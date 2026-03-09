import json
import math
from functools import partial
from pathlib import Path
from typing import Annotated

from datasets import load_from_disk, load_dataset, disable_caching
disable_caching()
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from PIL import Image
from transformers.utils import get_json_schema
from typing_extensions import TypedDict
import torch
from safetensors.torch import load_model, load_file
import os

from ..retrieve import ColQwen2Qdrant
from ..classify.li_classification_network import (
    LateInteractionClassificationNetwork,
    SigmoidLoss,
)
from ...utils.dirutils import get_data_dir, get_model_checkpoints_dir, get_models_dir
from . import Qwen2_5_VLChatModel


class Qwen2_5_VLRAGModel:
    def __init__(
        self,
        retriever_pretrained_model_name_or_path: str | Path,
        retriever_collection_name: str,
        model_pretrained_model_name_or_path: str | Path,
        licn_pretrained_path: str | Path,
        model_kwargs: dict = None,
        licn_kwargs: dict = None,
        licn_loss_kwargs: dict = None,
        dataset_path: str | Path = None,
        artgraph_path: str | Path = None,
        shot_path: str | Path = None,
    ):
        # Load retriever and model
        self.retriever = ColQwen2Qdrant(
            retriever_pretrained_model_name_or_path, retriever_collection_name
        )
        self.model = Qwen2_5_VLChatModel.from_pretrained(
            model_pretrained_model_name_or_path, **model_kwargs
        )

        # Load LICN
        self.licn = LateInteractionClassificationNetwork(**licn_kwargs)
        load_model(self.licn, Path(licn_pretrained_path) / "model.safetensors")
        self.licn.eval()
        self.licn.to(self.model.mllm.device)
        self.licn_loss = SigmoidLoss(**licn_loss_kwargs)
        load_model(self.licn_loss, Path(licn_pretrained_path) / "model_1.safetensors")
        self.licn_loss.eval()
        self.licn_loss.to(self.model.mllm.device)
        with open(Path(artgraph_path) / "class_lookups.json", "r") as f:
            class_lookups = json.load(f)
        with open(Path(artgraph_path) / "task_lookups.json", "r") as f:
            task_lookups = json.load(f)
        self.lookups = {}
        for task in task_lookups.keys():
            self.lookups[task] = {}
            for k, v in task_lookups[task].items():
                if k == "-100":
                    continue
                self.lookups[task][v] = class_lookups[task][k]
        self.task_matrices = load_file(
            Path(artgraph_path) / "task_matrices.safetensors",
            device=str(self.model.mllm.device),
        )

        # Load DS
        self.ds = load_dataset(dataset_path)

        # Load the one-shot example
        shot_path = Path(shot_path)
        with (shot_path / "ann.json").open("r") as f:
            shot_ann = json.load(f)

        # Read shot images
        shot_images = []
        for image_path in shot_path.glob("*.jpg"):
            image = Image.open(image_path)
            image = image.convert("RGB")
            shot_images.append(image)

        # Load shot
        self.shot = {
            "ann": shot_ann,
            "images": shot_images,
        }
        self._load_shot()

    def _doc_to_prompt(
        self, doc: tuple[dict, float], idx: int
    ) -> tuple[list[dict], list[str]]:
        """Convert a document-score pair to a prompt format.

        Args:
            doc (tuple[dict, float]): The document and its score.
            idx (int): The index of the document.

        Returns:
            tuple[list[dict], list[str]]: The prompt and the list of images to include.
        """
        prompt = []
        prompt.append(
            {
                "type": "text",
                "text": f"\n\n## Document {idx}: {doc[0]['title']}\nScore: {doc[1]:.2f}.\n\n",
            }
        )
        images = []
        for i, (image, caption) in enumerate(
            zip(doc[0]["images"]["image"], doc[0]["images"]["caption"]), start=1
        ):
            prompt.append({"type": "text", "text": f"### Image {i}\n"})
            prompt.append({"type": "image"})
            images.append(image)
            prompt.append({"type": "text", "text": f"Caption: {caption}\n\n"})
        prompt.append({"type": "text", "text": f"### Document Text\n{doc[0]['text']}"})
        return prompt, images

    def _get_relevant_documents_shot(self, query: str, image: Image.Image = None):
        """Get relevant documents and their scores for a given query.

        Args:
            query (str): The query string.
            image (str, optional): The image path. Defaults to None.

        Returns:
            list[tuple[dict, float]]: The relevant documents and their scores.
        """
        embeds = self.retriever.embed([query], [image] if image else None)
        response = self.retriever.query(embeds[0], prefetch_limit=100, limit=5)

        # Format
        context = [
            (self.ds["train"][record.payload["idx"]], record.score)
            for record in response.points
        ]
        
        # Save imgs for paper purpose
        os.makedirs("paper_imgs", exist_ok=True)            
        for doc, record in zip(context, response.points):   
            doc[0]["fragment"].save(f"paper_imgs/{record.payload['idx']}_{doc[0]['title']}.jpg")   
            
        context_prompts = [
            self._doc_to_prompt(doc, idx) for idx, doc in enumerate(context, start=1)
        ]
        context_prompt_texts = [prompt for prompt, _ in context_prompts]
        context_prompt_images = [images for _, images in context_prompts]
        context_prompt_texts = [
            item for sublist in context_prompt_texts for item in sublist
        ]
        context_prompt_images = [
            item for sublist in context_prompt_images for item in sublist
        ]

        # Add context prompt header
        context_prompt_texts.insert(0, {"type": "text", "text": "# Context"})
        return context_prompt_texts, context_prompt_images

    def _resize_to_target_pixels(
        self, image: Image.Image, target_pixels: int = 64 * 28 * 28
    ) -> Image.Image:
        W, H = image.size
        aspect_ratio = W / H

        # Compute new height and width
        new_H = math.sqrt(target_pixels / aspect_ratio)
        new_W = target_pixels / new_H

        # Round to nearest integer
        new_H = int(new_H)
        new_W = int(new_W)

        # Resize image
        return image.resize((new_W, new_H), Image.LANCZOS)

    def _load_shot(self):
        str_2_message = {
            "ai": AIMessage,
            "human": HumanMessage,
            "tool": ToolMessage,
        }

        tool_contents = []
        for query in self.shot["ann"]["queries"]:
            tool_contents.append(
                self._get_relevant_documents_shot(
                    query["text"],
                    self.shot["images"][0] if query["requires_image"] else None,
                )
            )

        messages = []
        for message in self.shot["ann"]["messages"]:
            if message["role"] == "tool":
                content = tool_contents[message["content"]][0]
                message = str_2_message[message["role"]](
                    content=content, tool_call_id=message["tool_call_id"]
                )
            else:
                message = str_2_message[message["role"]](content=message["content"])
            messages.append(message)

        self.shot["messages"] = messages
        for content in tool_contents:
            self.shot["images"] += content[1]
        self.shot["images"] = [
            self._resize_to_target_pixels(image) for image in self.shot["images"]
        ]

    @torch.no_grad()
    def classify(self, image: Image.Image):
        """Classify the input image using the LICN model.

        Args:
            image (Image.Image): The input image.

        Returns:
            dict: The classification results.
        """
        tasks = ("artist", "genre", "media", "style", "tag")
        multiclass = ("artist", "genre", "style")
        multilabel = ("media", "tag")
        embeds = torch.tensor(self.retriever.embed(images=[image])).to(
            self.model.mllm.device
        )
        visual_task_embeds = self.licn(visual_embeddings=embeds)[
            "visual_task_embeddings"
        ].squeeze()
        task_preds = {k: [] for k in tasks}

        for i, task in enumerate(tasks):
            visual_task_embed = visual_task_embeds[i]
            task_matrix = self.task_matrices[task]
            logits = self.licn_loss(
                visual_task_embed,
                task_matrix,
                None,
                return_logits=True,
            )
            # Apply sigmoid to the logits
            probs = torch.sigmoid(logits)
            if task in multiclass:
                preds = torch.argmax(probs).view(1)
                probs = probs[preds]
            elif task in multilabel:
                # get indices of preds > 0.4
                preds = torch.where(probs > 0.4)[0]
                probs = probs[preds]

            for pred, prob in zip(preds, probs):
                task_preds[task].append((self.lookups[task][pred.item()], prob))

        return task_preds


# Model and tools

MODEL = Qwen2_5_VLRAGModel(
    retriever_pretrained_model_name_or_path=get_models_dir() / "colqwen2-v1.0",
    retriever_collection_name="wikifragments-visual-arts-embeds",
    model_pretrained_model_name_or_path="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    licn_pretrained_path="data_/hf/hub/models--cilabuniba--artseek-licn/snapshots/b02c6e12a21fc9d4af84897a6ac1efbe46da2695",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
    },
    licn_kwargs={
        "activation": "gelu",
        "embedding_dim": 128,
        "num_tasks": 5,
        "num_encoder_layers": 6,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "output_dim": 512,
        "single_encoder": True,
        "single_embedding": False,
        "single_projection": True,
    },
    licn_loss_kwargs={
        "num_tasks": 5,
    },
    dataset_path="cilabuniba/wikifragments-visual-arts-embeds",
    artgraph_path="data_/hf/hub/datasets--cilabuniba--artseek-licn-data/snapshots/94ca4f9ec14941fa2cd45643ea351164e46b442e",
    shot_path=Path(__file__).parent / "shot",
)


@tool(response_format="content_and_artifact")
def get_relevant_documents(
    query: str, requires_image: bool, state: Annotated[dict, InjectedState]
):
    """This is an tool for retrieving relevant documents that can help answer user questions.
    It takes a text query and, optionally, the image to which it refers to retrieve useful contextual knowledge.
    Upon calling the tool, it returns relevant documents that can be used to generate an accurate response.
    Use this tool whenever possible to improve the quality of your answers.

    Args:
        query: A text query for information retrieval.
        requires_image: Whether the query should be combined with the input image or is a text-only query.
        state: The graph state.

    Returns:
        list: A list of retrieved documents relevant to the query.
    """
    # retrieve
    embeds = MODEL.retriever.embed(
        [query], [state["input_image"]] if requires_image else None
    )
    response = MODEL.retriever.query(embeds[0], prefetch_limit=100, limit=10)

    # Format
    idxs = [record.payload["idx"] for record in response.points]
    scores = [record.score for record in response.points]
    context = [(MODEL.ds["train"][idx], score) for idx, score in zip(idxs, scores)]
    context_prompts = [
        MODEL._doc_to_prompt(doc, idx) for idx, doc in enumerate(context, start=1)
    ]
    context_prompt_texts = [prompt for prompt, _ in context_prompts]
    context_prompt_images = [images for _, images in context_prompts]
    context_prompt_texts = [
        item for sublist in context_prompt_texts for item in sublist
    ]
    context_prompt_images = [
        item for sublist in context_prompt_images for item in sublist
    ]

    # Add context prompt header
    context_prompt_texts.insert(0, {"type": "text", "text": "# Context"})

    return context_prompt_texts, {
        "context_images": context_prompt_images,
        "context_idxs": idxs,
    }


# Langgraph definitions


def get_json_schema_no_state(f: callable):
    schema = get_json_schema(f)
    del schema["function"]["parameters"]["properties"]["state"]
    schema["function"]["parameters"]["required"] = [
        item for item in schema["function"]["parameters"]["required"] if item != "state"
    ]
    return schema


def manage_list(existing: list, updates: list):
    return existing + updates


class InputState(TypedDict):
    messages: Annotated[list, add_messages]
    input_image: Image.Image
    card: str


class State(TypedDict):
    messages: Annotated[list, add_messages]
    input_image: Image.Image
    context_images: Annotated[list, manage_list]


def set_input_image_classify(state: InputState) -> State:
    messages = state["messages"]
    card_dict = MODEL.classify(state["input_image"])
    card = "\n# Artwork card"
    for k, v in card_dict.items():
        card += f"\n{k}: "
        for i, label in enumerate(v):
            pred, prob = label
            card += f"{pred} ({int(prob * 100):.0f}%)"
            if i < len(v) - 1:
                card += ", "
    message_content = messages[-1].content
    message_content.insert(2, {"type": "text", "text": card})
    messages[-1] = HumanMessage(content=message_content)
    return {
        "messages": messages,
        "input_image": state["input_image"],
    }


def set_input_image_noclassify(state: InputState) -> State:
    return {
        "messages": state["messages"],
        "input_image": state["input_image"],
    }


def query_or_respond_classify_retrieve(state: State) -> State:
    model_with_tools = MODEL.model.bind_tools([
        get_json_schema_no_state(get_relevant_documents.func)
    ])
    system_message = SystemMessage(
        'You are a helpful assistant.\n\n# Rules\nWhen given a user query, analyze how best to respond.\n* Only very simple questions (e.g., naming clearly visible objects) should be answered directly.\n* An artwork card is provided with some information about the artwork. The artwork card might not be completely accurate, so when referring to the artist (for instance), always use terms such as "might be" or similar.\n* For most other questions—especially those involving the artist, historical context, stylistic analysis, or external references—you must retrieve and use relevant documents before responding.\n* Not all documents may be useful; select only the most relevant ones to support your answer.\n* To retrieve documents, use the tool "get_relevant_documents".\n* You may call the "get_relevant_documents" tool at most 3 times per user query.\n* Whenever possible, indicate which documents were used in your reasoning. Enclose your reasoning process within <think></think> XML tags.'
    )
    all_messages = [system_message] + MODEL.shot["messages"] + state["messages"]
    all_images = MODEL.shot["images"] + [state["input_image"]] + state["context_images"]
    context_images = []
    last_message = state["messages"][-1]
    if last_message.type == "tool":
        try:
            context_images = [
                MODEL._resize_to_target_pixels(item, 128 * 28 * 28)
                for item in last_message.artifact["context_images"]
            ]
            all_images += context_images
        except TypeError:
            print(last_message)
    response = model_with_tools.invoke(
        all_messages,
        images=all_images,
        max_new_tokens=512,
    )
    return {"messages": [response], "context_images": context_images}


def query_or_respond_classify_noretrieve(state: State) -> State:
    model_with_tools = MODEL.model
    system_message = SystemMessage(
        'You are a helpful assistant.\n\n# Rules\nWhen given a user query, analyze how best to respond.\n* Only very simple questions (e.g., naming clearly visible objects) should be answered directly.\n* An artwork card is provided with some information about the artwork. The artwork card might not be completely accurate, so when referring to the artist (for instance), always use terms such as "might be" or similar.\n* For most other questions—especially those involving the artist, historical context, or stylistic analysis—use the information in the artwork card and your own knowledge to answer.\n* Whenever possible, indicate your reasoning process within <think></think> XML tags.'
    )
    all_messages = [system_message] + MODEL.shot["messages"] + state["messages"]
    all_images = MODEL.shot["images"] + [state["input_image"]] + state["context_images"]
    response = model_with_tools.invoke(
        all_messages,
        images=all_images,
        max_new_tokens=512,
    )
    return {"messages": [response], "context_images": []}


def query_or_respond_noclassify_retrieve(state: State) -> State:
    model_with_tools = MODEL.model.bind_tools([
        get_json_schema_no_state(get_relevant_documents.func)
    ])
    system_message = SystemMessage(
        'You are a helpful assistant.\n\n# Rules\nWhen given a user query, analyze how best to respond.\n* Only very simple questions (e.g., naming clearly visible objects) should be answered directly.\n* For most other questions—especially those involving the artist, historical context, stylistic analysis, or external references—you must retrieve and use relevant documents before responding.\n* Not all documents may be useful; select only the most relevant ones to support your answer.\n* To retrieve documents, use the tool "get_relevant_documents".\n* You may call the "get_relevant_documents" tool at most 5 times per user query.\n* Whenever possible, indicate which documents were used in your reasoning. Enclose your reasoning process within <think></think> XML tags.'
    )
    all_messages = [system_message] + MODEL.shot["messages"] + state["messages"]
    all_images = MODEL.shot["images"] + [state["input_image"]] + state["context_images"]
    context_images = []
    last_message = state["messages"][-1]
    if last_message.type == "tool":
        try:
            context_images = [
                MODEL._resize_to_target_pixels(item, 128 * 28 * 28)
                for item in last_message.artifact["context_images"]
            ]
            all_images += context_images
        except TypeError:
            print(last_message)
    response = model_with_tools.invoke(
        all_messages,
        images=all_images,
        max_new_tokens=512,
    )
    return {"messages": [response], "context_images": context_images}


# def query_or_respond_noclassify_noretrieve(state: State) -> State:
#     model_with_tools = MODEL.model
#     system_message = SystemMessage(
#         'You are a helpful assistant.\n\n# Rules\nWhen given a user query, analyze how best to respond.\n* Only very simple questions (e.g., naming clearly visible objects) should be answered directly.\n* For most other questions—especially those involving the artist, historical context, or stylistic analysis—use your own knowledge to answer.\n* Whenever possible, indicate your reasoning process within <think></think> XML tags.'
#     )
#     all_messages = [system_message] + MODEL.shot["messages"] + state["messages"]
#     all_images = MODEL.shot["images"] + [state["input_image"]] + state["context_images"]
#     response = model_with_tools.invoke(
#         all_messages,
#         images=all_images,
#         max_new_tokens=512,
#     )
#     return {"messages": [response], "context_images": []}


def query_or_respond_noclassify_noretrieve(state: State) -> State:
    model_with_tools = MODEL.model
    system_message = SystemMessage(
        'You are a helpful assistant.\n\n# Rules\nWhen given a user query, analyze how best to respond.\n* Only very simple questions (e.g., naming clearly visible objects) should be answered directly.\n* For most other questions—especially those involving the artist, historical context, or stylistic analysis—use your own knowledge to answer.\n* Whenever possible, indicate your reasoning process within <think></think> XML tags.'
    )
    all_messages = [system_message] + state["messages"]
    all_images = [state["input_image"]] + state["context_images"]
    response = model_with_tools.invoke(
        all_messages,
        images=all_images,
        max_new_tokens=512,
    )
    return {"messages": [response], "context_images": []}


def build_graph(classify: bool = True, retrieve: bool = True):
    graph_builder = StateGraph(State, input=InputState)
    if classify:
        set_input = set_input_image_classify
        set_input_name = "set_input_image_classify"
        if retrieve:
            query_or_respond_fn = query_or_respond_classify_retrieve
            query_or_respond_name = "query_or_respond_classify_retrieve"
        else:
            query_or_respond_fn = query_or_respond_classify_noretrieve
            query_or_respond_name = "query_or_respond_classify_noretrieve"
    else:
        set_input = set_input_image_noclassify
        set_input_name = "set_input_image_noclassify"
        if retrieve:
            query_or_respond_fn = query_or_respond_noclassify_retrieve
            query_or_respond_name = "query_or_respond_noclassify_retrieve"
        else:
            query_or_respond_fn = query_or_respond_noclassify_noretrieve
            query_or_respond_name = "query_or_respond_noclassify_noretrieve"
    graph_builder.add_node(set_input)
    graph_builder.add_node(query_or_respond_fn)
    if retrieve:
        tools = ToolNode([get_relevant_documents])
        graph_builder.add_node(tools)
    graph_builder.set_entry_point(set_input_name)
    graph_builder.add_edge(set_input_name, query_or_respond_name)
    if retrieve:
        graph_builder.add_conditional_edges(
            query_or_respond_name,
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", query_or_respond_name)
    else:
        graph_builder.add_edge(query_or_respond_name, END)
    return graph_builder.compile()


# Start!

# input_message = "Make a formal analysis of this painting."

# for step in GRAPH.stream(
#     {
#         "input_image": Image.open(get_data_dir() / "images" / "nicolas.jpg"),
#         "messages": [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "Current Query:"},
#                     {"type": "image"},
#                     {"type": "text", "text": input_message},
#                 ],
#             },
#         ],
#     },
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()
