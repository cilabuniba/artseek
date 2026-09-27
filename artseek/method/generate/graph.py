"""LangGraph version of the pipeline, used for batch inference (`test.py`).

Equivalent to `Qwen2_5_VLRAGModel.chat_turn` for a single-turn conversation,
running on the shared model returned by `config.get_model()`.
"""

from functools import partial
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image
from typing_extensions import TypedDict

from .config import get_model
from .prompts import format_artwork_card, get_system_prompt
from .rag import CONTEXT_IMAGE_PIXELS
from .tools import RETRIEVAL_TOOL_SCHEMA, get_relevant_documents


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


def set_input_image(state: InputState, classify: bool) -> State:
    """Entry node: add the artwork card to the user message when classifying."""
    messages = state["messages"]
    if classify:
        card = format_artwork_card(get_model().classify(state["input_image"]))
        content = messages[-1].content
        content.insert(2, {"type": "text", "text": card})
        messages[-1] = HumanMessage(content=content)
    return {"messages": messages, "input_image": state["input_image"]}


def query_or_respond(state: State, classify: bool, retrieve: bool) -> State:
    """Agent node: answer, or emit tool calls when retrieval is enabled."""
    model = get_model()
    llm = model.model.bind_tools([RETRIEVAL_TOOL_SCHEMA]) if retrieve else model.model

    all_messages = [SystemMessage(get_system_prompt(classify, retrieve))]
    all_images = [state["input_image"]] + state["context_images"]
    if classify or retrieve:
        all_messages += model.shot["messages"]
        all_images = model.shot["images"] + all_images
    all_messages += state["messages"]

    # Images returned by the tool call just executed.
    context_images = []
    last_message = state["messages"][-1]
    if retrieve and last_message.type == "tool" and last_message.artifact:
        context_images = [
            model._resize_to_target_pixels(item, CONTEXT_IMAGE_PIXELS)
            for item in last_message.artifact["context_images"]
        ]
        all_images += context_images

    response = llm.invoke(all_messages, images=all_images, max_new_tokens=512)
    return {"messages": [response], "context_images": context_images}


def build_graph(classify: bool = True, retrieve: bool = True):
    """Compile the graph for a classification/retrieval configuration."""
    graph_builder = StateGraph(State, input=InputState)
    graph_builder.add_node("set_input_image", partial(set_input_image, classify=classify))
    graph_builder.add_node(
        "query_or_respond",
        partial(query_or_respond, classify=classify, retrieve=retrieve),
    )
    graph_builder.set_entry_point("set_input_image")
    graph_builder.add_edge("set_input_image", "query_or_respond")
    if retrieve:
        graph_builder.add_node("tools", ToolNode([get_relevant_documents]))
        graph_builder.add_conditional_edges(
            "query_or_respond", tools_condition, {END: END, "tools": "tools"}
        )
        graph_builder.add_edge("tools", "query_or_respond")
    else:
        graph_builder.add_edge("query_or_respond", END)
    return graph_builder.compile()
