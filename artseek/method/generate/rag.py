"""ArtSeek: an MLLM that classifies an artwork (LICN) and retrieves WikiFragments
documents (ColQwen2 + Qdrant) through tool calls before answering."""

import io
import json
import math
import time
from pathlib import Path

import torch
from datasets import disable_caching, load_dataset
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from PIL import Image
from safetensors.torch import load_file, load_model

from ..classify.li_classification_network import (
    LateInteractionClassificationNetwork,
    SigmoidLoss,
)
from ..retrieve import ColQwen2Qdrant
from .prompts import get_system_prompt
from .qwen2_5_vl import Qwen2_5_VLChatModel
from .tools import RETRIEVAL_TOOL_SCHEMA

disable_caching()

# Pixel budgets for images placed in the model context.
SHOT_IMAGE_PIXELS = 64 * 28 * 28
CONTEXT_IMAGE_PIXELS = 128 * 28 * 28
# Upper bound on model calls in a single turn (each may issue tool calls).
MAX_ITERATIONS_PER_TURN = 5


class Qwen2_5_VLRAGModel:
    """The ArtSeek pipeline.

    Holds the backbone MLLM, the ColQwen2 retriever connected to Qdrant, the
    WikiFragments dataset used to materialise retrieved documents, the LICN
    classifier, and the one-shot example that teaches the tool-calling policy.
    Use `chat_turn` to run one conversational turn and `classify` to obtain the
    artwork card. See `config.load_artseek` for the default configuration.
    """

    def __init__(
        self,
        retriever_pretrained_model_name_or_path: str | Path,
        retriever_collection_name: str,
        model_pretrained_model_name_or_path: str | Path,
        licn_pretrained_path: str | Path,
        engine: str = "hf",
        model_kwargs: dict = None,
        licn_kwargs: dict = None,
        licn_loss_kwargs: dict = None,
        dataset_path: str | Path = None,
        artgraph_path: str | Path = None,
        shot_path: str | Path = None,
        use_shot: bool = True,
        system_prompt_override: str | None = None,
        chat_model_cls=None,
    ):
        """
        Args:
            retriever_pretrained_model_name_or_path: ColQwen2 checkpoint.
            retriever_collection_name: Qdrant collection holding the fragment embeddings.
            model_pretrained_model_name_or_path: Local path of the backbone MLLM.
            licn_pretrained_path: Directory with the LICN weights
                (`model.safetensors`, `model_1.safetensors`).
            engine: "vllm" (recommended) or "hf" (transformers `generate`).
            model_kwargs: Forwarded to the backbone's `from_pretrained`
                (`vllm.LLM(...)` kwargs for the vLLM engine).
            licn_kwargs: LICN constructor arguments.
            licn_loss_kwargs: SigmoidLoss constructor arguments.
            dataset_path: HF dataset with the WikiFragments rows; row `i` must be
                the document whose Qdrant payload has `idx == i`.
            artgraph_path: Directory with `class_lookups.json`,
                `task_lookups.json` and `task_matrices.safetensors`.
            shot_path: Directory with the one-shot example (`ann.json` + image).
            use_shot: Inject the one-shot example. When False, the tool-calling
                policy is described in the system prompt instead.
            system_prompt_override: Replace the built-in system prompt.
            chat_model_cls: Backbone class for the vLLM engine (defaults to
                Qwen2.5-VL; see `gemma3_vllm.py` and `mistral3_vllm.py`).
        """
        self.retriever = ColQwen2Qdrant(
            retriever_pretrained_model_name_or_path, retriever_collection_name
        )
        self.use_shot = use_shot
        self.system_prompt_override = system_prompt_override

        # Backbone
        model_kwargs = model_kwargs or {}
        if engine == "hf":
            self.model = Qwen2_5_VLChatModel.from_pretrained(
                model_pretrained_model_name_or_path, **model_kwargs
            )
        elif engine == "vllm":
            # Imported lazily: importing vllm is slow.
            from .qwen2_5_vl_vllm import Qwen2_5_VLVLLMChatModel

            cls_ = chat_model_cls or Qwen2_5_VLVLLMChatModel
            self.model = cls_.from_pretrained(
                model_pretrained_model_name_or_path, **model_kwargs
            )
        else:
            raise ValueError(f"Unknown engine: {engine!r}. Use 'hf' or 'vllm'.")

        # LICN classifier and label lookups
        self.licn = LateInteractionClassificationNetwork(**licn_kwargs)
        load_model(self.licn, Path(licn_pretrained_path) / "model.safetensors")
        self.licn.eval()
        self.licn.to(self.model.device)
        self.licn_loss = SigmoidLoss(**licn_loss_kwargs)
        load_model(self.licn_loss, Path(licn_pretrained_path) / "model_1.safetensors")
        self.licn_loss.eval()
        self.licn_loss.to(self.model.device)
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
            device=str(self.model.device),
        )

        # WikiFragments rows, used to turn retrieved ids into documents
        self.ds = load_dataset(dataset_path)

        # One-shot example. It is a worked example of Qwen's Hermes-style tool
        # calls, so other backbones should be run with shot_path=None.
        if shot_path is None:
            if use_shot:
                raise ValueError("use_shot=True requires shot_path; got None.")
            self.shot = None
        else:
            shot_path = Path(shot_path)
            with (shot_path / "ann.json").open("r") as f:
                shot_ann = json.load(f)
            shot_images = [
                Image.open(image_path).convert("RGB")
                for image_path in shot_path.glob("*.jpg")
            ]
            self.shot = {"ann": shot_ann, "images": shot_images}
            if self.use_shot:
                # Building the shot issues real retrieval calls, so skip it
                # when the shot is not used.
                self._load_shot()

    # ── Documents ────────────────────────────────────────────────────────

    def _doc_to_prompt(
        self, doc: tuple[dict, float], idx: int
    ) -> tuple[list[dict], list[Image.Image]]:
        """Convert a (document, score) pair into prompt parts and its images."""
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
            if isinstance(image, dict):
                decoded_image = Image.open(io.BytesIO(image["bytes"]))
            else:
                decoded_image = image
            images.append(decoded_image)
            prompt.append({"type": "text", "text": f"Caption: {caption}\n\n"})
        prompt.append({"type": "text", "text": f"### Document Text\n{doc[0]['text']}"})
        return prompt, images

    def _resize_to_target_pixels(
        self, image: Image.Image, target_pixels: int = SHOT_IMAGE_PIXELS
    ) -> Image.Image:
        """Resize keeping the aspect ratio so that W*H ~= target_pixels."""
        W, H = image.size
        aspect_ratio = W / H
        new_H = math.sqrt(target_pixels / aspect_ratio)
        new_W = target_pixels / new_H
        return image.resize((int(new_W), int(new_H)), Image.LANCZOS)

    def _retrieve_hits(
        self,
        query: str,
        requires_image: bool,
        input_image: Image.Image,
        limit: int = 10,
    ) -> tuple[list[int], list[float], float, float]:
        """Embed one query and search Qdrant.

        Returns:
            (row ids, scores, embedding seconds, search seconds)
        """
        t_embed = time.perf_counter()
        embeds = self.retriever.embed(
            [query], [input_image] if requires_image else None
        )
        embed_duration = time.perf_counter() - t_embed

        t_search = time.perf_counter()
        response = self.retriever.query(embeds[0], prefetch_limit=100, limit=limit)
        search_duration = time.perf_counter() - t_search

        return (
            [r.payload["idx"] for r in response.points],
            [r.score for r in response.points],
            embed_duration,
            search_duration,
        )

    def _context_from_hits(
        self,
        idxs: list[int],
        scores: list[float],
        embed_duration: float,
        search_duration: float,
    ) -> tuple[list[dict], dict]:
        """Materialise retrieval hits.

        Returns:
            (content for the tool message, artifact) where the artifact holds
            the context images, the row ids, the documents for display and the
            timings.
        """
        context = [(self.ds["train"][idx], score) for idx, score in zip(idxs, scores)]
        context_prompts = [
            self._doc_to_prompt(doc, idx) for idx, doc in enumerate(context, start=1)
        ]
        context_prompt_texts = [part for prompt, _ in context_prompts for part in prompt]
        context_prompt_images = [img for _, imgs in context_prompts for img in imgs]
        context_prompt_texts.insert(0, {"type": "text", "text": "# Context"})

        documents = []
        for row, score in context:
            doc_images = []
            for img, cap in zip(row["images"]["image"], row["images"]["caption"]):
                if isinstance(img, dict):
                    img = Image.open(io.BytesIO(img["bytes"]))
                doc_images.append({"image": img, "caption": cap})
            documents.append(
                {
                    "title": row["title"],
                    "score": score,
                    "text": row["text"],
                    "images": doc_images,
                }
            )

        return context_prompt_texts, {
            "context_images": context_prompt_images,
            "context_idxs": idxs,
            "documents": documents,
            "embed_duration": embed_duration,
            "search_duration": search_duration,
        }

    def _call_retrieve_tool(
        self, query: str, requires_image: bool, input_image: Image.Image
    ) -> tuple[list[dict], dict]:
        """Execute one `get_relevant_documents` call (top-10 documents)."""
        idxs, scores, embed_duration, search_duration = self._retrieve_hits(
            query, requires_image, input_image
        )
        return self._context_from_hits(idxs, scores, embed_duration, search_duration)

    def _load_shot(self):
        """Build the one-shot conversation, running its retrieval calls for real."""
        str_2_message = {
            "ai": AIMessage,
            "human": HumanMessage,
            "tool": ToolMessage,
        }

        tool_contents = []
        for query in self.shot["ann"]["queries"]:
            idxs, scores, _, _ = self._retrieve_hits(
                query["text"],
                query["requires_image"],
                self.shot["images"][0],
                limit=5,
            )
            content, artifact = self._context_from_hits(idxs, scores, 0.0, 0.0)
            tool_contents.append((content, artifact["context_images"]))

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

    # ── Classification ───────────────────────────────────────────────────

    @torch.no_grad()
    def classify(self, image: Image.Image) -> dict:
        """Predict the artwork card with LICN.

        Returns:
            task -> list of (label, probability). Single-label tasks (artist,
            genre, style) return the argmax; multi-label tasks (media, tag)
            return every label with probability > 0.4.
        """
        tasks = ("artist", "genre", "media", "style", "tag")
        multiclass = ("artist", "genre", "style")
        multilabel = ("media", "tag")
        embeds = torch.tensor(self.retriever.embed(images=[image])).to(
            self.model.device
        )
        visual_task_embeds = self.licn(visual_embeddings=embeds)[
            "visual_task_embeddings"
        ].squeeze()
        task_preds = {k: [] for k in tasks}

        for i, task in enumerate(tasks):
            logits = self.licn_loss(
                visual_task_embeds[i],
                self.task_matrices[task],
                None,
                return_logits=True,
            )
            probs = torch.sigmoid(logits)
            if task in multiclass:
                preds = torch.argmax(probs).view(1)
                probs = probs[preds]
            elif task in multilabel:
                preds = torch.where(probs > 0.4)[0]
                probs = probs[preds]

            for pred, prob in zip(preds, probs):
                task_preds[task].append((self.lookups[task][pred.item()], prob))

        return task_preds

    # ── Conversation ─────────────────────────────────────────────────────

    def _get_system_message(
        self, classify: bool, retrieve: bool, use_shot: bool = True
    ) -> SystemMessage:
        return SystemMessage(get_system_prompt(classify, retrieve, use_shot))

    # Nudges used by `force_first_tool_call`, in escalation order. Generation
    # is greedy, so re-invoking the same prompt would return the same answer:
    # the prompt has to change for a retry to mean anything.
    _FORCE_NUDGES = (
        "Before answering, you must first search the document collection. "
        'Call the "get_relevant_documents" tool now with a short, focused '
        "query about the information this question needs.",
        "You did not call the tool. Respond with nothing but a single "
        "get_relevant_documents tool call.",
    )

    @staticmethod
    def _extract_query_text(messages: list) -> str:
        """Best-effort recovery of the user's question from the last human message."""
        for m in reversed(messages):
            if not isinstance(m, HumanMessage):
                continue
            content = m.content
            if isinstance(content, str):
                return content.strip()
            for part in reversed(content if isinstance(content, list) else []):
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    if "# Query" in text:
                        return text.split("# Query", 1)[1].strip()
            return " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            ).strip()
        return ""

    def _force_tool_call(
        self, model, all_messages, all_images, max_new_tokens, step_durations
    ):
        """Make the model issue a retrieval call it did not issue on its own.

        The answer without a tool call is dropped from the context, a short
        user nudge is appended and the model is re-invoked (up to two
        escalating nudges), so the model still writes its own query. If both
        nudges are refused, a tool call whose query is the raw question is
        synthesised.

        Returns:
            (response, mechanism, extra_messages) with mechanism one of
            "nudge_1", "nudge_2", "synthetic".
        """
        for i, nudge in enumerate(self._FORCE_NUDGES, start=1):
            nudge_msg = HumanMessage(content=[{"type": "text", "text": nudge}])
            all_messages.append(nudge_msg)

            t0 = time.perf_counter()
            response = model.invoke(
                all_messages, images=all_images, max_new_tokens=max_new_tokens
            )
            duration = time.perf_counter() - t0
            response.response_metadata["duration"] = duration
            step_durations.append(
                {"type": "generation", "duration": duration, "forced_retry": i}
            )
            if response.tool_calls:
                return response, f"nudge_{i}", [nudge_msg]
            all_messages.pop()

        query = self._extract_query_text(all_messages) or "artwork information"
        response = AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="call_forced_synthetic",
                    name="get_relevant_documents",
                    args={"query": query, "requires_image": False},
                )
            ],
        )
        return response, "synthetic", []

    def _add_context_images(self, artifact, new_context_images, all_images):
        resized = [
            self._resize_to_target_pixels(img, CONTEXT_IMAGE_PIXELS)
            for img in artifact["context_images"]
        ]
        new_context_images.extend(resized)
        all_images.extend(resized)

    @torch.no_grad()
    def chat_turn(
        self,
        messages: list,
        input_image: Image.Image,
        classify: bool = True,
        retrieve: bool = True,
        context_images: list | None = None,
        max_new_tokens: int = 512,
        force_first_tool_call: bool = False,
        max_tool_calls: int | None = None,
        seed_queries: list[dict] | None = None,
        seed_limit: int = 10,
        seed_merge_top_k: int | None = None,
        seed_context_into_user: bool = False,
    ) -> tuple[list, list]:
        """Run one turn of a (multi-turn) conversation, including tool calls.

        The caller keeps the conversation state: pass the full history in
        `messages` and the context images accumulated so far, then append the
        returned messages and images.

        Args:
            messages: Conversation history (langchain messages). The first user
                message holds the image placeholder (see `prompts.build_user_content`).
            input_image: The artwork image.
            classify: Whether the artwork card is part of the prompt (the card
                itself must already be in the user message).
            retrieve: Whether the retrieval tool is available.
            context_images: Context images accumulated in previous turns.
            max_new_tokens: Maximum new tokens per model call.
            force_first_tool_call: Do not let the model answer without
                retrieving at least once (see `_force_tool_call`).
            max_tool_calls: Cap on retrieval calls in this turn; once reached,
                the tool is unbound and the model must answer. The cap is
                checked between model calls, so one response emitting several
                tool calls executes all of them.
            seed_queries: Retrieval queries, each ``{"query": str,
                "requires_image": bool}``, executed before the model is first
                invoked and inserted as tool-call / tool-result pairs. The model
                may still issue further calls unless `max_tool_calls` forbids it.
                This is how backbones that cannot emit tool calls are given
                retrieved context.
            seed_limit: Documents retrieved per seed query.
            seed_merge_top_k: If set, merge the seed queries into a single tool
                result: the hits of all queries are deduplicated, sorted by
                score and truncated to this many documents.
            seed_context_into_user: With `seed_merge_top_k`, append the merged
                documents to the last user message instead of adding a tool
                message, for chat templates without a `tool` role.

        Returns:
            (new_messages, new_context_images)
        """
        if self.system_prompt_override is not None:
            system_message = SystemMessage(self.system_prompt_override)
        else:
            system_message = self._get_system_message(
                classify, retrieve, use_shot=self.use_shot
            )
        context_images = list(context_images) if context_images else []

        inject_shot = self.use_shot and (classify or retrieve)
        if inject_shot:
            all_messages = [system_message] + self.shot["messages"] + messages
            all_images = list(self.shot["images"]) + [input_image] + context_images
        else:
            all_messages = [system_message] + messages
            all_images = [input_image] + context_images

        model = self.model.bind_tools([RETRIEVAL_TOOL_SCHEMA]) if retrieve else self.model

        new_messages: list = []
        new_context_images: list = []
        step_durations: list[dict] = []
        force_mechanism = "not_forced"
        n_tool_calls = 0

        # Seeded retrieval, before the model speaks.
        seeds = [sq for sq in (seed_queries or []) if (sq.get("query") or "").strip()]

        if seeds and seed_merge_top_k:
            merged: dict[int, float] = {}
            embed_total = search_total = 0.0
            for sq in seeds:
                idxs, scores, e_dur, s_dur = self._retrieve_hits(
                    sq["query"], sq.get("requires_image", True), input_image,
                    limit=seed_limit,
                )
                embed_total += e_dur
                search_total += s_dur
                for idx, sc in zip(idxs, scores):
                    if sc > merged.get(idx, float("-inf")):
                        merged[idx] = sc
            ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
            ranked = ranked[:seed_merge_top_k]
            content, artifact = self._context_from_hits(
                [i for i, _ in ranked], [s for _, s in ranked],
                embed_total, search_total,
            )
            artifact["seed_queries"] = [sq["query"] for sq in seeds]
            merged_query = " | ".join(sq["query"] for sq in seeds)
            n_tool_calls += 1
            step_durations.append({
                "type": "tool", "duration": embed_total + search_total,
                "query": merged_query, "seeded": True, "merged_from": len(seeds),
            })
            resized = [self._resize_to_target_pixels(img, CONTEXT_IMAGE_PIXELS)
                       for img in artifact["context_images"]]

            if seed_context_into_user:
                # The artwork image is the first placeholder of that message and
                # the context images follow it, so all_images stays aligned.
                target = next((m for m in reversed(all_messages)
                               if isinstance(m, HumanMessage)), None)
                if target is None:
                    raise ValueError(
                        "seed_context_into_user needs a user message to append "
                        "to, but none was passed in `messages`.")
                if not isinstance(target.content, list):
                    target.content = [{"type": "text", "text": str(target.content)}]
                target.content = list(target.content) + list(content)
                # Keeps the retrieved ids and timings in the returned messages.
                new_messages.append(ToolMessage(
                    content="", artifact=artifact,
                    tool_call_id="call_seed_merged",
                    name="get_relevant_documents"))
            else:
                call_id = "call_seed_merged"
                seed_ai = AIMessage(
                    content="",
                    tool_calls=[ToolCall(
                        id=call_id, name="get_relevant_documents",
                        args={"query": merged_query, "requires_image": True})],
                )
                tool_msg = ToolMessage(content=content, artifact=artifact,
                                       tool_call_id=call_id,
                                       name="get_relevant_documents")
                new_messages += [seed_ai, tool_msg]
                all_messages += [seed_ai, tool_msg]

            new_context_images.extend(resized)
            all_images.extend(resized)
            seeds = []

        for i, seed in enumerate(seeds):
            query = seed["query"].strip()
            requires_image = seed.get("requires_image", True)
            call_id = f"call_seed_{i}"
            seed_ai = AIMessage(
                content="",
                tool_calls=[ToolCall(id=call_id, name="get_relevant_documents",
                                     args={"query": query,
                                           "requires_image": requires_image})],
            )
            new_messages.append(seed_ai)
            all_messages.append(seed_ai)

            n_tool_calls += 1
            t1 = time.perf_counter()
            idxs, scores, e_dur, s_dur = self._retrieve_hits(
                query, requires_image, input_image, limit=seed_limit
            )
            content, artifact = self._context_from_hits(idxs, scores, e_dur, s_dur)
            duration = time.perf_counter() - t1
            artifact["duration"] = duration
            step_durations.append(
                {"type": "tool", "query": query, "duration": duration, "seeded": True}
            )
            tool_msg = ToolMessage(content=content, artifact=artifact,
                                   tool_call_id=call_id,
                                   name="get_relevant_documents")
            new_messages.append(tool_msg)
            all_messages.append(tool_msg)
            self._add_context_images(artifact, new_context_images, all_images)

        # A seeded turn has already retrieved: nothing left to force.
        if seed_queries and force_first_tool_call:
            force_mechanism = "seeded"
            force_first_tool_call = False

        for iteration in range(MAX_ITERATIONS_PER_TURN):
            capped = max_tool_calls is not None and n_tool_calls >= max_tool_calls
            active_model = self.model if capped else model
            t0 = time.perf_counter()
            response = active_model.invoke(
                all_messages, images=all_images, max_new_tokens=max_new_tokens
            )
            duration = time.perf_counter() - t0
            response.response_metadata["duration"] = duration
            step_durations.append({"type": "generation", "duration": duration})

            if force_first_tool_call and iteration == 0 and not response.tool_calls:
                response, force_mechanism, nudges = self._force_tool_call(
                    model, all_messages, all_images, max_new_tokens, step_durations
                )
                new_messages.extend(nudges)
            elif force_first_tool_call and iteration == 0:
                force_mechanism = "none_needed"

            new_messages.append(response)
            all_messages.append(response)

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                n_tool_calls += 1
                query = tc["args"].get("query", "")
                requires_image = tc["args"].get("requires_image", False)
                t1 = time.perf_counter()
                content, artifact = self._call_retrieve_tool(
                    query, requires_image, input_image
                )
                duration = time.perf_counter() - t1
                artifact["duration"] = duration
                step_durations.append(
                    {"type": "tool", "query": query, "duration": duration}
                )
                tool_msg = ToolMessage(
                    content=content,
                    artifact=artifact,
                    tool_call_id=tc["id"],
                    name="get_relevant_documents",
                )
                new_messages.append(tool_msg)
                all_messages.append(tool_msg)
                self._add_context_images(artifact, new_context_images, all_images)

        # Per-turn timing breakdown, attached to the final answer.
        if new_messages and isinstance(new_messages[-1], AIMessage):
            meta = new_messages[-1].response_metadata
            meta["step_durations"] = step_durations
            meta["total_duration"] = sum(s["duration"] for s in step_durations)
            meta["force_mechanism"] = force_mechanism

        return new_messages, new_context_images


# Shorter alias.
ArtSeek = Qwen2_5_VLRAGModel
