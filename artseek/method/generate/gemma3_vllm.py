"""Gemma 3 served through vLLM, as an alternative ArtSeek backbone.

`Qwen2_5_VLVLLMChatModel` is generic apart from two Qwen-specific details in
loading, so this subclasses it and overrides only those:

  * Gemma 3 keeps its own chat template, which renders `{"type": "image"}`
    parts as `<start_of_image>` and folds a leading system turn into the first
    user turn;
  * `max_pixels` is a Qwen2-VL processor argument; Gemma 3's SigLIP processor
    uses fixed 896x896 images.

Gemma 3 does not emit Qwen's `<tool_call>` blocks, so use it with seeded
retrieval (`chat_turn(seed_queries=..., max_tool_calls=0)`) and without the
one-shot example (`use_shot=False, shot_path=None`).
"""

import re
from typing import Any

from langchain_core.messages import AIMessage
from transformers import AutoProcessor
from vllm import LLM

from artseek.method.generate.qwen2_5_vl_vllm import Qwen2_5_VLVLLMChatModel

GEMMA3_MODEL_ID = "google/gemma-3-27b-it"


class Gemma3VLLMChatModel(Qwen2_5_VLVLLMChatModel):
    """Gemma 3 (multimodal) behind the same chat interface as the Qwen model."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Gemma3VLLMChatModel":
        """Load Gemma 3 via vLLM.

        Args:
            pretrained_model_name_or_path: local snapshot path or hub id.
            **kwargs: forwarded to ``vllm.LLM(...)``. Gemma 3 27B is ~54 GB in
                bfloat16, so it needs ``tensor_parallel_size=2`` on 64 GB GPUs.

        Returns:
            A new instance of the model.
        """
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        # Deliberately *not* overriding processor.chat_template: Gemma 3 ships
        # its own, and the Qwen one would emit the wrong image sentinel.

        llm = LLM(
            model=str(pretrained_model_name_or_path),
            **kwargs,
        )
        return cls(processor=processor, llm=llm)

    def _try_parse_tool_calls(self, content: str):
        """Strip Gemma's end-of-turn marker (Gemma is not given tools)."""
        return AIMessage(
            content=re.sub(r"<end_of_turn>\s*$", "", content).strip(),
            additional_kwargs={},
            response_metadata={},
        )
