"""Mistral Small 3.1 served through vLLM, as an alternative ArtSeek backbone.

Uses RedHat's w4a16 build (~15 GB, one GPU), which is 4-bit like the AWQ Qwen
backbone.

`Qwen2_5_VLVLLMChatModel` is generic apart from two Qwen-specific details in
loading, so this subclasses it and overrides only those:

  * Mistral keeps its own chat template, which renders `{"type": "image"}`
    parts as `[IMG]`;
  * `max_pixels` is a Qwen2-VL processor argument that Mistral's processor
    does not accept.

Mistral does not emit Qwen's `<tool_call>` blocks and its chat template has no
`tool` role, so use it with merged seeded retrieval placed in the user turn
(`chat_turn(seed_queries=..., seed_merge_top_k=..., seed_context_into_user=True,
max_tool_calls=0)`) and without the one-shot example.
"""

import re
from typing import Any

from langchain_core.messages import AIMessage
from transformers import AutoProcessor
from vllm import LLM

from artseek.method.generate.qwen2_5_vl_vllm import Qwen2_5_VLVLLMChatModel

MISTRAL3_MODEL_ID = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16"


class Mistral3VLLMChatModel(Qwen2_5_VLVLLMChatModel):
    """Mistral Small 3.1 behind the same chat interface as the Qwen model."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "Mistral3VLLMChatModel":
        """Load Mistral Small 3.1 via vLLM.

        Args:
            pretrained_model_name_or_path: local snapshot path or hub id.
            **kwargs: forwarded to ``vllm.LLM(...)``.

        Returns:
            A new instance of the model.
        """
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            local_files_only=True,
        )
        processor.tokenizer.padding_side = "left"
        # Deliberately *not* overriding processor.chat_template: Mistral ships
        # its own, and the Qwen one would emit the wrong image sentinel.

        llm = LLM(
            model=str(pretrained_model_name_or_path),
            **kwargs,
        )
        return cls(processor=processor, llm=llm)

    def _try_parse_tool_calls(self, content: str):
        """Strip Mistral's end-of-sequence marker (Mistral is not given tools)."""
        return AIMessage(
            content=re.sub(r"</s>\s*$", "", content).strip(),
            additional_kwargs={},
            response_metadata={},
        )
