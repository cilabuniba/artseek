from .qwen2_5_vl import Qwen2_5_VLChatModel

__all__ = [
    "Qwen2_5_VLChatModel",
]

# The vLLM backends (qwen2_5_vl_vllm.py, gemma3_vllm.py, mistral3_vllm.py) are
# not imported here because importing vllm is slow; rag.py imports them only
# when the "vllm" engine is selected.
