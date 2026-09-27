"""Default ArtSeek configuration and model factory.

Every resource is fetched from the Hugging Face Hub (and cached under
`HF_HOME`) the first time it is needed. A few settings can be overridden with
environment variables, which is convenient for the demo:

    QDRANT_URL                        Qdrant server (default http://localhost)
    ARTSEEK_ENGINE                    "vllm" (default) or "hf"
    ARTSEEK_RETRIEVER                 ColQwen2 checkpoint (hub id or local path)
    ARTSEEK_GPU_MEMORY_UTILIZATION    vLLM GPU memory fraction (default 0.65)
    ARTSEEK_MAX_MODEL_LEN             vLLM context length (default 16384)
"""

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

MLLM_ID = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
RETRIEVER_ID = "vidore/colqwen2-v1.0"
COLLECTION_NAME = "wikifragments-visual-arts-embeds"
DATASET_ID = "cilabuniba/wikifragments-visual-arts-embeds"
LICN_ID = "cilabuniba/artseek-licn"
LICN_DATA_ID = "cilabuniba/artseek-licn-data"
SHOT_DIR = Path(__file__).resolve().parent / "shot"

LICN_KWARGS = {
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
}
LICN_LOSS_KWARGS = {"num_tasks": 5}

# vllm.LLM(...) arguments. The retriever and LICN share the GPU with vLLM, so
# vLLM must not take all of it: weights (~19.5 GB) plus a 16k-token KV cache for
# 4 sequences fit in 65% of a 64 GB A100 and leave room for ColQwen2.
VLLM_KWARGS = {
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.65,
    "max_model_len": 16384,
    "max_num_seqs": 4,
    # Shot images + input image + images from up to 5 retrieval calls.
    "limit_mm_per_prompt": {"image": 128},
    # Same image token budget as the transformers processor.
    "mm_processor_kwargs": {"max_pixels": 1280 * 28 * 28},
    # Quantization is auto-detected (AWQ, upgraded to awq_marlin on Ampere+).
    # If CUDA graph capture runs out of memory, add "enforce_eager": True.
}

# Qwen2_5_VLForConditionalGeneration.from_pretrained(...) arguments.
HF_KWARGS = {
    "torch_dtype": torch.bfloat16,
    "attn_implementation": "flash_attention_2",
    "device_map": "auto",
}


def load_artseek(engine: str | None = None, model_kwargs: dict | None = None, **kwargs):
    """Build the ArtSeek pipeline with the paper's configuration.

    Args:
        engine: "vllm" or "hf". Defaults to `$ARTSEEK_ENGINE`, then "vllm".
        model_kwargs: Replace the backbone kwargs (`VLLM_KWARGS` / `HF_KWARGS`).
        **kwargs: Override any other `Qwen2_5_VLRAGModel` argument.

    Returns:
        A `Qwen2_5_VLRAGModel`.
    """
    from .rag import Qwen2_5_VLRAGModel

    engine = engine or os.environ.get("ARTSEEK_ENGINE", "vllm")
    if model_kwargs is None:
        if engine == "vllm":
            model_kwargs = dict(VLLM_KWARGS)
            if "ARTSEEK_GPU_MEMORY_UTILIZATION" in os.environ:
                model_kwargs["gpu_memory_utilization"] = float(
                    os.environ["ARTSEEK_GPU_MEMORY_UTILIZATION"]
                )
            if "ARTSEEK_MAX_MODEL_LEN" in os.environ:
                model_kwargs["max_model_len"] = int(os.environ["ARTSEEK_MAX_MODEL_LEN"])
        else:
            model_kwargs = dict(HF_KWARGS)

    config = dict(
        retriever_pretrained_model_name_or_path=os.environ.get(
            "ARTSEEK_RETRIEVER", RETRIEVER_ID
        ),
        retriever_collection_name=COLLECTION_NAME,
        engine=engine,
        model_kwargs=model_kwargs,
        licn_kwargs=LICN_KWARGS,
        licn_loss_kwargs=LICN_LOSS_KWARGS,
        dataset_path=DATASET_ID,
        shot_path=SHOT_DIR,
    )
    config.update(kwargs)
    # Resolved only when not overridden, so nothing unused is downloaded.
    if "model_pretrained_model_name_or_path" not in config:
        config["model_pretrained_model_name_or_path"] = snapshot_download(MLLM_ID)
    if "licn_pretrained_path" not in config:
        config["licn_pretrained_path"] = snapshot_download(LICN_ID)
    if "artgraph_path" not in config:
        config["artgraph_path"] = snapshot_download(LICN_DATA_ID, repo_type="dataset")
    return Qwen2_5_VLRAGModel(**config)


_MODEL = None


def get_model():
    """Process-wide ArtSeek instance, built with `load_artseek()` on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_artseek()
    return _MODEL
