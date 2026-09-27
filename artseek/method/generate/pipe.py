"""Convenience entry point for the ArtSeek pipeline.

    from artseek.method.generate.pipe import get_model
    model = get_model()          # built on first use with the default config

`MODEL` is kept as a lazy alias of `get_model()` for older code.
"""

from .config import get_model, load_artseek
from .graph import build_graph
from .prompts import build_user_content, format_artwork_card, get_system_prompt
from .rag import ArtSeek, Qwen2_5_VLRAGModel
from .tools import get_json_schema_no_state, get_relevant_documents

__all__ = [
    "ArtSeek",
    "Qwen2_5_VLRAGModel",
    "build_graph",
    "build_user_content",
    "format_artwork_card",
    "get_json_schema_no_state",
    "get_model",
    "get_relevant_documents",
    "get_system_prompt",
    "load_artseek",
]


def __getattr__(name):
    if name == "MODEL":
        return get_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
