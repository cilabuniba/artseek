import os
from pathlib import Path


def get_project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Data directory (default `<repo>/data`, override with `ARTSEEK_DATA_DIR`)."""
    return Path(os.environ.get("ARTSEEK_DATA_DIR", get_project_dir() / "data"))


def get_fonts_dir() -> Path:
    return get_project_dir() / "fonts"


def get_store_dir() -> Path:
    return get_project_dir() / "store"


def get_models_dir() -> Path:
    return get_project_dir() / "models"


def get_model_checkpoints_dir() -> Path:
    return get_models_dir() / "checkpoints"
