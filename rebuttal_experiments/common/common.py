"""Shared helpers for the experiment scripts.

Scripts are run directly (`python rebuttal_experiments/common/<script>.py`),
so importing this module first puts the repository root on `sys.path`.
"""

import json
import os
import sys
import time
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
for _p in (str(_THIS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
from PIL import Image

# HF_HOME, QDRANT_URL, ... from the repository's .env (see .env.example).
load_dotenv(_REPO_ROOT / ".env")

# Some Wikimedia scans exceed PIL's decompression-bomb pixel limit.
Image.MAX_IMAGE_PIXELS = None

ARTPEDIA_JSON = _REPO_ROOT / "data" / "external" / "artpedia" / "artpedia.json"
ARTPEDIA_IMAGES_DIR = _REPO_ROOT / "data" / "external" / "artpedia" / "images"
# Images are looked up as <IMAGES_DIR>/<id>.jpg; set ARTSEEK_IMAGES_DIR for the
# other benchmarks.
IMAGES_DIR = Path(os.environ.get("ARTSEEK_IMAGES_DIR", ARTPEDIA_IMAGES_DIR))
# Defaults: the ArtPedia-VQA benchmark.
ARTPEDIA_VQA_DIR = _THIS_DIR.parent / "artpedia_vqa"
VQA_DATASET_PATH = ARTPEDIA_VQA_DIR / "data" / "artpedia_vqa.json"
RESULTS_DIR = ARTPEDIA_VQA_DIR / "results"


def load_artpedia_test_split() -> dict:
    """Load the test-split entries of artpedia.json, keyed by their id."""
    with open(ARTPEDIA_JSON, "r") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if v.get("split") == "test"}


def load_vqa_dataset(path: str | Path = VQA_DATASET_PATH) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def load_image(artpedia_id: str) -> Image.Image:
    path = IMAGES_DIR / f"{artpedia_id}.jpg"
    return Image.open(path).convert("RGB")


def load_image_or_none(artpedia_id: str) -> Image.Image | None:
    """Like load_image, but None when the file is missing (dead source URL)."""
    path = IMAGES_DIR / f"{artpedia_id}.jpg"
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def timed(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), returning (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def parse_ai_content(content: str) -> tuple[str | None, str]:
    """Split an AI message into (thinking, visible_response).

    The model often omits the opening <think> tag, so everything before
    </think> is treated as reasoning.
    """
    idx = content.find("</think>")
    if idx != -1:
        thinking = content[:idx].replace("<think>", "").strip()
        visible = content[idx + len("</think>"):].strip()
    else:
        thinking = None
        visible = content.strip()
    return thinking, visible


def split_reasoning_duration(
    raw_content: str, total_duration: float, tokenizer
) -> tuple[float, float]:
    """Split `total_duration` between the <think> part and the answer,
    proportionally to their token counts (an estimate: generation has no
    per-token timestamps)."""
    idx = raw_content.find("</think>")
    if idx == -1:
        return 0.0, total_duration
    thinking_part = raw_content[:idx]
    answer_part = raw_content[idx + len("</think>"):]
    n_think = len(tokenizer.encode(thinking_part)) if thinking_part.strip() else 0
    n_answer = len(tokenizer.encode(answer_part)) if answer_part.strip() else 0
    total_tokens = n_think + n_answer
    if total_tokens == 0:
        return 0.0, total_duration
    reasoning_duration = total_duration * (n_think / total_tokens)
    return reasoning_duration, total_duration - reasoning_duration


def _prob_value(prob) -> float:
    return prob.item() if hasattr(prob, "item") else float(prob)


def build_first_user_message(prompt: str, card: dict | None):
    """The user message: image placeholder, optional artwork card, query."""
    from langchain_core.messages import HumanMessage

    parts: list[dict] = [
        {"type": "text", "text": "# Current query image\n"},
        {"type": "image"},
    ]
    if card:
        card_text = "\n# Artwork card"
        for k, v in card.items():
            card_text += f"\n{k}: "
            for i, (pred, prob) in enumerate(v):
                card_text += f"{pred} ({int(_prob_value(prob) * 100)}%)"
                if i < len(v) - 1:
                    card_text += ", "
        parts.append({"type": "text", "text": card_text})
    parts.append({"type": "text", "text": f"\n# Query\n{prompt}"})
    return HumanMessage(content=parts)


def serialize_card(card: dict | None) -> dict | None:
    """Convert a classify() result (task -> [(label, prob_tensor), ...]) into
    plain JSON-serializable types."""
    if card is None:
        return None
    return {
        task: [[label, _prob_value(prob)] for label, prob in preds]
        for task, preds in card.items()
    }


def load_existing_results(out_path: str | Path) -> list[dict]:
    out_path = Path(out_path)
    if out_path.exists():
        with open(out_path, "r") as f:
            return json.load(f)
    return []


def save_results(out_path: str | Path, results: list[dict]):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
