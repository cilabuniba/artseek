"""Memorisation probe: can the bare backbone identify the painting?

Qwen2.5-VL-32B is shown only the image and asked for the title and the artist
(JSON, greedy decoding). ground_truth.py judge-probe then flags a painting as
obscure when both are wrong. Writes results/jsonl/obscure_probe.jsonl.

Needs a GPU (no Qdrant). Resumable.

Usage:
    python rebuttal_experiments/artpedia_vqa/obscure_probe.py
    python rebuttal_experiments/artpedia_vqa/obscure_probe.py --limit 10   # quick trial
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402,F401  (sets up sys.path)

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "analysis"))
import lib  # noqa: E402

SYSTEM_PROMPT = (
    "You are an expert art historian. You will be shown one painting. "
    "Identify it from your own knowledge.\n\n"
    "If you recognise the painting, give its commonly used title and the "
    "name of the artist who made it. If you do not recognise it, say so by "
    'using the exact string "UNKNOWN" for the field you cannot fill — do '
    "not guess wildly, but do give your best identification if you have "
    "one.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"title": "<title or UNKNOWN>", "artist": "<artist name or UNKNOWN>"}'
)

USER_PROMPT = "What painting is this, and who painted it?"


def extract_json(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


@click.command()
@click.option("--max-new-tokens", default=256, show_default=True)
@click.option("--limit", type=int, default=None)
def main(max_new_tokens, limit):
    from huggingface_hub import snapshot_download
    from tqdm import tqdm

    from artseek.method.generate.qwen2_5_vl_vllm import Qwen2_5_VLVLLMChatModel

    model_path = snapshot_download("Qwen/Qwen2.5-VL-32B-Instruct-AWQ")
    print(f"probe model: Qwen/Qwen2.5-VL-32B-Instruct-AWQ @ {Path(model_path).name}")
    model = Qwen2_5_VLVLLMChatModel.from_pretrained(
        model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=16384,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 8},
        mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
        seed=lib.SEED,
    )

    out_path = lib.JSONL_DIR / "obscure_probe.jsonl"
    rows = lib.read_jsonl(out_path)
    done = {r["painting_id"] for r in rows}

    vqa = common.load_vqa_dataset()
    items = vqa[:limit] if limit else vqa

    for item in tqdm(items, desc="obscure probe"):
        pid = str(item["id"])
        if pid in done:
            continue
        image = common.load_image_or_none(pid)
        if image is None:
            rows.append({"painting_id": pid, "gt_title": item["title"],
                         "error": "image not downloaded (dead source URL)"})
            lib.write_jsonl(out_path, rows)
            continue
        msgs = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(content=[{"type": "text", "text": "# Painting\n"},
                                  {"type": "image"},
                                  {"type": "text", "text": f"\n{USER_PROMPT}"}]),
        ]
        try:
            resp = model.invoke(msgs, images=[image], max_new_tokens=max_new_tokens)
            _, visible = common.parse_ai_content(resp.content)
            parsed = extract_json(visible) or {}
            rows.append({
                "painting_id": pid,
                "gt_title": item["title"],
                "gt_year": item.get("year"),
                "pred_title": str(parsed.get("title", "")).strip() or None,
                "pred_artist": str(parsed.get("artist", "")).strip() or None,
                "raw": visible[:600],
                "parse_ok": bool(parsed),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"painting_id": pid, "gt_title": item["title"],
                         "error": str(e)[:300]})
        lib.write_jsonl(out_path, rows)

    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
