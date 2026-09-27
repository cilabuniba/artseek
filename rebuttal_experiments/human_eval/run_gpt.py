"""Answers of GPT-5.5 for the human evaluation (OpenAI API).

Same prompt as the local systems (imported from run_local_models.py), one
request per painting, image passed inline, no tools (so no browsing). The
model id and the time of the run are stored in every row. Writes
results/gpt.json.

Needs OPENAI_API_KEY.

Usage:
    python rebuttal_experiments/human_eval/run_gpt.py
"""

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from tqdm import tqdm

HERE = Path(__file__).resolve().parent

# The prompt given to every system.
from run_local_models import PROMPT, PROMPT_VERSION  # noqa: E402

SYSTEM_PROMPT = (
    "You are an expert art historian. Answer questions about paintings based "
    "on the image provided. You have no access to the internet or any tools; "
    "answer from what you can see and what you know."
)


def data_uri(path: Path) -> str:
    return ("data:image/jpeg;base64,"
            + base64.b64encode(path.read_bytes()).decode())


@click.command()
@click.option("--model", default="gpt-5.5-2026-04-23", show_default=True,
              help="Model id (a dated snapshot).")
@click.option("--out-dir", default=str(HERE / "results"), show_default=True)
@click.option("--items-file", default="items.json", show_default=True,
              help="File under data/ listing the items to answer.")
@click.option("--max-tokens", default=4000, show_default=True,
              help="Budget for the whole completion, reasoning tokens included.")
def main(model: str, out_dir: str, max_tokens: int, items_file: str):
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")
    client = OpenAI()

    items = json.loads((HERE / "data" / items_file).read_text())
    out_path = Path(out_dir) / "gpt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else []
    # Only completed answers count as done; errors are retried.
    done = {r["id"] for r in results if r.get("answer")}
    results = [r for r in results if r.get("answer")]

    for it in tqdm(items, desc=model):
        if it["id"] in done:
            continue
        rec = {"id": it["id"], "tier": it["tier"], "title": it["title"],
               "system": "gpt", "model": model, "prompt": PROMPT, "prompt_version": PROMPT_VERSION,
               "run_utc": datetime.now(timezone.utc).isoformat()}
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url",
                         "image_url": {"url": data_uri(HERE / it["image"])}},
                    ]},
                ],
                max_completion_tokens=max_tokens,
                # No `tools` argument at all: nothing to browse with.
            )
            rec |= {
                "answer": (resp.choices[0].message.content or "").strip(),
                "finish_reason": resp.choices[0].finish_reason,
                "seconds": time.perf_counter() - t0,
                "usage": resp.usage.model_dump() if resp.usage else None,
            }
        except Exception as e:  # noqa: BLE001
            rec |= {"error": str(e)}
        results.append(rec)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        time.sleep(1.0)

    ok = sum(1 for r in results if r.get("answer"))
    refused = sum(1 for r in results
                  if r.get("answer") and len(r["answer"]) < 120)
    print(f"\n{ok}/{len(items)} answered -> {out_path}")
    if refused:
        print(f"!! {refused} suspiciously short answers, check for refusals.")


if __name__ == "__main__":
    main()
