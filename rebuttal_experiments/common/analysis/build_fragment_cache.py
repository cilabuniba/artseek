"""Cache the text of every WikiFragments row retrieved by any variant.

The runs store only row ids (`retrieved_doc_ids`); the evidence judge needs
the text. Only the light columns of the dataset are read. CPU only and
resumable. Writes results/cache/fragment_texts.jsonl.

Usage:
    ARTSEEK_EXP_DIR=rebuttal_experiments/artpedia_vqa python rebuttal_experiments/common/analysis/build_fragment_cache.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

DATASET = "cilabuniba/wikifragments-visual-arts-embeds"
KEEP_COLUMNS = ["id", "title", "text", "url"]
FLUSH_EVERY = 500


def main():
    runs = lib.load_runs()
    wanted = set()
    for r in runs:
        wanted.update(r.get("retrieved_doc_ids") or [])
    print(f"{len(wanted)} unique fragment ids referenced across all variants")

    out_path = lib.CACHE_DIR / "fragment_texts.jsonl"
    cached = lib.read_jsonl(out_path)
    have = {r["idx"] for r in cached}
    todo = sorted(wanted - have)
    print(f"{len(have)} already cached, {len(todo)} to fetch")
    if not todo:
        return

    from datasets import disable_caching, load_dataset
    from tqdm import tqdm

    disable_caching()
    ds = load_dataset(DATASET)["train"]
    # Drop the heavy columns (embeddings, rendered images).
    ds = ds.select_columns(KEEP_COLUMNS)
    print(f"dataset ready: {ds.num_rows} rows, columns {ds.column_names}")

    for i, idx in enumerate(tqdm(todo, desc="fragments"), start=1):
        try:
            row = ds[int(idx)]
            cached.append(
                {
                    "idx": int(idx),
                    "wiki_id": row.get("id"),
                    "title": row.get("title"),
                    "text": row.get("text"),
                    "url": row.get("url"),
                }
            )
        except Exception as e:  # noqa: BLE001
            cached.append({"idx": int(idx), "error": str(e)[:200],
                           "title": None, "text": "", "url": None})
        if i % FLUSH_EVERY == 0:
            lib.write_jsonl(out_path, cached)

    lib.write_jsonl(out_path, cached)
    print(f"wrote {len(cached)} fragments -> {out_path}")


if __name__ == "__main__":
    main()
