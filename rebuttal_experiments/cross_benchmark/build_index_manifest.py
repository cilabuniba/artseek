"""Enumerate the retrieval index by Wikipedia page.

Every row of `cilabuniba/wikifragments-visual-arts-embeds` carries the title
of the Wikipedia page it comes from. Projecting the light columns out of the
memory-mapped shards of the prepared dataset gives a manifest of the whole
index (8,166,323 fragments, 973,406 pages) without reading the embeddings.

Output: data/index_manifest.parquet with columns (idx, title, wiki_id,
paragraph_id, url). `idx` is the row index, the same `idx` stored in the
Qdrant payloads and in `retrieved_doc_ids`. Used by coverage_and_routing.py
and by the human-evaluation item selection.

Usage:
    python rebuttal_experiments/cross_benchmark/build_index_manifest.py --num-proc 16
"""

import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HERE = Path(__file__).resolve().parent
# Prepared copy of the dataset in the `datasets` cache (after load_dataset).
CACHE = os.path.join(
    os.environ.get("HF_DATASETS_CACHE",
                   os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets")),
    "cilabuniba___wikifragments-visual-arts-embeds/default/0.0.0/1484bbcff5017927874ab3a87bda0534ecb4e7d0",
)
KEEP = ["title", "url", "wiki_id", "paragraph_id"]


def _shard(args):
    """Project the light columns out of one memory-mapped shard."""
    path, offset = args
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import Dataset

    d = Dataset.from_file(path).select_columns(KEEP).to_dict()
    n = len(d["title"])
    return {
        "idx": list(range(offset, offset + n)),
        "title": d["title"],
        "url": d["url"],
        "wiki_id": d["wiki_id"],
        "paragraph_id": d["paragraph_id"],
    }


@click.command()
@click.option("--cache-dir", default=CACHE, show_default=True)
@click.option("--out", default=str(HERE / "data" / "index_manifest.parquet"),
              show_default=True)
@click.option("--num-proc", default=16, show_default=True)
def main(cache_dir: str, out: str, num_proc: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(glob.glob(
        f"{cache_dir}/wikifragments-visual-arts-embeds-train-*.arrow"))
    lengths = json.loads(
        (Path(cache_dir) / "dataset_info.json").read_text()
    )["splits"]["train"]["shard_lengths"]
    if len(files) != len(lengths):
        raise SystemExit(
            f"{len(files)} shards on disk but {len(lengths)} shard lengths in "
            "dataset_info.json — the row offsets would be wrong, refusing to "
            "write a manifest whose idx column does not match Qdrant.")

    # Row offsets must be the cumulative sum in the SAME order the dataset
    # concatenates shards, which is filename order. Getting this wrong would
    # silently shift every idx and poison the join against Qdrant payloads.
    offsets, run = [], 0
    for n in lengths:
        offsets.append(run)
        run += n
    print(f"{len(files)} shards, {run} rows total")

    with Pool(num_proc) as pool:
        parts = []
        for i, part in enumerate(
                pool.imap(_shard, list(zip(files, offsets)), chunksize=4), 1):
            parts.append(part)
            if i % 50 == 0:
                print(f"  {i}/{len(files)} shards", flush=True)

    table = pa.table({
        k: pa.array([v for p in parts for v in p[k]])
        for k in ("idx", "title", "url", "wiki_id", "paragraph_id")
    })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")

    n_titles = len(set(table["title"].to_pylist()))
    print(f"\nwrote {table.num_rows} rows, {n_titles} distinct pages -> {out}")


if __name__ == "__main__":
    main()
