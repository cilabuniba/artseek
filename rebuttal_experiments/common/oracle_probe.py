"""Oracle retrieval probe: query the index with the question + reference answer.

For every question of a benchmark, retrieves the top-N fragments for the text
query "<question> <reference answer>" and stores their ids in
results/jsonl/oracle_probe.jsonl. Judging them with `judge.py --pass evidence
--variant oracle` gives an upper bound on what better queries could retrieve
with this retriever (and a lower bound on the index coverage).

Needs a GPU and a running Qdrant server. Set ARTSEEK_EXP_DIR to the benchmark
folder (default: artpedia_vqa).

Usage:
    python rebuttal_experiments/common/oracle_probe.py [--limit N]
"""

import common  # noqa: F401  (sets up sys.path)

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import lib  # noqa: E402


@click.command()
@click.option("--top-n", default=10, show_default=True,
              help="Documents per probe; matches the pipeline's own limit.")
@click.option("--limit", type=int, default=None)
@click.option("--source-variant", default="full_classify", show_default=True,
              help="Variant whose question/reference rows define the probe set. "
                   "Any variant works — the probe does not use its answers.")
def main(top_n, limit, source_variant):
    from artseek.method.generate.config import COLLECTION_NAME, RETRIEVER_ID
    from artseek.method.retrieve import ColQwen2Qdrant
    from tqdm import tqdm

    rows = [r for r in lib.load_runs([source_variant]) if r["status"] == "ok"]
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} queries to probe (from `{source_variant}`)")

    out_path = lib.JSONL_DIR / "oracle_probe.jsonl"
    done = {r["query_id"] for r in lib.read_jsonl(out_path)}
    todo = [r for r in rows if r["query_id"] not in done]
    print(f"{len(done)} already probed, {len(todo)} to go")
    if not todo:
        return

    retriever = ColQwen2Qdrant(
        os.environ.get("ARTSEEK_RETRIEVER", RETRIEVER_ID), COLLECTION_NAME
    )

    results = lib.read_jsonl(out_path)
    for i, r in enumerate(tqdm(todo, desc="oracle probe"), start=1):
        # Text-only query built from the gold answer.
        query = f"{r['question']} {r['reference_answer']}"
        try:
            embeds = retriever.embed([query], None)
            resp = retriever.query(embeds[0], prefetch_limit=100, limit=top_n)
            idxs = [p.payload["idx"] for p in resp.points]
            scores = [p.score for p in resp.points]
            results.append({
                "query_id": r["query_id"],
                "painting_id": r["painting_id"],
                "question_type": r["question_type"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "variant": "oracle",
                "retrieved_doc_ids": idxs,
                "scores": scores,
                "status": "ok",
                "num_tool_calls": 1,
                "retrieved": True,
            })
        except Exception as e:  # noqa: BLE001
            results.append({"query_id": r["query_id"], "variant": "oracle",
                            "status": "failed", "error": str(e)[:200],
                            "retrieved_doc_ids": []})
        if i % 50 == 0:
            lib.write_jsonl(out_path, results)

    lib.write_jsonl(out_path, results)
    print(f"wrote {len(results)} probes -> {out_path}")


if __name__ == "__main__":
    main()
