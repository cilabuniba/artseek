"""Consolidate results/<variant>.json into results/jsonl/runs.jsonl.

One row per (variant, query_id) with every field the analysis scripts need,
including a failure category for items without an answer. Rows of the oracle
probe (results/jsonl/oracle_probe.jsonl), if present, are added as variant
"oracle". Run this before judge.py and before any analysis.

Usage:
    ARTSEEK_EXP_DIR=rebuttal_experiments/aqua python rebuttal_experiments/common/analysis/build_runs_jsonl.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# Failure taxonomy, first matching rule wins. Categories that never occur are
# still reported, with a count of 0.
FAILURE_CATEGORIES = (
    "image_download_failure",
    "context_length_overflow",
    "answer_truncated_empty",
    "tool_call_cap_exceeded",
    "malformed_tool_call",
    "generation_timeout",
    "classification_failure",
    "other_runtime_error",
)


def classify_failure(entry: dict) -> tuple[str, str] | tuple[None, None]:
    """Return (category, detail) for a failed item, or (None, None) if ok."""
    err = entry.get("error")
    if err:
        e = str(err)
        if "image not downloaded" in e:
            return "image_download_failure", e
        if "longer than the maximum model length" in e:
            return "context_length_overflow", e
        if "classify() failed" in e:
            return "classification_failure", e
        if "timeout" in e.lower():
            return "generation_timeout", e
        if "tool" in e.lower() and ("parse" in e.lower() or "malformed" in e.lower()):
            return "malformed_tool_call", e
        return "other_runtime_error", e

    # No exception but no answer: the model spent max_new_tokens inside
    # <think>. The judge skips these, so they are counted here.
    if not entry.get("model_answer"):
        detail = "empty visible answer after </think> (max_new_tokens exhausted by reasoning)"
        return "answer_truncated_empty", detail

    return None, None


def top_pred(card: dict | None, task: str):
    if not card or task not in card or not card[task]:
        return None, None
    label, prob = card[task][0]
    return label, float(prob)


def main():
    rows = []
    for variant in lib.VARIANTS:
        path = lib.RESULTS_DIR / f"{variant}.json"
        if not path.exists():
            print(f"  (skip {variant}: no {path.name} yet)")
            continue
        entries = json.loads(path.read_text())
        for e in entries:
            qid = lib.query_id(e["id"], e["question_type"])
            cat, detail = classify_failure(e)
            card = e.get("card")
            artist, artist_conf = top_pred(card, "artist")
            style, _ = top_pred(card, "style")
            genre, _ = top_pred(card, "genre")
            ref = e.get("reference_answer") or ""
            doc_ids = e.get("retrieved_doc_ids") or []
            row = {
                "query_id": qid,
                "painting_id": str(e["id"]),
                "variant": variant,
                "question_type": e["question_type"],
                "title": e.get("title"),
                "question": e.get("question"),
                "reference_answer": ref,
                "reference_answer_words": len(ref.split()) if ref else 0,
                "model_answer": e.get("model_answer"),
                "model_reasoning": e.get("model_reasoning"),
                "status": "ok" if cat is None else "failed",
                "failure_category": cat,
                "failure_detail": detail,
                "num_tool_calls": e.get("num_tool_calls"),
                "retrieved": (bool(e.get("num_tool_calls")) if cat is None else None),
                "retrieved_doc_ids": doc_ids,
                "n_unique_docs": len(set(doc_ids)),
                "licn_artist": artist,
                "licn_artist_conf": artist_conf,
                "licn_style": style,
                "licn_genre": genre,
                "card": card,
                "timing": e.get("timing"),
                "force_mechanism": e.get("force_mechanism"),
            }
            rows.append(row)
        print(f"  {variant}: {len(entries)} rows")

    # The oracle probe, as a pseudo-variant without answers.
    oracle = lib.read_jsonl(lib.JSONL_DIR / "oracle_probe.jsonl")
    for o in oracle:
        doc_ids = o.get("retrieved_doc_ids") or []
        rows.append({
            "query_id": o["query_id"],
            "painting_id": o.get("painting_id"),
            "variant": lib.ORACLE_VARIANT,
            "question_type": o.get("question_type"),
            "title": None,
            "question": o.get("question"),
            "reference_answer": o.get("reference_answer", ""),
            "reference_answer_words": len((o.get("reference_answer") or "").split()),
            "model_answer": None,
            "model_reasoning": None,
            "status": o.get("status", "ok"),
            "failure_category": None if o.get("status") == "ok" else "other_runtime_error",
            "failure_detail": o.get("error"),
            "num_tool_calls": 1 if doc_ids else 0,
            "retrieved": bool(doc_ids),
            "retrieved_doc_ids": doc_ids,
            "n_unique_docs": len(set(doc_ids)),
            "licn_artist": None, "licn_artist_conf": None,
            "licn_style": None, "licn_genre": None, "card": None,
            "timing": None, "force_mechanism": None,
        })
    if oracle:
        print(f"  oracle probe: {len(oracle)} rows merged")

    rows.sort(key=lambda r: (r["variant"], r["painting_id"] or "", r["question_type"] or ""))
    lib.write_jsonl(lib.JSONL_DIR / "runs.jsonl", rows)
    print(f"wrote {len(rows)} rows -> {lib.JSONL_DIR / 'runs.jsonl'}")


if __name__ == "__main__":
    main()
