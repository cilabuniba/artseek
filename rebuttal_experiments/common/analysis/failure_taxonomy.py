"""Failed items: counts per variant and category, cross-tabs against question
type and reference length, and a worst-case version of the main table where
every item a variant lost is scored 0.

Usage:
    python rebuttal_experiments/common/analysis/failure_taxonomy.py [--judge phi4]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib
from build_runs_jsonl import FAILURE_CATEGORIES

CAT_LABEL = {
    "image_download_failure": "image download failure",
    "context_length_overflow": "context-length overflow",
    "answer_truncated_empty": "answer truncated (empty after `</think>`)",
    "tool_call_cap_exceeded": "tool-call cap exceeded",
    "malformed_tool_call": "malformed tool call",
    "generation_timeout": "generation timeout",
    "classification_failure": "classification failure",
    "other_runtime_error": "other runtime error",
    "judge_parse_failure": "judge parse failure",
}


def build(judge: str):
    runs = lib.load_runs()
    variants = [v for v in lib.VARIANTS if any(r["variant"] == v for r in runs)]
    judged = lib.load_judge(judge, "answer")

    by_variant = defaultdict(list)
    for r in runs:
        by_variant[r["variant"]].append(r)

    # Skip variants that have not been judged yet.
    variants = [v for v in variants
                if any((v, r["query_id"]) in judged for r in by_variant[v])]

    # ── 1. counts per category ───────────────────────────────────────────
    counts = {v: {c: 0 for c in list(FAILURE_CATEGORIES) + ["judge_parse_failure"]}
              for v in variants}
    lost_items = defaultdict(list)  # variant -> [(row, category)]
    for v in variants:
        for r in by_variant[v]:
            if r["status"] == "failed":
                counts[v][r["failure_category"]] += 1
                lost_items[v].append((r, r["failure_category"]))
            elif (v, r["query_id"]) not in judged:
                counts[v]["judge_parse_failure"] += 1
                lost_items[v].append((r, "judge_parse_failure"))

    all_cats = [c for c in list(FAILURE_CATEGORIES) + ["judge_parse_failure"]]
    headers = ["failure mode"] + [f"`{v}`" for v in variants]
    rows = []
    for c in all_cats:
        rows.append([CAT_LABEL[c]] + [counts[v][c] for v in variants])
    rows.append(["**total lost**"] + [f"**{sum(counts[v].values())}**" for v in variants])
    rows.append(["**n scored**"] + [
        f"**{678 - sum(counts[v].values())}**" for v in variants
    ])
    tbl_counts = lib.table_block(
        "failure_counts",
        headers, rows)

    # ── 2. cross-tab: failure vs question class ──────────────────────────
    # Image-download failures are excluded: they are the same for every
    # variant and question type.
    ct_headers = ["variant", "lost (visual)", "lost (contextual)",
                  "mean ref-answer words (lost)", "mean ref-answer words (kept)"]
    ct_rows = []
    ct_json = {}
    for v in variants:
        lost = [r for r, c in lost_items[v] if c != "image_download_failure"]
        kept = [r for r in by_variant[v]
                if r["status"] == "ok" and (v, r["query_id"]) in judged]
        lv = sum(1 for r in lost if r["question_type"] == "visual")
        lc = sum(1 for r in lost if r["question_type"] == "contextual")
        wl = [r["reference_answer_words"] for r in lost if r["reference_answer_words"]]
        wk = [r["reference_answer_words"] for r in kept if r["reference_answer_words"]]
        mean_l = sum(wl) / len(wl) if wl else float("nan")
        mean_k = sum(wk) / len(wk) if wk else float("nan")
        ct_rows.append([f"`{v}`", lv, lc, lib.fmt(mean_l, 1), lib.fmt(mean_k, 1)])
        ct_json[v] = {"lost_visual": lv, "lost_contextual": lc,
                      "mean_ref_words_lost": mean_l, "mean_ref_words_kept": mean_k,
                      "n_lost": len(lost), "n_kept": len(kept)}
    tbl_crosstab = lib.table_block(
        "failure_crosstab",
        ct_headers, ct_rows)

    # ── 3. worst-case imputation ─────────────────────────────────────────
    # Every item a variant lost (except dead image links, shared by all
    # variants) is scored 0 / 0.0, so all variants have the same denominator.
    dead_images = {r["painting_id"] for r in runs
                   if r["failure_category"] == "image_download_failure"}
    universe = sorted({r["query_id"] for r in runs
                       if r["painting_id"] not in dead_images})

    imp_rows = []
    imp_json = {}
    for v in variants:
        got = {r["query_id"]: judged[(v, r["query_id"])]
               for r in by_variant[v]
               if r["status"] == "ok" and (v, r["query_id"]) in judged}
        rec = [got[q]["recall"] if q in got else 0.0 for q in universe]
        cor = [got[q]["correctness"] if q in got else 0 for q in universe]
        obs_rec = [got[q]["recall"] for q in universe if q in got]
        obs_cor = [got[q]["correctness"] for q in universe if q in got]
        imp_rows.append([
            f"`{v}`", len(universe), len(got),
            lib.fmt(sum(obs_rec) / len(obs_rec)), lib.fmt(sum(obs_cor) / len(obs_cor)),
            lib.fmt(sum(rec) / len(rec)), lib.fmt(sum(cor) / len(cor)),
        ])
        imp_json[v] = {
            "n_universe": len(universe), "n_observed": len(got),
            "observed_recall": sum(obs_rec) / len(obs_rec),
            "observed_correctness": sum(obs_cor) / len(obs_cor),
            "imputed_recall": sum(rec) / len(rec),
            "imputed_correctness": sum(cor) / len(cor),
        }
    tbl_imp = lib.table_block(
        "failure_imputation",
        ["variant", "n (common universe)", "n scored", "avg recall (observed)",
         "avg correctness (observed)", "avg recall (worst-case)",
         "avg correctness (worst-case)"],
        imp_rows,
    )

    rank_obs = sorted(imp_json, key=lambda v: -imp_json[v]["observed_correctness"])
    rank_imp = sorted(imp_json, key=lambda v: -imp_json[v]["imputed_correctness"])

    lib.write_table(f"failure_taxonomy__{judge.replace('/', '_')}", {
        "judge_model": judge,
        "counts": counts,
        "crosstab": ct_json,
        "worst_case_imputation": imp_json,
        "ranking_observed": rank_obs,
        "ranking_worst_case": rank_imp,
        "ranking_preserved": rank_obs == rank_imp,
        "n_universe": len(universe),
    })

    out = [
        f"### Failure taxonomy (judge: `{judge}`)\n",
        tbl_counts,
        "\n### Do failures concentrate on harder items?\n",
        tbl_crosstab,
        "\n### Worst-case imputation (every lost item scored 0 / 0.0)\n",
        tbl_imp,
        f"\nranking (observed):  {' > '.join(rank_obs)}",
        f"\nranking (worst-case): {' > '.join(rank_imp)}",
        f"\nranking preserved: {rank_obs == rank_imp}",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4
    print(build(judge))
