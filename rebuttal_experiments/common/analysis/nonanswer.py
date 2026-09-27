"""Non-answers: items where the model produced no final answer.

The judge skips items with an empty answer. This script reports the
non-answer rate per variant and re-runs the paired comparisons with
non-answers scored as correctness 0.

Usage:
    python rebuttal_experiments/common/analysis/nonanswer.py [--judge phi4]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# (baseline, comparison) — a positive delta favours the second.
PAIRS = [
    ("full_systemprompt", "full_classify"),
    ("full_noclassify", "full_classify"),
]


def nonanswers(variant: str) -> set[str]:
    """query_ids where the run produced no final answer (and no error)."""
    p = lib.RESULTS_DIR / f"{variant}.json"
    if not p.exists():
        return set()
    return {
        f"{e['id']}_{e['question_type']}"
        for e in json.loads(p.read_text())
        if "error" not in e and not str(e.get("model_answer", "")).strip()
    }


def totals(variant: str) -> int:
    p = lib.RESULTS_DIR / f"{variant}.json"
    if not p.exists():
        return 0
    return sum(1 for e in json.loads(p.read_text()) if "error" not in e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    J = lib.load_judge(judge, "answer")
    by = {}
    for (v, q), r in J.items():
        by.setdefault(v, {})[q] = r

    payload = {"judge_model": judge, "rate": {}, "sensitivity": {}}
    out = ["#### How often does each variant fail to answer at all?\n"]

    rows = []
    for v in lib.VARIANTS:
        n = totals(v)
        if not n:
            continue
        e = nonanswers(v)
        payload["rate"][v] = {"n_runs": n, "n_nonanswer": len(e),
                              "pct": len(e) / n * 100}
        rows.append([f"`{v}`", n, len(e), lib.fmt(len(e) / n * 100, 1) + "%"])
    out.append(lib.table_block(
        "nonanswer_rate",
        ["variant", "successful runs", "no final answer", "rate"], rows))
    out.append(
        "\n*A non-answer is a run that raised no error and produced reasoning "
        "but terminated with an empty answer. The judging pass skips these, "
        "so they are absent from every correctness number elsewhere in this "
        "document.*")

    # ── sensitivity of the paired comparisons ────────────────────────────
    rows = []
    for base, comp in PAIRS:
        A, B = by.get(base), by.get(comp)
        if not A or not B:
            continue
        zA = {q: 0 for q in nonanswers(base)}
        zB = {q: 0 for q in nonanswers(comp)}
        for label, addA, addB in (("as judged (non-answers dropped)", {}, {}),
                                  ("non-answers scored 0", zA, zB)):
            sA = {q: r["correctness"] for q, r in A.items()} | addA
            sB = {q: r["correctness"] for q, r in B.items()} | addB
            common = sorted(set(sA) & set(sB))
            if not common:
                continue
            r = lib.cluster_bootstrap_paired(
                [(lib.painting_of(q), sB[q] - sA[q]) for q in common],
                n_boot=a.n_boot)
            sig = not (r["ci_low"] <= 0 <= r["ci_high"])
            payload["sensitivity"][f"{base}->{comp}|{label}"] = {**r, "significant": sig}
            rows.append([
                f"`{base}` → `{comp}`", label, r["n"],
                lib.fmt(sum(sA[q] for q in common) / len(common)),
                lib.fmt(sum(sB[q] for q in common) / len(common)),
                f"{r['mean']:+.3f}", lib.fmt_ci(r),
                "**yes**" if sig else "no"])
    out.append("\n#### The comparison under both accountings\n")
    out.append(lib.table_block(
        "nonanswer_sensitivity",
        ["comparison", "accounting", "n", "baseline", "comparison",
         "mean Δ", "95% cluster CI", "CI excludes 0"], rows))

    lib.write_table(f"nonanswer__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
