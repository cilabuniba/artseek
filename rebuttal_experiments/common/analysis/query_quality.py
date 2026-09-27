"""Query quality: evidence recall of the documents retrieved by two variants,
paired over the questions for which both retrieved (so the retrieve/skip
decision is held fixed).

Usage:
    python rebuttal_experiments/common/analysis/query_quality.py [--judge phi4]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# (baseline, comparison) — a positive delta favours the second.
PAIRS = [
    ("full_systemprompt", "full_classify"),
    ("full_alwaysretrieve", "full_classify"),
    ("full_noclassify", "full_classify"),
    ("full_classify_noartist", "full_classify"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    ev = lib.load_judge(judge, "evidence")
    if not ev:
        raise SystemExit("no evidence judgments yet")

    by_variant = {}
    for (v, q), row in ev.items():
        by_variant.setdefault(v, {})[q] = row

    # ── per-variant retrieval quality ────────────────────────────────────
    rows, payload = [], {"judge_model": judge, "per_variant": {}, "paired": {}}
    for v in lib.VARIANTS:
        d = by_variant.get(v)
        if not d:
            continue
        vals = list(d.values())
        e = {
            "n": len(vals),
            "mean_evidence_recall": sum(r["evidence_recall"] for r in vals) / len(vals),
            "pct_present": sum(1 for r in vals if r["evidence_present"] == 2) / len(vals) * 100,
            "pct_absent": sum(1 for r in vals if r["evidence_present"] == 0) / len(vals) * 100,
        }
        payload["per_variant"][v] = e
        rows.append([f"`{v}`", e["n"], lib.fmt(e["mean_evidence_recall"]),
                     lib.fmt(e["pct_present"], 1) + "%",
                     lib.fmt(e["pct_absent"], 1) + "%"])
    out = ["#### Retrieval quality: how good is what each policy brings back?\n"]
    out.append(lib.table_block(
        "query_quality_per_variant",
        ["variant", "retrievals judged", "mean evidence recall",
         "% fully present", "% absent"], rows))

    # ── paired, holding the retrieval decision fixed ─────────────────────
    rows = []
    for base, comp in PAIRS:
        A, B = by_variant.get(base), by_variant.get(comp)
        if not A or not B:
            continue
        common = sorted(set(A) & set(B))
        if not common:
            continue
        for field, label in (("evidence_recall", "evidence recall"),
                             ("evidence_present", "evidence present (0–2)")):
            pairs = [(lib.painting_of(q), B[q][field] - A[q][field]) for q in common]
            r = lib.cluster_bootstrap_paired(pairs, n_boot=a.n_boot)
            sig = not (r["ci_low"] <= 0 <= r["ci_high"])
            payload["paired"][f"{base}->{comp}|{field}"] = {**r, "significant": sig}
            rows.append([f"`{base}` → `{comp}`", label, r["n"], r["n_clusters"],
                         f"{r['mean']:+.3f}", lib.fmt_ci(r),
                         lib.fmt(r["cohens_dz"]), "**yes**" if sig else "no"])
    out.append("\n#### Paired over queries both variants retrieved for\n")
    out.append(lib.table_block(
        "query_quality_paired",
        ["comparison", "metric", "n", "paintings", "mean Δ", "95% cluster CI",
         "d_z", "CI excludes 0"], rows))
    out.append(
        "\n*Restricted to queries where **both** variants issued a retrieval, "
        "so this isolates the quality of a retrieval from the decision to "
        "make one. A positive Δ means the second variant's retrievals "
        "contained more of what the answer needed.*"
    )

    lib.write_table(f"query_quality__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
