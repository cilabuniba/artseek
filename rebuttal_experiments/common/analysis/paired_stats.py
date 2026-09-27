"""Paired comparisons between variants: mean difference with a painting-level
cluster-bootstrap 95% CI, Cohen's d_z, and (uncorrected for clustering)
paired t-test and Wilcoxon p-values.

Each painting contributes several questions, so the bootstrap resamples
paintings. The bootstrap CI is the reported inference.

Usage:
    python rebuttal_experiments/common/analysis/paired_stats.py [--judge phi4]
                                                                [--subset obscure|nonobscure]
                                                                [--n-boot 10000]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# Bonferroni families, stated per table.
FAMILIES = {
    "retrieval_vs_base": {
        "title": "Retrieval vs. `base`",
        "pairs": [("base", v) for v in
                  ("full_classify", "full_noclassify", "full_systemprompt",
                   "full_classify_noartist", "full_alwaysretrieve")],
        "split_by_type": True,
    },
    "policy": {
        "title": "Tool-calling policy: ICL shot vs. system prompt",
        "pairs": [("full_systemprompt", "full_classify")],
        "split_by_type": False,
    },
    "classification": {
        "title": "Classification / artist-field ablation",
        "pairs": [("full_classify", "full_classify_noartist"),
                  ("full_classify_noartist", "full_noclassify"),
                  ("full_classify", "full_noclassify")],
        "split_by_type": False,
    },
    "alwaysretrieve": {
        "title": "Learned policy vs. forced retrieval",
        "pairs": [("full_alwaysretrieve", "full_classify")],
        "split_by_type": True,
    },
}


def compare(scored, a, b, field, qtype=None, n_boot=10000):
    """Mean difference b - a on `field`, with a painting-level cluster CI."""
    if a not in scored or b not in scored:
        return None
    A, B = scored[a], scored[b]
    if qtype:
        A = {k: v for k, v in A.items() if v["question_type"] == qtype}
        B = {k: v for k, v in B.items() if v["question_type"] == qtype}
    pairs = lib.paired_diffs(A, B, field)
    if not pairs:
        return None
    res = lib.cluster_bootstrap_paired(pairs, n_boot=n_boot)
    diffs = [d for _, d in pairs]
    res["p_ttest"] = lib.paired_ttest(diffs)
    res["p_wilcoxon"] = lib.wilcoxon(diffs)
    res["a"], res["b"], res["field"] = a, b, field
    res["question_type"] = qtype or "all"
    res["significant"] = not (res["ci_low"] <= 0 <= res["ci_high"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--subset", default=None, choices=["obscure", "nonobscure"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    scored = lib.load_scored(judge)
    tag = a.judge
    if a.subset:
        obs = lib.read_jsonl(lib.JSONL_DIR / "obscure.jsonl")
        if not obs:
            raise SystemExit("no obscure.jsonl yet — run artpedia_vqa/obscure_probe.py and ground_truth.py first")
        want = {r["painting_id"] for r in obs
                if r["is_obscure"] == (a.subset == "obscure")}
        scored = {v: {q: r for q, r in d.items() if r["painting_id"] in want}
                  for v, d in scored.items()}
        tag += f"__{a.subset}"

    out = []
    payload = {"judge_model": judge, "subset": a.subset or "all",
               "n_boot": a.n_boot, "families": {}}

    for fam_key, fam in FAMILIES.items():
        pairs = [(x, y) for x, y in fam["pairs"] if x in scored and y in scored]
        if not pairs:
            continue
        # Aggregate rows and per-question-type rows are separate families.
        n_agg = len(pairs)
        alpha_agg = 0.05 / n_agg
        out.append(f"\n#### {fam['title']}\n")
        if fam["split_by_type"]:
            n_split = len(pairs) * 2
            alpha_split = 0.05 / n_split
            note = (
                f"*Bonferroni families: the `all` rows are a {n_agg}-comparison "
                f"family (α={alpha_agg:.4f}); the visual/contextual rows are a "
                f"separate {n_split}-comparison family (α={alpha_split:.4f}).*"
            )
        else:
            note = f"*Bonferroni family: {n_agg} comparison(s), α={alpha_agg:.4f}.*"
        out.append(
            f"{note} *CIs are painting-level cluster bootstrap over "
            f"{a.n_boot} resamples, and are the reported inference; the "
            f"t-test column is an uncorrected legacy check.*\n"
        )
        fam_json = []
        for field in ("correctness", "recall"):
            rows = []
            for x, y in pairs:
                types = [None, "visual", "contextual"] if fam["split_by_type"] else [None]
                for qt in types:
                    r = compare(scored, x, y, field, qt, a.n_boot)
                    if r is None:
                        continue
                    fam_json.append(r)
                    rows.append([
                        f"`{x}` → `{y}`",
                        r["question_type"],
                        r["n"], r["n_clusters"],
                        f"{r['mean']:+.3f}",
                        lib.fmt_ci(r),
                        lib.fmt(r["cohens_dz"], 3),
                        lib.fmt_p(r["p_ttest"]),
                        "yes" if r["significant"] else "**no**",
                    ])
            if rows:
                out.append(f"\n**Δ {field}**\n")
                out.append(lib.table_block(
                    f"paired__{fam_key}__{field}",
                    ["comparison", "questions", "n", "paintings", "mean Δ",
                     "95% cluster CI", "d_z", "p (t-test, uncorrected)",
                     "CI excludes 0"],
                    rows,
                ))
        payload["families"][fam_key] = {
            "title": fam["title"],
            "n_tests_aggregate": n_agg, "alpha_aggregate": alpha_agg,
            "n_tests_split": (len(pairs) * 2) if fam["split_by_type"] else None,
            "alpha_split": (0.05 / (len(pairs) * 2)) if fam["split_by_type"] else None,
            "results": fam_json,
        }

    lib.write_table(f"paired_stats__{tag}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
