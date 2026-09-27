"""Sensitivity of the confidence intervals to the resampling unit.

Compares, for the paired differences and for each variant's mean, the 95% CI
of an item-level bootstrap with the painting-level cluster bootstrap used in
the other tables.

Usage:
    python rebuttal_experiments/common/analysis/clustering_sensitivity.py [--judge phi4]
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

COMPARISONS = [
    ("base", "full_classify"),
    ("base", "full_noclassify"),
    ("base", "full_systemprompt"),
    ("base", "full_classify_noartist"),
    ("base", "full_alwaysretrieve"),
    ("full_systemprompt", "full_classify"),
    ("full_classify", "full_noclassify"),
    ("full_classify", "full_classify_noartist"),
    ("full_alwaysretrieve", "full_classify"),
]


def item_bootstrap(values, n_boot=10000, seed=lib.SEED, alpha=0.05):
    """Resample observations as if independent."""
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    means.sort()
    return means[int(alpha / 2 * n_boot)], means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4
    scored = lib.load_scored(judge)

    out, payload = [], {"judge_model": judge, "paired": {}, "single_variant": {}}

    # ── paired differences ───────────────────────────────────────────────
    rows = []
    for x, y in COMPARISONS:
        if x not in scored or y not in scored:
            continue
        pairs = lib.paired_diffs(scored[x], scored[y], "correctness")
        if not pairs:
            continue
        diffs = [d for _, d in pairs]
        il, ih = item_bootstrap(diffs, a.n_boot)
        c = lib.cluster_bootstrap_paired(pairs, n_boot=a.n_boot)
        wi, wc = ih - il, c["ci_high"] - c["ci_low"]
        sig_i = not (il <= 0 <= ih)
        sig_c = not (c["ci_low"] <= 0 <= c["ci_high"])
        payload["paired"][f"{x}->{y}"] = {
            "mean": c["mean"], "item_ci": [il, ih],
            "cluster_ci": [c["ci_low"], c["ci_high"]],
            "width_item": wi, "width_cluster": wc,
            "width_ratio": wc / wi if wi else float("nan"),
            "significant_item": sig_i, "significant_cluster": sig_c,
            "verdict_changed": sig_i != sig_c,
        }
        rows.append([
            f"`{x}` → `{y}`", f"{c['mean']:+.3f}",
            f"[{il:+.3f}, {ih:+.3f}]", f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
            f"{wc / wi:.2f}×" if wi else "–",
            "**yes**" if sig_i != sig_c else "no",
        ])
    out.append("#### Did the clustering correction change any verdict? (Δ correctness)\n")
    out.append(lib.table_block(
        "clustering_paired",
        ["comparison", "mean Δ", "95% CI, item bootstrap",
         "95% CI, painting cluster bootstrap", "CI width ratio",
         "significance verdict changed?"], rows))

    # ── single-variant means ─────────────────────────────────────────────
    rows = []
    for v in lib.VARIANTS:
        if v not in scored or not scored[v]:
            continue
        items = list(scored[v].values())
        vals = [r["correctness"] for r in items]
        il, ih = item_bootstrap(vals, a.n_boot)
        c = lib.cluster_bootstrap_mean(
            [(r["painting_id"], r["correctness"]) for r in items], n_boot=a.n_boot)
        wi, wc = ih - il, c["ci_high"] - c["ci_low"]
        payload["single_variant"][v] = {
            "mean": c["mean"], "item_ci": [il, ih],
            "cluster_ci": [c["ci_low"], c["ci_high"]],
            "width_ratio": wc / wi if wi else float("nan"),
        }
        rows.append([f"`{v}`", lib.fmt(c["mean"]), f"[{il:.3f}, {ih:.3f}]",
                     f"[{c['ci_low']:.3f}, {c['ci_high']:.3f}]",
                     f"{wc / wi:.2f}×" if wi else "–"])
    out.append("\n#### The same, on each variant's own mean (correlation not differenced away)\n")
    out.append(lib.table_block(
        "clustering_single",
        ["variant", "mean correctness", "95% CI, item bootstrap",
         "95% CI, painting cluster bootstrap", "CI width ratio"], rows))

    ratios = [e["width_ratio"] for e in payload["paired"].values()]
    changed = [k for k, e in payload["paired"].items() if e["verdict_changed"]]
    payload["summary"] = {
        "paired_width_ratio_min": min(ratios) if ratios else None,
        "paired_width_ratio_max": max(ratios) if ratios else None,
        "n_verdicts_changed": len(changed),
        "verdicts_changed": changed,
    }
    lib.write_table(f"clustering_sensitivity__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
