"""Tests for lib.py: artist name matching and the cluster bootstrap.

Run with:  python rebuttal_experiments/common/analysis/test_lib.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# (predicted, gold, want_strict, want_lenient)
ARTIST_CASES = [
    # exact and slug-order forms
    ("vincent-van-gogh", "Vincent van Gogh", True, True),
    ("van-gogh-vincent", "Vincent van Gogh", True, True),
    ("clara-peeters", "Clara Peeters", True, True),
    ("duccio-di-buoninsegna", "Duccio di Buoninsegna", True, True),
    ("el-greco", "El Greco", True, True),
    ("cimabue", "Cimabue", True, True),
    ("albrecht-durer", "Albrecht Dürer", True, True),          # diacritics
    # genuinely different artists must not match
    ("edouard-manet", "Mary Cassatt", False, False),
    ("jean-hey", "Master of Moulins", False, False),
    ("clara-peeters", "unknown Dutch artist", False, False),
    ("titian", "Tiziano Vecellio", False, False),              # translated name: not recoverable by string match
    # surname-only: lenient but not strict
    ("pierre-auguste-renoir", "Auguste Renoir", False, True),
    # mononyms, where the gold carries the full form
    ("rembrandt", "Rembrandt van Rijn", False, True),
    ("caravaggio", "Michelangelo Merisi da Caravaggio", False, True),
    # a workshop attribution shares the surname: lenient but not strict
    ("workshop-of-rubens", "Peter Paul Rubens", False, True),
    # generic attribution words must never carry a match on their own
    ("master", "Master of Moulins", False, False),
    ("unknown", "unknown Dutch artist", False, False),
    ("hey", "Jean Hey", False, False),                          # token too short
    # empty inputs
    ("", "Cimabue", False, False),
    ("cimabue", "", False, False),
]


def test_artist_match():
    bad = []
    for pred, gold, ws, wl in ARTIST_CASES:
        s, l = lib.artist_match(pred, gold)
        if (s, l) != (ws, wl):
            bad.append(f"    {pred!r} vs {gold!r}: got strict={s}, lenient={l}; "
                       f"want strict={ws}, lenient={wl}")
    print(f"artist_match: {len(ARTIST_CASES) - len(bad)}/{len(ARTIST_CASES)} pass")
    for b in bad:
        print(b)
    return not bad


def test_cluster_bootstrap():
    """The cluster bootstrap must (a) recover the right point estimate,
    (b) count clusters rather than observations, and (c) produce a *wider*
    interval than an item-level bootstrap when within-painting correlation
    is strong — which is the entire reason for using it."""
    ok = True

    # (a) point estimate and cluster counting
    pairs = [("p1", 1.0), ("p1", 1.0), ("p2", 0.0), ("p2", 0.0), ("p3", 0.5)]
    r = lib.cluster_bootstrap_paired(pairs, n_boot=2000)
    if abs(r["mean"] - 0.5) > 1e-9:
        print(f"    mean wrong: {r['mean']} != 0.5"); ok = False
    if r["n"] != 5 or r["n_clusters"] != 3:
        print(f"    counts wrong: n={r['n']} clusters={r['n_clusters']}"); ok = False

    # (b) perfectly correlated within painting -> clustering must widen the CI
    #     relative to pretending the 2N questions are independent.
    import random
    rng = random.Random(0)
    corr, indep = [], []
    for i in range(200):
        v = rng.choice([0.0, 1.0])
        corr += [(f"p{i}", v), (f"p{i}", v)]      # both questions identical
        indep += [(f"a{i}", v), (f"b{i}", v)]     # same values, separate clusters
    rc = lib.cluster_bootstrap_paired(corr, n_boot=2000)
    ri = lib.cluster_bootstrap_paired(indep, n_boot=2000)
    wc, wi = rc["ci_high"] - rc["ci_low"], ri["ci_high"] - ri["ci_low"]
    if not wc > wi:
        print(f"    clustering did not widen the CI: {wc:.4f} vs {wi:.4f}"); ok = False
    else:
        print(f"cluster_bootstrap: correlated CI width {wc:.4f} > "
              f"independent {wi:.4f} (ratio {wc / wi:.2f}) — as expected")

    # (c) determinism under a fixed seed
    r1 = lib.cluster_bootstrap_paired(pairs, n_boot=500)
    r2 = lib.cluster_bootstrap_paired(pairs, n_boot=500)
    if (r1["ci_low"], r1["ci_high"]) != (r2["ci_low"], r2["ci_high"]):
        print("    bootstrap is not reproducible under a fixed seed"); ok = False

    print(f"cluster_bootstrap: {'pass' if ok else 'FAIL'}")
    return ok


def test_kappa_and_spearman():
    ok = True
    if abs(lib.cohens_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) - 1.0) > 1e-9:
        print("    kappa of a perfect match != 1"); ok = False
    # quadratic weighting must punish a 0-vs-2 disagreement harder than 0-vs-1
    k_adj = lib.cohens_kappa([0] * 5 + [2] * 5, [0] * 5 + [1] * 5)
    k_far = lib.cohens_kappa([0] * 5 + [2] * 5, [0] * 5 + [0] * 5)
    if not k_adj > k_far:
        print(f"    quadratic kappa not ordered: adjacent {k_adj} !> distant {k_far}"); ok = False
    if abs(lib.spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) > 1e-9:
        print("    spearman of an identical ranking != 1"); ok = False
    if abs(lib.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) > 1e-9:
        print("    spearman of a reversed ranking != -1"); ok = False
    print(f"kappa/spearman: {'pass' if ok else 'FAIL'}")
    return ok


def test_binary_metrics():
    m = lib.binary_metrics(tp=50, fp=10, fn=5, tn=35)
    ok = (abs(m["precision"] - 50 / 60) < 1e-9
          and abs(m["recall"] - 50 / 55) < 1e-9
          and 0 < m["mcc"] < 1)
    print(f"binary_metrics: {'pass' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [
        test_artist_match(),
        test_cluster_bootstrap(),
        test_kappa_and_spearman(),
        test_binary_metrics(),
    ]
    print(f"\n{sum(results)}/{len(results)} test groups pass")
    sys.exit(0 if all(results) else 1)
