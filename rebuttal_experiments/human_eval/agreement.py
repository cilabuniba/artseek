"""Agreement between raters, and between raters and the phi-4 judge.

  * humans with each other: Krippendorff's alpha (ordinal) over the 0-2 axis
    ratings, and pairwise Spearman correlation of the Sigma sums;
  * phi-4 with the humans: the same, with phi-4 (mean of its two presentation
    orders) as an additional rater, and its correlation with the human mean;
  * forced choice: pairwise agreement and Fleiss' kappa.

Item devanna-010 is excluded. Reads results/human_ratings.json,
results/judge_rubric.json and data/submissions.txt; writes
results/agreement.json.

Usage:
    python rebuttal_experiments/human_eval/agreement.py
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SYSTEMS = ["base", "artseek_multiquery", "gpt"]
DISPLAY = {"base": "base", "artseek_multiquery": "ArtSeek", "gpt": "GPT-5.5"}
AXES = ["subject", "figures", "placement", "evidence"]
SIGMA_AXES = ["subject", "figures", "placement"]
EXCLUDED = {"devanna-010"}


def krippendorff_ordinal(matrix):
    """Krippendorff's alpha for ordinal data.

    `matrix` is raters x units, with np.nan for missing. Implemented directly
    rather than pulled in as a dependency: the coincidence-matrix form is short
    and this way the number in the paper has visible provenance.
    """
    m = np.asarray(matrix, dtype=float)
    vals = np.unique(m[~np.isnan(m)])
    v_idx = {v: i for i, v in enumerate(vals)}
    n_v = len(vals)

    coincidence = np.zeros((n_v, n_v))
    for u in range(m.shape[1]):
        col = m[:, u]
        col = col[~np.isnan(col)]
        mu = len(col)
        if mu < 2:
            continue
        for a in col:
            for b in col:
                if a is b:
                    continue
                coincidence[v_idx[a], v_idx[b]] += 1.0 / (mu - 1)
        # remove the self-pairs counted above
        for a in col:
            coincidence[v_idx[a], v_idx[a]] -= 1.0 / (mu - 1)
    n_marg = coincidence.sum(axis=1)
    n_total = n_marg.sum()

    # Ordinal distance: squared sum of intervening marginal frequencies.
    def delta(i, j):
        lo, hi = (i, j) if i <= j else (j, i)
        s = n_marg[lo:hi + 1].sum() - (n_marg[lo] + n_marg[hi]) / 2.0
        return s ** 2

    d_obs = sum(coincidence[i, j] * delta(i, j)
                for i in range(n_v) for j in range(n_v))
    d_exp = sum(n_marg[i] * n_marg[j] * delta(i, j) / (n_total - 1)
                for i in range(n_v) for j in range(n_v))
    return 1.0 - d_obs / d_exp if d_exp else float("nan")


def fleiss_kappa(table):
    """table: units x categories, counts of raters choosing each category."""
    t = np.asarray(table, dtype=float)
    n = t.sum(axis=1)[0]
    p_j = t.sum(axis=0) / (t.shape[0] * n)
    P_i = (np.square(t).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), np.square(p_j).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float("nan")


def main():
    hr = json.loads((HERE / "results" / "human_ratings.json").read_text())
    # rater -> (item, system) -> {axis: score}
    human = defaultdict(dict)
    for r in hr["rows"]:
        human[r["rater"]][(r["item"], r["system"])] = {a: r[a] for a in AXES}
    raters = sorted(human)
    roles = {r: hr["per_rater"][r]["role"] for r in raters}
    experts = [r for r in raters if roles[r] == "expert"]
    lay = [r for r in raters if roles[r] != "expert"]

    # phi-4: average the two presentation orders per (item, system).
    judge_raw = json.loads((HERE / "results" / "judge_rubric.json").read_text())
    acc = defaultdict(list)
    best_phi = defaultdict(list)
    for rec in judge_raw:
        if rec["id"] in EXCLUDED:
            continue
        for sysname, sc in rec["scores"].items():
            acc[(rec["id"], sysname)].append(sc)
        best_phi[rec["id"]].append(rec["best"])
    phi = {k: {a: float(np.mean([s[a] for s in v])) for a in AXES}
           for k, v in acc.items()}

    items = sorted({k[0] for k in phi})
    cells = [(i, s) for i in items for s in SYSTEMS]
    print(f"{len(items)} items x {len(SYSTEMS)} systems = {len(cells)} cells; "
          f"{len(raters)} humans ({len(lay)} lay, {len(experts)} expert) + phi-4\n")

    def sigma(d):
        return sum(d[a] for a in SIGMA_AXES)

    # ---- 1. scores side by side -------------------------------------------
    print("=" * 72)
    print("SCORES — mean Sigma-correctness (0-6) by rater type")
    print("=" * 72)
    groups = {"phi-4 judge": None,
              f"lay (n={len(lay)})": lay,
              f"expert (n={len(experts)})": experts,
              f"all humans (n={len(raters)})": raters}
    print(f"{'system':10s}" + "".join(f"{g:>18s}" for g in groups))
    for s in SYSTEMS:
        row = f"{DISPLAY[s]:10s}"
        for g, members in groups.items():
            if members is None:
                v = np.mean([sigma(phi[(i, s)]) for i in items])
            else:
                v = np.mean([sigma(human[r][(i, s)])
                             for r in members for i in items])
            row += f"{v:18.2f}"
        print(row)

    print(f"\n{'gap vs base':10s}" + "".join(f"{g:>18s}" for g in groups))
    for s in ("artseek_multiquery", "gpt"):
        row = f"{DISPLAY[s]:10s}"
        for g, members in groups.items():
            if members is None:
                v = np.mean([sigma(phi[(i, s)]) - sigma(phi[(i, "base")])
                             for i in items])
            else:
                v = np.mean([sigma(human[r][(i, s)]) - sigma(human[r][(i, "base")])
                             for r in members for i in items])
            row += f"{v:+18.2f}"
        print(row)

    # ---- 2. agreement ------------------------------------------------------
    print("\n" + "=" * 72)
    print("AGREEMENT — Krippendorff's alpha (ordinal), axis-level 0-2 ratings")
    print("=" * 72)
    def alpha_for(members, with_phi=False):
        rows = []
        for r in members:
            rows.append([human[r][c][a] if human[r][c][a] is not None else np.nan
                         for c in cells for a in AXES])
        if with_phi:
            rows.append([phi[c][a] for c in cells for a in AXES])
        return krippendorff_ordinal(rows)

    print(f"  humans only (n={len(raters)})           alpha = {alpha_for(raters):.3f}")
    print(f"  lay only (n={len(lay)})               alpha = {alpha_for(lay):.3f}")
    print(f"  humans + phi-4 (n={len(raters)+1})       alpha = {alpha_for(raters, True):.3f}")

    print("\n  pairwise Spearman on Sigma (42 cells):")
    series = {r: np.array([sigma(human[r][c]) for c in cells]) for r in raters}
    series["phi-4"] = np.array([sigma(phi[c]) for c in cells])
    names = raters + ["phi-4"]
    hh, hp = [], []
    for a, b in combinations(names, 2):
        rho = spearmanr(series[a], series[b]).statistic
        tag = ""
        if "phi-4" in (a, b):
            hp.append(rho); tag = "   <- judge vs human"
        else:
            hh.append(rho)
        lab = f"{a} - {b}"
        print(f"    {lab:26s} rho = {rho:+.3f}{tag}")
    print(f"\n    mean human-human rho  = {np.mean(hh):+.3f}")
    print(f"    mean judge-human rho  = {np.mean(hp):+.3f}")

    human_mean = np.mean([series[r] for r in raters], axis=0)
    print(f"    phi-4 vs human MEAN   = "
          f"{spearmanr(series['phi-4'], human_mean).statistic:+.3f}")
    if experts:
        lay_mean = np.mean([series[r] for r in lay], axis=0)
        print(f"    expert vs lay mean    = "
              f"{spearmanr(series[experts[0]], lay_mean).statistic:+.3f}")
        print(f"    expert vs phi-4       = "
              f"{spearmanr(series[experts[0]], series['phi-4']).statistic:+.3f}")

    # ---- 3. the verdict ----------------------------------------------------
    print("\n" + "=" * 72)
    print("VERDICT — do they pick the same system?")
    print("=" * 72)
    best_h = {r: {} for r in raters}
    for r in raters:
        for row in hr["rows"]:
            pass
    # Forced choices, read back from the transcripts.
    import collect_ratings as cr
    subs, _, _ = cr.load(HERE / "data" / "submissions.txt")
    for s in subs:
        for iid, rec in s["items"].items():
            if iid in EXCLUDED:
                continue
            letters = dict(zip("ABC", cr.order_for(s["rater"], iid)))
            if rec["best"] in letters:
                best_h[s["rater"]][iid] = letters[rec["best"]]

    agree = []
    for a, b in combinations(raters, 2):
        same = sum(best_h[a][i] == best_h[b][i] for i in items) / len(items)
        agree.append(same)
        print(f"    {a} - {b}: {same*100:.0f}% same pick")
    print(f"    mean pairwise agreement (humans) = {np.mean(agree)*100:.0f}% "
          f"(chance = 33%)")

    phi_pick = {i: max(set(best_phi[i]), key=best_phi[i].count) for i in items}
    ph = [sum(best_h[r][i] == phi_pick[i] for i in items) / len(items)
          for r in raters]
    print(f"    phi-4 vs each human: " +
          ", ".join(f"{r} {v*100:.0f}%" for r, v in zip(raters, ph)) +
          f"  (mean {np.mean(ph)*100:.0f}%)")

    table = []
    for i in items:
        counts = [sum(best_h[r][i] == s for r in raters) for s in SYSTEMS]
        table.append(counts)
    print(f"    Fleiss' kappa on the forced choice (humans) = "
          f"{fleiss_kappa(table):+.3f}")

    # ---- 4. scale-free agreement ------------------------------------------
    # Within-item pairwise preferences (does the rater put system X above Y
    # on this painting?), insensitive to how much of the scale a rater uses.
    print("\n" + "=" * 72)
    print("SCALE-FREE — within-item pairwise preferences (ties dropped)")
    print("=" * 72)
    def prefs(get):
        out = []
        for i in items:
            for a, b in combinations(SYSTEMS, 2):
                d = get(i, a) - get(i, b)
                out.append(0 if d == 0 else (1 if d > 0 else -1))
        return np.array(out)
    P = {r: prefs(lambda i, s, r=r: sigma(human[r][(i, s)])) for r in raters}
    P["phi-4"] = prefs(lambda i, s: sigma(phi[(i, s)]))
    hh2, hp2 = [], []
    for a, b in combinations(list(P), 2):
        mask = (P[a] != 0) & (P[b] != 0)
        same = (P[a][mask] == P[b][mask]).mean() * 100
        (hp2 if "phi-4" in (a, b) else hh2).append(same)
        print(f"    {a + ' - ' + b:26s} n={mask.sum():3d}  {same:3.0f}% same direction")
    print(f"\n    mean human-human = {np.mean(hh2):.0f}%   "
          f"mean judge-human = {np.mean(hp2):.0f}%   (chance 50%)")

    out = HERE / "results" / "agreement.json"
    out.write_text(json.dumps({
        "n_items": len(items), "raters": raters, "roles": roles,
        "alpha_humans": alpha_for(raters), "alpha_lay": alpha_for(lay),
        "alpha_humans_plus_phi4": alpha_for(raters, True),
        "mean_rho_human_human": float(np.mean(hh)),
        "mean_rho_judge_human": float(np.mean(hp)),
        "rho_phi4_vs_human_mean": float(spearmanr(series["phi-4"], human_mean).statistic),
        "mean_pairwise_choice_agreement": float(np.mean(agree)),
        "fleiss_kappa_choice": float(fleiss_kappa(table)),
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
