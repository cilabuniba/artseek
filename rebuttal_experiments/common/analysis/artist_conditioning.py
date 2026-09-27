"""Effect of the artist field of the artwork card, split by whether LICN's
top-1 artist is correct.

Reports LICN's artist accuracy on the benchmark (strict and lenient, i.e.
surname-only, matching against the gold artists of ground_truth.py), the
paired comparison `full_classify_noartist` -> `full_classify` split by LICN
correctness, and a post-hoc simulated confidence gate over existing runs
(the executed version is `full_classify_gated`).

Usage:
    python rebuttal_experiments/common/analysis/artist_conditioning.py [--judge phi4]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    gold = {r["painting_id"]: r for r in lib.read_jsonl(lib.JSONL_DIR / "gold_artists.jsonl")}
    if not gold:
        raise SystemExit("no gold_artists.jsonl — run slurm/40_ground_truth.sh first")

    scored = lib.load_scored(judge)
    for v in ("full_classify", "full_classify_noartist", "full_noclassify"):
        if v not in scored:
            raise SystemExit(f"missing scored variant {v}")

    out, payload = [], {"judge_model": judge}

    # ── 1. LICN artist accuracy on this subset ───────────────────────────
    # The prediction is a property of the painting, so evaluate per painting
    # (both variants that show a card produce the same LICN output).
    per_painting = {}
    for r in scored["full_classify"].values():
        pid = r["painting_id"]
        if pid in per_painting or not r.get("licn_artist"):
            continue
        g = gold.get(pid)
        if not g:
            continue
        gold_name = g.get("gold_artist") or ""
        if g.get("gold_artist_unknown") or not gold_name:
            continue  # anonymous work: no artist to be right or wrong about
        strict, lenient = lib.artist_match(r["licn_artist"], gold_name)
        per_painting[pid] = {
            "licn_artist": r["licn_artist"],
            "licn_conf": r.get("licn_artist_conf"),
            "gold_artist": gold_name,
            "strict": strict,
            "lenient": lenient,
        }

    n = len(per_painting)
    n_strict = sum(1 for x in per_painting.values() if x["strict"])
    n_lenient = sum(1 for x in per_painting.values() if x["lenient"])
    n_anon = sum(1 for pid, g in gold.items()
                 if g.get("gold_artist_unknown") and pid in
                 {r["painting_id"] for r in scored["full_classify"].values()})
    payload["licn_artist_accuracy"] = {
        "n_paintings_with_named_gold_artist": n,
        "n_anonymous_excluded": n_anon,
        "top1_strict": n_strict / n if n else float("nan"),
        "top1_lenient": n_lenient / n if n else float("nan"),
        "mean_confidence_when_correct": (
            sum(x["licn_conf"] for x in per_painting.values() if x["strict"] and x["licn_conf"])
            / max(n_strict, 1)),
        "mean_confidence_when_wrong": (
            sum(x["licn_conf"] for x in per_painting.values() if not x["strict"] and x["licn_conf"])
            / max(n - n_strict, 1)),
    }
    p = payload["licn_artist_accuracy"]
    out.append("#### LICN top-1 artist accuracy on the ArtPedia test subset\n")
    out.append(lib.table_block(
        "artist_licn_accuracy",
        ["paintings with a named gold artist", "anonymous (excluded)",
         "top-1 accuracy (strict)", "top-1 accuracy (lenient, surname-only counts)",
         "mean confidence when right", "mean confidence when wrong"],
        [[n, n_anon, lib.fmt(p["top1_strict"]), lib.fmt(p["top1_lenient"]),
          lib.fmt(p["mean_confidence_when_correct"]),
          lib.fmt(p["mean_confidence_when_wrong"])]]))

    # ── 2. paired comparisons split by LICN correctness ──────────────────
    def stratified(a_name, b_name, field, correct_stratum, use="strict"):
        A, B = scored[a_name], scored[b_name]
        keep = {pid for pid, x in per_painting.items() if x[use] == correct_stratum}
        pairs = [(lib.painting_of(q), B[q][field] - A[q][field])
                 for q in sorted(set(A) & set(B))
                 if A[q]["painting_id"] in keep]
        if not pairs:
            return None
        return lib.cluster_bootstrap_paired(pairs, n_boot=a.n_boot)

    payload["stratified"] = {}
    for a_name, b_name in (("full_classify", "full_classify_noartist"),
                           ("full_classify", "full_noclassify")):
        rows = []
        for field in ("correctness", "recall"):
            for stratum, label in ((True, "LICN artist CORRECT"), (False, "LICN artist WRONG")):
                r = stratified(a_name, b_name, field, stratum)
                if not r:
                    continue
                payload["stratified"][f"{a_name}->{b_name}|{field}|{label}"] = r
                rows.append([label, field, r["n"], r["n_clusters"],
                             f"{r['mean']:+.3f}", lib.fmt_ci(r),
                             lib.fmt(r["cohens_dz"]),
                             "yes" if not (r["ci_low"] <= 0 <= r["ci_high"]) else "no"])
        out.append(f"\n#### `{a_name}` → `{b_name}`, split by whether LICN's artist was right\n")
        out.append(lib.table_block(
            f"artist_stratified__{b_name}",
            ["stratum", "metric", "n", "paintings", "mean Δ", "95% cluster CI",
             "d_z", "CI excludes 0"], rows))
        out.append(
            "\n*Δ is (withheld/absent artist) − (artist shown), so a "
            "**positive** Δ means showing the artist field HURT.*"
        )

    # ── 3. simulated confidence gate (post-hoc, no new inference) ────────
    A, B = scored["full_classify"], scored["full_classify_noartist"]
    common = sorted(set(A) & set(B))
    rows = []
    payload["confidence_gate_simulation"] = {}
    for tau in [i / 10 for i in range(10)]:
        vals_c, vals_r = [], []
        n_gated = 0
        for q in common:
            conf = A[q].get("licn_artist_conf")
            use_a = conf is not None and conf >= tau  # show artist
            n_gated += use_a
            src = A[q] if use_a else B[q]
            vals_c.append(src["correctness"])
            vals_r.append(src["recall"])
        e = {"tau": tau, "pct_showing_artist": n_gated / len(common) * 100,
             "mean_correctness": sum(vals_c) / len(vals_c),
             "mean_recall": sum(vals_r) / len(vals_r)}
        payload["confidence_gate_simulation"][f"{tau:.1f}"] = e
        rows.append([f"{tau:.1f}", lib.fmt(e["pct_showing_artist"], 1) + "%",
                     lib.fmt(e["mean_correctness"]), lib.fmt(e["mean_recall"])])
    out.append("\n#### Simulated confidence gate on the artist field (post-hoc)\n")
    out.append(lib.table_block(
        "artist_gate",
        ["τ (min LICN artist confidence to show the field)",
         "% of items showing the artist", "mean correctness", "mean recall"], rows))
    out.append(
        "\n*This is a **post-hoc simulation** built by selecting, per item, "
        "the score from a run we already have — τ=0.0 is exactly "
        "`full_classify` and τ=1.0 would be exactly `full_classify_noartist`. "
        "No confidence-gated variant was executed, and the model's retrieval "
        "trajectory would differ if one were.*"
    )

    lib.write_table(f"artist_conditioning__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
