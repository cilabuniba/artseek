"""Cross-benchmark summary tables.

For ArtPedia-VQA, AQUA, ArtQuest, LICNHeldOut and ArtCurate-AIC: provenance of
references and labels, mean scores of every variant, `base` -> `full_classify`
with painting-level 95% CIs, evidence and oracle statistics (where judged),
and `full_systemprompt` vs. `full_classify` with non-answers counted.

Usage:
    python rebuttal_experiments/cross_benchmark/summary.py [--judge phi4]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
ANALYSIS = EXPERIMENTS_DIR / "common" / "analysis"
sys.path.insert(0, str(ANALYSIS))

EXPERIMENTS = [
    ("ArtPedia-VQA", "artpedia_vqa", "LLM-generated from human sentences",
     "ours (visual/contextual)"),
    ("AQUA", "aqua", "dataset-provided (Garcia et al. 2020)",
     "dataset-provided (`need_external_knowledge`)"),
    ("ArtQuest", "artquest", "dataset-provided (Bleidt et al. 2024)",
     "n/a — all six types are metadata questions"),
    ("LICNHeldOut", "licn_heldout", "Wikidata statements (no LLM)",
     "n/a — every question targets one LICN head"),
    ("ArtCurate-AIC", "artcurate_aic", "LLM-generated from curatorial prose (Art Institute of Chicago, CC0)",
     "n/a — all questions are interpretive"),
]


def load_for(exp_dir: str):
    """Import lib fresh against one experiment directory."""
    os.environ["ARTSEEK_EXP_DIR"] = str(EXPERIMENTS_DIR / exp_dir)
    for mod in list(sys.modules):
        if mod in ("lib",):
            del sys.modules[mod]
    import lib  # noqa: PLC0415
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()

    out = ["## Cross-benchmark summary\n"]
    provenance_rows, result_rows, delta_rows = [], [], []

    for name, exp_dir, ref_prov, label_prov in EXPERIMENTS:
        lib = load_for(exp_dir)
        judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4
        try:
            scored = lib.load_scored(judge)
        except Exception:  # noqa: BLE001
            scored = {}
        if not scored:
            provenance_rows.append([f"**{name}**", ref_prov, label_prov, "—", "*not yet run*"])
            continue

        n_paintings = len({r["painting_id"] for d in scored.values() for r in d.values()})
        n_items = max(len(d) for d in scored.values())
        provenance_rows.append([
            f"**{name}**", ref_prov, label_prov, str(n_paintings),
            ", ".join(f"`{v}`" for v in scored),
        ])

        for v in lib.VARIANTS:
            if v not in scored or not scored[v]:
                continue
            items = list(scored[v].values())
            result_rows.append([
                name, f"`{v}`", len(items),
                lib.fmt(sum(r["correctness"] for r in items) / len(items)),
                lib.fmt(sum(r["recall"] for r in items) / len(items)),
                lib.fmt(sum(1 for r in items if r["correctness"] == 2) / len(items) * 100, 1) + "%",
            ])

        # the headline comparison, wherever both sides exist
        if "base" in scored and "full_classify" in scored:
            pairs = lib.paired_diffs(scored["base"], scored["full_classify"], "correctness")
            if pairs:
                r = lib.cluster_bootstrap_paired(pairs, n_boot=a.n_boot)
                delta_rows.append([
                    name, r["n"], r["n_clusters"], f"{r['mean']:+.3f}",
                    lib.fmt_ci(r), lib.fmt(r["cohens_dz"]),
                    "yes" if not (r["ci_low"] <= 0 <= r["ci_high"]) else "**no**",
                ])

    lib0 = load_for("artpedia_vqa")
    out.append("### What each benchmark controls for\n")
    out.append(lib0.table_block(
        "cross_provenance",
        ["benchmark", "reference answers", "retrieval-need label",
         "paintings", "variants run"], provenance_rows))

    if result_rows:
        out.append("\n### Absolute performance\n")
        out.append(lib0.table_block(
            "cross_results",
            ["benchmark", "variant", "n", "correctness (0–2)", "recall", "% fully correct"],
            result_rows))

    if delta_rows:
        out.append("\n### The headline comparison on every benchmark\n")
        out.append(lib0.table_block(
            "cross_delta",
            ["benchmark", "n", "paintings", "mean Δ correctness (`base` → `full_classify`)",
             "95% cluster CI", "d_z", "CI excludes 0"], delta_rows))
        out.append(
            "\n*Paired over items, resampled over paintings.*")

    ev_rows = []
    for name, exp_dir, _, _ in EXPERIMENTS:
        libx = load_for(exp_dir)
        ev = libx.load_judge(judge, "evidence")
        if not ev:
            continue
        by = {}
        for (v, q), r in ev.items():
            by.setdefault(v, {})[q] = r

        def stats(v):
            d = by.get(v)
            if not d:
                return None
            vals = list(d.values())
            n = len(vals)
            return (n,
                    sum(1 for r in vals if r["evidence_present"] == 0) / n * 100,
                    sum(1 for r in vals if r["evidence_present"] == 2) / n * 100,
                    sum(r["evidence_recall"] for r in vals) / n)

        s = stats("full_classify")
        o = stats(libx.ORACLE_VARIANT)
        if not s:
            continue
        ev_rows.append([
            name, s[0], libx.fmt(s[1], 1) + "%", libx.fmt(s[2], 1) + "%",
            libx.fmt(s[3]),
            libx.fmt(o[1], 1) + "%" if o else "—",
            libx.fmt(o[3]) if o else "—",
        ])

    if ev_rows:
        out.append("\n### Is the index able to answer at all?\n")
        out.append(lib0.table_block(
            "cross_evidence",
            ["benchmark", "retrievals judged", "% evidence absent",
             "% evidence fully present", "mean evidence recall",
             "oracle % absent", "oracle recall"], ev_rows))
        out.append(
            "\n*Oracle: the same index queried with question + reference "
            "answer (common/oracle_probe.py).*")

    na_rows = []
    for name, exp_dir, _, _ in EXPERIMENTS:
        libx = load_for(exp_dir)
        sys.path.insert(0, str(ANALYSIS))
        for mod in ("nonanswer",):
            sys.modules.pop(mod, None)
        import nonanswer  # noqa: PLC0415

        J = libx.load_judge(judge, "answer")
        by = {}
        for (v, q), r in J.items():
            by.setdefault(v, {})[q] = r
        if "full_systemprompt" not in by or "full_classify" not in by:
            continue

        n_sp, n_fc = nonanswer.totals("full_systemprompt"), nonanswer.totals("full_classify")
        e_sp, e_fc = nonanswer.nonanswers("full_systemprompt"), nonanswer.nonanswers("full_classify")

        cells = []
        for label, addA, addB in (("as judged", {}, {}),
                                  ("scored 0", {q: 0 for q in e_sp}, {q: 0 for q in e_fc})):
            sA = {q: r["correctness"] for q, r in by["full_systemprompt"].items()} | addA
            sB = {q: r["correctness"] for q, r in by["full_classify"].items()} | addB
            common = sorted(set(sA) & set(sB))
            r = libx.cluster_bootstrap_paired(
                [(libx.painting_of(q), sB[q] - sA[q]) for q in common], n_boot=a.n_boot)
            sig = not (r["ci_low"] <= 0 <= r["ci_high"])
            cells.append((f"{r['mean']:+.3f}", libx.fmt_ci(r), "**yes**" if sig else "no"))

        na_rows.append([
            name,
            f"{len(e_sp)}/{n_sp} ({libx.fmt(len(e_sp) / n_sp * 100, 1)}%)" if n_sp else "—",
            f"{len(e_fc)}/{n_fc} ({libx.fmt(len(e_fc) / n_fc * 100, 1)}%)" if n_fc else "—",
            cells[0][0], cells[0][1], cells[0][2],
            cells[1][0], cells[1][1], cells[1][2],
        ])

    if na_rows:
        out.append("\n### The in-context example, once non-answers are counted\n")
        out.append(lib0.table_block(
            "cross_nonanswer",
            ["benchmark", "`full_systemprompt` non-answers",
             "`full_classify` non-answers",
             "Δ as judged", "95% CI", "excludes 0",
             "Δ non-answers scored 0", "95% CI", "excludes 0"], na_rows))
        out.append(
            "\n*Non-answer: the run ended without a final answer. The judge "
            "skips these; the last three columns score them 0.*")

    print("\n".join(out))


if __name__ == "__main__":
    main()
