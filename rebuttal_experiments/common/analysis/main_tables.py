"""Main results table (mean correctness and recall per variant, with
painting-level 95% CIs) and its breakdown by question type.

Usage:
    python rebuttal_experiments/common/analysis/main_tables.py [--judge phi4]
                                                               [--subset obscure|nonobscure]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib


def summarise(items):
    n = len(items)
    if not n:
        return None
    return {
        "n": n,
        "avg_recall": sum(r["recall"] for r in items) / n,
        "avg_correctness": sum(r["correctness"] for r in items) / n,
        "pct_wrong": sum(1 for r in items if r["correctness"] == 0) / n * 100,
        "pct_partial": sum(1 for r in items if r["correctness"] == 1) / n * 100,
        "pct_correct": sum(1 for r in items if r["correctness"] == 2) / n * 100,
    }


def build(judge: str, subset: str | None, n_boot: int = 10000):
    scored = lib.load_scored(judge)
    if subset:
        obs = lib.read_jsonl(lib.JSONL_DIR / "obscure.jsonl")
        if not obs:
            raise SystemExit("no obscure.jsonl yet — run artpedia_vqa/obscure_probe.py and ground_truth.py first")
        want = {r["painting_id"] for r in obs if r["is_obscure"] == (subset == "obscure")}
        scored = {v: {q: r for q, r in d.items() if r["painting_id"] in want}
                  for v, d in scored.items()}

    variants = [v for v in lib.VARIANTS if v in scored and scored[v]]
    out, payload = [], {"judge_model": judge, "subset": subset or "all",
                        "overall": {}, "by_question_type": {}}

    # ── headline ─────────────────────────────────────────────────────────
    rows = []
    for v in variants:
        items = list(scored[v].values())
        s = summarise(items)
        # Painting-level CI on the variant's own mean.
        ci = lib.cluster_bootstrap_mean(
            [(r["painting_id"], r["correctness"]) for r in items], n_boot=n_boot
        )
        s["correctness_ci"] = [ci["ci_low"], ci["ci_high"]]
        payload["overall"][v] = s
        rows.append([
            f"`{v}`", s["n"], lib.fmt(s["avg_recall"]), lib.fmt(s["avg_correctness"]),
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]",
            lib.fmt(s["pct_wrong"], 1) + "%", lib.fmt(s["pct_partial"], 1) + "%",
            lib.fmt(s["pct_correct"], 1) + "%",
        ])
    out.append(lib.table_block(
        "main_overall",
        ["variant", "n", "avg recall", "avg correctness (0–2)",
         "95% CI (cluster)", "% wrong", "% partial", "% correct"], rows))

    # ── by question type ─────────────────────────────────────────────────
    # Question types as defined by the benchmark.
    qtypes = []
    for v in variants:
        for r in scored[v].values():
            if r["question_type"] not in qtypes:
                qtypes.append(r["question_type"])
    preferred = [q for q in ("visual", "contextual") if q in qtypes]
    qtypes = preferred + [q for q in sorted(qtypes) if q not in preferred]

    rows = []
    for v in variants:
        items = list(scored[v].values())
        per = {q: summarise([r for r in items if r["question_type"] == q]) for q in qtypes}
        if not any(per.values()):
            continue
        payload["by_question_type"][v] = {q: s for q, s in per.items() if s}
        cells = [f"`{v}`", " / ".join(str(per[q]["n"]) if per[q] else "–" for q in qtypes)]
        for q in qtypes:
            cells.append(lib.fmt(per[q]["avg_recall"]) if per[q] else "–")
        for q in qtypes:
            cells.append(lib.fmt(per[q]["avg_correctness"]) if per[q] else "–")
        for q in qtypes:
            cells.append(lib.fmt(per[q]["pct_wrong"], 1) + "%" if per[q] else "–")
        for q in qtypes:
            cells.append(lib.fmt(per[q]["pct_correct"], 1) + "%" if per[q] else "–")
        rows.append(cells)

    headers = (["variant", "n (" + " / ".join(qtypes) + ")"]
               + [f"recall ({q})" for q in qtypes]
               + [f"correctness ({q})" for q in qtypes]
               + [f"% wrong ({q})" for q in qtypes]
               + [f"% correct ({q})" for q in qtypes])
    out.append("\n" + lib.table_block("main_by_qtype", headers, rows))

    tag = ("phi4" if judge == lib.JUDGE_PHI4 else "qwen") + (f"__{subset}" if subset else "")
    lib.write_table(f"main__{tag}", payload)
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--subset", default=None, choices=["obscure", "nonobscure"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4
    print(build(judge, a.subset, a.n_boot))
