"""Retrieval-policy analysis (question types visual/contextual).

1. "The model retrieved" as a descriptive predictor of the contextual label:
   precision, recall, F1, balanced accuracy, MCC. This is not a target: visual
   questions can also benefit from retrieval.
2. Tool-call counts per question type.
3. On visual questions, correctness when the model retrieved vs. skipped
   (self-selected groups; see the difficulty controls).
4. Difficulty controls for that split.
5. Justified-skip rate: share of zero-call visual answers judged correct.

Usage:
    python rebuttal_experiments/common/analysis/routing.py [--judge phi4] [--subset ...]
"""

import argparse
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

# Crude question-form bucket, used only as a difficulty control when
# comparing retrieved-vs-skipped visual questions (those groups are
# self-selected, so we need *some* handle on whether they are comparable).
FORMS = [
    ("who", r"^\s*who\b"),
    ("what/which", r"^\s*(what|which)\b"),
    ("where", r"^\s*where\b"),
    ("when", r"^\s*when\b"),
    ("how many", r"^\s*how many\b"),
    ("how/why", r"^\s*(how|why)\b"),
]


def question_form(q: str) -> str:
    q = (q or "").strip().lower()
    for name, pat in FORMS:
        if re.search(pat, q):
            return name
    return "other"


def two_group_bootstrap(a_vals, b_vals, n_boot=10000, seed=lib.SEED, alpha=0.05):
    """Percentile CI on mean(b) - mean(a) for two independent groups.

    Within the visual-question subset there is exactly one question per
    painting, so item-level and painting-level resampling coincide here and
    no clustering correction is needed.
    """
    if not a_vals or not b_vals:
        return {"n_a": len(a_vals), "n_b": len(b_vals), "diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    rng = random.Random(seed)
    na, nb = len(a_vals), len(b_vals)
    diffs = []
    for _ in range(n_boot):
        sa = sum(a_vals[rng.randrange(na)] for _ in range(na)) / na
        sb = sum(b_vals[rng.randrange(nb)] for _ in range(nb)) / nb
        diffs.append(sb - sa)
    diffs.sort()
    return {
        "n_a": na, "n_b": nb,
        "mean_a": sum(a_vals) / na, "mean_b": sum(b_vals) / nb,
        "diff": sum(b_vals) / nb - sum(a_vals) / na,
        "ci_low": diffs[int(alpha / 2 * n_boot)],
        "ci_high": diffs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--subset", default=None, choices=["obscure", "nonobscure"])
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    scored = lib.load_scored(judge)
    scored.pop("base", None)  # no tools bound: it cannot route
    tag = a.judge
    if a.subset:
        obs = lib.read_jsonl(lib.JSONL_DIR / "obscure.jsonl")
        if not obs:
            raise SystemExit("no obscure.jsonl yet")
        want = {r["painting_id"] for r in obs if r["is_obscure"] == (a.subset == "obscure")}
        scored = {v: {q: r for q, r in d.items() if r["painting_id"] in want}
                  for v, d in scored.items()}
        tag += f"__{a.subset}"

    variants = [v for v in lib.VARIANTS if v in scored and scored[v]]
    out, payload = [], {"judge_model": judge, "subset": a.subset or "all"}

    # ── 1. routing as a (descriptive) classification problem ─────────────
    rows = []
    payload["routing"] = {}
    for v in variants:
        items = list(scored[v].values())
        tp = sum(1 for r in items if r["retrieved"] and r["question_type"] == "contextual")
        fp = sum(1 for r in items if r["retrieved"] and r["question_type"] == "visual")
        fn = sum(1 for r in items if not r["retrieved"] and r["question_type"] == "contextual")
        tn = sum(1 for r in items if not r["retrieved"] and r["question_type"] == "visual")
        m = lib.binary_metrics(tp, fp, fn, tn)
        payload["routing"][v] = m
        rows.append([f"`{v}`", tp, fp, fn, tn, lib.fmt(m["precision"]), lib.fmt(m["recall"]),
                     lib.fmt(m["f1"]), lib.fmt(m["balanced_accuracy"]), lib.fmt(m["mcc"])])
    out.append(
        "#### Retrieval decision vs. the gold visual/contextual label "
        "(descriptive — see caveat)\n"
    )
    out.append(lib.table_block(
        "routing_confusion",
        ["variant", "TP (ret/ctx)", "FP (ret/vis)", "FN (skip/ctx)", "TN (skip/vis)",
         "precision", "recall", "F1", "balanced acc.", "MCC"], rows))
    out.append(
        "\n*These are **not** a scorecard the policy should maximise. "
        "\"False positives\" here are visual questions the model chose to "
        "retrieve for, and retrieval measurably improves visual-question "
        "correctness, so many of them are good decisions. The table "
        "measures how closely the learned policy tracks a human "
        "annotation boundary, nothing more.*"
    )

    # ── 2. call counts by class ──────────────────────────────────────────
    rows = []
    payload["calls"] = {}
    for v in variants:
        cells = [f"`{v}`"]
        entry = {}
        for qt in ("visual", "contextual"):
            sub = [r for r in scored[v].values() if r["question_type"] == qt]
            calls = [r["num_tool_calls"] or 0 for r in sub]
            e = {"n": len(sub), "mean": sum(calls) / len(calls) if calls else float("nan"),
                 "max": max(calls) if calls else 0,
                 "pct_zero": sum(1 for c in calls if c == 0) / len(calls) * 100 if calls else float("nan")}
            entry[qt] = e
            cells += [lib.fmt(e["mean"], 2), str(e["max"]), lib.fmt(e["pct_zero"], 1) + "%"]
        payload["calls"][v] = entry
        rows.append(cells)
    out.append("\n#### Tool calls per query, by question class\n")
    out.append(lib.table_block(
        "routing_calls",
        ["variant", "vis: mean calls", "vis: max", "vis: % zero calls",
         "ctx: mean calls", "ctx: max", "ctx: % zero calls"], rows))

    # ── 3. does retrieving on a visual question actually help? ───────────
    rows = []
    payload["visual_retrieved_vs_skipped"] = {}
    for v in variants:
        vis = [r for r in scored[v].values() if r["question_type"] == "visual"]
        ret = [r for r in vis if r["retrieved"]]
        skip = [r for r in vis if not r["retrieved"]]
        if not ret or not skip:
            continue
        c = two_group_bootstrap([r["correctness"] for r in skip],
                                [r["correctness"] for r in ret])
        rc = two_group_bootstrap([r["recall"] for r in skip],
                                 [r["recall"] for r in ret])
        payload["visual_retrieved_vs_skipped"][v] = {"correctness": c, "recall": rc}
        rows.append([
            f"`{v}`", len(skip), len(ret),
            lib.fmt(c["mean_a"]), lib.fmt(c["mean_b"]), f"{c['diff']:+.3f}", lib.fmt_ci(c),
            lib.fmt(rc["mean_a"]), lib.fmt(rc["mean_b"]), f"{rc['diff']:+.3f}", lib.fmt_ci(rc),
        ])
    out.append("\n#### Visual questions: retrieved vs. skipped\n")
    out.append(lib.table_block(
        "routing_vis_ret_vs_skip",
        ["variant", "n skipped", "n retrieved", "correctness (skip)",
         "correctness (retrieved)", "Δ", "95% CI", "recall (skip)",
         "recall (retrieved)", "Δ", "95% CI"], rows))
    out.append(
        "\n*Self-selected comparison: the model chooses which group each "
        "item lands in, and it may skip precisely on the items it finds "
        "easy. The difficulty controls below exist to make that confound "
        "visible; they do not remove it.*"
    )

    # ── 4. difficulty controls for the split above ───────────────────────
    rows = []
    payload["difficulty_controls"] = {}
    for v in variants:
        vis = [r for r in scored[v].values() if r["question_type"] == "visual"]
        ret = [r for r in vis if r["retrieved"]]
        skip = [r for r in vis if not r["retrieved"]]
        if not ret or not skip:
            continue
        wl_s = [r["reference_answer_words"] for r in skip]
        wl_r = [r["reference_answer_words"] for r in ret]
        forms_s = {}
        forms_r = {}
        for r in skip:
            forms_s[question_form(r["question"])] = forms_s.get(question_form(r["question"]), 0) + 1
        for r in ret:
            forms_r[question_form(r["question"])] = forms_r.get(question_form(r["question"]), 0) + 1
        top_s = sorted(forms_s.items(), key=lambda kv: -kv[1])[:3]
        top_r = sorted(forms_r.items(), key=lambda kv: -kv[1])[:3]
        payload["difficulty_controls"][v] = {
            "mean_ref_words_skipped": sum(wl_s) / len(wl_s),
            "mean_ref_words_retrieved": sum(wl_r) / len(wl_r),
            "question_forms_skipped": forms_s,
            "question_forms_retrieved": forms_r,
        }
        rows.append([
            f"`{v}`",
            lib.fmt(sum(wl_s) / len(wl_s), 1), lib.fmt(sum(wl_r) / len(wl_r), 1),
            ", ".join(f"{k} {n / len(skip):.0%}" for k, n in top_s),
            ", ".join(f"{k} {n / len(ret):.0%}" for k, n in top_r),
        ])
    out.append("\n#### Difficulty controls for the retrieved/skipped split (visual only)\n")
    out.append(lib.table_block(
        "routing_difficulty",
        ["variant", "mean ref-answer words (skipped)", "mean ref-answer words (retrieved)",
         "top question forms (skipped)", "top question forms (retrieved)"], rows))

    # ── 5. justified-skip rate ───────────────────────────────────────────
    rows = []
    payload["justified_skip"] = {}
    for v in variants:
        skip = [r for r in scored[v].values()
                if r["question_type"] == "visual" and not r["retrieved"]]
        if not skip:
            continue
        n = len(skip)
        n_correct = sum(1 for r in skip if r["correctness"] == 2)
        n_partial = sum(1 for r in skip if r["correctness"] >= 1)
        e = {"n_skipped": n, "pct_correct": n_correct / n * 100,
             "pct_at_least_partial": n_partial / n * 100,
             "mean_recall": sum(r["recall"] for r in skip) / n}
        payload["justified_skip"][v] = e
        rows.append([f"`{v}`", n, lib.fmt(e["pct_correct"], 1) + "%",
                     lib.fmt(e["pct_at_least_partial"], 1) + "%",
                     lib.fmt(e["mean_recall"])])
    out.append("\n#### Justified-skip rate (visual questions answered with zero calls)\n")
    out.append(lib.table_block(
        "routing_justified_skip",
        ["variant", "n skipped", "% scored correct (2)",
         "% scored ≥ partial (≥1)", "mean recall"], rows))

    lib.write_table(f"routing__{tag}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
