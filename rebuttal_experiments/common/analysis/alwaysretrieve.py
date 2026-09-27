"""Forced retrieval (`full_alwaysretrieve`) vs. the learned policy
(`full_classify`): how often forcing was needed and by which mechanism,
correctness and cost (tool calls, wall-clock), and where forcing changed the
outcome.

Usage:
    python rebuttal_experiments/common/analysis/alwaysretrieve.py [--judge phi4]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

MECH_LABEL = {
    "none_needed": "model retrieved unprompted (no forcing needed)",
    "nudge_1": "retrieved after one nudge",
    "nudge_2": "retrieved after a second nudge",
    "synthetic": "refused both nudges — tool call synthesised",
    "not_forced": "forcing not applied (bug — should not appear)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    scored = lib.load_scored(judge)
    if "full_alwaysretrieve" not in scored or not scored["full_alwaysretrieve"]:
        raise SystemExit("no scored full_alwaysretrieve rows yet")

    out, payload = [], {"judge_model": judge}
    AR, FC = scored["full_alwaysretrieve"], scored["full_classify"]

    # ── 1. how often was forcing actually needed? ────────────────────────
    rows = []
    mech_all = Counter(r.get("force_mechanism") for r in AR.values())
    n = sum(mech_all.values())
    payload["force_mechanism"] = {}
    for mech in ("none_needed", "nudge_1", "nudge_2", "synthetic", "not_forced"):
        c = mech_all.get(mech, 0)
        if c == 0 and mech == "not_forced":
            continue
        vis = sum(1 for r in AR.values()
                  if r.get("force_mechanism") == mech and r["question_type"] == "visual")
        ctx = c - vis
        payload["force_mechanism"][mech] = {"n": c, "visual": vis, "contextual": ctx}
        rows.append([MECH_LABEL[mech], c, f"{c / n:.1%}", vis, ctx])
    zero_call = sum(1 for r in AR.values() if not r["retrieved"])
    payload["zero_call_items"] = zero_call
    out.append("#### How often did forcing actually have to intervene?\n")
    out.append(lib.table_block(
        "always_mechanism",
        ["outcome of the first turn", "n", "% of items", "visual", "contextual"],
        rows))
    out.append(
        f"\n*Items that still ended with zero retrieval calls: "
        f"**{zero_call}** (the invariant this variant exists to enforce).*"
    )

    # ── 2. accuracy and cost against the learned policy ──────────────────
    runs = {(r["variant"], r["query_id"]): r for r in lib.load_runs()}

    def cost(d, variant):
        calls = [r["num_tool_calls"] or 0 for r in d.values()]
        tot = [runs[(variant, q)]["timing"]["total"]
               for q in d if (variant, q) in runs and runs[(variant, q)].get("timing")]
        return (sum(calls) / len(calls) if calls else float("nan"),
                sum(tot) / len(tot) if tot else float("nan"))

    common = sorted(set(AR) & set(FC))
    rows = []
    payload["comparison"] = {}
    for label, keys in (("all", common),
                        ("visual", [q for q in common if FC[q]["question_type"] == "visual"]),
                        ("contextual", [q for q in common if FC[q]["question_type"] == "contextual"])):
        if not keys:
            continue
        sub_fc = {q: FC[q] for q in keys}
        sub_ar = {q: AR[q] for q in keys}
        # paired_diffs(A, B) yields B - A, so full_classify goes first to
        # make a POSITIVE delta mean "forcing retrieval helped".
        d = lib.cluster_bootstrap_paired(
            lib.paired_diffs(sub_fc, sub_ar, "correctness"), n_boot=a.n_boot)
        dr = lib.cluster_bootstrap_paired(
            lib.paired_diffs(sub_fc, sub_ar, "recall"), n_boot=a.n_boot)
        c_fc, t_fc = cost(sub_fc, "full_classify")
        c_ar, t_ar = cost(sub_ar, "full_alwaysretrieve")
        e = {"n": len(keys),
             "correctness_full_classify": sum(r["correctness"] for r in sub_fc.values()) / len(keys),
             "correctness_alwaysretrieve": sum(r["correctness"] for r in sub_ar.values()) / len(keys),
             "delta_correctness": d, "delta_recall": dr,
             "calls_full_classify": c_fc, "calls_alwaysretrieve": c_ar,
             "seconds_full_classify": t_fc, "seconds_alwaysretrieve": t_ar}
        payload["comparison"][label] = e
        rows.append([
            label, len(keys),
            lib.fmt(e["correctness_full_classify"]), lib.fmt(e["correctness_alwaysretrieve"]),
            f"{d['mean']:+.3f}", lib.fmt_ci(d),
            lib.fmt(c_fc, 2), lib.fmt(c_ar, 2),
            lib.fmt(t_fc, 1), lib.fmt(t_ar, 1),
            f"{(t_ar / t_fc - 1) * 100:+.0f}%" if t_fc else "–",
        ])
    out.append("\n#### Learned policy vs. forced retrieval: accuracy and cost\n")
    out.append(lib.table_block(
        "always_comparison",
        ["questions", "n", "correctness (`full_classify`)",
         "correctness (`full_alwaysretrieve`)", "Δ", "95% cluster CI",
         "calls (`full_classify`)", "calls (`full_alwaysretrieve`)",
         "seconds (`full_classify`)", "seconds (`full_alwaysretrieve`)",
         "extra wall-clock"], rows))
    out.append(
        "\n*Δ is `full_alwaysretrieve` − `full_classify`, so a positive Δ "
        "means forcing retrieval helped. The wall-clock for "
        "`full_alwaysretrieve` includes the rejected first turns — forcing "
        "is not free and is billed here.*"
    )

    # ── 3. where forcing changed the outcome ─────────────────────────────
    changed = [(q, FC[q]["correctness"], AR[q]["correctness"]) for q in common
               if FC[q]["correctness"] != AR[q]["correctness"]]
    helped = sum(1 for _, f, r in changed if r > f)
    hurt = sum(1 for _, f, r in changed if r < f)
    skipped_and_right = [q for q in common
                         if FC[q]["question_type"] == "visual" and not FC[q]["retrieved"]]
    sr_helped = sum(1 for q in skipped_and_right if AR[q]["correctness"] > FC[q]["correctness"])
    sr_hurt = sum(1 for q in skipped_and_right if AR[q]["correctness"] < FC[q]["correctness"])
    payload["outcome_changes"] = {
        "n_changed": len(changed), "helped": helped, "hurt": hurt,
        "visual_skipped_by_policy": len(skipped_and_right),
        "of_those_forcing_helped": sr_helped, "of_those_forcing_hurt": sr_hurt,
    }
    out.append("\n#### Where forcing changed the outcome\n")
    out.append(lib.table_block(
        "always_changes",
        ["items where the two variants disagree", "forcing helped", "forcing hurt",
         "visual questions the policy skipped", "…of those, forcing helped",
         "…of those, forcing hurt"],
        [[len(changed), helped, hurt, len(skipped_and_right), sr_helped, sr_hurt]]))

    lib.write_table(f"alwaysretrieve__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
