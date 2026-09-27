"""How many retrieval calls does the agent issue per question?

Distribution of tool calls (0, 1, >=2) of a variant on every benchmark,
compared with the three calls demonstrated by the one-shot example.

Usage:
    python rebuttal_experiments/cross_benchmark/decomposition.py [--variant full_classify]
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "rebuttal_experiments"
sys.path.insert(0, str(EXPERIMENTS / "common" / "analysis"))
import lib  # noqa: E402

BENCHMARKS = [
    ("ArtPedia-VQA", "artpedia_vqa", "single-hop"),
    ("AQUA", "aqua", "single-hop"),
    ("ArtQuest", "artquest", "single-hop"),
    ("LICNHeldOut", "licn_heldout", "single-hop"),
    ("ArtCurate-AIC", "artcurate_aic", "conceptual"),
]

# Number of calls demonstrated by the one-shot example.
SHOT_ANN = REPO / "artseek" / "method" / "generate" / "shot" / "ann.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full_classify")
    a = ap.parse_args()

    rows, payload = [], {"variant": a.variant, "benchmarks": {}}
    for name, exp, kind in BENCHMARKS:
        path = EXPERIMENTS / exp / "results" / f"{a.variant}.json"
        if not path.exists():
            continue
        ok = [x for x in json.loads(path.read_text()) if "error" not in x]
        if not ok:
            continue
        calls = [x.get("num_tool_calls") or 0 for x in ok]
        n = len(calls)
        e = {
            "n": n, "kind": kind,
            "mean_calls": sum(calls) / n,
            "pct_zero": sum(1 for c in calls if c == 0) / n * 100,
            "pct_one": sum(1 for c in calls if c == 1) / n * 100,
            "pct_two_plus": sum(1 for c in calls if c >= 2) / n * 100,
            "max_calls": max(calls),
        }
        payload["benchmarks"][name] = e
        rows.append([name, kind, n, lib.fmt(e["mean_calls"], 2),
                     lib.fmt(e["pct_zero"], 1) + "%",
                     lib.fmt(e["pct_one"], 1) + "%",
                     f"**{lib.fmt(e['pct_two_plus'], 1)}%**",
                     e["max_calls"]])

    n_shot = None
    if SHOT_ANN.exists():
        n_shot = len(json.loads(SHOT_ANN.read_text()).get("queries", []))
        payload["shot_demonstrated_calls"] = n_shot

    out = [f"#### How often does `{a.variant}` issue more than one retrieval?\n"]
    out.append(lib.table_block(
        "decomposition",
        ["benchmark", "question structure", "n", "mean calls", "0 calls",
         "1 call", "≥2 calls", "max"], rows))
    if n_shot:
        out.append(
            f"\n*For comparison, the in-context example the policy is taught "
            f"from demonstrates **{n_shot} retrieval calls**, the last of "
            f"which is a text-only follow-up derived from what the earlier "
            f"calls established.*"
        )
    lib.write_table("decomposition", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
