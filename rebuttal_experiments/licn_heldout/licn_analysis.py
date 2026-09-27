"""LICNHeldOut analysis.

Reports how often each LICN head is correct on the benchmark, the overall
results, and, per question type (= per head), the paired effect of showing the
artwork card (`full_noclassify` -> `full_classify`, painting-level bootstrap).

Usage:
    python rebuttal_experiments/licn_heldout/licn_analysis.py [--judge phi4]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("ARTSEEK_EXP_DIR", str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "analysis"))

import lib

HEADS = {"artist": "artist", "movement": "style", "genre": "genre",
         "material": "media"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def card_hits(entries: list[dict], vqa: dict) -> dict:
    """Per head: was LICN's top prediction present in the gold answer?"""
    out = defaultdict(lambda: [0, 0])
    for e in entries:
        card = e.get("card") or {}
        qt = e.get("question_type")
        head = HEADS.get(qt)
        if not head:
            continue
        preds = card.get(head) or []
        if not preds:
            continue
        gold = norm(e.get("reference_answer", ""))
        top = norm(preds[0][0]) if isinstance(preds[0], (list, tuple)) else norm(preds[0])
        # an artist slug is hyphenated; compare on tokens either way
        ok = bool(top) and (top in gold or all(w in gold.split() for w in top.split()))
        out[qt][0] += int(ok)
        out[qt][1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    J = lib.load_judge(judge, "answer")
    if not J:
        raise SystemExit("no LICNHeldOut judgments yet")
    by = defaultdict(dict)
    for (v, q), r in J.items():
        by[v][q] = r

    vqa = {r["id"]: r for r in json.loads(
        (Path(__file__).resolve().parent / "data" / "licn_vqa.json").read_text())}

    # ── is the card actually right here? ─────────────────────────────────
    cf = lib.RESULTS_DIR / "full_classify.json"
    rows = []
    if cf.exists():
        hits = card_hits(json.loads(cf.read_text()), vqa)
        for qt, head in HEADS.items():
            ok, n = hits.get(qt, [0, 0])
            if n:
                rows.append([qt, f"`{head}`", n, ok,
                             lib.fmt(ok / n * 100, 1) + "%"])
    out = ["#### Is LICN right here? (its top prediction vs the Wikidata answer)\n"]
    out.append(lib.table_block(
        "licn_card_accuracy",
        ["question type", "LICN head", "n", "top-1 correct", "accuracy"], rows))

    # ── overall ─────────────────────────────────────────────────────────
    base = by.get("base", {})
    rows = []
    for v in sorted(by, key=lambda x: -sum(r["correctness"] for r in by[x].values()) / max(len(by[x]), 1)):
        d = by[v]
        m = sum(r["correctness"] for r in d.values()) / len(d)
        row = [f"`{v}`", len(d), lib.fmt(m)]
        common = sorted(set(d) & set(base))
        if v != "base" and common:
            r = lib.cluster_bootstrap_paired(
                [(lib.painting_of(q), d[q]["correctness"] - base[q]["correctness"])
                 for q in common], n_boot=a.n_boot)
            sig = not (r["ci_low"] <= 0 <= r["ci_high"])
            row += [f"{r['mean']:+.3f}", lib.fmt_ci(r), "**yes**" if sig else "no"]
        else:
            row += ["—", "—", "—"]
        rows.append(row)
    out.append("\n#### Overall\n")
    out.append(lib.table_block(
        "licn_overall",
        ["variant", "n", "correctness (0–2)", "Δ vs `base`", "95% cluster CI",
         "CI excludes 0"], rows))

    # ── per head: does showing the card help? ───────────────────────────
    A, B = by.get("full_noclassify"), by.get("full_classify")
    rows = []
    if A and B:
        for qt, head in HEADS.items():
            common = sorted(q for q in set(A) & set(B) if q.endswith("_" + qt))
            if not common:
                continue
            r = lib.cluster_bootstrap_paired(
                [(lib.painting_of(q), B[q]["correctness"] - A[q]["correctness"])
                 for q in common], n_boot=a.n_boot)
            sig = not (r["ci_low"] <= 0 <= r["ci_high"])
            rows.append([qt, f"`{head}`", r["n"],
                         lib.fmt(sum(A[q]["correctness"] for q in common) / len(common)),
                         lib.fmt(sum(B[q]["correctness"] for q in common) / len(common)),
                         f"{r['mean']:+.3f}", lib.fmt_ci(r),
                         "**yes**" if sig else "no"])
    out.append("\n#### The card's effect per head (`full_noclassify` → `full_classify`)\n")
    out.append(lib.table_block(
        "licn_per_head",
        ["question type", "LICN head", "n", "no card", "with card", "mean Δ",
         "95% cluster CI", "CI excludes 0"], rows))

    lib.write_table(f"licn__{a.judge}", {"judge_model": judge})
    print("\n".join(out))


if __name__ == "__main__":
    main()
