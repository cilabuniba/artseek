"""ArtQuest scores, comparable with the published baselines.

Bleidt et al. report exact match against a short gold span; our models answer
in sentences. Three metrics, strictest first:

  em        gold == answer after casefolding and removing punctuation and
            a/an/the/of/on/in.
  contains  the normalised gold span occurs in the normalised answer (the
            closest analogue of exact match for sentence answers).
  judged    phi-4 grade 2 on the short-answer rubric.

Reports the overall table, the per-type comparison with the published
PrefixLM numbers (`contains`), and the per-type effect of the artwork card
(`full_noclassify` -> `full_classify`, painting-level bootstrap).

Usage:
    python rebuttal_experiments/artquest/artquest_score.py [--judge phi4]
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

TYPES = ("artist", "title", "technique", "school", "timeframe", "type")

# Their published PrefixLM numbers (Tab. 5 / Tab. 6), for the comparison table.
# Fine-tuned on ArtQuest's 115k training questions; ours is zero-shot.
THEIRS = {
    "overall": {"VIKING (FT, closed-book)": 37.9,
                "PrefixLM (FT, closed-book)": 50.2,
                "PrefixLM (FT, open-book)": 53.5,
                "OFA-base (zero-shot)": 1.9,
                "BLIP-base (zero-shot)": 2.4},
    "per_type": {  # PrefixLM open-book, EM
        "artist": 34.1, "title": 11.5, "technique": 61.2,
        "school": 69.4, "timeframe": 61.5, "type": 83.2},
}

# Per-type numbers are published only for the fine-tuned PrefixLM; the
# zero-shot baselines have overall figures only.
THEIRS_PER_TYPE_ZEROSHOT = {}

# Variants in the per-type table.
PER_TYPE_VARIANTS = ("base", "full_noclassify", "full_classify",
                     "full_systemprompt")

_ARTICLES = re.compile(r"\b(a|an|the|of|on|in)\b")


def norm(s: str) -> str:
    s = str(s).casefold()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    J = lib.load_judge(judge, "answer")
    judged = defaultdict(dict)
    for (v, q), r in J.items():
        judged[v][q] = r

    scores = defaultdict(lambda: defaultdict(list))  # variant -> metric -> [(qid, val)]
    per_type = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for p in sorted((lib.RESULTS_DIR).glob("*.json")):
        if p.name.endswith(("_judged.json", "_eval_summary.json")):
            continue
        entries = json.loads(p.read_text())
        if not entries or "model_answer" not in entries[0]:
            continue
        v = p.stem
        for e in entries:
            if "error" in e:
                continue
            qid = f"{e['id']}_{e['question_type']}"
            gold, ans = norm(e["reference_answer"]), norm(e.get("model_answer", ""))
            em = float(gold == ans)
            contains = float(bool(gold) and gold in ans)
            jr = judged.get(v, {}).get(qid)
            jd = float(jr["correctness"] == 2) if jr else None
            for metric, val in (("em", em), ("contains", contains), ("judged", jd)):
                if val is None:
                    continue
                scores[v][metric].append((qid, val))
                per_type[v][e["question_type"]][metric].append(val)

    if not scores:
        raise SystemExit("no ArtQuest results yet")

    out = ["#### ArtQuest test subset — overall accuracy\n"]
    rows = []
    for v in sorted(scores, key=lambda x: -sum(t[1] for t in scores[x]["contains"]) / max(len(scores[x]["contains"]), 1)):
        m = scores[v]
        rows.append([f"`{v}`", len(m["em"]),
                     lib.fmt(sum(x[1] for x in m["em"]) / len(m["em"]) * 100, 1) + "%",
                     lib.fmt(sum(x[1] for x in m["contains"]) / len(m["contains"]) * 100, 1) + "%",
                     (lib.fmt(sum(x[1] for x in m["judged"]) / len(m["judged"]) * 100, 1) + "%")
                     if m["judged"] else "—"])
    out.append(lib.table_block(
        "artquest_overall",
        ["variant", "n", "exact match", "contains gold span", "phi-4 judged"], rows))

    out.append("\n#### Against the published baselines (their metric: exact match)\n")
    rows = [[k, "fine-tuned" if "FT" in k else "zero-shot", lib.fmt(vv, 1) + "%", "—"]
            for k, vv in THEIRS["overall"].items()]
    for v in sorted(scores):
        m = scores[v]
        rows.append([f"ArtSeek `{v}`", "zero-shot",
                     lib.fmt(sum(x[1] for x in m["contains"]) / len(m["contains"]) * 100, 1) + "%",
                     (lib.fmt(sum(x[1] for x in m["judged"]) / len(m["judged"]) * 100, 1) + "%")
                     if m["judged"] else "—"])
    out.append(lib.table_block(
        "artquest_vs_published",
        ["model", "training", "exact match / contains", "phi-4 judged"], rows))
    out.append("\n*Their numbers are exact match for models fine-tuned on "
               "ArtQuest's 115k training questions; ours are zero-shot, scored "
               "by `contains` (the gold span must appear verbatim in our "
               "answer). `contains` is more permissive than EM for a "
               "sentence-form answer and stricter than the judge.*")

    def pct(v, t, metric):
        """Percentage for one (variant, type, metric), or None if not run."""
        vals = per_type.get(v, {}).get(t, {}).get(metric)
        return sum(vals) / len(vals) * 100 if vals else None

    def cell(x):
        return lib.fmt(x, 1) + "%" if x is not None else "—"

    # --- the per-type comparison against the published baselines -----------
    # Scored with `contains`, comparable with their exact match.
    out.append("\n#### Per question type — zero-shot ArtSeek against the "
               "fine-tuned baseline\n")
    rows = []
    for t in TYPES:
        theirs = THEIRS["per_type"][t]
        ours = {v: pct(v, t, "contains") for v in PER_TYPE_VARIANTS}
        run = {v: x for v, x in ours.items() if x is not None}
        best_v = max(run, key=run.get) if run else None
        row = [t, lib.fmt(theirs, 1) + "%"]
        for v in PER_TYPE_VARIANTS:
            x = ours[v]
            # Bold the best of our variants on this row.
            row.append(f"**{cell(x)}**" if v == best_v and x is not None
                       else cell(x))
        if best_v is None:
            row += ["—", "—"]
        else:
            delta = run[best_v] - theirs
            row.append(f"{delta:+.1f}pp")
            row.append("**yes**" if delta > 0 else "no")
        rows.append(row)
    out.append(lib.table_block(
        "artquest_per_type_vs_published",
        ["question type", "PrefixLM (FT, open-book) EM"]
        + [f"`{v}`" for v in PER_TYPE_VARIANTS]
        + ["best ArtSeek − theirs", "ArtSeek ahead"],
        rows))
    out.append(
        "\n*All ArtSeek columns are `contains` — the gold span must appear "
        "verbatim in our answer — which is the closest comparable analogue of "
        "their exact match for a model that answers in sentences. PrefixLM was "
        "fine-tuned on ArtQuest's 115k training questions; every ArtSeek "
        "column is zero-shot. Bold marks the best ArtSeek variant on that "
        "row.*")
    if not THEIRS_PER_TYPE_ZEROSHOT:
        out.append(
            "\n*Bleidt et al. publish OFA-base (1.9%) and BLIP-base (2.4%) as "
            "overall figures only, so the zero-shot comparison — the one that "
            "matches our training condition — lives in the overall table "
            "above and cannot be broken down per type from their paper.*")

    out.append("\n#### Per question type — the same rows under the phi-4 judge\n")
    rows = []
    for t in TYPES:
        rows.append([t] + [cell(pct(v, t, "judged")) for v in PER_TYPE_VARIANTS])
    out.append(lib.table_block(
        "artquest_per_type",
        ["question type"] + [f"`{v}`" for v in PER_TYPE_VARIANTS],
        rows))
    out.append(
        "\n*`contains` is unfair to `timeframe`: the gold span is a bracket "
        "like \"1601-1650\" and the model answers \"the early seventeenth "
        "century\", which is right and scores zero. The judged column is the "
        "meaningful one for that row, and it is not comparable to a published "
        "EM number — which is why it is tabled separately rather than pitted "
        "against theirs.*")

    # paired card effect per type, cluster-bootstrapped over paintings
    rows = []
    for t in TYPES:
        a_map = {q: val for (q, val) in scores.get("full_noclassify", {}).get("contains", [])
                 if q.endswith("_" + t)}
        b_map = {q: val for (q, val) in scores.get("full_classify", {}).get("contains", [])
                 if q.endswith("_" + t)}
        common = sorted(set(a_map) & set(b_map))
        if not common:
            continue
        r = lib.cluster_bootstrap_paired(
            [(lib.painting_of(q), b_map[q] - a_map[q]) for q in common], n_boot=a.n_boot)
        sig = not (r["ci_low"] <= 0 <= r["ci_high"])
        rows.append([t, r["n"], f"{r['mean'] * 100:+.1f}pp", lib.fmt_ci(r),
                     "**yes**" if sig else "no"])
    if rows:
        out.append("\n#### The card's effect, per question type "
                   "(`full_noclassify` → `full_classify`)\n")
        out.append(lib.table_block(
            "artquest_card_effect",
            ["question type", "n", "mean Δ", "95% cluster CI", "CI excludes 0"], rows))
        out.append(
            "\n*Registered before the run: LICN's `media` head covers 99.3% of "
            "`technique` answers by token and its `genre` head covers 36.8% of "
            "`type` answers exactly, while `school` and `title` have no "
            "corresponding head. The card should help on technique and type "
            "and do nothing on school and title.*")

    print("\n".join(out))


if __name__ == "__main__":
    main()
