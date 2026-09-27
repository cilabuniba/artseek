"""Evidence analysis: did the retrieved fragments contain the answer?

Uses the evidence judgments (judge.py --pass evidence) to report, per variant,
evidence recall and presence, the joint table evidence x answer correctness,
and the attribution of wrong answers:

  (i)   never retrieved                 -> retrieval policy
  (ii)  retrieved, evidence absent      -> query / knowledge-base coverage
  (iii) evidence present, answer wrong  -> generation / grounding
  (iv)  evidence present, answer right

Usage:
    python rebuttal_experiments/common/analysis/evidence.py [--judge phi4] [--subset ...]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib

PRESENT_LABEL = {0: "absent", 1: "partial", 2: "present"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--subset", default=None, choices=["obscure", "nonobscure"])
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    scored = lib.load_scored(judge)
    scored.pop("base", None)
    ev = lib.load_judge(judge, "evidence")
    if not ev:
        raise SystemExit(
            "no evidence judgments yet — run slurm/50_judge_evidence.sh first"
        )

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

    # ── 1. evidence quality per variant ──────────────────────────────────
    rows = []
    payload["evidence_quality"] = {}
    for v in variants:
        items = [r for q, r in scored[v].items() if (v, q) in ev]
        if not items:
            continue
        entry = {}
        cells = [f"`{v}`", len(items)]
        for scope, sub in (("all", items),
                           ("visual", [r for r in items if r["question_type"] == "visual"]),
                           ("contextual", [r for r in items if r["question_type"] == "contextual"])):
            if not sub:
                continue
            evs = [ev[(v, r["query_id"])] for r in sub]
            e = {
                "n": len(sub),
                "mean_evidence_recall": sum(x["evidence_recall"] for x in evs) / len(evs),
                "pct_absent": sum(1 for x in evs if x["evidence_present"] == 0) / len(evs) * 100,
                "pct_partial": sum(1 for x in evs if x["evidence_present"] == 1) / len(evs) * 100,
                "pct_present": sum(1 for x in evs if x["evidence_present"] == 2) / len(evs) * 100,
            }
            entry[scope] = e
            if scope == "all":
                cells += [lib.fmt(e["mean_evidence_recall"]),
                          lib.fmt(e["pct_absent"], 1) + "%",
                          lib.fmt(e["pct_partial"], 1) + "%",
                          lib.fmt(e["pct_present"], 1) + "%"]
            else:
                cells += [lib.fmt(e["mean_evidence_recall"])]
        payload["evidence_quality"][v] = entry
        rows.append(cells)
    out.append("#### Retrieval success: did the fragments contain the answer?\n")
    out.append(lib.table_block(
        "evidence_quality",
        ["variant", "n (with retrieval)", "mean evidence recall", "% absent (0)",
         "% partial (1)", "% present (2)", "ev. recall (vis)", "ev. recall (ctx)"],
        rows))

    # ── 2. joint table ───────────────────────────────────────────────────
    payload["joint"] = {}
    out.append("\n#### Joint decomposition: evidence present × answer correct\n")
    for v in variants:
        items = list(scored[v].values())
        if not any((v, r["query_id"]) in ev for r in items):
            continue
        cells, total = [], len(items)
        counts = {}
        for r in items:
            key_ev = ("no retrieval" if not r["retrieved"]
                      else PRESENT_LABEL[ev[(v, r["query_id"])]["evidence_present"]]
                      if (v, r["query_id"]) in ev else "unjudged")
            key_ans = {0: "wrong", 1: "partial", 2: "correct"}[r["correctness"]]
            counts[(key_ev, key_ans)] = counts.get((key_ev, key_ans), 0) + 1
        order_ev = ["no retrieval", "absent", "partial", "present", "unjudged"]
        order_ans = ["wrong", "partial", "correct"]
        for ke in order_ev:
            for ka in order_ans:
                n = counts.get((ke, ka), 0)
                if n:
                    cells.append([ke, ka, n, f"{n / total:.1%}"])
        payload["joint"][v] = {f"{k[0]}|{k[1]}": n for k, n in counts.items()}
        payload["joint"][v]["_total"] = total
        out.append(f"\n**`{v}`** (n={total})\n")
        out.append(lib.table_block(
            f"evidence_joint__{v}",
            ["evidence present?", "model answer", "count", "%"], cells))

    # ── 3. conditional accuracy + failure attribution ────────────────────
    rows = []
    payload["conditional"] = {}
    for v in variants:
        items = list(scored[v].values())
        judged = [r for r in items if (v, r["query_id"]) in ev]
        if not judged:
            continue
        present = [r for r in judged if ev[(v, r["query_id"])]["evidence_present"] == 2]
        atleast = [r for r in judged if ev[(v, r["query_id"])]["evidence_present"] >= 1]
        absent = [r for r in judged if ev[(v, r["query_id"])]["evidence_present"] == 0]
        no_ret = [r for r in items if not r["retrieved"]]
        wrong = [r for r in items if r["correctness"] == 0]
        e = {
            "acc_given_present": sum(r["correctness"] for r in present) / len(present) if present else float("nan"),
            "pct_correct_given_present": sum(1 for r in present if r["correctness"] == 2) / len(present) * 100 if present else float("nan"),
            "acc_given_atleast_partial": sum(r["correctness"] for r in atleast) / len(atleast) if atleast else float("nan"),
            "acc_given_absent": sum(r["correctness"] for r in absent) / len(absent) if absent else float("nan"),
            # attribution of the wrong answers
            "n_wrong": len(wrong),
            "wrong_no_retrieval": sum(1 for r in wrong if not r["retrieved"]),
            "wrong_evidence_absent": sum(1 for r in wrong if r["retrieved"] and (v, r["query_id"]) in ev
                                         and ev[(v, r["query_id"])]["evidence_present"] == 0),
            "wrong_evidence_present": sum(1 for r in wrong if r["retrieved"] and (v, r["query_id"]) in ev
                                          and ev[(v, r["query_id"])]["evidence_present"] >= 1),
            "n_no_retrieval": len(no_ret),
        }
        payload["conditional"][v] = e
        w = max(e["n_wrong"], 1)
        rows.append([
            f"`{v}`",
            lib.fmt(e["acc_given_absent"]), lib.fmt(e["acc_given_atleast_partial"]),
            lib.fmt(e["acc_given_present"]),
            lib.fmt(e["pct_correct_given_present"], 1) + "%",
            f"{e['wrong_no_retrieval'] / w:.0%}",
            f"{e['wrong_evidence_absent'] / w:.0%}",
            f"{e['wrong_evidence_present'] / w:.0%}",
        ])
    out.append("\n#### Conditional answer accuracy, and what the wrong answers are attributable to\n")
    out.append(lib.table_block(
        "evidence_conditional",
        ["variant", "correctness given evidence absent",
         "correctness given evidence ≥ partial",
         "correctness given evidence present",
         "% fully correct given evidence present",
         "wrong: no retrieval", "wrong: KB coverage", "wrong: grounding"], rows))
    out.append(
        "\n*The last three columns partition each variant's wrong answers. "
        "\"KB coverage\" means the policy retrieved but WikiFragments did not "
        "contain the fact — not a policy failure. \"Grounding\" means the "
        "evidence was there and the model still got it wrong.*"
    )

    lib.write_table(f"evidence__{tag}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
