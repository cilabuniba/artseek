"""Oracle probe analysis: is a missing fact absent from the index, or just not
retrieved?

Compares the evidence judgments of the documents retrieved by a variant with
those of the oracle probe (oracle_probe.py, query = question + reference
answer). The oracle query contains the answer, so it gives an upper bound on
retrievability, not an achievable operating point.

Usage:
    python rebuttal_experiments/common/analysis/oracle_analysis.py [--judge phi4]
                                                                   [--against full_classify]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--against", default="full_classify")
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    ev = lib.load_judge(judge, "evidence")
    oracle = {k[1]: v for k, v in ev.items() if k[0] == lib.ORACLE_VARIANT}
    actual = {k[1]: v for k, v in ev.items() if k[0] == a.against}
    if not oracle:
        raise SystemExit("no oracle-probe evidence judgments yet "
                         "— run slurm/60_oracle_probe.sh")
    common = sorted(set(oracle) & set(actual))
    if not common:
        raise SystemExit("oracle probe and actual retrieval share no queries")

    O = [oracle[q] for q in common]
    F = [actual[q] for q in common]

    def pct(rows, f):
        return sum(1 for r in rows if f(r)) / len(rows) * 100

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    rows = [
        ["evidence absent (0)",
         lib.fmt(pct(F, lambda r: r["evidence_present"] == 0), 1) + "%",
         lib.fmt(pct(O, lambda r: r["evidence_present"] == 0), 1) + "%"],
        ["evidence partially present (1)",
         lib.fmt(pct(F, lambda r: r["evidence_present"] == 1), 1) + "%",
         lib.fmt(pct(O, lambda r: r["evidence_present"] == 1), 1) + "%"],
        ["evidence fully present (2)",
         lib.fmt(pct(F, lambda r: r["evidence_present"] == 2), 1) + "%",
         lib.fmt(pct(O, lambda r: r["evidence_present"] == 2), 1) + "%"],
        ["mean evidence recall",
         lib.fmt(mean(F, "evidence_recall")), lib.fmt(mean(O, "evidence_recall"))],
    ]

    out = ["#### What the index holds vs. what the model's query found\n"]
    out.append(lib.table_block(
        "oracle_vs_actual",
        ["", f"model's own query (`{a.against}`)", "gold-informed query"], rows))

    miss = [q for q in common if actual[q]["evidence_present"] == 0]
    unreachable = [q for q in miss if oracle[q]["evidence_present"] == 0]
    reachable = [q for q in miss if oracle[q]["evidence_present"] > 0]
    n = max(len(miss), 1)
    out.append("\n#### Where the model's retrieval found nothing, was the fact there?\n")
    out.append(lib.table_block(
        "oracle_decomposition",
        ["queries where the model's retrieval found no evidence",
         "fact NOT reachable in the index", "fact WAS in the index, query missed it"],
        [[len(miss),
          f"{len(unreachable)} ({len(unreachable) / n:.1%})",
          f"{len(reachable)} ({len(reachable) / n:.1%})"]]))

    lib.write_table(f"oracle__{a.judge}", {
        "judge_model": judge, "against": a.against, "n": len(common),
        "actual": {"absent_pct": pct(F, lambda r: r["evidence_present"] == 0),
                   "present_pct": pct(F, lambda r: r["evidence_present"] == 2),
                   "mean_recall": mean(F, "evidence_recall")},
        "oracle": {"absent_pct": pct(O, lambda r: r["evidence_present"] == 0),
                   "present_pct": pct(O, lambda r: r["evidence_present"] == 2),
                   "mean_recall": mean(O, "evidence_recall")},
        "misses": len(miss), "unreachable": len(unreachable),
        "reachable_but_missed": len(reachable),
        "reachable_share": len(reachable) / n,
    })
    print("\n".join(out))


if __name__ == "__main__":
    main()
