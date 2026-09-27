"""Per-stage and end-to-end latency, aggregated from the `timing` recorded by
run_experiment.py on every item.

`reasoning` / `answer` are estimates (the completion is split by token count
at `</think>`). The Qdrant prefetch and rerank are a single request, so only
query embedding vs. search can be separated, and only for runs that recorded
it.

Usage:
    python rebuttal_experiments/common/analysis/latency.py [--judge phi4]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib


def pct(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    a = ap.parse_args()

    runs = [r for r in lib.load_runs() if r["status"] == "ok" and r.get("timing")]
    variants = [v for v in lib.VARIANTS if any(r["variant"] == v for r in runs)]
    by = {v: [r for r in runs if r["variant"] == v] for v in variants}

    out, payload = [], {"per_stage": {}, "end_to_end": {}, "per_call": {}}

    # ── per-stage means ──────────────────────────────────────────────────
    rows = []
    for v in variants:
        rs = by[v]
        t = lambda k: [r["timing"].get(k) or 0.0 for r in rs]  # noqa: E731
        entry = {
            "n": len(rs),
            "classification": mean(t("classification")),
            "generation_total": mean(t("generation_total")),
            "reasoning_est": mean(t("reasoning")),
            "answer_est": mean(t("answer")),
            "retrieval": mean(t("retrieval")),
            "total": mean(t("total")),
        }
        payload["per_stage"][v] = entry
        rows.append([f"`{v}`", entry["n"], lib.fmt(entry["classification"], 2),
                     lib.fmt(entry["generation_total"], 2), lib.fmt(entry["reasoning_est"], 2),
                     lib.fmt(entry["answer_est"], 2), lib.fmt(entry["retrieval"], 2),
                     lib.fmt(entry["total"], 2)])
    out.append("#### Mean per-stage wall-clock (seconds/query)\n")
    out.append(lib.table_block(
        "latency_stages",
        ["variant", "n", "LICN classify", "generation (all steps)",
         "· reasoning (est.)", "· answer (est.)", "retrieval (all calls)", "end-to-end"],
        rows))

    # ── end-to-end distribution, split by class ──────────────────────────
    rows = []
    for v in variants:
        entry = {}
        cells = [f"`{v}`"]
        for qt in ("visual", "contextual"):
            tot = [r["timing"]["total"] for r in by[v] if r["question_type"] == qt]
            entry[qt] = {"n": len(tot), "mean": mean(tot),
                         "p50": pct(tot, 0.50), "p95": pct(tot, 0.95)}
            cells += [lib.fmt(entry[qt]["p50"], 1), lib.fmt(entry[qt]["p95"], 1),
                      lib.fmt(entry[qt]["mean"], 1)]
        allt = [r["timing"]["total"] for r in by[v]]
        entry["all"] = {"n": len(allt), "mean": mean(allt),
                        "p50": pct(allt, 0.50), "p95": pct(allt, 0.95)}
        cells += [lib.fmt(entry["all"]["p50"], 1), lib.fmt(entry["all"]["p95"], 1),
                  lib.fmt(entry["all"]["mean"], 1)]
        payload["end_to_end"][v] = entry
        rows.append(cells)
    out.append("\n#### End-to-end latency by question class (seconds)\n")
    out.append(lib.table_block(
        "latency_e2e",
        ["variant", "vis p50", "vis p95", "vis mean", "ctx p50", "ctx p95",
         "ctx mean", "all p50", "all p95", "all mean"],
        rows))

    # ── cost decomposition: calls x per-call cost ────────────────────────
    rows = []
    for v in variants:
        rs = by[v]
        calls = [r["num_tool_calls"] or 0 for r in rs]
        retrieving = [r for r in rs if (r["num_tool_calls"] or 0) > 0]
        per_call = [r["timing"]["retrieval"] / r["num_tool_calls"] for r in retrieving]
        mc, mpc = mean(calls), mean(per_call)
        entry = {"mean_calls": mc, "mean_per_call_s": mpc,
                 "implied_retrieval_s": (mc * mpc) if per_call else 0.0,
                 "observed_retrieval_s": mean([r["timing"]["retrieval"] for r in rs]),
                 "pct_of_end_to_end": (mean([r["timing"]["retrieval"] for r in rs])
                                       / mean([r["timing"]["total"] for r in rs]) * 100)}
        payload["per_call"][v] = entry
        rows.append([f"`{v}`", lib.fmt(mc, 2), lib.fmt(mpc, 2),
                     lib.fmt(entry["observed_retrieval_s"], 2),
                     lib.fmt(entry["pct_of_end_to_end"], 1) + "%"])
    out.append("\n#### Retrieval cost decomposition\n")
    out.append(lib.table_block(
        "latency_cost",
        ["variant", "mean tool calls", "mean cost per call (s)",
         "mean retrieval time (s)", "retrieval as % of end-to-end"],
        rows))

    # ── embed vs. search split, where instrumented ───────────────────────
    stage_rows = []
    for v in variants:
        embeds, searches = [], []
        for r in by[v]:
            for s in (r.get("retrieval_stages") or []):
                if s.get("embed") is not None:
                    embeds.append(s["embed"])
                if s.get("search") is not None:
                    searches.append(s["search"])
        if embeds or searches:
            stage_rows.append([f"`{v}`", len(embeds), lib.fmt(mean(embeds), 3),
                               lib.fmt(mean(searches), 3)])
            payload.setdefault("retrieval_substages", {})[v] = {
                "n_calls": len(embeds), "mean_embed_s": mean(embeds),
                "mean_search_s": mean(searches)}
    if stage_rows:
        out.append("\n#### Retrieval sub-stages, where instrumented\n")
        out.append(lib.table_block(
        "latency_substages",
        ["variant", "calls", "mean query-embedding (s)", "mean Qdrant search (s)"],
            stage_rows))
        out.append(
            "\n*The Qdrant search column covers prefetch **and** rerank: they "
            "are issued as a single request and cannot be timed apart from "
            "the client.*"
        )

    lib.write_table("latency", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
