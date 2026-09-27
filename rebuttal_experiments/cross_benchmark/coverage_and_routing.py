"""Help/hurt, routable headroom and knowledge-base coverage per benchmark.

1. Help / hurt: per item, how often `full_classify` scores higher / lower than
   `base` (phi-4 judge), and their ratio.
2. Routable headroom: the gain of an oracle that picks the better of the two
   per item, minus a noise floor (the same oracle over two near-equivalent
   ArtSeek variants). Picking the max of two noisy arms gains even when they
   are equivalent; the floor removes that.
3. Knowledge-base coverage: share of a benchmark's paintings whose title
   matches a page of the retrieval index (index_manifest.parquet, from
   build_index_manifest.py).

Writes results/coverage_and_routing.json.

Usage:
    python rebuttal_experiments/cross_benchmark/coverage_and_routing.py
"""

import collections
import json
import re
import statistics
import unicodedata
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
MANIFEST = HERE / "data" / "index_manifest.parquet"
BENCHMARKS = [("artpedia_vqa", "ArtPedia"), ("licn_heldout", "LICNHeldOut"),
              ("aqua", "AQUA"), ("artquest", "ArtQuest"),
              ("artcurate_aic", "ArtCurate-AIC")]
JUDGE = "microsoft/phi-4"
# Near-equivalent pairs for the noise floor, first available is used.
CONTROL_PAIRS = [("full_classify", "full_classify_noartist"),
                 ("full_classify", "full_classify_notags"),
                 ("full_classify", "full_noclassify")]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


def scores(exp: str) -> dict:
    """{variant: {query_id: score}} for the phi-4 judge."""
    p = EXPERIMENTS / exp / "results" / "jsonl" / "judge_answer.jsonl"
    out = collections.defaultdict(dict)
    if not p.exists():
        return out
    for line in p.open():
        r = json.loads(line)
        if r.get("judge_model") != JUDGE:
            continue
        s = r.get("score", r.get("correctness"))
        if s is not None:
            out[r["variant"]][r["query_id"]] = s
    return out


def max_gain(a: dict, b: dict) -> float | None:
    """Mean of the per-item max, minus the better arm's own mean."""
    q = sorted(set(a) & set(b))
    if len(q) < 100:
        return None
    ma, mb = statistics.mean(a[k] for k in q), statistics.mean(b[k] for k in q)
    return statistics.mean(max(a[k], b[k]) for k in q) - max(ma, mb)


@click.command()
@click.option("--manifest", default=str(MANIFEST), show_default=True)
def main(manifest: str):
    import pyarrow.parquet as pq

    frag = collections.Counter(
        pq.read_table(manifest, columns=["title"])["title"].to_pylist())
    by_norm = collections.Counter()
    for t, n in frag.items():
        by_norm[norm(t)] += n

    rows = []
    for exp, name in BENCHMARKS:
        sc = scores(exp)
        base, full = sc.get("base", {}), sc.get("full_classify", {})
        q = sorted(set(base) & set(full))
        if not q:
            print(f"{name}: no judged pair, skipped")
            continue

        helped = sum(1 for k in q if full[k] > base[k])
        hurt = sum(1 for k in q if full[k] < base[k])
        m_base = statistics.mean(base[k] for k in q)
        m_full = statistics.mean(full[k] for k in q)

        oracle = max_gain(base, full)
        floor, pair = None, None
        for a, b in CONTROL_PAIRS:
            if a in sc and b in sc:
                floor = max_gain(sc[a], sc[b])
                if floor is not None:
                    pair = f"{a}|{b}"
                    break

        runs = EXPERIMENTS / exp / "results" / "jsonl" / "runs.jsonl"
        titles = {json.loads(l)["title"] for l in runs.open()
                  if json.loads(l).get("title")} if runs.exists() else set()
        covered = [by_norm.get(norm(t), 0) for t in titles]
        covered = [c for c in covered if c > 0]
        cov = 100 * len(covered) / max(len(titles), 1)

        rows.append({
            "benchmark": name, "n": len(q),
            "base": m_base, "full": m_full, "delta": m_full - m_base,
            "helped_pct": 100 * helped / len(q), "hurt_pct": 100 * hurt / len(q),
            "ratio": helped / hurt if hurt else float("inf"),
            "oracle_gain": oracle, "noise_floor": floor,
            "routable": (oracle - floor) if (oracle is not None and floor is not None) else None,
            "control_pair": pair,
            "paintings": len(titles), "in_index": len(covered), "coverage_pct": cov,
        })

    print(f"\n{'benchmark':15}{'base':>7}{'full':>7}{'Δ':>8}"
          f"{'helped':>8}{'hurt':>7}{'ratio':>7}{'routable':>10}{'KB cov':>8}")
    for r in rows:
        rt = f"{r['routable']:+.3f}" if r["routable"] is not None else "n/a"
        print(f"{r['benchmark']:15}{r['base']:7.3f}{r['full']:7.3f}{r['delta']:+8.3f}"
              f"{r['helped_pct']:7.1f}%{r['hurt_pct']:6.1f}%{r['ratio']:7.2f}"
              f"{rt:>10}{r['coverage_pct']:7.1f}%")

    out = HERE / "results" / "coverage_and_routing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}")

    # Coverage range of the benchmarks with and without a gain.
    gain = [r for r in rows if r["delta"] > 0.05]
    null = [r for r in rows if r["delta"] <= 0.05]
    if gain and null:
        print(f"\nbenchmarks where retrieval gains  : coverage "
              f"{min(r['coverage_pct'] for r in gain):.0f}-"
              f"{max(r['coverage_pct'] for r in gain):.0f}%")
        print(f"benchmarks where it does not      : coverage "
              f"{min(r['coverage_pct'] for r in null):.0f}-"
              f"{max(r['coverage_pct'] for r in null):.0f}%")


if __name__ == "__main__":
    main()
