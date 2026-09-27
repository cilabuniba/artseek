"""Candidate paintings for the Wikipedia tier of the human evaluation.

Queries Wikidata for paintings with an English Wikipedia article and at most
--max-sitelinks language editions, and joins them with the index manifest
(cross_benchmark/build_index_manifest.py) to count the fragments each article
contributed to the retrieval index.

The query is split by sitelink count (a single query times out). WDQS may
return a truncated body with HTTP 200, so results are requested as TSV and an
incomplete last line is dropped.

Writes data/candidate_pool.json.

Usage:
    python rebuttal_experiments/human_eval/build_candidate_pool.py
"""

import csv
import io
import json
import time
from pathlib import Path
from urllib.parse import unquote

import click
import requests

HERE = Path(__file__).resolve().parent
WDQS = "https://query.wikidata.org/sparql"
UA = "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"

QUERY = """
SELECT ?item ?article ?image ?sitelinks ?inc WHERE {
  ?item wdt:P31 wd:Q3305213 ; wdt:P18 ?image ; wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks = %(s)d)
  OPTIONAL { ?item wdt:P571 ?inc }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
}
"""


def sparql_tsv(query: str, tries: int = 4) -> list[dict]:
    """Return rows, tolerating the truncated bodies WDQS sends on slow queries."""
    for attempt in range(tries):
        try:
            r = requests.get(
                WDQS, params={"query": query},
                headers={"User-Agent": UA, "Accept": "text/tab-separated-values"},
                timeout=300)
            if r.status_code == 200:
                text = r.text
                # A truncated body ends mid-row. Dropping the last line costs at
                # most one candidate and keeps every complete row before it.
                if not text.endswith("\n"):
                    text = text[:text.rfind("\n") + 1]
                    print("    (truncated response, kept complete rows)")
                rd = csv.DictReader(io.StringIO(text), delimiter="\t")
                # WDQS names TSV columns "?item", "?article", ... — keying on
                # the bare variable name yields None for every row and produces
                # an empty pool that looks exactly like "no candidates exist".
                return [{k.lstrip("?"): v for k, v in row.items() if k}
                        for row in rd]
            wait = 10 * (attempt + 1)
            print(f"    HTTP {r.status_code}, retry in {wait}s")
            time.sleep(wait)
        except requests.RequestException as e:
            print(f"    {type(e).__name__}, retry")
            time.sleep(10 * (attempt + 1))
    return []


def clean(v: str) -> str:
    return (v or "").strip().strip('"').strip("<>")


@click.command()
@click.option("--max-sitelinks", default=6, show_default=True,
              help="Upper bound of the low-profile band to enumerate.")
@click.option("--out", default=str(HERE / "data" / "candidate_pool.json"),
              show_default=True)
@click.option("--manifest", default=str(HERE.parent / "cross_benchmark" / "data" / "index_manifest.parquet"),
              show_default=True)
def main(max_sitelinks: int, out: str, manifest: str):
    import collections

    import pyarrow.parquet as pq

    frag_counts = collections.Counter(
        pq.read_table(manifest, columns=["title"])["title"].to_pylist())
    print(f"index manifest: {len(frag_counts)} distinct pages")

    seen, cands = set(), []
    for s in range(1, max_sitelinks + 1):
        got = sparql_tsv(QUERY % {"s": s})
        added = 0
        for b in got:
            qid = clean(b.get("item", "")).rsplit("/", 1)[-1]
            art = clean(b.get("article", ""))
            if not qid or not art or qid in seen:
                continue
            seen.add(qid)
            # The article URL is percent-encoded with underscores; the manifest
            # stores the human-readable page title. Both transformations are
            # needed or the join silently yields zero everywhere.
            title = unquote(art.rsplit("/", 1)[-1]).replace("_", " ")
            cands.append({
                "qid": qid,
                "wikipedia_title": title,
                "image_url": clean(b.get("image", "")),
                "inception": clean(b.get("inc", ""))[:4] or None,
                "sitelinks": s,
                "n_fragments": frag_counts.get(title, 0),
            })
            added += 1
        n_in = sum(1 for c in cands[-added:] if c["n_fragments"] > 0) if added else 0
        print(f"  sitelinks={s}: {len(got)} rows, {added} new, {n_in} in index",
              flush=True)

    cands.sort(key=lambda c: (-c["n_fragments"], c["sitelinks"]))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cands, indent=2, ensure_ascii=False))

    n_in = sum(1 for c in cands if c["n_fragments"] > 0)
    print(f"\n{len(cands)} low-profile paintings with an en-wiki article")
    print(f"{n_in} ({100*n_in/max(len(cands),1):.1f}%) are in the index")
    for lo in (5, 8, 15):
        print(f"  n_fragments>={lo}: "
              f"{sum(1 for c in cands if c['n_fragments'] >= lo)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
