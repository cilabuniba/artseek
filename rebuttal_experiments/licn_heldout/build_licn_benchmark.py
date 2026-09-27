"""Build LICNHeldOut: paintings inside LICN's label space but outside its
training set, with questions that each target one LICN head.

  in distribution   every painting is by one of LICN's 2,501 artist classes
                    (the ArtGraph artists);
  held out          no painting is among the 116,475 ArtGraph training
                    artworks (checked against artgraph_index.json, see
                    build_artgraph_index.py);
  one head each     questions come from Wikidata properties matching a head:

                        artist    P170 creator    -> head `artist`
                        movement  P135 movement   -> head `style`
                        genre     P136 genre      -> head `genre`
                        material  P186 material   -> head `media`

Reference answers are Wikidata statements. Artist QIDs (data/artist_qids.json,
included) and the Wikidata query results (data/wd_paintings.json) are cached.
Wikidata changes over time, so data/licn_vqa.json is the benchmark we used.

Usage:
    python rebuttal_experiments/licn_heldout/build_licn_benchmark.py
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
HERE = Path(__file__).resolve().parent
WDQS = "https://query.wikidata.org/sparql"
HDR = {"User-Agent": "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)",
       "Accept": "application/sparql-results+json"}
SEED = 20260828

QUESTIONS = {
    "artist":   "Who is the artist who created this painting?",
    "movement": "To which artistic movement or style does this painting belong?",
    "genre":    "What is the genre of this painting?",
    "material": "What material or medium was used to make this painting?",
}


def slug_to_name(slug: str) -> str:
    """ArtGraph slugs are lowercase-hyphenated names: camille-pissarro."""
    return " ".join(w.capitalize() for w in slug.split("-"))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def sparql(query: str, tries: int = 4) -> dict:
    for i in range(tries):
        try:
            r = requests.get(WDQS, params={"query": query, "format": "json"},
                             headers=HDR, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def resolve_artists(names: list[str], cache: Path) -> dict:
    """name -> artist QID, via exact English label/alias match."""
    if cache.exists():
        return json.loads(cache.read_text())
    out = {}
    for i in range(0, len(names), 60):
        batch = names[i:i + 60]
        vals = " ".join('"%s"@en' % n.replace('"', '') for n in batch)
        q = f"""
SELECT ?a ?label WHERE {{
  VALUES ?label {{ {vals} }}
  ?a rdfs:label ?label .
  ?a wdt:P106 ?occ . VALUES ?occ {{ wd:Q1028181 wd:Q483501 }}
}}"""
        try:
            res = sparql(q)
        except Exception as e:
            print(f"  artist batch {i} failed: {e}")
            continue
        for b in res["results"]["bindings"]:
            out[b["label"]["value"]] = b["a"]["value"].rsplit("/", 1)[-1]
        print(f"  resolved {len(out)} artists after {min(i+60,len(names))}", flush=True)
        time.sleep(1.0)
    cache.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def fetch_paintings(qids: list[str], cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for i in range(0, len(qids), 25):
        batch = qids[i:i + 25]
        vals = " ".join(f"wd:{q}" for q in batch)
        q = f"""
SELECT ?w ?wLabel ?creator ?creatorLabel ?img ?article
       ?movementLabel ?genreLabel ?materialLabel ?date WHERE {{
  VALUES ?creator {{ {vals} }}
  ?w wdt:P31 wd:Q3305213 ; wdt:P170 ?creator ; wdt:P18 ?img .
  ?article schema:about ?w ; schema:isPartOf <https://en.wikipedia.org/> .
  OPTIONAL {{ ?w wdt:P135 ?movement }}
  OPTIONAL {{ ?w wdt:P136 ?genre }}
  OPTIONAL {{ ?w wdt:P186 ?material }}
  OPTIONAL {{ ?w wdt:P571 ?date }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 4000"""
        try:
            res = sparql(q)
        except Exception as e:
            print(f"  painting batch {i} failed: {e}")
            continue
        for b in res["results"]["bindings"]:
            rows.append({k: b[k]["value"] for k in b})
        print(f"  {len(rows)} rows after {min(i+25,len(qids))}/{len(qids)} artists", flush=True)
        time.sleep(1.2)
    cache.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paintings", type=int, default=250)
    ap.add_argument("--n-artists", type=int, default=700,
                    help="LICN artist classes to search (sampled, seeded).")
    ap.add_argument("--max-per-artist", type=int, default=1,
                    help="Cap paintings per artist so no painter dominates.")
    a = ap.parse_args()

    import random
    rng = random.Random(SEED)

    ag = json.loads((REPO / "data/external/artgraph/artgraph_index.json").read_text())
    trained = {slug: set(map(norm, titles)) for slug, titles in ag["artists"].items()}
    from huggingface_hub import snapshot_download

    licn_data = Path(snapshot_download("cilabuniba/artseek-licn-data", repo_type="dataset"))
    cl = json.loads((licn_data / "class_lookups.json").read_text())
    slugs = sorted(set(cl["artist"].values()))
    print(f"LICN artist classes: {len(slugs)}; ArtGraph training artworks: {ag['n_names']}")

    pick = sorted(rng.sample(slugs, min(a.n_artists, len(slugs))))
    names = [slug_to_name(s) for s in pick]
    by_name = dict(zip(names, pick))

    print("\nresolving artist QIDs...")
    resolved = resolve_artists(names, HERE / "data" / "artist_qids.json")
    print(f"resolved {len(resolved)}/{len(names)}")

    qid_to_slug = {q: by_name[n] for n, q in resolved.items() if n in by_name}
    print("\nfetching paintings...")
    rows = fetch_paintings(sorted(qid_to_slug), HERE / "data" / "wd_paintings.json")
    print(f"{len(rows)} painting rows")

    # group by painting, merge multi-valued properties
    merged = defaultdict(lambda: defaultdict(set))
    meta = {}
    for r in rows:
        w = r["w"].rsplit("/", 1)[-1]
        meta[w] = {"title": r.get("wLabel"), "img": r.get("img"),
                   "artist": r.get("creatorLabel"),
                   "slug": qid_to_slug.get(r["creator"].rsplit("/", 1)[-1]),
                   "article": r.get("article"), "date": r.get("date")}
        for k, key in (("movementLabel", "movement"), ("genreLabel", "genre"),
                       ("materialLabel", "material")):
            if r.get(k) and not r[k].startswith("Q"):
                merged[w][key].add(r[k])

    kept, dropped_seen, dropped_thin = [], 0, 0
    for w, m in meta.items():
        slug = m["slug"]
        if not slug or not m["title"] or m["title"].startswith("Q"):
            continue
        # the exclusion: was this exact painting in LICN's training set?
        if norm(m["title"]) in trained.get(slug, ()):
            dropped_seen += 1
            continue
        props = merged[w]
        if not props:
            dropped_thin += 1
            continue
        rec = {"id": w, "title": m["title"], "artist": m["artist"],
               "artist_slug": slug, "image_url": m["img"],
               "article": m["article"], "year": (m["date"] or "")[:4],
               "artist_question": QUESTIONS["artist"],
               "artist_answer": f"This painting was created by {m['artist']}."}
        for key in ("movement", "genre", "material"):
            if props.get(key):
                vals = sorted(props[key])
                rec[f"{key}_question"] = QUESTIONS[key]
                rec[f"{key}_answer"] = (
                    f"The {key} of this painting is "
                    f"{', '.join(vals)}." if key != "material"
                    else f"This painting is made of {', '.join(vals)}.")
        kept.append(rec)

    print(f"\ncandidates: {len(kept)}   excluded as LICN-trained: {dropped_seen}   "
          f"excluded for no card-addressable property: {dropped_thin}")

    # one painting per artist, so that no painter dominates the sample
    per_artist = defaultdict(list)
    for r in kept:
        per_artist[r["artist_slug"]].append(r)
    pool = []
    for v in per_artist.values():
        rng.shuffle(v)
        pool.extend(v[:a.max_per_artist])
    rng.shuffle(pool)
    sample = sorted(pool[:a.n_paintings], key=lambda r: r["id"])

    out = HERE / "data" / "licn_vqa.json"
    out.write_text(json.dumps(sample, indent=2, ensure_ascii=False))
    nq = sum(1 for r in sample for k in r if k.endswith("_question"))
    print(f"wrote {len(sample)} paintings / {nq} questions -> {out}")
    counts = defaultdict(int)
    for r in sample:
        for k in r:
            if k.endswith("_question"):
                counts[k[:-9]] += 1
    print("per type:", dict(counts))


if __name__ == "__main__":
    main()
