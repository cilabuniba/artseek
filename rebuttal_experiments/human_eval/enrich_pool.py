"""Add creator, movement and genre (Wikidata) to the candidates with at least
--min-fragments fragments, and flag whether the artist is one of LICN's 2,501
classes (`artist_in_licn`, matched by ArtGraph slug, so a lower bound).

Updates data/candidate_pool.json in place.

Usage:
    python rebuttal_experiments/human_eval/enrich_pool.py
"""

import csv
import io
import json
import re
import time
import unicodedata
from pathlib import Path

import click
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")
WDQS = "https://query.wikidata.org/sparql"
UA = "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"

QUERY = """
SELECT ?item ?creatorLabel ?movementLabel ?genreLabel WHERE {
  VALUES ?item { %s }
  OPTIONAL { ?item wdt:P170 ?creator .
             ?creator rdfs:label ?creatorLabel . FILTER(LANG(?creatorLabel)="en") }
  OPTIONAL { ?item wdt:P135 ?movement .
             ?movement rdfs:label ?movementLabel . FILTER(LANG(?movementLabel)="en") }
  OPTIONAL { ?item wdt:P136 ?genre .
             ?genre rdfs:label ?genreLabel . FILTER(LANG(?genreLabel)="en") }
}
"""


def slug(name: str) -> str:
    """ArtGraph-style slug: 'Michiel van Musscher' -> 'michiel-van-musscher'."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.lower()).strip("-")
    return s


def sparql_tsv(query: str, tries: int = 4) -> list[dict]:
    for attempt in range(tries):
        try:
            r = requests.get(WDQS, params={"query": query},
                             headers={"User-Agent": UA,
                                      "Accept": "text/tab-separated-values"},
                             timeout=300)
            if r.status_code == 200:
                text = r.text
                if not text.endswith("\n"):
                    text = text[:text.rfind("\n") + 1]
                rd = csv.DictReader(io.StringIO(text), delimiter="\t")
                return [{k.lstrip("?"): v for k, v in row.items() if k} for row in rd]
            time.sleep(10 * (attempt + 1))
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    return []


def clean(v: str) -> str:
    """Strip TSV literal decoration, including the language tag.

    WDQS returns plain literals as `Jean Metzinger@en`. Leaving the tag on
    slugs the artist to `jean-metzinger-en`, which matches nothing in
    ArtGraph — the failure looked exactly like "no low-profile artist is in
    the label space" and reported 0/451.
    """
    v = (v or "").strip().strip('"').strip("<>")
    return re.sub(r"@[a-z]{2}(-[a-zA-Z]+)?$", "", v).strip()


@click.command()
@click.option("--pool", default=str(HERE / "data" / "candidate_pool.json"),
              show_default=True)
@click.option("--min-fragments", default=8, show_default=True)
@click.option("--batch", default=200, show_default=True,
              help="QIDs per VALUES clause; larger batches start timing out.")
def main(pool: str, min_fragments: int, batch: int):
    cands = json.loads(Path(pool).read_text())
    band = [c for c in cands if c["n_fragments"] >= min_fragments]
    print(f"{len(band)} candidates with >={min_fragments} fragments")

    from huggingface_hub import snapshot_download

    licn_data = Path(snapshot_download("cilabuniba/artseek-licn-data", repo_type="dataset"))
    licn = json.loads((licn_data / "class_lookups.json").read_text())
    licn_artists = {slug(a) for a in licn["artist"].values()}
    print(f"LICN artist label space: {len(licn_artists)} slugs")

    info: dict[str, dict] = {}
    for i in range(0, len(band), batch):
        chunk = band[i:i + batch]
        vals = " ".join(f"wd:{c['qid']}" for c in chunk)
        for row in sparql_tsv(QUERY % vals):
            qid = clean(row.get("item", "")).rsplit("/", 1)[-1]
            if not qid:
                continue
            d = info.setdefault(qid, {"artist": None, "movements": set(),
                                      "genres": set()})
            if clean(row.get("creatorLabel", "")):
                d["artist"] = clean(row["creatorLabel"])
            if clean(row.get("movementLabel", "")):
                d["movements"].add(clean(row["movementLabel"]))
            if clean(row.get("genreLabel", "")):
                d["genres"].add(clean(row["genreLabel"]))
        print(f"  batch {i//batch + 1}: {len(info)} enriched", flush=True)

    n_licn = 0
    for c in band:
        d = info.get(c["qid"], {})
        c["artist"] = d.get("artist")
        c["movements"] = sorted(d.get("movements", []))
        c["genres"] = sorted(d.get("genres", []))
        c["artist_in_licn"] = bool(c["artist"]) and slug(c["artist"]) in licn_artists
        n_licn += c["artist_in_licn"]

    Path(pool).write_text(json.dumps(cands, indent=2, ensure_ascii=False))
    print(f"\n{sum(1 for c in band if c['artist'])}/{len(band)} have a creator")
    print(f"{n_licn}/{len(band)} artists are inside LICN's label space "
          "(lower bound — slug matching is approximate)")
    print(f"-> {pool}")


if __name__ == "__main__":
    main()
