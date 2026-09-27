"""Index the ArtGraph artworks (LICN's training set) by artist.

Every ArtGraph artwork is named `<artist-slug>_<title-slug>.jpg` in the RDF
dump. Writes data/external/artgraph/artgraph_index.json:
{"n_names": <number of artworks>, "artists": {artist-slug: [title-slug, ...]}},
which build_licn_benchmark.py uses to exclude training paintings.

Usage:
    python rebuttal_experiments/licn_heldout/build_artgraph_index.py \\
        --facts data/external/artgraph/artgraph-rdf/artgraph-facts.ttl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "external" / "artgraph" / "artgraph_index.json"
NAME = re.compile(r'artgraph:name "([^"]+\.jpg)"')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", required=True, help="artgraph-facts.ttl from the RDF dump")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    names = set()
    with open(a.facts, encoding="utf-8") as f:
        for line in f:
            m = NAME.search(line)
            if m:
                names.add(m.group(1))

    artists = defaultdict(list)
    for name in sorted(names):
        artist, title = name[: -len(".jpg")].split("_", 1)
        artists[artist].append(title)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"n_names": len(names), "artists": artists}))
    print(f"{len(names)} artworks by {len(artists)} artists -> {a.out}")


if __name__ == "__main__":
    main()
