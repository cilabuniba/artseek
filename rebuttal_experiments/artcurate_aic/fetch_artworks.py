"""Fetch the ArtCurate-AIC corpus from the Art Institute of Chicago API.

Public-domain paintings with a curatorial `description` of at least
--min-desc characters. For each painting we keep the description and, as a
second independent text for the multihop questions, the provenance or
exhibition history (`did_you_know`). Images are 843px IIIF derivatives saved to
data/external/artcurate_aic/images/{id}.jpg. The AIC data is CC0.

Writes data/artcurate_corpus.json.

Usage:
    python rebuttal_experiments/artcurate_aic/fetch_artworks.py --n 260
"""

import argparse
import html
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
IMAGES = REPO / "data" / "external" / "artcurate_aic" / "images"
API = "https://api.artic.edu/api/v1/artworks/search"
IIIF = "https://www.artic.edu/iiif/2/{}/full/843,/0/default.jpg"
HDR = {"AIC-User-Agent": "ArtSeek (https://github.com/cilabuniba/artseek)"}
SEED = 20260828

FIELDS = ",".join([
    "id", "title", "artist_title", "artist_display", "date_display",
    "medium_display", "classification_title", "classification_titles",
    "style_title", "place_of_origin", "description", "short_description",
    "provenance_text", "exhibition_history", "image_id", "is_public_domain",
    "credit_line",
])


def strip_html(s: str | None) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    ap.add_argument("--min-desc", type=int, default=400,
                    help="Minimum description length; short blurbs cannot "
                         "support six interpretive questions.")
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)

    params = {
        "query[bool][must][0][term][is_public_domain]": "true",
        "query[bool][must][1][match][classification_titles]": "painting",
        "query[bool][must][2][exists][field]": "description",
        "fields": FIELDS, "limit": 100,
    }
    rows, page = [], 1
    while len(rows) < 900 and page <= 10:
        r = requests.get(API, params={**params, "page": page}, headers=HDR, timeout=90)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            break
        rows.extend(data)
        print(f"  page {page}: +{len(data)} (total {len(rows)})", flush=True)
        page += 1
        time.sleep(0.4)

    corpus = []
    for x in rows:
        desc = strip_html(x.get("description"))
        if len(desc) < a.min_desc or not x.get("image_id"):
            continue
        second = strip_html(x.get("provenance_text")) or strip_html(x.get("exhibition_history"))
        if not second:
            continue
        corpus.append({
            "id": str(x["id"]),
            "title": x.get("title"),
            "artist": x.get("artist_display") or x.get("artist_title"),
            "culture": [x["place_of_origin"]] if x.get("place_of_origin") else [],
            "creation_date": x.get("date_display"),
            "type": x.get("classification_title"),
            "technique": x.get("medium_display"),
            "tombstone": x.get("credit_line"),
            "description": desc,
            "did_you_know": second,
            "image_url": IIIF.format(x["image_id"]),
            "record_url": f"https://www.artic.edu/artworks/{x['id']}",
            "source": "Art Institute of Chicago (public domain / CC0)",
        })
    print(f"\n{len(corpus)} paintings with a >={a.min_desc}-char description "
          f"and an independent second statement")

    import random
    random.Random(SEED).shuffle(corpus)
    corpus = sorted(corpus[:a.n], key=lambda r: int(r["id"]))

    ok = 0
    for i, r in enumerate(corpus, 1):
        dst = IMAGES / f"{r['id']}.jpg"
        if dst.exists():
            ok += 1
            continue
        try:
            im = requests.get(r["image_url"], headers=HDR, timeout=120)
            im.raise_for_status()
            dst.write_bytes(im.content)
            ok += 1
        except Exception as e:
            print(f"  image {r['id']} failed: {e}")
        if i % 25 == 0:
            print(f"  images {i}/{len(corpus)}", flush=True)
        time.sleep(0.25)

    corpus = [r for r in corpus if (IMAGES / f"{r['id']}.jpg").exists()]
    out = HERE / "data" / "artcurate_corpus.json"
    out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(corpus)} paintings ({ok} images on disk) -> {out}")


if __name__ == "__main__":
    main()
