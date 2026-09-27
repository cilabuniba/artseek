"""Select the Wikipedia paintings of the human evaluation (10 items).

A painting is eligible when its English Wikipedia article contributed at least
MIN_FRAGMENTS fragments to the retrieval index and at most MAX_SITELINKS
Wikipedia language editions have an article about it (documented in the
corpus, but not famous). Paintings whose artist is among LICN's 2,501 classes
are preferred. From a seeded shuffle, at most two paintings per century and
one per artist are taken.

Reads data/candidate_pool.json (build_candidate_pool.py + enrich_pool.py) and
writes the Wikipedia tier of data/items.json, keeping any gallery items already
there (see select_gallery.py).

Usage:
    python rebuttal_experiments/human_eval/select_items.py
"""

import collections
import json
import random
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

SEED = 20260827
MIN_FRAGMENTS = 8
MAX_SITELINKS = 6
N_TIER_A = 10


def century(year: str | None) -> str:
    try:
        return str((int(year) - 1) // 100 + 1)
    except (TypeError, ValueError):
        return "unknown"


MAX_PER_CENTURY = 2


def pick_diverse(pool: list[dict], n: int) -> list[dict]:
    """At most MAX_PER_CENTURY per century and one work per artist."""
    picked, per_century, seen_artist = [], collections.Counter(), set()
    for c in pool:
        cen = century(c.get("inception"))
        art = (c.get("artist") or "").lower()
        if per_century[cen] >= MAX_PER_CENTURY or (art and art in seen_artist):
            continue
        per_century[cen] += 1
        if art:
            seen_artist.add(art)
        picked.append(c)
        if len(picked) == n:
            break
    return picked


@click.command()
@click.option("--pool", default=str(HERE / "data" / "candidate_pool.json"),
              show_default=True)
@click.option("--out", default=str(HERE / "data" / "items.json"),
              show_default=True)
def main(pool: str, out: str):
    cands = json.loads(Path(pool).read_text())
    old = json.loads(Path(out).read_text()) if Path(out).exists() else []

    band = [c for c in cands
            if c["n_fragments"] >= MIN_FRAGMENTS
            and c["sitelinks"] <= MAX_SITELINKS]
    in_licn = [c for c in band if c.get("artist_in_licn")]
    print(f"{len(cands)} candidates -> {len(band)} in band "
          f"(>={MIN_FRAGMENTS} frags, <={MAX_SITELINKS} sitelinks), "
          f"{len(in_licn)} of them inside LICN's label space")

    rng = random.Random(SEED)
    rng.shuffle(in_licn)
    rng.shuffle(band)

    picked = pick_diverse(in_licn, N_TIER_A)
    if len(picked) < N_TIER_A:
        # Fall back to artists outside LICN's label space.
        print(f"  only {len(picked)} in-label-space items survive the diversity "
              f"constraint; filling {N_TIER_A - len(picked)} from the wider band")
        chosen = {c["qid"] for c in picked}
        picked += pick_diverse([c for c in band if c["qid"] not in chosen],
                               N_TIER_A - len(picked))
    if len(picked) < N_TIER_A:
        raise SystemExit(
            f"Only {len(picked)} candidates survive; loosen MIN_FRAGMENTS or the "
            "diversity constraint.")

    items = [{
        "tier": "indexed_lowprofile",
        "id": c["qid"],
        "title": c["wikipedia_title"],
        "wikipedia_title": c["wikipedia_title"],
        "artist": c.get("artist"),
        "artist_in_licn": bool(c.get("artist_in_licn")),
        "year": c.get("inception"),
        "movements": c.get("movements") or [],
        "genres": c.get("genres") or [],
        "sitelinks": c["sitelinks"],
        "n_fragments": c["n_fragments"],
        "image_url": c["image_url"],
        "image": f"images/{c['qid']}.jpg",
        "source": "wikidata",
    } for c in picked]

    items += [dict(i) for i in old if i["tier"] == "gallery"]

    Path(out).write_text(json.dumps(items, indent=2, ensure_ascii=False))
    for t in ("indexed_lowprofile", "gallery"):
        sel = [i for i in items if i["tier"] == t]
        print(f"\n{t} ({len(sel)}):")
        for i in sel:
            note = ""
            if t == "indexed_lowprofile":
                note = (f"  {i['n_fragments']:3d} frags, {i['sitelinks']} sitelinks, "
                        f"LICN={'y' if i['artist_in_licn'] else 'n'}, "
                        f"{i.get('artist') or '?'}")
            print(f"   {i['title'][:52]:54}{note}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
