"""Select the gallery paintings of the human evaluation.

Galleria Nazionale della Puglia "Girolamo e Rosaria Devanna" (Bitonto). The
works have no Wikipedia article and their painters are outside LICN's label
space. A painting is eligible when:

  * it is a painting on display (not in storage), with an ICCD scheda and a
    photograph;
  * its catalogued subject (`Soggetto`) maps, through a fixed lexicon of
    standard iconographic subjects (SUBJECT_LEXICON), to an English Wikipedia
    page with at least MIN_SUBJECT_FRAGMENTS fragments in the retrieval index.
    Genre pages ("Still life", "Portrait", ...) do not count.

Five works are then drawn with a seeded shuffle, one per subject category
first, one per artist and one per subject page. The tier tests whether the
depicted subject can be read, not whether the painting can be identified.

--gallery-dir must hold the gallery catalogue (inventory.xlsx,
schede_iccd_*/ and images/), which is not distributed; data/gallery/ contains
the records and photographs of the selected works only.

Usage:
    python rebuttal_experiments/human_eval/select_gallery.py --gallery-dir <catalogue>
"""

import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import click
import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent

SEED = 20260827
MIN_SUBJECT_FRAGMENTS = 8
N_GALLERY = 5

# Subject categories, used to spread the selection.
CATEGORIES = [
    ("sacred_event", r"annunciazione|nativit|battesimo|flagellazione|"
                     r"adorazione|crocifissione|deposizione|resurrezione|"
                     r"incoronazione|assunzione|transito|pentecoste|"
                     r"circoncisione|presentazione|fuga in egitto|strage|"
                     r"estasi|martirio|ultima cena|cenacolo"),
    ("sacred_figure", r"madonna|crist|sant|maddalena|piet|redentore|"
                      r"immacolata|angelo|arcangelo"),
    ("mythological", r"pan e siringa|diana|atteone|ninf|mitolog|venere|apollo|"
                     r"orfeo|bacco|marsia|europa|dafne"),
    ("still_life", r"natura morta|ghirlanda|fiori|frutta|vanitas"),
    ("landscape", r"paesaggio|sottobosco|veduta|marina|incendio"),
    ("modern", r"composizione|ritmi|dinamic|uomini e cavallo|nudo|astratt"),
]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


# Italian catalogue subject -> English Wikipedia page of the iconographic
# subject. Hand-written from the standard subject vocabulary (full-text search
# returns pages that merely mention a phrase). The first matching pattern
# wins, so specific subjects precede general ones.
SUBJECT_LEXICON = [
    (r"riposo.*fuga in egitto", "Rest on the Flight into Egypt"),
    (r"fuga in egitto", "Flight into Egypt"),
    (r"adorazione dei magi", "Adoration of the Magi"),
    (r"adorazione dei pastori", "Adoration of the Shepherds"),
    (r"annunciazione", "Annunciation"),
    (r"nativit|presepe", "Nativity of Jesus in art"),
    (r"flagellazione", "Flagellation of Christ"),
    (r"crocifiss|cristo in croce", "Crucifixion of Jesus"),
    (r"deposizione|piet\b", "Pietà"),
    (r"resurrezione|risurrezione", "Resurrection of Jesus"),
    (r"incoronazione di maria|maria regina", "Coronation of the Virgin"),
    (r"assunzione|dormitio|transito.*madonna", "Assumption of Mary"),
    (r"strage degli innocenti", "Massacre of the Innocents"),
    (r"ultima cena|cenacolo", "The Last Supper (Leonardo)"),
    (r"stimmate", "Stigmata"),
    (r"estasi.*teresa", "Ecstasy of Saint Teresa"),
    (r"strage degli innocenti", "Massacre of the Innocents"),
    (r"sacra famiglia", "Holy Family"),
    (r"madonna con bambino|madonna|vergine", "Madonna (art)"),
    (r"apollo e marsia|marsia", "Flaying of Marsyas (Titian)"),
    (r"pan e siringa", "Pan (god)"),
    (r"diana.*endimione|endimione", "Endymion (mythology)"),
    (r"diana|atteone", "Diana (mythology)"),
    (r"venere", "Venus (mythology)"),
    (r"bacco", "Dionysus"),
    (r"orfeo", "Orpheus"),
    (r"vanitas", "Vanitas"),
    # Genre pages: listed for completeness, they do not make a work eligible.
    (r"natura morta|fiori|frutta|ghirlanda", "Still life"),
    (r"paesaggio|veduta|sottobosco", "Landscape painting"),
    (r"marina", "Marine art"),
    (r"autoritratto", "Self-portrait"),
    (r"ritratto", "Portrait"),
]


# Genre pages: they clear the fragment threshold but cannot serve as subject
# evidence, so they do not make a work eligible.
GENERIC_TARGETS = {"Still life", "Landscape painting", "Marine art",
                   "Self-portrait", "Portrait"}


def lexicon_title(subject: str) -> str | None:
    low = norm(subject)
    for pat, title in SUBJECT_LEXICON:
        if re.search(pat, low):
            return title
    return None


def it_to_en_title(subject: str, cache: dict) -> str | None:
    """Italian subject phrase -> English Wikipedia page (lexicon only)."""
    return lexicon_title(subject)

def load_inventory(gallery: Path) -> pd.DataFrame:
    raw = pd.read_excel(gallery / "inventory.xlsx", header=None)
    hdr = [str(v) for v in raw.iloc[2].tolist()]
    x = raw.iloc[3:].copy()
    x.columns = hdr
    x = x.dropna(how="all")
    x["inv"] = x["N. Inventario"].astype(str).str.strip()
    return x


def scheda_paths(gallery: Path) -> dict:
    """Inventory number -> ICCD scheda (PDF)."""
    out = {}
    for sub in ("schede_iccd_quadri_e_sculture", "schede_iccd_disegni",
                "schede_iccd_incisioni"):
        for p in (gallery / sub).glob("*.pdf"):
            out.setdefault(p.stem, p)
    return out


def image_paths(gallery: Path, inventory: pd.DataFrame) -> dict:
    """Inventory number -> photographs of that work.

    File names start with the inventory number, and inventory numbers have
    different lengths ("010" and "010.010"), so the longest matching number
    wins and the match must end at ".", " " or "_".
    """
    known = sorted((str(v).strip() for v in inventory["inv"]), key=len, reverse=True)
    out = defaultdict(list)
    for p in (gallery / "images").rglob("*"):
        if not (p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif"}):
            continue
        stem = p.name[: -len(p.suffix)]
        for inv in known:
            if stem == inv or stem[len(inv):len(inv) + 1] in (".", " ", "_") \
                    and stem.startswith(inv):
                out[inv].append(p)
                break
    return out


def scheda_nctn(path: Path) -> str | None:
    """The NCTN code (catalogue number) printed in an ICCD scheda."""
    try:
        t = "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)
    except Exception:
        return None
    m = re.search(r"generale\s*(\d{6,})", t)
    return m.group(1) if m else None



@click.command()
@click.option("--gallery-dir", required=True, type=click.Path(exists=True),
              help="Gallery catalogue: inventory.xlsx, schede_iccd_*/, images/.")
@click.option("--items-file", default="items.json", show_default=True,
              help="Item file (under data/) whose gallery tier is rewritten.")
@click.option("--manifest", default=str(HERE.parent / "cross_benchmark" / "data" / "index_manifest.parquet"),
              show_default=True)
def main(gallery_dir: str, items_file: str, manifest: str):
    import collections

    import pyarrow.parquet as pq

    gallery_dir = Path(gallery_dir)

    frag = collections.Counter(
        pq.read_table(manifest, columns=["title"])["title"].to_pylist())
    print(f"index manifest: {len(frag)} pages")

    inv = load_inventory(gallery_dir)
    schede, imgs = scheda_paths(gallery_dir), image_paths(gallery_dir, inv)
    on_show = inv[~inv["Collocazione"].astype(str).str.strip()
                  .isin(["Cassaforte", "Deposito"])]
    paintings = on_show[on_show["Tipologia"].astype(str).str.strip() == "Dipinto"]
    print(f"{len(paintings)} paintings on display")

    cands = []
    for _, r in paintings.iterrows():
        if r["inv"] not in schede or r["inv"] not in imgs:
            continue
        subject = str(r.get("Soggetto") or "").strip()
        if not subject:
            continue
        en = it_to_en_title(subject, {})
        n = frag.get(en, 0) if en else 0
        cat = next((c for c, pat in CATEGORIES
                    if re.search(pat, subject.lower())), None)
        cands.append({"row": r, "subject": subject, "en_title": en,
                      "n_fragments": n, "category": cat})

    covered = [c for c in cands
               if c["n_fragments"] >= MIN_SUBJECT_FRAGMENTS
               and c["en_title"] not in GENERIC_TARGETS]
    n_generic = sum(1 for c in cands if c["en_title"] in GENERIC_TARGETS
                    and c["n_fragments"] >= MIN_SUBJECT_FRAGMENTS)
    print(f"({n_generic} excluded: subject resolves only to a genre page)")
    # Stable order before the seeded shuffle.
    covered.sort(key=lambda c: str(c["row"]["inv"]))
    print(f"{len(covered)}/{len(cands)} have a subject page with "
          f">={MIN_SUBJECT_FRAGMENTS} fragments in the index\n")
    for c in sorted(covered, key=lambda x: -x["n_fragments"])[:25]:
        print(f"  {c['n_fragments']:4d}  {c['subject'][:44]:46} -> {c['en_title']}")

    rng = random.Random(SEED)
    rng.shuffle(covered)
    # Spread across categories first, then fill; one work per artist throughout.
    by_cat = defaultdict(list)
    for c in covered:
        by_cat[c["category"]].append(c)
    picked, used_artists, used_subjects = [], set(), set()

    def take(c) -> bool:
        """One work per artist and one per subject page."""
        artist = str(c["row"].get("Autore") or "").strip()
        if (artist and artist in used_artists) or c["en_title"] in used_subjects:
            return False
        picked.append(c)
        used_artists.add(artist)
        used_subjects.add(c["en_title"])
        return True

    for cat in [c for c, _ in CATEGORIES] + [None]:
        for c in by_cat.get(cat, []):
            if take(c):
                break
        if len(picked) == N_GALLERY:
            break
    for c in covered:
        if len(picked) >= N_GALLERY:
            break
        take(c)

    if len(picked) < N_GALLERY:
        raise SystemExit(
            f"only {len(picked)} eligible gallery works; lower "
            "MIN_SUBJECT_FRAGMENTS.")

    gallery = []
    for c in picked[:N_GALLERY]:
        r = c["row"]
        nctn = scheda_nctn(schede[r["inv"]])
        gallery.append({
            "tier": "gallery",
            "id": f"devanna-{r['inv']}",
            "inv": r["inv"],
            "title": str(r["Soggetto"]),
            "artist": str(r["Autore"]),
            "year": str(r.get("Datazione") or ""),
            "century": str(r.get("Secolo") or ""),
            "material": str(r.get("Materia") or ""),
            "subject": c["category"],
            "subject_wikipedia_title": c["en_title"],
            "subject_n_fragments": c["n_fragments"],
            "room": str(r["Collocazione"]).strip(),
            "nctn": nctn,
            "arco_uri": (f"https://w3id.org/arco/resource/"
                         f"HistoricOrArtisticProperty/16{nctn}" if nctn else None),
            "scheda": str(schede[r["inv"]].relative_to(gallery_dir)),
            "image_local": str(sorted(imgs[r["inv"]])[0].relative_to(gallery_dir)),
            "source": "devanna",
        })

    path = HERE / "data" / items_file
    items = json.loads(path.read_text())
    items = [i for i in items if i["tier"] != "gallery"] + gallery
    path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    print(f"\nselected gallery tier ({len(gallery)}):")
    for g in gallery:
        print(f"  [{g['subject']}] {g['inv']} {g['title'][:40]:42} "
              f"subject={g['subject_wikipedia_title']} "
              f"({g['subject_n_fragments']} frags)")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
