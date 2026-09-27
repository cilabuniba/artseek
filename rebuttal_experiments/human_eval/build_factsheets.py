"""Reference material ("fact sheets") for the human-evaluation paintings.

Source material only, collected without any language model:

    Wikipedia tier   the English Wikipedia summary and Wikidata claims
    gallery tier     the fields of the ICCD scheda (verbatim, Italian) and the
                     English label of the ArCo record

`key_facts` are added by derive_key_facts.py and the Italian fields are
translated by translate_factsheets.py. Writes data/factsheets.json.

Usage:
    python rebuttal_experiments/human_eval/build_factsheets.py
"""

import click
import json
import re
import time
from pathlib import Path

import requests
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
GALLERY = HERE / "data" / "gallery"   # ICCD schede of the selected works
HDR = {"User-Agent": "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"}
WDQS = "https://query.wikidata.org/sparql"
ARCO = "https://dati.cultura.gov.it/sparql"

# The ICCD fields worth carrying into a fact sheet, with plain-English glosses.
ICCD_FIELDS = {
    "OGTD": "object type",
    "SGTI": "subject / title",
    "AUTN": "author (as catalogued)",
    "AUTA": "author dates",
    "AUTM": "basis of attribution",
    "DTZG": "century",
    "DTZS": "part of century",
    "DTSI": "dated from",
    "DTSF": "dated to",
    "DTM": "basis of dating",
    "MTC": "materials and technique",
    "MISA": "height (cm)",
    "MISL": "width (cm)",
    "DESI": "Iconclass code",
    "DESS": "subject notes",
    "NSC": "critical note",
    "LDCM": "collection",
    "STCC": "condition",
}


def wikipedia_extract(title: str) -> dict:
    """Lead section of the English article, verbatim."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + title.replace(" ", "_"),
            headers=HDR, timeout=60)
        if r.status_code != 200:
            return {}
        d = r.json()
        return {"wikipedia_title": d.get("title"),
                "wikipedia_extract": d.get("extract"),
                "wikipedia_url": (d.get("content_urls", {})
                                  .get("desktop", {}).get("page"))}
    except Exception:
        return {}


def wikidata_claims(qid: str) -> dict:
    q = f"""
SELECT ?depictsLabel ?genreLabel ?movementLabel ?materialLabel ?collectionLabel
       ?inception ?countryLabel WHERE {{
  OPTIONAL {{ wd:{qid} wdt:P180 ?depicts }}
  OPTIONAL {{ wd:{qid} wdt:P136 ?genre }}
  OPTIONAL {{ wd:{qid} wdt:P135 ?movement }}
  OPTIONAL {{ wd:{qid} wdt:P186 ?material }}
  OPTIONAL {{ wd:{qid} wdt:P195 ?collection }}
  OPTIONAL {{ wd:{qid} wdt:P571 ?inception }}
  OPTIONAL {{ wd:{qid} wdt:P495 ?country }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} LIMIT 40"""
    try:
        r = requests.get(WDQS, params={"query": q, "format": "json"},
                         headers={**HDR, "Accept": "application/sparql-results+json"},
                         timeout=120)
        r.raise_for_status()
        out = {}
        for b in r.json()["results"]["bindings"]:
            for k, v in b.items():
                key = k.replace("Label", "")
                out.setdefault(key, set()).add(v["value"])
        return {k: sorted(v) for k, v in out.items()}
    except Exception:
        return {}


def arco_label(nctn: str) -> str | None:
    q = f"""SELECT ?l WHERE {{
  <https://w3id.org/arco/resource/HistoricOrArtisticProperty/16{nctn}>
    <http://www.w3.org/2000/01/rdf-schema#label> ?l .
  FILTER(lang(?l)='en') }} LIMIT 1"""
    try:
        r = requests.get(ARCO, params={"query": q},
                         headers={**HDR, "Accept": "application/sparql-results+json"},
                         timeout=120)
        r.raise_for_status()
        b = r.json()["results"]["bindings"]
        return b[0]["l"]["value"] if b else None
    except Exception:
        return None


def parse_scheda(path: Path) -> dict:
    """Pull the catalogued fields out of an ICCD PDF, verbatim."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    text = re.sub(r"Pagina \d+ di \d+", "", text)
    flat = re.sub(r"\s*\n\s*", " ", text)
    out = {}
    for code, gloss in ICCD_FIELDS.items():
        # "NSC - Notizie storico-critiche <value>" up to the next CODE - marker
        m = re.search(rf"\b{code}\b\s*-?\s*[A-Za-zàèéìòù'\s]*?\s{{1,}}(.+?)(?=\s[A-Z]{{2,4}}\s*-\s|$)",
                      flat)
        if m:
            v = m.group(1).strip(" -")
            # Field labels wrap across lines ("AUTN - Nome scelto <value>"):
            # strip the known continuations.
            v = re.sub(r"^(scelto|anagrafici|dell'attribuzione|e tecnica|"
                       r"cronologia|storico-critiche|di conservazione|"
                       r"sul soggetto|raccolta|generica|specifica)\s+",
                       "", v, flags=re.I).strip()
            if v and not re.fullmatch(r"[A-Z]{2,4}", v):
                out[code] = {"gloss": gloss, "value": v[:1200]}
    return out


@click.command()
@click.option("--items-file", default="items.json", show_default=True,
              help="File under data/ listing the items to build sheets for.")
@click.option("--out-file", default="factsheets.json", show_default=True)
def main(items_file: str, out_file: str):
    items = json.loads((HERE / "data" / items_file).read_text())
    sheets = []
    for it in items:
        s = {k: it[k] for k in ("id", "tier", "title", "artist", "image")
             if k in it}
        s["year"] = it.get("year") or it.get("century")
        if it["tier"] != "gallery":
            s |= wikipedia_extract(it.get("wikipedia_title") or it["title"])
            s |= {"wikidata": wikidata_claims(it["id"])}
            time.sleep(1.2)
        else:
            s["iccd"] = parse_scheda(GALLERY / it["scheda"])
            s["arco_uri"] = it.get("arco_uri")
            s["arco_label_en"] = arco_label(it["nctn"]) if it.get("nctn") else None
            s["room"] = it.get("room")
            s["subject_category"] = it.get("subject")
            time.sleep(1.0)
        # Filled by derive_key_facts.py from catalogue data (never by a model).
        s["key_facts"] = []
        s["traps"] = []
        s["iconography_notes"] = ""
        s["validated_by_expert"] = False
        sheets.append(s)
        print(f"  {it['tier']:8s} {it['title'][:44]:46s} "
              f"{'wiki' if s.get('wikipedia_extract') else ''}"
              f"{'iccd:' + str(len(s.get('iccd', {}))) if it['tier']=='gallery' else ''}")

    out = HERE / "data" / out_file
    out.write_text(json.dumps(sheets, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(sheets)} sheets -> {out}")
    print("next: derive_key_facts.py, then translate_factsheets.py")


if __name__ == "__main__":
    main()
