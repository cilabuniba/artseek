"""Fill the `key_facts` of each fact sheet from catalogue data, without a model.

  Wikipedia tier   Wikidata claims (P180 depicts, P921 main subject, P136 genre,
                   P571 inception, P195 collection) and the first sentence of
                   the English Wikipedia summary.
  gallery tier     ICCD fields written by the cataloguers: SGTI subject, DESS
                   subject notes (naming the figures), DESI Iconclass code,
                   DTZG/DTSI/DTSF dating.

Fields are copied, not paraphrased; a missing field gives no line. `traps` are
left empty. Updates data/factsheets.json in place.

Usage:
    python rebuttal_experiments/human_eval/derive_key_facts.py
"""

import json
import re
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

# Wikidata claim -> the label a rater sees. Ordered most to least decisive:
# what is depicted is the thing the SUBJECT axis scores.
WD_FIELDS = [
    ("depicts", "Depicts"),
    ("main subject", "Main subject"),
    ("genre", "Genre"),
    ("movement", "Movement"),
    ("inception", "Date"),
    ("collection", "Collection"),
]

# ICCD code -> label. DESS names the figures, DESI carries the Iconclass code;
# between them they are the gallery tier's SUBJECT and FIGURES ground truth.
ICCD_FIELDS = [
    ("SGTI", "Subject"),
    ("SGTD", "Subject"),
    ("DESS", "Figures / subject notes"),
    ("DESI", "Iconclass"),
    ("DTZG", "Century"),
    ("MTC", "Materials"),
]


def clean_year(v: str) -> str:
    """Wikidata inception arrives as a full timestamp; a rater wants the year."""
    m = re.match(r"(-?\d{3,4})", str(v))
    return m.group(1) if m else str(v)


def first_sentence(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    m = re.match(r"(.+?[.!?])(\s|$)", text, re.S)
    s = (m.group(1) if m else text).strip()
    return s if len(s) > 20 else None


@click.command()
@click.option("--sheets", default=str(HERE / "data" / "factsheets.json"),
              show_default=True)
def main(sheets: str):
    path = Path(sheets)
    data = json.loads(path.read_text())

    for s in data:
        facts: list[str] = []

        if s["tier"] == "gallery":
            iccd = s.get("iccd") or {}
            seen_labels = set()
            for code, label in ICCD_FIELDS:
                f = iccd.get(code)
                if not f:
                    continue
                val = (f.get("value_en") or f.get("value") or "").strip()
                # SGTI and SGTD share a label; keep the first that exists so a
                # sheet does not show "Subject:" twice with different wording.
                if not val or label in seen_labels:
                    continue
                seen_labels.add(label)
                facts.append(f"{label}: {val}")
            # Dating range, when the scheda gives one, is more use to a rater
            # than the century alone.
            lo = (iccd.get("DTSI") or {}).get("value")
            hi = (iccd.get("DTSF") or {}).get("value")
            if lo and hi:
                facts.append(f"Dated: {lo}-{hi}")
        else:
            wd = s.get("wikidata") or {}
            for key, label in WD_FIELDS:
                vals = wd.get(key)
                if not vals:
                    continue
                if key == "inception":
                    vals = [clean_year(v) for v in vals]
                facts.append(f"{label}: {', '.join(str(v) for v in vals)}")
            fs = first_sentence(s.get("wikipedia_extract"))
            if fs:
                facts.append(fs)

        s["key_facts"] = facts
        # Not derivable from catalogue data.
        s["traps"] = []
        s["validated_by_expert"] = False
        s["key_facts_source"] = ("ICCD scheda" if s["tier"] == "gallery"
                                 else "Wikidata claims + Wikipedia extract")

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote key_facts for {len(data)} sheets -> {path}\n")
    for s in data:
        print(f"  {s['tier'][:12]:13} {s['title'][:38]:40} "
              f"{len(s['key_facts'])} facts")
        for f in s["key_facts"][:3]:
            print(f"       - {f[:88]}")
    empty = [s["title"] for s in data if not s["key_facts"]]
    if empty:
        print(f"\n!! {len(empty)} sheets have NO key facts: {empty}")


if __name__ == "__main__":
    main()
