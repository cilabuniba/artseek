"""Build the ArtQuest evaluation subset.

ArtQuest (Bleidt, Eslami & de Melo, WACV 2024) asks six question types about
SemArt paintings: technique, type, school, timeframe, artist and title. From
the test split (1,069 paintings) we sample 200 paintings with all six types
(seed 20260828), i.e. 1,200 questions. The images are extracted from SemArt
to data/external/semart/images/.

Inputs: data/external/artquest/artquest_test.json (from the ArtQuest archive,
Zenodo record 10453925, CC BY 4.0) and SemArt.zip (by default the copy in the
Hugging Face dataset leo20000306/SemArt).

Usage:
    python rebuttal_experiments/artquest/build_artquest_subset.py --n-paintings 200
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")
ARTQUEST = REPO / "data" / "external" / "artquest" / "artquest_test.json"
IMAGES = REPO / "data" / "external" / "semart" / "images"
OUT = Path(__file__).resolve().parent / "data" / "artquest_vqa.json"
SEED = 20260828

# ArtQuest's six question types.
TYPES = ("artist", "title", "technique", "school", "timeframe", "type")


def semart_zip(path: str | None = None) -> zipfile.ZipFile:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("leo20000306/SemArt", "SemArt.zip", repo_type="dataset")
    return zipfile.ZipFile(path)


def extract_images(zf: zipfile.ZipFile, images: list[str]) -> None:
    """Extract the given SemArt images to IMAGES (skipping existing files)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    members = set(zf.namelist())
    for img in images:
        dst = IMAGES / img
        member = f"SemArt/Images/{img}"
        if not dst.exists() and member in members:
            dst.write_bytes(zf.read(member))


def semart_metadata(zf: zipfile.ZipFile) -> dict:
    """title/artist/date/technique/school per image, from the SemArt CSVs."""
    meta = {}
    for name in zf.namelist():
        if not name.endswith(("semart_train.csv", "semart_val.csv", "semart_test.csv")):
            continue
        import csv
        import io
        text = io.TextIOWrapper(zf.open(name), encoding="latin-1")
        for row in csv.DictReader(text, delimiter="\t"):
            meta[row["IMAGE_FILE"]] = row
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paintings", type=int, default=200,
                    help="Paintings to sample; each contributes 6 questions.")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--semart-zip", default=None,
                    help="SemArt.zip (default: download leo20000306/SemArt from the HF Hub)")
    a = ap.parse_args()

    rows = json.loads(ARTQUEST.read_text())
    by_img = defaultdict(dict)
    for r in rows:
        by_img[r["image"]][r["question_type"]] = r

    zf = semart_zip(a.semart_zip)
    extract_images(zf, sorted(by_img))

    # Keep only paintings with all six question types and an image, so the
    # per-type comparison is balanced.
    complete = sorted(img for img, qs in by_img.items()
                      if len(qs) == len(TYPES) and (IMAGES / img).exists())
    print(f"paintings with all 6 types and a local image: {len(complete)}/{len(by_img)}")

    rng = random.Random(SEED)
    sample = sorted(rng.sample(complete, min(a.n_paintings, len(complete))))

    meta = semart_metadata(zf)
    out = []
    for img in sample:
        qs = by_img[img]
        m = meta.get(img, {})
        rec = {
            "id": img.rsplit(".", 1)[0],
            "semart_image": img,
            "title": m.get("TITLE") or qs["title"]["answer"],
            "artist": m.get("AUTHOR") or qs["artist"]["answer"],
            "year": m.get("DATE"),
            "technique": m.get("TECHNIQUE"),
            "school": m.get("SCHOOL"),
            "img_url": None,
        }
        for t in TYPES:
            rec[f"{t}_question"] = qs[t]["question"]
            rec[f"{t}_answer"] = qs[t]["answer"]
        out.append(rec)

    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {len(out)} paintings x {len(TYPES)} questions "
          f"= {len(out) * len(TYPES)} -> {a.out}")


if __name__ == "__main__":
    main()
