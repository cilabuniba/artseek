"""Build the AQUA evaluation subset.

AQUA (Garcia et al., "A Dataset and Baselines for Visual Question Answering on
Art", ECCV Workshops 2020) is built on SemArt. Every QA pair carries the
dataset's `need_external_knowledge` flag; here it becomes the question type
(`contextual` if True, `visual` if False).

Of the 1,032 test paintings, 663 have at least one question of each type. We
sample 326 of them (seed 20260828) and take the first question of each type,
giving 652 questions. Title and artist come from the SemArt metadata. The
images are extracted to data/external/semart/images/.

Inputs: AQUA's JSON files (https://github.com/noagarcia/ArtVQA) and SemArt.zip
(by default the copy in the Hugging Face dataset leo20000306/SemArt).

Usage:
    python rebuttal_experiments/aqua/build_aqua_subset.py --aqua-dir <path to ArtVQA/AQUA>
"""

import argparse
import collections
import csv
import json
import random
import zipfile
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
load_dotenv(REPO / ".env")

SEED = 20260828
N_PAINTINGS = 326  # same size as ArtPedia-VQA

IMAGES_DIR = REPO / "data" / "external" / "semart" / "images"
OUT_PATH = Path(__file__).resolve().parent / "data" / "aqua_vqa.json"


def load_semart_metadata(zf: zipfile.ZipFile) -> dict:
    """Author/title/date per image filename, from SemArt's own CSVs."""
    meta = {}
    for name in ("semart_train.csv", "semart_val.csv", "semart_test.csv"):
        path = f"SemArt/{name}"
        if path not in zf.namelist():
            continue
        with zf.open(path) as fh:
            text = fh.read().decode("latin-1").splitlines()
        for row in csv.DictReader(text, delimiter="\t"):
            fn = row.get("IMAGE_FILE")
            if not fn:
                continue
            meta[fn] = {
                "title": (row.get("TITLE") or "").strip(),
                "artist": (row.get("AUTHOR") or "").strip(),
                "date": (row.get("DATE") or "").strip(),
                "technique": (row.get("TECHNIQUE") or "").strip(),
                "type": (row.get("TYPE") or "").strip(),
                "school": (row.get("SCHOOL") or "").strip(),
                "timeframe": (row.get("TIMEFRAME") or "").strip(),
            }
    return meta


def clean(text: str) -> str:
    """AQUA's generated questions carry raw HTML entities and stray spacing."""
    import html
    return " ".join(html.unescape(text or "").split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aqua-dir", required=True,
                    help="Directory containing AQUA's train/val/test.json")
    ap.add_argument("--semart-zip", default=None,
                    help="SemArt.zip (default: download leo20000306/SemArt from the HF Hub)")
    ap.add_argument("--n-paintings", type=int, default=N_PAINTINGS)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    semart_zip = a.semart_zip
    if semart_zip is None:
        from huggingface_hub import hf_hub_download

        semart_zip = hf_hub_download("leo20000306/SemArt", "SemArt.zip", repo_type="dataset")
    print(f"SemArt archive: {semart_zip}")

    qa = json.loads((Path(a.aqua_dir) / f"{a.split}.json").read_text())
    print(f"AQUA {a.split}: {len(qa)} QA pairs")

    by_image = collections.defaultdict(lambda: {"contextual": [], "visual": []})
    for x in qa:
        cls = "contextual" if x["need_external_knowledge"] else "visual"
        by_image[x["image"]][cls].append(x)

    both = sorted(i for i, v in by_image.items() if v["contextual"] and v["visual"])
    print(f"{len(both)} paintings carry at least one question of each class")

    rng = random.Random(SEED)
    chosen = sorted(rng.sample(both, min(a.n_paintings, len(both))))
    print(f"sampling {len(chosen)} paintings -> {2 * len(chosen)} examples")

    zf = zipfile.ZipFile(semart_zip)
    meta = load_semart_metadata(zf)
    print(f"SemArt metadata rows: {len(meta)}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    entries, missing_img, missing_meta = [], 0, 0

    for fn in chosen:
        member = f"SemArt/Images/{fn}"
        # A stable id that is filesystem-safe and traceable back to SemArt.
        pid = fn.rsplit(".", 1)[0]
        dst = IMAGES_DIR / f"{pid}.jpg"
        if not dst.exists():
            try:
                with zf.open(member) as src, dst.open("wb") as out:
                    out.write(src.read())
            except KeyError:
                missing_img += 1
                continue
        m = meta.get(fn)
        if not m:
            missing_meta += 1
            m = {"title": "", "artist": "", "date": ""}

        # One question of each class per painting; deterministic pick.
        c = by_image[fn]["contextual"][0]
        v = by_image[fn]["visual"][0]
        entries.append({
            "id": pid,
            "semart_image": fn,
            "title": m["title"] or pid,
            "artist": m["artist"],
            "year": m["date"],
            "technique": m.get("technique", ""),
            "school": m.get("school", ""),
            "img_url": None,          # local only; SemArt is redistributed as a zip
            "visual_question": clean(v["question"]),
            "visual_answer": clean(v["answer"]),
            "contextual_question": clean(c["question"]),
            "contextual_answer": clean(c["answer"]),
            "n_candidates_visual": len(by_image[fn]["visual"]),
            "n_candidates_contextual": len(by_image[fn]["contextual"]),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(entries)} paintings ({2 * len(entries)} examples) -> {OUT_PATH}")
    print(f"images -> {IMAGES_DIR}")
    if missing_img:
        print(f"  {missing_img} images missing from the archive")
    if missing_meta:
        print(f"  {missing_meta} paintings had no SemArt metadata row")

    with_artist = sum(1 for e in entries if e["artist"])
    print(f"  {with_artist}/{len(entries)} have a ground-truth artist from SemArt")


if __name__ == "__main__":
    main()
