"""Collect the evaluation images into images/<id>.jpg (JPEG, longest side 1024).

Wikipedia paintings are downloaded from Wikimedia Commons (scaled copies).
Gallery paintings are re-encoded from the gallery photographs (--gallery-dir,
not distributed; the resulting images are in images/). Existing files are
kept.

Usage:
    python rebuttal_experiments/human_eval/fetch_images.py
"""

import click
import json
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
MAX_SIDE = 1024      # plenty for both the VLMs and the rating page
JPEG_QUALITY = 88
HDR = {"User-Agent": "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"}


def thumb_url(url: str, width: int = MAX_SIDE) -> str:
    """A scaled copy through Special:FilePath?width= instead of the master."""
    if "Special:FilePath" in url:
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}width={width}"
    if "upload.wikimedia.org" in url:
        name = url.rsplit("/", 1)[-1]
        return ("https://commons.wikimedia.org/wiki/Special:FilePath/"
                f"{name}?width={width}")
    return url


def save(img: Image.Image, dest: Path) -> tuple[int, int]:
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return img.size


@click.command()
@click.option("--items-file", default="items.json", show_default=True,
              help="File under data/ listing the items to fetch.")
@click.option("--gallery-dir", default=None,
              help="Gallery catalogue (image_local paths are relative to it).")
@click.option("--images-dir", default=None,
              help="Directory under human_eval/ for the JPEGs. "
                   "Defaults to the module-level IMAGES.")
def main(items_file: str, gallery_dir: str | None, images_dir: str | None):
    global IMAGES
    if images_dir:
        IMAGES = HERE / images_dir
    IMAGES.mkdir(exist_ok=True)
    items_path = HERE / "data" / items_file
    items = json.loads(items_path.read_text())
    updated = []
    for it in items:
        dest = IMAGES / f"{it['id']}.jpg"
        if dest.exists():
            print(f"  have {dest.name}")
            it["image"] = f"{IMAGES.name}/{dest.name}"
            updated.append(it)
            continue
        try:
            if it.get("image_local"):
                if not gallery_dir:
                    raise ValueError("gallery photograph needs --gallery-dir")
                img = Image.open(Path(gallery_dir) / it["image_local"])
            else:
                # Retry with back-off (Commons rate-limits).
                img = None
                for attempt in range(5):
                    try:
                        r = requests.get(thumb_url(it["image_url"]),
                                         headers=HDR, timeout=180)
                        r.raise_for_status()
                        img = Image.open(BytesIO(r.content))
                        break
                    except Exception as e:
                        if attempt == 4:
                            raise
                        wait = 15 * (attempt + 1)
                        print(f"     retry {attempt+1}/4 in {wait}s "
                              f"({str(e)[:50]})", flush=True)
                        time.sleep(wait)
                time.sleep(3.0)
            w, h = save(img, dest)
            it["image"] = f"{IMAGES.name}/{dest.name}"
            print(f"  {dest.name:28s} {w}x{h}  {dest.stat().st_size//1024} KB  "
                  f"[{it['tier']}] {it['title'][:34]}")
        except Exception as e:
            print(f"  !! {it['id']}: {str(e)[:90]}")
            it["image"] = None
        updated.append(it)

    items_path.write_text(json.dumps(updated, indent=2, ensure_ascii=False))
    ok = sum(1 for i in updated if i.get("image"))
    total_kb = sum((IMAGES / f"{i['id']}.jpg").stat().st_size
                   for i in updated if i.get("image")) // 1024
    print(f"\n{ok}/{len(updated)} images ready, {total_kb} KB total")


if __name__ == "__main__":
    main()
