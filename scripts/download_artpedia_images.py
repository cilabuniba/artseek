"""Download the ArtPedia images listed in artpedia.json.

Images are saved as <data-dir>/images/<id>.jpg, where <id> is the key of the
painting in artpedia.json (the layout ArtpediaDataset expects). Existing files
are skipped, so re-run to retry failures. Some source URLs no longer exist;
those paintings are skipped by the dataset loader.

    python scripts/download_artpedia_images.py                 # test split
    python scripts/download_artpedia_images.py --split all
"""

import io
import json
import time
from pathlib import Path

import click
import requests
from PIL import Image
from tqdm import tqdm

# Some Wikimedia scans exceed PIL's decompression-bomb pixel limit.
Image.MAX_IMAGE_PIXELS = None
# Wikimedia rejects requests without a descriptive User-Agent.
HEADERS = {"User-Agent": "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"}


def get_with_backoff(url: str, timeout: int, retries: int) -> requests.Response:
    """GET with exponential backoff on HTTP 429 (honouring Retry-After)."""
    backoff, last_err = 5.0, None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429 and attempt < retries:
                time.sleep(float(resp.headers.get("Retry-After", backoff)))
                backoff *= 2
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


@click.command()
@click.option("--data-dir", default="data/external/artpedia", show_default=True,
              type=click.Path(exists=True, file_okay=False),
              help="Folder containing artpedia.json.")
@click.option("--split", type=click.Choice(["test", "val", "train", "all"]),
              default="test", show_default=True)
@click.option("--timeout", default=30, show_default=True)
@click.option("--retries", default=5, show_default=True)
@click.option("--delay", default=1.0, show_default=True,
              help="Seconds between requests, to respect Wikimedia's rate limit.")
def main(data_dir: str, split: str, timeout: int, retries: int, delay: float):
    data_dir = Path(data_dir)
    entries = json.loads((data_dir / "artpedia.json").read_text())
    if split != "all":
        entries = {k: v for k, v in entries.items() if v.get("split") == split}
    images_dir = data_dir / "images"
    images_dir.mkdir(exist_ok=True)

    ok, skipped, failed = 0, 0, []
    for key, entry in tqdm(sorted(entries.items(), key=lambda kv: int(kv[0]))):
        out = images_dir / f"{key}.jpg"
        if out.exists():
            skipped += 1
            continue
        try:
            resp = get_with_backoff(entry["img_url"], timeout, retries)
            Image.open(io.BytesIO(resp.content)).convert("RGB").save(out, "JPEG", quality=95)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((key, entry["img_url"], str(e)))
        time.sleep(delay)

    print(f"\n{ok} downloaded, {skipped} already present, {len(failed)} failed")
    for key, url, err in failed:
        print(f"  {key}: {url} ({err[:80]})")


if __name__ == "__main__":
    main()
