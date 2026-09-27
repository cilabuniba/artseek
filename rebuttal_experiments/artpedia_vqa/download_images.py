"""Download the ArtPedia test-split images to data/external/artpedia/images/{id}.jpg.

Already downloaded images are skipped, so re-run to retry failures. A few
source URLs are dead; those paintings are skipped by the runner.

Usage:
    python rebuttal_experiments/artpedia_vqa/download_images.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402,F401  (sets up sys.path)

import io
import time

import click
import requests
from PIL import Image
from tqdm import tqdm

# Some Wikimedia scans exceed PIL's decompression-bomb pixel limit.
Image.MAX_IMAGE_PIXELS = None

HEADERS = {
    # Wikimedia rejects requests with no / a generic User-Agent.
    "User-Agent": "ArtSeekResearchBot/1.0 (https://github.com/cilabuniba/artseek)"
}


def _get_with_backoff(url: str, timeout: int, retries: int):
    """GET url, retrying on 429 with exponential backoff (respecting
    Retry-After when Wikimedia sends one) rather than hammering it."""
    backoff = 5.0
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", backoff))
                if attempt < retries:
                    time.sleep(wait)
                    backoff *= 2
                    continue
            resp.raise_for_status()
            return resp
        except Exception as e:  # noqa: BLE001 — collect and report, don't abort the run
            last_err = e
    raise last_err


@click.command()
@click.option("--timeout", default=30, show_default=True, help="Per-request timeout in seconds.")
@click.option("--retries", default=5, show_default=True, help="Retries per image before giving up.")
@click.option(
    "--delay",
    default=1.0,
    show_default=True,
    type=float,
    help="Seconds to sleep between successful requests, to stay well under Wikimedia's rate limit.",
)
def main(timeout: int, retries: int, delay: float):
    test_entries = common.load_artpedia_test_split()
    common.ARTPEDIA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    ok, skipped, failed = 0, 0, []
    for artpedia_id, entry in tqdm(sorted(test_entries.items(), key=lambda kv: int(kv[0]))):
        out_path = common.ARTPEDIA_IMAGES_DIR / f"{artpedia_id}.jpg"
        if out_path.exists():
            skipped += 1
            continue

        url = entry["img_url"]
        try:
            resp = _get_with_backoff(url, timeout, retries)
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            image.save(out_path, "JPEG", quality=95)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((artpedia_id, url, str(e)))
        time.sleep(delay)

    print(f"\nDownloaded {ok} new images, {skipped} already present, {len(failed)} failed.")
    if failed:
        print("Failed downloads:")
        for artpedia_id, url, err in failed:
            print(f"  id={artpedia_id} url={url} error={err}")


if __name__ == "__main__":
    main()
