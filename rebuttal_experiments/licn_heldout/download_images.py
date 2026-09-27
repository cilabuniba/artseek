"""Download the LICNHeldOut images (1280px Wikimedia Commons thumbnails) to
data/external/licn_heldout/images/{id}.jpg.

Sequential with a pause between requests (Wikimedia rate-limits). Existing
files are skipped.

Usage:
    python rebuttal_experiments/licn_heldout/download_images.py
"""
import argparse, json, time
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parents[1] / "data" / "external" / "licn_heldout" / "images"
# Wikimedia requires a descriptive User-Agent.
HDR = {"User-Agent": "ArtSeekResearchBot/1.0 "
                     "(https://github.com/cilabuniba/artseek) python-requests"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pace", type=float, default=1.5)
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    rows = json.loads((HERE / "data" / "licn_vqa.json").read_text())
    ok = fail = 0
    for i, r in enumerate(rows, 1):
        dst = IMAGES / f"{r['id']}.jpg"
        if dst.exists() and dst.stat().st_size > 1000:
            ok += 1
            continue
        url = r["image_url"].replace("http://", "https://")
        if "Special:FilePath" in url and "?" not in url:
            url += "?width=1280"
        for attempt in range(3):
            time.sleep(a.pace)
            try:
                resp = requests.get(url, headers=HDR, timeout=120)
                if resp.status_code == 429:
                    time.sleep(5 * (attempt + 1)); continue
                resp.raise_for_status()
                dst.write_bytes(resp.content); ok += 1; break
            except Exception as e:
                if attempt == 2:
                    print(f"  {r['id']}: {e}"); fail += 1
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}  ok={ok} fail={fail}", flush=True)
    print(f"\n{ok} images on disk, {fail} failed -> {IMAGES}")


if __name__ == "__main__":
    main()
