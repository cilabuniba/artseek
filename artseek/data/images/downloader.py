import concurrent.futures
import json
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from langchain_community.graphs import Neo4jGraph
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from wikidata.client import Client
import time


import concurrent.futures
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import requests
from tqdm import tqdm

from typing import List, Union
from requests import get


def download_image_wikidata(entity_id: str, output_dir: Path, img_res: int, client: Client, image_prop, headers: dict) -> str:
    """Download and save a single image from WikiData."""
    image_path = output_dir / f"{entity_id}.jpg"
    if image_path.exists():
        return f"Already downloaded {entity_id}"

    try:
        entity = client.get(entity_id, load=True)
        image_data = entity[image_prop]
        image_url = image_data.image_url

        with get(image_url, headers=headers, stream=True) as response:
            response.raise_for_status()
            with Image.open(response.raw) as image:
                image = image.convert("RGB").resize((img_res, img_res))
                image.save(image_path)
        return f"Downloaded {entity_id}"

    except KeyError:
        return f"No image for {entity_id}"
    except UnidentifiedImageError as e:
        return f"Error downloading {entity_id}: Unidentified image - {e}"
    except Exception as e:
        return f"Error processing {entity_id}: {e}"

def download_and_save_images_wikidata(
    entity_ids: List[str],
    output_dir: Union[Path, str],
    img_res: int = 224,
    num_workers: int = 8,
) -> None:
    """Download and save images from WikiData using multiple workers.

    Args:
        entity_ids: List of WikiData entity IDs.
        output_dir: Directory to save the downloaded images.
        img_res: Resolution to resize the images to (default: 224x224).
        num_workers: Number of concurrent workers (default: 8).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = Client()
    image_prop = client.get("P18")
    headers = {"User-Agent": "Knowledge Graph Image Downloader"}

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_image_wikidata, entity_id, output_dir, img_res, client, image_prop, headers): entity_id for entity_id in entity_ids}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            print(future.result())


def download_image_wikiart(record, output_dir, existing_images, modifiers):
    """Download and save an image from WikiArt."""
    image_url = record["image_url"]
    name = record["name"]

    # Skip if the image is already downloaded
    if name in existing_images:
        return None

    # Remove existing modifiers
    for modifier in modifiers:
        modifier_idx = image_url.find(f"!{modifier}")
        if modifier_idx != -1:
            image_url = image_url[:modifier_idx]

    ext = image_url.split(".")[-1]

    # Try different image resolutions
    for modifier in modifiers:
        modified_url = f"{image_url}!{modifier}.{ext}"
        try:
            response = requests.get(modified_url, stream=True, timeout=10)
            if response.status_code == 200:
                with response:
                    image = Image.open(response.raw)
                    image = image.convert("RGB")
                    if name.split(".")[-1] != "jpg":
                        name = name.split(".")[0] + ".jpg"
                    image.save(output_dir / name)
                    image.close()
                return f"Downloaded {name}"
        except Exception as e:
            time.sleep(0.5)  # Small delay to avoid overloading
            continue

    return f"Error downloading {name}"


def download_and_save_images_wikiart(graph, output_dir: Path | str, num_threads=16):
    """Download and save images from WikiArt using a fixed number of threads."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_images = {img.name for img in output_dir.glob("*.jpg")}
    modifiers = [
        "HD",
        "HalfHD",
        "Large",
        "Blog",
        "Portrait",
        "PinterestLarge",
        "PinterestSmall",
    ]

    query = """
    MATCH (a:Artwork)
    WHERE a.image_url IS NOT NULL
    RETURN a.image_url AS image_url, a.name AS name
    """
    result = graph.query(query)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(
            tqdm(
                executor.map(
                    lambda record: download_image_wikiart(
                        record, output_dir, existing_images, modifiers
                    ),
                    result,
                ),
                total=len(result),
            )
        )

    for res in results:
        if res is not None:
            print(res)


def download_and_save_images_wikipedia(
    input_dir: Path | str, output_dir: Path | str
) -> None:
    """Download and save images from Wikipedia.

    Args:
        input_dir (Path | str): The directory with the Wikipedia URLs.
        output_dir (Path | str): The directory to save the images.
    """
    # get all files which start with "wiki" in the dir and subdir of input_dir
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # create output dir if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(list(input_dir.glob("**/wiki*"))):
        with open(file, "r") as f:
            pages = f.readlines()
        for page in pages:
            page = json.loads(page)
            text = page["text"]
            links = re.findall(r'<a href="([^"]+)">(.*?)</a>', text)
            for url, cap in links:
                if url.startswith("File"):
                    chars_to_skip = 7
                elif url.startswith("InfoboxHeader"):
                    chars_to_skip = 16
                else:
                    continue
                filename = url[chars_to_skip:].replace("%20", "_")
                # check if image is already saved
                try:
                    if (output_dir / f"{filename}.jpg").exists():
                        continue
                except:
                    continue
                try:
                    image = Image.open(
                        requests.get(
                            f"http://127.0.0.1:8080/content/wikipedia_en_all_maxi_2024-01/I/{filename}.webp",
                            stream=True,
                        ).raw
                    ).convert("RGB")
                    image.save(output_dir / f"{filename}.jpg")
                except:
                    continue
