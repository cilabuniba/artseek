#!/usr/bin/env python3
"""
Fetch image metadata (license, author, credit, etc.) for wikifragments dataset.
Queries Wikipedia and Wikimedia Commons APIs in batches for efficiency.
"""

import time
import ctypes
from multiprocessing import Value, Lock
from urllib.parse import unquote

import requests
from datasets import load_dataset, disable_caching, Features, Sequence, Value as DSValue, Image
from datasets.utils.file_utils import get_datasets_user_agent
from tqdm.auto import tqdm


# Configuration
MAX_RETRIES = 10
WAIT_SECONDS = 10
API_BATCH_SIZE = 50  # Wikimedia API supports up to 50 titles per request
MAP_BATCH_SIZE = 2048
NUM_PROC = 12

# Query Wikipedia first (can resolve both local and Commons images), fallback to Commons
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_API_URL = "https://commons.wikimedia.org/w/api.php"

# Create a session with proper User-Agent (important for higher rate limits!)
session = requests.Session()
USER_AGENT = "WikifragmentsBot/1.0 (https://huggingface.co/datasets/nicolafan/wikifragments; your-email@example.com) " + get_datasets_user_agent()
session.headers.update({"User-Agent": USER_AGENT})

# Shared counters for multiprocessing
success_counter = Value(ctypes.c_int, 0)
failure_counter = Value(ctypes.c_int, 0)

# Lock and path for failed URLs file (append mode)
failed_urls_lock = Lock()
failed_urls_path = None

# Progress bars (created lazily in first process that needs them)
pbar_success = None
pbar_failure = None


def get_progress_bars():
    global pbar_success, pbar_failure
    if pbar_success is None:
        pbar_success = tqdm(desc="✓ Retrieved", unit=" images", position=0, colour="green")
        pbar_failure = tqdm(desc="✗ Failed", unit=" images", position=1, colour="red")
    return pbar_success, pbar_failure


def normalize_filename(filename):
    """Decode and normalize filename for API lookup."""
    decoded = unquote(filename)
    if decoded.startswith("InfoboxHeader:"):
        decoded = "File:" + decoded[len("InfoboxHeader:"):]
    return decoded


def fetch_bulk_image_metadata(all_filenames, max_retries=MAX_RETRIES, wait_seconds=WAIT_SECONDS):
    """
    Fetch metadata for many Wikimedia files using batched API calls.
    all_filenames: list of all unique filenames to fetch
    Returns dict mapping normalized_filename -> metadata dict (or None if failed).
    """
    if not all_filenames:
        return {}
    
    # Normalize all filenames and deduplicate
    normalized_map = {fn: normalize_filename(fn) for fn in all_filenames}
    unique_titles = list(set(normalized_map.values()))
    
    # Results keyed by normalized title
    results_by_title = {}
    
    # Try Wikipedia first, then Commons
    api_urls = [WIKIPEDIA_API_URL, WIKIMEDIA_API_URL]
    remaining_titles = set(unique_titles)
    
    for api_url in api_urls:
        if not remaining_titles:
            break
        
        # Process in batches of API_BATCH_SIZE
        titles_list = list(remaining_titles)
        for i in range(0, len(titles_list), API_BATCH_SIZE):
            batch_titles = titles_list[i:i + API_BATCH_SIZE]
            
            for attempt in range(max_retries):
                try:
                    params = {
                        "action": "query",
                        "titles": "|".join(batch_titles),
                        "prop": "imageinfo",
                        "iiprop": "url|extmetadata",
                        "format": "json"
                    }
                    r = session.get(api_url, params=params, timeout=60)
                    r.raise_for_status()
                    data = r.json()

                    pages = data.get("query", {}).get("pages", {})
                    
                    # Build a map from normalized titles back to our original titles
                    # The API may normalize underscores to spaces, change case, etc.
                    normalized_info = data.get("query", {}).get("normalized", [])
                    normalized_to_original = {}
                    for norm in normalized_info:
                        normalized_to_original[norm["to"]] = norm["from"]
                    
                    for page in pages.values():
                        title = page.get("title", "")
                        imageinfo_list = page.get("imageinfo", [])
                        
                        # Find the original title we asked for (before API normalization)
                        original_title = normalized_to_original.get(title, title)
                        
                        if imageinfo_list and original_title in remaining_titles:
                            info = imageinfo_list[0]
                            metadata = info.get("extmetadata", {})
                            
                            results_by_title[original_title] = {
                                "image_url": info.get("url"),
                                "source_url": info.get("descriptionurl"),
                                "author": metadata.get("Artist", {}).get("value", None),
                                "license": metadata.get("LicenseShortName", {}).get("value", None),
                                "license_url": metadata.get("LicenseUrl", {}).get("value", None),
                                "credit": metadata.get("Credit", {}).get("value", None),
                            }
                            remaining_titles.discard(original_title)
                    
                    break  # Success for this batch
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(wait_seconds)
                    # else move on
    
    # Map back to original filenames
    results = {}
    for orig_fn, norm_title in normalized_map.items():
        results[orig_fn] = results_by_title.get(norm_title, None)
    
    return results


def process_images_column_batched(batch):
    """
    Process a batch of rows. batch is a dict of lists where each key has batch_size elements.
    batch["images"] is a list of dicts, each dict has keys: caption, image, type, url (all lists)
    """
    global success_counter, failure_counter, failed_urls
    
    images_list = batch["images"]  # List of image dicts, one per row
    
    # Collect ALL unique filenames across all rows in this batch
    all_filenames = set()
    for images in images_list:
        urls = images.get("url", [])
        if urls:
            all_filenames.update(urls)
    
    # Fetch metadata for all filenames in bulk
    metadata_cache = fetch_bulk_image_metadata(list(all_filenames))
    
    # Count successes and failures, write failed URLs to file
    successes = 0
    failures = 0
    failed_batch = []
    for fn, v in metadata_cache.items():
        if v is not None:
            successes += 1
        else:
            failures += 1
            failed_batch.append(fn)
    
    # Write failed URLs to file in append mode (with lock for multiprocessing)
    if failed_batch and failed_urls_path is not None:
        with failed_urls_lock:
            with open(failed_urls_path, "a") as f:
                for url in failed_batch:
                    f.write(url + "\n")
    
    with success_counter.get_lock():
        success_counter.value += successes
    with failure_counter.get_lock():
        failure_counter.value += failures
    
    pbar_s, pbar_f = get_progress_bars()
    if successes > 0:
        pbar_s.update(successes)
    if failures > 0:
        pbar_f.update(failures)
    
    # Build images_temp for each row using the cached metadata
    images_temp_list = []
    for images in images_list:
        filenames = images.get("url", [])
        
        if not filenames:
            images_temp_list.append({
                "caption": [],
                "image": [],
                "type": [],
                "url": [],
                "image_url": [],
                "source_url": [],
                "author": [],
                "license": [],
                "license_url": [],
                "credit": [],
            })
        else:
            images_temp_list.append({
                # Original fields
                "caption": images.get("caption", []),
                "image": images.get("image", []),
                "type": images.get("type", []),
                "url": images.get("url", []),
                # New metadata fields from cache
                "image_url": [metadata_cache[fn]["image_url"] if metadata_cache.get(fn) else None for fn in filenames],
                "source_url": [metadata_cache[fn]["source_url"] if metadata_cache.get(fn) else None for fn in filenames],
                "author": [metadata_cache[fn]["author"] if metadata_cache.get(fn) else None for fn in filenames],
                "license": [metadata_cache[fn]["license"] if metadata_cache.get(fn) else None for fn in filenames],
                "license_url": [metadata_cache[fn]["license_url"] if metadata_cache.get(fn) else None for fn in filenames],
                "credit": [metadata_cache[fn]["credit"] if metadata_cache.get(fn) else None for fn in filenames],
            })
    
    batch["images_temp"] = images_temp_list
    return batch


def get_new_features(ds):
    """Build the complete features schema with images_temp."""
    new_features = ds["train"].features.copy()
    
    new_features["images_temp"] = {
        "caption": Sequence(DSValue("string")),
        "image": Sequence(Image()),
        "type": Sequence(DSValue("string")),
        "url": Sequence(DSValue("string")),
        "image_url": Sequence(DSValue("string")),
        "source_url": Sequence(DSValue("string")),
        "author": Sequence(DSValue("string")),
        "license": Sequence(DSValue("string")),
        "license_url": Sequence(DSValue("string")),
        "credit": Sequence(DSValue("string")),
    }
    
    return new_features


def main():
    global pbar_success, pbar_failure, failed_urls_path
    
    print("Loading dataset...")
    disable_caching()
    ds = load_dataset("nicolafan/wikifragments")
    print(f"Loaded dataset: {ds}")
    
    # Reset counters and progress bars
    success_counter.value = 0
    failure_counter.value = 0
    pbar_success = None
    pbar_failure = None
    
    # Initialize failed URLs file (clear it first)
    failed_urls_path = "data/wikifragments_failed_urls.txt"
    with open(failed_urls_path, "w") as f:
        pass  # Clear the file
    
    # Build the new features schema
    new_features = get_new_features(ds)
    
    print(f"\nProcessing with batch_size={MAP_BATCH_SIZE}, num_proc={NUM_PROC}...")
    
    # Step 1: Map to create images_temp with old + new fields (batched!)
    ds = ds.map(
        process_images_column_batched,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        num_proc=NUM_PROC,
        features=new_features
    )
    
    # Close progress bars
    if pbar_success:
        pbar_success.close()
    if pbar_failure:
        pbar_failure.close()
    
    print(f"\n✓ Total retrieved: {success_counter.value}")
    print(f"✗ Total failed: {failure_counter.value}")
    
    # Step 2: Drop old 'images' column and rename 'images_temp' to 'images'
    ds = ds.remove_columns(["images"])
    ds = ds.rename_column("images_temp", "images")
    
    print(f"\nFinal dataset: {ds}")
    
    # Step 3: Save to disk
    output_path = "data/wikifragments_licensed"
    print(f"\nSaving to {output_path}...")
    ds.save_to_disk(output_path, num_proc=NUM_PROC)
    
    print(f"\nFailed URLs written to {failed_urls_path}")
    print("Done!")


if __name__ == "__main__":
    main()
