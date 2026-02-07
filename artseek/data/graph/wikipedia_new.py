import logging
from pathlib import Path
import re
import pandas as pd
from wikidata.client import Client

import click
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.graphs import Neo4jGraph
from ...utils.dirutils import get_data_dir
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from ...utils.lookups import BAD_WIKI_TITLES_SUBSTRS_LOOKUP

# Thread-local storage for session reuse
_thread_local = threading.local()

# Rate limiting: Wikipedia allows ~200 requests/minute
_rate_limiter_lock = threading.Lock()
_last_request_time = 0
_min_request_interval = 0.05  # 50ms between requests = ~20 requests/second max

def _get_session(user_agent: str) -> requests.Session:
    """Get or create a thread-local session."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
        _thread_local.session.headers.update({"User-Agent": user_agent})
    return _thread_local.session


def _rate_limited_request(session: requests.Session, url: str, params: dict, timeout: int = 10):
    """Make a rate-limited request with retry logic for 429 errors."""
    global _last_request_time
    
    attempt = 0
    while True:
        # Rate limiting
        with _rate_limiter_lock:
            now = time.time()
            wait_time = _min_request_interval - (now - _last_request_time)
            if wait_time > 0:
                time.sleep(wait_time)
            _last_request_time = time.time()
        
        response = session.get(url=url, params=params, timeout=timeout)
        
        if response.status_code == 429:
            attempt += 1
            wait = 10 * attempt
            print(f"Rate limited, waiting {wait}s before retry (attempt {attempt})...")
            time.sleep(wait)
            continue
        
        response.raise_for_status()
        return response


def find_wiki_styles(graph: Neo4jGraph) -> None:
    """Find Wikipedia documents for each style in the database.

    Args:
        graph (Neo4jGraph): The graph database.
    """
    query = """
    MATCH (s:Style)
    RETURN ID(s) as id, s.name as name
    """
    result = graph.query(query)
    styles = {record["id"]: record["name"] for record in result}
    style_queries = [re.sub(r"\(.*\)", "", style).title() for style in styles.values()]

    style_queries[style_queries.index("Realism")] = "Realism (arts)"
    style_queries[style_queries.index("Symbolism")] = "Symbolism (arts)"

    documents = [
        WikipediaLoader(query=query, load_max_docs=1).load() for query in style_queries
    ]

    data = []
    for style_id, document in zip(styles.keys(), documents):
        if not document:
            print(f"No Wikipedia document found for {styles[style_id]}")
            continue
        summary = document[0].metadata["summary"]
        source = document[0].metadata["source"]
        data.append({"id": style_id, "summary": summary.strip(), "wikipedia_url": source.strip()})

    df = pd.DataFrame(data)
    df.to_csv(get_data_dir() / "graph" / "styles.csv", index=False)


def _fetch_category_pages(category: str, user_agent: str) -> list:
    """Fetch all pages in a single category (handles pagination).
    
    Args:
        category (str): Name of the category.
        user_agent (str): User-Agent string for the API.
        
    Returns:
        list: List of pages in the category.
    """
    session = _get_session(user_agent)
    url = "https://en.wikipedia.org/w/api.php"
    pages = []
    
    params = {
        "action": "query",
        "cmtitle": category,
        "cmlimit": "500",
        "cmtype": "page",
        "list": "categorymembers",
        "format": "json",
    }
    
    while True:
        try:
            response = _rate_limited_request(session, url, params)
            data = response.json()
            
            if "error" in data:
                print(f"API error for {category}: {data['error']}")
                return []
                
        except Exception as e:
            print(f"Request error for {category}: {e}")
            return []
        
        pages.extend(data["query"]["categorymembers"])
        
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    
    return pages


def _fetch_subcategories(category: str, user_agent: str) -> list:
    """Fetch all subcategories of a category (handles pagination).
    
    Args:
        category (str): Name of the category.
        user_agent (str): User-Agent string for the API.
        
    Returns:
        list: List of subcategory titles.
    """
    session = _get_session(user_agent)
    url = "https://en.wikipedia.org/w/api.php"
    subcats = []
    
    params = {
        "action": "query",
        "cmtitle": category,
        "cmlimit": "500",
        "cmtype": "subcat",
        "list": "categorymembers",
        "format": "json",
    }
    
    while True:
        try:
            response = _rate_limited_request(session, url, params)
            data = response.json()
            
            if "error" in data:
                print(f"API error for {category}: {data['error']}")
                return []
                
        except Exception as e:
            print(f"Request error for {category}: {e}")
            return []
        
        subcats.extend([s["title"] for s in data["query"]["categorymembers"]])
        
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    
    return subcats


def _find_pages_in_category_parallel(
    category: str,
    max_depth: int = 0,
    max_workers: int = 5,
    user_agent: str = "ArtSeek/1.0 (research project) Python/requests",
) -> list:
    """Find all pages in a Wikipedia category recursively using parallel processing.

    Args:
        category (str): Name of the category.
        max_depth (int, optional): Maximum search depth (included). Defaults to 0.
        max_workers (int, optional): Number of parallel workers. Defaults to 10.
        user_agent (str, optional): User-Agent string for the API.

    Returns:
        list: List of pages in the category.
    """
    all_pages = []
    visited = set()
    visited_lock = threading.Lock()
    
    # Categories to process at current depth level
    current_level = [category]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for depth in range(max_depth + 1):
            if not current_level:
                break
                
            # Filter out already visited categories
            with visited_lock:
                categories_to_process = [c for c in current_level if c not in visited]
                visited.update(categories_to_process)
            
            if not categories_to_process:
                print(f"Depth {depth}: All {len(current_level)} categories already visited, stopping.")
                break
            
            print(f"Depth {depth}: Processing {len(categories_to_process)} categories ({len(current_level) - len(categories_to_process)} already visited)...")
            
            # Fetch pages from all categories at this level in parallel
            page_futures = {
                executor.submit(_fetch_category_pages, cat, user_agent): cat 
                for cat in categories_to_process
            }
            
            for future in tqdm(as_completed(page_futures), total=len(page_futures), desc=f"Fetching pages (depth {depth})"):
                try:
                    pages = future.result()
                    all_pages.extend(pages)
                except Exception as e:
                    print(f"Error fetching pages: {e}")
            
            # Fetch subcategories from all categories at this level in parallel
            next_level = []
            if depth < max_depth:
                subcat_futures = {
                    executor.submit(_fetch_subcategories, cat, user_agent): cat 
                    for cat in categories_to_process
                }
                
                for future in tqdm(as_completed(subcat_futures), total=len(subcat_futures), desc=f"Fetching subcategories (depth {depth})"):
                    try:
                        subcats = future.result()
                        next_level.extend(subcats)
                    except Exception as e:
                        print(f"Error fetching subcategories: {e}")
                
                # Show how many subcategories were found
                with visited_lock:
                    new_subcats = [c for c in next_level if c not in visited]
                print(f"Depth {depth}: Found {len(next_level)} subcategories, {len(new_subcats)} are new")
            
            current_level = next_level
    
    return all_pages


def _find_pages_in_category_recursive(
    category: str,
    session: requests.Session,
    depth: int = 0,
    max_depth: int = 0,
    visited: set = set(),
):
    """Find all pages in a Wikipedia category recursively.

    Args:
        category (str): Name of the category.
        session (requests.Session): Requests session.
        depth (int, optional): Current search depth. Defaults to 0.
        max_depth (int, optional): Maximum search depth (included). Defaults to 0.
        visited (set, optional): Already visited categories. Defaults to set().

    Returns:
        list: List of pages in the category.
    """
    url = "https://en.wikipedia.org/w/api.php"
    pages = []

    if category in visited:
        return set()
    visited.add(category)

    get_pages_params = {
        "action": "query",
        "cmtitle": category,
        "cmlimit": "500",
        "cmtype": "page",
        "list": "categorymembers",
        "format": "json",
    }
    get_subcats_params = {
        "action": "query",
        "cmtitle": category,
        "cmlimit": "500",
        "cmtype": "subcat",
        "list": "categorymembers",
        "format": "json",
    }

    while True:
        try:
            R = session.get(url=url, params=get_pages_params, timeout=5)
            DATA = R.json()
        except:
            return []

        pages += DATA["query"]["categorymembers"]

        if not "continue" in DATA:
            break
        get_pages_params["cmcontinue"] = DATA["continue"]["cmcontinue"]

    if depth <= max_depth:
        while True:
            R = session.get(url=url, params=get_subcats_params)
            DATA = R.json()

            subcats = DATA["query"]["categorymembers"]
            print(f"Found {len(subcats)} subcategories at depth {depth}")

            for subcat in subcats:
                pages += _find_pages_in_category_recursive(
                    subcat["title"],
                    session,
                    depth=depth + 1,
                    max_depth=max_depth,
                    visited=visited,
                )

            if not "continue" in DATA:
                break
            get_subcats_params["cmcontinue"] = DATA["continue"]["cmcontinue"]

    return pages


def select_category_pages(category: str, max_depth: int = 0, max_workers: int = 5):
    """Create a file of Wikipedia pages from a category.

    Args:
        category (str): Name of the category.
        max_depth (int, optional): Maximum search depth (included). Defaults to 0.
        max_workers (int, optional): Number of parallel workers. Defaults to 10.
    """
    pages = _find_pages_in_category_parallel(category, max_depth=max_depth, max_workers=max_workers)
    print(f"Found {len(pages)} pages in the category {category}")

    for page in pages:
        page["id"] = page["pageid"]

    df = pd.DataFrame(pages, columns=["id", "title"])
    # remove duplicates
    df = df.drop_duplicates(subset="id")

    # remove pages with bad titles
    for substr in BAD_WIKI_TITLES_SUBSTRS_LOOKUP:
        df = df[~df["title"].str.contains(substr)]

    df.to_csv(get_data_dir() / "graph" / f"{category}.csv", index=False)
