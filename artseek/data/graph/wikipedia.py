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

from ...utils.lookups import BAD_WIKI_TITLES_SUBSTRS_LOOKUP


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


def select_category_pages(category: str, max_depth: int = 0):
    """Create a file of Wikipedia pages from a category.

    Args:
        category (str): Name of the category.
        max_depth (int, optional): Maximum search depth (included). Defaults to 0.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "ArtSeek/1.0 (contact: add your contact information here)"
    })
    pages = _find_pages_in_category_recursive(category, session, max_depth=max_depth)
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
