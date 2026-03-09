import json
import logging
import time
from pathlib import Path

import click
import pandas as pd
import os
import torch
from datasets import load_from_disk, disable_caching
from dotenv import find_dotenv, load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from PIL import Image
from qdrant_client import QdrantClient, models
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from wikidata.client import Client
from sklearn.model_selection import train_test_split
from multiprocess import set_start_method

from ..utils.dirutils import get_data_dir, get_fonts_dir, get_project_dir, get_store_dir
from .datasets.processing import FragmentCreator
from .graph import ops as graph_ops
from .graph import wikidata as graph_wikidata
from .graph import wikipedia as graph_wikipedia
from .graph import wikipedia_new as graph_wikipedia_new
from .images import downloader as images_downloader
from .images import embed as images_embed
from .texts import extractor as texts_extractor
from .texts import wikifragments_creator as texts_wiki_full_img_text_creator


@click.group()
@click.pass_context
def cli(ctx):
    # set logging and load env
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    load_dotenv(find_dotenv())

    # check if context object exists
    ctx.ensure_object(dict)

    ctx.obj["logger"] = logger
    try:
        ctx.obj["graph"] = Neo4jGraph(
            url=os.environ["NEO4J_URI"],
            username=os.environ["NEO4J_USERNAME"],
            password=os.environ["NEO4J_PASSWORD"],
        )
    except Exception as e:
        logger.error(f"Error creating Neo4j graph: {e}")
        ctx.obj["graph"] = None


@cli.command()
@click.pass_context
def download_wikiart_images(ctx):
    logger = ctx.obj["logger"]
    graph = ctx.obj["graph"]

    images_downloader.download_and_save_images_wikiart(
        graph,
        get_data_dir() / "artgraph" / "images",
    )
    logger.info(
        f"Images downloaded and saved at {get_data_dir() / 'artgraph' / 'images'}"
    )


@cli.command()
@click.pass_context
def make_artgraph_dataset(ctx):
    # TODO: add shuffling for the dataset
    logger = ctx.obj["logger"]
    graph = ctx.obj["graph"]

    data_dir = get_data_dir() / "artgraph"

    # from the graph, get all artworks with connected Artist, Style,
    relationships = [
        ("createdBy", "Artist"),
        ("about", "Tag"),
        ("hasGenre", "Genre"),
        ("hasStyle", "Style"),
        ("madeOf", "Media"),
    ]
    artgraph = {"relationships": {}}

    for relationship in relationships:
        query = f"""
        MATCH (a:Artwork)-[:{relationship[0]}]->(b:{relationship[1]})
        RETURN ID(a) as artwork_id, ID(b) as {relationship[1].lower()}_id
        """
        result = graph.query(query)

        for record in result:
            if record["artwork_id"] not in artgraph["relationships"]:
                artgraph["relationships"][record["artwork_id"]] = {}
            if (
                relationship[1].lower()
                not in artgraph["relationships"][record["artwork_id"]]
            ):
                artgraph["relationships"][record["artwork_id"]][
                    relationship[1].lower()
                ] = []
            artgraph["relationships"][record["artwork_id"]][
                relationship[1].lower()
            ].append(record[f"{relationship[1].lower()}_id"])

    node_types = ["Artwork", "Artist", "Tag", "Genre", "Style", "Media"]

    for node_type in node_types:
        query = f"""
        MATCH (n:{node_type})
        RETURN ID(n) as id, n.name as name
        """
        result = graph.query(query)

        artgraph[node_type.lower()] = {
            record["id"]: record["name"] for record in result
        }

    # make a stratified split by style using the collected data
    artwork_style = []
    for artwork_id, relationships in artgraph["relationships"].items():
        if "style" in relationships:
            style_id = relationships["style"][0]
            artwork_style.append((artwork_id, style_id))

    df = pd.DataFrame(artwork_style, columns=["artwork_id", "style_id"])
    train, test = train_test_split(
        df, test_size=0.3, stratify=df["style_id"], random_state=42
    )
    val, test = train_test_split(
        test, test_size=0.5, stratify=test["style_id"], random_state=42
    )

    # now make a set of artwork ids for each split
    split_sets = {
        "train": set(train["artwork_id"]),
        "val": set(val["artwork_id"]),
        "test": set(test["artwork_id"]),
    }

    # make the splits by selecting only the artworks in the split sets
    for split_name, split_set in split_sets.items():
        split_artgraph = {"relationships": {}}
        for artwork_id, relationships in artgraph["relationships"].items():
            if artwork_id in split_set:
                split_artgraph["relationships"][artwork_id] = relationships
        for node_type in node_types:
            split_artgraph[node_type.lower()] = {
                k: v for k, v in artgraph[node_type.lower()].items()
            }
        with open(data_dir / f"{split_name}.json", "w") as f:
            json.dump(split_artgraph, f)
        logger.info(f"Artgraph dataset saved at {data_dir / split_name}.json")


@cli.command()
@click.pass_context
def define_valid_labels_artgraph_dataset(ctx):
    logger = ctx.obj["logger"]
    data_dir = get_data_dir() / "artgraph"

    with open(data_dir / "train.json", "r") as f:
        train = json.load(f)
    with open(data_dir / "val.json", "r") as f:
        val = json.load(f)
    with open(data_dir / "test.json", "r") as f:
        test = json.load(f)

    label_counts = {}
    tasks = ["artist", "genre", "media", "style", "tag"]
    label_counts = {task: {} for task in tasks}
    for split in [train, val, test]:
        for artwork_id, relationships in split["relationships"].items():
            for task in tasks:
                if task in relationships:
                    for label in relationships[task]:
                        if label not in label_counts[task]:
                            label_counts[task][label] = 0
                        label_counts[task][label] += 1

    valid_labels = {}
    for task in tasks:
        # keep only labels that appear in at least 100 artworks
        valid_labels[task] = [k for k, v in label_counts[task].items() if v >= 100]

    with open(data_dir / "valid_labels.json", "w") as f:
        json.dump(valid_labels, f, indent=4)
    logger.info(f"Valid labels saved at {data_dir / 'valid_labels.json'}")


@cli.command()
@click.pass_context
def get_visual_arts_dataset_pages(ctx):
    logger = ctx.obj["logger"]

    category = "Category:Visual arts"
    graph_wikipedia_new.select_category_pages(category, 5)
    logger.info(
        f"Visual arts dataset pages saved at {get_data_dir() / 'graph' / category}.csv"
    )


# ----------------- AFTER WIKIEXTRACTOR -------------------------------------------------


@cli.command()
@click.pass_context
def download_and_save_images_wikipedia(ctx):
    logger = ctx.obj["logger"]

    images_downloader.download_and_save_images_wikipedia(
        get_data_dir() / "texts" / "text_en",
        get_data_dir() / "wikipedia_images",
    )
    logger.info(f"Images downloaded and saved at {get_data_dir() / 'wikipedia_images'}")


@cli.command()
@click.pass_context
def create_wikifragments_dataset(ctx):
    logger = ctx.obj["logger"]

    texts_wiki_full_img_text_creator.make_wikipedia_dataset(
        get_data_dir() / "texts" / "text_en",
        get_data_dir() / "wikipedia_images",
        get_data_dir() / "wikifragments_dataset",
    )
    logger.info(
        f"Wiki full img text dataset saved at {get_data_dir() / 'wikifragments_dataset'}"
    )


@cli.command()
@click.pass_context
def create_wikifragments_visual_arts_full_dataset(ctx):
    logger = ctx.obj["logger"]

    fragment_creator = FragmentCreator(font_path=get_fonts_dir() / "arial.ttf")
    ds = load_from_disk(get_data_dir() / "wikifragments_dataset")

    df = pd.read_csv(get_data_dir() / "graph" / "Category:Visual arts.csv")
    visual_arts_ids = set(df["id"].tolist())

    ds = ds.filter(
        lambda x: x["wiki_id"] in visual_arts_ids,
        num_proc=32,
        load_from_cache_file=False,
    )

    def create_fragment(example):
        example["fragment"] = fragment_creator.example_to_image(example)
        return example

    ds = ds.map(create_fragment, num_proc=32, load_from_cache_file=False)
    ds.save_to_disk(get_data_dir() / "wikifragments_visual_arts_dataset", num_proc=32)


@cli.command()
@click.pass_context
def colqwen_embed(ctx):
    logger = ctx.obj["logger"]

    images_embed.colqwen_embed(get_data_dir() / "wikifragments_visual_arts_dataset")
    logger.info("Embeddings created")


@cli.command()
@click.option(
    "--start-idx", type=int, default=None, help="Start index for dataset selection"
)
@click.option(
    "--end-idx", type=int, default=None, help="End index for dataset selection"
)
@click.option(
    "--step", type=click.Choice(["extract", "merge", "all"]), default="extract"
)
@click.pass_context
def colqwen_embed_new(ctx, start_idx, end_idx, step):
    logger = ctx.obj["logger"]

    set_start_method("spawn")
    embedder = images_embed.ColQwenPipeline(
        str(get_data_dir() / "wikifragments_visual_arts_dataset"),
    )

    embedder.run(step=step, start_idx=start_idx, end_idx=end_idx)


@cli.command()
@click.pass_context
@click.option("--process-idx", default=0, help="Process index")
@click.option("--num-proc", default=1, help="Number of processes")
def make_qdrant_store(ctx, process_idx, num_proc):
    logger = ctx.obj["logger"]

    images_embed.make_qdrant_store(
        "cilabuniba/wikifragments-visual-arts-embeds",
        process_idx=process_idx,
        num_proc=num_proc,
    )
    logger.info("Qdrant index created")


@cli.command()
@click.pass_context
def add_qdrant_index(ctx):
    logger = ctx.obj["logger"]

    client = QdrantClient(url="http://localhost", prefer_grpc=True)
    try:
        client.get_collection("wikifragments-visual-arts-embeds")
    except Exception:
        raise ValueError("Collection does not exist")
    client.update_collection(
        collection_name="wikifragments-visual-arts-embeds",
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        hnsw_config=models.HnswConfigDiff(m=16),
    )

    # every 5 minutes get the collection and check the state
    while True:
        info = client.get_collection("wikifragments-visual-arts-embeds")
        logger.info(
            f"Qdrant index status: {info.status} - {info.indexed_vectors_count} / {info.points_count}"
        )
        if info.status != "yellow":
            logger.info(f"Qdrant index updated to {info.status}")
            break
        time.sleep(300)

    logger.info("Qdrant index updated")


@cli.command()
@click.pass_context
def add_qdrant_metadata(ctx):
    # TODO: add metadata to the qdrant index, in the current moment I would just use the HF dataset
    raise NotImplementedError


if __name__ == "__main__":
    cli(obj={})
