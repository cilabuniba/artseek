"""Retrieval evaluation (paper Sec. 5.1 and supplementary Tab. S1).

Steps, each a command of this module:

  sample     10,000 WikiFragments with images: 5,000 with exactly one image
             and 5,000 with two or more (the first half becomes text-only once
             its image is removed).
  questions  remove the first image of every fragment, re-render the fragment,
             and let Qwen2.5-VL-32B write a question about the removed image
             ("this image") answerable from the fragment.
  embed      ColQwen2 embeddings of the re-rendered fragments (full and pooled).
  store      Qdrant collections: full multi-vector, pooled two-stage, or CLIP.
  evaluate   NDCG@5, R@1 and query time for a store / query-encoding pair.

Usage:
    python -m artseek.method.retrieve.eval sample --out data/retrieval_eval/sample
    python -m artseek.method.retrieve.eval questions --dataset data/retrieval_eval/sample --out data/retrieval_eval/qa
    python -m artseek.method.retrieve.eval embed --dataset data/retrieval_eval/qa --out data/retrieval_eval/qa_embeds
    python -m artseek.method.retrieve.eval store --dataset data/retrieval_eval/qa_embeds --kind pooled
    python -m artseek.method.retrieve.eval evaluate --dataset data/retrieval_eval/qa --store pooled --query filtered
"""

import io
import json
import os
import shutil
import time
from pathlib import Path

import click

import numpy as np
import pandas as pd
import torch
from accelerate import infer_auto_device_map
from colpali_engine.models import ColQwen2, ColQwen2Processor
from datasets import concatenate_datasets, load_dataset, load_from_disk
from langchain_core.messages import HumanMessage, SystemMessage
from multiprocess import set_start_method
from PIL import Image
from qdrant_client import QdrantClient, models
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.aggregation import SumMetric

from ...data.datasets.processing import FragmentCreator
from ...method.generate import Qwen2_5_VLChatModel
from ...utils.dirutils import get_fonts_dir
from .metrics import RetrievalMetrics

EMBEDDING_COLUMNS = ("embedding", "vcount", "pooled_embedding",
                     "full_embeddings", "pooled_embeddings")
# Qdrant collection per store kind (binary quantization, as in the paper).
COLLECTIONS = {
    "full": "retrieval_eval_full_binary",
    "pooled": "retrieval_eval_reduced_binary",
    "clip": "retrieval_eval_clip_binary",
}


def _load(path):
    """A dataset saved with save_to_disk, or a Hugging Face Hub id."""
    return load_from_disk(path) if Path(path).exists() else load_dataset(str(path))


def _to_pil(image):
    """Dataset images come either decoded or as {"bytes": ...}."""
    if isinstance(image, dict):
        return Image.open(io.BytesIO(image["bytes"]))
    return image


def stratified_sample(df, column_name, n_samples, random_state=42):
    # Get the proportion of each class
    proportions = df[column_name].value_counts(normalize=True)

    # Compute how many samples to take from each class
    sample_counts = (proportions * n_samples).round().astype(int)

    # Ensure the total adds up to n_samples (due to rounding issues)
    diff = n_samples - sample_counts.sum()
    if diff != 0:
        # Adjust the most frequent class
        most_common_class = sample_counts.idxmax()
        sample_counts[most_common_class] += diff

    # Perform sampling
    sampled_df = pd.concat(
        [
            df[df[column_name] == label].sample(n=count, random_state=random_state)
            for label, count in sample_counts.items()
        ]
    )

    # Shuffle the result
    return sampled_df.sample(frac=1, random_state=random_state).reset_index(drop=True)


def make_sample_dataset(dataset_path, out_path, sample_size=10000, min_images=1):
    """Sample fragments with at least `min_images` images: half with exactly
    `min_images`, half with more (stratified by image count, seed 42)."""
    ds = _load(dataset_path)
    ds = ds.remove_columns([c for c in EMBEDDING_COLUMNS if c in ds["train"].column_names])
    ds = ds.filter(
        lambda x: len(x["images"]["image"]) >= min_images,
        num_proc=32,
    )

    ds_for_sample = ds.map(
        lambda x, idx: {"idx": idx, "num_images": len(x["images"]["image"])},
        num_proc=32,
        with_indices=True,
        remove_columns=ds["train"].column_names,
    )
    df_for_sample = ds_for_sample["train"].to_pandas()

    # Take half samples_size of examples with num_images==1 and half randomly from the rest
    df_for_sample_min = df_for_sample[df_for_sample["num_images"] == min_images]
    df_for_sample_rest = df_for_sample[df_for_sample["num_images"] > min_images]

    # Sample the minimum images
    df_for_sample_min = stratified_sample(
        df_for_sample_min, "num_images", sample_size // 2
    )
    # Sample the rest
    df_for_sample_rest = stratified_sample(
        df_for_sample_rest, "num_images", sample_size // 2
    )
    # Concatenate the samples
    df_for_sample = pd.concat([df_for_sample_min, df_for_sample_rest])

    # Get the indices of the sample
    sample_indices = df_for_sample["idx"].tolist()

    # Select the sample examples
    ds_sample = ds["train"].select(sample_indices)
    ds_sample.save_to_disk(out_path, num_proc=min(32, max(1, len(ds_sample) // 100)))


def make_question_answer_dataset(dataset_path, out_path):
    ds = load_from_disk(dataset_path)
    fragment_creator = FragmentCreator(font_path=get_fonts_dir() / "arial.ttf")

    def make_eval_fragment(example):
        example["old_fragment"] = example["fragment"]
        example["old_images"] = example["images"]
        example["images"] = {k: v[1:] for k, v in example["images"].items()}
        example["fragment"] = fragment_creator.example_to_image(example)
        return example

    ds = ds.map(make_eval_fragment, num_proc=8)

    model = Qwen2_5_VLChatModel.from_pretrained(
        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="sequential",
    )
    model.processor.min_pixels = 1

    system_prompt = """You are a helpful assistant.

    # Task  
    You are given a **query image** and a **document image**. Your goal is to generate a **question about the query image** that can be used to test whether a system can correctly retrieve the associated document image based on the query.

    - The question **must include the phrase "this image"**.  
    - **Do not mention or reference the document image** in the question.  
    - The **answer to the question must be found within the document image** and must be a **single sentence**.

    # Format  
    Return your response as a JSON object in the following format:

    {
    "question": "Your generated question here.",
    "answer": "The correct answer found in the document image."
    }
    """
    messages = [
        SystemMessage(content=[{"type": "text", "text": system_prompt}]),
        HumanMessage(
            content=[
                {"type": "text", "text": "Query image: "},
                {"type": "image"},
                {"type": "text", "text": "Document image: "},
                {"type": "image"},
            ]
        ),
    ]

    def make_question_answer(example):
        images = [
            _to_pil(example["old_images"]["image"][0]),
            example["fragment"],
        ]

        # If image height or width is less than 28, pad it to 28
        for i in range(len(images)):
            if images[i].height < 28 or images[i].width < 28:
                new_size = (
                    (28, images[i].height)
                    if images[i].width < 28
                    else (images[i].width, 28)
                )
                images[i] = Image.new("RGB", new_size, (255, 255, 255))
                images[i].paste(images[i], (0, 0))

        response = model.invoke(messages, images=images)
        try:
            response = json.loads(response.content)
        except json.JSONDecodeError:
            example["question"] = ""
            example["answer"] = ""
            return example
        example["question"] = response["question"]
        example["answer"] = response["answer"]
        return example

    # Ten shards, each saved when done, so an interrupted run resumes.
    shard_paths = [f"{out_path}_shard_{i}" for i in range(10)]
    for i, shard_path in enumerate(shard_paths):
        if os.path.exists(shard_path):
            continue
        shard = ds.shard(10, i, contiguous=True)
        shard = shard.map(make_question_answer)
        shard.save_to_disk(shard_path, num_proc=32)

    ds = concatenate_datasets([load_from_disk(p) for p in shard_paths])
    ds.save_to_disk(out_path, num_proc=32)
    for shard_path in shard_paths:
        shutil.rmtree(shard_path)


@torch.no_grad()
def colqwen_embed(dataset_path: Path | str, out_path: Path | str):
    set_start_method("spawn")

    ds = load_from_disk(dataset_path)
    model_name = "vidore/colqwen2-v1.0"

    model = ColQwen2.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    processor = ColQwen2Processor.from_pretrained(model_name)
    model = model.eval()

    def embed(examples, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        model.to(device)

        inputs = processor.process_images(examples["fragment"]).to(model.device)
        embeddings = model(**inputs)
        examples["embedding"] = embeddings.tolist()

        vcounts = [
            processor.process_images([fragment]).input_ids.shape[1]
            for fragment in examples["fragment"]
        ]
        examples["vcount"] = vcounts
        examples["embedding"] = [
            embedding[-vcount:]
            for embedding, vcount in zip(examples["embedding"], vcounts)
        ]

        pooled_embeddings = []
        for full_embedding in examples["embedding"]:
            special_embeddings = np.array(full_embedding[:4] + full_embedding[-7:])
            content_embeddings = np.array(full_embedding[4:-7])
            pooled_special_embedding = np.mean(special_embeddings, axis=0)

            # Agglomerative clustering
            n_clusters = 8
            clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(
                content_embeddings
            )
            pooled_base_embeddings = np.array(
                [
                    np.mean(content_embeddings[clustering.labels_ == i], axis=0)
                    for i in range(n_clusters)
                ]
            )
            pooled_embeddings.append(
                np.concatenate(
                    (pooled_special_embedding.reshape(1, -1), pooled_base_embeddings)
                ).tolist()
            )
        examples["pooled_embedding"] = pooled_embeddings
        return examples

    new_ds = ds.map(
        embed,
        batched=True,
        batch_size=4,
        with_rank=True,
        num_proc=torch.cuda.device_count(),
    )
    new_ds.save_to_disk(out_path, num_proc=torch.cuda.device_count())


def make_full_qdrant_store(dataset_path: Path | str, binary: bool = True):
    batch_size = 32
    dataset_path = Path(dataset_path)

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost"), prefer_grpc=True)
    collection_name = f"retrieval_eval_full{'_binary' if binary else ''}"
    ds = load_from_disk(dataset_path)

    if binary:
        quantization_config = models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(),
        )
    else:
        quantization_config = None
    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=quantization_config,
        ),
        shard_number=4,
    )

    def upload_points(examples, idxs, client=None):
        if client is None:
            raise RuntimeError("Client is None")
        retries = 5
        for attempt in range(retries):
            try:
                client.upload_points(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=id,
                            vector=vector,
                            payload={"idx": idx},
                        )
                        for id, vector, idx in zip(
                            examples["id"],
                            examples["embedding"],
                            idxs,
                        )
                    ],
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise
        return examples

    # Split the dataset into shards
    ds.map(
        upload_points,
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        fn_kwargs={"client": client},
        load_from_cache_file=False,
    )


def make_reduced_qdrant_store(dataset_path: Path | str, binary: bool = True):
    batch_size = 32
    dataset_path = Path(dataset_path)

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost"), prefer_grpc=True)
    collection_name = f"retrieval_eval_reduced{'_binary' if binary else ''}"
    ds = load_from_disk(dataset_path)

    if binary:
        quantization_config = models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(),
        )
    else:
        quantization_config = None

    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config={
            "pooled": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=quantization_config,
            ),
            "full": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                quantization_config=quantization_config,
            ),
        },
        shard_number=4,
    )

    def upload_points(examples, idxs, client=None):
        if client is None:
            raise RuntimeError("Client is None")
        retries = 5
        for attempt in range(retries):
            try:
                client.upload_points(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=id,
                            vector={"full": vector, "pooled": pooled_vector},
                            payload={"idx": idx},
                        )
                        for id, pooled_vector, vector, idx in zip(
                            examples["id"],
                            examples["pooled_embedding"],
                            examples["embedding"],
                            idxs,
                        )
                    ],
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise
        return examples

    ds.map(
        upload_points,
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        fn_kwargs={"client": client},
        load_from_cache_file=False,
    )


@torch.no_grad()
def make_clip_qdrant_store(dataset_path: Path | str, binary: bool = True):
    batch_size = 32
    dataset_path = Path(dataset_path)

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost"), prefer_grpc=True)
    collection_name = f"retrieval_eval_clip{'_binary' if binary else ''}"
    ds = load_from_disk(dataset_path)

    if binary:
        quantization_config = models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(),
        )
    else:
        quantization_config = None

    client.create_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=512,  # CLIP ViT-B/32 embedding size
            distance=models.Distance.COSINE,
            quantization_config=quantization_config,
        ),
        shard_number=4,
    )

    # Here we don't have precomputed embeddings, so we need to compute them on the fly
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.eval()

    def upload_points(examples, idxs, client=None):
        if client is None:
            raise RuntimeError("Client is None")
        retries = 5
        for attempt in range(retries):
            try:
                images = [img for img in examples["fragment"]]
                inputs = processor(images=images, return_tensors="pt", padding=True).to(
                    model.device
                )
                embeddings = model.get_image_features(**inputs).cpu().numpy().tolist()

                client.upload_points(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=id,
                            vector=embedding,
                            payload={"idx": idx},
                        )
                        for id, embedding, idx in zip(
                            examples["id"],
                            embeddings,
                            idxs,
                        )
                    ],
                )
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    raise
        return examples

    ds.map(
        upload_points,
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        fn_kwargs={"client": client},
        load_from_cache_file=False,
    )


@torch.no_grad()
def eval(dataset_path: Path | str, collection_name: str, query_type: str = "full",
         out_dir: Path | str = "."):
    assert query_type in [
        "full",
        "reduced",
        "text",
        "clip",
    ], "query_type must be either 'full' or 'reduced'"
    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost"), prefer_grpc=True)

    metrics = RetrievalMetrics()
    total_time = SumMetric()

    if query_type != "clip":
        model = ColQwen2.from_pretrained(
            "vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16
        )
        processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
        model = model.eval()
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.eval()

    dataset_path = Path(dataset_path)
    ds = load_from_disk(dataset_path)
    # Metrics over the first half of the sample (the text-only fragments).
    metrics_dict_half = {}

    def process_example_clip(example, i):
        nonlocal metrics_dict_half
        image = _to_pil(example["old_images"]["image"][0])
        question = example["question"]
        
        # Embed the query
        image_inputs = processor(images=image, return_tensors="pt").to(model.device)
        text_inputs = processor(text=question, return_tensors="pt").to(model.device)
        image_outputs = model.get_image_features(**image_inputs)
        text_outputs = model.get_text_features(**text_inputs)
        
        # Sum the embeddings
        embeddings = (image_outputs + text_outputs).squeeze().tolist()

        # Perform the search
        start_time = time.time()
        response = client.query_points(
            collection_name=collection_name,
            query=embeddings,
            limit=5,
            with_payload=True,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=120,
        )
        end_time = time.time()
        total_time.update(end_time - start_time)

        # Compute metrics
        preds, real = [record.score for record in response.points], [
            record.payload["idx"] for record in response.points
        ]
        target = [1 if r == i else 0 for r in real]
        indexes = [i] * 5
        metrics.update(
            torch.tensor(preds), torch.tensor(target), indexes=torch.tensor(indexes)
        )
        if i == (len(ds) // 2) - 1:
            metrics_dict_half = metrics.compute()
            metrics_dict_half = {f"half_{k}": v for k, v in metrics_dict_half.items()}

    def process_example(example, i):
        nonlocal metrics_dict_half
        image = _to_pil(example["old_images"]["image"][0])
        question = example["question"]

        # Embed the query
        image_inputs = processor.process_images([image]).to(model.device)
        text_inputs = processor.process_queries([question]).to(model.device)
        inputs = image_inputs
        inputs["input_ids"] = torch.cat(
            (
                inputs["input_ids"][:, :-6],
                text_inputs["input_ids"],
                inputs["input_ids"][:, -2:],
            ),
            dim=1,
        )
        inputs["attention_mask"] = torch.cat(
            (
                inputs["attention_mask"][:, :-6],
                text_inputs["attention_mask"],
                inputs["attention_mask"][:, -2:],
            ),
            dim=1,
        )
        outputs = model(**inputs)
        if query_type == "reduced":
            embeddings = outputs[
                :, -(text_inputs["input_ids"].shape[1] + 2) : -2
            ].tolist()[0]
        else:
            embeddings = outputs.tolist()[0]

        # Perform the search
        start_time = time.time()
        if "reduced" in collection_name:
            prefetch_limit = 100
            limit = 5
            prefetch_oversampling = 2.0
            oversampling = 2.0
            response = client.query_points(
                collection_name=collection_name,
                prefetch=models.Prefetch(
                    query=embeddings,
                    using="pooled",
                    limit=prefetch_limit,
                    params=models.SearchParams(
                        quantization=models.QuantizationSearchParams(
                            ignore=False,
                            rescore=False,
                            oversampling=prefetch_oversampling,
                        )
                    ),
                ),
                query=embeddings,
                using="full",
                limit=limit,
                with_payload=True,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=oversampling,
                    )
                ),
                timeout=120,
            )
        else:
            response = client.query_points(
                collection_name=collection_name,
                query=embeddings,
                limit=5,
                with_payload=True,
                search_params=models.SearchParams(
                    quantization=models.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    )
                ),
                timeout=120,
            )
        end_time = time.time()
        total_time.update(end_time - start_time)

        # Compute metrics
        preds, real = [record.score for record in response.points], [
            record.payload["idx"] for record in response.points
        ]
        target = [1 if r == i else 0 for r in real]
        indexes = [i] * 5
        metrics.update(
            torch.tensor(preds), torch.tensor(target), indexes=torch.tensor(indexes)
        )
        if i == (len(ds) // 2) - 1:
            metrics_dict_half = metrics.compute()
            metrics_dict_half = {f"half_{k}": v for k, v in metrics_dict_half.items()}

    if query_type == "clip":
        ds.map(process_example_clip, with_indices=True, num_proc=1, load_from_cache_file=False)
    else:
        ds.map(process_example, with_indices=True, num_proc=1, load_from_cache_file=False)

    # Save the metrics
    metrics_dict = metrics.compute()
    metrics_dict.update(metrics_dict_half)
    metrics_dict["total_time"] = total_time.compute()
    metrics_dict["average_time"] = metrics_dict["total_time"] / len(ds)
    metrics_dict = {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in metrics_dict.items()
    }
    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(out_dir, f"{collection_name}__{query_type}_metrics.json")
    with open(file_name, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(json.dumps(metrics_dict, indent=2))
    print(f"saved {file_name}")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dataset", default="cilabuniba/wikifragments-visual-arts-embeds",
              show_default=True, help="Hub id or save_to_disk path.")
@click.option("--out", required=True, type=click.Path())
@click.option("--sample-size", default=10000, show_default=True)
def sample(dataset, out, sample_size):
    """Sample the evaluation fragments."""
    make_sample_dataset(dataset, out, sample_size=sample_size)


@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def questions(dataset, out):
    """Remove the first image and generate the questions (GPU)."""
    make_question_answer_dataset(dataset, out)


@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def embed(dataset, out):
    """ColQwen2 full and pooled embeddings of the fragments (GPU)."""
    colqwen_embed(dataset, out)


@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True),
              help="Output of `embed` (full/pooled) or of `questions` (clip).")
@click.option("--kind", type=click.Choice(list(COLLECTIONS)), required=True)
def store(dataset, kind):
    """Create a Qdrant collection for the evaluation (QDRANT_URL)."""
    {"full": make_full_qdrant_store, "pooled": make_reduced_qdrant_store,
     "clip": make_clip_qdrant_store}[kind](dataset, binary=True)


@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True),
              help="Output of `questions`.")
@click.option("--store", "store_kind", type=click.Choice(list(COLLECTIONS)), required=True)
@click.option("--query", type=click.Choice(["full", "filtered", "clip"]), required=True,
              help="full: whole multimodal query; filtered: its text tokens only.")
@click.option("--out-dir", default="data/retrieval_eval/results", show_default=True)
def evaluate(dataset, store_kind, query, out_dir):
    """NDCG@5, R@1 and query time."""
    query_type = {"full": "full", "filtered": "reduced", "clip": "clip"}[query]
    eval(dataset, COLLECTIONS[store_kind], query_type, out_dir=out_dir)


if __name__ == "__main__":
    cli()
