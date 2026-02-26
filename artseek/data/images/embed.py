from datasets import Dataset, load_from_disk
from pathlib import Path
import os
from colpali_engine.models import ColQwen2, ColQwen2Processor
import torch
from multiprocess import set_start_method
from qdrant_client import QdrantClient, models
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# TODO: update this with the code in retrieve.eval which unifies
# embedding, unpadding and pooling
@torch.no_grad()
def colqwen_embed(dataset_path: Path | str):
    set_start_method("spawn")

    ds = load_from_disk(dataset_path)
    model_name = "models/colqwen2-v1.0"

    model = ColQwen2.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    processor = ColQwen2Processor.from_pretrained(model_name)
    model = model.eval()

    def embed(examples, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        model.to(device)

        inputs = processor.process_images(examples["fragment"]).to(model.device)
        embeddings = model(**inputs)
        examples["embeddings"] = embeddings.tolist()

        vcounts = [
            processor.process_images([fragment]).input_ids.shape[1]
            for fragment in examples["fragment"]
        ]
        examples["vcount"] = vcounts
        examples["embedding"] = [
            embedding[-vcount:]
            for embedding, vcount in zip(examples["embeddings"], vcounts)
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
                    (pooled_special_embedding.reshape(1, -1), pooled_base_embeddings),
                    axis=0,
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
    new_ds.save_to_disk(
        dataset_path.parent / "wikipedia_visual_arts_dataset_embeds",
        num_proc=torch.cuda.device_count(),
    )


def make_qdrant_store(dataset_path: Path | str, process_idx: int, num_proc: int):
    batch_size = 512
    dataset_path = Path(dataset_path)

    client = QdrantClient(url="http://localhost", prefer_grpc=True)
    ds = load_from_disk(dataset_path)

    if process_idx == 0:
        client.create_collection(
            collection_name=dataset_path.stem,
            on_disk_payload=True,
            vectors_config={
                "pooled": models.VectorParams(
                    size=128,  # size of each vector produced by ColBERT
                    distance=models.Distance.COSINE,  # similarity metric between each vector
                    on_disk=True,
                    datatype=models.Datatype.FLOAT16,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM  # similarity metric between multivectors (matrices)
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(
                            always_ram=True,
                        ),
                    ),
                ),
                "full": models.VectorParams(
                    size=128,  # size of each vector produced by ColBERT
                    distance=models.Distance.COSINE,  # similarity metric between each vector
                    on_disk=True,
                    datatype=models.Datatype.FLOAT16,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM  # similarity metric between multivectors (matrices)
                    ),
                    quantization_config=models.BinaryQuantization(
                        binary=models.BinaryQuantizationConfig(),
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        m=0,
                    ),
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,
            ),
            shard_number=4,
        )

    def upload_points(examples, idxs, start_idx=None, client=None):
        if client is None:
            raise RuntimeError("Client is None")
        retries = 5
        for attempt in range(retries):
            try:
                client.upload_points(
                    collection_name=dataset_path.stem,
                    points=[
                        models.PointStruct(
                            id=id,
                            vector={"full": vector, "pooled": pooled_vector},
                            payload={"idx": start_idx + idx},
                        )
                        for id, pooled_vector, vector, idx in zip(
                            examples["id"],
                            examples["pooled_embedding"],
                            examples["embedding"],
                            idxs,
                        )
                    ],
                )
                break  # Exit the loop if upload is successful
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    print(f"Attempt {attempt + 1} failed: {e}. No more retries left.")
                    raise e
        return examples

    # Split the dataset into shards
    shards = [
        ds["train"].shard(num_shards=num_proc, index=i, contiguous=True)
        for i in range(num_proc)
    ]
    shard = shards[process_idx]
    shard_lens = [len(shard) for shard in shards]
    shard_start_idxs = [sum(shard_lens[:i]) for i in range(num_proc)]
    shard_start_idx = shard_start_idxs[process_idx]

    shard.map(
        upload_points,
        with_indices=True,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        fn_kwargs={"start_idx": shard_start_idx, "client": client},
        load_from_cache_file=False,
    )
