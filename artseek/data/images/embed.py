from datasets import Dataset, load_from_disk
from pathlib import Path
import os
import torch
from transformers.utils.import_utils import is_flash_attn_2_available

from torchvision.transforms import PILToTensor
from transformers import ColQwen2ForRetrieval, ColQwen2Processor
from datasets import DatasetDict, concatenate_datasets

# from colpali_engine.models import ColQwen2, ColQwen2Processor
from multiprocess import set_start_method
from qdrant_client import QdrantClient, models
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from safetensors.torch import save_file
import atexit


# TODO: update this with the code in retrieve.eval which unifies
# embedding, unpadding and pooling
@torch.no_grad()
def colqwen_embed(dataset_path: Path | str):
    set_start_method("spawn")

    ds = load_from_disk(dataset_path)
    model_name = "vidore/colqwen2-v1.0-hf"

    model = ColQwen2ForRetrieval.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=(
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        ),
    )
    max_pixels = 768 * 28 * 28
    processor = ColQwen2Processor.from_pretrained(model_name, max_pixels=max_pixels)
    model = model.eval()

    def embed(examples, rank):
        device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
        model.to(device)

        inputs = processor(images=examples["fragment"]).to(model.device)
        embeddings = model(**inputs).embeddings
        examples["embeddings"] = embeddings.tolist()

        vcounts = [
            processor(images=[fragment]).input_ids.shape[1]
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


class ColQwenPipeline:
    def __init__(self, dataset_path: str, model_name: str = "vidore/colqwen2-v1.0-hf"):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name

    def run_extract_and_pool(
        self, input_path, batch_size=4, start_idx=None, end_idx=None
    ):
        print(f">>> Starting Phase A: Extraction (Range: {start_idx} to {end_idx})")
        ds_dict = load_from_disk(input_path)

        # Select the split (assuming 'train')
        ds = ds_dict["train"]

        # Apply selection if indices are provided
        if start_idx is not None or end_idx is not None:
            start = start_idx if start_idx is not None else 0
            end = end_idx if end_idx is not None else len(ds)
            ds = ds.select(range(start, end))
            suffix = f"_{start}_{end}"
        else:
            suffix = "_full"

        def _process_batch(examples, rank):
            # ... (Your existing model loading logic remains exactly the same) ...
            if not hasattr(_process_batch, "model"):
                device = f"cuda:{rank % torch.cuda.device_count()}"

                _process_batch.model = (
                    ColQwen2ForRetrieval.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                    )
                    .to(device)
                    .eval()
                )
                _process_batch.processor = ColQwen2Processor.from_pretrained(
                    self.model_name, min_pixels=56 * 56, max_pixels=768 * 28 * 28
                )

            model = _process_batch.model
            processor = _process_batch.processor

            images = [
                PILToTensor()(img).to(model.device) for img in examples["fragment"]
            ]
            inputs = processor(
                images=images, device=model.device, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                embeddings = model(**inputs).embeddings
                attention_mask = inputs.get("attention_mask")

            full_embs_list = []
            pooled_embs_list = []

            for i in range(len(examples["fragment"])):
                vcount = attention_mask[i].sum().item()
                unpadded = embeddings[i, -vcount:, :].cpu().numpy().astype(np.float16)

                # ... (Your existing Pooling Logic remains the same) ...
                special = np.concatenate([unpadded[:4], unpadded[-7:]], axis=0)
                p_special = np.mean(special, axis=0)
                content = unpadded[4:-7]
                n_clusters = 8

                if len(content) < n_clusters:
                    p_content = np.zeros(
                        (n_clusters, unpadded.shape[-1]), dtype=np.float16
                    )
                else:
                    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(
                        content.astype(np.float32)
                    )
                    p_content = [
                        np.mean(content[clustering.labels_ == j], axis=0)
                        for j in range(n_clusters)
                    ]
                    p_content = np.array(p_content, dtype=np.float16)

                pooled_vec = np.vstack([p_special, p_content]).astype(np.float16)
                full_embs_list.append(unpadded)
                pooled_embs_list.append(pooled_vec)

            return {
                "full_embeddings": full_embs_list,
                "pooled_embeddings": pooled_embs_list,
            }

        processed_ds = ds.map(
            _process_batch,
            batched=True,
            batch_size=batch_size,
            with_rank=True,
            num_proc=torch.cuda.device_count(),
            desc="Extracting & Pooling",
        )

        # Build output path with indices
        output_name = f"{Path(input_path).name}_embeds{suffix}"
        output_path = Path(input_path).parent / output_name

        # Save as DatasetDict to maintain compatibility
        DatasetDict({"train": processed_ds}).save_to_disk(output_path)
        print(f"Extraction Complete. Saved to {output_path}")
        return output_path

    def merge_selections(self, base_path):
        """Finds all chunks in the directory and merges them into one."""
        parent_dir = Path(base_path).parent
        base_name = Path(base_path).name

        # Find directories matching the pattern: base_name_embeds_start_end
        chunk_paths = sorted(
            [p for p in parent_dir.glob(f"{base_name}_embeds_*_*") if p.is_dir()],
            key=lambda x: int(x.name.split("_")[-2]),  # Sort by start_idx
        )

        if not chunk_paths:
            print("No chunks found to merge.")
            return

        print(f"Merging {len(chunk_paths)} chunks...")
        datasets_to_combine = [load_from_disk(str(p))["train"] for p in chunk_paths]

        merged_ds = concatenate_datasets(datasets_to_combine)
        final_path = parent_dir / f"{base_name}_embeds_final"

        DatasetDict({"train": merged_ds}).save_to_disk(final_path, num_proc=8)
        print(f"Merge Complete. Final dataset saved at: {final_path}")
        return final_path

    def run(
        self, step: str = "all", input_path: str = None, start_idx=None, end_idx=None
    ):
        current_path = Path(input_path) if input_path else self.dataset_path

        if step in ["extract", "all"]:
            current_path = self.run_extract_and_pool(
                current_path, batch_size=16, start_idx=start_idx, end_idx=end_idx
            )

        if step == "merge":
            current_path = self.merge_selections(self.dataset_path)

        return current_path


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
