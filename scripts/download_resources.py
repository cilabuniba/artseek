"""Download every resource ArtSeek needs from the Hugging Face Hub.

Files go to the standard Hugging Face cache (`$HF_HOME`, read from the
environment or from `.env`). The WikiFragments dataset is also prepared with
`datasets.load_dataset`, exactly as the pipeline loads it.

    python scripts/download_resources.py                  # everything
    python scripts/download_resources.py --skip-dataset   # models only (~30 GB)
"""

import os

from dotenv import load_dotenv

load_dotenv()
# Downloading requires the hub even if .env enables offline mode.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

import click
from huggingface_hub import snapshot_download

MODELS = [
    ("Qwen/Qwen2.5-VL-32B-Instruct-AWQ", "model"),  # backbone MLLM
    ("vidore/colqwen2-v1.0", "model"),  # retriever (LoRA adapter)
    ("vidore/colqwen2-base", "model"),  # retriever base weights
    ("cilabuniba/artseek-licn", "model"),  # LICN classifier
    ("cilabuniba/artseek-licn-data", "dataset"),  # LICN label spaces
]
DATASET_ID = "cilabuniba/wikifragments-visual-arts-embeds"


@click.command()
@click.option("--skip-dataset", is_flag=True, help="Do not download WikiFragments.")
@click.option("--num-proc", default=8, show_default=True, help="Processes for dataset preparation.")
def main(skip_dataset: bool, num_proc: int):
    print(f"HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")
    for repo_id, repo_type in MODELS:
        print(f"Downloading {repo_id} ...")
        snapshot_download(repo_id, repo_type=repo_type)

    if not skip_dataset:
        from datasets import load_dataset

        print(f"Downloading and preparing {DATASET_ID} (this takes several hours) ...")
        print(load_dataset(DATASET_ID, num_proc=num_proc))


if __name__ == "__main__":
    main()
