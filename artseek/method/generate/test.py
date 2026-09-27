"""Zero-shot artwork explanation (captioning) on ArtPedia, SemArt v2.0 and
PaintingForm with the LangGraph pipeline.

The config (models/configs/generate/*.yaml) sets the dataset, the prompt,
whether classification and retrieval are used, and the output directory.
Predictions go to <out_dir>/preds.json (saved every 50 images; a re-run
resumes). Score them with `python -m artseek.method.generate.eval score`.

Usage:
    python -m artseek.method.generate.test inference --config-path models/configs/generate/artpedia_short.yaml
"""

import json
from pathlib import Path

import click
import torch
import yaml
from hydra.utils import instantiate
from langchain_core.messages import HumanMessage
from tqdm import tqdm

from ...data.datasets.explain_me import ExplainMeDataset
from ...data.datasets.painting_form import PaintingFormDataset
from ...utils.dirutils import get_data_dir


@click.group
def cli():
    pass


def load_existing_results(out_dir):
    preds_file = Path(out_dir) / "preds.json"
    if preds_file.exists():
        with open(preds_file, "r") as f:
            preds = json.load(f)
        return preds
    return []


def save_results(out_dir, preds):
    preds_file = Path(out_dir) / "preds.json"
    with open(preds_file, "w") as f:
        json.dump(preds, f)
    print(f"Saved results to {preds_file}.")


@cli.command
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the config file.",
)
@torch.no_grad()
def inference(config_path: Path | str):
    from .graph import build_graph

    # Load the config
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = instantiate(config_dict)

    GRAPH = build_graph(
        classify=config.classify,
        retrieve=config.retrieve,
    )

    dataset = config.dataset
    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    preds = load_existing_results(config.out_dir)
    processed_ids = {entry["image_id"] for entry in preds}

    for i, sample in enumerate(tqdm(dataset)):
        if i in processed_ids:
            continue

        response = GRAPH.invoke(
            {
                "input_image": sample["image"],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "# Current query image\n"},
                            {"type": "image"},
                            {
                                "type": "text",
                                "text": f"\n# Query\n{config.prompt}",
                            },
                        ],
                    },
                ],
            },
            {"recursion_limit": 50},
        )

        # Take all messages after the last human message
        messages = response["messages"]
        generated_messages = []
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            generated_messages.append(message)
        generated_messages.reverse()

        message_preds = []
        for message in generated_messages:
            message_pred = {"type": message.type, "content": message.content}
            if hasattr(message, "tool_calls") and message.tool_calls:
                message_pred["tool_calls"] = message.tool_calls
            if hasattr(message, "artifact"):
                if "context_idxs" in message.artifact:
                    message_pred["doc_idxs"] = message.artifact["context_idxs"]
            message_preds.append(message_pred)
        preds.append(
            {
                "image_id": i,
                "messages": message_preds,
            }
        )
        # Save the results every 50 samples
        if i % 50 == 0:
            print(f"Processed {i} samples.")
            save_results(config.out_dir, preds)

    # Save the results at the end
    save_results(config.out_dir, preds)


if __name__ == "__main__":
    cli()
