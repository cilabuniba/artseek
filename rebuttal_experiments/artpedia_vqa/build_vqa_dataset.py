"""Build ArtPedia-VQA from the ArtPedia test split.

For every test painting, a text LLM (Qwen3-8B, vLLM) is given the painting's
human-written visual and contextual sentences and writes one visual and one
contextual question-answer pair grounded in them (no image is shown). Output:
data/artpedia_vqa.json, a list of {id, title, img_url, year, visual_question,
visual_answer, contextual_question, contextual_answer}.

Needs a GPU and data/external/artpedia/artpedia.json. Resumable.

Usage:
    python rebuttal_experiments/artpedia_vqa/build_vqa_dataset.py
    python rebuttal_experiments/artpedia_vqa/build_vqa_dataset.py --limit 10   # quick trial
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402,F401  (sets up sys.path)

import json
import os
import re
from pathlib import Path


import click

# Question/answer generator (override with --model).
DEFAULT_MODEL = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You write visual-question-answering examples for artwork images from "
    "descriptive sentences that are given to you.\n"
    "You will receive two groups of sentences about a single painting:\n"
    "- VISUAL sentences, which describe what can actually be seen in the "
    "image (subjects, composition, colors, objects, style elements).\n"
    "- CONTEXTUAL sentences, which describe facts that are NOT visible in "
    "the image itself (the artist's biography, history, provenance, "
    "influences, current location, etc.).\n\n"
    "From the VISUAL sentences, write one question that could be answered "
    "by looking at the painting, plus a reference answer grounded only in "
    "those sentences.\n"
    "From the CONTEXTUAL sentences, write one question about facts that "
    "cannot be seen in the image but are stated in those sentences, plus a "
    "reference answer grounded only in those sentences.\n\n"
    "Reference answers can be a full sentence or two, not just a single "
    "word or phrase — write them the way a knowledgeable person would "
    "actually answer.\n"
    'Do not mention "the sentences" or "the text" in the questions or '
    "answers; phrase them as if you were an independent expert looking at "
    "and knowing about the painting.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"visual_question": "...", "visual_answer": "...", '
    '"contextual_question": "...", "contextual_answer": "..."}'
)


def build_prompt(entry: dict) -> str:
    visual = " ".join(entry["visual_sentences"])
    contextual = " ".join(entry["contextual_sentences"])
    return (
        f"Painting title: {entry['title']}\n\n"
        f"VISUAL sentences:\n{visual}\n\n"
        f"CONTEXTUAL sentences:\n{contextual}"
    )


def extract_json(text: str) -> dict | None:
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


REQUIRED_KEYS = (
    "visual_question",
    "visual_answer",
    "contextual_question",
    "contextual_answer",
)


@click.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option(
    "--out-path", default=str(common.VQA_DATASET_PATH), show_default=True
)
@click.option("--max-new-tokens", default=1024, show_default=True)
@click.option(
    "--limit", type=int, default=None, help="Only process the first N test entries (quick trial run)."
)
@click.option(
    "--gpu-memory-utilization", default=0.85, show_default=True, type=float
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="Paintings per generate() call. Results are saved after every batch, "
    "so a killed/timed-out job (e.g. hitting the SLURM time limit) loses at "
    "most one batch of progress — re-running the script picks up exactly "
    "where it left off.",
)
def main(
    model: str,
    out_path: str,
    max_new_tokens: int,
    limit: int | None,
    gpu_memory_utilization: float,
    batch_size: int,
):
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    test_entries = common.load_artpedia_test_split()
    ids = sorted(test_entries.keys(), key=int)
    if limit:
        ids = ids[:limit]

    out_path_p = Path(out_path)
    dataset = []
    if out_path_p.exists():
        dataset = json.loads(out_path_p.read_text())
    done_ids = {d["id"] for d in dataset}
    todo = [i for i in ids if i not in done_ids]

    if not todo:
        print(f"Nothing to do — {out_path_p} already has all {len(ids)} requested entries.")
        return

    print(f"Generating VQA pairs for {len(todo)} paintings (model={model})...")
    from huggingface_hub import snapshot_download

    llm = LLM(
        model=snapshot_download(model),
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=max_new_tokens)

    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for batch_start in tqdm(range(0, len(todo), batch_size), desc="VQA batches"):
        batch_ids = todo[batch_start : batch_start + batch_size]
        prompts = []
        for i in batch_ids:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(test_entries[i])},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    # Thinking mode off: not needed, and it could
                    # exhaust max_new_tokens before the JSON.
                    enable_thinking=False,
                )
            )

        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for i, output in zip(batch_ids, outputs):
            entry = test_entries[i]
            raw = output.outputs[0].text
            parsed = extract_json(raw)
            if not parsed or not all(k in parsed for k in REQUIRED_KEYS):
                failures.append((i, raw))
                continue
            dataset.append(
                {
                    "id": i,
                    "title": entry["title"],
                    "img_url": entry["img_url"],
                    "year": entry.get("year"),
                    "visual_question": parsed["visual_question"],
                    "visual_answer": parsed["visual_answer"],
                    "contextual_question": parsed["contextual_question"],
                    "contextual_answer": parsed["contextual_answer"],
                }
            )

        # Save after every batch, not just at the end.
        out_path_p.write_text(json.dumps(dataset, indent=2))

    print(f"Saved {len(dataset)} entries to {out_path_p}.")
    if failures:
        print(f"Failed to parse {len(failures)} entries (unparseable JSON output): {[i for i, _ in failures]}")
        print("Re-run this script to retry only the missing entries (unparsed ones aren't recorded as done).")


if __name__ == "__main__":
    main()
