"""Write the ArtCurate-AIC questions from curatorial prose.

Six conceptual question types, one question each where the source supports it:

  iconography   what a depicted element, dress, gesture or motif signifies
  narrative     which event, story or scene is depicted or alluded to
  person        who a depicted figure is and why they matter (not the artist)
  technique     how a formal or material choice serves the work's meaning
  context       the historical, social or biographical circumstances
  multihop      needs both the `description` and the `did_you_know` texts

Questions and reference answers are written by Qwen3-8B (vLLM, greedy) from
the Art Institute of Chicago's curatorial texts (data/artcurate_corpus.json,
from fetch_artworks.py); both source texts are stored with every question.
Writes data/artcurate_vqa.json.

Needs a GPU. Resumable.

Usage:
    python rebuttal_experiments/artcurate_aic/build_questions.py
"""

import json
import os
import re
import sys
from pathlib import Path


import click

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rebuttal_experiments" / "common"))
import common  # noqa: E402,F401  (loads .env)
sys.path.insert(0, str(REPO / "rebuttal_experiments" / "common" / "analysis"))
import lib  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "data" / "artcurate_corpus.json"
OUT = Path(__file__).resolve().parent / "data" / "artcurate_vqa.json"

GENERATOR = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You write demanding, conceptual exam questions about artworks for an "
    "art-history assessment. You are given a museum's own catalogue text "
    "about one artwork: an interpretive DESCRIPTION and a separate "
    "DID_YOU_KNOW fact.\n\n"
    "Write questions of the following types, and a reference answer for "
    "each, grounded ONLY in the given text:\n\n"
    "1. iconography — what a depicted element, garment, gesture, object, or "
    "compositional choice signifies or conveys. Not \"what is shown\" but "
    "\"what does it mean\".\n"
    "2. narrative — what event, story, activity, or scene is depicted or "
    "alluded to.\n"
    "3. person — who a depicted figure is (the SITTER or SUBJECT, never the "
    "artist) and why that person matters.\n"
    "4. technique — how a formal, material, or stylistic choice serves the "
    "work's meaning or effect.\n"
    "5. context — the historical, social, or biographical circumstances "
    "that shaped the work.\n"
    "6. multihop — a question that CANNOT be answered from the DESCRIPTION "
    "alone or from the DID_YOU_KNOW alone, but requires combining a fact "
    "from EACH. This is the most important one: make the dependency on both "
    "sources genuine and explicit in the reference answer.\n\n"
    "Rules:\n"
    "* Ask about the artwork as if speaking to someone looking at it. Never "
    "mention \"the text\", \"the description\", or \"the catalogue\".\n"
    "* Never ask which museum holds the work, what year it was made, or who "
    "painted it — those are trivial lookups and are covered elsewhere.\n"
    "* Questions must be answerable from the given text. Do not invent "
    "facts. If the text does not support a type, omit that type entirely "
    "rather than inventing one.\n"
    "* Reference answers should be one to three sentences, specific, and "
    "state the substance rather than gesturing at it.\n\n"
    "Respond with ONLY a JSON object mapping each type you could write to "
    "an object with \"question\" and \"answer\". Omit types you could not "
    "support. Example shape:\n"
    '{"iconography": {"question": "...", "answer": "..."}, '
    '"multihop": {"question": "...", "answer": "..."}}'
)

TYPES = ("iconography", "narrative", "person", "technique", "context", "multihop")


def build_prompt(a: dict) -> str:
    bits = [f"TITLE: {a.get('title')}"]
    if a.get("artist"):
        bits.append(f"ARTIST: {a['artist']}")
    if a.get("creation_date"):
        bits.append(f"DATE: {a['creation_date']}")
    if a.get("type"):
        bits.append(f"OBJECT TYPE: {a['type']}")
    if a.get("technique"):
        bits.append(f"TECHNIQUE: {a['technique']}")
    bits.append(f"\nDESCRIPTION:\n{a['description']}")
    bits.append(f"\nDID_YOU_KNOW:\n{a['did_you_know']}")
    return "\n".join(bits)


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


@click.command()
@click.option("--model", default=GENERATOR, show_default=True)
@click.option("--max-new-tokens", default=1400, show_default=True)
@click.option("--batch-size", default=16, show_default=True)
@click.option("--gpu-memory-utilization", default=0.85, type=float, show_default=True)
@click.option("--limit", type=int, default=None)
def main(model, max_new_tokens, batch_size, gpu_memory_utilization, limit):
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    corpus = json.loads(CORPUS.read_text())
    if limit:
        corpus = corpus[:limit]
    print(f"{len(corpus)} artworks in the corpus")

    existing = json.loads(OUT.read_text()) if OUT.exists() else []
    done = {e["id"] for e in existing}
    todo = [a for a in corpus if a["id"] not in done]
    print(f"{len(done)} already written, {len(todo)} to go")
    if not todo:
        return

    path = snapshot_download(model)
    print(f"generator: {model} @ {Path(path).name}")
    llm = LLM(model=path, dtype="bfloat16",
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=8192, seed=lib.SEED)
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=lib.SEED)

    entries, failures = list(existing), []
    for i in tqdm(range(0, len(todo), batch_size), desc="writing questions"):
        batch = todo[i: i + batch_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_prompt(a)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for a in batch
        ]
        outs = llm.generate(prompts, params, use_tqdm=False)
        for a, o in zip(batch, outs):
            parsed = extract_json(o.outputs[0].text)
            if not parsed:
                failures.append(a["id"])
                continue
            qa = {}
            for t in TYPES:
                v = parsed.get(t)
                if isinstance(v, dict) and v.get("question") and v.get("answer"):
                    qa[f"{t}_question"] = str(v["question"]).strip()
                    qa[f"{t}_answer"] = str(v["answer"]).strip()
            if not qa:
                failures.append(a["id"])
                continue
            entries.append({
                "id": a["id"], "title": a["title"], "artist": a.get("artist"),
                "year": a.get("creation_date"), "culture": a.get("culture"),
                "type": a.get("type"), "img_url": a.get("image_url"),
                # Verbatim human-written sources, retained for auditing.
                "source_description": a["description"],
                "source_did_you_know": a["did_you_know"],
                "source": a["source"], "record_url": a["record_url"],
                "generator_model": model,
                **qa,
            })
        OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    n_q = sum(1 for e in entries for k in e if k.endswith("_question"))
    print(f"\nwrote {len(entries)} artworks, {n_q} questions -> {OUT}")
    from collections import Counter
    c = Counter(k[:-9] for e in entries for k in e if k.endswith("_question"))
    for t in TYPES:
        print(f"  {t:12s}: {c.get(t, 0)}")
    if failures:
        print(f"  {len(failures)} artworks produced no usable questions")


if __name__ == "__main__":
    main()
