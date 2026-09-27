"""Gold artists and scoring of the memorisation probe (ArtPedia-VQA).

ArtPedia has no artist field. Two subcommands, both with microsoft/phi-4:

  extract-artists  Recover the artist of each test painting from its
                   contextual sentences, cross-checked with the Wikimedia file
                   name in img_url. Stores a confidence and the supporting
                   sentence (results/jsonl/gold_artists.jsonl).

  judge-probe      Score the title/artist guesses of obscure_probe.py against
                   the gold (alternative titles and name variants count as
                   correct) and set `is_obscure` when both are wrong
                   (results/jsonl/obscure.jsonl).

Needs a GPU. Resumable.

Usage:
    python rebuttal_experiments/artpedia_vqa/ground_truth.py extract-artists
    python rebuttal_experiments/artpedia_vqa/ground_truth.py judge-probe
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import common  # noqa: E402,F401  (sets up sys.path)

import json
import os
import re
import sys
from pathlib import Path


import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "analysis"))
import lib  # noqa: E402

DEFAULT_MODEL = lib.JUDGE_PHI4

ARTIST_SYSTEM_PROMPT = (
    "You extract structured metadata about paintings from encyclopedia "
    "text. You will be given a painting's title, the filename of its image, "
    "and a set of sentences from its Wikipedia article.\n\n"
    "Identify the artist who painted it, as stated in the given material. "
    "Rules:\n"
    "* Report the artist's name in its normal full form (e.g. \"Vincent van "
    "Gogh\", \"Duccio di Buoninsegna\"), not a slug or an abbreviation.\n"
    "* The title and the image filename often contain the artist's name — "
    "use them, but prefer what the sentences state explicitly when they "
    "disagree.\n"
    "* If the work is explicitly described as by an unknown, anonymous, or "
    "unidentified artist, report exactly \"UNKNOWN\".\n"
    "* If the material genuinely does not indicate an artist at all, also "
    "report \"UNKNOWN\". Do not use outside knowledge to fill the gap.\n"
    "* Set confidence to \"high\" when a sentence names the artist "
    "directly, \"medium\" when you inferred it from the title or filename, "
    "and \"low\" when you are unsure.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"artist": "<name or UNKNOWN>", "confidence": "<high|medium|low>", '
    '"evidence": "<the short phrase or sentence you took it from>"}'
)

PROBE_SYSTEM_PROMPT = (
    "You are grading whether a vision model correctly identified a "
    "painting. You will be given the true title and true artist of a "
    "painting, and the model's guesses for each.\n\n"
    "Judge the title and the artist independently.\n\n"
    "Count a TITLE as correct if it names the same work, allowing for:\n"
    "* alternative or historical titles for the same painting,\n"
    "* translations between languages (e.g. \"La Primavera\" / \"Spring\"),\n"
    "* the presence or absence of a parenthetical disambiguator, an "
    "artist's name, or a date,\n"
    "* minor differences in wording, articles, spelling, or punctuation.\n"
    "Count it as WRONG if it names a different artwork, or is a generic "
    "description rather than a title (e.g. \"Portrait of a woman\" when the "
    "real title is specific), or is UNKNOWN.\n\n"
    "Count an ARTIST as correct if it names the same person, allowing for:\n"
    "* standard name variants, transliterations, and spellings (e.g. "
    "\"Bruegel\"/\"Brueghel\", \"El Greco\"/\"Doménikos Theotokópoulos\"),\n"
    "* first-name/surname-only forms where the reference is unambiguous,\n"
    "* honorifics, diacritics, and ordering differences.\n"
    "Count it as WRONG if it names a different person, or is UNKNOWN, or is "
    "merely a school/workshop/follower attribution when the true artist is "
    "a named individual.\n"
    "If the true artist is UNKNOWN (the work is anonymous), count the "
    "model's artist as correct only if it also says the artist is unknown "
    "or anonymous.\n\n"
    "Be reasonably lenient: the question is whether the model knows the "
    "work, not whether it matched a string.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"title_correct": <true|false>, "artist_correct": <true|false>, '
    '"justification": "<one brief sentence>"}'
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


def build_llm(model, gpu_memory_utilization, max_model_len):
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams

    path = snapshot_download(model)
    print(f"model: {model} @ {Path(path).name}")
    llm = LLM(model=path, dtype="bfloat16",
              gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len, seed=lib.SEED)
    return llm, llm.get_tokenizer(), SamplingParams(
        temperature=0.0, max_tokens=256, seed=lib.SEED
    ), Path(path).name


def run_batched(llm, tokenizer, params, system_prompt, prompts, batch_size=64):
    from tqdm import tqdm

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    out = []
    for i in tqdm(range(0, len(rendered), batch_size), desc="generating"):
        outs = llm.generate(rendered[i : i + batch_size], params, use_tqdm=False)
        out.extend(o.outputs[0].text for o in outs)
    return out


@click.group()
def cli():
    pass


@cli.command("extract-artists")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--gpu-memory-utilization", default=0.90, type=float, show_default=True)
@click.option("--max-model-len", default=8192, show_default=True)
@click.option("--batch-size", default=32, show_default=True)
def extract_artists(model, gpu_memory_utilization, max_model_len, batch_size):
    out_path = lib.JSONL_DIR / "gold_artists.jsonl"
    rows = lib.read_jsonl(out_path)
    done = {r["painting_id"] for r in rows}

    test = common.load_artpedia_test_split()
    todo = [(pid, e) for pid, e in sorted(test.items()) if pid not in done]
    if not todo:
        print("nothing to do")
        return
    print(f"extracting gold artist for {len(todo)} paintings")

    llm, tok, params, revision = build_llm(model, gpu_memory_utilization, max_model_len)

    prompts = []
    for pid, e in todo:
        fname = (e.get("img_url") or "").rsplit("/", 1)[-1]
        ctx = " ".join(e.get("contextual_sentences", []))[:4000]
        prompts.append(
            f"Painting title: {e['title']}\n"
            f"Image filename: {fname}\n"
            f"Year: {e.get('year')}\n\n"
            f"Sentences from the Wikipedia article:\n{ctx}"
        )

    texts = run_batched(llm, tok, params, ARTIST_SYSTEM_PROMPT, prompts, batch_size)
    for (pid, e), text in zip(todo, texts):
        parsed = extract_json(text) or {}
        artist = str(parsed.get("artist", "")).strip()
        rows.append({
            "painting_id": pid,
            "title": e["title"],
            "year": e.get("year"),
            "img_url": e.get("img_url"),
            "gold_artist": artist or None,
            "gold_artist_unknown": artist.upper() == "UNKNOWN",
            "extraction_confidence": str(parsed.get("confidence", "")).lower() or None,
            "extraction_evidence": str(parsed.get("evidence", ""))[:300],
            "extractor_model": model,
            "extractor_revision": revision,
            "parse_ok": bool(parsed),
        })
    lib.write_jsonl(out_path, rows)
    n_unknown = sum(1 for r in rows if r.get("gold_artist_unknown"))
    print(f"wrote {len(rows)} rows -> {out_path} ({n_unknown} anonymous works)")


@cli.command("judge-probe")
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--gpu-memory-utilization", default=0.90, type=float, show_default=True)
@click.option("--max-model-len", default=4096, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
def judge_probe(model, gpu_memory_utilization, max_model_len, batch_size):
    probe = {r["painting_id"]: r for r in lib.read_jsonl(lib.JSONL_DIR / "obscure_probe.jsonl")}
    gold = {r["painting_id"]: r for r in lib.read_jsonl(lib.JSONL_DIR / "gold_artists.jsonl")}
    if not probe:
        raise SystemExit("no obscure_probe.jsonl — run obscure_probe.py first")
    if not gold:
        raise SystemExit("no gold_artists.jsonl — run `extract-artists` first")

    out_path = lib.JSONL_DIR / "obscure.jsonl"
    rows = lib.read_jsonl(out_path)
    done = {r["painting_id"] for r in rows}

    todo = [p for pid, p in sorted(probe.items())
            if pid not in done and "error" not in p and pid in gold]
    if not todo:
        print("nothing to do")
        return
    print(f"judging {len(todo)} identification attempts")

    llm, tok, params, revision = build_llm(model, gpu_memory_utilization, max_model_len)

    prompts = []
    for p in todo:
        g = gold[p["painting_id"]]
        prompts.append(
            f"True title: {g['title']}\n"
            f"True artist: {g.get('gold_artist') or 'UNKNOWN'}\n"
            f"Year: {g.get('year')}\n\n"
            f"Model's guessed title: {p.get('pred_title') or 'UNKNOWN'}\n"
            f"Model's guessed artist: {p.get('pred_artist') or 'UNKNOWN'}"
        )

    texts = run_batched(llm, tok, params, PROBE_SYSTEM_PROMPT, prompts, batch_size)
    for p, text in zip(todo, texts):
        g = gold[p["painting_id"]]
        parsed = extract_json(text) or {}
        tc = bool(parsed.get("title_correct", False))
        ac = bool(parsed.get("artist_correct", False))
        rows.append({
            "painting_id": p["painting_id"],
            "title": g["title"],
            "gold_artist": g.get("gold_artist"),
            "pred_title": p.get("pred_title"),
            "pred_artist": p.get("pred_artist"),
            "title_correct": tc,
            "artist_correct": ac,
            # Obscure == the backbone knows neither what it is nor who made
            # it, so any correct answer downstream came from retrieval.
            "is_obscure": (not tc) and (not ac),
            "justification": str(parsed.get("justification", ""))[:300],
            "judge_model": model,
            "judge_revision": revision,
            "parse_ok": bool(parsed),
        })

    lib.write_jsonl(out_path, rows)
    n_obs = sum(1 for r in rows if r["is_obscure"])
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"obscure: {n_obs}/{len(rows)} ({n_obs / len(rows):.1%})")
    print(f"title top-1:  {sum(r['title_correct'] for r in rows) / len(rows):.1%}")
    print(f"artist top-1: {sum(r['artist_correct'] for r in rows) / len(rows):.1%}")


if __name__ == "__main__":
    cli()
