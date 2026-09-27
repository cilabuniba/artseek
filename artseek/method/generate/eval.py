"""Captioning metrics for the artwork-explanation experiments (ArtPedia,
SemArt v2.0 / ExplainMe, PaintingForm).

Reads <out_dir>/preds.json written by `test.py inference`, applies the paper's
normalisation (one sentence per aspect for ArtPedia, every aspect sentence for
SemArt v2.0, the whole paragraph for PaintingForm) and computes BLEU@1-4,
METEOR, ROUGE-L, CIDEr and SPICE. Scores are printed and saved to
<out_dir>/metrics.json. METEOR and SPICE need Java.

Usage:
    python -m artseek.method.generate.eval score --config-path models/configs/generate/artpedia_short.yaml
"""

import json
import re
import string
import unicodedata
from pathlib import Path

import click
import nltk
import yaml
from hydra.utils import instantiate
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

from ...data.datasets.artpedia import ArtpediaDataset
from ...data.datasets.explain_me import ExplainMeDataset
from ...data.datasets.painting_form import PaintingFormDataset
from .metrics import Evaluator


def _ensure_nltk_data():
    for resource, package in (("tokenizers/punkt", "punkt"),
                              ("tokenizers/punkt_tab", "punkt_tab")):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(package, quiet=True)


@click.group()
def cli():
    pass


@cli.command("score")
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the config file.",
)
@click.option("--no-spice", is_flag=True,
              help="Skip SPICE (slow on long texts; the paper skips it for PaintingForm).")
def score(config_path, no_spice):
    """Compute captioning metrics for the predictions of a config."""
    _ensure_nltk_data()
    config = load_config(config_path)
    preds = load_predictions(config.out_dir)
    messages = extract_messages(preds)

    ds = config.dataset
    if isinstance(ds, ExplainMeDataset):
        jsons = [extract_json_from_message(message["content"]) for message in messages]
        hypotheses, references = process_explain_me_dataset(ds, jsons)
    elif isinstance(ds, ArtpediaDataset):
        jsons = [extract_json_from_message(message["content"]) for message in messages]
        hypotheses, references = process_artpedia_dataset(ds, jsons)
    elif isinstance(ds, PaintingFormDataset):
        hypotheses, references = process_painting_form_dataset(ds, messages)
    else:
        raise click.UsageError(f"Unsupported dataset: {type(ds).__name__}")

    hypotheses = filter_hypotheses_by_reference_vocab(hypotheses, references)

    report = evaluate_results(references, hypotheses, with_spice=not no_spice)
    out = Path(config.out_dir) / "metrics.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


# Former name of the command.
cli.add_command(score, name="pred-message-to-str")


def filter_hypotheses_by_reference_vocab(hypotheses, references):
    # Step 1: Build vocabulary from all reference sentences
    vocab = set()
    for refs in references:
        for ref in refs:
            tokens = word_tokenize(ref.lower())
            vocab.update(tokens)

    # Step 2: Filter each hypothesis to keep only tokens in the vocab
    detokenizer = TreebankWordDetokenizer()
    filtered_hyps = []
    for hyp in hypotheses:
        tokens = word_tokenize(hyp.lower())
        filtered_tokens = [tok for tok in tokens if tok in vocab]
        detok = detokenizer.detokenize(filtered_tokens)
        filtered_hyps.append(detok)

    return filtered_hyps


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return instantiate(config_dict)


def load_predictions(out_dir):
    preds_path = Path(out_dir) / "preds.json"
    with open(preds_path, "r") as f:
        return json.load(f)


def extract_messages(preds):
    return [pred["messages"][-1] for pred in preds]


def extract_json_from_message(content):
    if not isinstance(content, str):
        content = content.get("content", "")
    stack, start = [], -1

    for i, char in enumerate(content):
        if char == "{":
            if not stack:
                start = i
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack:
                    json_str = content[start : i + 1]
                    return parse_json(json_str)
    return parse_json(content[start:]) if start != -1 else None


def parse_json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return parse_partial_json(json_str)


def parse_partial_json(json_str):
    try:
        decoder = json.JSONDecoder(strict=False)
        obj, _ = decoder.raw_decode(json_str)
        return obj
    except json.JSONDecodeError:
        return extract_key_value_pairs(json_str)


def extract_key_value_pairs(json_str):
    try:
        matches = re.findall(
            r'"(.*?)"\s*:\s*("(.*?)"|[\d.]+|true|false|null)', json_str
        )
        return {
            key: (
                json.loads(value)
                if value in {"true", "false", "null"}
                else value.strip('"')
            )
            for key, value, _ in matches
        }
    except Exception:
        return {"error": "incomplete_json", "data": json_str}


# dimension-level evaluation - first sentence of aspect vs aspect references (KALE)
def process_explain_me_dataset(dataset, jsons):
    hypotheses, references = [], []
    for i, data in enumerate(dataset):
        if i >= len(jsons):
            break
        if jsons[i] is None:
            continue
        for field in ["content", "form", "context"]:
            if field not in jsons[i]:
                continue
            hypothesis = preprocess_text(jsons[i][field])
            reference = preprocess_references(data[field])
            if reference:
                for i, hyp in enumerate(hypothesis):
                    references.append(reference + [f"{field}_{i}"])
                    hypotheses.append(hyp)
    return hypotheses, references


# dimension-level evaluation - first sentence of aspect vs aspect references (KALE)
def process_artpedia_dataset(dataset, jsons):
    hypotheses, references = [], []
    for i, data in enumerate(dataset):
        if i >= len(jsons):
            break
        if jsons[i] is None:
            continue
        for field in ["content", "context"]:
            if field not in jsons[i]:
                continue
            hypothesis = preprocess_text(jsons[i][field])[0]
            reference = preprocess_references(data[field])
            if reference:
                references.append(reference)
                hypotheses.append(hypothesis)
    return hypotheses, references


# for the painting form dataset, we only take the hypothesis after </think>
# and the references are the two formal analyses from Gemini and GPT
def process_painting_form_dataset(dataset, messages):
    hypotheses, references = [], []
    for i, data in enumerate(dataset):
        if not messages[i]:
            continue
        # hypothesis is what the model wrote after </think>
        hypothesis = messages[i]["content"].split("</think>")[-1].strip()
        hypothesis = " ".join(preprocess_text(hypothesis))
        reference = preprocess_references(
            [data["formal_analysis_gemini"], data["formal_analysis_gpt"]]
        )
        if reference:
            references.append(reference)
            hypotheses.append(hypothesis)
    return hypotheses, references


def preprocess_text(text):
    text = text.lower().strip()
    text = normalize_accents(text)
    text = re.sub(r"\s+", " ", text).strip()
    # Use nltk to split into sentences intelligently
    sentences = nltk.sent_tokenize(text)
    return sentences


def preprocess_references(references):
    if not references:
        return None
    references = [x.lower().strip() for x in references]
    references = [normalize_accents(x) for x in references]
    # if there are multiple periods, replace them with a single period
    references = [re.sub(r"\.{2,}", "", x) for x in references]
    # if there are periods in the reference not followed by a space, add a space
    references = [re.sub(r"\.(?!\s)", ". ", x) for x in references]
    references = [re.sub(r"\s+", " ", x).strip() for x in references]
    return references


def normalize_accents(text):
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")


def build_vocab(references):
    vocab = set()
    table = str.maketrans("", "", string.punctuation)
    for ref_list in references:
        for ref in ref_list:
            # Remove punctuation before splitting into words
            cleaned = ref.translate(table)
            vocab.update(cleaned.split())
    return vocab


def filter_vocab(hypotheses, references, vocab):
    table = str.maketrans("", "", string.punctuation)

    def keep_word(word):
        return word.translate(table) in vocab

    filtered_hypotheses = [
        " ".join([word for word in h.split() if keep_word(word)])
        for h in hypotheses
    ]
    filtered_references = [
        [
            " ".join([word for word in r.split() if keep_word(word)])
            for r in ref_list
        ]
        for ref_list in references
    ]
    filtered_hypotheses = [
        truncate_caption(h, max_words=30) for h in filtered_hypotheses
    ]
    return filtered_hypotheses, filtered_references


def truncate_caption(caption, max_words=30):
    words = caption.strip().split()
    truncated = words[:max_words]
    result = " ".join(truncated)
    if caption.strip().endswith(".") and not result.endswith("."):
        result += "."
    return result


def evaluate_results(references, hypotheses, with_spice=True):
    evaluator = Evaluator(with_spice=with_spice)
    evaluator.do_the_thing(references, hypotheses)
    report = {k: float(v) for k, v in evaluator.evaluation_report.items()}
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    cli()
