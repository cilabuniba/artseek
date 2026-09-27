"""LLM-as-a-judge with microsoft/phi-4 (served with vLLM, greedy).

Two passes:

  --pass answer    recall (0.0-1.0) and correctness (0/1/2) of the model's
                   answer against the reference answer. We used `--rubric
                   artpedia` for ArtPedia-VQA and `--rubric aqua` (written for
                   short references) for AQUA, ArtQuest, LICNHeldOut and
                   ArtCurate-AIC.
  --pass evidence  whether the retrieved fragments contain what is needed to
                   produce the reference answer: evidence_recall (0.0-1.0) and
                   evidence_present (0/1/2). Needs analysis/build_fragment_cache.py.

Judgments are keyed by (judge_model, variant, query_id) and appended to
results/jsonl/judge_{answer,evidence}.jsonl, so a re-run only judges what is
missing. Run analysis/build_runs_jsonl.py first. Set ARTSEEK_EXP_DIR to the
benchmark folder (default: artpedia_vqa).

Usage:
    python rebuttal_experiments/common/judge.py --pass answer --variant base --variant full_classify
    python rebuttal_experiments/common/judge.py --pass evidence --variant full_classify
"""

import common  # noqa: F401  (sets up sys.path)

import json
import os
import re
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import lib  # noqa: E402

DEFAULT_MODEL = "microsoft/phi-4"
SEED = lib.SEED

# ── answer rubric (long reference answers) ───────────────────────────────
JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator grading an AI assistant's answers to "
    "questions about paintings. You will be given a question, a reference "
    "answer (treat it as ground truth), and the assistant's answer to "
    "evaluate.\n\n"
    "Score the assistant's answer on two dimensions:\n\n"
    "1. RECALL (a number between 0.0 and 1.0): First identify the specific "
    "factual claims in the reference answer (names of people, places, "
    "dates, specific visual details, specific artistic influences, or "
    "other concrete facts — not filler words). Then determine what "
    "fraction of those specific facts are correctly and unambiguously "
    "conveyed in the assistant's answer, allowing for different "
    "wording/paraphrasing as long as the meaning is preserved. A fact only "
    "counts as recalled if the assistant's answer states it correctly, not "
    "just related or nearby information.\n"
    "   - 1.0 = every specific fact from the reference is correctly present.\n"
    "   - 0.0 = none of the specific facts from the reference are present.\n"
    "   - Use values in between for partial recall.\n\n"
    "2. CORRECTNESS (an integer 0, 1, or 2):\n"
    "   - 0 = WRONG: the assistant's answer contradicts the reference "
    "answer's key facts, gives a substantially different or false claim "
    "(e.g. wrong artist, wrong museum, wrong date, wrong description of "
    "what's depicted), or is otherwise not a valid answer to the question.\n"
    "   - 1 = PARTIALLY CORRECT: the assistant's answer gets some of the "
    "key facts right but misses others, is vague or hedged where the "
    "reference is specific, or mixes correct and incorrect claims.\n"
    "   - 2 = CORRECT: the assistant's answer correctly conveys the key "
    "facts of the reference answer, even if phrased differently, includes "
    "extra non-contradictory detail, or differs in verbosity.\n\n"
    "Guidelines:\n"
    "* Judge factual alignment with the reference, not writing style or length.\n"
    "* Extra correct elaboration beyond the reference should NOT be penalized.\n"
    "* Hallucinated specifics that contradict the reference (wrong names, "
    "places, dates) should be penalized even if other parts of the answer "
    "are fine.\n"
    "* If the assistant explicitly says it cannot determine the answer or "
    "lacks sufficient information, treat that as WRONG (0) — the reference "
    "answer demonstrates the information was determinable.\n"
    "* Be strict but fair: don't require verbatim wording, but do require "
    "factual accuracy on the substance.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"recall": <float 0.0-1.0>, "correctness": <0, 1, or 2>, '
    '"justification": "<one brief sentence>"}'
)

# ── short-answer rubric ──────────────────────────────────────────────────
# The references are one word or a short span while the models answer in
# sentences, so the judge grades whether the reference is conveyed. AQUA's
# questions are generated automatically and some are ill-posed, so the judge
# grades against the reference's content.
AQUA_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator grading an AI assistant's answers to "
    "questions about paintings. You will be given a question, a short "
    "reference answer (treat it as ground truth), and the assistant's "
    "answer to evaluate.\n\n"
    "Important context about this data:\n"
    "* The reference answers are SHORT — often a single word or a brief "
    "phrase — while the assistant typically replies in full sentences. "
    "Judge whether the assistant CONVEYS the reference answer, not whether "
    "the two strings look alike. An assistant answer of \"The figure "
    "seated on the couch is a woman\" fully conveys the reference answer "
    "\"woman\" and is CORRECT.\n"
    "* The questions were generated automatically and some are awkward, "
    "truncated, or ungrammatical. Grade against the reference answer's "
    "content; do not penalise the assistant for a badly-posed question.\n\n"
    "Score the assistant's answer on two dimensions:\n\n"
    "1. RECALL (a number between 0.0 and 1.0): the fraction of the "
    "reference answer's content that is correctly conveyed by the "
    "assistant, allowing any wording. For a one-word reference this is "
    "effectively 1.0 if that thing is correctly identified and 0.0 if it "
    "is not. For a longer reference, use intermediate values when only "
    "part of it is conveyed.\n\n"
    "2. CORRECTNESS (an integer 0, 1, or 2):\n"
    "   - 0 = WRONG: the assistant identifies something different from the "
    "reference, contradicts it, or does not answer.\n"
    "   - 1 = PARTIALLY CORRECT: the assistant conveys part of the "
    "reference, or gives a related but less specific term (e.g. "
    "\"person\" where the reference says \"woman\"), or hedges between "
    "the right answer and a wrong one.\n"
    "   - 2 = CORRECT: the assistant conveys the reference answer, in any "
    "wording and at any length.\n\n"
    "Guidelines:\n"
    "* Do NOT penalise extra detail, explanation, or length. A long answer "
    "containing the right short answer is CORRECT.\n"
    "* Accept synonyms, plurals, and equivalent phrasings.\n"
    "* A MORE specific correct answer than the reference is still correct "
    "(e.g. \"a young woman\" for reference \"woman\").\n"
    "* A LESS specific answer that omits what was asked is partial, not "
    "correct.\n"
    "* If the assistant says it cannot determine the answer, that is "
    "WRONG (0).\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"recall": <float 0.0-1.0>, "correctness": <0, 1, or 2>, '
    '"justification": "<one brief sentence>"}'
)

# ── evidence rubric (same structure as the answer rubric) ────────────────
EVIDENCE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing whether a set of retrieved "
    "reference documents contains the information needed to answer a "
    "question about a painting. You will be given a question, a reference "
    "answer (treat it as ground truth), and the text of the documents that "
    "a retrieval system returned. You are NOT grading any assistant's "
    "answer here — only whether the retrieved text contains the needed "
    "information.\n\n"
    "Score the retrieved documents on two dimensions:\n\n"
    "1. EVIDENCE_RECALL (a number between 0.0 and 1.0): First identify the "
    "specific factual claims in the reference answer (names of people, "
    "places, dates, specific visual details, specific artistic influences, "
    "or other concrete facts — not filler words). Then determine what "
    "fraction of those specific facts appear somewhere in the retrieved "
    "documents, allowing for different wording/paraphrasing as long as the "
    "meaning is preserved. A fact only counts if the documents state it "
    "correctly, not merely mention a related or nearby topic.\n"
    "   - 1.0 = every specific fact from the reference is present somewhere "
    "in the retrieved text.\n"
    "   - 0.0 = none of the specific facts from the reference are present.\n"
    "   - Use values in between for partial coverage.\n\n"
    "2. EVIDENCE_PRESENT (an integer 0, 1, or 2):\n"
    "   - 0 = ABSENT: the retrieved documents do not contain the "
    "information needed to produce the reference answer. They may be about "
    "an unrelated subject, or about the right subject but silent on the "
    "specific point asked.\n"
    "   - 1 = PARTIALLY PRESENT: the documents contain some of what is "
    "needed but not enough to produce the reference answer in full — some "
    "key facts are there and others are missing.\n"
    "   - 2 = FULLY PRESENT: a careful reader could produce the reference "
    "answer's key facts from the retrieved documents alone.\n\n"
    "Guidelines:\n"
    "* Judge only what the documents actually state. Do not credit "
    "information you happen to know but that is not written in them.\n"
    "* The documents are Wikipedia fragments and are often long and mostly "
    "irrelevant; a single relevant sentence buried in an otherwise "
    "unrelated document still counts as present.\n"
    "* Documents mentioning the painting or artist by name but not the "
    "asked-about fact are ABSENT (0) for that fact, not partially present.\n"
    "* Be strict but fair: don't require verbatim wording, but do require "
    "the substance to actually be stated.\n\n"
    "Respond with ONLY a JSON object with this exact shape, nothing else:\n"
    '{"evidence_recall": <float 0.0-1.0>, "evidence_present": <0, 1, or 2>, '
    '"justification": "<one brief sentence>"}'
)

# Retrieved fragments are long: cap each document and the total length.
MAX_DOC_CHARS = 1800
MAX_TOTAL_DOC_CHARS = 24000


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


def valid_answer(p: dict | None) -> bool:
    if not p or "recall" not in p or "correctness" not in p:
        return False
    try:
        r, c = float(p["recall"]), int(p["correctness"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= r <= 1.0 and c in (0, 1, 2)


def valid_evidence(p: dict | None) -> bool:
    if not p or "evidence_recall" not in p or "evidence_present" not in p:
        return False
    try:
        r, c = float(p["evidence_recall"]), int(p["evidence_present"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= r <= 1.0 and c in (0, 1, 2)


def build_answer_prompt(row: dict) -> str:
    return (
        f"Question type: {row['question_type']}\n"
        f"Question: {row['question']}\n\n"
        f"Reference answer: {row['reference_answer']}\n\n"
        f"Assistant's answer to evaluate: {row['model_answer']}"
    )


def build_evidence_prompt(row: dict, docs: list[dict]) -> str:
    parts, total = [], 0
    for i, d in enumerate(docs, start=1):
        text = (d.get("text") or "").strip()
        if len(text) > MAX_DOC_CHARS:
            text = text[:MAX_DOC_CHARS] + " […truncated]"
        chunk = f"## Document {i}: {d.get('title', '?')}\n{text}\n"
        if total + len(chunk) > MAX_TOTAL_DOC_CHARS:
            parts.append(f"\n[{len(docs) - i + 1} further documents omitted for length]")
            break
        parts.append(chunk)
        total += len(chunk)
    return (
        f"Question type: {row['question_type']}\n"
        f"Question: {row['question']}\n\n"
        f"Reference answer: {row['reference_answer']}\n\n"
        f"Retrieved documents:\n\n" + "\n".join(parts)
    )


def load_doc_cache() -> dict:
    path = lib.CACHE_DIR / "fragment_texts.jsonl"
    return {str(r["idx"]): r for r in lib.read_jsonl(path)}


@click.command()
@click.option("--pass", "pass_", type=click.Choice(["answer", "evidence"]), required=True)
@click.option("--rubric", type=click.Choice(["artpedia", "aqua"]), default="artpedia",
              show_default=True,
              help="Answer rubric. 'aqua' grades short reference answers "
                   "against verbose model answers (see AQUA_JUDGE_SYSTEM_PROMPT).")
@click.option("--variant", "variants", multiple=True, required=True)
@click.option("--model", default=DEFAULT_MODEL, show_default=True)
@click.option("--max-new-tokens", default=256, show_default=True)
@click.option("--batch-size", default=64, show_default=True)
@click.option("--gpu-memory-utilization", default=0.90, show_default=True, type=float)
@click.option("--max-model-len", default=16384, show_default=True)
@click.option("--limit", type=int, default=None, help="Pilot mode: first N items per variant.")
def main(pass_, rubric, variants, model, max_new_tokens, batch_size,
         gpu_memory_utilization, max_model_len, limit):
    from huggingface_hub import snapshot_download
    from tqdm import tqdm
    from vllm import LLM, SamplingParams

    model_path = snapshot_download(model)
    revision = Path(model_path).name  # snapshot dir is the commit hash
    print(f"judge model: {model} @ {revision}")

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=SEED,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=SEED)

    out_path = lib.JSONL_DIR / f"judge_{pass_}.jsonl"
    existing = lib.read_jsonl(out_path)
    done = {(r["judge_model"], r["variant"], r["query_id"]) for r in existing}

    runs = lib.load_runs()
    by_variant = {}
    for r in runs:
        by_variant.setdefault(r["variant"], []).append(r)

    doc_cache = load_doc_cache() if pass_ == "evidence" else {}
    if pass_ == "evidence" and not doc_cache:
        raise SystemExit(
            "results/cache/fragment_texts.jsonl is empty — run "
            "analysis/build_fragment_cache.py first."
        )

    if pass_ == "answer":
        system_prompt = (AQUA_JUDGE_SYSTEM_PROMPT if rubric == "aqua"
                         else JUDGE_SYSTEM_PROMPT)
    else:
        system_prompt = EVIDENCE_SYSTEM_PROMPT
    print(f"pass={pass_} rubric={rubric}")
    new_rows, failures = [], []

    for variant in variants:
        rows = by_variant.get(variant)
        if not rows:
            print(f"skip {variant}: no runs")
            continue

        todo = []
        for r in rows:
            if r["status"] != "ok":
                continue
            if pass_ == "answer" and not r.get("model_answer"):
                continue
            if pass_ == "evidence" and not r.get("retrieved_doc_ids"):
                continue  # nothing was retrieved — nothing to judge
            if (model, variant, r["query_id"]) in done:
                continue
            todo.append(r)
        if limit:
            todo = todo[:limit]

        if not todo:
            print(f"{variant}/{pass_}: nothing to do")
            continue
        print(f"{variant}/{pass_}: judging {len(todo)}")

        for start in tqdm(range(0, len(todo), batch_size), desc=f"{variant}/{pass_}"):
            batch = todo[start : start + batch_size]
            prompts, ctx = [], []
            for r in batch:
                if pass_ == "answer":
                    user = build_answer_prompt(r)
                    n_docs = None
                else:
                    seen, docs = set(), []
                    for idx in r["retrieved_doc_ids"]:
                        if idx in seen:
                            continue
                        seen.add(idx)
                        d = doc_cache.get(str(idx))
                        if d:
                            docs.append(d)
                    if not docs:
                        continue
                    user = build_evidence_prompt(r, docs)
                    n_docs = len(docs)
                prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                ctx.append((r, n_docs))

            if not prompts:
                continue
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for (r, n_docs), output in zip(ctx, outputs):
                parsed = extract_json(output.outputs[0].text)
                ok = valid_answer(parsed) if pass_ == "answer" else valid_evidence(parsed)
                if not ok:
                    failures.append((variant, r["query_id"]))
                    continue
                base = {
                    "judge_model": model,
                    "judge_revision": revision,
                    "rubric": rubric,
                    "variant": variant,
                    "query_id": r["query_id"],
                    "painting_id": r["painting_id"],
                    "question_type": r["question_type"],
                    "justification": str(parsed.get("justification", ""))[:400],
                }
                if pass_ == "answer":
                    base["recall"] = float(parsed["recall"])
                    base["correctness"] = int(parsed["correctness"])
                else:
                    base["evidence_recall"] = float(parsed["evidence_recall"])
                    base["evidence_present"] = int(parsed["evidence_present"])
                    base["n_docs_judged"] = n_docs
                    base["num_tool_calls"] = r["num_tool_calls"]
                new_rows.append(base)

            # Flush after every batch, so an interrupted job can resume.
            lib.write_jsonl(out_path, existing + new_rows)

    lib.write_jsonl(out_path, existing + new_rows)
    print(f"wrote {len(new_rows)} new judgments -> {out_path}")
    if failures:
        print(f"{len(failures)} unparseable judge outputs (re-run to retry): {failures[:20]}")


if __name__ == "__main__":
    main()
