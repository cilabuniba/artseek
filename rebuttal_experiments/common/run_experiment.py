"""Run one system variant over a VQA benchmark and record answers and timings.

Every (image, question) pair is an independent single-turn conversation.
Results are saved after every item, so an interrupted run resumes where it
stopped.

Variants:
    base                      Qwen2.5-VL-32B-AWQ alone (no LICN, no retrieval).
    base_gemma3               Gemma 3 27B alone.
    base_mistral3             Mistral Small 3.1 24B (w4a16) alone.
    full_classify             ArtSeek: artwork card + retrieval, tool-calling
                              policy taught by the one-shot example.
    full_noclassify           ArtSeek without the artwork card.
    full_systemprompt         ArtSeek with the policy described in the system
                              prompt instead of the one-shot example.
    full_alwaysretrieve       ArtSeek, at least one retrieval is forced.
    full_singleretrieve       ArtSeek, exactly one retrieval, then answer.
    full_classify_noartist    ArtSeek, artist field removed from the card.
    full_classify_notags      ArtSeek, tag field removed from the card.
    full_classify_reliable    ArtSeek, artist and tag fields removed.
    full_classify_gated       ArtSeek, artist shown only if confidence >= 0.6.
    full_seeded               ArtSeek with seeded retrieval (Qwen backbone).
    full_gemma3               ArtSeek with seeded retrieval on Gemma 3.
    full_mistral3             ArtSeek with seeded retrieval on Mistral Small 3.1.

Seeded retrieval runs before the model's first turn (queries: the card's artist
if its confidence is >= 0.8, and the question; 20 documents each, merged by
score, top 10 kept) and allows no further tool calls. It is needed for backbones
that cannot emit Qwen-style tool calls; full_seeded is the matched Qwen run.

All variants except base* need a running Qdrant server (QDRANT_URL).

Usage:
    python rebuttal_experiments/common/run_experiment.py --variant full_classify
    python rebuttal_experiments/common/run_experiment.py --variant base \\
        --vqa-path rebuttal_experiments/aqua/data/aqua_vqa.json \\
        --out-dir rebuttal_experiments/aqua/results
"""

import common  # noqa: F401  (sets up sys.path)

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import click
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from tqdm import tqdm

VARIANTS = (
    "base",
    "base_gemma3",
    "base_mistral3",
    "full_classify",
    "full_noclassify",
    "full_systemprompt",
    "full_alwaysretrieve",
    "full_singleretrieve",
    "full_classify_noartist",
    "full_classify_notags",
    "full_classify_reliable",
    "full_classify_gated",
    "full_seeded",
    "full_gemma3",
    "full_mistral3",
)
SEEDED_VARIANTS = ("full_seeded", "full_gemma3", "full_mistral3")

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about paintings and "
    "artworks based on the provided image. Enclose your reasoning process "
    "within <think></think> XML tags, then give your final answer."
)

# With seeded retrieval the documents are already in the context and no tool
# is available, so the prompt must not mention the tool.
SEEDED_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about "
    "paintings and artworks.\n\n# Rules\n"
    "* Relevant documents have ALREADY been retrieved for you and "
    "appear in the conversation. You have no tools and cannot search; "
    "do not announce or attempt a search.\n"
    "* Answer the question directly, using the image and those "
    "documents. Cite a document when you rely on it.\n"
    "* If the documents do not contain the answer, answer from your "
    "own knowledge and say so.\n"
    "* Enclose your reasoning within <think></think> XML tags, then "
    "give your final answer."
)

# Settings shared by the bare backbones.
BASE_VLLM_KWARGS = {
    "gpu_memory_utilization": 0.85,
    "max_model_len": 16384,
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"image": 8},
}


def question_types(vqa: list[dict]) -> tuple[str, ...]:
    """Question classes of a benchmark (keys `<type>_question`), in order."""
    seen = []
    for item in vqa:
        for key in item:
            if key.endswith("_question"):
                qt = key[: -len("_question")]
                if qt not in seen:
                    seen.append(qt)
    return tuple(seen)


def build_base_model(variant: str):
    """A bare backbone served with vLLM."""
    from huggingface_hub import snapshot_download

    if variant == "base_gemma3":
        from artseek.method.generate.gemma3_vllm import GEMMA3_MODEL_ID, Gemma3VLLMChatModel

        return Gemma3VLLMChatModel.from_pretrained(
            snapshot_download(GEMMA3_MODEL_ID),
            dtype="bfloat16",
            tensor_parallel_size=2,
            # vLLM's custom all-reduce kernel failed at startup on our nodes.
            disable_custom_all_reduce=True,
            **BASE_VLLM_KWARGS,
        )
    if variant == "base_mistral3":
        from artseek.method.generate.mistral3_vllm import (
            MISTRAL3_MODEL_ID,
            Mistral3VLLMChatModel,
        )

        return Mistral3VLLMChatModel.from_pretrained(
            snapshot_download(MISTRAL3_MODEL_ID), **BASE_VLLM_KWARGS
        )

    from artseek.method.generate.config import MLLM_ID
    from artseek.method.generate.qwen2_5_vl_vllm import Qwen2_5_VLVLLMChatModel

    return Qwen2_5_VLVLLMChatModel.from_pretrained(
        snapshot_download(MLLM_ID),
        dtype="bfloat16",
        mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
        **BASE_VLLM_KWARGS,
    )


def build_full_model(variant: str):
    """The ArtSeek pipeline for a full_* variant."""
    from huggingface_hub import snapshot_download

    from artseek.method.generate.config import load_artseek

    if variant == "full_mistral3":
        from artseek.method.generate.mistral3_vllm import (
            MISTRAL3_MODEL_ID,
            Mistral3VLLMChatModel,
        )

        return load_artseek(
            model_pretrained_model_name_or_path=snapshot_download(MISTRAL3_MODEL_ID),
            chat_model_cls=Mistral3VLLMChatModel,
            model_kwargs={
                "gpu_memory_utilization": 0.65,
                "max_model_len": 16384,
                "max_num_seqs": 4,
                "limit_mm_per_prompt": {"image": 128},
            },
            shot_path=None,
            use_shot=False,
        )
    if variant == "full_gemma3":
        from artseek.method.generate.gemma3_vllm import GEMMA3_MODEL_ID, Gemma3VLLMChatModel

        return load_artseek(
            model_pretrained_model_name_or_path=snapshot_download(GEMMA3_MODEL_ID),
            chat_model_cls=Gemma3VLLMChatModel,
            model_kwargs={
                "dtype": "bfloat16",
                "tensor_parallel_size": 2,
                "disable_custom_all_reduce": True,
                "gpu_memory_utilization": 0.60,
                "max_model_len": 16384,
                "max_num_seqs": 4,
                "limit_mm_per_prompt": {"image": 32},
            },
            shot_path=None,
            use_shot=False,
        )
    use_shot = variant != "full_systemprompt" and variant not in SEEDED_VARIANTS
    return load_artseek(engine="vllm", use_shot=use_shot)


def error_record(item: dict, variant: str, qtype: str, error: str) -> dict:
    return {
        "id": item["id"],
        "title": item["title"],
        "variant": variant,
        "question_type": qtype,
        "error": error,
    }


def run_base(vqa: list[dict], out_path: Path, max_new_tokens: int,
             limit: int | None, variant: str = "base"):
    qtypes = question_types(vqa)
    model = build_base_model(variant)
    tokenizer = model.processor.tokenizer

    results = common.load_existing_results(out_path)
    done = {(r["id"], r["question_type"]) for r in results}

    items = vqa[:limit] if limit else vqa
    for item in tqdm(items, desc=variant):
        item_qtypes = [q for q in qtypes if f"{q}_question" in item]
        if all((item["id"], qtype) in done for qtype in item_qtypes):
            continue

        image = common.load_image_or_none(item["id"])
        if image is None:
            for qtype in item_qtypes:
                if (item["id"], qtype) not in done:
                    results.append(error_record(
                        item, variant, qtype, "image not downloaded (dead source URL)"))
            common.save_results(out_path, results)
            continue

        for qtype in item_qtypes:
            if (item["id"], qtype) in done:
                continue
            question = item[f"{qtype}_question"]
            user_msg_content = [
                {"type": "text", "text": "# Artwork image\n"},
                {"type": "image"},
                {"type": "text", "text": f"\n# Question\n{question}"},
            ]
            try:
                response, duration = common.timed(
                    model.invoke,
                    [SystemMessage(BASE_SYSTEM_PROMPT), HumanMessage(content=user_msg_content)],
                    images=[image],
                    max_new_tokens=max_new_tokens,
                )
                reasoning, answer = common.parse_ai_content(response.content)
                reasoning_dur, answer_dur = common.split_reasoning_duration(
                    response.content, duration, tokenizer
                )
                results.append({
                    "id": item["id"],
                    "title": item["title"],
                    "variant": variant,
                    "question_type": qtype,
                    "question": question,
                    "reference_answer": item[f"{qtype}_answer"],
                    "model_answer": answer,
                    "model_reasoning": reasoning,
                    "num_tool_calls": 0,
                    "retrieved_doc_ids": [],
                    "timing": {
                        "classification": 0.0,
                        "generation_total": duration,
                        "reasoning": reasoning_dur,
                        "answer": answer_dur,
                        "retrieval": 0.0,
                        "total": duration,
                    },
                })
            except Exception as e:  # noqa: BLE001
                results.append(error_record(item, variant, qtype, str(e)))
            common.save_results(out_path, results)


def seed_queries_for(card: dict | None, question: str) -> list[dict]:
    """Seed queries: the card's artist when LICN is confident, and the question."""
    qs = []
    preds = (card or {}).get("artist") or []
    if preds:
        top = preds[0][1]
        conf = top.item() if hasattr(top, "item") else float(top)
        if conf >= 0.80:
            qs.append(str(preds[0][0]).replace("-", " "))
    qs.append(question)
    return [{"query": q, "requires_image": True}
            for q in dict.fromkeys(q for q in qs if q.strip())]


def run_full(variant: str, vqa: list[dict], out_path: Path, max_new_tokens: int,
             limit: int | None):
    qtypes = question_types(vqa)
    do_classify = variant != "full_noclassify"
    # Card fields hidden from the model (the full prediction is still saved).
    omit_fields = {
        "full_classify_noartist": {"artist"},
        "full_classify_notags": {"tag"},
        "full_classify_reliable": {"artist", "tag"},
    }.get(variant, set())
    artist_conf_gate = 0.6 if variant == "full_classify_gated" else None
    force_retrieve = variant in ("full_alwaysretrieve", "full_singleretrieve")
    call_cap = 1 if variant == "full_singleretrieve" else None
    seeded = variant in SEEDED_VARIANTS
    if seeded:
        call_cap = 0

    model = build_full_model(variant)
    if seeded:
        model.system_prompt_override = SEEDED_SYSTEM_PROMPT
    tokenizer = model.model.processor.tokenizer

    results = common.load_existing_results(out_path)
    done = {(r["id"], r["question_type"]) for r in results}

    items = vqa[:limit] if limit else vqa
    for item in tqdm(items, desc=variant):
        item_qtypes = [q for q in qtypes if f"{q}_question" in item]
        if all((item["id"], qtype) in done for qtype in item_qtypes):
            continue

        image = common.load_image_or_none(item["id"])
        if image is None:
            for qtype in item_qtypes:
                if (item["id"], qtype) not in done:
                    results.append(error_record(
                        item, variant, qtype, "image not downloaded (dead source URL)"))
            common.save_results(out_path, results)
            continue

        card = None
        classify_duration = 0.0
        if do_classify:
            try:
                card, classify_duration = common.timed(model.classify, image)
            except Exception as e:  # noqa: BLE001
                for qtype in item_qtypes:
                    if (item["id"], qtype) not in done:
                        results.append(error_record(
                            item, variant, qtype, f"classify() failed: {e}"))
                common.save_results(out_path, results)
                continue

        for qtype in item_qtypes:
            if (item["id"], qtype) in done:
                continue
            question = item[f"{qtype}_question"]

            card_for_prompt = card
            if card_for_prompt and omit_fields:
                card_for_prompt = {k: v for k, v in card.items() if k not in omit_fields}
            if card_for_prompt and artist_conf_gate is not None:
                preds = card_for_prompt.get("artist") or []
                top = preds[0][1] if preds else 0.0
                conf = top.item() if hasattr(top, "item") else float(top)
                if conf < artist_conf_gate:
                    card_for_prompt = {k: v for k, v in card_for_prompt.items() if k != "artist"}
            user_msg = common.build_first_user_message(
                question, card_for_prompt if do_classify else None
            )

            try:
                new_messages, _ = model.chat_turn(
                    messages=[user_msg],
                    input_image=image,
                    classify=do_classify,
                    retrieve=True,
                    context_images=[],
                    max_new_tokens=max_new_tokens,
                    force_first_tool_call=force_retrieve,
                    max_tool_calls=call_cap,
                    seed_queries=seed_queries_for(card, question) if seeded else None,
                    seed_limit=20 if seeded else 10,
                    seed_merge_top_k=10 if seeded else None,
                    # Gemma's and Mistral's chat templates have no `tool` role.
                    seed_context_into_user=variant in ("full_mistral3", "full_gemma3"),
                )

                final_ai_msg = next(
                    m for m in reversed(new_messages) if isinstance(m, AIMessage)
                )
                reasoning, answer = common.parse_ai_content(final_ai_msg.content)

                step_durations = final_ai_msg.response_metadata.get("step_durations", [])
                total_duration = final_ai_msg.response_metadata.get(
                    "total_duration", sum(s["duration"] for s in step_durations)
                )
                generation_total = sum(
                    s["duration"] for s in step_durations if s["type"] == "generation"
                )
                retrieval_total = sum(
                    s["duration"] for s in step_durations if s["type"] == "tool"
                )
                num_tool_calls = sum(1 for s in step_durations if s["type"] == "tool")

                retrieved_doc_ids = []
                retrieval_stages = []
                for m in new_messages:
                    if isinstance(m, ToolMessage) and m.artifact:
                        retrieved_doc_ids.extend(m.artifact.get("context_idxs", []))
                        retrieval_stages.append({
                            "embed": m.artifact.get("embed_duration"),
                            "search": m.artifact.get("search_duration"),
                            "total": m.artifact.get("duration"),
                        })

                # The reasoning/answer split refers to the final generation step.
                last_gen_duration = next(
                    (s["duration"] for s in reversed(step_durations) if s["type"] == "generation"),
                    0.0,
                )
                reasoning_dur, answer_dur = common.split_reasoning_duration(
                    final_ai_msg.content, last_gen_duration, tokenizer
                )

                results.append({
                    "id": item["id"],
                    "title": item["title"],
                    "variant": variant,
                    "question_type": qtype,
                    "question": question,
                    "reference_answer": item[f"{qtype}_answer"],
                    "model_answer": answer,
                    "model_reasoning": reasoning,
                    "card": common.serialize_card(card) if do_classify else None,
                    "num_tool_calls": num_tool_calls,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    # For forced retrieval: "none_needed", "nudge_1", "nudge_2"
                    # or "synthetic" (see Qwen2_5_VLRAGModel._force_tool_call).
                    "force_mechanism": final_ai_msg.response_metadata.get(
                        "force_mechanism", "not_forced"
                    ),
                    "retrieval_stages": retrieval_stages,
                    "timing": {
                        "classification": classify_duration,
                        "generation_total": generation_total,
                        "reasoning": reasoning_dur,
                        "answer": answer_dur,
                        "retrieval": retrieval_total,
                        "total": classify_duration + total_duration,
                    },
                })
            except Exception as e:  # noqa: BLE001
                results.append(error_record(item, variant, qtype, str(e)))
            common.save_results(out_path, results)


@click.command()
@click.option("--variant", type=click.Choice(VARIANTS), required=True)
@click.option("--vqa-path", default=str(common.VQA_DATASET_PATH), show_default=True)
@click.option("--out-dir", default=str(common.RESULTS_DIR), show_default=True)
@click.option("--max-new-tokens", default=512, show_default=True)
@click.option("--limit", type=int, default=None,
              help="Only process the first N paintings (quick trial).")
def main(variant: str, vqa_path: str, out_dir: str, max_new_tokens: int,
         limit: int | None):
    vqa = common.load_vqa_dataset(vqa_path)
    out_path = Path(out_dir) / f"{variant}.json"

    if variant.startswith("base"):
        run_base(vqa, out_path, max_new_tokens, limit, variant=variant)
    else:
        run_full(variant, vqa, out_path, max_new_tokens, limit)

    print(f"Done. Results saved to {out_path}.")


if __name__ == "__main__":
    main()
