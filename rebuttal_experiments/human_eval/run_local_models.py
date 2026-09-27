"""Answers of the local systems for the human evaluation.

Systems:
    base                Qwen2.5-VL-32B-AWQ alone.
    artseek_multiquery  ArtSeek (LICN card + retrieval + one-shot example) with
                        three seeded retrieval queries issued before the first
                        turn: the card's artist (if its confidence is >= 0.80)
                        with the predicted style and genre; style and genre
                        with the model's own <=12-word guess of the subject;
                        and "What is depicted in this painting?". Each query
                        retrieves 20 fragments; the best 15 overall are kept.
                        The model may retrieve further on its own.

Every system gets the same prompt (PROMPT, also used by run_gpt.py). Writes
results/<system>.json. Needs a GPU; artseek_multiquery also needs Qdrant.

Usage:
    python rebuttal_experiments/human_eval/run_local_models.py --system base
    python rebuttal_experiments/human_eval/run_local_models.py --system artseek_multiquery
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

# Four short parts with word caps, so that answers are comparable part by part.
PROMPT = (
    "Look at this painting and answer in four short numbered parts.\n\n"
    "1. SUBJECT - What event, story or scene is depicted? Name it as "
    "specifically as you can. (at most 40 words)\n"
    "2. FIGURES - Who or what are the figures, and which attributes, symbols "
    "or details identify them? (at most 50 words)\n"
    "3. PLACEMENT - What tradition, school and approximate period does the "
    "work belong to, and what in the picture indicates that? (at most 40 "
    "words)\n"
    "4. EVIDENCE - Name up to three specific artworks, artists or sources "
    "your reading rests on, and say in a few words why each is relevant. "
    "(at most 50 words)\n\n"
    "Give your best reading even if you do not recognise the work: commit to "
    "your most likely answer rather than declining. Mark anything you are "
    "unsure of with \"(uncertain)\". Do not refuse and do not answer only "
    "that you cannot identify the work. Keep to the four numbered parts."
)
PROMPT_VERSION = "v3-structured"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about paintings and "
    "artworks based on the provided image. Enclose your reasoning process "
    "within <think></think> XML tags, then give your final answer."
)

MAX_NEW_TOKENS = 700   # the answer plus the <think> block


@click.command()
@click.option("--system", required=True, type=click.Choice(
    ["base", "artseek_multiquery"]))
@click.option("--out-dir", default=str(HERE / "results"), show_default=True)
@click.option("--items-file", default="items.json", show_default=True,
              help="File under data/ listing the items to answer.")
def main(system: str, out_dir: str, items_file: str):
    import common
    import run_experiment as R
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    items = json.loads((HERE / "data" / items_file).read_text())
    out_path = Path(out_dir) / f"{system}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = json.loads(out_path.read_text()) if out_path.exists() else []
    # Only completed answers count as done; error rows are retried.
    done = {r["id"] for r in results if r.get("answer")}
    results = [r for r in results if r.get("answer")]

    do_classify = multiquery = system == "artseek_multiquery"
    model = (R.build_base_model("base") if system == "base"
             else R.build_full_model("full_classify"))

    # A short, title-like guess of the subject, asked before any retrieval.
    HYPOTHESIS_PROMPT = (
        "In at most 12 words, name the subject of this painting as a title "
        "would: the event, story, figure or scene. Reply with the phrase "
        "only, no explanation."
    )
    CARD_TAU = 0.80  # the card's artist is used only above this confidence

    def build_seed_queries(card, hypothesis: str) -> list[dict]:
        """The three seed queries (artist gated at CARD_TAU; style and genre
        ungated)."""
        def top(head):
            v = ((card or {}).get(head) or [["", 0.0]])[0]
            return (v[0] or ""), (v[1] or 0.0)

        artist, conf = top("artist")
        style, _ = top("style")
        genre, _ = top("genre")
        style_genre = " ".join(x for x in (style, genre) if x).strip()

        queries = []
        if artist and conf >= CARD_TAU:
            queries.append(f"{artist.replace('-', ' ')} {style_genre}".strip())
        if hypothesis:
            queries.append(f"{style_genre} {hypothesis}".strip())
        queries.append("What is depicted in this painting?")
        # Deduplicate, preserving order.
        seen, out = set(), []
        for q in queries:
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append({"query": q, "requires_image": True})
        return out

    for it in tqdm(items, desc=system):
        if it["id"] in done:
            continue
        image = Image.open(HERE / it["image"]).convert("RGB")
        rec = {"id": it["id"], "tier": it["tier"], "title": it["title"],
               "system": system, "prompt": PROMPT,
               "prompt_version": PROMPT_VERSION}
        try:
            t0 = time.perf_counter()
            if system == "base":
                content = [
                    {"type": "text", "text": "# Artwork image\n"},
                    {"type": "image"},
                    {"type": "text", "text": f"\n# Question\n{PROMPT}"},
                ]
                response = model.invoke(
                    [SystemMessage(SYSTEM_PROMPT), HumanMessage(content=content)],
                    images=[image], max_new_tokens=MAX_NEW_TOKENS)
                reasoning, answer = common.parse_ai_content(response.content)
                rec |= {"answer": answer, "reasoning": reasoning,
                        "num_tool_calls": 0, "retrieved_doc_ids": []}
            else:
                # Same code path as run_experiment.run_full.
                card, _ = common.timed(model.classify, image)

                seeds, hypothesis = None, None
                if multiquery:
                    hyp_msg = HumanMessage(content=[
                        {"type": "text", "text": "# Artwork image\n"},
                        {"type": "image"},
                        {"type": "text", "text": f"\n{HYPOTHESIS_PROMPT}"},
                    ])
                    hyp_resp = model.model.invoke(
                        [SystemMessage("You are a helpful assistant."), hyp_msg],
                        images=[image], max_new_tokens=48)
                    _, hypothesis = common.parse_ai_content(hyp_resp.content)
                    # Strip lead-ins such as "The subject of the painting
                    # appears to be", then keep at most 12 words.
                    hypothesis = re.sub(
                        r"^\W*(the\s+)?(subject\s+(of\s+(the\s+|this\s+)?"
                        r"(painting|work|image)\s+)?(appears\s+to\s+be|seems"
                        r"\s+to\s+be|is)|(this\s+|the\s+)?(painting|work|"
                        r"image|scene)\s+(appears\s+to\s+)?(depicts?|shows?|"
                        r"portrays?|represents?)|it\s+(appears|seems)\s+to\s+"
                        r"be|a\s+painting\s+of)\s*",
                        "", (hypothesis or "").strip(), flags=re.I)
                    hypothesis = " ".join(hypothesis.split()[:12])
                    seeds = build_seed_queries(
                        common.serialize_card(card), hypothesis)

                user_msg = common.build_first_user_message(
                    PROMPT, card if do_classify else None)
                new_messages, _ = model.chat_turn(
                    messages=[user_msg], input_image=image,
                    classify=do_classify, retrieve=True, context_images=[],
                    max_new_tokens=MAX_NEW_TOKENS,
                    seed_queries=seeds,
                    # 20 fragments per query, best 15 overall kept.
                    seed_limit=20,
                    seed_merge_top_k=15)
                final = next(m for m in reversed(new_messages)
                             if isinstance(m, AIMessage))
                reasoning, answer = common.parse_ai_content(final.content)

                doc_ids, queries = [], []
                for m in new_messages:
                    if isinstance(m, ToolMessage) and m.artifact:
                        doc_ids.extend(m.artifact.get("context_idxs", []))
                    if isinstance(m, AIMessage):
                        for tc in (getattr(m, "tool_calls", None) or []):
                            queries.append(tc.get("args", {}))
                steps = final.response_metadata.get("step_durations", [])
                rec |= {
                    "answer": answer,
                    "reasoning": reasoning,
                    "card": common.serialize_card(card),
                    "card_shown_to_model": do_classify,
                    "retrieved_doc_ids": doc_ids,
                    "tool_queries": queries,
                    "num_tool_calls": sum(1 for s in steps if s["type"] == "tool"),
                    "hypothesis": hypothesis,
                    "seed_queries": seeds,
                }
            rec["seconds"] = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            rec |= {"error": str(e)}
        results.append(rec)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    ok = sum(1 for r in results if r.get("answer"))
    print(f"\n{ok}/{len(items)} answered -> {out_path}")


if __name__ == "__main__":
    main()
