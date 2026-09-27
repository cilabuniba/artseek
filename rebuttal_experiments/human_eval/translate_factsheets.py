"""Translate the Italian ICCD fields of the gallery fact sheets into English
with microsoft/phi-4 (vLLM, greedy).

Translation only: the prompt forbids adding or resolving anything (hedges such
as "attribuibile a" or "(?)" must be kept). Codes and names (Iconclass, NCTN,
inventory numbers, author) are not translated, and a translation much longer
than its source is discarded. The English text is added as `value_en` next to
the Italian `value`. Updates data/factsheets.json in place. Needs a GPU.

Usage:
    python rebuttal_experiments/human_eval/translate_factsheets.py
"""

import os

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import common  # noqa: E402,F401  (loads .env)

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

HERE = Path(__file__).resolve().parent
JUDGE = "microsoft/phi-4"

# Free-text ICCD fields only; codes (e.g. Iconclass) and names are copied as
# they are.
TRANSLATABLE = ("SGTI", "SGTD", "DESS", "NSC", "DTZG", "MTC", "LDCN",
                "ATB", "DESO")

PROMPT = """\
Translate the following Italian museum-catalogue text into English.

Museum-cataloguing vocabulary (use these exact readings):
- "tavola" as a support means "panel" (a wood panel), never "table".
- "tela" means "canvas". "pittura a olio" means "oil painting".
- "Pan" is the Greek god Pan. It is NEVER "bread".
- "ambito" means "milieu" or "circle". "bottega" means "workshop".
- "sec." abbreviates "secolo", i.e. "century".

Rules:
- Translate only. Do not add, explain, interpret or complete anything.
- Never expand, gloss or restate a classification code or a proper name.
- Keep every hedge exactly as hedged. "attribuibile a" is "attributable to",
  "(?)" stays "(?)", "forse" is "perhaps". Do not resolve any uncertainty.
- Keep proper names, place names, saints' names and codes unchanged.
- If the text is already English, return it unchanged.
- Reply with the translation only, no preamble and no quotation marks.

Italian text:
{text}"""


@click.command()
@click.option("--sheets", default=str(HERE / "data" / "factsheets.json"),
              show_default=True)
@click.option("--max-new-tokens", default=900, show_default=True)
def main(sheets: str, max_new_tokens: int):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = json.loads(Path(sheets).read_text())

    jobs = []
    for s in data:
        for code, field in (s.get("iccd") or {}).items():
            if code in TRANSLATABLE and (field.get("value") or "").strip():
                jobs.append((s["id"], code, field["value"]))
    if not jobs:
        print("nothing to translate")
        return
    print(f"{len(jobs)} ICCD fields to translate across "
          f"{len({j[0] for j in jobs})} gallery items")

    path = snapshot_download(JUDGE)
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.85, disable_custom_all_reduce=True)
    outs = llm.generate(
        [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=text)}],
            tokenize=False, add_generation_prompt=True) for _, _, text in jobs],
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens))

    by_id = {s["id"]: s for s in data}
    rejected = 0
    for (item_id, code, src), o in zip(jobs, outs):
        en = o.outputs[0].text.strip()
        # Much longer than the source: the model expanded rather than
        # translated, so keep the Italian.
        if len(en) > 2.5 * len(src) + 40:
            rejected += 1
            print(f"  rejected {item_id}/{code}: {len(src)} chars -> {len(en)}")
            continue
        by_id[item_id]["iccd"][code]["value_en"] = en

    Path(sheets).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {len(jobs) - rejected} translations "
          f"({rejected} rejected as over-long) -> {sheets}")
    for s in data:
        if s.get("iccd"):
            done = sum(1 for f in s["iccd"].values() if f.get("value_en"))
            print(f"  {s['title'][:44]:46} {done}/{len(s['iccd'])} fields en")


if __name__ == "__main__":
    main()
