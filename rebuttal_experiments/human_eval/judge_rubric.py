"""Score the human-evaluation answers with microsoft/phi-4, using the raters' rubric.

The judge gets the same reference material and the same rubric as the human
raters: four axes (subject, figures, placement, evidence) scored 0/1/2 by what
is claimed against the reference, plus a forced choice of the best
description. The primary metric is the 0-6 sum of subject, figures and
placement.

All descriptions of a painting are scored together in one prompt, labelled
A/B/C without system names. Every painting is judged twice, in opposite
presentation orders. Writes results/judge_rubric.json.

Usage (GPU):
    python rebuttal_experiments/human_eval/judge_rubric.py --systems base,artseek_multiquery,gpt
"""

import os

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import common  # noqa: E402,F401  (loads .env)

import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import click  # noqa: E402

HERE = Path(__file__).resolve().parent
JUDGE = "microsoft/phi-4"
LET = ["A", "B", "C", "D"]
AXES = ("subject", "figures", "placement", "evidence")
CORRECTNESS_AXES = ("subject", "figures", "placement")

RUBRIC = """\
You are evaluating descriptions of a painting written by different assistants.

REFERENCE INFORMATION (what the painting actually is):
{reference}

THE DESCRIPTIONS:
{descriptions}

Each description answers four numbered questions: SUBJECT, FIGURES, PLACEMENT
and EVIDENCE.

Score EVERY description on EVERY axis, using 0, 1 or 2. Judge what is CLAIMED
against the reference, not how well it is written. Fluent writing earns
nothing; a confident wrong claim is worse than an accurate cautious one.

1. subject - Is the depicted event, story or scene correctly identified?
   2 = names the specific subject the reference gives.
   1 = right general category (a religious scene, a portrait, a landscape)
       but the specific subject is not identified.
   0 = confidently states a subject the reference contradicts.
2. figures - Are the figures correctly identified, with the attributes that
   identify them?
   2 = names the actual figures and cites identifying attributes.
   1 = describes them only generically ("a man", "a woman", "a saint").
   0 = names the wrong figures.
3. placement - Is the tradition, school and period right?
   2 = correct tradition AND approximately correct period.
   1 = one of the two right, or right but very vague ("European", "old").
   0 = wrong tradition or wrong century.
4. evidence - Are the named works, artists or sources real and relevant?
   2 = names specific real works or artists that genuinely bear on this
       painting.
   1 = names something real but only loosely relevant, or is vague.
   0 = names nothing specific, or names works that do not exist.

Then choose the single best description overall - the one you would rather be
given as a gallery visitor. Weigh being right above being fluent.

Reply with ONLY this JSON, no other text:
{{"scores": {{{score_slots}}}, "best": "<letter>", "why": "<one sentence>"}}"""


def build_reference(sheet: dict) -> str:
    """The reference shown to the raters, as text (English translation of
    the ICCD fields when available)."""
    out = []
    for k, label in (("title", "Catalogued title"), ("year", "Date"),
                     ("artist", "Artist")):
        if sheet.get(k):
            out.append(f"{label}: {sheet[k]}")
    if sheet.get("wikipedia_extract"):
        out.append(f"Encyclopaedia summary: {sheet['wikipedia_extract']}")
    for code, f in (sheet.get("iccd") or {}).items():
        val = f.get("value_en") or f.get("value")
        if val:
            out.append(f"{f['gloss'].capitalize()}: {val}")
    for kf in sheet.get("key_facts") or []:
        out.append(f"Key fact: {kf}")
    return "\n".join(out) or "(no reference material available)"


def parse(text: str, letters: list) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d.get("scores"), dict) or d.get("best") not in letters:
        return None
    # Reject judgements with a missing axis.
    for L in letters:
        row = d["scores"].get(L)
        if not isinstance(row, dict) or any(a not in row for a in AXES):
            return None
    return d


@click.command()
@click.option("--systems", default="base,artseek_multiquery,gpt",
              show_default=True)
@click.option("--items", default=str(HERE / "data" / "items.json"),
              show_default=True)
@click.option("--sheets", default=str(HERE / "data" / "factsheets.json"),
              show_default=True)
@click.option("--results-dir", default=str(HERE / "results"),
              show_default=True)
@click.option("--out", default=str(HERE / "results" / "judge_rubric.json"),
              show_default=True)
@click.option("--max-new-tokens", default=800, show_default=True)
def main(systems: str, items: str, sheets: str, results_dir: str, out: str,
         max_new_tokens: int):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    S = [s.strip() for s in systems.split(",") if s.strip()]
    if len(S) < 2:
        raise SystemExit(
            f"--systems resolved to {S!r}; at least two systems are needed.")
    rows = json.loads(Path(items).read_text())
    sheet_by_id = {x["id"]: x for x in json.loads(Path(sheets).read_text())}

    ans: dict[str, dict[str, str]] = {}
    for s in S:
        p = Path(results_dir) / f"{s}_normalized.json"
        if not p.exists():
            p = Path(results_dir) / f"{s}.json"
        if not p.exists():
            raise SystemExit(f"missing answers for {s}: {p}")
        for r in json.loads(p.read_text()):
            ans.setdefault(r["id"], {})[s] = (
                r.get("answer_normalized") or r.get("answer") or "")

    missing = [it["id"] for it in rows
               if any(not ans.get(it["id"], {}).get(s) for s in S)]
    if missing:
        raise SystemExit(
            f"{len(missing)} items lack an answer from at least one system "
            f"(first: {missing[:3]}).")

    letters = LET[:len(S)]
    slots = ", ".join(
        f'"{L}": {{' + ", ".join(f'"{a}": <0-2>' for a in AXES) + "}}"
        for L in letters)

    jobs = []
    for it in rows:
        base_order = sorted(S, key=lambda s: hash((it["id"], s)) % 997)
        for rev in (False, True):
            order = list(reversed(base_order)) if rev else base_order
            desc = "\n\n".join(
                f"Description {letters[k]}:\n{ans[it['id']][sys]}"
                for k, sys in enumerate(order))
            jobs.append({
                "id": it["id"], "tier": it["tier"], "reversed": rev,
                "mapping": dict(zip(letters, order)),
                "prompt": RUBRIC.format(
                    reference=build_reference(sheet_by_id[it["id"]]),
                    descriptions=desc, score_slots=slots),
            })

    print(f"{len(rows)} items x {len(S)} systems x 2 orders = {len(jobs)} judgements")
    path = snapshot_download(JUDGE)
    tok = AutoTokenizer.from_pretrained(path)
    llm = LLM(model=path, dtype="bfloat16", max_model_len=16384,
              gpu_memory_utilization=0.85, disable_custom_all_reduce=True)
    outs = llm.generate(
        [tok.apply_chat_template([{"role": "user", "content": j["prompt"]}],
                                 tokenize=False, add_generation_prompt=True)
         for j in jobs],
        SamplingParams(temperature=0.0, max_tokens=max_new_tokens))

    out_rows, bad = [], 0
    for j, o in zip(jobs, outs):
        d = parse(o.outputs[0].text, letters)
        if d is None:
            bad += 1
            continue
        out_rows.append({
            "id": j["id"], "tier": j["tier"], "reversed": j["reversed"],
            "scores": {j["mapping"][L]: v for L, v in d["scores"].items()
                       if L in j["mapping"]},
            "best": j["mapping"].get(d["best"]),
            "why": d.get("why", ""),
        })
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(out_rows, indent=2, ensure_ascii=False))
    print(f"wrote {len(out_rows)} judgements ({bad} unparseable) -> {out}\n")

    def report(subset, label):
        if not subset:
            return
        print(f"--- {label} ({len({r['id'] for r in subset})} items) ---")
        head = "".join(f"{a[:9]:>11}" for a in AXES)
        print(f"{'system':22}{head}{'SUM(0-6)':>11}")
        for s in S:
            per = {a: [r["scores"][s][a] for r in subset if s in r["scores"]]
                   for a in AXES}
            cells = "".join(f"{statistics.mean(per[a]):11.2f}" if per[a] else
                            f"{'-':>11}" for a in AXES)
            tot = sum(statistics.mean(per[a]) for a in CORRECTNESS_AXES
                      if per[a])
            print(f"{s:22}{cells}{tot:11.2f}")
        wins = {s: sum(1 for r in subset if r["best"] == s) for s in S}
        print("  forced choice: " +
              "  ".join(f"{s}={wins[s]}" for s in S) +
              f"   (of {len(subset)})\n")

    report(out_rows, "ALL")
    for tier in ("indexed_lowprofile", "gallery"):
        report([r for r in out_rows if r["tier"] == tier], tier)


if __name__ == "__main__":
    main()
