"""Remove formatting differences between systems before blinding.

A deterministic, rule-based pass (no model) applied identically to every
system: markdown headings, bold/italics and bullets are removed, the four part
labels are normalised ("1. SUBJECT: ..."), "Final answer:" markers and
mentions of the pipeline (artwork card, retrieved documents, confidence
values) are removed. The original text is kept in `answer_raw`. Writes
results/<system>_normalized.json.

Usage:
    python rebuttal_experiments/human_eval/normalize_answers.py
"""

import json
import re
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

RATED = ("base", "artseek_multiquery", "gpt")

RULES = [
    # House formatting
    (re.compile(r"^\s*#{1,6}\s*", re.M), ""),
    (re.compile(r"\*\*(.+?)\*\*", re.S), r"\1"),
    (re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.S), r"\1"),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    # Keep the "1." numbering, drop the emphasis around the part label.
    (re.compile(r"^(\s*\d\.\s*)\**\s*(SUBJECT|FIGURES|PLACEMENT|EVIDENCE)"
                r"\s*\**\s*[:\-]?\s*", re.M | re.I),
     lambda m: f"{m.group(1)}{m.group(2).upper()}: "),
    # "Final Answer:" markers (only the marker is removed, not the text).
    (re.compile(r"^\s*(final answer|answer)\s*:\s*", re.I | re.M), ""),
    # Mentions of the pipeline
    (re.compile(r"\s*\(?\b\d{1,3}(\.\d+)?\s*%\s*confidence\b\)?", re.I), ""),
    (re.compile(r"\bconfidence level of \d{1,3}(\.\d+)?%?\b", re.I), ""),
    (re.compile(r"\bthe artwork card\b", re.I), "the available evidence"),
    (re.compile(r"\bretrieved documents?\b", re.I), "sources"),
    (re.compile(r"\bbased on the information provided,?\s*", re.I), ""),
    # Whitespace
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def normalize(text: str) -> str:
    for pat, rep in RULES:
        text = pat.sub(rep, text)
    return text.strip()


@click.command()
@click.option("--results-dir", default=str(HERE / "results"), show_default=True)
def main(results_dir: str):
    out = Path(results_dir)
    changed = 0
    for system in RATED:
        p = out / f"{system}.json"
        if not p.exists():

            continue
        rows = json.loads(p.read_text())
        for r in rows:
            if not r.get("answer"):
                continue
            r["answer_raw"] = r["answer"]
            r["answer_normalized"] = normalize(r["answer"])
            if r["answer_normalized"] != r["answer_raw"]:
                changed += 1
        q = out / f"{system}_normalized.json"
        q.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        n = sum(1 for r in rows if r.get("answer_normalized"))
        print(f"  {system:20s} {n:3d} answers -> {q.name}")
    print(f"\n{changed} answers altered by the pass (rest were already clean)")


if __name__ == "__main__":
    main()
