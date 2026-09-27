"""Decode the rating transcripts into scores per system.

The rating page gives each rater a short transcript of their scores. Letters
A/B/C are assigned per rater and per painting by a seeded shuffle of the
rater code and the item id (`orderFor` in build_rating_app.py), reimplemented
here to map letters back to systems. Identical duplicate transcripts are
counted once.

Exclusions:
  * item devanna-010: the photograph shown was not the catalogued work it was
    annotated with (inventory 010.010 "La Canonica" instead of 010
    "Annunciazione"), so its ratings are not comparable;
  * rater L3: never used the score 0 (0 of 168 axis ratings) and never gave a
    Sigma below 3, so the ratings cannot distinguish wrong answers.

Writes results/human_ratings.json.

Usage:
    python rebuttal_experiments/human_eval/collect_ratings.py
    python rebuttal_experiments/human_eval/collect_ratings.py --include-excluded
"""

import json
from collections import defaultdict
from pathlib import Path

import click

HERE = Path(__file__).resolve().parent

# The pre-shuffle system list of build_rating_app.py (order matters).
SYSTEMS = ["base", "artseek_multiquery", "gpt"]
DISPLAY = {"base": "base", "artseek_multiquery": "ArtSeek", "gpt": "GPT-5.5"}
AXES = ["subject", "figures", "placement", "evidence"]
# Sigma (0-6), the primary metric: subject + figures + placement.
SIGMA_AXES = ["subject", "figures", "placement"]

EXCLUDED = {
    "devanna-010": "annotation did not match the image (see the note in data/items.json)",
}

# Raters excluded on a scale-usage criterion (see the module docstring).
EXCLUDED_RATERS = {
    "L3": "never used the score 0 (0/168 ratings); Sigma range restricted to 3-6",
}


def fnv1a(s: str) -> int:
    """FNV-1a over UTF-16 code units, matching JS String.charCodeAt."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def order_for(rater_code: str, item_id: str) -> list:
    """Reproduce the app's per-rater, per-item letter assignment.

    JS does the LCG step in doubles, but x*1664525 peaks near 7.2e15, inside
    the 2^53 exact-integer range, so plain integer arithmetic is identical.
    """
    sys = list(SYSTEMS)
    x = fnv1a(rater_code.upper() + "|" + item_id)
    for i in range(len(sys) - 1, 0, -1):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        j = x % (i + 1)
        sys[i], sys[j] = sys[j], sys[i]
    return sys


def parse(line: str) -> dict:
    magic, rater, role, mode, ts, body = line.strip().split("|", 5)
    if magic != "ARTSEEKv3":
        raise ValueError(f"not an ARTSEEKv3 transcript: {magic}")
    items = {}
    for rec in body.split(";"):
        iid, rest = rec.split(":")
        groups = rest.split(",")
        scores = {}
        for letter, g in zip("ABC", groups[:3]):
            scores[letter] = {
                ax: (None if c == "-" else int(c)) for ax, c in zip(AXES, g)
            }
        items[iid] = {"scores": scores, "best": groups[3]}
    return {"rater": rater.upper(), "role": role, "mode": mode,
            "submitted_utc": ts, "items": items, "raw": line.strip()}


def load(path: Path) -> tuple:
    seen, subs, dupes, conflicts = {}, [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = parse(line)
        if s["rater"] in EXCLUDED_RATERS:
            continue
        key = (s["rater"], s["submitted_utc"], s["raw"])
        if key in seen:
            dupes.append(s["rater"])
            continue
        prior = [t for t in seen if t[0] == s["rater"]]
        if prior:
            conflicts.append(s["rater"])
        seen[key] = True
        subs.append(s)
    return subs, dupes, conflicts


@click.command()
@click.option("--subs-file", default="data/submissions.txt", show_default=True)
@click.option("--out", default="results/human_ratings.json", show_default=True)
@click.option("--include-excluded", is_flag=True,
              help="Keep items in EXCLUDED (for inspecting what was dropped).")
def main(subs_file, out, include_excluded):
    subs, dupes, conflicts = load(HERE / subs_file)
    print(f"{len(subs)} unique submissions "
          f"({len(dupes)} duplicate line(s) ignored: {sorted(set(dupes)) or '-'})")
    if conflicts:
        print(f"  !! same rater submitted differing transcripts: "
              f"{sorted(set(conflicts))} — resolve before trusting totals")
    if not include_excluded:
        for iid, why in EXCLUDED.items():
            print(f"  excluding item {iid}: {why}")
        for r, why in EXCLUDED_RATERS.items():
            print(f"  excluding rater {r}: {why}")

    per_sys = defaultdict(lambda: defaultdict(list))   # sys -> axis -> [score]
    sigma = defaultdict(list)                          # sys -> [0-6]
    pref = defaultdict(int)
    by_rater = {}
    rows = []

    for s in subs:
        r_sigma, r_pref = defaultdict(list), defaultdict(int)
        for iid, rec in s["items"].items():
            if iid in EXCLUDED and not include_excluded:
                continue
            letters = dict(zip("ABC", order_for(s["rater"], iid)))
            for L, sysname in letters.items():
                sc = rec["scores"][L]
                for ax in AXES:
                    if sc[ax] is not None:
                        per_sys[sysname][ax].append(sc[ax])
                tot = sum(sc[ax] for ax in SIGMA_AXES if sc[ax] is not None)
                sigma[sysname].append(tot)
                r_sigma[sysname].append(tot)
                rows.append({"rater": s["rater"], "role": s["role"],
                             "item": iid, "letter": L, "system": sysname,
                             **{ax: sc[ax] for ax in AXES}, "sigma": tot})
            if rec["best"] in letters:
                pref[letters[rec["best"]]] += 1
                r_pref[letters[rec["best"]]] += 1
        by_rater[s["rater"]] = {
            "role": s["role"],
            "n_items": len(r_sigma[SYSTEMS[0]]),
            "sigma": {DISPLAY[k]: round(sum(v) / len(v), 2)
                      for k, v in r_sigma.items()},
            "preferred": {DISPLAY[k]: r_pref[k] for k in SYSTEMS},
        }

    n_items = len(sigma[SYSTEMS[0]])
    print(f"\n{n_items} judgements per system "
          f"({len(subs)} raters x {n_items // max(len(subs), 1)} items)\n")
    hdr = f"{'system':10s} " + " ".join(f"{ax:>10s}" for ax in AXES) + \
          f" {'Sigma/6':>8s} {'preferred':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for k in SYSTEMS:
        means = " ".join(f"{sum(per_sys[k][ax]) / len(per_sys[k][ax]):10.2f}"
                         for ax in AXES)
        print(f"{DISPLAY[k]:10s} {means} "
              f"{sum(sigma[k]) / len(sigma[k]):8.2f} "
              f"{pref[k]:>7d}/{sum(pref.values())}")

    print("\nper rater (Sigma/6, then forced choice):")
    for r, d in by_rater.items():
        sig = "  ".join(f"{k} {v}" for k, v in d["sigma"].items())
        pr = "  ".join(f"{k} {v}" for k, v in d["preferred"].items())
        print(f"  {r:10s} ({d['role']:6s}, {d['n_items']} items)  {sig}   |   {pr}")

    payload = {
        "n_raters": len(subs), "n_items_per_rater": n_items // max(len(subs), 1),
        "excluded_items": EXCLUDED if not include_excluded else {},
        "duplicates_ignored": sorted(set(dupes)),
        "conflicting_raters": sorted(set(conflicts)),
        "overall": {DISPLAY[k]: {
            **{ax: round(sum(per_sys[k][ax]) / len(per_sys[k][ax]), 3)
               for ax in AXES},
            "sigma": round(sum(sigma[k]) / len(sigma[k]), 3),
            "preferred": pref[k],
        } for k in SYSTEMS},
        "per_rater": by_rater,
        "rows": rows,
    }
    dest = HERE / out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
