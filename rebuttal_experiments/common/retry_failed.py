"""Remove transient failures from result files so that a resubmitted run redoes them.

run_experiment.py skips every (id, question_type) already in its output,
errors included. Permanent errors (dead image URL, prompt longer than the
model context) are kept; everything else (e.g. Qdrant connection errors) is
removed. Dry run by default.

Usage:
    python rebuttal_experiments/common/retry_failed.py --results-dir rebuttal_experiments/aqua/results
    python rebuttal_experiments/common/retry_failed.py --results-dir ... --apply
"""

import argparse
import json
import shutil
from pathlib import Path

PERMANENT = (
    "image not downloaded",
    "longer than the maximum model length",
)


def is_permanent(err: str) -> bool:
    e = str(err).lower()
    return any(p.lower() in e for p in PERMANENT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--variant", default=None,
                    help="Only this variant; default is every *.json in the dir.")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    d = Path(a.results_dir)
    files = ([d / f"{a.variant}.json"] if a.variant
             else sorted(p for p in d.glob("*.json")
                         if not p.name.endswith(("_judged.json", "_eval_summary.json"))))

    total_kept = total_dropped = 0
    for f in files:
        if not f.exists():
            continue
        entries = json.loads(f.read_text())
        keep, drop = [], []
        for e in entries:
            if "error" in e and not is_permanent(e["error"]):
                drop.append(e)
            else:
                keep.append(e)
        if not drop:
            print(f"  {f.name:38s} nothing retryable ({len(entries)} entries)")
            continue
        total_dropped += len(drop)
        total_kept += len(keep)
        kinds = {}
        for e in drop:
            k = str(e["error"])[:60]
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  {f.name:38s} {len(drop):4d} retryable failures removed, {len(keep)} kept")
        for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:3]:
            print(f"        {n:4d} x {k}")
        if a.apply:
            shutil.copy(f, f.with_suffix(".json.bak"))
            f.write_text(json.dumps(keep, indent=2, default=str))

    if not a.apply:
        print(f"\nDRY RUN — would remove {total_dropped} entries. Re-run with --apply.")
    else:
        print(f"\nRemoved {total_dropped} retryable failures "
              f"(originals saved alongside as .json.bak). Resubmit the run jobs.")


if __name__ == "__main__":
    main()
