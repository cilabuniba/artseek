"""Obscure-painting split (ArtPedia-VQA).

A painting is "obscure" when the bare backbone, shown only the image, names
neither its title nor its artist (artpedia_vqa/obscure_probe.py +
ground_truth.py). Reports the size and examples of the split and tests whether
the gain of the RAG variants over `base` differs between obscure and
non-obscure paintings (difference-of-differences, painting-level bootstrap).
The other tables are restricted to the split with `--subset obscure`.

Usage:
    python rebuttal_experiments/common/analysis/obscure_report.py [--judge phi4]
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="phi4", choices=["phi4"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-examples", type=int, default=8)
    a = ap.parse_args()
    judge = lib.JUDGE_QWEN if a.judge == "qwen" else lib.JUDGE_PHI4

    obs = lib.read_jsonl(lib.JSONL_DIR / "obscure.jsonl")
    if not obs:
        raise SystemExit("no obscure.jsonl — run slurm/30 + slurm/40 first")

    out, payload = [], {"judge_model": judge}

    # ── 1. subset composition ────────────────────────────────────────────
    n = len(obs)
    n_obscure = sum(1 for r in obs if r["is_obscure"])
    n_title = sum(1 for r in obs if r["title_correct"])
    n_artist = sum(1 for r in obs if r["artist_correct"])
    n_both = sum(1 for r in obs if r["title_correct"] and r["artist_correct"])
    payload["composition"] = {
        "n_paintings_probed": n, "n_obscure": n_obscure,
        "pct_obscure": n_obscure / n * 100,
        "title_top1": n_title / n * 100, "artist_top1": n_artist / n * 100,
        "both_correct": n_both / n * 100,
    }
    out.append("#### Backbone identification of the ArtPedia test paintings\n")
    out.append(lib.table_block(
        "obscure_composition",
        ["paintings probed", "title correct", "artist correct", "both correct",
         "**neither correct → obscure**"],
        [[n, f"{n_title} ({n_title / n:.1%})", f"{n_artist} ({n_artist / n:.1%})",
          f"{n_both} ({n_both / n:.1%})",
          f"**{n_obscure} ({n_obscure / n:.1%})**"]]))

    # ── 2. example works, for sanity-checking the split ──────────────────
    def examples(flag, k):
        sel = [r for r in obs if r["is_obscure"] == flag][:k]
        return [[r["painting_id"], r["title"][:44], (r["gold_artist"] or "—")[:24],
                 (r["pred_title"] or "—")[:34], (r["pred_artist"] or "—")[:22]]
                for r in sel]

    out.append("\n**Flagged obscure** (backbone got both wrong)\n")
    out.append(lib.table_block(
        "obscure_examples_yes",
        ["id", "true title", "true artist", "model's guessed title", "model's guessed artist"],
        examples(True, a.n_examples)))
    out.append("\n**Not obscure** (backbone identified it)\n")
    out.append(lib.table_block(
        "obscure_examples_no",
        ["id", "true title", "true artist", "model's guessed title", "model's guessed artist"],
        examples(False, a.n_examples)))

    # ── 3. the widening test ─────────────────────────────────────────────
    scored = lib.load_scored(judge)
    subsets = {
        "obscure": {r["painting_id"] for r in obs if r["is_obscure"]},
        "non-obscure": {r["painting_id"] for r in obs if not r["is_obscure"]},
    }
    rows = []
    payload["widening"] = {}
    for target in lib.RAG_VARIANTS:
        if target not in scored:
            continue
        cells = [f"`base` → `{target}`"]
        entry = {}
        for label, pids in subsets.items():
            A = {q: r for q, r in scored["base"].items() if r["painting_id"] in pids}
            B = {q: r for q, r in scored[target].items() if r["painting_id"] in pids}
            pairs = lib.paired_diffs(A, B, "correctness")
            if not pairs:
                cells += ["–", "–", "–"]
                continue
            res = lib.cluster_bootstrap_paired(pairs, n_boot=a.n_boot)
            entry[label] = res
            cells += [f"{res['n']}", f"{res['mean']:+.3f}", lib.fmt_ci(res)]
        if "obscure" in entry and "non-obscure" in entry:
            entry["widening"] = entry["obscure"]["mean"] - entry["non-obscure"]["mean"]
            # Bootstrap the difference-of-differences, resampling paintings
            # independently within each stratum.
            A = {q: r for q, r in scored["base"].items()}
            B = {q: r for q, r in scored[target].items()}
            rng = random.Random(lib.SEED)
            by_s = {}
            for label, pids in subsets.items():
                cl = {}
                for q in sorted(set(A) & set(B)):
                    if A[q]["painting_id"] not in pids:
                        continue
                    cl.setdefault(A[q]["painting_id"], []).append(
                        B[q]["correctness"] - A[q]["correctness"])
                by_s[label] = list(cl.values())
            diffs = []
            for _ in range(a.n_boot):
                m = {}
                for label, groups in by_s.items():
                    tot = cnt = 0
                    for _ in range(len(groups)):
                        g = groups[rng.randrange(len(groups))]
                        tot += sum(g); cnt += len(g)
                    m[label] = tot / cnt if cnt else float("nan")
                diffs.append(m["obscure"] - m["non-obscure"])
            diffs.sort()
            lo = diffs[int(0.025 * a.n_boot)]
            hi = diffs[min(a.n_boot - 1, int(0.975 * a.n_boot))]
            entry["widening_ci"] = [lo, hi]
            entry["widening_significant"] = not (lo <= 0 <= hi)
            cells.append(f"{entry['widening']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        else:
            cells.append("–")
        payload["widening"][target] = entry
        rows.append(cells)
    out.append("\n#### Does retrieval's benefit widen on paintings the backbone cannot name?\n")
    out.append(lib.table_block(
        "obscure_widening",
        ["comparison", "n (obscure)", "Δ correctness (obscure)", "95% CI (obscure)",
         "n (non-obscure)", "Δ correctness (non-obscure)", "95% CI (non-obscure)",
         "widening (95% CI)"], rows))
    out.append(
        "\n*Power in the obscure stratum is materially lower than in the "
        "full benchmark — the CIs are correspondingly wider, and a "
        "difference-of-differences that does not reach significance here is "
        "weak evidence either way rather than evidence of no effect.*"
    )

    # ── 4. base's absolute performance in each stratum ───────────────────
    rows = []
    payload["by_stratum"] = {}
    for v in lib.VARIANTS:
        if v not in scored:
            continue
        cells = [f"`{v}`"]
        entry = {}
        for label, pids in subsets.items():
            items = [r for r in scored[v].values() if r["painting_id"] in pids]
            if not items:
                cells += ["–", "–"]
                continue
            e = {"n": len(items),
                 "correctness": sum(r["correctness"] for r in items) / len(items),
                 "recall": sum(r["recall"] for r in items) / len(items)}
            entry[label] = e
            cells += [lib.fmt(e["correctness"]), lib.fmt(e["recall"])]
        payload["by_stratum"][v] = entry
        rows.append(cells)
    out.append("\n#### Absolute performance in each stratum\n")
    out.append(lib.table_block(
        "obscure_by_stratum",
        ["variant", "correctness (obscure)", "recall (obscure)",
         "correctness (non-obscure)", "recall (non-obscure)"], rows))

    lib.write_table(f"obscure__{a.judge}", payload)
    print("\n".join(out))


if __name__ == "__main__":
    main()
