"""Shared loading and statistics helpers for the analysis scripts.

Every script reads the JSONL files under ``<ARTSEEK_EXP_DIR>/results/jsonl/``
(built by build_runs_jsonl.py and judge.py) and writes machine-readable
tables to ``results/tables/``.

Questions are nested within paintings, so confidence intervals resample
paintings, not questions (see ``cluster_bootstrap_paired``).
"""

import json
import math
import os
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
# Benchmark folder the scripts read from and write to (default: ArtPedia-VQA).
EXP_DIR = Path(
    os.environ.get("ARTSEEK_EXP_DIR", ANALYSIS_DIR.parents[1] / "artpedia_vqa")
).resolve()
RESULTS_DIR = EXP_DIR / "results"
JSONL_DIR = RESULTS_DIR / "jsonl"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
ASSETS_DIR = RESULTS_DIR / "assets"

VARIANTS = (
    "base",
    "base_gemma3",
    "base_mistral3",
    "full_seeded",
    "full_mistral3",
    "full_gemma3",
    "full_classify",
    "full_noclassify",
    "full_systemprompt",
    "full_classify_noartist",
    "full_classify_notags",
    "full_classify_reliable",
    "full_classify_gated",
    "full_alwaysretrieve",
    "full_singleretrieve",
)

BASE_VARIANTS = ("base", "base_gemma3", "base_mistral3")

RAG_VARIANTS = tuple(v for v in VARIANTS if v not in BASE_VARIANTS)

# The oracle retrieval probe (oracle_probe.py). It goes through runs.jsonl so
# the evidence judge can score it, but it has no answers.
ORACLE_VARIANT = "oracle"

# Judge identifiers (every reported number uses phi-4).
JUDGE_QWEN = "Qwen/Qwen3-8B"
JUDGE_PHI4 = "microsoft/phi-4"

SEED = 20260828


# ── io ───────────────────────────────────────────────────────────────────


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_table(name: str, payload) -> None:
    """Persist a machine-readable copy of a rendered table."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))


def load_runs(variants=None) -> list[dict]:
    rows = read_jsonl(JSONL_DIR / "runs.jsonl")
    if variants is not None:
        keep = set(variants)
        rows = [r for r in rows if r["variant"] in keep]
    return rows


def load_judge(judge: str, kind: str = "answer") -> dict:
    """Return {(variant, query_id): row} for one judge and one judging pass."""
    fname = {"answer": "judge_answer.jsonl", "evidence": "judge_evidence.jsonl"}[kind]
    return {
        (r["variant"], r["query_id"]): r
        for r in read_jsonl(JSONL_DIR / fname)
        if r["judge_model"] == judge
    }


def load_scored(judge: str, variants=None) -> dict:
    """Join runs with answer-judge scores.

    Returns {variant: {query_id: row}} containing only items that both ran
    successfully *and* were scored by ``judge``.
    """
    scores = load_judge(judge, "answer")
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in load_runs(variants):
        key = (r["variant"], r["query_id"])
        if r["status"] != "ok" or key not in scores:
            continue
        merged = dict(r)
        merged["recall"] = scores[key]["recall"]
        merged["correctness"] = scores[key]["correctness"]
        merged["judge_justification"] = scores[key].get("justification", "")
        out[r["variant"]][r["query_id"]] = merged
    return dict(out)


# ── ids ──────────────────────────────────────────────────────────────────


def query_id(painting_id, question_type: str) -> str:
    return f"{painting_id}_{question_type}"


def painting_of(qid: str) -> str:
    return qid.rsplit("_", 1)[0]


# ── name normalisation ────────────────────────────────────────────────────


def strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def normalise_name(s: str) -> str:
    """Lowercase, de-accent, de-punctuate, and un-slug an artist name.

    Handles ArtGraph's ``lastname-firstname`` slug form and ordinary
    ``Firstname Lastname`` alike by reducing both to a sorted token bag is
    *not* done here — token order is preserved, and the caller compares
    with :func:`artist_match`, which tries both orders explicitly.
    """
    if not s:
        return ""
    s = strip_diacritics(str(s)).lower()
    s = s.replace("-", " ").replace("_", " ")
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return " ".join(s.split())


# Honorifics/particles that shouldn't drive a surname match on their own.
_PARTICLES = {
    "de", "di", "da", "del", "della", "van", "von", "der", "den", "le", "la",
    "el", "il", "dos", "das", "do", "y", "of", "the", "il", "lo",
}


# Generic attribution words that must never drive a name match on their own
# ("Master of Moulins", "workshop of Rubens", "follower of Rembrandt").
_GENERIC = {
    "master", "workshop", "follower", "circle", "school", "studio",
    "anonymous", "unknown", "attributed", "after", "manner", "painter",
    "artist",
}


def surname_of(name: str) -> str:
    toks = [t for t in normalise_name(name).split() if t not in _PARTICLES]
    return toks[-1] if toks else ""


def artist_match(pred: str, gold: str) -> tuple[bool, bool]:
    """Return (strict_match, lenient_match) for a predicted vs. gold artist.

    strict  — full normalised name equal, in either token order (so
              ``van-gogh-vincent`` matches ``Vincent van Gogh``).
    lenient — strict, or the surnames agree (and the surname is not a
              trivially short token). Reported separately because a
              surname-only match is genuinely weaker evidence.
    """
    p, g = normalise_name(pred), normalise_name(gold)
    if not p or not g:
        return False, False
    strict = p == g or " ".join(reversed(p.split())) == g or p.split() == list(
        reversed(g.split())
    )
    if not strict:
        strict = sorted(p.split()) == sorted(g.split())
    ps, gs = surname_of(pred), surname_of(gold)
    lenient = strict or (len(ps) >= 4 and ps not in _GENERIC and ps == gs)

    # Mononyms: artists conventionally known by one name ("Rembrandt",
    # "Caravaggio", "Cimabue") whose gold label carries the full form.
    # Accept when one side is a single substantive token that appears among
    # the other side's tokens.
    if not lenient:
        pt = [t for t in p.split() if t not in _PARTICLES]
        gt = [t for t in g.split() if t not in _PARTICLES]
        if len(pt) == 1 and len(pt[0]) >= 4 and pt[0] not in _GENERIC and pt[0] in gt:
            lenient = True
        elif len(gt) == 1 and len(gt[0]) >= 4 and gt[0] not in _GENERIC and gt[0] in pt:
            lenient = True

    return strict, lenient


# ── statistics ───────────────────────────────────────────────────────────


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cluster_bootstrap_paired(
    pairs: list[tuple[str, float]],
    n_boot: int = 10000,
    seed: int = SEED,
    alpha: float = 0.05,
):
    """Percentile bootstrap CI on a mean paired difference, resampling
    **paintings** with replacement rather than individual questions.

    ``pairs`` is a list of (painting_id, difference) — one entry per
    question, several questions sharing a painting_id. Each bootstrap
    replicate draws len(paintings) painting ids with replacement and takes
    *every* question belonging to each drawn painting, which propagates the
    within-painting correlation into the interval.
    """
    if not pairs:
        return {"n": 0, "n_clusters": 0, "mean": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}

    by_cluster: dict[str, list[float]] = defaultdict(list)
    for cid, diff in pairs:
        by_cluster[cid].append(diff)
    cluster_ids = list(by_cluster)
    values = [by_cluster[c] for c in cluster_ids]
    k = len(cluster_ids)

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total, count = 0.0, 0
        for _ in range(k):
            vs = values[rng.randrange(k)]
            total += sum(vs)
            count += len(vs)
        if count:
            means.append(total / count)
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    diffs = [d for _, d in pairs]
    return {
        "n": len(pairs),
        "n_clusters": k,
        "mean": _mean(diffs),
        "ci_low": lo,
        "ci_high": hi,
        "cohens_dz": cohens_dz(diffs),
    }


def cluster_bootstrap_mean(
    pairs: list[tuple[str, float]], n_boot: int = 10000, seed: int = SEED, alpha: float = 0.05
):
    """Same machinery, for a single (unpaired) mean rather than a difference."""
    return cluster_bootstrap_paired(pairs, n_boot=n_boot, seed=seed, alpha=alpha)


def cohens_dz(diffs: list[float]) -> float:
    """Standardised effect size for a paired difference: mean(d)/sd(d).

    d_z is the right paired-design statistic here (not d_av): it is what a
    paired t is a monotone function of, and it is what makes a raw
    difference like +0.065 on a 0-2 scale interpretable.
    """
    if len(diffs) < 2:
        return float("nan")
    sd = _stdev(diffs)
    if sd == 0:
        return float("nan")
    return _mean(diffs) / sd


def paired_diffs(a: dict, b: dict, field: str) -> list[tuple[str, float]]:
    """(painting_id, b[field] - a[field]) over query ids present in both."""
    common = sorted(set(a) & set(b))
    return [(painting_of(q), b[q][field] - a[q][field]) for q in common]


def paired_ttest(diffs: list[float]) -> float:
    """Two-sided p-value for mean(diffs) == 0 (ignores clustering; the
    bootstrap CIs are the reported inference)."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    sd = _stdev(diffs)
    if sd == 0:
        return 0.0 if _mean(diffs) != 0 else 1.0
    t = _mean(diffs) / (sd / math.sqrt(n))
    return _t_sf(abs(t), n - 1) * 2


def wilcoxon(diffs: list[float]) -> float:
    """Two-sided normal-approximation Wilcoxon signed-rank p-value."""
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n < 10:
        return float("nan")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k2 in range(i, j + 1):
            ranks[order[k2]] = avg
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return float("nan")
    z = (w_plus - mu) / sigma
    return 2 * (1 - _norm_cdf(abs(z)))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _t_sf(t: float, df: int) -> float:
    """Upper-tail probability of Student's t, via the regularised
    incomplete beta function."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta)
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - math.exp(
        b * math.log(1 - x) + a * math.log(x) - lbeta
    ) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            break
    return h


# ── agreement metrics ────────────────────────────────────────────────────


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    if len(xs) < 2:
        return float("nan")
    rx, ry = rank(xs), rank(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def cohens_kappa(a: list[int], b: list[int], labels=(0, 1, 2), weights="quadratic") -> float:
    """Weighted Cohen's kappa. ``quadratic`` is the right choice for the
    ordered 0/1/2 correctness label — it penalises a 0-vs-2 disagreement
    four times as heavily as an adjacent one."""
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    obs = [[c / n for c in row] for row in obs]
    ra = [sum(row) for row in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    if weights == "quadratic":
        w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    elif weights == "linear":
        w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    else:
        w = [[0.0 if i == j else 1.0 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ra[i] * cb[j] for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


# ── classification metrics ───────────────────────────────────────────────


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec + rec) else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    bal = (tpr + tnr) / 2 if not (math.isnan(tpr) or math.isnan(tnr)) else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / den if den else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "balanced_accuracy": bal, "mcc": mcc,
    }


# ── formatting ───────────────────────────────────────────────────────────


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        if math.isnan(x):
            return "–"
        return f"{x:.{nd}f}"
    return str(x)


def fmt_ci(d, nd=3):
    return f"[{d['ci_low']:+.{nd}f}, {d['ci_high']:+.{nd}f}]"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    if p < 1e-15:
        return "<1e-15"
    if p >= 0.001:
        return f"{p:.3f}"
    exp = math.floor(math.log10(p))
    return f"{p / 10 ** exp:.0f}×10⁻{abs(exp)}".replace("-", "")


def table_block(name: str, headers, rows) -> str:
    """A markdown table preceded by an HTML comment with its name."""
    return f"<!--TABLE:{name}-->\n" + markdown_table(headers, rows)


def markdown_table(headers, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
