"""Metrics and statistics over graded records.

Pure functions, no file IO and no plotting, so every number in the paper is
reproducible from `graded.jsonl` alone.

Deliberate choices a reviewer will ask about:

* `pass_at_k` is the unbiased Chen et al. estimator, not "did any of k succeed"
  measured on the k we happened to draw. The naive version is biased upward.
* No scipy. The exact binomial p-value for McNemar uses `math.comb`, and the
  chi-square survival function for 1 degree of freedom is `erfc(sqrt(x/2))`.
  This keeps results identical across environments, which matters because the
  Kaggle image's scipy version is not under our control.
* Every function that returns an interval also returns the `seed` and
  `n_bootstrap` it used. Uncertainty without a stated procedure is not
  reportable.
* Records with `error` set count as incorrect rather than being dropped;
  dropping failures silently inflates accuracy.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

Record = Dict[str, Any]

DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05
#: Random draws averaged over when estimating maj@k from more than k samples.
DEFAULT_N_SUBSET_DRAWS = 200


def _rng(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)


def _correct_flags(records: Sequence[Record], key: str = "is_correct") -> List[int]:
    return [1 if r.get(key) else 0 for r in records]


# ------------------------------------------------------------------- accuracy
def accuracy(records: Sequence[Record]) -> float:
    """Fraction of records whose final answer was correct."""
    if not records:
        return 0.0
    flags = _correct_flags(records)
    return sum(flags) / len(flags)


def vote_accuracy(records: Sequence[Record]) -> float:
    """Accuracy of the majority vote over each record's samples."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("vote_correct")) / len(records)


# --------------------------------------------------------------------- pass@k
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the probability that k of n samples contain a success.

    `1 - C(n-c, k) / C(n, k)`, evaluated as a product to stay numerically stable
    for large n.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        raise ValueError(f"k must be <= n, got k={k}, n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def pass_at_k_records(records: Sequence[Record], k: int) -> float:
    """Mean pass@k over records, skipping any with fewer than k samples."""
    values = [
        pass_at_k(int(r.get("n_samples_graded") or 0), int(r.get("n_correct_samples") or 0), k)
        for r in records
        if int(r.get("n_samples_graded") or 0) >= k
    ]
    if not values:
        return float("nan")
    return sum(values) / len(values)


def pass_at_k_curve(records: Sequence[Record], ks: Sequence[int]) -> Dict[str, Any]:
    """pass@k for each k, with the number of records each estimate is based on."""
    out: Dict[str, Any] = {"ks": [], "values": [], "n_records": []}
    for k in ks:
        eligible = [r for r in records if int(r.get("n_samples_graded") or 0) >= k]
        if not eligible:
            continue
        out["ks"].append(int(k))
        out["values"].append(pass_at_k_records(eligible, k))
        out["n_records"].append(len(eligible))
    return out


# ------------------------------------------------------------- majority vote @k
def _cluster_vote(
    answers: Sequence[Optional[str]], answer_type: str, choices: Optional[Sequence[str]]
) -> Optional[str]:
    """Equivalence-aware plurality, falling back to exact string counting.

    The lazy import keeps `metrics` free of a hard dependency on the grader, so
    metrics can be computed from a JSONL file in isolation.
    """
    try:
        from .answers import majority_vote

        winner, _info = majority_vote(answers, answer_type, choices)
        return winner
    except Exception:  # noqa: BLE001 - degrade rather than fail
        counts: Dict[str, int] = {}
        for a in answers:
            if a is None:
                continue
            counts[str(a)] = counts.get(str(a), 0) + 1
        if not counts:
            return None
        return max(counts, key=lambda key: (counts[key], -list(counts).index(key)))


def majority_vote_at_k(
    sample_correct: Sequence[bool],
    sample_answers: Sequence[Optional[str]],
    k: int,
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
    indices: Optional[Sequence[int]] = None,
) -> Optional[bool]:
    """Was the plurality answer over k samples correct?

    Correctness of the winner is read off `sample_correct` for a member of the
    winning cluster, so no grader call is needed here: `sample_correct[i]` is
    already the judgement for `sample_answers[i]`.

    Returns None when there are fewer than k samples or no parseable answer,
    which the callers treat as "no estimate" rather than "wrong".
    """
    n = min(len(sample_answers), len(sample_correct))
    if n < k or k <= 0:
        return None
    positions = list(indices) if indices is not None else list(range(k))
    positions = [p for p in positions if 0 <= p < n][:k]
    if not positions:
        return None
    answers = [sample_answers[p] for p in positions]
    winner = _cluster_vote(answers, answer_type, choices)
    if winner is None:
        return None
    for p in positions:
        if sample_answers[p] is not None and str(sample_answers[p]) == str(winner):
            return bool(sample_correct[p])
    return None


def majority_accuracy_curve(
    records: Sequence[Record],
    ks: Sequence[int],
    n_bootstrap: int = 0,
    seed: int = 0,
    n_draws: int = DEFAULT_N_SUBSET_DRAWS,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Majority-vote accuracy as a function of the number of samples k.

    When a record has n > k samples there are many possible size-k subsets, so
    the estimate averages over `n_draws` subsets drawn **without replacement**
    (drawing with replacement would let one chain vote twice and inflate
    agreement). Draws use a fixed seed, so the curve is reproducible.

    With `n_bootstrap > 0`, a bootstrap over *records* adds a confidence band;
    records are the unit of resampling because they are the population we
    generalise to.
    """
    rng = _rng(seed)
    out: Dict[str, Any] = {
        "ks": [],
        "values": [],
        "n_records": [],
        "ci_low": [],
        "ci_high": [],
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "n_draws": n_draws,
    }
    for k in ks:
        per_record: List[float] = []
        for r in records:
            answers = r.get("sample_answers") or []
            correct = r.get("sample_correct") or []
            n = min(len(answers), len(correct))
            if n < k:
                continue
            answer_type = str(r.get("answer_type") or "math")
            choices = r.get("choices")
            if n == k:
                verdicts = [majority_vote_at_k(correct, answers, k, answer_type, choices)]
            else:
                verdicts = []
                for _ in range(n_draws):
                    picks = rng.choice(n, size=k, replace=False)
                    verdicts.append(
                        majority_vote_at_k(
                            correct, answers, k, answer_type, choices, indices=[int(p) for p in picks]
                        )
                    )
            usable = [v for v in verdicts if v is not None]
            # An unparseable vote counts as incorrect, matching how `is_correct`
            # treats a missing answer.
            per_record.append(
                sum(1 for v in usable if v) / len(verdicts) if verdicts else 0.0
            )
        if not per_record:
            continue
        out["ks"].append(int(k))
        value = sum(per_record) / len(per_record)
        out["values"].append(value)
        out["n_records"].append(len(per_record))
        if n_bootstrap > 0:
            low, high, _point = bootstrap_ci(
                per_record, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            )
            out["ci_low"].append(low)
            out["ci_high"].append(high)
        else:
            out["ci_low"].append(None)
            out["ci_high"].append(None)
    return out


# ----------------------------------------------------------------- token costs
def token_cost(records: Sequence[Record]) -> Dict[str, Any]:
    """Total and per-example token usage. The x-axis of every budget comparison."""
    if not records:
        return {
            "n": 0,
            "total_prompt": 0,
            "total_completion": 0,
            "total": 0,
            "mean_prompt": 0.0,
            "mean_completion": 0.0,
            "mean_total": 0.0,
            "mean_samples": 0.0,
            "mean_calls": 0.0,
            "mean_latency_s": 0.0,
        }
    prompt = sum(int(r.get("tokens_prompt") or 0) for r in records)
    completion = sum(int(r.get("tokens_completion") or 0) for r in records)
    n = len(records)
    return {
        "n": n,
        "total_prompt": prompt,
        "total_completion": completion,
        "total": prompt + completion,
        "mean_prompt": prompt / n,
        "mean_completion": completion / n,
        "mean_total": (prompt + completion) / n,
        "mean_samples": sum(int(r.get("n_samples") or 0) for r in records) / n,
        "mean_calls": sum(int(r.get("n_calls") or 0) for r in records) / n,
        "mean_latency_s": sum(float(r.get("latency_s") or 0.0) for r in records) / n,
    }


def tokens_per_correct(records: Sequence[Record]) -> float:
    """Completion tokens spent per correct answer; inf when nothing is correct.

    The efficiency number that stops "sample 64 times" from looking free.
    """
    n_correct = sum(_correct_flags(records))
    total = sum(int(r.get("tokens_completion") or 0) for r in records)
    if n_correct == 0:
        return float("inf")
    return total / n_correct


def accuracy_vs_compute(
    records_by_setting: Dict[str, Sequence[Record]],
    x_key: str = "tokens_completion",
) -> Dict[str, Any]:
    """Points for the accuracy-versus-compute plot, one per setting."""
    out: Dict[str, Any] = {"labels": [], "x": [], "y": [], "n": [], "x_key": x_key}
    for label, records in records_by_setting.items():
        if not records:
            continue
        out["labels"].append(label)
        out["x"].append(sum(int(r.get(x_key) or 0) for r in records) / len(records))
        out["y"].append(accuracy(records))
        out["n"].append(len(records))
    return out


# ------------------------------------------------------------ confidence intervals
def wilson_ci(
    n_success: int, n: int, alpha: float = DEFAULT_ALPHA
) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because accuracies near 0 or 1 are
    common on hard benchmarks, where the Wald interval can leave [0, 1].
    """
    if n <= 0:
        return (0.0, 0.0)
    z = _normal_quantile(1.0 - alpha / 2.0)
    p = n_success / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation).

    Hand-rolled to avoid scipy; accurate to ~1e-9 over the range we use.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (low, high, point_estimate)."""
    import numpy as np

    if not values:
        return (0.0, 0.0, 0.0)
    array = np.asarray(list(values), dtype=float)
    point = float(statistic(array.tolist()))
    if len(array) == 1 or n_bootstrap <= 0:
        return (point, point, point)
    rng = _rng(seed)
    picks = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    if statistic is statistics.fmean:
        stats = array[picks].mean(axis=1)
    else:
        stats = np.array([statistic(array[row].tolist()) for row in picks])
    low = float(np.quantile(stats, alpha / 2.0))
    high = float(np.quantile(stats, 1.0 - alpha / 2.0))
    return (low, high, point)


def accuracy_ci(
    records: Sequence[Record], alpha: float = DEFAULT_ALPHA
) -> Tuple[float, float]:
    """Wilson interval for a run's accuracy: cheap, deterministic, seed-free."""
    flags = _correct_flags(records)
    return wilson_ci(sum(flags), len(flags), alpha=alpha)


# --------------------------------------------------------------- paired testing
def mcnemar(
    a: Sequence[bool],
    b: Sequence[bool],
    exact: bool = True,
    correction: bool = True,
) -> Dict[str, Any]:
    """McNemar's test on paired binary outcomes.

    `b01` counts examples where a is wrong and b is right; `b10` the reverse.
    Concordant pairs carry no information about a difference and are ignored,
    which is exactly why this test is the right one for paired accuracies.

    Uses the exact two-sided binomial p-value when asked, or whenever the number
    of discordant pairs is small (< 25), where the chi-square approximation is
    unreliable.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs must have equal length, got {len(a)} and {len(b)}")
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))
    n_disc = b01 + b10
    out: Dict[str, Any] = {
        "n_pairs": len(a),
        "b01": b01,
        "b10": b10,
        "n_discordant": n_disc,
        "method": "exact_binomial",
        "statistic": None,
        "p_value": 1.0,
    }
    if n_disc == 0:
        return out

    if exact or n_disc < 25:
        smaller = min(b01, b10)
        tail = sum(math.comb(n_disc, i) for i in range(smaller + 1)) / (2.0**n_disc)
        out["p_value"] = min(1.0, 2.0 * tail)
        out["statistic"] = float(smaller)
        return out

    diff = abs(b01 - b10)
    if correction:
        diff = max(0.0, diff - 1.0)  # Edwards' continuity correction
    chi2 = (diff * diff) / n_disc
    out["method"] = "chi2_1df" + ("_continuity_corrected" if correction else "")
    out["statistic"] = float(chi2)
    # Survival function of chi-square with 1 df, without scipy.
    out["p_value"] = float(math.erfc(math.sqrt(chi2 / 2.0)))
    return out


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired bootstrap on the per-example difference b - a.

    Example *indices* are resampled, so the two methods are always compared on
    the same resampled examples. Resampling each method independently would
    destroy the pairing and inflate the interval.
    """
    import numpy as np

    if len(a) != len(b):
        raise ValueError(f"paired inputs must have equal length, got {len(a)} and {len(b)}")
    n = len(a)
    out: Dict[str, Any] = {
        "n_pairs": n,
        "diff": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "p_value": 1.0,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "alpha": alpha,
    }
    if n == 0:
        return out
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    diffs = arr_b - arr_a
    observed = float(diffs.mean())
    out["diff"] = observed
    out["mean_a"] = float(arr_a.mean())
    out["mean_b"] = float(arr_b.mean())
    if n_bootstrap <= 0 or n == 1:
        out["ci_low"] = out["ci_high"] = observed
        return out

    rng = _rng(seed)
    picks = rng.integers(0, n, size=(n_bootstrap, n))
    resampled = diffs[picks].mean(axis=1)
    out["ci_low"] = float(np.quantile(resampled, alpha / 2.0))
    out["ci_high"] = float(np.quantile(resampled, 1.0 - alpha / 2.0))
    frac_le = float((resampled <= 0).mean())
    frac_ge = float((resampled >= 0).mean())
    p = 2.0 * min(frac_le, frac_ge)
    # A bootstrap cannot resolve p below 1/n_bootstrap; reporting 0 would
    # overstate the evidence.
    out["p_value"] = float(min(1.0, max(p, 1.0 / n_bootstrap)))
    return out


def _pair_key(record: Record) -> Tuple[Any, ...]:
    """Alignment key that is independent of model and strategy.

    uids include the strategy, so two strategies never share one; pairing is done
    on the measurement's coordinates instead.
    """
    example_id = record.get("example_id")
    if example_id is None:
        example_id = record.get("index")
    return (record.get("dataset"), record.get("seed"), example_id)


def compare_strategies(
    records_a: Sequence[Record],
    records_b: Sequence[Record],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
    label_a: str = "a",
    label_b: str = "b",
    strict: bool = False,
) -> Dict[str, Any]:
    """Paired comparison of two strategies on the examples they share.

    Refuses to invent a comparison: unmatched examples are reported in
    `n_only_a`/`n_only_b` and excluded, and an empty intersection yields
    `n_paired=0` with a warning (or an exception when `strict`).
    """
    map_a = {_pair_key(r): r for r in records_a}
    map_b = {_pair_key(r): r for r in records_b}
    shared = [k for k in map_a if k in map_b]
    shared.sort(key=lambda t: tuple(str(x) for x in t))

    out: Dict[str, Any] = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": len(shared),
        "n_only_a": len(map_a) - len(shared),
        "n_only_b": len(map_b) - len(shared),
        "n_a": len(map_a),
        "n_b": len(map_b),
        "seed": seed,
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
    }
    if not shared:
        message = (
            f"no shared examples between {label_a!r} ({len(map_a)} records) and "
            f"{label_b!r} ({len(map_b)} records); refusing to compare misaligned sets"
        )
        if strict:
            raise ValueError(message)
        log.warning(message)
        out.update({"accuracy_a": accuracy(records_a), "accuracy_b": accuracy(records_b)})
        return out
    if out["n_only_a"] or out["n_only_b"]:
        log.warning(
            "comparing %r vs %r on %d shared examples (%d only in %s, %d only in %s)",
            label_a, label_b, len(shared), out["n_only_a"], label_a,
            out["n_only_b"], label_b,
        )

    flags_a = [1 if map_a[k].get("is_correct") else 0 for k in shared]
    flags_b = [1 if map_b[k].get("is_correct") else 0 for k in shared]
    ci_a = wilson_ci(sum(flags_a), len(flags_a), alpha)
    ci_b = wilson_ci(sum(flags_b), len(flags_b), alpha)

    out.update(
        {
            "accuracy_a": sum(flags_a) / len(flags_a),
            "accuracy_b": sum(flags_b) / len(flags_b),
            "ci_a": [ci_a[0], ci_a[1]],
            "ci_b": [ci_b[0], ci_b[1]],
            "delta": sum(flags_b) / len(flags_b) - sum(flags_a) / len(flags_a),
            "mcnemar": mcnemar(
                [bool(x) for x in flags_a], [bool(x) for x in flags_b], exact=True
            ),
            "paired_bootstrap": paired_bootstrap(
                flags_a, flags_b, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
            ),
            "tokens_per_correct_a": tokens_per_correct([map_a[k] for k in shared]),
            "tokens_per_correct_b": tokens_per_correct([map_b[k] for k in shared]),
        }
    )
    return out


def holm_bonferroni(
    p_values: Sequence[float], alpha: float = DEFAULT_ALPHA
) -> Dict[str, Any]:
    """Holm-Bonferroni step-down adjustment over a family of comparisons.

    Any table that compares several methods against one baseline is a family, and
    the family size must be stated alongside the adjusted values.
    """
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * p_values[index]
        running = max(running, min(1.0, value))
        adjusted[index] = running
    return {
        "family_size": n,
        "alpha": alpha,
        "p_adjusted": adjusted,
        "reject": [p <= alpha for p in adjusted],
    }


# ------------------------------------------------- empirical logit (METHOD_SPEC 5.1)
#: Haldane-Anscombe continuity correction. 0.5 is the value the spec fixes.
HALDANE = 0.5


def haldane_logit(k: int, n: int, correction: float = HALDANE) -> float:
    """The empirical logit `log((k + c) / (n - k + c))`.

    All variance modelling happens on this scale rather than on the raw
    proportion, because a proportion is bounded and its variance is a function of
    its mean, which would make an additive variance decomposition of `p_hat`
    meaningless. The correction keeps the transform finite at `k = 0` and `k = n`.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    return math.log((k + correction) / (n - k + correction))


def haldane_logit_var(k: int, n: int, correction: float = HALDANE) -> float:
    """Sampling variance of `haldane_logit`: `1/(k+c) + 1/(n-k+c)`.

    Worth understanding rather than just calling, because this expression drove
    the spec's budget decision. At an interior cell (`p_hat = 0.5`) it halves when
    `N` doubles: 0.16 at N=24, 0.08 at N=48. At a **saturated** cell (`k = 0` or
    `k = n`) it is `1/c + 1/(n + c)`, which is approximately 2.0 and essentially
    flat in `N` -- 2.04 at N=24, 2.02 at N=48. Saturated cells therefore dominate
    the sampling-noise floor and cannot be rescued by drawing more samples, which
    is why METHOD_SPEC section 8.4 spends the headroom on items rather than on N.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    return 1.0 / (k + correction) + 1.0 / (n - k + correction)


def is_saturated(k: int, n: int) -> bool:
    """True when a cell carries no within-cell information (`k = 0` or `k = n`)."""
    return k <= 0 or k >= n


# ------------------------------------------------------------ rank correlation
def ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, ties shared. The basis of Spearman's rho.

    Hand-rolled rather than taken from scipy so that a Kaggle image upgrade cannot
    change a number in the paper.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for position in range(i, j + 1):
            out[order[position]] = shared
        i = j + 1
    return out


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError(f"paired inputs must have equal length, got {len(x)}, {len(y)}")
    n = len(x)
    if n < 2:
        return float("nan")
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    dx = [xi - mean_x for xi in x]
    dy = [yi - mean_y for yi in y]
    denom = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denom == 0.0:
        # One side is constant: the correlation is undefined, and returning 0
        # would quietly report "no relationship" for a degenerate measurement.
        return float("nan")
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho as Pearson on average ranks (correct under ties).

    Ties matter here: on an easy benchmark a large fraction of items sit at
    `p_hat = 0` or `p_hat = 1`, so the tie-corrected form is the only defensible
    one.
    """
    if len(x) < 2:
        return float("nan")
    return pearson(ranks(x), ranks(y))


# --------------------------------------------------------------------- BCa bootstrap
def bca_ci(
    values: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """Bias-corrected and accelerated bootstrap interval over `values`.

    `values` is the resampling unit -- items for every interval in this paper, or
    templates for GSM-Symbolic, where instances from one template are not
    independent. BCa rather than a plain percentile interval because the
    statistics here (variance shares, ratios of correlations) are skewed and
    biased, which is precisely the case percentile intervals get wrong.

    Falls back to the percentile interval when the acceleration is undefined (a
    constant jackknife, which happens when every element is identical).
    """
    import numpy as np

    n = len(values)
    out: Dict[str, Any] = {
        "point": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n": n,
        "n_bootstrap": int(n_bootstrap),
        "alpha": alpha,
        "seed": seed,
        "method": "bca",
    }
    if n == 0:
        return out
    point = float(statistic(list(values)))
    out["point"] = point
    if n < 3 or n_bootstrap <= 0 or not math.isfinite(point):
        out["ci_low"] = out["ci_high"] = point
        out["method"] = "degenerate"
        return out

    rng = _rng(seed)
    replicates = np.empty(n_bootstrap, dtype=float)
    picks = rng.integers(0, n, size=(n_bootstrap, n))
    for b in range(n_bootstrap):
        replicates[b] = statistic([values[i] for i in picks[b]])
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 10:
        out["ci_low"] = out["ci_high"] = point
        out["method"] = "degenerate"
        return out

    # Bias correction: how far the point estimate sits from the median replicate.
    prop_below = float((finite < point).mean())
    prop_below = min(max(prop_below, 1.0 / (finite.size + 1)), 1.0 - 1.0 / (finite.size + 1))
    z0 = _normal_quantile(prop_below)

    # Acceleration from the jackknife's third moment.
    jack = np.array(
        [statistic([values[i] for i in range(n) if i != leave]) for leave in range(n)],
        dtype=float,
    )
    jack = jack[np.isfinite(jack)]
    centred = jack.mean() - jack
    denom = 6.0 * float((centred**2).sum()) ** 1.5
    accel = float((centred**3).sum()) / denom if denom > 0 else 0.0

    z_lo = _normal_quantile(alpha / 2.0)
    z_hi = _normal_quantile(1.0 - alpha / 2.0)
    lo = _bca_adjust(z0, accel, z_lo)
    hi = _bca_adjust(z0, accel, z_hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        out["method"] = "percentile"
        lo, hi = alpha / 2.0, 1.0 - alpha / 2.0
    out["ci_low"] = float(np.quantile(finite, lo))
    out["ci_high"] = float(np.quantile(finite, hi))
    out["bias_correction_z0"] = z0
    out["acceleration"] = accel
    out["n_finite_replicates"] = int(finite.size)
    return out


def _bca_adjust(z0: float, accel: float, z: float) -> float:
    """The BCa percentile for a nominal normal quantile `z`."""
    denom = 1.0 - accel * (z0 + z)
    if denom == 0.0:
        return float("nan")
    adjusted = z0 + (z0 + z) / denom
    return _normal_cdf(adjusted)


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ------------------------------------------------------------- permutation test
def permutation_test(
    statistic: Callable[[Any], float],
    observed_input: Any,
    permute: Callable[[Any, Any], Any],
    n_permutations: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
    alternative: str = "greater",
) -> Dict[str, Any]:
    """Generic one- or two-sided permutation test.

    `permute(observed_input, rng)` must return a permuted copy of the input under
    the null. The p-value uses the `(1 + #at-least-as-extreme) / (1 + B)` form, so
    it can never be reported as exactly 0 -- a permutation test cannot resolve
    below `1/(B+1)` and claiming otherwise overstates the evidence.
    """
    import numpy as np

    rng = _rng(seed)
    observed = float(statistic(observed_input))
    draws = np.array(
        [statistic(permute(observed_input, rng)) for _ in range(max(0, n_permutations))],
        dtype=float,
    )
    draws = draws[np.isfinite(draws)]
    n = int(draws.size)
    if n == 0 or not math.isfinite(observed):
        return {
            "observed": observed,
            "p_value": 1.0,
            "n_permutations": n,
            "seed": seed,
            "alternative": alternative,
        }
    if alternative == "greater":
        n_extreme = int((draws >= observed).sum())
    elif alternative == "less":
        n_extreme = int((draws <= observed).sum())
    elif alternative == "two-sided":
        centre = float(draws.mean())
        n_extreme = int((np.abs(draws - centre) >= abs(observed - centre)).sum())
    else:
        raise ValueError(
            f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}"
        )
    return {
        "observed": observed,
        "null_mean": float(draws.mean()),
        "null_std": float(draws.std(ddof=1)) if n > 1 else 0.0,
        "p_value": (1.0 + n_extreme) / (1.0 + n),
        "n_permutations": n,
        "seed": seed,
        "alternative": alternative,
    }


def mann_whitney_u(a: Sequence[float], b: Sequence[float]) -> Dict[str, Any]:
    """Tie-corrected Mann-Whitney U with a normal-approximation p-value.

    Used for the mechanism prediction that mode reorderings concentrate on
    small-margin items: the two groups (reordered / not reordered) are unpaired
    and unequal in size, and the margin distribution is far from normal.
    """
    n_a, n_b = len(a), len(b)
    out: Dict[str, Any] = {
        "n_a": n_a,
        "n_b": n_b,
        "u": float("nan"),
        "p_value": 1.0,
        "effect_size_auc": float("nan"),
    }
    if n_a == 0 or n_b == 0:
        return out
    combined = list(a) + list(b)
    r = ranks(combined)
    rank_sum_a = sum(r[:n_a])
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    out["u"] = float(u_a)
    # AUC = P(random a > random b), the interpretable effect size.
    out["effect_size_auc"] = float(u_a / (n_a * n_b))
    mean_u = n_a * n_b / 2.0
    counts: Dict[float, int] = {}
    for value in combined:
        counts[value] = counts.get(value, 0) + 1
    n = n_a + n_b
    tie_term = sum(t**3 - t for t in counts.values())
    var_u = (n_a * n_b / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0:
        return out
    z = (u_a - mean_u) / math.sqrt(var_u)
    out["z"] = float(z)
    out["p_value"] = float(2.0 * (1.0 - _normal_cdf(abs(z))))
    return out


def jaccard(a: Sequence[Any], b: Sequence[Any]) -> float:
    """|A and B| / |A or B|. Returns nan for two empty sets, never 1.0."""
    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return float("nan")
    return len(set_a & set_b) / len(union)


# ------------------------------------------------------------------- summaries
def summarize_run(records: Sequence[Record], alpha: float = DEFAULT_ALPHA) -> Dict[str, Any]:
    """The per-run row used by every table and figure."""
    if not records:
        return {"n": 0, "accuracy": 0.0}
    first = records[0]
    flags = _correct_flags(records)
    ci = wilson_ci(sum(flags), len(flags), alpha)
    costs = token_cost(records)
    max_samples = max(int(r.get("n_samples_graded") or 0) for r in records)
    n_unparsed = sum(1 for r in records if r.get("pred_answer") is None)
    summary: Dict[str, Any] = {
        "dataset": first.get("dataset"),
        "model": first.get("model"),
        "strategy": first.get("strategy"),
        "seed": first.get("seed"),
        "config_hash": first.get("config_hash"),
        "n": len(records),
        "n_correct": sum(flags),
        "accuracy": sum(flags) / len(flags),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "ci_method": "wilson",
        "alpha": alpha,
        "vote_accuracy": vote_accuracy(records),
        "max_samples": max_samples,
        "tokens_per_correct": tokens_per_correct(records),
        "n_errors": sum(1 for r in records if r.get("error")),
        "n_extraction_failures": n_unparsed,
        "extraction_failure_rate": n_unparsed / len(records),
        "grader_version": first.get("grader_version"),
        "grader_backend": first.get("grader_backend"),
    }
    summary.update({f"tokens_{k}": v for k, v in costs.items() if k != "n"})
    if max_samples > 1:
        summary["pass_at_1"] = pass_at_k_records(records, 1)
        summary["pass_at_max"] = pass_at_k_records(records, max_samples)
    return summary


def aggregate_seeds(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean, standard deviation and CI of accuracy across seeds.

    Across-seed spread is a separate source of uncertainty from the within-run
    interval, and on small benchmarks it is often the larger one, so both are
    reported.
    """
    if not summaries:
        return {"n_seeds": 0}
    accuracies = [float(s.get("accuracy") or 0.0) for s in summaries]
    n_seeds = len(accuracies)
    mean = statistics.fmean(accuracies)
    stdev = statistics.stdev(accuracies) if n_seeds > 1 else 0.0
    out: Dict[str, Any] = {
        "dataset": summaries[0].get("dataset"),
        "model": summaries[0].get("model"),
        "strategy": summaries[0].get("strategy"),
        "seeds": [s.get("seed") for s in summaries],
        "n_seeds": n_seeds,
        "accuracy_mean": mean,
        "accuracy_std": stdev,
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "n_total": sum(int(s.get("n") or 0) for s in summaries),
    }
    if n_seeds > 1:
        # Standard error over seeds, with the t-like 1.96 factor left explicit.
        half = 1.96 * stdev / math.sqrt(n_seeds)
        out["accuracy_ci_low"] = max(0.0, mean - half)
        out["accuracy_ci_high"] = min(1.0, mean + half)
    else:
        out["accuracy_ci_low"] = out["accuracy_ci_high"] = mean
    return out


def group_records(
    records: Iterable[Record], keys: Sequence[str] = ("dataset", "model", "strategy", "seed")
) -> Dict[Tuple[Any, ...], List[Record]]:
    """Bucket records by the given fields, for per-cell summaries."""
    out: Dict[Tuple[Any, ...], List[Record]] = {}
    for r in records:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out
