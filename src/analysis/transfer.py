"""Algorithms 2 and 3: does an item-difficulty estimate transfer? (METHOD_SPEC 5.3-5.4)

Two statistics, and in both cases the *null* is what makes the number mean
anything.

**Disattenuated difficulty transfer.** The raw Spearman correlation between
`p_hat` under two configurations is attenuated by sampling noise, so a low raw
correlation is not evidence of a configuration effect: re-estimating the same
configuration would also correlate imperfectly. The seed replicates give the
reliability `r_mm` -- how much correlation sampling noise alone permits -- and
dividing by it removes the attenuation. `r_mm` is the denominator of the headline
statistic, so its own uncertainty is propagated into every interval rather than
being treated as known.

**Hard-subset overlap.** The bottom-quartile "hard subset" under one configuration
is compared against the hard subset under another. The comparison is *not* against
1.0, which no finite sample would reach; it is against `J_seed`, the overlap
between two hard subsets estimated from independent seeds of the same
configuration. `J_seed - J_config` is the excess instability attributable to the
configuration rather than to re-estimation noise.

**On the significance test for the overlap, where this module departs from the
spec.** METHOD_SPEC 5.4 step 5 asks for "a permutation test that shuffles the
config label within item, one-sided". That test has no power against the hypothesis
of interest, and the reason is structural rather than a matter of sample size:
permuting a single item's values across configuration columns leaves that item's
marginal distribution untouched, and every column remains an equally noisy estimate
of the same item difficulty. The Jaccard statistic is therefore very nearly
distribution-invariant under the permutation, so the null lands on the observed
value and the p-value is uninformative whatever the truth is. It is computed and
reported as `permutation` for faithfulness, with a flag, and
`test_transfer.py` asserts that it is centred on the observed value so the defect
cannot be mistaken for a null result.

What answers the actual question is a **re-estimation null**: hold each item's
difficulty fixed at its across-configuration mean, redraw as many independent
columns as there are configurations at the same `N`, and ask how much overlap pure
re-estimation noise produces. That is the same null `J_seed` estimates, but with the
column count and pair count matched to `J_config`, and it is reported as
`null_test`. Its centre should land near `J_seed`, which is an internal consistency
check worth reading in the output.

A note on the disattenuation formula. METHOD_SPEC 5.3 step 2 writes
`rho_disatt = rho_raw / r_mm`. The classical correction is
`rho_raw / sqrt(r_xx * r_yy)`; for two configurations of the *same* model both
reliabilities are `r_mm`, so `sqrt(r_mm * r_mm) = r_mm` and the spec's form is
exactly the classical one. For **model** pairs the two reliabilities differ, so
this module uses the square-root form there. That is a generalisation of the spec,
not a departure from it, and it is flagged in the output as
`disattenuation="sqrt(r_x*r_y)"`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import (
    DEFAULT_ALPHA,
    DEFAULT_N_BOOTSTRAP,
    bca_ci,
    jaccard,
    permutation_test,
    spearman,
)
from ..utils import stable_hash
from .corpus import Corpus

log = logging.getLogger(__name__)

#: The "hard subset" quantile from METHOD_SPEC 5.4.
HARD_QUANTILE = 0.25


def model_family(model: str) -> str:
    """The pretraining lineage, taken as the HuggingFace org.

    The scientific requirement in METHOD_SPEC 1.2 is only that family 2 has a
    different lineage from Qwen, and the org prefix carries that faithfully for
    every candidate the spec lists (Qwen, meta-llama, microsoft, google, mistralai)
    without hardcoding a model list -- which matters because the model ladder is
    being revised.
    """
    text = str(model or "")
    return text.split("/")[0] if "/" in text else text.split("-")[0]


# --------------------------------------------------------------------- reliability
@dataclass
class Reliability:
    """`r_mm`: the correlation ceiling that sampling noise alone permits."""

    dataset: str
    subset: Optional[str]
    model: str
    config: str
    r_mm: float
    ci_low: float
    ci_high: float
    n_seed_pairs: int
    seeds: List[int]
    n_items: int
    per_pair: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "config": self.config,
            "r_mm": self.r_mm,
            "r_mm_ci_low": self.ci_low,
            "r_mm_ci_high": self.ci_high,
            "n_seed_pairs": self.n_seed_pairs,
            "seeds": ",".join(str(s) for s in self.seeds),
            "n_items": self.n_items,
        }


def _aligned(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    columns: Sequence[Tuple[str, str, int]],
) -> Tuple[List[str], List[List[float]]]:
    """`p_hat` vectors for several (model, config, seed) columns on shared items.

    Only items present in every requested column are returned. Correlating
    partially-overlapping item sets would compare different populations, which is
    exactly the mistake that makes a transfer correlation uninterpretable.
    """
    per_column: List[Dict[str, float]] = []
    for model, config, seed in columns:
        index = {
            cell.item_id: cell.p_hat
            for cell in corpus.cells.values()
            if cell.dataset == dataset
            and cell.subset == subset
            and cell.model == model
            and cell.config == config
            and cell.seed == seed
            and cell.n > 0
        }
        per_column.append(index)
    if not per_column:
        return [], []
    shared = set(per_column[0])
    for index in per_column[1:]:
        shared &= set(index)
    items = sorted(shared)
    return items, [[index[item] for item in items] for index in per_column]


def _column_sample_counts(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    columns: Sequence[Tuple[str, str, int]],
) -> List[int]:
    """Modal `N` per (model, config, seed) column, for the re-estimation null.

    The mode rather than the mean because `N` is a design constant that only varies
    when a cell was truncated by the wall-clock guard; the mode is the intended
    value and a stray short cell should not shift the null.
    """
    counts: List[int] = []
    for model, config, seed in columns:
        sizes: Dict[int, int] = {}
        for cell in corpus.cells.values():
            if (
                cell.dataset == dataset
                and cell.subset == subset
                and cell.model == model
                and cell.config == config
                and cell.seed == seed
                and cell.n > 0
            ):
                sizes[cell.n] = sizes.get(cell.n, 0) + 1
        counts.append(max(sizes, key=lambda k: sizes[k]) if sizes else 0)
    return counts


def measure_reliability(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    config: str = REFERENCE_CONFIG_ID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Optional[Reliability]:
    """Mean pairwise Spearman of `p_hat` across independent seeds of one config."""
    seeds = sorted(
        {
            cell.seed
            for cell in corpus.cells.values()
            if cell.dataset == dataset
            and cell.subset == subset
            and cell.model == model
            and cell.config == config
            and cell.n > 0
        }
    )
    if len(seeds) < 2:
        log.warning(
            "cannot measure reliability for %s on %s/%s at %s: %d seed(s) present. "
            "rho_disatt has no denominator without this arm.",
            model,
            dataset,
            subset,
            config,
            len(seeds),
        )
        return None

    items, vectors = _aligned(
        corpus, dataset, subset, [(model, config, s) for s in seeds]
    )
    if len(items) < 3:
        log.warning(
            "reliability for %s on %s/%s: only %d item(s) shared across seeds",
            model, dataset, subset, len(items),
        )
        return None

    pairs = list(combinations(range(len(seeds)), 2))
    per_pair = {
        f"s{seeds[a]}-s{seeds[b]}": spearman(vectors[a], vectors[b]) for a, b in pairs
    }

    def statistic(rows: Sequence[Tuple[float, ...]]) -> float:
        # Each element is one item's values across seeds; resampling items is the
        # clustering unit, exactly as for every other interval in the paper.
        if len(rows) < 3:
            return float("nan")
        columns = list(zip(*rows))
        values = [spearman(columns[a], columns[b]) for a, b in pairs]
        finite = [v for v in values if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    rows = [tuple(vector[i] for vector in vectors) for i in range(len(items))]
    interval = bca_ci(rows, statistic, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
    return Reliability(
        dataset=dataset,
        subset=subset,
        model=model,
        config=config,
        r_mm=float(interval["point"]),
        ci_low=float(interval["ci_low"]),
        ci_high=float(interval["ci_high"]),
        n_seed_pairs=len(pairs),
        seeds=seeds,
        n_items=len(items),
        per_pair=per_pair,
    )


def all_reliabilities(
    corpus: Corpus,
    config: str = REFERENCE_CONFIG_ID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    **kw: Any,
) -> Dict[Tuple[str, Optional[str], str], Reliability]:
    out: Dict[Tuple[str, Optional[str], str], Reliability] = {}
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            measured = measure_reliability(
                corpus, dataset, subset, model, config, n_bootstrap=n_bootstrap, **kw
            )
            if measured is not None:
                out[(dataset, subset, model)] = measured
    return out


# ------------------------------------------------------------ difficulty transfer
@dataclass
class TransferRow:
    """One `rho_raw` / `rho_disatt` pair, with the reliability that scaled it."""

    dataset: str
    subset: Optional[str]
    kind: str  # "config" | "model" | "family"
    label_a: str
    label_b: str
    model: str
    rho_raw: float
    rho_disatt: float
    reliability: float
    reliability_note: str
    ci_low: float
    ci_high: float
    n_items: int
    n_bootstrap: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "pair_kind": self.kind,
            "a": self.label_a,
            "b": self.label_b,
            "model": self.model,
            "rho_raw": self.rho_raw,
            "rho_disatt": self.rho_disatt,
            "rho_disatt_ci_low": self.ci_low,
            "rho_disatt_ci_high": self.ci_high,
            "r_mm": self.reliability,
            "disattenuation": self.reliability_note,
            "n_items": self.n_items,
            "n_bootstrap": self.n_bootstrap,
        }


def _disattenuated_row(
    dataset: str,
    subset: Optional[str],
    kind: str,
    label_a: str,
    label_b: str,
    model: str,
    x: Sequence[float],
    y: Sequence[float],
    denominator: float,
    note: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> TransferRow:
    rho_raw = spearman(x, y)
    rows = list(zip(x, y))

    def statistic(sample: Sequence[Tuple[float, float]]) -> float:
        if len(sample) < 3 or not math.isfinite(denominator) or denominator <= 0:
            return float("nan")
        xs = [r[0] for r in sample]
        ys = [r[1] for r in sample]
        raw = spearman(xs, ys)
        return raw / denominator if math.isfinite(raw) else float("nan")

    interval = bca_ci(rows, statistic, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
    disatt = (
        rho_raw / denominator
        if math.isfinite(rho_raw) and math.isfinite(denominator) and denominator > 0
        else float("nan")
    )
    return TransferRow(
        dataset=dataset,
        subset=subset,
        kind=kind,
        label_a=label_a,
        label_b=label_b,
        model=model,
        rho_raw=rho_raw,
        rho_disatt=disatt,
        reliability=denominator,
        reliability_note=note,
        ci_low=float(interval["ci_low"]),
        ci_high=float(interval["ci_high"]),
        n_items=len(rows),
        n_bootstrap=int(n_bootstrap),
    )


def transfer_table(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    reference_config: str = REFERENCE_CONFIG_ID,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
    reliabilities: Optional[Dict[Tuple[str, Optional[str], str], Reliability]] = None,
) -> List[TransferRow]:
    """`rho_raw` and `rho_disatt` for config pairs, model pairs and family pairs."""
    configs = list(configs) if configs else [
        c for c in corpus.configs() if c in PRIMARY_CONFIG_IDS
    ] or corpus.configs()
    reliabilities = (
        reliabilities
        if reliabilities is not None
        else all_reliabilities(
            corpus, reference_config, n_bootstrap=min(n_bootstrap, 2000), alpha=alpha
        )
    )
    rows: List[TransferRow] = []

    for dataset, subset in corpus.benchmarks():
        # --- configuration pairs, within model: the headline statistic
        for model in corpus.models():
            reliability = reliabilities.get((dataset, subset, model))
            denominator = reliability.r_mm if reliability else float("nan")
            for config_a, config_b in combinations(configs, 2):
                items, vectors = _aligned(
                    corpus, dataset, subset,
                    [(model, config_a, seed), (model, config_b, seed)],
                )
                if len(items) < 3:
                    continue
                rows.append(
                    _disattenuated_row(
                        dataset, subset, "config", config_a, config_b, model,
                        vectors[0], vectors[1], denominator,
                        "rho_raw / r_mm (METHOD_SPEC 5.3; both configs share r_mm)",
                        n_bootstrap, alpha, bootstrap_seed,
                    )
                )

        # --- model pairs at the reference configuration
        for model_a, model_b in combinations(corpus.models(), 2):
            rel_a = reliabilities.get((dataset, subset, model_a))
            rel_b = reliabilities.get((dataset, subset, model_b))
            denominator = (
                math.sqrt(rel_a.r_mm * rel_b.r_mm)
                if rel_a and rel_b and rel_a.r_mm > 0 and rel_b.r_mm > 0
                else float("nan")
            )
            items, vectors = _aligned(
                corpus, dataset, subset,
                [(model_a, reference_config, seed), (model_b, reference_config, seed)],
            )
            if len(items) < 3:
                continue
            kind = (
                "family"
                if model_family(model_a) != model_family(model_b)
                else "model"
            )
            rows.append(
                _disattenuated_row(
                    dataset, subset, kind, model_a, model_b, f"{model_a}|{model_b}",
                    vectors[0], vectors[1], denominator,
                    "rho_raw / sqrt(r_a*r_b) (reliabilities differ across models)",
                    n_bootstrap, alpha, bootstrap_seed,
                )
            )
    return rows


def transfer_summary(rows: Sequence[TransferRow], kind: str = "config") -> Dict[str, Any]:
    """Prediction P2: mean `rho_disatt` over config pairs below 0.85, `r_mm` above 0.95."""
    selected = [r for r in rows if r.kind == kind and math.isfinite(r.rho_disatt)]
    if not selected:
        return {"kind": kind, "n_pairs": 0}
    disatt = [r.rho_disatt for r in selected]
    raw = [r.rho_raw for r in selected if math.isfinite(r.rho_raw)]
    reliabilities = [
        r.reliability for r in selected if math.isfinite(r.reliability)
    ]
    mean_disatt = sum(disatt) / len(disatt)
    mean_reliability = (
        sum(reliabilities) / len(reliabilities) if reliabilities else float("nan")
    )
    return {
        "kind": kind,
        "n_pairs": len(selected),
        "mean_rho_raw": sum(raw) / len(raw) if raw else float("nan"),
        "mean_rho_disatt": mean_disatt,
        "min_rho_disatt": min(disatt),
        "max_rho_disatt": max(disatt),
        "mean_r_mm": mean_reliability,
        "prediction": "P2",
        "statement": "rho_disatt for config pairs below 0.85 while r_mm exceeds 0.95",
        "rho_disatt_below_085": bool(mean_disatt < 0.85),
        "r_mm_above_095": bool(mean_reliability > 0.95),
        "supported": bool(mean_disatt < 0.85 and mean_reliability > 0.95),
    }


# ------------------------------------------------------------- hard-subset overlap
def hard_subset(
    items: Sequence[str], p_hat: Sequence[float], quantile: float = HARD_QUANTILE
) -> List[str]:
    """The bottom-`quantile` items by `p_hat`, with deterministic tie-breaking.

    Ties are the norm rather than the exception: on GSM8K a strong model saturates
    many items at `p_hat = 0` or `1`, so the quartile boundary usually falls inside
    a tied block. Breaking ties by a hash of the item id is arbitrary but *stable*
    across configurations and seeds, which is what the comparison requires -- a
    tie-break that varied per configuration would manufacture the very instability
    the statistic is trying to measure.
    """
    if not items:
        return []
    size = max(1, int(round(quantile * len(items))))
    order = sorted(
        range(len(items)),
        key=lambda i: (
            float("inf") if not math.isfinite(p_hat[i]) else p_hat[i],
            stable_hash(items[i], 8),
        ),
    )
    return sorted(items[i] for i in order[:size])


def boundary_tie_fraction(
    p_hat: Sequence[float], quantile: float = HARD_QUANTILE
) -> float:
    """Fraction of items tied with the value at the quartile boundary.

    Reported because a large value means the hard subset is partly an artefact of
    the tie-break, which caps how high `J_config` and `J_seed` can both go.
    """
    values = [v for v in p_hat if math.isfinite(v)]
    if not values:
        return float("nan")
    size = max(1, int(round(quantile * len(values))))
    threshold = sorted(values)[min(size, len(values)) - 1]
    return sum(1 for v in values if v == threshold) / len(values)


@dataclass
class OverlapResult:
    """`J_config` against its seed-pair null, per model, plus the excess."""

    dataset: str
    subset: Optional[str]
    model: str
    quantile: float
    j_config: float
    j_config_ci: Tuple[float, float]
    j_seed: float
    j_seed_ci: Tuple[float, float]
    excess: float
    excess_ci: Tuple[float, float]
    n_config_pairs: int
    n_seed_pairs: int
    n_items: int
    boundary_tie_fraction: float
    #: Corrected interaction permutation (centred within-item shuffle).
    permutation: Dict[str, Any] = field(default_factory=dict)
    #: Legacy spec permutation (uninformative); regression reference only.
    spec_permutation: Dict[str, Any] = field(default_factory=dict)
    #: The re-estimation null, which is the one to read.
    null_test: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "quantile": self.quantile,
            "j_config": self.j_config,
            "j_config_ci_low": self.j_config_ci[0],
            "j_config_ci_high": self.j_config_ci[1],
            "j_seed_null": self.j_seed,
            "j_seed_ci_low": self.j_seed_ci[0],
            "j_seed_ci_high": self.j_seed_ci[1],
            "excess_instability": self.excess,
            "excess_ci_low": self.excess_ci[0],
            "excess_ci_high": self.excess_ci[1],
            "n_config_pairs": self.n_config_pairs,
            "n_seed_pairs": self.n_seed_pairs,
            "n_items": self.n_items,
            "boundary_tie_fraction": self.boundary_tie_fraction,
            "null_p": self.null_test.get("p_value"),
            "null_mean_j": self.null_test.get("null_mean"),
            "null_n_draws": self.null_test.get("n_draws"),
            "spec_permutation_p": self.spec_permutation.get("p_value"),
            "spec_permutation_uninformative": self.spec_permutation.get(
                "uninformative_by_construction", False
            ),
            "interaction_permutation_p": self.permutation.get("p_value"),
        }


def reestimation_null(
    items: Sequence[str],
    vectors: Sequence[Sequence[float]],
    sample_counts: Sequence[int],
    observed: float,
    quantile: float = HARD_QUANTILE,
    n_draws: int = 2_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """How much hard-subset overlap does re-estimation noise alone produce?

    Each item's difficulty is held fixed at its mean across the observed columns, and
    as many fresh columns as there are configurations are redrawn as
    `Binomial(N_c, p_bar_i) / N_c`. Under that null every configuration shares one
    difficulty per item, so the only thing breaking the overlap is sampling -- which
    is precisely the comparison METHOD_SPEC 5.4 wants and the label permutation in
    step 5 cannot deliver.

    One-sided: the prediction is that a real configuration effect drives `J_config`
    *below* the null.
    """
    import numpy as np

    usable = [n for n in sample_counts if n > 0]
    if len(vectors) < 2 or not usable or not math.isfinite(observed):
        return {"p_value": float("nan"), "n_draws": 0, "available": False}

    p_bar = np.asarray(
        [
            float(np.nanmean([v[i] for v in vectors]))
            for i in range(len(items))
        ],
        dtype=float,
    )
    p_bar = np.clip(np.nan_to_num(p_bar, nan=0.5), 0.0, 1.0)
    counts = [n if n > 0 else usable[0] for n in sample_counts]

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for b in range(n_draws):
        columns = [
            (rng.binomial(n, p_bar) / float(n)).tolist() for n in counts
        ]
        draws[b] = _mean_jaccard(items, columns, quantile)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"p_value": float("nan"), "n_draws": 0, "available": False}
    n_extreme = int((finite <= observed).sum())
    return {
        "p_value": (1.0 + n_extreme) / (1.0 + finite.size),
        "null_mean": float(finite.mean()),
        "null_std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "null_q05": float(np.quantile(finite, 0.05)),
        "observed": float(observed),
        "n_draws": int(finite.size),
        "sample_counts": list(counts),
        "available": True,
        "alternative": "less",
        "note": (
            "re-estimation null: each item's difficulty fixed at its "
            "across-configuration mean, columns redrawn binomially at the same N. "
            "One-sided; a small p-value means J_config is lower than sampling noise "
            "alone explains. null_mean should sit near J_seed."
        ),
    }


def _mean_jaccard(
    items: Sequence[str], vectors: Sequence[Sequence[float]], quantile: float
) -> float:
    subsets = [hard_subset(items, v, quantile) for v in vectors]
    values = [
        jaccard(subsets[a], subsets[b]) for a, b in combinations(range(len(subsets)), 2)
    ]
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def hard_subset_overlap(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    reference_config: str = REFERENCE_CONFIG_ID,
    quantile: float = HARD_QUANTILE,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_permutations: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
) -> Optional[OverlapResult]:
    """Algorithm 3 for one (dataset, model): `J_config`, `J_seed`, excess, p-value."""
    configs = list(configs) if configs else [
        c for c in corpus.configs() if c in PRIMARY_CONFIG_IDS
    ] or corpus.configs()
    config_items, config_vectors = _aligned(
        corpus, dataset, subset, [(model, c, seed) for c in configs]
    )
    seeds = sorted(
        {
            cell.seed
            for cell in corpus.cells.values()
            if cell.dataset == dataset
            and cell.subset == subset
            and cell.model == model
            and cell.config == reference_config
            and cell.n > 0
        }
    )
    seed_items, seed_vectors = (
        _aligned(corpus, dataset, subset, [(model, reference_config, s) for s in seeds])
        if len(seeds) >= 2
        else ([], [])
    )
    if len(config_items) < 4 or len(config_vectors) < 2:
        return None

    # Compare on the item set shared by both arms, so J_config and J_seed are not
    # quartiles of different populations.
    if seed_items:
        shared = sorted(set(config_items) & set(seed_items))
        if len(shared) >= 4:
            keep_c = [config_items.index(i) for i in shared]
            keep_s = [seed_items.index(i) for i in shared]
            config_vectors = [[v[j] for j in keep_c] for v in config_vectors]
            seed_vectors = [[v[j] for j in keep_s] for v in seed_vectors]
            config_items = shared
            seed_items = shared
        else:
            seed_items, seed_vectors = [], []

    items = config_items
    j_config = _mean_jaccard(items, config_vectors, quantile)
    j_seed = (
        _mean_jaccard(items, seed_vectors, quantile) if seed_vectors else float("nan")
    )

    # Bootstrap the clustering unit. Resampling items changes the subset *size*
    # consistently for both arms, which is what keeps the difference meaningful.
    rows = [
        (
            items[i],
            tuple(v[i] for v in config_vectors),
            tuple(v[i] for v in seed_vectors) if seed_vectors else (),
        )
        for i in range(len(items))
    ]

    def _rebuild(sample: Sequence[Any], which: int) -> Tuple[List[str], List[List[float]]]:
        # De-duplicate ids so a resampled item cannot appear twice in one set: a
        # Jaccard over a multiset is not defined, and silently collapsing would bias
        # both arms in the same direction but by different amounts.
        labels = [f"{row[0]}#{i}" for i, row in enumerate(sample)]
        width = len(sample[0][which]) if sample and sample[0][which] else 0
        vectors = [[row[which][c] for row in sample] for c in range(width)]
        return labels, vectors

    def stat_config(sample: Sequence[Any]) -> float:
        if len(sample) < 4:
            return float("nan")
        labels, vectors = _rebuild(sample, 1)
        return _mean_jaccard(labels, vectors, quantile)

    def stat_seed(sample: Sequence[Any]) -> float:
        if len(sample) < 4 or not sample[0][2]:
            return float("nan")
        labels, vectors = _rebuild(sample, 2)
        return _mean_jaccard(labels, vectors, quantile)

    def stat_excess(sample: Sequence[Any]) -> float:
        return stat_seed(sample) - stat_config(sample)

    ci_config = bca_ci(rows, stat_config, n_bootstrap=n_bootstrap, alpha=alpha, seed=bootstrap_seed)
    ci_seed = bca_ci(rows, stat_seed, n_bootstrap=n_bootstrap, alpha=alpha, seed=bootstrap_seed)
    ci_excess = bca_ci(rows, stat_excess, n_bootstrap=n_bootstrap, alpha=alpha, seed=bootstrap_seed)

    # Corrected interaction permutation: centre each configuration to its own mean
    # before permuting within item, so the null targets the interaction alone
    # (review §2.4).
    grand = sum(sum(v) for v in config_vectors) / (len(items) * len(config_vectors))
    col_means = [
        sum(config_vectors[c][i] for c in range(len(config_vectors))) / len(config_vectors)
        for i in range(len(items))
    ]
    config_means = [
        sum(config_vectors[c][i] for i in range(len(items))) / len(items)
        for c in range(len(config_vectors))
    ]
    centered = [
        [
            config_vectors[c][i] - config_means[c] - col_means[i] + grand
            for i in range(len(items))
        ]
        for c in range(len(config_vectors))
    ]
    matrix = [[centered[c][i] for c in range(len(config_vectors))] for i in range(len(items))]

    def permuted_statistic(rows_in: Sequence[Sequence[float]]) -> float:
        vectors = [[row[c] for row in rows_in] for c in range(len(rows_in[0]))]
        return -_mean_jaccard(items, vectors, quantile)

    def permute(rows_in: Sequence[Sequence[float]], rng: Any) -> List[List[float]]:
        out = []
        for row in rows_in:
            order = rng.permutation(len(row))
            out.append([row[j] for j in order])
        return out

    perm = permutation_test(
        permuted_statistic,
        matrix,
        permute,
        n_permutations=n_permutations,
        seed=bootstrap_seed,
        alternative="greater",
    )
    perm["observed_j_config"] = -perm.pop("observed")
    if "null_mean" in perm:
        perm["null_mean_j_config"] = -perm.pop("null_mean")
    perm["uninformative_by_construction"] = False
    perm["note"] = (
        "corrected within-item permutation: configuration columns centred before "
        "label shuffle, testing the interaction alone (review §2.4)"
    )
    perm["interaction_test"] = True

    # Legacy spec permutation (uninformative); kept for regression checks.
    raw_matrix = [[v[i] for v in config_vectors] for i in range(len(items))]

    def raw_permuted(rows_in: Sequence[Sequence[float]]) -> float:
        vectors = [[row[c] for row in rows_in] for c in range(len(rows_in[0]))]
        return -_mean_jaccard(items, vectors, quantile)

    def raw_permute(rows_in: Sequence[Sequence[float]], rng: Any) -> List[List[float]]:
        out = []
        for row in rows_in:
            order = rng.permutation(len(row))
            out.append([row[j] for j in order])
        return out

    spec_perm = permutation_test(
        raw_permuted,
        raw_matrix,
        raw_permute,
        n_permutations=min(200, n_permutations),
        seed=bootstrap_seed + 1,
        alternative="greater",
    )
    spec_perm["observed_j_config"] = -spec_perm.pop("observed")
    if "null_mean" in spec_perm:
        spec_perm["null_mean_j_config"] = -spec_perm.pop("null_mean")
    spec_perm["uninformative_by_construction"] = True
    spec_perm["note"] = "legacy METHOD_SPEC 5.4 step 5 permutation (uninformative)"

    null_test = reestimation_null(
        items,
        config_vectors,
        _column_sample_counts(
            corpus, dataset, subset, [(model, c, seed) for c in configs]
        ),
        observed=j_config,
        quantile=quantile,
        n_draws=max(200, min(n_permutations, 2_000)),
        seed=bootstrap_seed,
    )

    return OverlapResult(
        dataset=dataset,
        subset=subset,
        model=model,
        quantile=quantile,
        j_config=j_config,
        j_config_ci=(float(ci_config["ci_low"]), float(ci_config["ci_high"])),
        j_seed=j_seed,
        j_seed_ci=(float(ci_seed["ci_low"]), float(ci_seed["ci_high"])),
        excess=j_seed - j_config if math.isfinite(j_seed) else float("nan"),
        excess_ci=(float(ci_excess["ci_low"]), float(ci_excess["ci_high"])),
        n_config_pairs=len(list(combinations(range(len(config_vectors)), 2))),
        n_seed_pairs=len(list(combinations(range(len(seed_vectors)), 2))),
        n_items=len(items),
        boundary_tie_fraction=boundary_tie_fraction(config_vectors[0], quantile),
        permutation=perm,
        spec_permutation=spec_perm,
        null_test=null_test,
    )


def overlap_table(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_permutations: int = DEFAULT_N_BOOTSTRAP,
    **kw: Any,
) -> List[OverlapResult]:
    out: List[OverlapResult] = []
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            result = hard_subset_overlap(
                corpus,
                dataset,
                subset,
                model,
                configs=configs,
                seed=seed,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                **kw,
            )
            if result is not None:
                out.append(result)
    return out


def overlap_summary(results: Sequence[OverlapResult]) -> Dict[str, Any]:
    """Prediction P3: `J_config` at least 0.10 below `J_seed`."""
    usable = [r for r in results if math.isfinite(r.excess)]
    if not usable:
        return {"n_cells": 0, "prediction": "P3", "supported": None}
    excess = [r.excess for r in usable]
    mean_excess = sum(excess) / len(excess)
    null_ps = [
        float(r.null_test.get("p_value"))
        for r in usable
        if r.null_test.get("available") and math.isfinite(float(r.null_test.get("p_value") or float("nan")))
    ]
    return {
        "n_cells": len(usable),
        "mean_j_config": sum(r.j_config for r in usable) / len(usable),
        "mean_j_seed": sum(r.j_seed for r in usable) / len(usable),
        "mean_excess": mean_excess,
        "n_cells_below_reestimation_null": sum(1 for p in null_ps if p < 0.05),
        "n_cells_tested": len(null_ps),
        "min_excess": min(excess),
        "max_excess": max(excess),
        "prediction": "P3",
        "statement": "J_config is at least 0.10 lower than J_seed",
        "supported": bool(mean_excess >= 0.10),
    }
