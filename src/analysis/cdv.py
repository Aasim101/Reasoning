"""Algorithm 5: configuration-diversified voting, at matched completion tokens.

This is the intervention, and it costs **zero additional GPU time**: every method
here is a different way of partitioning chains that the sampling run already
generated and persisted. Nothing in this module generates text.

The one rule that makes the comparison honest is METHOD_SPEC 5.6's matched-budget
rule: methods are compared at equal **total completion tokens**, never at equal
sample count. Configurations induce chains of different mean length, so matching
on `N` would hand a free win to whichever configuration happens to be terse -- a
reviewer would spot it immediately, and it would be the kind of error that
invalidates the headline claim rather than a detail. Every allocator below
therefore takes a *token* budget and draws chains until one more would exceed it,
and every result carries the realised token spend so the matching is auditable
rather than asserted.

Two consequences of that rule are worth stating because they look like bugs and
are not:

* Methods under-spend the budget by up to one chain, and by different amounts.
  Overshooting would let a method exceed the budget it is being compared at, which
  is worse. The realised spend is reported per method.
* A method whose pool runs out cannot use more budget however large the budget
  gets. Self-consistency has `N` chains at one configuration; CDV has `N` at each
  of `C_use`. The token grid is therefore capped at what the *narrowest* method can
  actually spend, and `pool_exhausted_rate` flags any point where it binds. This
  cap is what makes the large-budget end of Figure F5 -- the falsification hook --
  a fair test instead of an artefact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS
from ..elicitation import (
    DEFAULT_CDV_CONFIG_IDS,
    PARAPHRASE_CONFIG_ID,
    REFERENCE_CONFIG_ID,
    TEMPERATURE_ARM_IDS,
)
from ..metrics import DEFAULT_ALPHA, DEFAULT_N_BOOTSTRAP, holm_bonferroni, mcnemar
from .corpus import Cell, Corpus, SampleRow

log = logging.getLogger(__name__)

#: Sample-equivalent budgets for the accuracy-vs-tokens curve. Converted to a token
#: budget by multiplying by the mean completion tokens per chain across the
#: configurations in the comparison, so every method faces the identical number of
#: tokens (see `token_budgets`).
DEFAULT_BUDGET_MULTIPLES: Tuple[int, ...] = (1, 2, 4, 8, 16, 24)

#: The three budgets Table T5 reports at: small (where any variance reduction
#: flatters), mid, and the largest the design supports (where the falsification
#: hook lives).
DEFAULT_TABLE_MULTIPLES: Tuple[int, ...] = (4, 8, 24)

#: Random subsets per (item, method, budget). METHOD_SPEC 7.1 specifies 200 for
#: `maj@n`; 100 halves the analysis runtime and the residual Monte-Carlo error on
#: the *item-averaged* accuracy is under 0.2 points, well inside the bootstrap CI.
#: Raise to 200 for the final camera-ready pass via `n_repeats`.
DEFAULT_N_REPEATS = 100

#: Probe samples per configuration for Adaptive-CDV's first stage (METHOD_SPEC 5.6).
ADAPTIVE_PROBE_N = 4

#: Slack allowed above the narrowest method's mean capacity before a budget point is
#: dropped as unspendable. See `token_budgets`.
CAP_TOLERANCE = 0.05

#: Minimum accuracy gain of configuration-aware over uniform allocation before the
#: "fraction of gain retained" ratio in Table T6 means anything (0.5 points).
MIN_REPORTABLE_GAIN = 0.005


# ------------------------------------------------------------------- item pools
@dataclass
class ItemPools:
    """Every chain available for one (item, model, seed), grouped by configuration.

    The unit every allocator consumes. Chains are stored as parallel arrays because
    the draw loop runs millions of times across a full analysis and per-sample
    object attribute access dominated the profile.
    """

    item_id: str
    dataset: str
    subset: Optional[str]
    model: str
    seed: int
    bootstrap_unit: str
    #: config -> (tokens, class_label, is_correct) arrays.
    tokens: Dict[str, Any] = field(default_factory=dict)
    classes: Dict[str, List[str]] = field(default_factory=dict)
    correct: Dict[str, Any] = field(default_factory=dict)
    #: config -> per-chain self-certainty weight (from mean logprob), or None.
    certainty: Dict[str, Optional[Any]] = field(default_factory=dict)

    def configs(self) -> List[str]:
        return sorted(self.tokens)

    def n_available(self, config: str) -> int:
        return len(self.classes.get(config, ()))

    def capacity_tokens(self, configs: Sequence[str]) -> float:
        """Total tokens this item could spend if every chain in `configs` were used."""
        return float(sum(float(self.tokens[c].sum()) for c in configs if c in self.tokens))

    def mean_tokens(self, configs: Sequence[str]) -> float:
        total, count = 0.0, 0
        for c in configs:
            if c in self.tokens and len(self.classes[c]):
                total += float(self.tokens[c].sum())
                count += len(self.classes[c])
        return total / count if count else float("nan")


def build_seed_pooled_pools(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    reference: str = REFERENCE_CONFIG_ID,
    include_separate_arms: bool = True,
) -> List[ItemPools]:
    """Merge seed replicates at `reference` into one pool per item (SC control)."""
    import numpy as np

    grouped: Dict[str, List[Cell]] = {}
    for cell in corpus.select(
        dataset=dataset,
        subset=subset,
        model=model,
        include_separate_arms=include_separate_arms,
    ):
        if cell.n == 0 or cell.config != reference:
            continue
        grouped.setdefault(cell.item_id, []).append(cell)

    pools: List[ItemPools] = []
    for item_id, cells in sorted(grouped.items()):
        pool = ItemPools(
            item_id=item_id,
            dataset=dataset,
            subset=subset,
            model=model,
            seed=0,
            bootstrap_unit=cells[0].bootstrap_unit(),
        )
        tokens: List[float] = []
        labels: List[str] = []
        correct: List[bool] = []
        logprobs: List[float] = []
        for cell in sorted(cells, key=lambda c: c.seed):
            for row in cell.samples:
                tokens.append(float(max(1, row.tokens_completion)))
                labels.append(row.canonical_class)
                correct.append(bool(row.is_correct))
                if row.mean_logprob is not None:
                    logprobs.append(float(row.mean_logprob))
        pool.tokens[reference] = np.array(tokens, dtype=float)
        pool.classes[reference] = labels
        pool.correct[reference] = np.array(correct, dtype=bool)
        pool.certainty[reference] = (
            np.array(logprobs, dtype=float)
            if logprobs and len(logprobs) == len(labels)
            else None
        )
        pools.append(pool)
    return pools


def build_item_pools(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    seed: int = 0,
    configs: Optional[Sequence[str]] = None,
    include_separate_arms: bool = True,
) -> List[ItemPools]:
    """Group the corpus into per-item pools for one (dataset, model, seed)."""
    import numpy as np

    wanted = set(configs) if configs else None
    grouped: Dict[str, List[Cell]] = {}
    for cell in corpus.select(
        dataset=dataset,
        subset=subset,
        model=model,
        seed=seed,
        include_separate_arms=include_separate_arms,
    ):
        if cell.n == 0 or (wanted is not None and cell.config not in wanted):
            continue
        grouped.setdefault(cell.item_id, []).append(cell)

    pools: List[ItemPools] = []
    for item_id, cells in sorted(grouped.items()):
        pool = ItemPools(
            item_id=item_id,
            dataset=dataset,
            subset=subset,
            model=model,
            seed=seed,
            bootstrap_unit=cells[0].bootstrap_unit(),
        )
        for cell in cells:
            rows: List[SampleRow] = cell.samples
            pool.tokens[cell.config] = np.array(
                [max(1, r.tokens_completion) for r in rows], dtype=float
            )
            pool.classes[cell.config] = [r.canonical_class for r in rows]
            pool.correct[cell.config] = np.array(
                [bool(r.is_correct) for r in rows], dtype=bool
            )
            logprobs = [r.mean_logprob for r in rows]
            pool.certainty[cell.config] = (
                np.array([float(v) for v in logprobs], dtype=float)
                if all(v is not None and math.isfinite(float(v)) for v in logprobs)
                else None
            )
        pools.append(pool)
    return pools


def _zero_token_warning(pools: Sequence[ItemPools]) -> Optional[str]:
    """Detect a corpus with no token accounting, which silently breaks matching."""
    total = sum(float(arr.sum()) for p in pools for arr in p.tokens.values())
    n = sum(len(v) for p in pools for v in p.classes.values())
    if n and total <= n:  # every chain fell back to the 1-token floor
        return (
            "no completion-token counts found in the corpus, so the budget "
            "matching is by sample count in disguise. Re-run grading from results "
            "that carry `sample_stats`, or treat every token-matched number as "
            "invalid."
        )
    return None


# ------------------------------------------------------------------- draw result
@dataclass
class DrawResult:
    """One method's draw for one item at one budget."""

    predicted: Optional[str]
    correct: bool
    tokens: float
    n_chains: int
    exhausted: bool
    #: True when the budget did not cover a single chain.
    empty: bool = False


def _vote(
    labels: Sequence[str],
    correct: Sequence[bool],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[Optional[str], bool]:
    """Plurality over canonical classes; ties break by first occurrence.

    First-occurrence tie-breaking matches `Cell.modal_class`, so the plateau
    reported by the modal-ceiling analysis and the vote reported here agree on the
    same corpus. The extraction-failure class is allowed to win: suppressing it
    would let a method claim credit for chains that produced no parseable answer.
    """
    if not labels:
        return None, False
    totals: Dict[str, float] = {}
    for i, label in enumerate(labels):
        totals[label] = totals.get(label, 0.0) + (1.0 if weights is None else float(weights[i]))
    best = max(totals.values())
    winner = next(label for label in labels if totals[label] >= best - 1e-12)
    if winner == EXTRACTION_FAILURE_CLASS:
        return winner, False
    is_correct = any(bool(correct[i]) for i, label in enumerate(labels) if label == winner)
    return winner, is_correct


def _take_within_budget(
    order: Sequence[int],
    tokens: Any,
    budget: float,
    spent: float = 0.0,
) -> Tuple[List[int], float]:
    """Greedily take indices in `order` while the token budget allows.

    Skips a chain that does not fit rather than stopping, so one unusually long
    chain does not truncate the draw and make the method look budget-starved. This
    is the only place the budget is enforced.
    """
    taken: List[int] = []
    for idx in order:
        cost = float(tokens[idx])
        if spent + cost <= budget + 1e-9:
            taken.append(idx)
            spent += cost
    return taken, spent


Allocator = Callable[[ItemPools, float, Any], DrawResult]


def _single_config(config: str, weighted: bool = False) -> Allocator:
    """B3 self-consistency at one configuration; B6 when `weighted`."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        if config not in pool.classes:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        order = rng.permutation(len(pool.classes[config]))
        taken, spent = _take_within_budget(order, pool.tokens[config], budget)
        labels = [pool.classes[config][i] for i in taken]
        correct = [bool(pool.correct[config][i]) for i in taken]
        weights = None
        if weighted:
            certainty = pool.certainty.get(config)
            if certainty is not None:
                weights = _certainty_weights([float(certainty[i]) for i in taken])
        predicted, is_correct = _vote(labels, correct, weights)
        return DrawResult(
            predicted,
            is_correct,
            spent,
            len(taken),
            exhausted=len(taken) >= len(order),
            empty=not taken,
        )

    return alloc


def _certainty_weights(mean_logprobs: Sequence[float]) -> List[float]:
    """Self-certainty vote weights from per-chain mean logprob (B6, approximate).

    METHOD_SPEC 6 defines B6 as `KL(answer-token distribution || uniform)`, which
    needs the full per-token distribution. The harness logs the mean logprob of the
    sampled tokens, and mean logprob is a monotone proxy for that KL (both measure
    how far the chain's token distribution sits from uniform). The weights below are
    a softmax over mean logprob, shifted by the maximum for numerical stability.
    Reported as an approximation, not as B6 exactly -- see the discrepancy note.
    """
    if not mean_logprobs:
        return []
    top = max(mean_logprobs)
    exp = [math.exp(min(0.0, v - top)) for v in mean_logprobs]
    total = sum(exp)
    return [e / total for e in exp] if total > 0 else [1.0] * len(mean_logprobs)


def _pooled_configs(configs: Sequence[str], round_robin: bool = True) -> Allocator:
    """CDV and the diversity ablations: spread the budget over several pools.

    Round-robin rather than a shuffled union, because METHOD_SPEC 5.6 specifies a
    uniform allocation (`per_cfg = floor(B_s / C_use)`) with a fixed configuration
    order and no per-item selection. A shuffled union would spend more on whichever
    configuration is terse, quietly turning uniform CDV into a length-biased
    allocation.
    """

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        available = [c for c in configs if c in pool.classes]
        if not available:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        orders = {c: list(rng.permutation(len(pool.classes[c]))) for c in available}
        cursors = {c: 0 for c in available}
        labels: List[str] = []
        correct: List[bool] = []
        spent = 0.0
        progress = True
        while progress:
            progress = False
            for c in available:
                order = orders[c]
                while cursors[c] < len(order):
                    idx = order[cursors[c]]
                    cursors[c] += 1
                    cost = float(pool.tokens[c][idx])
                    if spent + cost <= budget + 1e-9:
                        spent += cost
                        labels.append(pool.classes[c][idx])
                        correct.append(bool(pool.correct[c][idx]))
                        progress = True
                        break
        predicted, is_correct = _vote(labels, correct)
        exhausted = all(cursors[c] >= len(orders[c]) for c in available)
        return DrawResult(
            predicted, is_correct, spent, len(labels), exhausted, empty=not labels
        )

    return alloc


def _adaptive_cdv(configs: Sequence[str], probe_n: int = ADAPTIVE_PROBE_N) -> Allocator:
    """Adaptive-CDV: probe cheaply, commit when the configurations already agree."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        available = [c for c in configs if c in pool.classes]
        if not available:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        orders = {c: list(rng.permutation(len(pool.classes[c]))) for c in available}
        spent = 0.0
        probe_labels: Dict[str, List[str]] = {}
        probe_correct: Dict[str, List[bool]] = {}
        cursors = {c: 0 for c in available}
        for c in available:
            probe_labels[c] = []
            probe_correct[c] = []
            # The cursor counts *positions consumed*, not chains taken: a chain
            # skipped for not fitting must not be reconsidered by the second stage,
            # or it would be counted twice.
            while cursors[c] < min(probe_n, len(orders[c])):
                idx = orders[c][cursors[c]]
                cursors[c] += 1
                cost = float(pool.tokens[c][idx])
                if spent + cost <= budget + 1e-9:
                    spent += cost
                    probe_labels[c].append(pool.classes[c][idx])
                    probe_correct[c].append(bool(pool.correct[c][idx]))

        modes = {c: _vote(probe_labels[c], probe_correct[c])[0] for c in available}
        present = [m for m in modes.values() if m is not None]
        all_labels = [l for c in available for l in probe_labels[c]]
        all_correct = [v for c in available for v in probe_correct[c]]
        if present and len(set(present)) == 1:
            predicted, is_correct = _vote(all_labels, all_correct)
            return DrawResult(
                predicted, is_correct, spent, len(all_labels), False, empty=not all_labels
            )

        # Disagreement: spend what is left across the disagreeing configurations.
        disagreeing = [c for c in available if modes[c] is not None] or available
        progress = True
        while progress:
            progress = False
            for c in disagreeing:
                order = orders[c]
                while cursors[c] < len(order):
                    idx = order[cursors[c]]
                    cursors[c] += 1
                    cost = float(pool.tokens[c][idx])
                    if spent + cost <= budget + 1e-9:
                        spent += cost
                        all_labels.append(pool.classes[c][idx])
                        all_correct.append(bool(pool.correct[c][idx]))
                        progress = True
                        break
        predicted, is_correct = _vote(all_labels, all_correct)
        exhausted = all(cursors[c] >= len(orders[c]) for c in disagreeing)
        return DrawResult(
            predicted, is_correct, spent, len(all_labels), exhausted, empty=not all_labels
        )

    return alloc


def _esc(config: str, window: int = 4) -> Allocator:
    """B8 early-stopping self-consistency: stop once a window agrees unanimously."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        if config not in pool.classes:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        order = list(rng.permutation(len(pool.classes[config])))
        labels: List[str] = []
        correct: List[bool] = []
        spent = 0.0
        for idx in order:
            cost = float(pool.tokens[config][idx])
            if spent + cost > budget + 1e-9:
                continue
            spent += cost
            labels.append(pool.classes[config][idx])
            correct.append(bool(pool.correct[config][idx]))
            if len(labels) >= window and len(set(labels[-window:])) == 1:
                break
        predicted, is_correct = _vote(labels, correct)
        return DrawResult(
            predicted,
            is_correct,
            spent,
            len(labels),
            exhausted=len(labels) >= len(order),
            empty=not labels,
        )

    return alloc


def _oracle_coverage(configs: Sequence[str]) -> Allocator:
    """B10 pass@n: correct if *any* drawn chain is correct."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        available = [c for c in configs if c in pool.classes]
        if not available:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        orders = {c: list(rng.permutation(len(pool.classes[c]))) for c in available}
        cursors = {c: 0 for c in available}
        spent, n, any_correct = 0.0, 0, False
        progress = True
        while progress:
            progress = False
            for c in available:
                while cursors[c] < len(orders[c]):
                    idx = orders[c][cursors[c]]
                    cursors[c] += 1
                    cost = float(pool.tokens[c][idx])
                    if spent + cost <= budget + 1e-9:
                        spent += cost
                        n += 1
                        any_correct = any_correct or bool(pool.correct[c][idx])
                        progress = True
                        break
        return DrawResult(
            "<oracle>" if any_correct else None,
            any_correct,
            spent,
            n,
            all(cursors[c] >= len(orders[c]) for c in available),
            empty=n == 0,
        )

    return alloc


def _oracle_best_single(configs: Sequence[str]) -> Allocator:
    """Oracle best *global* single configuration: pick the config with highest π_mode."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        best: Optional[DrawResult] = None
        best_rate = -1.0
        for c in configs:
            if c not in pool.classes:
                continue
            result = _single_config(c)(pool, budget, rng)
            rate = float(pool.correct[c].mean()) if len(pool.correct[c]) else 0.0
            if rate > best_rate or (rate == best_rate and result.correct):
                best_rate = rate
                best = result
        return best or DrawResult(None, False, 0.0, 0, True, empty=True)

    return alloc


def _seed_pooled(reference: str = REFERENCE_CONFIG_ID) -> Allocator:
    """Pool seed replicates at c0: identical components, must match SC at one seed."""

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        return _single_config(reference)(pool, budget, rng)

    return alloc


def _pools_for_method(
    method: Method,
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    seed: int,
    pools: Sequence[ItemPools],
) -> Sequence[ItemPools]:
    """Return item pools appropriate for a method (seed-pooled SC merges replicates)."""
    if method.name == "seed_pooled_sc":
        ref = method.configs[0] if method.configs else REFERENCE_CONFIG_ID
        merged = build_seed_pooled_pools(corpus, dataset, subset, model, reference=ref)
        return merged or pools
    return pools


def _oracle_config(configs: Sequence[str]) -> Allocator:
    """B11 oracle configuration selection: spend it all in the best configuration.

    The whole budget goes to one configuration, chosen with hindsight per item. The
    upper bound for any configuration-selection policy, so the CDV-to-B11 gap is the
    headroom a better selector could still win.
    """

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        best: Optional[DrawResult] = None
        for c in configs:
            if c not in pool.classes:
                continue
            result = _single_config(c)(pool, budget, rng)
            if best is None or (result.correct and not best.correct):
                best = result
        return best or DrawResult(None, False, 0.0, 0, True, empty=True)

    return alloc


def _random_config(configs: Sequence[str]) -> Allocator:
    """Control: one configuration per item, chosen at random.

    Separates "CDV works" from "the reference configuration is just a bad one". If
    this control matches CDV, the gain was never about diversity.
    """

    def alloc(pool: ItemPools, budget: float, rng: Any) -> DrawResult:
        available = [c for c in configs if c in pool.classes]
        if not available:
            return DrawResult(None, False, 0.0, 0, True, empty=True)
        chosen = available[int(rng.integers(0, len(available)))]
        return _single_config(chosen)(pool, budget, rng)

    return alloc


# ------------------------------------------------------------------ method table
@dataclass
class Method:
    """A named allocator plus the baseline id it implements."""

    name: str
    baseline: str
    alloc: Allocator
    description: str
    #: Configurations it draws from; used for the shared-capacity token cap.
    configs: Tuple[str, ...] = ()


def single_config_method(config: str) -> Method:
    """Self-consistency restricted to one configuration.

    Exposed so Figure F4 can draw a `maj@n`-versus-tokens curve per configuration and
    show the plateau spread, without reaching into the private allocators.
    """
    return Method(
        f"sc_{config}", "B3", _single_config(config),
        f"self-consistency at {config}", (config,),
    )


def build_methods(
    available_configs: Sequence[str],
    cdv_configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
) -> List[Method]:
    """The B1-B11 + CDV method set, restricted to what the corpus actually has."""
    have = set(available_configs)
    cdv_ids = [c for c in (cdv_configs or DEFAULT_CDV_CONFIG_IDS) if c in have]
    temp_ids = [c for c in TEMPERATURE_ARM_IDS if c in have]
    para_ids = [c for c in (reference, PARAPHRASE_CONFIG_ID) if c in have]
    ref = reference if reference in have else (sorted(have)[0] if have else reference)

    methods: List[Method] = []
    if ref in have:
        methods += [
            Method("sc", "B3", _single_config(ref),
                   f"self-consistency at {ref} (primary baseline)", (ref,)),
            Method("certainty_vote", "B6", _single_config(ref, weighted=True),
                   f"self-certainty-weighted vote at {ref} (approximate)", (ref,)),
            Method("esc", "B8", _esc(ref),
                   f"early-stopping self-consistency at {ref}", (ref,)),
        ]
    if len(temp_ids) > 1:
        methods.append(
            Method("temp_sc", "B4", _pooled_configs(temp_ids),
                   "temperature-diversified self-consistency, same prompt",
                   tuple(temp_ids))
        )
    if len(para_ids) > 1:
        methods.append(
            Method("para_sc", "B5", _pooled_configs(para_ids),
                   "self-para-consistency (vote over paraphrases)", tuple(para_ids))
        )
    if len(cdv_ids) > 1:
        methods += [
            Method("cdv", "CDV", _pooled_configs(cdv_ids),
                   f"configuration-diversified voting over {','.join(cdv_ids)}",
                   tuple(cdv_ids)),
            Method("adaptive_cdv", "CDV-A", _adaptive_cdv(cdv_ids),
                   "adaptive CDV: probe, commit on agreement", tuple(cdv_ids)),
            Method("random_config", "control", _random_config(cdv_ids),
                   "one randomly chosen configuration per item", tuple(cdv_ids)),
            Method("oracle_config", "B11", _oracle_config(cdv_ids),
                   "oracle per-item configuration selection", tuple(cdv_ids)),
        ]
    if ref in have:
        methods.append(
            Method("oracle_coverage", "B10", _oracle_coverage([ref]),
                   f"oracle pass@n coverage at {ref}", (ref,))
        )
        methods.append(
            Method(
                "oracle_best_single",
                "B12",
                _oracle_best_single(cdv_ids or [ref]),
                "oracle best single global configuration",
                tuple(cdv_ids or [ref]),
            )
        )
        methods.append(
            Method(
                "seed_pooled_sc",
                "control",
                _seed_pooled(ref),
                "seed-pooled voting at c0 (must equal SC)",
                (ref,),
            )
        )
    return methods


# ------------------------------------------------------------------ token budgets
def token_budgets(
    pools: Sequence[ItemPools],
    methods: Sequence[Method],
    multiples: Sequence[int] = DEFAULT_BUDGET_MULTIPLES,
    scale_method: str = "sc",
) -> Dict[str, Any]:
    """The shared token grid, capped by the narrowest method's capacity.

    `multiples` are sample-equivalents, converted to tokens with **one** scalar --
    the mean chain cost of the reference method's configuration -- so a budget point
    is literally the same token count for every method, which is what the
    matched-budget rule requires.

    Scaling off the reference configuration rather than the pooled mean is
    deliberate: it makes multiple `N` equal to self-consistency's entire pool, so the
    grid's top end is exactly where the primary baseline runs out. Scaling off the
    pooled mean instead would put the top budget beyond what a terse reference
    configuration could ever spend, and the cap below would silently delete the
    largest budget point -- the one the falsification hook needs.
    """
    scale_configs = next(
        (m.configs for m in methods if m.name == scale_method),
        tuple(sorted({c for m in methods for c in m.configs})),
    )
    per_chain = [p.mean_tokens(scale_configs) for p in pools]
    per_chain = [v for v in per_chain if math.isfinite(v)]
    mean_tokens = sum(per_chain) / len(per_chain) if per_chain else float("nan")

    # Mean per-item capacity per method; the smallest of those caps the grid. The
    # tolerance keeps the top budget point, which by construction sits exactly at
    # the reference method's full pool: at that point self-consistency is meant to
    # be exhausted (`pool_exhausted_rate` says so), and dropping it would remove the
    # large-budget end of Figure F5, which is where the falsification hook lives.
    caps: Dict[str, float] = {}
    for method in methods:
        capacities = [p.capacity_tokens(method.configs) for p in pools]
        caps[method.name] = sum(capacities) / len(capacities) if capacities else 0.0
    binding = min(caps.values()) if caps else 0.0
    limiting = min(caps, key=lambda k: caps[k]) if caps else ""

    grid = [float(m) * mean_tokens for m in multiples]
    ceiling = binding * (1.0 + CAP_TOLERANCE)
    kept = [b for b in grid if b <= ceiling]
    dropped = [b for b in grid if b > ceiling]
    if dropped:
        log.info(
            "capping the token grid at %.0f tokens/item (%s is the narrowest "
            "method); dropped %d budget point(s) that only the wider pools could "
            "spend",
            binding,
            limiting,
            len(dropped),
        )
    return {
        "mean_tokens_per_chain": mean_tokens,
        "scale_configs": list(scale_configs),
        "budgets": kept or grid[:1],
        "multiples": [m for m, b in zip(multiples, grid) if b in kept] or list(multiples[:1]),
        "capacity_by_method": caps,
        "binding_capacity": binding,
        "limiting_method": limiting,
        "dropped_budgets": dropped,
    }


# --------------------------------------------------------------------- evaluation
@dataclass
class MethodPoint:
    """One (method, budget) point: accuracy, realised tokens, per-item scores."""

    method: str
    baseline: str
    dataset: str
    subset: Optional[str]
    model: str
    seed: int
    budget_tokens: float
    budget_multiple: float
    accuracy: float
    mean_tokens_used: float
    mean_chains_used: float
    pool_exhausted_rate: float
    empty_draw_rate: float
    n_items: int
    #: item_id -> mean correctness over the repeated draws, in [0, 1].
    scores: Dict[str, float] = field(default_factory=dict)
    #: item_id -> resampling unit, for the clustered bootstrap.
    units: Dict[str, str] = field(default_factory=dict)

    def verdicts(self) -> Dict[str, bool]:
        """Binary per-item outcome for McNemar: correct on most repeated draws."""
        return {item: score > 0.5 for item, score in self.scores.items()}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "seed": self.seed,
            "method": self.method,
            "baseline": self.baseline,
            "budget_multiple": self.budget_multiple,
            "budget_tokens": self.budget_tokens,
            "accuracy": self.accuracy,
            "mean_tokens_used": self.mean_tokens_used,
            "mean_chains_used": self.mean_chains_used,
            "pool_exhausted_rate": self.pool_exhausted_rate,
            "empty_draw_rate": self.empty_draw_rate,
            "n_items": self.n_items,
        }


def evaluate_method(
    method: Method,
    pools: Sequence[ItemPools],
    budget_tokens: float,
    budget_multiple: float,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = 0,
) -> MethodPoint:
    """Repeated token-bounded draws for one method at one budget."""
    import numpy as np

    scores: Dict[str, float] = {}
    units: Dict[str, str] = {}
    tokens_used = 0.0
    chains_used = 0.0
    exhausted = 0.0
    empty = 0.0
    for p_index, pool in enumerate(pools):
        # Seeded per item so a method's draws are reproducible independently of the
        # order items are evaluated in, and so two methods see the same stream.
        rng = np.random.default_rng((seed, p_index))
        hits = 0.0
        for _ in range(n_repeats):
            result = method.alloc(pool, budget_tokens, rng)
            hits += 1.0 if result.correct else 0.0
            tokens_used += result.tokens
            chains_used += result.n_chains
            exhausted += 1.0 if result.exhausted else 0.0
            empty += 1.0 if result.empty else 0.0
        scores[pool.item_id] = hits / n_repeats
        units[pool.item_id] = pool.bootstrap_unit
    n_draws = max(1, len(pools) * n_repeats)
    return MethodPoint(
        method=method.name,
        baseline=method.baseline,
        dataset=pools[0].dataset if pools else "",
        subset=pools[0].subset if pools else None,
        model=pools[0].model if pools else "",
        seed=pools[0].seed if pools else 0,
        budget_tokens=budget_tokens,
        budget_multiple=budget_multiple,
        accuracy=sum(scores.values()) / len(scores) if scores else float("nan"),
        mean_tokens_used=tokens_used / n_draws,
        mean_chains_used=chains_used / n_draws,
        pool_exhausted_rate=exhausted / n_draws,
        empty_draw_rate=empty / n_draws,
        n_items=len(scores),
        scores=scores,
        units=units,
    )


def budget_curves(
    corpus: Corpus,
    seed: int = 0,
    cdv_configs: Optional[Sequence[str]] = None,
    multiples: Sequence[int] = DEFAULT_BUDGET_MULTIPLES,
    n_repeats: int = DEFAULT_N_REPEATS,
    draw_seed: int = 0,
) -> Tuple[List[MethodPoint], List[str]]:
    """Accuracy-vs-tokens for every method, benchmark and model. Backs Figure F5."""
    points: List[MethodPoint] = []
    warnings: List[str] = []
    for dataset, subset in corpus.benchmarks(include_separate_arms=True):
        for model in corpus.models():
            pools = build_item_pools(corpus, dataset, subset, model, seed=seed)
            if not pools:
                continue
            zero = _zero_token_warning(pools)
            if zero:
                warnings.append(f"{dataset}/{model}: {zero}")
            available = sorted({c for p in pools for c in p.configs()})
            methods = build_methods(available, cdv_configs=cdv_configs)
            if not methods:
                continue
            grid = token_budgets(pools, methods, multiples)
            for multiple, budget in zip(grid["multiples"], grid["budgets"]):
                for method in methods:
                    method_pools = _pools_for_method(
                        method, corpus, dataset, subset, model, seed, pools
                    )
                    points.append(
                        evaluate_method(
                            method, method_pools, budget, float(multiple),
                            n_repeats=n_repeats, seed=draw_seed,
                        )
                    )
            if grid["dropped_budgets"]:
                warnings.append(
                    f"{dataset}/{model}: token grid capped at "
                    f"{grid['binding_capacity']:.0f} tokens/item by "
                    f"{grid['limiting_method']}; {len(grid['dropped_budgets'])} "
                    "requested budget point(s) exceed what it can spend"
                )
    return points, warnings


# ------------------------------------------------------- paired comparison vs SC
def compare_to_sc(
    points: Sequence[MethodPoint],
    reference_method: str = "sc",
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
) -> List[Dict[str, Any]]:
    """Paired differences against self-consistency at each matched budget.

    Effect size with a clustered bootstrap CI leads; McNemar's p-value follows, as
    METHOD_SPEC 7.2 requires. Holm-Bonferroni is applied within each
    (benchmark, model, budget) family and the family size is reported, because a
    reader cannot interpret an adjusted p-value without it.
    """
    from ..metrics import bootstrap_ci

    grouped: Dict[Tuple[str, str, str, int, float], List[MethodPoint]] = {}
    for point in points:
        grouped.setdefault(
            (point.dataset, point.subset or "", point.model, point.seed, point.budget_multiple),
            [],
        ).append(point)

    rows: List[Dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        reference = next((p for p in group if p.method == reference_method), None)
        if reference is None:
            continue
        family: List[Dict[str, Any]] = []
        for point in group:
            if point.method == reference_method:
                continue
            items = sorted(set(point.scores) & set(reference.scores))
            if not items:
                continue
            diffs = [point.scores[i] - reference.scores[i] for i in items]
            # Resample the clustering unit (template for GSM-Symbolic, else item).
            unit_of = {i: reference.units.get(i, i) for i in items}
            clusters: Dict[str, List[float]] = {}
            for item, diff in zip(items, diffs):
                clusters.setdefault(unit_of[item], []).append(diff)
            ci_low, ci_high, _point = bootstrap_ci(
                [sum(v) / len(v) for v in clusters.values()],
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                seed=bootstrap_seed,
            )
            a_verdicts = point.verdicts()
            b_verdicts = reference.verdicts()
            test = mcnemar(
                [a_verdicts[i] for i in items], [b_verdicts[i] for i in items]
            )
            family.append(
                {
                    "dataset": key[0],
                    "subset": key[1],
                    "model": key[2],
                    "seed": key[3],
                    "budget_multiple": key[4],
                    "budget_tokens": point.budget_tokens,
                    "method": point.method,
                    "baseline": point.baseline,
                    "accuracy": point.accuracy,
                    "reference_method": reference_method,
                    "reference_accuracy": reference.accuracy,
                    "delta": point.accuracy - reference.accuracy,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "n_items": len(items),
                    "n_clusters": len(clusters),
                    "mcnemar_p": test.get("p_value"),
                    "mcnemar_b01": test.get("b01"),
                    "mcnemar_b10": test.get("b10"),
                    "mcnemar_n_discordant": test.get("n_discordant"),
                    "tokens_used": point.mean_tokens_used,
                    "reference_tokens_used": reference.mean_tokens_used,
                    "token_match_error": (
                        abs(point.mean_tokens_used - reference.mean_tokens_used)
                        / reference.mean_tokens_used
                        if reference.mean_tokens_used
                        else float("nan")
                    ),
                }
            )
        adjusted = holm_bonferroni([r["mcnemar_p"] or 1.0 for r in family], alpha=alpha)
        for i, row in enumerate(family):
            row["p_holm"] = adjusted["p_adjusted"][i]
            row["significant_holm"] = adjusted["reject"][i]
            row["family_size"] = adjusted["family_size"]
        rows.extend(family)
    return rows


def cdv_summary(
    points: Sequence[MethodPoint], comparisons: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Prediction P5 and the falsification hook at the largest matched budget."""
    if not points:
        return {"n_points": 0, "cdv_beats_sc": None}
    largest = max(p.budget_multiple for p in points)
    at_largest = [c for c in comparisons if c["budget_multiple"] == largest]
    cdv_rows = [c for c in at_largest if c["method"] == "cdv"]
    temp_rows = [c for c in at_largest if c["method"] == "temp_sc"]
    mean_delta = (
        sum(c["delta"] for c in cdv_rows) / len(cdv_rows) if cdv_rows else float("nan")
    )
    temp_delta = (
        sum(c["delta"] for c in temp_rows) / len(temp_rows) if temp_rows else float("nan")
    )
    worst_match = max(
        (c.get("token_match_error") or 0.0 for c in comparisons), default=float("nan")
    )
    return {
        "n_points": len(points),
        "largest_budget_multiple": largest,
        "cdv_minus_sc_at_largest": mean_delta,
        "temp_sc_minus_sc_at_largest": temp_delta,
        "cdv_minus_temp_sc_at_largest": mean_delta - temp_delta,
        "n_cells_cdv_significant": sum(1 for c in cdv_rows if c.get("significant_holm")),
        "n_cells_cdv_compared": len(cdv_rows),
        "worst_token_match_error": worst_match,
        "prediction": "P5",
        "statement": (
            "at matched completion-token budget, configuration-diversified voting "
            "plateaus above single-configuration self-consistency, and above "
            "temperature-diversified sampling"
        ),
        "cdv_beats_sc": bool(mean_delta > 0) if cdv_rows else None,
        "cdv_beats_temperature_diversity": (
            bool(mean_delta > temp_delta) if cdv_rows and temp_rows else None
        ),
        "falsification_hook": (
            "if cdv_minus_sc_at_largest is <= 0, the mechanism claim of METHOD_SPEC "
            "5.5 is refuted at this budget and must be reported as such"
        ),
    }


# ----------------------------------------------- B9: transferred-difficulty budget
def transferred_difficulty_allocation(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    source_config: str = REFERENCE_CONFIG_ID,
    target_config: str = "c1",
    seed: int = 0,
    hard_fraction: float = 0.5,
    boost: float = 2.0,
    rule: str = "uncertainty",
    n_repeats: int = DEFAULT_N_REPEATS,
    draw_seed: int = 0,
) -> Dict[str, Any]:
    """Baseline B9 and the applied consequence behind Table T6.

    Adaptive test-time compute estimates per-item difficulty once and then spends
    more on the hard items. The question this paper's result raises is what happens
    when the estimate came from a *different* elicitation configuration than the one
    being run. Three allocations are compared at the same total tokens:

    * `uniform`     -- the same budget everywhere; no difficulty estimate needed.
    * `transferred` -- hard set taken from `source_config`, spent in `target_config`.
    * `aware`       -- hard set taken from `target_config` itself, the ceiling any
      transferred estimate is trying to reach.

    If difficulty were a property of the item, `transferred` and `aware` would
    coincide. The gap is the cost of the assumption.

    `rule` selects which items count as worth extra compute:

    * `uncertainty` (default) targets `p_hat` near 0.5, where extra chains can
      actually change the vote.
    * `difficulty` targets the lowest `p_hat`, the literal reading of
      "allocate to the hard items".

    The default is `uncertainty` because the literal difficulty rule is known to be
    self-defeating: on an item the model gets right a quarter of the time, more
    chains drive the plurality *further* onto the wrong class, so a difficulty-ranked
    allocation can lose to uniform and the comparison then measures nothing about
    transfer. Both are available and METHOD_SPEC does not specify which; see the
    discrepancy note.
    """
    import numpy as np

    pools = build_item_pools(
        corpus, dataset, subset, model, seed=seed,
        configs=[source_config, target_config],
    )
    pools = [p for p in pools if source_config in p.classes and target_config in p.classes]
    out: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "source_config": source_config,
        "target_config": target_config,
        "n_items": len(pools),
        "hard_fraction": hard_fraction,
        "boost": boost,
        "rule": rule,
    }
    if len(pools) < 8:
        out["error"] = "too few crossed items for the allocation comparison"
        return out
    if rule not in ("uncertainty", "difficulty"):
        raise ValueError(f"rule must be 'uncertainty' or 'difficulty', got {rule!r}")

    def priority(pool: ItemPools, config: str) -> float:
        """Lower sorts first, i.e. gets the extra budget."""
        p_hat = float(pool.correct[config].mean())
        if rule == "difficulty":
            return p_hat
        return abs(2.0 * p_hat - 1.0)  # 0 at p_hat = 0.5, 1 at a saturated cell

    n_hard = max(1, int(round(hard_fraction * len(pools))))

    def hard_set(config: str) -> set:
        # Ties broken by item id so the selection is deterministic and does not
        # depend on dict ordering.
        ranked = sorted(pools, key=lambda p: (priority(p, config), p.item_id))
        return {p.item_id for p in ranked[:n_hard]}

    hard_transferred = hard_set(source_config)
    hard_aware = hard_set(target_config)

    mean_tokens = float(
        np.mean([p.tokens[target_config].mean() for p in pools])
    )
    base_multiple = 8.0
    total_budget = base_multiple * mean_tokens * len(pools)

    def allocate(hard: Optional[set]) -> Dict[str, float]:
        if hard is None:
            per_item = {p.item_id: total_budget / len(pools) for p in pools}
        else:
            weights = {p.item_id: (boost if p.item_id in hard else 1.0) for p in pools}
            scale = total_budget / sum(weights.values())
            per_item = {item: w * scale for item, w in weights.items()}
        return per_item

    method = Method("sc", "B3", _single_config(target_config), "", (target_config,))
    results: Dict[str, Dict[str, float]] = {}
    for name, hard in (
        ("uniform", None),
        ("transferred", hard_transferred),
        ("aware", hard_aware),
    ):
        per_item = allocate(hard)
        hits, tokens, capped = 0.0, 0.0, 0
        for p_index, pool in enumerate(pools):
            rng = np.random.default_rng((draw_seed, p_index))
            budget = per_item[pool.item_id]
            for _ in range(n_repeats):
                result = method.alloc(pool, budget, rng)
                hits += 1.0 if result.correct else 0.0
                tokens += result.tokens
                capped += 1 if result.exhausted else 0
            # Unspendable budget (the pool ran out) is *not* redistributed: a real
            # scheduler could not know in advance that an item would exhaust its
            # pool, and redistributing it would let the adaptive arms quietly
            # consume more than the uniform arm.
        n_draws = len(pools) * n_repeats
        results[name] = {
            "accuracy": hits / n_draws,
            "mean_tokens_used": tokens / n_draws,
            "pool_exhausted_rate": capped / n_draws,
        }

    out.update(
        {
            "hard_set_jaccard_source_vs_target": (
                len(hard_transferred & hard_aware) / len(hard_transferred | hard_aware)
            ),
            "accuracy_uniform": results["uniform"]["accuracy"],
            "accuracy_transferred": results["transferred"]["accuracy"],
            "accuracy_aware": results["aware"]["accuracy"],
            "tokens_uniform": results["uniform"]["mean_tokens_used"],
            "tokens_transferred": results["transferred"]["mean_tokens_used"],
            "tokens_aware": results["aware"]["mean_tokens_used"],
            "gain_aware_over_uniform": results["aware"]["accuracy"]
            - results["uniform"]["accuracy"],
            "gain_transferred_over_uniform": results["transferred"]["accuracy"]
            - results["uniform"]["accuracy"],
            "pool_exhausted_rate_uniform": results["uniform"]["pool_exhausted_rate"],
            "pool_exhausted_rate_aware": results["aware"]["pool_exhausted_rate"],
            # When both arms routinely run out of chains, they are both pinned at the
            # pool ceiling and the comparison measures nothing. Better to say so than
            # to report a difference of zero as evidence of transfer.
            "valid": bool(results["aware"]["pool_exhausted_rate"] < 0.5),
            "transfer_degradation": results["aware"]["accuracy"]
            - results["transferred"]["accuracy"],
            # A ratio to a near-zero denominator is noise dressed as a finding: if
            # configuration-aware allocation barely beats uniform on this cell, there
            # is no gain to retain and the ratio is not reportable.
            "fraction_of_gain_retained": (
                (results["transferred"]["accuracy"] - results["uniform"]["accuracy"])
                / (results["aware"]["accuracy"] - results["uniform"]["accuracy"])
                if results["aware"]["accuracy"] - results["uniform"]["accuracy"]
                >= MIN_REPORTABLE_GAIN
                else float("nan")
            ),
            "gain_reportable": bool(
                results["aware"]["accuracy"] - results["uniform"]["accuracy"]
                >= MIN_REPORTABLE_GAIN
            ),
            "prediction": "P6",
            "statement": (
                "difficulty estimated under one configuration retains materially "
                "less than all of the adaptive-allocation gain available under the "
                "configuration actually run"
            ),
        }
    )
    return out


def cdv_gain_decomposition(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Split CDV gains into new modal mass vs configuration selection."""
    from .modes import mixture_pi_mode_per_item

    configs = list(configs) if configs else list(DEFAULT_CDV_CONFIG_IDS)
    pools = build_item_pools(corpus, dataset, subset, model, seed=seed, configs=configs)
    if not pools:
        return {"n_items": 0}

    cdv_method = Method("cdv", "CDV", _pooled_configs(configs), "", tuple(configs))
    sc_method = Method("sc", "B3", _single_config(REFERENCE_CONFIG_ID), "", (REFERENCE_CONFIG_ID,))
    grid = token_budgets(pools, build_methods(configs, cdv_configs=configs))
    budget = grid["budgets"][-1] if grid["budgets"] else 1.0

    import numpy as np

    new_mass = selection = both_wrong = 0
    for p_index, pool in enumerate(pools):
        rng = np.random.default_rng((seed, p_index))
        cdv = cdv_method.alloc(pool, budget, rng)
        sc = sc_method.alloc(pool, budget, rng)
        per_cfg_correct = {
            c: bool(pool.correct[c].mean() > 0.5) if c in pool.correct else False
            for c in configs
        }
        any_single = any(per_cfg_correct.values())
        if cdv.correct and not sc.correct:
            if not any_single:
                new_mass += 1
            else:
                selection += 1
        if not cdv.correct and not sc.correct:
            both_wrong += 1

    n = len(pools)
    mixture = mixture_pi_mode_per_item(corpus, dataset, subset, model, configs, seed)
    sc_plug = sum(1 for v in mixture.values() if v.get("sc_modal_correct")) / max(1, n)
    mix_plug = sum(1 for v in mixture.values() if v.get("mixture_modal_correct")) / max(1, n)
    return {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_items": n,
        "cdv_wins_over_sc": new_mass + selection,
        "new_modal_mass": new_mass,
        "configuration_selection": selection,
        "fraction_new_modal_mass": new_mass / max(1, new_mass + selection),
        "pi_mode_sc": sc_plug,
        "pi_mode_mixture": mix_plug,
        "pi_mode_gain": mix_plug - sc_plug,
        "prediction": "F3_primary",
        "statement": "plug-in pi_mode(mixture) exceeds pi_mode(c0)",
        "supported": bool(mix_plug > sc_plug),
    }


def plug_in_plateau_summary(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Primary F3/P5 test: π_mode(c0) vs π_mode(mixture) per (dataset, model)."""
    configs = list(configs) if configs else list(DEFAULT_CDV_CONFIG_IDS)
    rows: List[Dict[str, Any]] = []
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            row = cdv_gain_decomposition(
                corpus, dataset, subset, model, configs=configs, seed=seed
            )
            if row.get("n_items", 0) > 0:
                rows.append(row)
    return rows


def greedy_accuracy(corpus: Corpus) -> List[Dict[str, Any]]:
    """Baseline B2 from the auxiliary greedy pass, if one was run."""
    rows: Dict[Tuple[str, str, str, str], List[Cell]] = {}
    for cell in corpus.aux_cells("greedy_pass"):
        rows.setdefault(
            (cell.dataset, cell.subset or "", cell.model, cell.config), []
        ).append(cell)
    out: List[Dict[str, Any]] = []
    for (dataset, subset, model, config), cells in sorted(rows.items()):
        out.append(
            {
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "config": config,
                "baseline": "B2",
                "n_items": len(cells),
                "greedy_accuracy": sum(c.k for c in cells) / max(1, sum(c.n for c in cells)),
                "mean_tokens": sum(c.tokens_completion for c in cells) / max(1, len(cells)),
            }
        )
    return out
