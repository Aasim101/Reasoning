"""Algorithm 4 and the mechanism: modal ceilings and mode transitions (METHOD_SPEC 5.5).

The paper's mechanism claim is that changing the elicitation configuration
*reorders* the top competing answer classes. That is what separates it from "any
diversity reduces variance": i.i.d. resampling at a fixed configuration cannot
move the modal class, so it cannot move the self-consistency plateau, whereas a
reordering can.

This module supplies the two halves of the evidence.

**The ceiling.** `pi_mode(m, c)` is the fraction of items whose modal canonical
class is correct -- the self-consistency plateau for that cell. Reported next to
`pass@N` so the identifiability gap (`pass@N - pi_mode`) is visible per cell, and
across configurations so the spread of the plateau is visible at all. If
`pi_mode` moves with the configuration then it is not the model-level constant the
ceiling literature's framing implies.

**The transitions.** For a configuration pair, each item's mode either stays or
moves. A move is *corrective* (wrong to right), *destructive* (right to wrong), or
*benign* (the class changed but the verdict did not, or the class did not change).
The prediction under test is that moves concentrate on **small-margin** items --
items where the top-1 and top-2 classes hold comparable mass, which are the only
items where a mode can move at all. A confirmed concentration is what makes the
mechanism a mechanism rather than a correlation, and it is also what predicts
*which* items configuration-diversified voting can help.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import DEFAULT_ALPHA, bca_ci, mann_whitney_u, pass_at_k
from .corpus import Cell, Corpus

log = logging.getLogger(__name__)

TRANSITION_KINDS: Tuple[str, ...] = ("benign", "corrective", "destructive")

#: Items with a top1-top2 frequency gap at or below this are "small margin".
#: One sample out of 24 is 0.042, so 0.25 is six samples: comfortably inside the
#: range where a configuration change can plausibly flip the ordering, and well
#: away from the 1.0 of a unanimous cell.
SMALL_MARGIN_THRESHOLD = 0.25

#: Bootstrap replicates for the per-row proportions here. Lower than the 10k used
#: for the headline variance shares because these are simple means over items,
#: whose interval is stable by 2k, and there are hundreds of rows: the full count
#: would add minutes of CPU for no change in the third decimal place.
ROW_N_BOOTSTRAP = 2_000


# ------------------------------------------------------------------ modal ceiling
@dataclass
class CeilingRow:
    """`pi_mode`, coverage and the gap between them, for one cell column."""

    dataset: str
    subset: Optional[str]
    model: str
    config: str
    seed: int
    n_items: int
    n_samples: int
    avg_at_1: float
    pi_mode: float
    pi_mode_ci: Tuple[float, float]
    pass_at_n: float
    identifiability_gap: float
    modal_tie_rate: float
    extraction_failure_rate: float
    unparsed_mode_rate: float
    saturated_rate: float
    mean_margin: float
    mean_tokens_per_sample: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "config": self.config,
            "seed": self.seed,
            "n_items": self.n_items,
            "n_samples_per_item": self.n_samples,
            "avg_at_1": self.avg_at_1,
            "pi_mode": self.pi_mode,
            "pi_mode_ci_low": self.pi_mode_ci[0],
            "pi_mode_ci_high": self.pi_mode_ci[1],
            "pass_at_n": self.pass_at_n,
            "identifiability_gap": self.identifiability_gap,
            "modal_tie_rate": self.modal_tie_rate,
            "extraction_failure_rate": self.extraction_failure_rate,
            "unparsed_mode_rate": self.unparsed_mode_rate,
            "saturated_rate": self.saturated_rate,
            "mean_margin": self.mean_margin,
            "mean_tokens_per_sample": self.mean_tokens_per_sample,
        }


def ceiling_rows(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_bootstrap: int = ROW_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
    include_separate_arms: bool = True,
) -> List[CeilingRow]:
    """`pi_mode` and `pass@N` per (dataset, model, configuration)."""
    rows: List[CeilingRow] = []
    wanted = set(configs) if configs else None
    grouped: Dict[Tuple[str, Optional[str], str, str, int], List[Cell]] = {}
    for cell in corpus.select(include_separate_arms=include_separate_arms):
        if cell.seed != seed or cell.n == 0:
            continue
        if wanted is not None and cell.config not in wanted:
            continue
        grouped.setdefault(
            (cell.dataset, cell.subset, cell.model, cell.config, cell.seed), []
        ).append(cell)

    for (dataset, subset, model, config, cell_seed), cells in sorted(
        grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        n_min = min(c.n for c in cells)
        modal_flags = [1.0 if c.modal_correct() else 0.0 for c in cells]
        interval = bca_ci(
            modal_flags,
            lambda sample: sum(sample) / len(sample) if sample else float("nan"),
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=bootstrap_seed,
        )
        coverage = [pass_at_k(c.n, c.k, n_min) for c in cells if c.n >= n_min]
        pi_mode = sum(modal_flags) / len(modal_flags)
        pass_n = sum(coverage) / len(coverage) if coverage else float("nan")
        total_samples = sum(c.n for c in cells)
        rows.append(
            CeilingRow(
                dataset=dataset,
                subset=subset,
                model=model,
                config=config,
                seed=cell_seed,
                n_items=len(cells),
                n_samples=n_min,
                avg_at_1=sum(c.p_hat for c in cells) / len(cells),
                pi_mode=pi_mode,
                pi_mode_ci=(float(interval["ci_low"]), float(interval["ci_high"])),
                pass_at_n=pass_n,
                identifiability_gap=pass_n - pi_mode,
                modal_tie_rate=sum(1 for c in cells if c.modal_tie()) / len(cells),
                extraction_failure_rate=sum(c.n_extraction_failures for c in cells)
                / max(1, total_samples),
                unparsed_mode_rate=sum(
                    1 for c in cells if c.modal_class() == EXTRACTION_FAILURE_CLASS
                )
                / len(cells),
                saturated_rate=sum(1 for c in cells if c.saturated) / len(cells),
                mean_margin=sum(c.margin() for c in cells) / len(cells),
                mean_tokens_per_sample=sum(c.tokens_completion for c in cells)
                / max(1, total_samples),
            )
        )
    return rows


def union_ceiling(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Fraction of items where *some* configuration's mode is correct.

    An oracle upper bound for any configuration-selection policy (baseline B11), so
    the gap between it and configuration-diversified voting sizes the remaining
    headroom.
    """
    configs = list(configs) if configs else [
        c for c in corpus.configs() if c in PRIMARY_CONFIG_IDS
    ] or corpus.configs()
    per_item: Dict[str, Dict[str, bool]] = {}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.model != model
            or cell.seed != seed
            or cell.config not in configs
            or cell.n == 0
        ):
            continue
        per_item.setdefault(cell.item_id, {})[cell.config] = cell.modal_correct()
    complete = {
        item: flags for item, flags in per_item.items() if len(flags) == len(configs)
    }
    if not complete:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "n_items": 0,
            "union_ceiling": float("nan"),
        }
    best_single = max(
        sum(1 for flags in complete.values() if flags[c]) / len(complete)
        for c in configs
    )
    union = sum(1 for flags in complete.values() if any(flags.values())) / len(complete)
    return {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "configs": ",".join(configs),
        "n_items": len(complete),
        "union_ceiling": union,
        "best_single_config_pi_mode": best_single,
        "oracle_headroom_over_best_config": union - best_single,
    }


def mixture_pi_mode_per_item(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Sequence[str],
    seed: int = 0,
    reference: str = REFERENCE_CONFIG_ID,
) -> Dict[str, Dict[str, Any]]:
    """Per-item plug-in π_mode for c0 alone vs the uniform configuration mixture."""
    from collections import Counter

    per_item: Dict[str, Dict[str, List[str]]] = {}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.model != model
            or cell.seed != seed
            or cell.config not in configs
            or cell.n == 0
        ):
            continue
        per_item.setdefault(cell.item_id, {})[cell.config] = [
            s.canonical_class for s in cell.samples
        ]

    ref_classes: Dict[str, List[str]] = {}
    for cell in corpus.cells.values():
        if (
            cell.dataset == dataset
            and cell.subset == subset
            and cell.model == model
            and cell.seed == seed
            and cell.config == reference
            and cell.n > 0
        ):
            ref_classes[cell.item_id] = [s.canonical_class for s in cell.samples]

    out: Dict[str, Dict[str, Any]] = {}
    for item_id, by_config in per_item.items():
        if len(by_config) < len(configs):
            continue
        pooled: List[str] = []
        for c in configs:
            pooled.extend(by_config.get(c, []))
        ref = ref_classes.get(item_id, [])
        if not pooled or not ref:
            continue

        def modal_correct(classes: List[str]) -> bool:
            if not classes:
                return False
            counts = Counter(classes)
            top = max(counts.values())
            mode = next(cl for cl, ct in counts.items() if ct == top)
            if mode == EXTRACTION_FAILURE_CLASS:
                return False
            return any(
                s.is_correct
                for cell in corpus.cells.values()
                if cell.item_id == item_id
                and cell.dataset == dataset
                and cell.subset == subset
                and cell.model == model
                and cell.seed == seed
                for s in cell.samples
                if s.canonical_class == mode
            )

        out[item_id] = {
            "sc_modal_correct": modal_correct(ref),
            "mixture_modal_correct": modal_correct(pooled),
        }
    return out


def ceiling_spread(rows: Sequence[CeilingRow]) -> List[Dict[str, Any]]:
    """Prediction P4: `pi_mode` varies by more than 3 points across configurations."""
    grouped: Dict[Tuple[str, Optional[str], str], List[CeilingRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.subset, row.model), []).append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, subset, model), group in sorted(
        grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        values = [r.pi_mode for r in group if math.isfinite(r.pi_mode)]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        out.append(
            {
                "dataset": dataset,
                "subset": subset or "",
                "model": model,
                "n_configs": len(values),
                "pi_mode_min": min(values),
                "pi_mode_max": max(values),
                "pi_mode_spread_points": 100.0 * spread,
                "argmin_config": min(group, key=lambda r: r.pi_mode).config,
                "argmax_config": max(group, key=lambda r: r.pi_mode).config,
                "prediction": "P4",
                "statement": (
                    "pi_mode varies by more than 3 accuracy points across "
                    "configurations for a fixed (model, dataset)"
                ),
                "supported": bool(100.0 * spread > 3.0),
            }
        )
    return out


# ------------------------------------------------------------ transition taxonomy
def classify_transition(
    mode_a: Optional[str],
    correct_a: bool,
    mode_b: Optional[str],
    correct_b: bool,
) -> Dict[str, Any]:
    """Label one item's mode transition between two configurations.

    METHOD_SPEC 5.5 step 3 defines the split over the items whose mode *changed*:
    benign when both verdicts agree, corrective for wrong to right, destructive for
    right to wrong. An item whose mode did not change is recorded as `reordered:
    False` and also carries the `benign` label, because a table of the three kinds
    has to sum to the item count.
    """
    reordered = mode_a != mode_b
    if correct_a and not correct_b:
        kind = "destructive"
    elif (not correct_a) and correct_b:
        kind = "corrective"
    else:
        kind = "benign"
    return {
        "reordered": bool(reordered),
        "kind": kind,
        "mode_a": mode_a,
        "mode_b": mode_b,
        "correct_a": bool(correct_a),
        "correct_b": bool(correct_b),
        "unparsed_involved": EXTRACTION_FAILURE_CLASS in (mode_a, mode_b),
    }


@dataclass
class TransitionRow:
    """Mode-transition taxonomy for one (dataset, model, configuration pair)."""

    dataset: str
    subset: Optional[str]
    model: str
    config_a: str
    config_b: str
    n_items: int
    reorder_rate: float
    reorder_rate_ci: Tuple[float, float]
    n_benign: int
    n_corrective: int
    n_destructive: int
    net_corrective_rate: float
    pi_mode_a: float
    pi_mode_b: float
    mean_margin_reordered: float
    mean_margin_stable: float
    small_margin_reorder_rate: float
    large_margin_reorder_rate: float
    margin_test: Dict[str, Any] = field(default_factory=dict)
    reorder_rate_by_margin_quartile: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "config_a": self.config_a,
            "config_b": self.config_b,
            "n_items": self.n_items,
            "reorder_rate": self.reorder_rate,
            "reorder_rate_ci_low": self.reorder_rate_ci[0],
            "reorder_rate_ci_high": self.reorder_rate_ci[1],
            "n_benign": self.n_benign,
            "n_corrective": self.n_corrective,
            "n_destructive": self.n_destructive,
            "benign_rate": self.n_benign / self.n_items if self.n_items else float("nan"),
            "corrective_rate": self.n_corrective / self.n_items if self.n_items else float("nan"),
            "destructive_rate": self.n_destructive / self.n_items if self.n_items else float("nan"),
            "net_corrective_rate": self.net_corrective_rate,
            "pi_mode_a": self.pi_mode_a,
            "pi_mode_b": self.pi_mode_b,
            "mean_margin_reordered": self.mean_margin_reordered,
            "mean_margin_stable": self.mean_margin_stable,
            "small_margin_reorder_rate": self.small_margin_reorder_rate,
            "large_margin_reorder_rate": self.large_margin_reorder_rate,
            "margin_auc": self.margin_test.get("effect_size_auc"),
            "margin_p_value": self.margin_test.get("p_value"),
            "reorder_rate_by_margin_quartile": ";".join(
                f"{v:.4f}" for v in self.reorder_rate_by_margin_quartile
            ),
        }


def transition_records(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[Tuple[str, Optional[str], str, str, str], List[Dict[str, Any]]]:
    """Item-level transition verdicts, keyed by (dataset, subset, model, cA, cB).

    Exposed separately from `transition_rows` because the margin figure bins the
    individual items rather than reading an aggregate.
    """
    configs = list(configs) if configs else [
        c for c in corpus.configs() if c in PRIMARY_CONFIG_IDS
    ] or corpus.configs()

    # (dataset, subset, model, config) -> {item_id: cell}, built in one pass so the
    # pair loop below is O(items) per pair rather than O(corpus) per pair.
    columns: Dict[Tuple[str, Optional[str], str, str], Dict[str, Cell]] = {}
    wanted = set(configs)
    for cell in corpus.cells.values():
        if cell.seed != seed or cell.n == 0 or cell.config not in wanted:
            continue
        columns.setdefault(
            (cell.dataset, cell.subset, cell.model, cell.config), {}
        )[cell.item_id] = cell

    out: Dict[Tuple[str, Optional[str], str, str, str], List[Dict[str, Any]]] = {}
    keys = sorted({(d, s, m) for (d, s, m, _c) in columns})
    for dataset, subset, model in keys:
        for config_a, config_b in combinations(configs, 2):
            col_a = columns.get((dataset, subset, model, config_a))
            col_b = columns.get((dataset, subset, model, config_b))
            if not col_a or not col_b:
                continue
            items = sorted(set(col_a) & set(col_b))
            if len(items) < 4:
                continue
            records: List[Dict[str, Any]] = []
            for item in items:
                cell_a, cell_b = col_a[item], col_b[item]
                verdict = classify_transition(
                    cell_a.modal_class(),
                    cell_a.modal_correct(),
                    cell_b.modal_class(),
                    cell_b.modal_correct(),
                )
                verdict["item_id"] = item
                # The margin of the *source* configuration is the covariate: the
                # prediction is about how fragile the mode was before the
                # configuration changed. `min` of the two would mix the outcome
                # into the predictor, which would make the test circular.
                verdict["margin"] = cell_a.margin()
                verdict["margin_b"] = cell_b.margin()
                verdict["p_hat_a"] = cell_a.p_hat
                verdict["p_hat_b"] = cell_b.p_hat
                verdict["bootstrap_unit"] = cell_a.bootstrap_unit()
                records.append(verdict)
            out[(dataset, subset, model, config_a, config_b)] = records
    return out


def transition_rows(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_bootstrap: int = ROW_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
    small_margin_threshold: float = SMALL_MARGIN_THRESHOLD,
) -> List[TransitionRow]:
    """The full taxonomy plus the small-margin concentration test, per config pair."""
    rows: List[TransitionRow] = []
    grouped = transition_records(corpus, configs=configs, seed=seed)
    for (dataset, subset, model, config_a, config_b), records in sorted(
        grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append(
            _transition_row(
                dataset, subset, model, config_a, config_b, records,
                n_bootstrap, alpha, bootstrap_seed, small_margin_threshold,
            )
        )
    return rows


def _transition_row(
    dataset: str,
    subset: Optional[str],
    model: str,
    config_a: str,
    config_b: str,
    records: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    alpha: float,
    bootstrap_seed: int,
    small_margin_threshold: float,
) -> TransitionRow:
    import numpy as np

    n = len(records)
    reordered = [r for r in records if r["reordered"]]
    stable = [r for r in records if not r["reordered"]]
    counts = {kind: sum(1 for r in records if r["kind"] == kind) for kind in TRANSITION_KINDS}

    flags = [1.0 if r["reordered"] else 0.0 for r in records]
    interval = bca_ci(
        flags,
        lambda sample: sum(sample) / len(sample) if sample else float("nan"),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=bootstrap_seed,
    )

    margins_reordered = [r["margin"] for r in reordered if math.isfinite(r["margin"])]
    margins_stable = [r["margin"] for r in stable if math.isfinite(r["margin"])]
    test = mann_whitney_u(margins_reordered, margins_stable)
    test["direction"] = (
        "reordered items have SMALLER margins (predicted)"
        if margins_reordered
        and margins_stable
        and sum(margins_reordered) / len(margins_reordered)
        < sum(margins_stable) / len(margins_stable)
        else "reordered items do NOT have smaller margins"
    )

    small = [r for r in records if math.isfinite(r["margin"]) and r["margin"] <= small_margin_threshold]
    large = [r for r in records if math.isfinite(r["margin"]) and r["margin"] > small_margin_threshold]
    all_margins = [r["margin"] for r in records if math.isfinite(r["margin"])]
    quartile_rates: List[float] = []
    if len(all_margins) >= 8:
        edges = np.quantile(all_margins, [0.25, 0.5, 0.75])
        buckets: List[List[Dict[str, Any]]] = [[], [], [], []]
        for r in records:
            margin = r["margin"]
            if not math.isfinite(margin):
                continue
            bucket = int(np.searchsorted(edges, margin, side="right"))
            buckets[min(bucket, 3)].append(r)
        quartile_rates = [
            (sum(1 for r in b if r["reordered"]) / len(b)) if b else float("nan")
            for b in buckets
        ]

    return TransitionRow(
        dataset=dataset,
        subset=subset,
        model=model,
        config_a=config_a,
        config_b=config_b,
        n_items=n,
        reorder_rate=len(reordered) / n if n else float("nan"),
        reorder_rate_ci=(float(interval["ci_low"]), float(interval["ci_high"])),
        n_benign=counts["benign"],
        n_corrective=counts["corrective"],
        n_destructive=counts["destructive"],
        net_corrective_rate=(counts["corrective"] - counts["destructive"]) / n if n else float("nan"),
        pi_mode_a=sum(1 for r in records if r["correct_a"]) / n if n else float("nan"),
        pi_mode_b=sum(1 for r in records if r["correct_b"]) / n if n else float("nan"),
        mean_margin_reordered=(
            sum(margins_reordered) / len(margins_reordered) if margins_reordered else float("nan")
        ),
        mean_margin_stable=(
            sum(margins_stable) / len(margins_stable) if margins_stable else float("nan")
        ),
        small_margin_reorder_rate=(
            sum(1 for r in small if r["reordered"]) / len(small) if small else float("nan")
        ),
        large_margin_reorder_rate=(
            sum(1 for r in large if r["reordered"]) / len(large) if large else float("nan")
        ),
        margin_test=test,
        reorder_rate_by_margin_quartile=quartile_rates,
    )


#: Fixed margin bins for the mechanism figure. Fixed rather than quantile-derived
#: so the same x-axis is comparable across models, datasets and pairs; the
#: quantile version lives on `TransitionRow` for the table.
MARGIN_BIN_EDGES: Tuple[float, ...] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)


def reorder_rate_by_margin_bin(
    records: Sequence[Dict[str, Any]],
    edges: Sequence[float] = MARGIN_BIN_EDGES,
) -> List[Dict[str, Any]]:
    """Reorder rate within fixed source-margin bins; the shape of the mechanism.

    The prediction is a monotone decrease: the more decisively one answer class
    already won, the less a configuration change can move it.
    """
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Left-closed, right-open, except the final bin which must include 1.0
        # (unanimous cells, whose reorder rate is the most informative single
        # number in the figure).
        last = hi >= edges[-1]

        def _in_bin(margin: float, lo: float = lo, hi: float = hi, last: bool = last) -> bool:
            if not math.isfinite(margin):
                return False
            return lo <= margin < hi or (last and margin == hi)

        members = [
            r for r in records if _in_bin(float(r.get("margin", float("nan"))))
        ]
        n_reordered = sum(1 for r in members if r["reordered"])
        out.append(
            {
                "margin_low": lo,
                "margin_high": hi,
                "n_items": len(members),
                "reorder_rate": n_reordered / len(members) if members else float("nan"),
                "n_corrective": sum(1 for r in members if r["kind"] == "corrective"),
                "n_destructive": sum(1 for r in members if r["kind"] == "destructive"),
            }
        )
    return out


def transition_summary(rows: Sequence[TransitionRow]) -> Dict[str, Any]:
    """Whether the mechanism prediction holds: reorderings concentrate at small margin."""
    usable = [
        r
        for r in rows
        if math.isfinite(r.small_margin_reorder_rate)
        and math.isfinite(r.large_margin_reorder_rate)
    ]
    if not usable:
        return {"n_pairs": 0, "mechanism_supported": None}
    small = sum(r.small_margin_reorder_rate for r in usable) / len(usable)
    large = sum(r.large_margin_reorder_rate for r in usable) / len(usable)
    def _value(row: TransitionRow, key: str, default: float) -> float:
        # Not `x or default`: an AUC of exactly 0.0 is the strongest possible
        # evidence for the prediction and is also falsy, so the idiom would discard
        # precisely the cases the test is looking for.
        raw = row.margin_test.get(key)
        return float(raw) if raw is not None and math.isfinite(float(raw)) else default

    significant = [
        r
        for r in usable
        if _value(r, "p_value", 1.0) < 0.05 and _value(r, "effect_size_auc", 0.5) < 0.5
    ]
    return {
        "n_pairs": len(usable),
        "mean_reorder_rate": sum(r.reorder_rate for r in rows) / len(rows),
        "mean_corrective_rate": sum(
            r.n_corrective / r.n_items for r in rows if r.n_items
        ) / max(1, len(rows)),
        "mean_destructive_rate": sum(
            r.n_destructive / r.n_items for r in rows if r.n_items
        ) / max(1, len(rows)),
        "small_margin_reorder_rate": small,
        "large_margin_reorder_rate": large,
        "small_over_large_ratio": small / large if large else float("inf"),
        "n_pairs_significant": len(significant),
        "statement": (
            "mode reorderings concentrate on small-margin items (small gap between "
            "the top-1 and top-2 answer-class frequencies)"
        ),
        "mechanism_supported": bool(small > large),
    }
