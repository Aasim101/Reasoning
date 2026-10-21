"""Algorithm 1: variance decomposition of item difficulty (METHOD_SPEC 5.2).

The paper's headline figure. We decompose the Haldane-corrected empirical logit of
each cell's success rate over a fully crossed (item x model x configuration)
design, subtract the sampling-noise floor measured from the seed replicates, and
report each component as a share of the *corrected* total.

Three things here are worth reading before trusting the output.

**1. Why the noise correction changes only the residual, and why that is right.**
The estimator subtracts the noise floor from every mean square, exactly as the
spec directs. In a balanced crossed design that operation is analytically
equivalent to removing the floor from the residual component alone: each main and
interaction component is a contrast of mean squares whose coefficients sum to zero
(`sigma2_item = (MS_item - MS_im - MS_ic + MS_resid) / (M*C)`, and
`+1 -1 -1 +1 = 0`), so a constant subtracted from all mean squares cancels. Only
`sigma2_resid = MS_resid - floor` survives it. That is not a shortcut, it is where
independent per-cell sampling noise actually lives: it inflates the residual and
nothing else. What the correction buys is the *denominator* -- the shares are
taken over corrected total variance, so an uncorrected fit would understate every
structural share by attributing sampling noise to "explainable" variance.
`test_variance.py` asserts this identity on synthetic data with an injected floor.

**2. Saturated cells set the floor and more samples cannot lower it.** The
Haldane sampling variance is `1/(k+1/2) + 1/(N-k+1/2)`. At an interior cell it
halves when N doubles; at `k=0` or `k=N` it is approximately 2.0 and flat in N.
So the floor is dominated by decided cells. This is the arithmetic behind
METHOD_SPEC section 8.4's decision to buy items rather than samples, and
`saturated_cell_rate` is reported alongside every fit so the reader can see it.

**3. Items saturated in every cell carry no information and are excluded.** They
would contribute pure floor to every mean square and nothing to any structural
component. The censoring rate is reported, per METHOD_SPEC and the gap analysis's
Risk 4, rather than being quietly absorbed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import DEFAULT_ALPHA, DEFAULT_N_BOOTSTRAP, haldane_logit, haldane_logit_var
from .corpus import Cell, Corpus

log = logging.getLogger(__name__)

#: The components, in the order F1 stacks them.
COMPONENT_NAMES: Tuple[str, ...] = (
    "item",
    "model",
    "config",
    "item_model",
    "item_config",
    "model_config",
    "residual",
)

#: Bootstrap replicates for the shares. Reduced automatically for tiny designs.
DEFAULT_BOOTSTRAP_CHUNK = 250


# ------------------------------------------------------------------ mean squares
def mean_squares(z: Any) -> Dict[str, Any]:
    """ANOVA mean squares for a crossed (item x model x config) array, n=1 per cell.

    `z` has shape `(..., I, M, C)`; leading axes are treated as independent fits,
    which is what makes the item bootstrap fast enough to run at B = 10000.

    With one observation per cell the three-way interaction is not separable from
    error, so `residual` is `item x model x config` plus sampling noise. That is
    stated in the output rather than hidden, because the noise correction then
    subtracts the sampling part and leaves the three-way interaction behind.
    """
    import numpy as np

    z = np.asarray(z, dtype=float)
    if z.ndim < 3:
        raise ValueError(f"z must have at least 3 dims (I, M, C), got shape {z.shape}")
    n_items, n_models, n_configs = z.shape[-3], z.shape[-2], z.shape[-1]
    if min(n_items, n_models, n_configs) < 2:
        raise ValueError(
            f"the crossed design needs at least 2 levels on every factor, got "
            f"I={n_items}, M={n_models}, C={n_configs}. With a single level on any "
            "factor its main effect and every interaction involving it are "
            "unidentifiable."
        )

    mu = z.mean(axis=(-3, -2, -1), keepdims=True)
    m_i = z.mean(axis=(-2, -1), keepdims=True)
    m_m = z.mean(axis=(-3, -1), keepdims=True)
    m_c = z.mean(axis=(-3, -2), keepdims=True)
    m_im = z.mean(axis=-1, keepdims=True)
    m_ic = z.mean(axis=-2, keepdims=True)
    m_mc = z.mean(axis=-3, keepdims=True)

    def total(x: Any) -> Any:
        return (x**2).sum(axis=(-3, -2, -1))

    ss_item = n_models * n_configs * total(m_i - mu)
    ss_model = n_items * n_configs * total(m_m - mu)
    ss_config = n_items * n_models * total(m_c - mu)
    ss_im = n_configs * total(m_im - m_i - m_m + mu)
    ss_ic = n_models * total(m_ic - m_i - m_c + mu)
    ss_mc = n_items * total(m_mc - m_m - m_c + mu)
    residual = z - m_im - m_ic - m_mc + m_i + m_m + m_c - mu
    ss_resid = total(residual)

    df = {
        "item": n_items - 1,
        "model": n_models - 1,
        "config": n_configs - 1,
        "item_model": (n_items - 1) * (n_models - 1),
        "item_config": (n_items - 1) * (n_configs - 1),
        "model_config": (n_models - 1) * (n_configs - 1),
        "residual": (n_items - 1) * (n_models - 1) * (n_configs - 1),
    }
    ss = {
        "item": ss_item,
        "model": ss_model,
        "config": ss_config,
        "item_model": ss_im,
        "item_config": ss_ic,
        "model_config": ss_mc,
        "residual": ss_resid,
    }
    return {
        "ms": {name: ss[name] / df[name] for name in COMPONENT_NAMES},
        "ss": ss,
        "df": df,
        "dims": {"n_items": n_items, "n_models": n_models, "n_configs": n_configs},
        "residual_note": (
            "one observation per cell: the residual is item x model x config plus "
            "sampling noise; the noise correction removes the sampling part"
        ),
    }


def components_from_ms(
    ms: Dict[str, Any],
    n_items: int,
    n_models: int,
    n_configs: int,
    noise_floor: Any = 0.0,
) -> Dict[str, Any]:
    """Method-of-moments variance components for the crossed random-effects model.

    Inverting the expected mean squares of the fully crossed random model with one
    observation per cell:

        E[MS_resid]  = s2_e
        E[MS_mc]     = s2_e + I*s2_mc
        E[MS_ic]     = s2_e + M*s2_ic
        E[MS_im]     = s2_e + C*s2_im
        E[MS_item]   = s2_e + C*s2_im + M*s2_ic + M*C*s2_item
        E[MS_model]  = s2_e + C*s2_im + I*s2_mc + I*C*s2_model
        E[MS_config] = s2_e + M*s2_ic + I*s2_mc + I*M*s2_config

    `noise_floor` is subtracted from every mean square first, per METHOD_SPEC
    section 5.2 step 3. Broadcasting is preserved so the whole item bootstrap runs
    as array arithmetic.
    """
    corrected = {name: ms[name] - noise_floor for name in COMPONENT_NAMES}
    s2_e = corrected["residual"]
    s2_mc = (corrected["model_config"] - s2_e) / n_items
    s2_ic = (corrected["item_config"] - s2_e) / n_models
    s2_im = (corrected["item_model"] - s2_e) / n_configs
    s2_item = (
        corrected["item"] - corrected["item_model"] - corrected["item_config"] + s2_e
    ) / (n_models * n_configs)
    s2_model = (
        corrected["model"] - corrected["item_model"] - corrected["model_config"] + s2_e
    ) / (n_items * n_configs)
    s2_config = (
        corrected["config"] - corrected["item_config"] - corrected["model_config"] + s2_e
    ) / (n_items * n_models)
    return {
        "item": s2_item,
        "model": s2_model,
        "config": s2_config,
        "item_model": s2_im,
        "item_config": s2_ic,
        "model_config": s2_mc,
        "residual": s2_e,
    }


def shares_from_components(
    components: Dict[str, Any], clamp_negative: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalise components to shares of corrected total variance.

    Method-of-moments components can go negative when the true component is near
    zero; clamping to 0 is standard and is *reported* rather than hidden, because a
    heavily clamped fit is a signal that the design is underpowered rather than a
    finding.
    """
    import numpy as np

    clamped = {}
    was_clamped = {}
    for name in COMPONENT_NAMES:
        value = np.asarray(components[name], dtype=float)
        negative = value < 0
        clamped[name] = np.where(negative, 0.0, value) if clamp_negative else value
        was_clamped[name] = negative
    total = sum(clamped[name] for name in COMPONENT_NAMES)
    total = np.where(np.asarray(total) <= 0, np.nan, total)
    shares = {name: clamped[name] / total for name in COMPONENT_NAMES}
    return shares, {"clamped": was_clamped, "total": total}


# ------------------------------------------------------------------- noise floor
@dataclass
class NoiseFloor:
    """The sampling-variance floor of the empirical logit.

    Two independent estimates, and the paper should report both:

    * `replicate` -- the variance of `z` across independent sampling seeds at the
      reference configuration, per (item, model), averaged. This is the *measured*
      floor and the one that is subtracted, because it needs no distributional
      assumption. It is the quantity the gap analysis calls uncuttable.
    * `analytic` -- the mean of `1/(k+1/2) + 1/(N-k+1/2)` over cells. Available
      without replicates, so it exists to *validate* the measured floor. A large
      disagreement means the chains within a cell are not behaving like independent
      Bernoulli draws, which would itself be a finding.
    """

    replicate: float = float("nan")
    analytic: float = float("nan")
    used: str = "replicate"
    n_replicate_cells: int = 0
    n_seeds: int = 0
    n_pairs: int = 0
    #: item_id -> per-item floor, so the item bootstrap propagates its uncertainty.
    per_item: Dict[str, float] = field(default_factory=dict)

    @property
    def value(self) -> float:
        chosen = self.replicate if self.used == "replicate" else self.analytic
        if not math.isfinite(chosen):
            return self.analytic if math.isfinite(self.analytic) else 0.0
        return chosen

    def as_dict(self) -> Dict[str, Any]:
        return {
            "noise_floor_used": self.used,
            "noise_floor": self.value,
            "noise_floor_replicate": self.replicate,
            "noise_floor_analytic": self.analytic,
            "noise_floor_ratio_replicate_over_analytic": (
                self.replicate / self.analytic
                if math.isfinite(self.replicate) and self.analytic
                else float("nan")
            ),
            "n_replicate_cells": self.n_replicate_cells,
            "n_replicate_seeds": self.n_seeds,
            "n_replicate_pairs": self.n_pairs,
        }


def measure_noise_floor(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    config: str = REFERENCE_CONFIG_ID,
    prefer: str = "replicate",
) -> NoiseFloor:
    """Estimate the sampling-noise floor from the same-configuration seed arm."""
    import numpy as np

    models = list(models) if models else corpus.models()
    by_item_model: Dict[Tuple[str, str], List[float]] = {}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.config != config
            or cell.model not in models
            or cell.n == 0
        ):
            continue
        by_item_model.setdefault((cell.item_id, cell.model), []).append(cell.z)

    usable = {key: values for key, values in by_item_model.items() if len(values) >= 2}
    per_item: Dict[str, List[float]] = {}
    variances: List[float] = []
    seeds = sorted(
        {c.seed for c in corpus.cells.values()
         if c.dataset == dataset and c.subset == subset and c.config == config}
    )
    for (item_id, _model), values in usable.items():
        variance = float(np.var(values, ddof=1))
        variances.append(variance)
        per_item.setdefault(item_id, []).append(variance)

    analytic_cells = [
        c
        for c in corpus.cells.values()
        if c.dataset == dataset and c.subset == subset and c.model in models and c.n > 0
    ]
    analytic = (
        float(np.mean([c.sampling_var for c in analytic_cells])) if analytic_cells else float("nan")
    )
    replicate = float(np.mean(variances)) if variances else float("nan")

    n_seeds = len(seeds)
    floor = NoiseFloor(
        replicate=replicate,
        analytic=analytic,
        used=prefer if (prefer == "analytic" or math.isfinite(replicate)) else "analytic",
        n_replicate_cells=len(usable),
        n_seeds=n_seeds,
        n_pairs=n_seeds * (n_seeds - 1) // 2,
        per_item={k: float(np.mean(v)) for k, v in per_item.items()},
    )
    if not variances:
        log.warning(
            "no seed replicates found for %s/%s at %s: falling back to the analytic "
            "binomial floor. METHOD_SPEC section 8.5 forbids cutting the seed arm -- "
            "without it the noise correction rests on the binomial assumption alone.",
            dataset,
            subset,
            config,
        )
    elif math.isfinite(analytic) and analytic > 0:
        ratio = replicate / analytic
        if not 0.4 <= ratio <= 2.5:
            log.warning(
                "measured noise floor (%.4f) and analytic binomial floor (%.4f) "
                "disagree by %.2fx. The chains within a cell may not be behaving "
                "like independent draws; report this rather than picking one.",
                replicate,
                analytic,
                ratio,
            )
    return floor


# ---------------------------------------------------------------- the decomposition
@dataclass
class VarianceResult:
    """Everything F1 and T3 need, plus the caveats the paper must print."""

    dataset: str
    subset: Optional[str]
    models: List[str]
    configs: List[str]
    seed: int
    n_items: int
    components: Dict[str, float]
    shares: Dict[str, float]
    shares_ci: Dict[str, Tuple[float, float]]
    components_ci: Dict[str, Tuple[float, float]]
    mean_squares: Dict[str, float]
    noise_floor: Dict[str, Any]
    censoring: Dict[str, Any]
    clamping: Dict[str, Any]
    design: Dict[str, Any]
    n_bootstrap: int
    alpha: float
    bootstrap_seed: int
    saturated_cell_rate: float

    def as_rows(self) -> List[Dict[str, Any]]:
        """One row per component: the CSV backing of F1 (`tab3_variance_components`)."""
        rows = []
        for name in COMPONENT_NAMES:
            low, high = self.shares_ci.get(name, (float("nan"), float("nan")))
            c_low, c_high = self.components_ci.get(name, (float("nan"), float("nan")))
            rows.append(
                {
                    "dataset": self.dataset,
                    "subset": self.subset or "",
                    "component": name,
                    "variance": self.components.get(name),
                    "variance_ci_low": c_low,
                    "variance_ci_high": c_high,
                    "share": self.shares.get(name),
                    "share_ci_low": low,
                    "share_ci_high": high,
                    "mean_square": self.mean_squares.get(name),
                    "clamped_fraction": self.clamping.get("fraction", {}).get(name),
                    "n_items": self.n_items,
                    "n_models": len(self.models),
                    "n_configs": len(self.configs),
                    "noise_floor": self.noise_floor.get("noise_floor"),
                    "noise_floor_source": self.noise_floor.get("noise_floor_used"),
                    "censoring_rate": self.censoring.get("censoring_rate"),
                    "saturated_cell_rate": self.saturated_cell_rate,
                    "n_bootstrap": self.n_bootstrap,
                }
            )
        return rows

    def prediction_p1(self) -> Dict[str, Any]:
        """Pre-registered prediction P1, evaluated (METHOD_SPEC section 10)."""
        # Explicit None checks: a share clamped to exactly 0.0 is a legitimate value
        # and `x or nan` would silently turn it into a missing one.
        raw_ic = self.shares.get("item_config")
        raw_im = self.shares.get("item_model")
        item_config = float(raw_ic) if raw_ic is not None else float("nan")
        item_model = float(raw_im) if raw_im is not None else float("nan")
        low, _high = self.shares_ci.get("item_config", (float("nan"), float("nan")))
        return {
            "prediction": "P1",
            "statement": (
                "item x configuration share exceeds 10% of noise-corrected variance "
                "and exceeds the item x model share"
            ),
            "item_config_share": item_config,
            "item_config_ci_low": low,
            "item_model_share": item_model,
            "exceeds_10pct": bool(item_config > 0.10),
            "exceeds_10pct_ci_excludes": bool(low > 0.10),
            "exceeds_item_model": bool(item_config > item_model),
            "supported": bool(item_config > 0.10 and item_config > item_model),
        }


def variance_decomposition(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str] = None,
    models: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    bootstrap_seed: int = 0,
    noise_floor: Optional[NoiseFloor] = None,
    clamp_negative: bool = True,
    exclude_all_saturated: bool = True,
) -> VarianceResult:
    """Fit Algorithm 1 on one benchmark and return everything F1/T3 report."""
    import numpy as np

    models = list(models) if models else corpus.models()
    configs = list(configs) if configs else [
        c for c in corpus.configs() if c in PRIMARY_CONFIG_IDS
    ] or corpus.configs()

    design = corpus.design_report(dataset, subset, models, configs, seed)
    items = corpus.crossed_items(dataset, subset, models, configs, seed)
    if len(items) < 3:
        raise ValueError(
            f"only {len(items)} item(s) are present in every (model, config) cell for "
            f"{dataset}/{subset} at seed {seed}; the crossed design is too incomplete "
            f"to fit. Design report: {design}"
        )

    k, n = corpus.counts_matrix(dataset, subset, models, configs, items, seed)
    if np.isnan(k).any():
        raise AssertionError("crossed_items returned an item with a missing cell")

    # Censoring: an item saturated in every cell contributes only floor.
    saturated = (k <= 0) | (k >= n)
    all_saturated = saturated.reshape(len(items), -1).all(axis=1)
    all_zero = (k <= 0).reshape(len(items), -1).all(axis=1)
    all_one = (k >= n).reshape(len(items), -1).all(axis=1)
    censoring = {
        "n_items_before_censoring": len(items),
        "n_items_all_saturated": int(all_saturated.sum()),
        "n_items_all_zero": int(all_zero.sum()),
        "n_items_all_one": int(all_one.sum()),
        "censoring_rate": float(all_saturated.mean()) if len(items) else 0.0,
        "excluded": bool(exclude_all_saturated),
        "note": (
            "METHOD_SPEC and the gap analysis Risk 4 exclude items saturated in every "
            "cell: they carry no within-item information and would contribute pure "
            "sampling floor to every mean square. RESEARCH_GAP 2.5 words this as "
            "p_q = 0 in all cells; we also exclude p_q = 1 in all cells, which is "
            "the same argument applied at the other boundary and matters on GSM8K."
        ),
    }
    keep = ~all_saturated if exclude_all_saturated else np.ones(len(items), dtype=bool)
    if keep.sum() < 3:
        raise ValueError(
            f"{dataset}/{subset}: only {int(keep.sum())} item(s) survive censoring "
            f"({censoring['n_items_all_saturated']} of {len(items)} are saturated in "
            "every cell). The model is too easy or too hard on this benchmark for a "
            "variance decomposition; METHOD_SPEC section 2 selects benchmarks to "
            "avoid exactly this."
        )
    kept_items = [item for item, flag in zip(items, keep) if flag]
    k, n = k[keep], n[keep]

    # Empirical logits and the per-cell analytic floor.
    z = np.log((k + 0.5) / (n - k + 0.5))
    per_cell_var = 1.0 / (k + 0.5) + 1.0 / (n - k + 0.5)

    floor = noise_floor or measure_noise_floor(corpus, dataset, subset, models)
    per_item_floor = np.array(
        [floor.per_item.get(item, floor.value) for item in kept_items], dtype=float
    )
    if not np.isfinite(per_item_floor).all():
        per_item_floor = np.where(
            np.isfinite(per_item_floor), per_item_floor, float(np.nanmean(per_cell_var))
        )

    fit = mean_squares(z)
    dims = fit["dims"]
    point_floor = float(np.mean(per_item_floor))
    components = components_from_ms(
        {name: float(fit["ms"][name]) for name in COMPONENT_NAMES},
        dims["n_items"],
        dims["n_models"],
        dims["n_configs"],
        noise_floor=point_floor,
    )
    shares, clamp_info = shares_from_components(components, clamp_negative=clamp_negative)

    # ------------------------------------------------------- item bootstrap (BCa-free)
    # Percentile intervals over refits, resampling the clustering unit: templates for
    # GSM-Symbolic (instances of one template are not independent), items elsewhere.
    units = corpus.bootstrap_units(kept_items)
    boot_shares, boot_components, n_clamped = _bootstrap_refit(
        z=z,
        per_item_floor=per_item_floor,
        units=units,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
        clamp_negative=clamp_negative,
    )
    shares_ci: Dict[str, Tuple[float, float]] = {}
    components_ci: Dict[str, Tuple[float, float]] = {}
    for name in COMPONENT_NAMES:
        shares_ci[name] = _percentile_interval(boot_shares.get(name), alpha)
        components_ci[name] = _percentile_interval(boot_components.get(name), alpha)

    clamping = {
        "clamp_negative": clamp_negative,
        "point_clamped": {
            name: bool(np.asarray(clamp_info["clamped"][name]).any())
            for name in COMPONENT_NAMES
        },
        "fraction": n_clamped,
        "note": (
            "method-of-moments components can go negative when the true component is "
            "near zero; clamping at 0 is standard and the clamped fraction is "
            "reported because a heavily clamped fit means the design is underpowered"
        ),
    }

    return VarianceResult(
        dataset=dataset,
        subset=subset,
        models=models,
        configs=configs,
        seed=seed,
        n_items=len(kept_items),
        components={name: float(components[name]) for name in COMPONENT_NAMES},
        shares={name: float(shares[name]) for name in COMPONENT_NAMES},
        shares_ci=shares_ci,
        components_ci=components_ci,
        mean_squares={name: float(fit["ms"][name]) for name in COMPONENT_NAMES},
        noise_floor=floor.as_dict(),
        censoring=censoring,
        clamping=clamping,
        design=design,
        n_bootstrap=int(n_bootstrap),
        alpha=alpha,
        bootstrap_seed=bootstrap_seed,
        saturated_cell_rate=float(((k <= 0) | (k >= n)).mean()),
    )


def _bootstrap_refit(
    z: Any,
    per_item_floor: Any,
    units: Sequence[str],
    n_bootstrap: int,
    seed: int,
    clamp_negative: bool,
    chunk: int = DEFAULT_BOOTSTRAP_CHUNK,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """Refit steps 1-4 on `n_bootstrap` resamples of the clustering unit.

    Resampling is done in chunks so the working array stays small: a
    `(B, I, M, C)` array at B = 10000 would be hundreds of megabytes, which on a
    Kaggle kernel is a real risk, while 250 at a time is a few megabytes and the
    numpy call overhead is amortised.

    The noise floor is recomputed on each resample from the per-item floors, so the
    uncertainty of the *denominator* propagates into the share intervals instead of
    being treated as known. That matters: the floor is estimated from a handful of
    seed replicates.
    """
    import numpy as np

    n_items = z.shape[0]
    if n_bootstrap <= 0 or n_items < 3:
        empty: Dict[str, Any] = {name: None for name in COMPONENT_NAMES}
        return empty, empty, {name: 0.0 for name in COMPONENT_NAMES}

    # Group items by clustering unit so a resample draws whole templates.
    groups: Dict[str, List[int]] = {}
    for index, unit in enumerate(units):
        groups.setdefault(unit, []).append(index)
    unit_keys = sorted(groups)
    unit_indices = [np.asarray(groups[u], dtype=int) for u in unit_keys]
    n_units = len(unit_keys)

    rng = np.random.default_rng(seed)
    share_out = {name: [] for name in COMPONENT_NAMES}
    component_out = {name: [] for name in COMPONENT_NAMES}
    clamp_counts = {name: 0 for name in COMPONENT_NAMES}
    n_done = 0
    while n_done < n_bootstrap:
        batch = min(chunk, n_bootstrap - n_done)
        rows: List[Any] = []
        floors: List[float] = []
        for _ in range(batch):
            picks = rng.integers(0, n_units, size=n_units)
            selection = np.concatenate([unit_indices[p] for p in picks])
            rows.append(selection)
            floors.append(float(per_item_floor[selection].mean()))
        # Ragged when units differ in size (GSM-Symbolic); fall back to a loop then.
        lengths = {len(r) for r in rows}
        if len(lengths) == 1:
            stacked = z[np.stack(rows)]
            fit = mean_squares(stacked)
            floor_arr = np.asarray(floors)[:, None]
            ms = {name: np.asarray(fit["ms"][name])[:, None] for name in COMPONENT_NAMES}
            dims = fit["dims"]
            components = components_from_ms(
                {name: ms[name] for name in COMPONENT_NAMES},
                dims["n_items"],
                dims["n_models"],
                dims["n_configs"],
                noise_floor=floor_arr,
            )
            shares, info = shares_from_components(components, clamp_negative=clamp_negative)
            for name in COMPONENT_NAMES:
                share_out[name].extend(np.ravel(shares[name]).tolist())
                component_out[name].extend(np.ravel(components[name]).tolist())
                clamp_counts[name] += int(np.asarray(info["clamped"][name]).sum())
        else:
            for selection, floor_value in zip(rows, floors):
                fit = mean_squares(z[selection])
                dims = fit["dims"]
                components = components_from_ms(
                    {name: float(fit["ms"][name]) for name in COMPONENT_NAMES},
                    dims["n_items"],
                    dims["n_models"],
                    dims["n_configs"],
                    noise_floor=floor_value,
                )
                shares, info = shares_from_components(
                    components, clamp_negative=clamp_negative
                )
                for name in COMPONENT_NAMES:
                    share_out[name].append(float(shares[name]))
                    component_out[name].append(float(components[name]))
                    clamp_counts[name] += int(bool(np.asarray(info["clamped"][name]).any()))
        n_done += batch

    fractions = {
        name: clamp_counts[name] / max(1, n_bootstrap) for name in COMPONENT_NAMES
    }
    return (
        {name: np.asarray(share_out[name], dtype=float) for name in COMPONENT_NAMES},
        {name: np.asarray(component_out[name], dtype=float) for name in COMPONENT_NAMES},
        fractions,
    )


def _percentile_interval(values: Any, alpha: float) -> Tuple[float, float]:
    import numpy as np

    if values is None:
        return (float("nan"), float("nan"))
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def decompose_all(
    corpus: Corpus,
    models: Optional[Sequence[str]] = None,
    configs: Optional[Sequence[str]] = None,
    seed: int = 0,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    **kw: Any,
) -> List[VarianceResult]:
    """Fit every benchmark present, skipping the ones the design cannot support."""
    out: List[VarianceResult] = []
    for dataset, subset in corpus.benchmarks():
        try:
            out.append(
                variance_decomposition(
                    corpus,
                    dataset,
                    subset,
                    models=models,
                    configs=configs,
                    seed=seed,
                    n_bootstrap=n_bootstrap,
                    **kw,
                )
            )
        except (ValueError, AssertionError) as exc:
            log.warning("skipping variance decomposition for %s/%s: %s", dataset, subset, exc)
    return out
