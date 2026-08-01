"""Every figure and table the paper needs, from one command (METHOD_SPEC 9).

The filenames here are load-bearing. `paper/main.tex` cites them by name, so a
typo does not raise -- it silently produces a paper with a missing float. They are
therefore declared once, as constants, in `FIGURE_FILENAMES` and `TABLE_FILENAMES`,
and `test_paper.py` asserts the exact set.

Three design points.

**Analysis is separate from plotting.** `analyse()` computes every statistic once
and returns a bundle; the figure and table functions only format it. So a plotting
change never re-runs a ten-minute bootstrap, and the numbers behind a figure are
always the numbers in its companion CSV.

**A missing input skips one artefact, never the run.** Tier A alone cannot produce
the distribution-shift figure or the precision control, and a session that dies
halfway leaves a partial corpus. Each artefact records why it was skipped in
`artefacts.json` instead of taking the whole build down with it -- the alternative
is discovering at camera-ready that one absent benchmark cost you all thirteen
floats.

**Every pre-registered prediction is evaluated and written down, including when it
fails.** `predictions.csv` holds P1-P7 with the numbers behind each verdict. The
falsification hooks in METHOD_SPEC 5.6 and 10 are only meaningful if the refuting
value is as easy to read as the confirming one.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import DEFAULT_ALPHA, DEFAULT_N_BOOTSTRAP
from ..utils import setup_logging, utc_now_iso, write_json_atomic
from . import cdv as cdv_mod
from . import modes as modes_mod
from . import transfer as transfer_mod
from . import variance as variance_mod
from .confounds import (
    extractor_agreement_table,
    per_config_failure_rates,
    sensitivity_glmm_shares,
)
from .corpus import Corpus, corpus_summary, load_corpus
from .glmm import GLMMResult, glmm_decomposition, parametric_bootstrap_null

log = logging.getLogger(__name__)

#: METHOD_SPEC 9 figure names. Byte-exact; `paper/main.tex` cites these strings.
FIGURE_FILENAMES: Dict[str, str] = {
    "F1": "fig1_variance_components.pdf",
    "F2": "fig2_transfer_scatter.pdf",
    "F3": "fig3_hard_subset_overlap.pdf",
    "F4": "fig4_ceiling_vs_coverage.pdf",
    "F5": "fig5_cdv_vs_sc.pdf",
    "F6": "fig6_reordering.pdf",
    "F7": "fig7_shift.pdf",
    "F8": "fig8_precision_control.pdf",
}

#: METHOD_SPEC 9 table names. Byte-exact.
TABLE_FILENAMES: Dict[str, str] = {
    "T1": "tab1_setup.csv",
    "T2": "tab2_main_accuracy.csv",
    "T3": "tab3_variance_components.csv",
    "T4": "tab4_transfer.csv",
    "T5": "tab5_method_comparison.csv",
    "T6": "tab6_downstream.csv",
    "T7": "tab7_precision_control.csv",
}

#: Emitted alongside the numbered artefacts; not cited by main.tex.
EXTRA_FILENAMES: Dict[str, str] = {
    "predictions": "predictions.csv",
    "artefacts": "artefacts.json",
    "corpus_health": "corpus_health.json",
}


# ------------------------------------------------------------------------ config
@dataclass
class PaperConfig:
    """Knobs for the whole build. Defaults are the spec's values."""

    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    alpha: float = DEFAULT_ALPHA
    #: Sampling seed whose cells form the primary crossed design.
    seed: int = 0
    bootstrap_seed: int = 0
    configs: Optional[Sequence[str]] = None
    cdv_configs: Optional[Sequence[str]] = None
    budget_multiples: Sequence[int] = cdv_mod.DEFAULT_BUDGET_MULTIPLES
    table_multiples: Sequence[int] = cdv_mod.DEFAULT_TABLE_MULTIPLES
    n_repeats: int = cdv_mod.DEFAULT_N_REPEATS
    hard_quantile: float = transfer_mod.HARD_QUANTILE
    reference_config: str = REFERENCE_CONFIG_ID
    #: The B9 target configuration: difficulty is estimated in `reference_config`
    #: and spent here.
    allocation_target: str = "c1"
    also_png: bool = True

    def resolved_configs(self, corpus: Corpus) -> List[str]:
        if self.configs:
            return [c for c in self.configs if c in set(corpus.configs(True))]
        present = corpus.configs()
        primary = [c for c in PRIMARY_CONFIG_IDS if c in present]
        return primary or present


@dataclass
class Analysis:
    """Every statistic the artefacts need, computed once."""

    corpus: Corpus
    config: PaperConfig
    configs: List[str] = field(default_factory=list)
    health: Dict[str, Any] = field(default_factory=dict)
    variance: List[variance_mod.VarianceResult] = field(default_factory=list)
    glmm: List[GLMMResult] = field(default_factory=list)
    null_calibration: List[Dict[str, Any]] = field(default_factory=list)
    plug_in_plateau: List[Dict[str, Any]] = field(default_factory=list)
    extractor_agreement: List[Dict[str, Any]] = field(default_factory=list)
    per_config_failures: List[Dict[str, Any]] = field(default_factory=list)
    glmm_sensitivity: List[Dict[str, Any]] = field(default_factory=list)
    cdv_decomposition: List[Dict[str, Any]] = field(default_factory=list)
    noise_floors: Dict[Tuple[str, Optional[str]], variance_mod.NoiseFloor] = field(
        default_factory=dict
    )
    reliabilities: Dict[Any, transfer_mod.Reliability] = field(default_factory=dict)
    transfer: List[transfer_mod.TransferRow] = field(default_factory=list)
    overlaps: List[transfer_mod.OverlapResult] = field(default_factory=list)
    ceilings: List[modes_mod.CeilingRow] = field(default_factory=list)
    unions: List[Dict[str, Any]] = field(default_factory=list)
    transitions: List[modes_mod.TransitionRow] = field(default_factory=list)
    transition_items: Dict[Any, List[Dict[str, Any]]] = field(default_factory=dict)
    cdv_points: List[cdv_mod.MethodPoint] = field(default_factory=list)
    per_config_points: List[cdv_mod.MethodPoint] = field(default_factory=list)
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    greedy: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def predictions(self) -> List[Dict[str, Any]]:
        """P1-P6, each with the numbers that decided it."""
        rows: List[Dict[str, Any]] = []
        for result in self.glmm or self.variance:
            if isinstance(result, GLMMResult):
                row = result.prediction_p1()
            else:
                row = result.prediction_p1()
            row.update({"dataset": result.dataset, "subset": result.subset or ""})
            rows.append(row)
        for row in self.plug_in_plateau:
            rows.append(row)
        if self.transfer:
            rows.append(transfer_mod.transfer_summary(self.transfer, kind="config"))
        if self.overlaps:
            rows.append(transfer_mod.overlap_summary(self.overlaps))
        rows.extend(modes_mod.ceiling_spread(self.ceilings))
        if self.cdv_points:
            rows.append(cdv_mod.cdv_summary(self.cdv_points, self.comparisons))
        if self.transitions:
            summary = modes_mod.transition_summary(self.transitions)
            summary["prediction"] = "P5-mechanism"
            rows.append(summary)
        for allocation in self.allocations:
            if "error" not in allocation:
                rows.append(allocation)
        return rows


# ---------------------------------------------------------------------- analysis
def analyse(corpus: Corpus, cfg: Optional[PaperConfig] = None) -> Analysis:
    """Run the whole statistical pipeline over a loaded corpus. CPU only."""
    cfg = cfg or PaperConfig()
    out = Analysis(corpus=corpus, config=cfg)
    out.configs = cfg.resolved_configs(corpus)
    out.health = corpus_summary(corpus)
    out.warnings.extend(corpus.warnings)
    if not out.configs:
        out.warnings.append("no elicitation configurations found in the corpus")
        return out
    log.info(
        "analysing %d cells over configs %s at seed %d",
        len(corpus), out.configs, cfg.seed,
    )

    # --- variance: GLMM primary (5.2), moment decomposition robustness
    for dataset, subset in corpus.benchmarks():
        floor = variance_mod.measure_noise_floor(
            corpus, dataset, subset, config=cfg.reference_config
        )
        out.noise_floors[(dataset, subset)] = floor
        try:
            out.glmm.append(
                glmm_decomposition(
                    corpus, dataset, subset,
                    configs=out.configs, seed=cfg.seed,
                    n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha,
                    bootstrap_seed=cfg.bootstrap_seed, compare_moments=True,
                )
            )
        except (ValueError, ImportError, AssertionError) as exc:
            out.warnings.append(f"GLMM decomposition skipped for {dataset}/{subset}: {exc}")
        try:
            out.variance.append(
                variance_mod.variance_decomposition(
                    corpus, dataset, subset,
                    configs=out.configs, seed=cfg.seed,
                    n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha,
                    bootstrap_seed=cfg.bootstrap_seed, noise_floor=floor,
                )
            )
        except (ValueError, AssertionError) as exc:
            out.warnings.append(f"moment decomposition skipped for {dataset}/{subset}: {exc}")
        try:
            out.null_calibration.append(
                parametric_bootstrap_null(
                    corpus, dataset, subset,
                    configs=out.configs, seed=cfg.seed,
                    n_draws=min(50, max(20, cfg.n_bootstrap // 20)),
                    draw_seed=cfg.bootstrap_seed,
                )
            )
        except (ValueError, ImportError, AssertionError) as exc:
            out.warnings.append(f"parametric null skipped for {dataset}/{subset}: {exc}")
        try:
            out.glmm_sensitivity.append(
                sensitivity_glmm_shares(corpus, dataset, subset, seed=cfg.seed)
            )
        except (ValueError, ImportError, AssertionError) as exc:
            out.warnings.append(f"GLMM sensitivity skipped for {dataset}/{subset}: {exc}")

    out.extractor_agreement = [r.as_dict() for r in extractor_agreement_table(corpus)]
    out.per_config_failures = per_config_failure_rates(corpus)

    # --- transfer and overlap (5.3, 5.4)
    out.reliabilities = transfer_mod.all_reliabilities(
        corpus, cfg.reference_config,
        n_bootstrap=min(cfg.n_bootstrap, 2000), alpha=cfg.alpha,
    )
    out.transfer = transfer_mod.transfer_table(
        corpus, configs=out.configs, seed=cfg.seed,
        reference_config=cfg.reference_config, n_bootstrap=cfg.n_bootstrap,
        alpha=cfg.alpha, bootstrap_seed=cfg.bootstrap_seed,
        reliabilities=out.reliabilities,
    )
    out.overlaps = transfer_mod.overlap_table(
        corpus, configs=out.configs, seed=cfg.seed,
        n_bootstrap=cfg.n_bootstrap, n_permutations=cfg.n_bootstrap,
        quantile=cfg.hard_quantile, alpha=cfg.alpha,
        bootstrap_seed=cfg.bootstrap_seed, reference_config=cfg.reference_config,
    )

    # --- modal ceilings and transitions (5.5)
    out.ceilings = modes_mod.ceiling_rows(
        corpus, configs=None, seed=cfg.seed,
        alpha=cfg.alpha, bootstrap_seed=cfg.bootstrap_seed,
    )
    out.transitions = modes_mod.transition_rows(
        corpus, configs=out.configs, seed=cfg.seed,
        alpha=cfg.alpha, bootstrap_seed=cfg.bootstrap_seed,
    )
    out.transition_items = modes_mod.transition_records(
        corpus, configs=out.configs, seed=cfg.seed
    )
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            union = modes_mod.union_ceiling(
                corpus, dataset, subset, model, out.configs, seed=cfg.seed
            )
            if union.get("n_items"):
                out.unions.append(union)

    # --- the intervention (5.6) and baselines (6)
    out.cdv_points, cdv_warnings = cdv_mod.budget_curves(
        corpus, seed=cfg.seed, cdv_configs=cfg.cdv_configs,
        multiples=cfg.budget_multiples, n_repeats=cfg.n_repeats,
    )
    out.warnings.extend(cdv_warnings)
    out.comparisons = cdv_mod.compare_to_sc(
        out.cdv_points, n_bootstrap=cfg.n_bootstrap, alpha=cfg.alpha,
        bootstrap_seed=cfg.bootstrap_seed,
    )
    out.plug_in_plateau = cdv_mod.plug_in_plateau_summary(
        corpus, configs=cfg.cdv_configs, seed=cfg.seed
    )
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            out.cdv_decomposition.append(
                cdv_mod.cdv_gain_decomposition(
                    corpus, dataset, subset, model,
                    configs=cfg.cdv_configs or None, seed=cfg.seed,
                )
            )
    out.per_config_points = _per_config_curves(corpus, out.configs, cfg)
    out.greedy = cdv_mod.greedy_accuracy(corpus)

    # --- the applied consequence (B9)
    target = cfg.allocation_target if cfg.allocation_target in out.configs else None
    if target and target != cfg.reference_config:
        for dataset, subset in corpus.benchmarks():
            for model in corpus.models():
                for rule in ("uncertainty", "difficulty"):
                    out.allocations.append(
                        cdv_mod.transferred_difficulty_allocation(
                            corpus, dataset, subset, model,
                            source_config=cfg.reference_config, target_config=target,
                            seed=cfg.seed, rule=rule, n_repeats=cfg.n_repeats,
                        )
                    )
    else:
        out.warnings.append(
            "no B9 allocation comparison: it needs the reference configuration plus a "
            f"distinct target ({cfg.allocation_target!r}) in the corpus"
        )
    return out


def _per_config_curves(
    corpus: Corpus, configs: Sequence[str], cfg: PaperConfig
) -> List[cdv_mod.MethodPoint]:
    """`maj@n` versus tokens for each configuration separately, for Figure F4."""
    points: List[cdv_mod.MethodPoint] = []
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            pools = cdv_mod.build_item_pools(
                corpus, dataset, subset, model, seed=cfg.seed, configs=list(configs)
            )
            if not pools:
                continue
            methods = [cdv_mod.single_config_method(c) for c in configs
                       if any(c in p.classes for p in pools)]
            if not methods:
                continue
            grid = cdv_mod.token_budgets(
                pools, methods, cfg.budget_multiples, scale_method=methods[0].name
            )
            for multiple, budget in zip(grid["multiples"], grid["budgets"]):
                for method in methods:
                    points.append(
                        cdv_mod.evaluate_method(
                            method, pools, budget, float(multiple),
                            n_repeats=cfg.n_repeats, seed=cfg.bootstrap_seed,
                        )
                    )
    return points


# ------------------------------------------------------------------- csv writing
def write_csv(
    rows: Sequence[Dict[str, Any]], path: str | Path, columns: Optional[Sequence[str]] = None
) -> Path:
    """Write dict rows to CSV with a stable column order.

    Column order is first-seen across all rows rather than sorted, so a reader sees
    identifiers before statistics. Missing keys become empty cells instead of
    raising: a partial corpus should still produce a readable table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    log.info("wrote %s (%d row(s))", path, len(rows))
    return path


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


# ----------------------------------------------------------------------- figures
def _plt():
    from .figures import plt, set_style

    set_style()
    return plt


def _save(fig: Any, path: Path, also_png: bool = True) -> Path:
    from .figures import _save as save_figure

    return save_figure(fig, path, also_png=also_png)


def _grid(n_panels: int, width_per: float = 2.4, height: float = 2.5, max_cols: int = 3):
    plt = _plt()
    cols = min(max_cols, max(1, n_panels))
    rows = max(1, math.ceil(n_panels / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(width_per * cols + 0.6, height * rows), squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat[:n_panels]


def fig1_variance_components(analysis: Analysis, path: Path) -> Optional[Path]:
    """Stacked share of variance per benchmark. GLMM primary; moment as fallback."""
    from .figures import PALETTE

    results = analysis.glmm or analysis.variance
    if not results:
        return None
    plt = _plt()
    names = variance_mod.COMPONENT_NAMES
    labels = [f"{r.dataset}{'/' + r.subset if r.subset else ''}" for r in results]
    fig, ax = plt.subplots(figsize=(max(3.4, 1.1 * len(results) + 1.8), 3.0))
    bottoms = [0.0] * len(results)
    for index, name in enumerate(names):
        values = [max(0.0, float(r.shares.get(name) or 0.0)) for r in results]
        ax.bar(
            labels, values, bottom=bottoms,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white", linewidth=0.4,
            label=name.replace("_", r"$\times$"),
        )
        if name == "item_config":
            # CIs on the interaction segment only: it is the claim under test, and
            # error bars on all seven segments would be unreadable.
            for x, (value, bottom, result) in enumerate(zip(values, bottoms, results)):
                low, high = result.shares_ci.get(name, (float("nan"), float("nan")))
                if math.isfinite(low) and math.isfinite(high):
                    ax.errorbar(
                        x, bottom + value,
                        yerr=[[max(0.0, value - (low - 0.0))], [max(0.0, high - value)]],
                        fmt="none", ecolor="black", elinewidth=0.8, capsize=2,
                    )
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("variance component share (GLMM primary)")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)
    floors = {}
    for r in analysis.glmm or analysis.variance:
        if isinstance(r, GLMMResult):
            floors[r.dataset] = (r.noise_floor or {}).get("noise_floor")
        else:
            floors[r.dataset] = r.noise_floor.get("noise_floor")
    ax.set_title(
        "noise floor "
        + ", ".join(f"{k}: {v:.2f}" for k, v in floors.items() if v is not None),
        fontsize=7,
    )
    return _save(fig, path, analysis.config.also_png)


def fig2_transfer_scatter(analysis: Analysis, path: Path) -> Optional[Path]:
    """`p_hat` under the reference configuration against each other configuration."""
    from .figures import PALETTE, _short

    corpus = analysis.corpus
    cfg = analysis.config
    reference = cfg.reference_config
    others = [c for c in analysis.configs if c != reference]
    if not others:
        return None
    benchmarks = corpus.benchmarks()
    if not benchmarks:
        return None
    dataset, subset = benchmarks[0]
    models = corpus.models()
    seeds = sorted(
        {c.seed for c in corpus.cells.values() if c.config == reference and c.n > 0}
    )

    n_panels = len(others) + (1 if len(seeds) >= 2 else 0)
    fig, axes = _grid(n_panels)
    by_pair = {
        (r.label_a, r.label_b, r.model): r for r in analysis.transfer if r.kind == "config"
    }
    for index, config in enumerate(others):
        ax = axes[index]
        for m_index, model in enumerate(models):
            items, vectors = transfer_mod._aligned(
                corpus, dataset, subset,
                [(model, reference, cfg.seed), (model, config, cfg.seed)],
            )
            if not items:
                continue
            ax.scatter(
                vectors[0], vectors[1], s=5, alpha=0.45,
                color=PALETTE[m_index % len(PALETTE)],
                edgecolors="none", label=_short(model, 14),
            )
        ax.plot([0, 1], [0, 1], color="0.4", linewidth=0.6, linestyle="--")
        row = next(
            (v for k, v in by_pair.items() if k[0] == reference and k[1] == config), None
        )
        annotation = (
            rf"$\rho_{{raw}}$={row.rho_raw:.2f}" + "\n" + rf"$\rho_{{dis}}$={row.rho_disatt:.2f}"
            if row
            else ""
        )
        ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va="top", fontsize=7)
        ax.set_title(f"{reference} vs {config}", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    if len(seeds) >= 2:
        ax = axes[-1]
        for m_index, model in enumerate(models):
            items, vectors = transfer_mod._aligned(
                corpus, dataset, subset,
                [(model, reference, seeds[0]), (model, reference, seeds[1])],
            )
            if not items:
                continue
            ax.scatter(
                vectors[0], vectors[1], s=5, alpha=0.45,
                color=PALETTE[m_index % len(PALETTE)], edgecolors="none",
            )
        ax.plot([0, 1], [0, 1], color="0.4", linewidth=0.6, linestyle="--")
        r_mm = [
            v.r_mm for k, v in analysis.reliabilities.items() if k[0] == dataset
        ]
        ax.text(
            0.04, 0.96,
            rf"$r_{{mm}}$={sum(r_mm) / len(r_mm):.2f}" if r_mm else "",
            transform=ax.transAxes, va="top", fontsize=7,
        )
        ax.set_title(f"seed {seeds[0]} vs {seeds[1]} (noise ceiling)", fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel(r"$\hat{p}$ (other)")
    for ax in axes:
        ax.set_xlabel(rf"$\hat{{p}}$ ({reference})")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{dataset}{'/' + subset if subset else ''}", fontsize=8)
    return _save(fig, path, analysis.config.also_png)


def fig3_hard_subset_overlap(analysis: Analysis, path: Path) -> Optional[Path]:
    """`J_config` against the seed-pair null, per model. Shows the excess."""
    from .figures import PALETTE, _short

    results = analysis.overlaps
    if not results:
        return None
    plt = _plt()
    labels = [
        f"{_short(r.model, 12)}\n{r.dataset}{'/' + r.subset if r.subset else ''}"
        for r in results
    ]
    x = list(range(len(results)))
    fig, ax = plt.subplots(figsize=(max(3.4, 1.2 * len(results) + 1.0), 2.8))
    width = 0.38
    for offset, (key, colour, label) in enumerate(
        (("j_seed", PALETTE[2], r"$J_{seed}$ (null)"), ("j_config", PALETTE[1], r"$J_{config}$"))
    ):
        values = [getattr(r, key) for r in results]
        errors = [
            [
                max(0.0, getattr(r, key) - getattr(r, f"{key}_ci")[0])
                for r in results
            ],
            [
                max(0.0, getattr(r, f"{key}_ci")[1] - getattr(r, key))
                for r in results
            ],
        ]
        ax.bar(
            [v + (offset - 0.5) * width for v in x], values, width=width,
            color=colour, label=label, edgecolor="white", linewidth=0.4,
            yerr=errors, error_kw={"elinewidth": 0.7, "capsize": 2},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("hard-subset Jaccard")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title(
        f"bottom-{int(100 * analysis.config.hard_quantile)}% subset; the gap is the "
        "excess instability",
        fontsize=7,
    )
    return _save(fig, path, analysis.config.also_png)


def fig4_ceiling_vs_coverage(analysis: Analysis, path: Path) -> Optional[Path]:
    """`maj@n` and `pass@n` against output tokens, with `pi_mode` as a plateau."""
    from .figures import PALETTE, _short, _style_for

    points = analysis.per_config_points
    if not points:
        return None
    models = sorted({p.model for p in points})
    fig, axes = _grid(len(models), width_per=2.6, height=2.6)
    ceilings = {(c.model, c.config): c for c in analysis.ceilings}
    for index, model in enumerate(models):
        ax = axes[index]
        configs = sorted({p.method.replace("sc_", "") for p in points if p.model == model})
        for c_index, config in enumerate(configs):
            series = sorted(
                (p for p in points if p.model == model and p.method == f"sc_{config}"),
                key=lambda p: p.budget_tokens,
            )
            if not series:
                continue
            ax.plot(
                [p.mean_tokens_used for p in series],
                [p.accuracy for p in series],
                label=config, **_style_for(c_index), markersize=3,
            )
            ceiling = ceilings.get((model, config))
            if ceiling is not None and math.isfinite(ceiling.pi_mode):
                ax.axhline(
                    ceiling.pi_mode, color=PALETTE[c_index % len(PALETTE)],
                    linewidth=0.5, linestyle=":", alpha=0.8,
                )
        reference = ceilings.get((model, analysis.config.reference_config))
        if reference is not None and math.isfinite(reference.pass_at_n):
            ax.axhline(
                reference.pass_at_n, color="0.2", linewidth=0.8, linestyle="--",
                label=r"pass@$N$",
            )
        ax.set_xscale("log")
        ax.set_title(_short(model, 16), fontsize=8)
        ax.set_xlabel("completion tokens / item")
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
               fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "dotted horizontals: $\\pi_{mode}$ per configuration (the SC plateau)",
        fontsize=7,
    )
    return _save(fig, path, analysis.config.also_png)


def fig5_cdv_vs_sc(analysis: Analysis, path: Path) -> Optional[Path]:
    """Accuracy against completion tokens for every method. One panel per model."""
    from .figures import _short, _style_for

    points = analysis.cdv_points
    if not points:
        return None
    models = sorted({p.model for p in points})
    fig, axes = _grid(len(models), width_per=2.6, height=2.6)
    order = ["sc", "temp_sc", "para_sc", "cdv", "adaptive_cdv", "random_config",
             "oracle_config", "oracle_coverage", "certainty_vote", "esc"]
    for index, model in enumerate(models):
        ax = axes[index]
        present = [m for m in order if any(p.method == m and p.model == model for p in points)]
        for m_index, method in enumerate(present):
            series = sorted(
                (p for p in points if p.model == model and p.method == method),
                key=lambda p: p.budget_tokens,
            )
            oracle = method.startswith("oracle")
            ax.plot(
                [p.mean_tokens_used for p in series],
                [p.accuracy for p in series],
                label=method, **_style_for(m_index), markersize=3,
                alpha=0.55 if oracle else 1.0,
                linewidth=0.8 if oracle else 1.2,
            )
        ax.set_xscale("log")
        ax.set_title(_short(model, 16), fontsize=8)
        ax.set_xlabel("completion tokens / item")
    axes[0].set_ylabel("accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               fontsize=7, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle(
        "matched completion-token budget; the test is whether CDV plateaus above SC",
        fontsize=7,
    )
    return _save(fig, path, analysis.config.also_png)


def fig6_reordering(analysis: Analysis, path: Path) -> Optional[Path]:
    """The mechanism: transition taxonomy per model, and reordering versus margin."""
    from .figures import PALETTE, _short

    rows = analysis.transitions
    if not rows:
        return None
    plt = _plt()
    models = sorted({r.model for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    width = 0.6
    bottoms = [0.0] * len(models)
    for index, kind in enumerate(modes_mod.TRANSITION_KINDS):
        values = []
        for model in models:
            subset = [r for r in rows if r.model == model and r.n_items]
            values.append(
                sum(getattr(r, f"n_{kind}") / r.n_items for r in subset) / len(subset)
                if subset
                else 0.0
            )
        ax.bar(
            [_short(m, 12) for m in models], values, bottom=bottoms, width=width,
            color=PALETTE[index % len(PALETTE)], edgecolor="white", linewidth=0.4,
            label=kind,
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_ylabel("fraction of items")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.set_title("mode transitions across configuration pairs", fontsize=8)
    ax.tick_params(axis="x", labelrotation=20)

    ax = axes[1]
    pooled: List[Dict[str, Any]] = [
        record for records in analysis.transition_items.values() for record in records
    ]
    bins = modes_mod.reorder_rate_by_margin_bin(pooled)
    centres = [(b["margin_low"] + b["margin_high"]) / 2 for b in bins]
    ax.bar(
        centres, [b["reorder_rate"] for b in bins],
        width=[b["margin_high"] - b["margin_low"] for b in bins],
        color=PALETTE[0], edgecolor="white", linewidth=0.5, align="center",
    )
    for centre, b in zip(centres, bins):
        if b["n_items"]:
            ax.text(centre, 0.02, f"n={b['n_items']}", ha="center", fontsize=6, rotation=90)
    ax.set_xlabel("top-1 minus top-2 frequency (source configuration)")
    ax.set_ylabel("reorder rate")
    ax.set_title("reorderings concentrate at small margin", fontsize=8)
    return _save(fig, path, analysis.config.also_png)


def fig7_shift(analysis: Analysis, path: Path) -> Optional[Path]:
    """Does symbolic perturbation amplify the item x configuration interaction?"""
    from .figures import PALETTE

    results = analysis.glmm or analysis.variance
    if len(results) < 2:
        return None
    plt = _plt()
    labels = [f"{r.dataset}{'/' + r.subset if r.subset else ''}" for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax = axes[0]
    shares = [float(r.shares.get("item_config") or 0.0) for r in results]
    errors = [
        [max(0.0, s - r.shares_ci["item_config"][0]) for s, r in zip(shares, results)],
        [max(0.0, r.shares_ci["item_config"][1] - s) for s, r in zip(shares, results)],
    ]
    ax.bar(labels, shares, color=PALETTE[1], edgecolor="white", linewidth=0.4,
           yerr=errors, error_kw={"elinewidth": 0.7, "capsize": 2})
    ax.set_ylabel(r"item$\times$config share")
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("interaction share by benchmark", fontsize=8)

    ax = axes[1]
    by_benchmark: Dict[str, List[float]] = {}
    for row in analysis.transfer:
        if row.kind != "config" or not math.isfinite(row.rho_disatt):
            continue
        key = f"{row.dataset}{'/' + row.subset if row.subset else ''}"
        by_benchmark.setdefault(key, []).append(row.rho_disatt)
    keys = [k for k in labels if k in by_benchmark]
    ax.bar(
        keys, [sum(by_benchmark[k]) / len(by_benchmark[k]) for k in keys],
        color=PALETTE[0], edgecolor="white", linewidth=0.4,
    )
    ax.axhline(1.0, color="0.3", linewidth=0.6, linestyle="--")
    ax.set_ylabel(r"mean $\rho_{disatt}$")
    ax.tick_params(axis="x", labelrotation=20)
    ax.set_title("difficulty transfer by benchmark", fontsize=8)
    return _save(fig, path, analysis.config.also_png)


def fig8_precision_control(
    fp16: Analysis, quantised: Analysis, path: Path
) -> Optional[Path]:
    """Paired fp16 versus 4-bit panels. The margin panel is the decisive one."""
    from .figures import PALETTE

    if not fp16.variance or not quantised.variance:
        return None
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    ax = axes[0]
    names = variance_mod.COMPONENT_NAMES
    x = list(range(len(names)))
    for offset, (analysis, label, colour) in enumerate(
        ((fp16, "fp16", PALETTE[0]), (quantised, "4-bit", PALETTE[1]))
    ):
        values = [
            sum(float(r.shares.get(n) or 0.0) for r in analysis.variance) / len(analysis.variance)
            for n in names
        ]
        ax.bar([v + (offset - 0.5) * 0.4 for v in x], values, width=0.4,
               color=colour, label=label, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "x") for n in names], rotation=60, fontsize=6)
    ax.set_ylabel("variance share")
    ax.legend(fontsize=7)
    ax.set_title("components", fontsize=8)

    ax = axes[1]
    for offset, (analysis, label, colour) in enumerate(
        ((fp16, "fp16", PALETTE[0]), (quantised, "4-bit", PALETTE[1]))
    ):
        values = [r.rho_disatt for r in analysis.transfer
                  if r.kind == "config" and math.isfinite(r.rho_disatt)]
        if values:
            ax.bar([offset], [sum(values) / len(values)], width=0.5, color=colour,
                   label=label, edgecolor="white", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["fp16", "4-bit"], fontsize=7)
    ax.set_ylabel(r"mean $\rho_{disatt}$")
    ax.set_title("difficulty transfer", fontsize=8)

    ax = axes[2]
    for analysis, label, colour in (
        (fp16, "fp16", PALETTE[0]), (quantised, "4-bit", PALETTE[1])
    ):
        margins = [
            cell.margin() for cell in analysis.corpus.all_cells() if cell.n > 0
        ]
        if margins:
            ax.hist(margins, bins=20, range=(0, 1), histtype="step", linewidth=1.0,
                    color=colour, label=label, density=True)
    ax.set_xlabel("top-two margin")
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    ax.set_title("margin distribution (the direct test)", fontsize=8)
    return _save(fig, path, fp16.config.also_png)


# ------------------------------------------------------------------------ tables
def tab1_setup(analysis: Analysis) -> List[Dict[str, Any]]:
    """Reproducibility table: what ran, on what, with which backend branch."""
    rows: List[Dict[str, Any]] = []
    for run_dir, manifest in sorted(analysis.corpus.manifests.items()):
        cfg = manifest.get("config") or {}
        hardware = manifest.get("hardware") or {}
        rows.append(
            {
                "run_dir": Path(run_dir).name,
                "model": (cfg.get("model") or {}).get("name"),
                "dtype": (cfg.get("model") or {}).get("dtype"),
                "quantization": (cfg.get("model") or {}).get("quantization"),
                "backend": (cfg.get("model") or {}).get("backend"),
                "backend_version": hardware.get("backend_version")
                or manifest.get("backend_version"),
                "attention_backend": hardware.get("attention_backend"),
                "parallelism": (cfg.get("model") or {}).get("parallelism")
                or hardware.get("parallelism"),
                "tensor_parallel_size": (cfg.get("model") or {}).get("tensor_parallel_size"),
                "dataset": (cfg.get("data") or {}).get("name"),
                "subset": (cfg.get("data") or {}).get("subset"),
                "split": (cfg.get("data") or {}).get("split"),
                "n_items": (cfg.get("data") or {}).get("limit"),
                "tier": (cfg.get("data") or {}).get("tier"),
                "elicitation": (cfg.get("elicitation") or {}).get("id"),
                "strategy": (cfg.get("strategy") or {}).get("name"),
                "n_samples": (cfg.get("strategy") or {}).get("n_samples"),
                "seed": (cfg.get("runtime") or {}).get("seed"),
                "config_hash": manifest.get("config_hash"),
                "gpu_name": hardware.get("gpu_name"),
                "n_gpus": hardware.get("n_gpus"),
                "elapsed_seconds": manifest.get("elapsed_seconds"),
                "tokens_completion": manifest.get("tokens_completion"),
                "grader_version": manifest.get("grader_version"),
                "harness_commit": manifest.get("git_commit"),
            }
        )
    health = dict(analysis.health)
    health.pop("warnings", None)
    rows.append({"run_dir": "TOTAL", **health})
    for (dataset, subset), floor in sorted(
        analysis.noise_floors.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append(
            {
                "run_dir": f"NOISE_FLOOR:{dataset}{'/' + subset if subset else ''}",
                **floor.as_dict(),
            }
        )
    return rows


def tab2_main_accuracy(analysis: Analysis) -> List[Dict[str, Any]]:
    """Per (model, dataset, config): avg@1, greedy, maj@N, `pi_mode`, pass@N."""
    greedy = {
        (g["dataset"], g["subset"], g["model"], g["config"]): g["greedy_accuracy"]
        for g in analysis.greedy
    }
    rows: List[Dict[str, Any]] = []
    for row in analysis.ceilings:
        record = row.as_dict()
        record["greedy_accuracy"] = greedy.get(
            (row.dataset, row.subset or "", row.model, row.config)
        )
        # maj@N over all N samples is the modal class by definition, so it equals
        # pi_mode exactly. Reported under both names because METHOD_SPEC's T2 asks
        # for both; see the discrepancy note rather than reading them as independent.
        record["maj_at_n"] = row.pi_mode
        record["maj_at_n_equals_pi_mode_by_construction"] = True
        rows.append(record)
    for union in analysis.unions:
        rows.append({**union, "config": "UNION"})
    return rows


def tab3_variance_components(analysis: Analysis) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in analysis.glmm:
        rows.extend(result.as_rows())
        rows.append(
            {
                "dataset": result.dataset,
                "subset": result.subset or "",
                "component": "DESIGN",
                "estimator": "glmm",
                "n_items": result.n_items,
                "n_models": len(result.models),
                "n_configs": len(result.configs),
                "censoring_rate": result.censoring.get("censoring_rate"),
                "backend": result.backend,
                "converged": result.converged,
                "n_bootstrap": result.n_bootstrap,
            }
        )
    for result in analysis.variance:
        for row in result.as_rows():
            row = dict(row)
            row["estimator"] = "moment"
            rows.append(row)
        rows.append(
            {
                "dataset": result.dataset,
                "subset": result.subset or "",
                "component": "DESIGN",
                "estimator": "moment",
                "n_items": result.n_items,
                "n_models": len(result.models),
                "n_configs": len(result.configs),
                "censoring_rate": result.censoring.get("censoring_rate"),
                "saturated_cell_rate": result.saturated_cell_rate,
                "n_items_all_zero": result.censoring.get("n_items_all_zero"),
                "n_items_all_one": result.censoring.get("n_items_all_one"),
                "n_items_dropped_incomplete": result.design.get("n_items_dropped"),
                "balanced": result.design.get("balanced"),
                "noise_floor": result.noise_floor.get("noise_floor"),
                "noise_floor_source": result.noise_floor.get("noise_floor_used"),
                "noise_floor_analytic": result.noise_floor.get("noise_floor_analytic"),
                "noise_floor_replicate": result.noise_floor.get("noise_floor_replicate"),
                "n_bootstrap": result.n_bootstrap,
            }
        )
    for row in analysis.null_calibration:
        rows.append({"component": "NULL_CALIBRATION", **row})
    return rows


def tab4_transfer(analysis: Analysis) -> List[Dict[str, Any]]:
    rows = [r.as_dict() for r in analysis.transfer]
    for key, reliability in sorted(
        analysis.reliabilities.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        rows.append({"pair_kind": "reliability", **reliability.as_dict()})
    rows.extend({"pair_kind": "overlap", **r.as_dict()} for r in analysis.overlaps)
    return rows


def tab5_method_comparison(analysis: Analysis) -> List[Dict[str, Any]]:
    """Budget-matched accuracy for every method, plus the paired tests against SC."""
    wanted = set(float(m) for m in analysis.config.table_multiples)
    rows = [
        p.as_dict()
        for p in analysis.cdv_points
        if not wanted or p.budget_multiple in wanted
    ]
    for row in rows:
        row["row_kind"] = "accuracy"
    for comparison in analysis.comparisons:
        if wanted and comparison["budget_multiple"] not in wanted:
            continue
        rows.append({"row_kind": "comparison_vs_sc", **comparison})
    return rows


def tab6_downstream(analysis: Analysis) -> List[Dict[str, Any]]:
    return list(analysis.allocations)


def tab7_precision_control(
    fp16: Analysis, quantised: Analysis
) -> List[Dict[str, Any]]:
    """Paired fp16 / 4-bit numbers and the P7 verdict."""

    def summarise(analysis: Analysis, label: str) -> Dict[str, Any]:
        transfer = [
            r.rho_disatt for r in analysis.transfer
            if r.kind == "config" and math.isfinite(r.rho_disatt)
        ]
        reliabilities = [r.r_mm for r in analysis.reliabilities.values()]
        overlaps = [r.j_config for r in analysis.overlaps if math.isfinite(r.j_config)]
        pi_modes = [r.pi_mode for r in analysis.ceilings if math.isfinite(r.pi_mode)]
        reorder = [r.reorder_rate for r in analysis.transitions if math.isfinite(r.reorder_rate)]
        margins = [c.margin() for c in analysis.corpus.all_cells() if c.n > 0]
        shares = {
            f"share_{name}": (
                sum(float(r.shares.get(name) or 0.0) for r in analysis.variance)
                / len(analysis.variance)
                if analysis.variance
                else float("nan")
            )
            for name in variance_mod.COMPONENT_NAMES
        }
        return {
            "precision": label,
            **shares,
            "rho_disatt": sum(transfer) / len(transfer) if transfer else float("nan"),
            "r_mm": sum(reliabilities) / len(reliabilities) if reliabilities else float("nan"),
            "j_config": sum(overlaps) / len(overlaps) if overlaps else float("nan"),
            "pi_mode": sum(pi_modes) / len(pi_modes) if pi_modes else float("nan"),
            "reorder_rate": sum(reorder) / len(reorder) if reorder else float("nan"),
            "mean_margin": sum(margins) / len(margins) if margins else float("nan"),
            "n_cells": len(analysis.corpus),
        }

    a = summarise(fp16, "fp16")
    b = summarise(quantised, "4bit")
    delta = {"precision": "delta_4bit_minus_fp16"}
    for key in a:
        if key == "precision":
            continue
        try:
            delta[key] = float(b[key]) - float(a[key])
        except (TypeError, ValueError):
            delta[key] = None
    # P7 asks whether the *conclusions* survive quantisation, so the verdict is about
    # the interaction share and the transfer statistic, not about raw accuracy --
    # accuracy is expected to drop and that is not what the control tests.
    share_shift = abs(float(delta.get("share_item_config") or float("nan")))
    rho_shift = abs(float(delta.get("rho_disatt") or float("nan")))
    delta["prediction"] = "P7"
    delta["statement"] = (
        "the variance decomposition and difficulty-transfer conclusions are "
        "unchanged at 4-bit: item x config share moves by under 5 points and "
        "rho_disatt by under 0.10"
    )
    delta["supported"] = bool(
        math.isfinite(share_shift) and share_shift < 0.05
        and math.isfinite(rho_shift) and rho_shift < 0.10
    )
    return [a, b, delta]


# -------------------------------------------------------------------- the driver
@dataclass
class Artefacts:
    """What was written, what was skipped and why."""

    out_dir: Path
    written: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "generated_at": self.generated_at,
            "written": self.written,
            "skipped": self.skipped,
            "warnings": self.warnings,
            "expected_figures": FIGURE_FILENAMES,
            "expected_tables": TABLE_FILENAMES,
            "n_written": len(self.written),
            "n_expected": len(FIGURE_FILENAMES) + len(TABLE_FILENAMES),
        }

    def report(self) -> str:
        lines = [f"artefacts in {self.out_dir}"]
        for key in list(FIGURE_FILENAMES) + list(TABLE_FILENAMES):
            name = FIGURE_FILENAMES.get(key) or TABLE_FILENAMES.get(key)
            if key in self.written:
                lines.append(f"  [ok]   {key:<3} {name}")
            else:
                lines.append(
                    f"  [skip] {key:<3} {name}  -- {self.skipped.get(key, 'not attempted')}"
                )
        if self.warnings:
            lines.append("warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def build(
    analysis: Analysis,
    out_dir: str | Path,
    precision_analysis: Optional[Analysis] = None,
) -> Artefacts:
    """Emit every artefact METHOD_SPEC 9 names, skipping what the data cannot support."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artefacts = Artefacts(out_dir=out_dir, warnings=list(analysis.warnings))

    figure_builders = {
        "F1": lambda p: fig1_variance_components(analysis, p),
        "F2": lambda p: fig2_transfer_scatter(analysis, p),
        "F3": lambda p: fig3_hard_subset_overlap(analysis, p),
        "F4": lambda p: fig4_ceiling_vs_coverage(analysis, p),
        "F5": lambda p: fig5_cdv_vs_sc(analysis, p),
        "F6": lambda p: fig6_reordering(analysis, p),
        "F7": lambda p: fig7_shift(analysis, p),
        "F8": (
            (lambda p: fig8_precision_control(analysis, precision_analysis, p))
            if precision_analysis is not None
            else None
        ),
    }
    table_builders = {
        "T1": lambda: tab1_setup(analysis),
        "T2": lambda: tab2_main_accuracy(analysis),
        "T3": lambda: tab3_variance_components(analysis),
        "T4": lambda: tab4_transfer(analysis),
        "T5": lambda: tab5_method_comparison(analysis),
        "T6": lambda: tab6_downstream(analysis),
        "T7": (
            (lambda: tab7_precision_control(analysis, precision_analysis))
            if precision_analysis is not None
            else None
        ),
    }

    for key, builder in figure_builders.items():
        name = FIGURE_FILENAMES[key]
        if builder is None:
            artefacts.skipped[key] = (
                "needs a second corpus at the other precision; pass "
                "--precision-results-dir"
            )
            continue
        try:
            path = builder(out_dir / name)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not lose twelve
            log.exception("figure %s failed", key)
            artefacts.skipped[key] = f"error: {type(exc).__name__}: {exc}"
            continue
        if path is None:
            artefacts.skipped[key] = "not enough data in the corpus"
        else:
            artefacts.written[key] = str(Path(path).name)

    for key, table_builder in table_builders.items():
        name = TABLE_FILENAMES[key]
        if table_builder is None:
            artefacts.skipped[key] = (
                "needs a second corpus at the other precision; pass "
                "--precision-results-dir"
            )
            continue
        try:
            rows = table_builder()
        except Exception as exc:  # noqa: BLE001
            log.exception("table %s failed", key)
            artefacts.skipped[key] = f"error: {type(exc).__name__}: {exc}"
            continue
        if not rows:
            artefacts.skipped[key] = "no rows"
            continue
        write_csv(rows, out_dir / name)
        artefacts.written[key] = name

    predictions = analysis.predictions()
    if predictions:
        write_csv(predictions, out_dir / EXTRA_FILENAMES["predictions"])
    write_json_atomic(
        out_dir / EXTRA_FILENAMES["corpus_health"],
        {
            "health": analysis.health,
            "configs": analysis.configs,
            "null_calibration": analysis.null_calibration,
            "extractor_agreement": analysis.extractor_agreement,
            "per_config_failures": analysis.per_config_failures,
        },
    )
    write_json_atomic(out_dir / EXTRA_FILENAMES["artefacts"], artefacts.as_dict())
    log.info("\n%s", artefacts.report())
    return artefacts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build every METHOD_SPEC section 9 figure and table from a results tree. "
            "CPU only: run this with the Kaggle accelerator switched OFF."
        )
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="paper_artefacts")
    parser.add_argument(
        "--precision-results-dir",
        default=None,
        help="second results tree at the other precision, for F8/T7",
    )
    parser.add_argument("--grade", action="store_true",
                        help="grade any ungraded results in place first")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--n-repeats", type=int, default=cdv_mod.DEFAULT_N_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--configs", default=None,
                        help="comma-separated configuration ids (default: primary c0-c2)")
    parser.add_argument("--cdv-configs", default=None,
                        help="comma-separated configurations CDV spreads over")
    parser.add_argument("--no-png", action="store_true",
                        help="write only the vector PDF, no preview PNG")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(args.log_level)
    cfg = PaperConfig(
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=args.seed,
        n_repeats=args.n_repeats,
        configs=args.configs.split(",") if args.configs else None,
        cdv_configs=args.cdv_configs.split(",") if args.cdv_configs else None,
        also_png=not args.no_png,
    )
    corpus = load_corpus(args.results_dir, grade=args.grade)
    if not len(corpus):
        log.error(
            "no graded cells found under %s. Run the experiment first, or pass "
            "--grade if only raw results.jsonl files exist.",
            args.results_dir,
        )
        return 1
    analysis = analyse(corpus, cfg)
    precision = None
    if args.precision_results_dir:
        other = load_corpus(args.precision_results_dir, grade=args.grade)
        if len(other):
            precision = analyse(other, cfg)
        else:
            log.warning("no cells under %s; skipping F8/T7", args.precision_results_dir)
    artefacts = build(analysis, args.out_dir, precision_analysis=precision)
    print(artefacts.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
