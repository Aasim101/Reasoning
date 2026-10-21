"""Publication figures: matplotlib only, vector PDF, no chartjunk.

Constraints encoded here:

* No seaborn. matplotlib alone, so the figures are reproducible from
  `requirements.txt` on any machine, including a Kaggle image.
* `Agg` backend, set before pyplot is imported, so this works headless.
* `pdf.fonttype=42` / `ps.fonttype=42` embed TrueType rather than Type 3 fonts.
  Several venues reject Type 3, and it is easier to set once than to discover at
  camera-ready.
* Series are distinguished by marker *and* linestyle as well as colour, and the
  palette is colourblind-safe, so a printed greyscale figure is still readable.
* Fonts are requested with a fallback chain: a missing serif font must not raise
  on someone else's machine.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

#: Okabe-Ito: colourblind-safe and distinguishable in greyscale.
PALETTE: Tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C564B",
    "#000000",
)
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES: Tuple[str, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))

#: Single-column and double-column figure sizes, in inches.
SINGLE_COLUMN = (3.4, 2.6)
WIDE = (7.0, 2.8)


def set_style() -> None:
    """Apply paper rcParams. Safe to call repeatedly."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "serif",
            ],
            "mathtext.fontset": "dejavuserif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.figsize": SINGLE_COLUMN,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Embed editable TrueType fonts; Type 3 is rejected by some venues.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "errorbar.capsize": 2,
        }
    )


def _save(fig: Any, out_path: str | Path, also_png: bool = True) -> Path:
    out_path = Path(out_path).with_suffix(".pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    if also_png:
        # PNG only for inline notebook preview; the paper always uses the PDF.
        fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    log.info("wrote %s", out_path)
    return out_path


def _style_for(index: int) -> Dict[str, Any]:
    return {
        "color": PALETTE[index % len(PALETTE)],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
    }


def _short(name: Any, limit: int = 18) -> str:
    text = str(name or "?")
    text = text.split("/")[-1]
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def _errors(summaries: Sequence[Dict[str, Any]]) -> Optional[List[List[float]]]:
    """Asymmetric error bars from ci_low/ci_high, or None when absent."""
    lows, highs = [], []
    for s in summaries:
        acc = float(s.get("accuracy") or 0.0)
        low = s.get("ci_low")
        high = s.get("ci_high")
        if low is None or high is None:
            return None
        lows.append(max(0.0, acc - float(low)))
        highs.append(max(0.0, float(high) - acc))
    return [lows, highs]


# --------------------------------------------------------------------- figures
def fig_accuracy_bar(
    summaries: Sequence[Dict[str, Any]],
    out_path: str | Path,
    title: Optional[str] = None,
) -> Path:
    """Accuracy per strategy with confidence intervals, grouped by dataset."""
    set_style()
    datasets = sorted({str(s.get("dataset")) for s in summaries})
    strategies = sorted({str(s.get("strategy")) for s in summaries})
    width = max(SINGLE_COLUMN[0], 1.1 * len(datasets) * max(1, len(strategies)) * 0.55)
    fig, ax = plt.subplots(figsize=(min(width, WIDE[0]), SINGLE_COLUMN[1]))

    bar_width = 0.8 / max(1, len(strategies))
    for si, strategy in enumerate(strategies):
        xs, ys, rows = [], [], []
        for di, dataset in enumerate(datasets):
            match = [
                s
                for s in summaries
                if str(s.get("dataset")) == dataset and str(s.get("strategy")) == strategy
            ]
            if not match:
                continue
            row = match[0]
            xs.append(di + (si - (len(strategies) - 1) / 2) * bar_width)
            ys.append(float(row.get("accuracy") or 0.0))
            rows.append(row)
        if not xs:
            continue
        style = _style_for(si)
        ax.bar(
            xs,
            ys,
            width=bar_width * 0.92,
            label=_short(strategy),
            color=style["color"],
            edgecolor="black",
            linewidth=0.4,
            yerr=_errors(rows),
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([_short(d, 14) for d in datasets])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    if title:
        ax.set_title(title)
    if len(strategies) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, out_path)


def fig_accuracy_vs_compute(
    summaries: Sequence[Dict[str, Any]],
    out_path: str | Path,
    log_x: bool = True,
    annotate: bool = True,
) -> Path:
    """Accuracy against mean completion tokens per example.

    The figure that decides whether a method is worth its compute: a method that
    is only better at 8x the tokens belongs above the self-consistency curve, not
    merely above the single-sample baseline.
    """
    set_style()
    datasets = sorted({str(s.get("dataset")) for s in summaries})
    strategies = sorted({str(s.get("strategy")) for s in summaries})
    fig, ax = plt.subplots(figsize=WIDE if len(strategies) > 5 else SINGLE_COLUMN)

    # Each method is a separate point, NOT a line through the others: joining
    # different strategies would imply a compute-scaling curve that does not
    # exist. A curve is only meaningful within one method (see fig_majority_vs_k).
    for si, strategy in enumerate(strategies):
        rows = [s for s in summaries if str(s.get("strategy")) == strategy]
        if not rows:
            continue
        xs = [max(1e-9, float(s.get("tokens_mean_completion") or 0.0)) for s in rows]
        ys = [float(s.get("accuracy") or 0.0) for s in rows]
        style = _style_for(si)
        errors = _errors(rows)
        ax.errorbar(
            xs,
            ys,
            yerr=errors,
            fmt=style["marker"],
            color=style["color"],
            markersize=5,
            elinewidth=0.7,
            linestyle="none",
            label=_short(strategy, 20),
        )
        if annotate and len(datasets) > 1:
            for x, y, row in zip(xs, ys, rows):
                ax.annotate(
                    _short(row.get("dataset"), 10),
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, -2),
                    fontsize=6,
                    color=style["color"],
                )
    if log_x and any(float(s.get("tokens_mean_completion") or 0) > 0 for s in summaries):
        ax.set_xscale("log")
    ax.set_xlabel("Mean completion tokens per example")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    if len(strategies) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.4)
    return _save(fig, out_path)


def fig_majority_vs_k(
    curves: Dict[str, Dict[str, Any]],
    out_path: str | Path,
    ylabel: str = "Majority-vote accuracy",
) -> Path:
    """Majority-vote accuracy against k, with a shaded confidence band."""
    set_style()
    fig, ax = plt.subplots()
    plotted = 0
    for i, (label, curve) in enumerate(sorted(curves.items())):
        ks = curve.get("ks") or []
        values = curve.get("values") or []
        if not ks:
            continue
        style = _style_for(i)
        ax.plot(ks, values, label=_short(label), **style)
        lows = curve.get("ci_low") or []
        highs = curve.get("ci_high") or []
        if lows and highs and all(v is not None for v in lows + highs):
            ax.fill_between(ks, lows, highs, color=style["color"], alpha=0.15, linewidth=0)
        plotted += 1
    ax.set_xlabel("Samples $k$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    if plotted > 1:
        ax.legend(loc="best")
    return _save(fig, out_path)


def fig_pass_at_k(
    curves: Dict[str, Dict[str, Any]], out_path: str | Path, log_x: bool = True
) -> Path:
    """pass@k against k. The coverage ceiling any selection method aims at."""
    set_style()
    fig, ax = plt.subplots()
    plotted = 0
    for i, (label, curve) in enumerate(sorted(curves.items())):
        ks = curve.get("ks") or []
        values = curve.get("values") or []
        if not ks:
            continue
        ax.plot(ks, values, label=_short(label), **_style_for(i))
        plotted += 1
    if log_x and plotted:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("Samples $k$")
    ax.set_ylabel("pass@$k$")
    ax.set_ylim(0, 1.0)
    if plotted > 1:
        ax.legend(loc="best")
    return _save(fig, out_path)


def fig_token_cost(
    summaries: Sequence[Dict[str, Any]], out_path: str | Path
) -> Path:
    """Completion tokens per correct answer, horizontal bars (lower is better)."""
    set_style()
    rows = [
        s
        for s in summaries
        if math.isfinite(float(s.get("tokens_per_correct") or float("inf")))
    ]
    rows.sort(key=lambda s: float(s.get("tokens_per_correct")))
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no finite token cost", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return _save(fig, out_path)

    height = max(SINGLE_COLUMN[1], 0.22 * len(rows) + 0.8)
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN[0], height))
    labels = [
        f"{_short(s.get('strategy'), 16)} ({_short(s.get('dataset'), 10)})" for s in rows
    ]
    values = [float(s.get("tokens_per_correct")) for s in rows]
    ax.barh(
        range(len(rows)),
        values,
        color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))],
        edgecolor="black",
        linewidth=0.4,
        height=0.7,
    )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Completion tokens per correct answer")
    return _save(fig, out_path)


def write_all(
    summaries: Sequence[Dict[str, Any]],
    out_dir: str | Path,
    majority_curves: Optional[Dict[str, Dict[str, Any]]] = None,
    pass_curves: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Path]:
    """Write every figure that the available data supports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    if summaries:
        written.append(fig_accuracy_bar(summaries, out_dir / "fig_accuracy"))
        written.append(
            fig_accuracy_vs_compute(summaries, out_dir / "fig_accuracy_vs_compute")
        )
        written.append(fig_token_cost(summaries, out_dir / "fig_token_cost"))
    if majority_curves:
        written.append(fig_majority_vs_k(majority_curves, out_dir / "fig_majority_vs_k"))
    if pass_curves:
        written.append(fig_pass_at_k(pass_curves, out_dir / "fig_pass_at_k"))
    return written
