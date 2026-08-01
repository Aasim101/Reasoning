"""Walk a results tree, summarise every run, and emit paper artefacts.

    python -m src.analysis.aggregate --results-dir results --out-dir paper_assets \\
        --figures --tables --baseline cot_zeroshot

Reads `graded.jsonl` (or grades on demand with `--grade`), writes `summary.json`,
`summary.csv` and `comparisons.json`, and optionally the figures and LaTeX
tables. Uses only the standard library plus numpy/matplotlib: no pandas, so it
runs anywhere the harness runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..checkpointing import load_records
from ..metrics import (
    DEFAULT_N_BOOTSTRAP,
    compare_strategies,
    group_records,
    majority_accuracy_curve,
    pass_at_k_curve,
    summarize_run,
)
from ..utils import read_json, setup_logging, write_json_atomic

log = logging.getLogger(__name__)

#: Sample counts used for the pass@k and maj@k curves.
DEFAULT_KS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

#: Columns of summary.csv, in order.
CSV_COLUMNS = (
    "dataset",
    "model",
    "strategy",
    "seed",
    "config_hash",
    "n",
    "n_correct",
    "accuracy",
    "ci_low",
    "ci_high",
    "vote_accuracy",
    "max_samples",
    "pass_at_1",
    "pass_at_max",
    "tokens_mean_prompt",
    "tokens_mean_completion",
    "tokens_total",
    "tokens_per_correct",
    "tokens_mean_latency_s",
    "n_errors",
    "extraction_failure_rate",
    "elapsed_seconds",
    "gpu_hours",
    "grader_version",
    "grader_backend",
    "run_dir",
)


def find_run_dirs(results_dir: str | Path) -> List[Path]:
    """Every directory under `results_dir` that looks like a run."""
    root = Path(results_dir)
    if not root.exists():
        return []
    found = {
        p.parent
        for pattern in ("**/graded.jsonl", "**/results.jsonl", "**/manifest.json")
        for p in root.glob(pattern)
    }
    return sorted(found)


def load_run(run_dir: Path, grade: bool = False) -> Optional[Dict[str, Any]]:
    """Load one run's graded records plus its manifest metadata."""
    graded = run_dir / "graded.jsonl"
    raw = run_dir / "results.jsonl"
    if not graded.exists():
        if not raw.exists():
            return None
        if grade:
            from ..grading import grade_file

            log.info("grading %s on demand", raw)
            grade_file(raw, graded)
        else:
            log.warning(
                "%s has results.jsonl but no graded.jsonl; skipping (pass --grade to "
                "grade it now, or run `python -m src.grading --run-dir %s`)",
                run_dir,
                run_dir,
            )
            return None

    records = load_records(graded, dedupe=True)
    if not records:
        log.warning("%s contains no graded records; skipping", graded)
        return None
    manifest = read_json(run_dir / "manifest.json", default={}) or {}
    return {"run_dir": run_dir, "records": records, "manifest": manifest}


def summarize(run: Dict[str, Any], ks: Sequence[int] = DEFAULT_KS) -> Dict[str, Any]:
    """Per-run summary row, enriched with manifest cost data and curves."""
    records = run["records"]
    manifest = run["manifest"]
    summary = summarize_run(records)
    summary["run_dir"] = str(run["run_dir"])
    summary["run_name"] = manifest.get("run_name")
    summary["elapsed_seconds"] = manifest.get("elapsed_seconds")
    summary["gpu_hours"] = manifest.get("gpu_hours")
    summary["status"] = manifest.get("status")
    summary["n_total_planned"] = manifest.get("n_total")
    max_samples = int(summary.get("max_samples") or 1)
    usable = [k for k in ks if k <= max_samples]
    if max_samples > 1 and usable:
        summary["pass_curve"] = pass_at_k_curve(records, usable)
        summary["majority_curve"] = majority_accuracy_curve(
            records, usable, n_bootstrap=1000, seed=0
        )
    return summary


def build_comparisons(
    runs: Sequence[Dict[str, Any]],
    baseline: str,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Compare every strategy against `baseline` within each (dataset, model)."""
    by_cell: Dict[Tuple[Any, Any], Dict[str, List[Dict[str, Any]]]] = {}
    for run in runs:
        for (dataset, model, strategy, _seed), records in group_records(
            run["records"]
        ).items():
            cell = by_cell.setdefault((dataset, model), {})
            cell.setdefault(str(strategy), []).extend(records)

    out: List[Dict[str, Any]] = []
    for (dataset, model), strategies in sorted(
        by_cell.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        if baseline not in strategies:
            log.info(
                "no baseline %r for dataset=%s model=%s (have %s); skipping comparisons",
                baseline,
                dataset,
                model,
                sorted(strategies),
            )
            continue
        for strategy, records in sorted(strategies.items()):
            if strategy == baseline:
                continue
            comparison = compare_strategies(
                strategies[baseline],
                records,
                n_bootstrap=n_bootstrap,
                seed=seed,
                label_a=baseline,
                label_b=strategy,
            )
            comparison["dataset"] = dataset
            comparison["model"] = model
            out.append(comparison)
    return out


def write_csv(summaries: Sequence[Dict[str, Any]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in summaries:
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
    log.info("wrote %s", out_path)
    return out_path


def print_console_table(summaries: Sequence[Dict[str, Any]]) -> None:
    if not summaries:
        print("no runs found")
        return
    header = (
        f"{'dataset':<14} {'strategy':<20} {'model':<22} {'seed':>4} {'n':>5} "
        f"{'acc':>6} {'95% CI':>15} {'tok/ex':>8} {'tok/correct':>11}"
    )
    print(header)
    print("-" * len(header))
    for s in sorted(
        summaries, key=lambda r: (str(r.get("dataset")), -float(r.get("accuracy") or 0))
    ):
        cost = float(s.get("tokens_per_correct") or float("inf"))
        print(
            f"{str(s.get('dataset'))[:14]:<14} "
            f"{str(s.get('strategy'))[:20]:<20} "
            f"{str(s.get('model')).split('/')[-1][:22]:<22} "
            f"{str(s.get('seed')):>4} "
            f"{int(s.get('n') or 0):>5} "
            f"{float(s.get('accuracy') or 0):>6.3f} "
            f"[{float(s.get('ci_low') or 0):.3f},{float(s.get('ci_high') or 0):.3f}] "
            f"{float(s.get('tokens_mean_completion') or 0):>8.0f} "
            # ASCII only: the Windows console defaults to cp1252 and a nicer
            # infinity glyph would raise UnicodeEncodeError there.
            f"{(f'{cost:>11.0f}' if cost != float('inf') else 'inf'.rjust(11))}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.analysis.aggregate",
        description="Summarise a results tree and emit paper figures and tables.",
    )
    p.add_argument("--results-dir", default="results", help="root of the results tree")
    p.add_argument("--out-dir", default="paper_assets", help="where to write artefacts")
    p.add_argument("--figures", action="store_true", help="write PDF figures")
    p.add_argument("--tables", action="store_true", help="write LaTeX tables")
    p.add_argument(
        "--grade",
        action="store_true",
        help="grade runs that only have results.jsonl (CPU only, no GPU needed)",
    )
    p.add_argument(
        "--baseline",
        default="cot_zeroshot",
        help="strategy that significance tests compare against",
    )
    p.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (recorded in output)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    run_dirs = find_run_dirs(args.results_dir)
    if not run_dirs:
        log.error("no run directories found under %s", args.results_dir)
        return 1
    log.info("found %d candidate run director(ies)", len(run_dirs))

    runs = [r for r in (load_run(d, grade=args.grade) for d in run_dirs) if r]
    if not runs:
        log.error(
            "no graded runs found. Grade them first: "
            "python -m src.grading --results-root %s",
            args.results_dir,
        )
        return 1

    summaries = [summarize(r) for r in runs]
    comparisons = build_comparisons(
        runs, args.baseline, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "summary.json", summaries)
    write_csv(summaries, out_dir / "summary.csv")
    write_json_atomic(out_dir / "comparisons.json", comparisons)

    if args.figures:
        from . import figures

        majority_curves = {
            f"{s.get('strategy')} ({s.get('dataset')})": s["majority_curve"]
            for s in summaries
            if s.get("majority_curve")
        }
        pass_curves = {
            f"{s.get('strategy')} ({s.get('dataset')})": s["pass_curve"]
            for s in summaries
            if s.get("pass_curve")
        }
        figures.write_all(summaries, out_dir, majority_curves, pass_curves)

    if args.tables:
        from . import tables

        tables.write_all(summaries, comparisons, out_dir, baseline=args.baseline)

    print()
    print_console_table(summaries)
    print(f"\nwrote artefacts to {out_dir.resolve()}")
    if comparisons:
        print(f"\ncomparisons against {args.baseline!r}:")
        for c in comparisons:
            mc = c.get("mcnemar") or {}
            boot = c.get("paired_bootstrap") or {}
            if not c.get("n_paired"):
                continue
            print(
                f"  {c.get('dataset')}: {c.get('label_b')} vs {c.get('label_a')}: "
                f"delta={100 * float(c.get('delta') or 0):+.1f} pts "
                f"[{100 * float(boot.get('ci_low', 0)):+.1f}, "
                f"{100 * float(boot.get('ci_high', 0)):+.1f}] "
                f"McNemar p={float(mc.get('p_value', 1)):.4f} "
                f"(n={c.get('n_paired')})"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
