"""The sampling corpus: one tidy table that every downstream analysis reads.

METHOD_SPEC section 4.4 specifies one Parquet file per (model, dataset, config)
with per-sample rows. This harness's durable artefact is append-only JSONL
instead, because JSONL is what survives a session killed mid-write -- a
half-written Parquet file is unreadable, and losing a cell to that would cost
GPU quota. JSONL is therefore the source of truth and `export_parquet` below
writes the spec's layout as a *derived* view, so anything expecting Parquet still
works. (Reported as a deliberate deviation; nothing downstream depends on it.)

The shapes this module produces:

* `SampleRow`  -- one generated chain: its canonical answer class, correctness and
  completion tokens. The unit the CDV re-partitioning consumes.
* `Cell`       -- one (item, model, config, seed): `k` successes out of `N`, plus
  the modal class and the top-1/top-2 margin. The unit the variance
  decomposition and the transfer correlations consume.
* `Corpus`     -- both, indexed, with the design-completeness checks that the
  balanced method-of-moments fit in `variance.py` silently depends on.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS, canonical_classes
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID, is_separate_arm
from ..metrics import haldane_logit, haldane_logit_var, is_saturated

log = logging.getLogger(__name__)

#: (dataset, subset) -- GSM-Symbolic `main` and `p2` are separate benchmarks for
#: every purpose in this paper, so they are never pooled by accident.
BenchmarkKey = Tuple[str, Optional[str]]
#: (dataset, subset, item_id, model, config_id, seed). The benchmark is part of the
#: key because item ids are only unique *within* a benchmark: GSM-Symbolic `main`
#: and `p2` are the same fifty templates instantiated twice, so keying on the item
#: alone would have one silently overwrite the other -- and the two are precisely
#: what the distribution-shift arm compares.
CellKey = Tuple[str, Optional[str], str, str, str, int]
#: `CellKey` prefixed with the strategy, for the auxiliary passes below.
AuxKey = Tuple[str, str, Optional[str], str, str, str, int]

#: Strategies whose records *are* the sampling corpus: many chains per cell at the
#: configuration's decoding parameters.
SAMPLING_STRATEGIES: Tuple[str, ...] = (
    "configured_sampling",
    "cdv_corpus",
    "self_consistency",
    "majority_vote",
)

#: Single-pass strategies that share a `(item, model, config, seed)` key with a
#: sampling cell but are a different measurement. Baseline B2's greedy pass is the
#: case that matters: at `N=1` it is saturated by construction, so letting it into
#: the crossed design would both collide with the real cell and drive the noise
#: floor to nonsense. They are kept, separately, in `Corpus.aux`.
AUX_STRATEGIES: Tuple[str, ...] = ("greedy_pass", "direct", "cot_zeroshot", "cot_fewshot")


@dataclass
class SampleRow:
    """One sampled chain of thought, reduced to what the analysis needs."""

    item_id: str
    dataset: str
    subset: Optional[str]
    model: str
    config: str
    seed: int
    sample_idx: int
    answer: Optional[str]
    canonical_class: str
    is_correct: bool
    tokens_completion: int
    mean_logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    truncated: bool = False
    rendered_prompt_hash: Optional[str] = None

    @property
    def extraction_failed(self) -> bool:
        return self.canonical_class == EXTRACTION_FAILURE_CLASS


@dataclass
class Cell:
    """One (item, model, configuration, seed) cell of the crossed design."""

    item_id: str
    dataset: str
    subset: Optional[str]
    model: str
    config: str
    seed: int
    gold_answer: str
    answer_type: str
    samples: List[SampleRow] = field(default_factory=list)
    template_id: Optional[str] = None
    tier: Optional[str] = None
    strategy: Optional[str] = None
    errored: bool = False

    # ------------------------------------------------------------------ counts
    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def k(self) -> int:
        return sum(1 for s in self.samples if s.is_correct)

    @property
    def p_hat(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def z(self) -> float:
        """The Haldane-corrected empirical logit of the cell's success rate."""
        return haldane_logit(self.k, self.n)

    @property
    def sampling_var(self) -> float:
        """`1/(k+1/2) + 1/(N-k+1/2)`: the analytic noise floor for this cell."""
        return haldane_logit_var(self.k, self.n)

    @property
    def saturated(self) -> bool:
        return is_saturated(self.k, self.n)

    @property
    def tokens_completion(self) -> int:
        return sum(s.tokens_completion for s in self.samples)

    @property
    def n_extraction_failures(self) -> int:
        return sum(1 for s in self.samples if s.extraction_failed)

    @property
    def extraction_failure_rate(self) -> float:
        return self.n_extraction_failures / self.n if self.n else 0.0

    # ------------------------------------------------------- modes and margins
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.samples:
            counts[s.canonical_class] = counts.get(s.canonical_class, 0) + 1
        return counts

    def modal_class(self) -> Optional[str]:
        """The most frequent canonical class; ties break by first occurrence.

        The extraction-failure class competes on equal terms, because a cell whose
        most common outcome is an unparseable answer really does have an
        unparseable mode, and hiding that would flatter the modal ceiling.
        """
        if not self.samples:
            return None
        counts = self.class_counts()
        top = max(counts.values())
        for s in self.samples:  # input order == first occurrence
            if counts[s.canonical_class] == top:
                return s.canonical_class
        return None

    def modal_correct(self) -> bool:
        """Is the modal class the correct answer? This is `pi_mode`'s indicator."""
        mode = self.modal_class()
        if mode is None or mode == EXTRACTION_FAILURE_CLASS:
            return False
        return any(s.is_correct for s in self.samples if s.canonical_class == mode)

    def modal_tie(self) -> bool:
        counts = sorted(self.class_counts().values(), reverse=True)
        return len(counts) > 1 and counts[0] == counts[1]

    def margin(self) -> float:
        """`(count_top1 - count_top2) / N`, the mechanism analysis's key covariate.

        A configuration change can only move the mode where two classes hold
        comparable mass, so the paper predicts reorderings concentrate where this
        is small. A cell with a single class has margin 1.0 and cannot reorder.
        """
        if not self.samples:
            return float("nan")
        counts = sorted(self.class_counts().values(), reverse=True)
        second = counts[1] if len(counts) > 1 else 0
        return (counts[0] - second) / self.n

    def key(self) -> CellKey:
        return (
            self.dataset, self.subset, self.item_id, self.model, self.config, self.seed
        )

    def aux_key(self) -> AuxKey:
        return (str(self.strategy),) + self.key()

    def benchmark(self) -> BenchmarkKey:
        return (self.dataset, self.subset)

    def bootstrap_unit(self) -> str:
        """The resampling unit: the template where one exists, else the item.

        GSM-Symbolic instances generated from one template are not independent, so
        METHOD_SPEC section 7.2 requires resampling templates there. Returning the
        item id elsewhere means every caller can just resample this value.
        """
        return str(self.template_id) if self.template_id else self.item_id


@dataclass
class Corpus:
    """Every cell of the design, indexed for the analyses that consume it."""

    cells: Dict[CellKey, Cell] = field(default_factory=dict)
    #: Single-pass cells (baseline B2's greedy decode, chiefly), kept out of the
    #: crossed design. See `AUX_STRATEGIES`.
    aux: Dict[AuxKey, Cell] = field(default_factory=dict)
    #: Non-fatal problems found while loading, surfaced in `tab1_setup.csv`.
    warnings: List[str] = field(default_factory=list)
    #: run_dir -> manifest, for the reproducibility table.
    manifests: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.cells)

    def all_cells(self) -> List[Cell]:
        return list(self.cells.values())

    def samples(self) -> List[SampleRow]:
        return [s for cell in self.cells.values() for s in cell.samples]

    def aux_cells(self, strategy: str = "greedy_pass") -> List[Cell]:
        return [c for c in self.aux.values() if c.strategy == strategy]

    def select(
        self,
        dataset: Optional[str] = None,
        subset: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[str] = None,
        seed: Optional[int] = None,
        configs: Optional[Sequence[str]] = None,
        models: Optional[Sequence[str]] = None,
        include_separate_arms: bool = False,
    ) -> List[Cell]:
        """Filtered cell list. `subset` is matched only when `dataset` is given."""
        wanted_configs = set(configs) if configs is not None else None
        wanted_models = set(models) if models is not None else None
        out: List[Cell] = []
        for cell in self.cells.values():
            if dataset is not None and cell.dataset != dataset:
                continue
            if dataset is not None and subset is not None and cell.subset != subset:
                continue
            if model is not None and cell.model != model:
                continue
            if config is not None and cell.config != config:
                continue
            if seed is not None and cell.seed != seed:
                continue
            if wanted_configs is not None and cell.config not in wanted_configs:
                continue
            if wanted_models is not None and cell.model not in wanted_models:
                continue
            if not include_separate_arms and is_separate_arm(cell.config):
                continue
            out.append(cell)
        return out

    def benchmarks(self, include_separate_arms: bool = False) -> List[BenchmarkKey]:
        keys = {
            c.benchmark()
            for c in self.cells.values()
            if include_separate_arms or not is_separate_arm(c.config)
        }
        return sorted(keys, key=lambda t: (str(t[0]), str(t[1])))

    def models(self) -> List[str]:
        return sorted({c.model for c in self.cells.values()})

    def configs(self, include_separate_arms: bool = False) -> List[str]:
        found = {
            c.config
            for c in self.cells.values()
            if include_separate_arms or not is_separate_arm(c.config)
        }
        # Registry order first (c0, c1, ...), then anything unrecognised.
        ordered = [c for c in PRIMARY_CONFIG_IDS if c in found]
        return ordered + sorted(found - set(ordered))

    def seeds(self, config: str = REFERENCE_CONFIG_ID) -> List[int]:
        return sorted({c.seed for c in self.cells.values() if c.config == config})

    def items(
        self, dataset: str, subset: Optional[str] = None, seed: int = 0
    ) -> List[str]:
        return sorted(
            {
                c.item_id
                for c in self.cells.values()
                if c.dataset == dataset and c.subset == subset and c.seed == seed
            }
        )

    # --------------------------------------------------------- design geometry
    def crossed_items(
        self,
        dataset: str,
        subset: Optional[str],
        models: Sequence[str],
        configs: Sequence[str],
        seed: int = 0,
    ) -> List[str]:
        """Items present in **every** (model, config) cell, in sorted order.

        The variance decomposition is a balanced method-of-moments fit, so a
        missing cell does not merely lose precision, it biases the components.
        Restricting to the fully crossed intersection is the honest fix, and the
        number dropped is reported rather than silently absorbed.
        """
        per_combination: List[Set[str]] = []
        for model in models:
            for config in configs:
                per_combination.append(
                    {
                        c.item_id
                        for c in self.cells.values()
                        if c.dataset == dataset
                        and c.subset == subset
                        and c.model == model
                        and c.config == config
                        and c.seed == seed
                        and c.n > 0
                    }
                )
        if not per_combination:
            return []
        common = set.intersection(*per_combination) if per_combination else set()
        return sorted(common)

    def design_report(
        self,
        dataset: str,
        subset: Optional[str],
        models: Sequence[str],
        configs: Sequence[str],
        seed: int = 0,
    ) -> Dict[str, Any]:
        union: Set[str] = {
            c.item_id
            for c in self.cells.values()
            if c.dataset == dataset and c.subset == subset and c.seed == seed
        }
        crossed = self.crossed_items(dataset, subset, models, configs, seed)
        sample_counts = sorted(
            {
                c.n
                for c in self.cells.values()
                if c.dataset == dataset
                and c.subset == subset
                and c.seed == seed
                and c.config in set(configs)
                and c.model in set(models)
            }
        )
        return {
            "dataset": dataset,
            "subset": subset,
            "models": list(models),
            "configs": list(configs),
            "seed": seed,
            "n_items_union": len(union),
            "n_items_crossed": len(crossed),
            "n_items_dropped": len(union) - len(crossed),
            "balanced": len(crossed) == len(union) and len(union) > 0,
            "sample_counts_seen": sample_counts,
            "ragged_sample_counts": len(sample_counts) > 1,
        }

    def counts_matrix(
        self,
        dataset: str,
        subset: Optional[str],
        models: Sequence[str],
        configs: Sequence[str],
        items: Sequence[str],
        seed: int = 0,
    ) -> Tuple[Any, Any]:
        """`(k, n)` arrays of shape (items, models, configs) for the ANOVA fit."""
        import numpy as np

        index = {(c.item_id, c.model, c.config): c for c in self.cells.values()
                 if c.dataset == dataset and c.subset == subset and c.seed == seed}
        k = np.zeros((len(items), len(models), len(configs)), dtype=float)
        n = np.zeros_like(k)
        for i, item in enumerate(items):
            for m, model in enumerate(models):
                for c, config in enumerate(configs):
                    cell = index.get((item, model, config))
                    if cell is None:
                        k[i, m, c] = np.nan
                        n[i, m, c] = np.nan
                    else:
                        k[i, m, c] = cell.k
                        n[i, m, c] = cell.n
        return k, n

    def p_hat_vector(
        self,
        dataset: str,
        subset: Optional[str],
        model: str,
        config: str,
        items: Sequence[str],
        seed: int = 0,
    ) -> List[float]:
        """`p_hat` for a fixed cell column, aligned to `items` (nan when absent)."""
        index = {
            c.item_id: c
            for c in self.cells.values()
            if c.dataset == dataset
            and c.subset == subset
            and c.model == model
            and c.config == config
            and c.seed == seed
        }
        return [
            index[item].p_hat if item in index else float("nan") for item in items
        ]

    def bootstrap_units(self, items: Sequence[str]) -> List[str]:
        """Map items to their resampling unit (template where one exists)."""
        unit_of: Dict[str, str] = {}
        for cell in self.cells.values():
            unit_of.setdefault(cell.item_id, cell.bootstrap_unit())
        return [unit_of.get(item, item) for item in items]


# ----------------------------------------------------------------------- loading
def _sample_rows(record: Dict[str, Any]) -> List[SampleRow]:
    answers = record.get("sample_answers")
    if not isinstance(answers, list):
        answers = []
    correct = record.get("sample_correct") or []
    stats = record.get("sample_stats") or []
    classes = canonical_classes(
        answers,
        str(record.get("answer_type") or "math"),
        record.get("choices") or (record.get("extra") or {}).get("choices"),
    )
    rows: List[SampleRow] = []
    for i, answer in enumerate(answers):
        stat = stats[i] if i < len(stats) else {}
        rows.append(
            SampleRow(
                item_id=str(record.get("example_id")),
                dataset=str(record.get("dataset")),
                subset=record.get("subset"),
                model=str(record.get("model")),
                config=str(record.get("elicitation") or "c?"),
                seed=int(record.get("seed") or 0),
                sample_idx=i,
                answer=None if answer is None else str(answer),
                canonical_class=classes[i],
                is_correct=bool(correct[i]) if i < len(correct) else False,
                tokens_completion=int((stat or {}).get("tokens_completion") or 0),
                mean_logprob=(stat or {}).get("mean_logprob"),
                finish_reason=(stat or {}).get("finish_reason"),
                truncated=bool((stat or {}).get("truncated")),
                rendered_prompt_hash=(stat or {}).get("rendered_prompt_hash")
                or (record.get("extra") or {}).get("rendered_prompt_hash"),
            )
        )
    return rows


def cell_from_record(record: Dict[str, Any]) -> Cell:
    meta = record.get("meta") or {}
    return Cell(
        item_id=str(record.get("example_id")),
        dataset=str(record.get("dataset")),
        subset=record.get("subset"),
        model=str(record.get("model")),
        config=str(record.get("elicitation") or "c?"),
        seed=int(record.get("seed") or 0),
        gold_answer=str(record.get("gold_answer") or ""),
        answer_type=str(record.get("answer_type") or "math"),
        samples=_sample_rows(record),
        template_id=meta.get("template_id"),
        tier=meta.get("tier"),
        strategy=record.get("strategy"),
        errored=bool(record.get("error")),
    )


def load_corpus(
    results_dir: str | Path,
    grade: bool = False,
    require_elicitation: bool = True,
) -> Corpus:
    """Build a `Corpus` from a results tree of graded JSONL files.

    Records without an `elicitation` id are skipped by default: they come from a
    run where the configuration factor was not in play, so pooling them into the
    crossed design would introduce a cell with an unknown configuration and
    corrupt every variance component.
    """
    from ..checkpointing import load_records
    from ..utils import read_json
    from .aggregate import find_run_dirs

    corpus = Corpus()
    n_skipped_no_config = 0
    for run_dir in find_run_dirs(results_dir):
        graded = _graded_path(run_dir, grade=grade)
        if graded is None:
            continue
        manifest = read_json(run_dir / "manifest.json", default={}) or {}
        corpus.manifests[str(run_dir)] = manifest
        for record in load_records(graded, dedupe=True):
            config = record.get("elicitation")
            if not config:
                n_skipped_no_config += 1
                if require_elicitation:
                    continue
            cell = cell_from_record(record)
            if cell.n == 0 and not cell.errored:
                corpus.warnings.append(
                    f"{run_dir.name}: cell {cell.key()} has no graded samples"
                )
                continue
            if cell.strategy in AUX_STRATEGIES:
                corpus.aux[cell.aux_key()] = cell
                continue
            existing = corpus.cells.get(cell.key())
            if existing is not None and existing.n != cell.n:
                corpus.warnings.append(
                    f"duplicate cell {cell.key()} with differing sample counts "
                    f"({existing.n} vs {cell.n}); keeping the larger"
                )
                if existing.n >= cell.n:
                    continue
            corpus.cells[cell.key()] = cell
    if n_skipped_no_config:
        message = (
            f"skipped {n_skipped_no_config} record(s) with no elicitation "
            "configuration; they cannot enter the crossed design"
        )
        log.warning("%s", message)
        corpus.warnings.append(message)
    log.info(
        "corpus: %d cells (+%d auxiliary single-pass), %d samples, %d model(s), "
        "%d config(s), benchmarks %s",
        len(corpus),
        len(corpus.aux),
        sum(c.n for c in corpus.cells.values()),
        len(corpus.models()),
        len(corpus.configs(include_separate_arms=True)),
        corpus.benchmarks(include_separate_arms=True),
    )
    return corpus


def _graded_path(run_dir: Path, grade: bool) -> Optional[Path]:
    for name in ("graded.jsonl", "graded.jsonl.gz"):
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    for name in ("results.jsonl", "results.jsonl.gz"):
        raw = run_dir / name
        if not raw.exists():
            continue
        if not grade:
            log.warning(
                "%s has %s but no graded file; skipping (pass --grade, or run "
                "`python -m src.grading --run-dir %s` -- it is CPU-only)",
                run_dir,
                name,
                run_dir,
            )
            return None
        from ..grading import grade_file

        target = run_dir / ("graded.jsonl.gz" if name.endswith(".gz") else "graded.jsonl")
        grade_file(raw, target)
        return target
    return None


# ----------------------------------------------------------------- corpus summary
def corpus_summary(corpus: Corpus) -> Dict[str, Any]:
    """Coverage and health of the corpus; the backing numbers for `tab1_setup`."""
    cells = corpus.all_cells()
    if not cells:
        return {"n_cells": 0, "n_samples": 0}
    n_saturated = sum(1 for c in cells if c.saturated)
    failures = sum(c.n_extraction_failures for c in cells)
    total = sum(c.n for c in cells)
    per_cell_failure = [c.extraction_failure_rate for c in cells]
    return {
        "n_cells": len(cells),
        "n_samples": total,
        "n_items": len({c.item_id for c in cells}),
        "n_models": len(corpus.models()),
        "n_configs": len(corpus.configs(include_separate_arms=True)),
        "benchmarks": [f"{d}/{s}" if s else str(d) for d, s in
                       corpus.benchmarks(include_separate_arms=True)],
        "sample_counts": sorted({c.n for c in cells}),
        "tokens_completion": sum(c.tokens_completion for c in cells),
        "saturated_cell_rate": n_saturated / len(cells),
        "extraction_failure_rate": failures / total if total else 0.0,
        "max_cell_extraction_failure_rate": max(per_cell_failure) if per_cell_failure else 0.0,
        # METHOD_SPEC section 4.3: flag any cell above 5%, because an inflated
        # failure class biases the modal ceiling downward.
        "n_cells_over_5pct_failures": sum(1 for r in per_cell_failure if r > 0.05),
        "modal_tie_rate": sum(1 for c in cells if c.modal_tie()) / len(cells),
        "warnings": list(corpus.warnings),
    }


# ------------------------------------------------------- spec-shaped Parquet view
def export_parquet(corpus: Corpus, out_dir: str | Path) -> List[Path]:
    """Write `runs/{model}/{dataset}/{config}.parquet`, the METHOD_SPEC 4.4 layout.

    A derived view, not the source of truth (see the module docstring). Returns an
    empty list with a warning when pyarrow is unavailable, because losing an
    optional export must never fail an analysis run.
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception:  # noqa: BLE001 - optional dependency
        log.warning(
            "pyarrow is not installed, so the Parquet view of the corpus was not "
            "written. The JSONL corpus is the source of truth and every analysis "
            "reads it directly; install pyarrow only if an external tool needs "
            "Parquet."
        )
        return []

    out_dir = Path(out_dir)
    grouped: Dict[Tuple[str, str, str], List[SampleRow]] = defaultdict(list)
    for row in corpus.samples():
        dataset = f"{row.dataset}-{row.subset}" if row.subset else row.dataset
        grouped[(row.model, dataset, row.config)].append(row)

    written: List[Path] = []
    for (model, dataset, config), rows in sorted(grouped.items()):
        safe_model = model.replace("/", "__")
        path = out_dir / safe_model / dataset / f"{config}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.table(
            {
                "item_id": [r.item_id for r in rows],
                "model": [r.model for r in rows],
                "config": [r.config for r in rows],
                "seed": [r.seed for r in rows],
                "sample_idx": [r.sample_idx for r in rows],
                "extracted_answer": [r.answer for r in rows],
                "canonical_class": [r.canonical_class for r in rows],
                "is_correct": [r.is_correct for r in rows],
                "n_output_tokens": [r.tokens_completion for r in rows],
                "mean_logprob": [r.mean_logprob for r in rows],
            }
        )
        pq.write_table(table, path)
        written.append(path)
    log.info("wrote %d Parquet file(s) under %s", len(written), out_dir)
    return written


def iter_cells(cells: Iterable[Cell]) -> Iterable[Cell]:
    """Stable iteration order, so every artefact is byte-reproducible."""
    return sorted(cells, key=lambda c: (c.dataset, str(c.subset), c.model, c.config, c.seed, c.item_id))
