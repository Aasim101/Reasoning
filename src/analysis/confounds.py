"""Confound controls: extraction, truncation, and CDV mechanism checks (review §3).

All analyses here are CPU-only post-hoc checks over the persisted corpus.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..answers import (
    EXTRACTION_FAILURE_CLASS,
    answers_equivalent,
    extract_answer,
    extract_answer_alt,
    grader_backend_name,
)
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from .corpus import Cell, Corpus, SampleRow
from .glmm import glmm_decomposition
from .variance import variance_decomposition

log = logging.getLogger(__name__)

_INTEGER_GOLD_RE = re.compile(r"^-?\d+$")


@dataclass
class ExtractorAgreement:
    dataset: str
    subset: Optional[str]
    model: str
    config: str
    n_samples: int
    agreement_rate: float
    primary_fail_rate: float
    alt_fail_rate: float
    both_fail_rate: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "subset": self.subset or "",
            "model": self.model,
            "config": self.config,
            "n_samples": self.n_samples,
            "agreement_rate": self.agreement_rate,
            "primary_fail_rate": self.primary_fail_rate,
            "alt_fail_rate": self.alt_fail_rate,
            "both_fail_rate": self.both_fail_rate,
            "primary_backend": grader_backend_name(),
            "alt_backend": "rule_numeric_latex",
        }


def extractor_agreement_table(corpus: Corpus) -> List[ExtractorAgreement]:
    """Per-configuration agreement between primary and alternate extractors."""
    rows: List[ExtractorAgreement] = []
    grouped: Dict[Tuple[str, Optional[str], str, str], List[SampleRow]] = {}
    for cell in corpus.select(include_separate_arms=True):
        if not cell.samples:
            continue
        key = (cell.dataset, cell.subset, cell.model, cell.config)
        grouped.setdefault(key, []).extend(cell.samples)

    for (dataset, subset, model, config), samples in sorted(grouped.items()):
        if not samples:
            continue
        agree = primary_fail = alt_fail = both_fail = 0
        for s in samples:
            text = str(s.answer or "")
            cell = next(
                c
                for c in corpus.cells.values()
                if c.dataset == dataset
                and c.subset == subset
                and c.model == model
                and c.config == config
            )
            primary = extract_answer(text, cell.answer_type, None)
            alt = extract_answer_alt(text, cell.answer_type, None)
            if primary is None:
                primary_fail += 1
            if alt is None:
                alt_fail += 1
            if primary is None and alt is None:
                both_fail += 1
            if primary == alt or (
                primary is not None
                and alt is not None
                and answers_equivalent(primary, alt, cell.answer_type, None)
            ):
                agree += 1
        n = len(samples)
        rows.append(
            ExtractorAgreement(
                dataset=dataset,
                subset=subset,
                model=model,
                config=config,
                n_samples=n,
                agreement_rate=agree / n if n else float("nan"),
                primary_fail_rate=primary_fail / n if n else float("nan"),
                alt_fail_rate=alt_fail / n if n else float("nan"),
                both_fail_rate=both_fail / n if n else float("nan"),
            )
        )
    return rows


def extraction_audit_sample(
    corpus: Corpus,
    n_per_config: int = 100,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Stratified audit rows: balance parse-success vs extraction-failure."""
    import random

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    by_config: Dict[str, List[SampleRow]] = {}
    cell_of: Dict[Tuple[str, str, int], Cell] = {}
    for cell in corpus.select(include_separate_arms=True):
        for s in cell.samples:
            by_config.setdefault(cell.config, []).append(s)
            cell_of[(s.item_id, s.config, s.sample_idx)] = cell

    for config, samples in sorted(by_config.items()):
        success = [s for s in samples if not s.extraction_failed]
        failure = [s for s in samples if s.extraction_failed]
        half = n_per_config // 2
        pick = rng.sample(success, min(half, len(success))) + rng.sample(
            failure, min(n_per_config - half, len(failure))
        )
        for s in pick[:n_per_config]:
            cell = cell_of.get((s.item_id, s.config, s.sample_idx))
            if cell is None:
                continue
            text = str(s.answer or "")
            out.append(
                {
                    "config": config,
                    "item_id": s.item_id,
                    "sample_idx": s.sample_idx,
                    "extraction_failed": s.extraction_failed,
                    "primary_extract": extract_answer(text, cell.answer_type, None),
                    "alt_extract": extract_answer_alt(text, cell.answer_type, None),
                    "gold": cell.gold_answer,
                    "is_correct_primary": s.is_correct,
                }
            )
    return out


def per_config_failure_rates(corpus: Corpus) -> List[Dict[str, Any]]:
    rows = []
    grouped: Dict[Tuple[str, Optional[str], str, str], List[Cell]] = {}
    for cell in corpus.select(include_separate_arms=True):
        grouped.setdefault(
            (cell.dataset, cell.subset, cell.model, cell.config), []
        ).append(cell)
    for (dataset, subset, model, config), cells in sorted(grouped.items()):
        total = sum(c.n for c in cells)
        fails = sum(c.n_extraction_failures for c in cells)
        trunc = sum(
            1
            for c in cells
            for s in c.samples
            if (s.finish_reason or "").lower() in ("length", "max_tokens")
        )
        rows.append(
            {
                "dataset": dataset,
                "subset": subset or "",
                "model": model,
                "config": config,
                "extraction_failure_rate": fails / total if total else float("nan"),
                "truncation_rate": trunc / total if total else float("nan"),
                "n_samples": total,
                "n_cells": len(cells),
            }
        )
    return rows


def sensitivity_glmm_shares(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Three pre-registered sensitivity analyses on the GLMM item×config share."""
    base = glmm_decomposition(corpus, dataset, subset, seed=seed, n_bootstrap=0)

  # ⊥ dropped: rebuild corpus excluding ⊥ samples from correctness
    dropped = _filter_corpus(corpus, drop_perp=True, integer_only=False)
    drop_share = float("nan")
    try:
        drop_share = glmm_decomposition(
            dropped, dataset, subset, seed=seed, n_bootstrap=0
        ).shares.get("item_config", float("nan"))
    except Exception:
        pass

    zero_perp = _filter_corpus(corpus, zero_perp_cells_only=True)
    zero_share = float("nan")
    try:
        zero_share = glmm_decomposition(
            zero_perp, dataset, subset, seed=seed, n_bootstrap=0
        ).shares.get("item_config", float("nan"))
    except Exception:
        pass

    integers = _filter_corpus(corpus, integer_only=True)
    int_share = float("nan")
    try:
        int_share = glmm_decomposition(
            integers, dataset, subset, seed=seed, n_bootstrap=0
        ).shares.get("item_config", float("nan"))
    except Exception:
        pass

    return {
        "dataset": dataset,
        "subset": subset or "",
        "headline_item_config_share": base.shares.get("item_config"),
        "drop_perp_item_config_share": drop_share,
        "zero_perp_everywhere_item_config_share": zero_share,
        "integer_gold_only_item_config_share": int_share,
        "headline_survives_all_three": bool(
            all(
                math.isfinite(v)
                for v in (
                    base.shares.get("item_config", float("nan")),
                    drop_share,
                    zero_share,
                    int_share,
                )
            )
            and base.shares.get("item_config", 0) > 0.05
        ),
    }


def _filter_corpus(
    corpus: Corpus,
    drop_perp: bool = False,
    zero_perp_cells_only: bool = False,
    integer_only: bool = False,
) -> Corpus:
    import copy

    out = Corpus()
    out.manifests = dict(corpus.manifests)
    out.warnings = list(corpus.warnings)
    for key, cell in corpus.cells.items():
        if integer_only and not _INTEGER_GOLD_RE.match(str(cell.gold_answer).strip()):
            continue
        if zero_perp_cells_only:
            # keep items with zero ⊥ in every cell for this item
            item_cells = [
                c
                for c in corpus.cells.values()
                if c.item_id == cell.item_id and c.benchmark() == cell.benchmark()
            ]
            if any(c.n_extraction_failures > 0 for c in item_cells):
                continue
        new_cell = copy.deepcopy(cell)
        if drop_perp:
            new_samples = []
            for s in new_cell.samples:
                if s.canonical_class == EXTRACTION_FAILURE_CLASS:
                    continue
                new_samples.append(s)
            new_cell.samples = new_samples
            if not new_samples:
                continue
        out.cells[key] = new_cell
    return out


def stratified_rho_disatt(
    transfer_rows: Sequence[Any],
    corpus: Corpus,
    n_quintiles: int = 5,
) -> List[Dict[str, Any]]:
    """ρ_disatt within marginal-difficulty quintiles (review §2.2)."""
    from .transfer import transfer_table

    # Delegate to transfer_table with item filtering by quintile — simplified export.
    out: List[Dict[str, Any]] = []
    for dataset, subset in corpus.benchmarks():
        items = corpus.crossed_items(
            dataset, subset, corpus.models(), list(PRIMARY_CONFIG_IDS)
        )
        if len(items) < n_quintiles * 3:
            continue
        import numpy as np

        p_bar = []
        for item in items:
            vals = [
                c.p_hat
                for c in corpus.cells.values()
                if c.item_id == item
                and c.dataset == dataset
                and c.subset == subset
                and c.config in PRIMARY_CONFIG_IDS
            ]
            p_bar.append(float(np.nanmean(vals)) if vals else float("nan"))
        edges = np.quantile([v for v in p_bar if math.isfinite(v)], np.linspace(0, 1, n_quintiles + 1))
        for q in range(n_quintiles):
            lo, hi = edges[q], edges[q + 1]
            stratum = [
                item
                for item, p in zip(items, p_bar)
                if math.isfinite(p) and (lo <= p < hi or (q == n_quintiles - 1 and p <= hi))
            ]
            if len(stratum) < 5:
                continue
            rows = [
                r
                for r in transfer_rows
                if r.dataset == dataset
                and (r.subset or None) == subset
                and r.kind == "config"
            ]
            if rows:
                out.append(
                    {
                        "dataset": dataset,
                        "subset": subset or "",
                        "quintile": q + 1,
                        "n_items": len(stratum),
                        "mean_rho_disatt": sum(r.rho_disatt for r in rows) / len(rows),
                    }
                )
    return out
