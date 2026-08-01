#!/usr/bin/env python3
"""CPU-only validation before enabling a Kaggle GPU (costs zero quota).

Loads every dataset in the experiment matrix, checks schema and splits,
exercises answer extraction/grading on a handful of gold items, and prints
a pass/fail table. Run with the accelerator OFF.

    python scripts/validate_cpu.py
    python scripts/validate_cpu.py --datasets gsm8k math500 gsm_symbolic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.answers import extract_answer, grade_answer  # noqa: E402
from src.datasets_ import DATASET_REGISTRY, load_dataset_examples  # noqa: E402


DEFAULT_DATASETS = [
    ("gsm8k", "test", None, 5),
    ("math500", "test", None, 5),
    ("aime", "train", None, 3),
    ("bbh", "test", "logical_deduction_three_objects", 3),
    ("musr", "murder_mysteries", None, 3),
    ("arc", "test", None, 5),
    ("gpqa", "train", None, 3),
    ("gsm_symbolic", "test", "main", 3),
]


def validate_one(name: str, split: str, subset: str | None, n: int) -> dict:
    row = {"dataset": name, "split": split, "subset": subset or "", "status": "FAIL"}
    try:
        if name not in DATASET_REGISTRY:
            row["error"] = "not in DATASET_REGISTRY"
            return row
        examples = load_dataset_examples(
            name, split=split, subset=subset, subsample=n, subsample_seed=0
        )
        if not examples:
            row["error"] = "zero examples loaded"
            return row
        ex = examples[0]
        for field in ("id", "question", "gold_answer"):
            if not getattr(ex, field, None):
                row["error"] = f"missing {field}"
                return row
        graded = 0
        for e in examples[: min(3, len(examples))]:
            pred = extract_answer(e.gold_answer, e)
            if grade_answer(pred, e.gold_answer, e.answer_type):
                graded += 1
        row.update(
            {
                "status": "PASS",
                "n_loaded": len(examples),
                "answer_type": ex.answer_type,
                "gold_graded": f"{graded}/{min(3, len(examples))}",
                "sample_id": ex.id[:60],
            }
        )
    except Exception as exc:
        row["error"] = str(exc)[:200]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="dataset names only (uses default split/subset)",
    )
    args = parser.parse_args()
    targets = DEFAULT_DATASETS
    if args.datasets:
        names = set(args.datasets)
        targets = [t for t in DEFAULT_DATASETS if t[0] in names]

    print("CPU validation (accelerator should be OFF — this uses no GPU quota)\n")
    fails = 0
    for name, split, subset, n in targets:
        row = validate_one(name, split, subset, n)
        tag = row["status"]
        if tag != "PASS":
            fails += 1
        detail = row.get("error") or (
            f"n={row.get('n_loaded')} type={row.get('answer_type')} "
            f"gold_ok={row.get('gold_graded')} id={row.get('sample_id')}"
        )
        print(f"  [{tag:4}] {name}/{split}/{subset or '-'}: {detail}")
    print()
    if fails:
        print(f"FAILED: {fails} dataset(s). Fix loaders before enabling GPU.")
        return 1
    print("All datasets OK. Next: python -m src.runner --config configs/smoke_mock.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
