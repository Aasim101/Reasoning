"""The grading pass: cached, re-runnable, and completely separate from inference.

This separation is the single biggest quota saver in the project. Raw generations
in `results.jsonl` are the expensive artefact; whether a given string counts as
the right answer is a cheap CPU judgement that we expect to revise. So grading
reads `results.jsonl` (never modifies it) and writes `graded.jsonl`, keyed by
uid + `GRADER_VERSION`. Improve `src/answers.py`, bump the version, re-run this,
and every number in the paper updates without touching a GPU.

Every trace is graded, not just the strategy's chosen answer, because per-sample
correctness is what `pass@k` and the majority-vote-versus-k curves are computed
from - and re-deriving it later would otherwise require re-running inference.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .answers import (
    GRADER_VERSION,
    answers_equivalent,
    extract_answer,
    grader_backend_name,
    majority_vote,
    normalize_answer,
)
from .checkpointing import JsonlAppender, load_records
from .utils import iter_jsonl, setup_logging, utc_now_iso

log = logging.getLogger(__name__)

#: Keys added to a raw record by grading.
GRADED_KEYS = (
    "is_correct",
    "pred_answer",
    "pred_normalized",
    "gold_normalized",
    "sample_answers",
    "sample_correct",
    "n_samples_graded",
    "n_correct_samples",
    "vote_answer",
    "vote_correct",
    "grader_version",
    "grader_backend",
    "graded_at",
)


def grade_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Grade one raw record. Never raises: a grader crash must not lose a run."""
    out = dict(record)
    gold = str(record.get("gold_answer", ""))
    answer_type = str(record.get("answer_type") or "math")
    choices = record.get("choices") or (record.get("extra") or {}).get("choices")
    traces = record.get("reasoning_traces") or []
    if not isinstance(traces, list):
        traces = [str(traces)]

    # An example that errored is incorrect, not missing: dropping it would
    # silently inflate accuracy over the examples that happened to succeed.
    errored = bool(record.get("error"))

    final = record.get("final_answer")
    pred = final if final is not None else None
    if pred is None and traces and not errored and str(
        (record.get("extra") or {}).get("trace_policy") or "full"
    ).lower() == "full":
        # The strategy failed to name an answer; fall back to the last trace so a
        # parsing bug in a strategy does not masquerade as a wrong answer.
        pred = extract_answer(str(traces[-1]), answer_type, choices)

    # Under a trace-shrinking policy the raw chains can no longer be re-parsed, so
    # the per-sample answers extracted at generation time are authoritative. They
    # are what pass@k, the modal ceiling and the whole mode-reordering analysis are
    # computed from, so they are recorded before the text is ever discarded.
    extra = record.get("extra") or {}
    stored_answers = extra.get("sample_answers")
    policy = str(extra.get("trace_policy") or "full").lower()
    reparse = policy == "full" or not isinstance(stored_answers, list)
    if reparse:
        sample_answers: List[Optional[str]] = [
            extract_answer(str(trace), answer_type, choices) for trace in traces
        ]
    else:
        sample_answers = [None if a is None else str(a) for a in stored_answers]
    sample_correct: List[bool] = [
        bool(ans is not None and answers_equivalent(ans, gold, answer_type, choices))
        for ans in sample_answers
    ]

    vote_answer, _vote_info = majority_vote(sample_answers, answer_type, choices)

    is_correct = bool(
        not errored
        and pred is not None
        and answers_equivalent(str(pred), gold, answer_type, choices)
    )

    out.update(
        {
            "is_correct": is_correct,
            "pred_answer": None if pred is None else str(pred),
            "pred_normalized": (
                None if pred is None else normalize_answer(str(pred), answer_type)
            ),
            "gold_normalized": normalize_answer(gold, answer_type),
            "sample_answers": sample_answers,
            "sample_correct": sample_correct,
            "n_samples_graded": len(sample_answers),
            "n_correct_samples": sum(sample_correct),
            "vote_answer": vote_answer,
            "vote_correct": bool(
                vote_answer is not None
                and answers_equivalent(vote_answer, gold, answer_type, choices)
            ),
            "grader_version": GRADER_VERSION,
            "grader_backend": grader_backend_name(),
            "graded_at": utc_now_iso(),
        }
    )
    return out


def grade_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [grade_record(r) for r in records]


def _already_graded(graded_path: Path) -> Dict[str, str]:
    """uid -> grader_version for records already present."""
    out: Dict[str, str] = {}
    for row in iter_jsonl(graded_path, tolerant=True):
        uid = row.get("uid")
        if uid:
            out[uid] = str(row.get("grader_version", ""))
    return out


def grade_file(
    results_path: str | Path,
    graded_path: Optional[str | Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Grade `results.jsonl` into `graded.jsonl`, skipping up-to-date records.

    Returns a summary dict. Records graded by an older `GRADER_VERSION` are
    regraded automatically, which is why bumping that constant is the correct way
    to roll out a grader fix.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"no results file at {results_path}")
    graded_path = Path(graded_path) if graded_path else results_path.with_name("graded.jsonl")

    records = load_records(results_path, dedupe=True)
    existing = {} if force else _already_graded(graded_path)
    if force and graded_path.exists():
        # Rewriting is safe here: graded.jsonl is derived data, reproducible from
        # results.jsonl at any time. results.jsonl itself is never rewritten.
        graded_path.unlink()

    stale = [uid for uid, ver in existing.items() if ver != GRADER_VERSION]
    if stale:
        log.info(
            "%d record(s) were graded by an older grader (%s -> %s); regrading them",
            len(stale),
            sorted({existing[u] for u in stale}),
            GRADER_VERSION,
        )

    n_graded_now = 0
    n_skipped = 0
    graded_rows: List[Dict[str, Any]] = []
    with JsonlAppender(graded_path, flush_every=50) as writer:
        for record in records:
            uid = record.get("uid")
            if uid and existing.get(uid) == GRADER_VERSION:
                n_skipped += 1
                continue
            graded = grade_record(record)
            writer.write(graded)
            graded_rows.append(graded)
            n_graded_now += 1

    all_rows = load_records(graded_path, dedupe=True)
    n_total = len(all_rows)
    n_correct = sum(1 for r in all_rows if r.get("is_correct"))
    n_errors = sum(1 for r in all_rows if r.get("error"))
    n_unparsed = sum(1 for r in all_rows if r.get("pred_answer") is None)
    summary = {
        "results_path": str(results_path),
        "graded_path": str(graded_path),
        "n_total": n_total,
        "n_graded_now": n_graded_now,
        "n_skipped": n_skipped,
        "n_correct": n_correct,
        "accuracy": round(n_correct / n_total, 6) if n_total else 0.0,
        "vote_accuracy": (
            round(sum(1 for r in all_rows if r.get("vote_correct")) / n_total, 6)
            if n_total
            else 0.0
        ),
        "n_errors": n_errors,
        "n_extraction_failures": n_unparsed,
        "extraction_failure_rate": round(n_unparsed / n_total, 6) if n_total else 0.0,
        "grader_version": GRADER_VERSION,
        "grader_backend": grader_backend_name(),
    }
    log.info(
        "graded %s: %d new, %d cached, accuracy %.3f (%d/%d), extraction failures %.1f%%",
        results_path,
        n_graded_now,
        n_skipped,
        summary["accuracy"],
        n_correct,
        n_total,
        100 * summary["extraction_failure_rate"],
    )
    if summary["extraction_failure_rate"] > 0.05:
        log.warning(
            "extraction failure rate is %.1f%% (> 5%%): the answer extractor is "
            "probably missing a format this model uses, which biases accuracy "
            "DOWNWARD. Inspect a few traces before trusting these numbers.",
            100 * summary["extraction_failure_rate"],
        )
    return summary


def load_graded(graded_path: str | Path) -> List[Dict[str, Any]]:
    return load_records(graded_path, dedupe=True)


def find_results_files(root: str | Path) -> List[Path]:
    """Every results.jsonl under a results root, sorted for stable output."""
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.rglob("results.jsonl"))


# ------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.grading",
        description=(
            "Grade raw generations. Cheap, cached and CPU-only: fix src/answers.py, "
            "bump GRADER_VERSION, re-run this instead of re-running inference."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--results", type=str, help="path to a single results.jsonl")
    src.add_argument("--run-dir", type=str, help="a run directory containing results.jsonl")
    src.add_argument(
        "--results-root",
        type=str,
        help="a results tree; every results.jsonl beneath it is graded",
    )
    p.add_argument("--graded", type=str, default=None, help="output path (single-file mode)")
    p.add_argument(
        "--force",
        action="store_true",
        help="regrade everything, ignoring the cache",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)

    if args.results:
        targets = [Path(args.results)]
    elif args.run_dir:
        targets = [Path(args.run_dir) / "results.jsonl"]
    else:
        targets = find_results_files(args.results_root)

    if not targets:
        log.error("no results.jsonl found")
        return 1

    failures = 0
    totals = {"n_total": 0, "n_correct": 0, "n_graded_now": 0}
    for path in targets:
        try:
            summary = grade_file(
                path,
                args.graded if (args.results and args.graded) else None,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001 - keep grading the rest of the tree
            log.error("failed to grade %s: %s", path, exc)
            failures += 1
            continue
        for key in totals:
            totals[key] += summary[key]
        print(
            f"{summary['accuracy']:.3f}  {summary['n_correct']:>4}/{summary['n_total']:<4} "
            f"(+{summary['n_graded_now']} new, {summary['n_skipped']} cached, "
            f"{100 * summary['extraction_failure_rate']:.1f}% unparsed)  "
            f"{path.parent.name}"
        )

    if len(targets) > 1 and totals["n_total"]:
        print(
            f"\ntotal: {totals['n_correct']}/{totals['n_total']} = "
            f"{totals['n_correct'] / totals['n_total']:.3f} across {len(targets)} run(s), "
            f"{totals['n_graded_now']} newly graded, grader {GRADER_VERSION}/"
            f"{grader_backend_name()}"
        )
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
