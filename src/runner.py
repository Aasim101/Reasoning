"""The experiment driver: resumable, time-budgeted, one example at a time.

Operational contract (this is what makes a 9-12h Kaggle session survivable):

* Work unit = one example. Nothing is ever partially recorded.
* Before each example the wall-clock guard is consulted; when the remaining
  budget cannot safely cover another example the loop exits cleanly, state is
  written, and a copy-pasteable resume command is printed.
* Records are appended to `results.jsonl` and fsync'd every
  `runtime.flush_every` examples. Re-running the identical command skips
  everything already present, so `python -m src.runner --config X` is both
  "start" and "resume".
* SIGTERM/SIGINT are caught and turned into a clean stop at the next example
  boundary, so a manually stopped notebook still leaves a consistent run.
* Grading is *not* part of the loop; it runs afterwards (or later, offline) so a
  grader fix never costs GPU time.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .budget import TimeGuard
from .checkpointing import (
    JsonlAppender,
    RunState,
    build_resume_command,
    completed_uids,
    make_uid,
)
from .config import RunConfig, add_config_args, config_from_args, load_config
from .generation import GenerationBackend
from .types import Example, GenParams, ResultRecord, StrategyResult
from .utils import human_time, seed_everything, setup_logging, truncate, utc_now_iso

log = logging.getLogger(__name__)


@dataclass
class RunSummary:
    """What one session accomplished; also printed as the closing report."""

    run_dir: Path
    config_hash: str
    n_total: int
    n_completed_total: int
    n_completed_this_session: int
    n_errors_this_session: int
    n_remaining: int
    stop_reason: str
    elapsed_seconds: float
    gpu_hours_this_session: float
    tokens_prompt: int
    tokens_completion: int
    resume_command: str
    grading: Optional[Dict[str, Any]] = None
    timing: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.n_remaining == 0

    def report(self) -> str:
        lines = [
            "",
            "=" * 78,
            f"RUN {'COMPLETE' if self.is_complete else 'INCOMPLETE'}  "
            f"({self.stop_reason})",
            "=" * 78,
            f"  run dir            : {self.run_dir}",
            f"  config hash        : {self.config_hash}",
            f"  progress           : {self.n_completed_total}/{self.n_total} examples "
            f"({self.n_remaining} remaining)",
            f"  this session       : {self.n_completed_this_session} done, "
            f"{self.n_errors_this_session} errors, {human_time(self.elapsed_seconds)} "
            f"wall-clock",
            f"  gpu-hours (session): {self.gpu_hours_this_session:.2f}",
            f"  tokens             : {self.tokens_prompt:,} prompt / "
            f"{self.tokens_completion:,} completion",
        ]
        if self.timing.get("mean_seconds_per_example") is not None:
            lines.append(
                f"  throughput         : "
                f"{self.timing['mean_seconds_per_example']:.2f} s/example"
                + (
                    f", eta for remainder {self.timing['eta_remaining_human']}"
                    if self.timing.get("eta_remaining_human")
                    else ""
                )
            )
        if self.timing.get("sessions_needed"):
            lines.append(
                f"  sessions needed    : ~{self.timing['sessions_needed']:.1f} more "
                f"at this budget"
            )
        if self.grading:
            lines.append(
                f"  graded accuracy    : {self.grading.get('accuracy')} "
                f"({self.grading.get('n_correct')}/{self.grading.get('n_total')}, "
                f"grader {self.grading.get('grader_version')}/"
                f"{self.grading.get('grader_backend')})"
            )
        if self.is_complete:
            lines += [
                "",
                "  Nothing left to do. Regenerate figures/tables with:",
                "    python -m src.analysis.aggregate --results-dir "
                f"{self.run_dir.parent} --out-dir paper_assets --figures --tables",
            ]
        else:
            lines += [
                "",
                "  RESUME IN THE NEXT SESSION WITH EXACTLY THIS COMMAND:",
                f"    {self.resume_command}",
                "",
                "  (Completed examples are skipped automatically. First download this",
                "   run directory from /kaggle/working, then re-attach it as a Kaggle",
                "   Dataset input and point --set runtime.out_dir at it.)",
            ]
        lines.append("=" * 78)
        return "\n".join(lines)


class _StopSignal:
    """Turns SIGINT/SIGTERM into a cooperative stop at an example boundary."""

    def __init__(self) -> None:
        self.requested = False
        self.reason = ""
        self._previous: Dict[int, Any] = {}

    def install(self) -> "_StopSignal":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError, AttributeError):
                # Not on the main thread, or the platform lacks the signal.
                pass
        return self

    def restore(self) -> None:
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def _handle(self, signum: int, _frame: Any) -> None:
        self.requested = True
        self.reason = f"signal_{signal.Signals(signum).name.lower()}"
        log.warning(
            "received %s; finishing the current example then stopping cleanly "
            "(press again to force-quit)",
            signal.Signals(signum).name,
        )
        # A second signal restores default behaviour so the user can force-quit.
        try:
            signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
        except (ValueError, OSError):
            pass


def plan_examples(cfg: RunConfig) -> Tuple[List[Example], List[str]]:
    """Load the target examples and their deterministic result uids."""
    from .datasets_ import load_dataset_examples

    examples = load_dataset_examples(cfg.data)
    uids = [uid_for(cfg, ex, position) for position, ex in enumerate(examples)]
    return examples, uids


def uid_for(cfg: RunConfig, example: Example, position: int) -> str:
    """Deterministic result key for one measurement.

    The example's *original* dataset row index is preferred over its position in
    the (sub)sample, so the key describes the measurement rather than the
    iteration order.
    """
    index = example.meta.get("orig_index", position)
    return make_uid(
        model=cfg.model.name_or_path,
        strategy=cfg.strategy.name,
        dataset=cfg.data.name,
        index=int(index),
        seed=cfg.runtime.seed,
        config_hash=cfg.config_hash,
    )


def example_seed(cfg: RunConfig, example: Example, position: int) -> int:
    """Per-example sampling seed: deterministic, but different per example."""
    index = int(example.meta.get("orig_index", position))
    return (cfg.runtime.seed * 1_000_003 + index * 7919) % (2**31 - 1)


def run(
    cfg: RunConfig,
    config_path: Optional[str] = None,
    overrides: Sequence[str] = (),
    backend: Optional[GenerationBackend] = None,
    grade: Optional[bool] = None,
) -> RunSummary:
    """Execute (or resume) one experiment cell. Returns a session summary.

    Pass `backend` to reuse an already-loaded model across several runs in one
    notebook session — loading a 7B model twice wastes several minutes of quota.
    """
    setup_logging(cfg.runtime.log_level)
    seed_everything(cfg.runtime.seed)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s", cfg.describe())

    examples, uids = plan_examples(cfg)
    if not examples:
        raise RuntimeError(
            f"dataset {cfg.data.name!r} produced 0 examples; check split/subset/subsample"
        )

    done = completed_uids(cfg.results_path)
    pending: List[Tuple[int, Example, str]] = [
        (i, ex, uid) for i, (ex, uid) in enumerate(zip(examples, uids)) if uid not in done
    ]
    n_total = len(examples)
    n_already = n_total - len(pending)
    if n_already:
        log.info(
            "resuming: %d/%d examples already recorded in %s",
            n_already,
            n_total,
            cfg.results_path,
        )
    cap = cfg.runtime.max_examples
    if cap is not None and len(pending) > cap:
        log.info("--max-examples=%d caps this session (of %d pending)", cap, len(pending))
        pending = pending[:cap]

    # ---------------------------------------------------------------- state
    state = RunState.load(cfg.manifest_path) or RunState(
        run_name=cfg.run_name, config_hash=cfg.config_hash, config=cfg.to_dict()
    )
    drift = state.config_drift(cfg.to_dict())
    if drift:
        log.warning(
            "config differs from the stored manifest in non-semantic fields: %s "
            "(results remain compatible; recorded in the manifest)",
            ", ".join(drift[:12]),
        )
    state.config = cfg.to_dict()
    state.run_name = cfg.run_name
    state.config_hash = cfg.config_hash
    state.n_total = n_total
    state.n_completed = n_already
    resume_command = build_resume_command(config_path, overrides)
    state.resume_command = resume_command

    # -------------------------------------------------------------- backend
    from .models import build_backend, log_hardware

    hardware = log_hardware()
    owns_backend = backend is None
    if backend is None:
        backend = build_backend(cfg, cfg.generation, examples=examples)
    elif hasattr(backend, "register_golds"):
        backend.register_golds(examples)  # keeps MockBackend useful when reused

    from .strategies import build_strategy

    # The elicitation configuration composes with the strategy: it supplies the
    # prompt style and the decoding temperature, the strategy keeps deciding how
    # many chains to draw and how to aggregate them.
    strategy = build_strategy(cfg.strategy.name, **cfg.strategy_params())
    log.info("strategy: %s", strategy.describe())
    elicitation_id = cfg.elicitation_id
    if elicitation_id:
        log.info(
            "elicitation configuration %s (axis %s)",
            elicitation_id,
            (cfg.resolved_elicitation() or {}).axis,  # type: ignore[union-attr]
        )

    base_params = cfg.gen_params()
    guard = TimeGuard(
        budget_hours=cfg.runtime.time_budget_hours,
        reserve_minutes=cfg.runtime.reserve_minutes,
        n_gpus=int(getattr(hardware, "n_gpus", 0) or 0),
        gpu_hour_multiplier=cfg.runtime.gpu_hour_multiplier,
    )
    session = state.begin_session(
        hardware=_as_dict(hardware), backend=dict(backend.info)
    )
    state.save(cfg.manifest_path)

    stopper = _StopSignal().install()
    writer = JsonlAppender(cfg.results_path, flush_every=cfg.runtime.flush_every)
    n_done = n_err = 0
    tokens_prompt = tokens_completion = 0
    stop_reason = "completed"
    guard.start()

    try:
        if not pending:
            stop_reason = "nothing_to_do"
            log.info("nothing to do: all %d examples already completed", n_total)
        for position, example, uid in pending:
            if stopper.requested:
                stop_reason = stopper.reason or "interrupted"
                break
            if guard.should_stop():
                stop_reason = "time_budget"
                log.warning(
                    "stopping at an example boundary: %s of budget left, "
                    "next example needs ~%s",
                    human_time(max(0.0, guard.remaining_seconds)),
                    human_time(guard.projected_next_seconds()),
                )
                break

            params = base_params.with_(seed=example_seed(cfg, example, position))
            t0 = time.perf_counter()
            error: Optional[str] = None
            try:
                result = strategy.run(example, backend, params)
            except Exception as exc:  # noqa: BLE001 - one bad example must not kill a session
                if not cfg.runtime.continue_on_error:
                    raise
                error = f"{type(exc).__name__}: {truncate(str(exc), 500)}"
                log.exception("example %s failed", example.id)
                result = StrategyResult(final_answer=None, reasoning_traces=[])
                n_err += 1
            latency = time.perf_counter() - t0

            record = ResultRecord(
                uid=uid,
                example_id=example.id,
                dataset=cfg.data.name,
                model=cfg.model.name_or_path,
                strategy=cfg.strategy.name,
                seed=cfg.runtime.seed,
                config_hash=cfg.config_hash,
                index=int(example.meta.get("orig_index", position)),
                question=example.question,
                gold_answer=example.gold_answer,
                answer_type=example.answer_type,
                final_answer=result.final_answer,
                reasoning_traces=apply_trace_policy(
                    result.reasoning_traces,
                    policy=cfg.runtime.trace_policy,
                    max_chars=cfg.runtime.trace_max_chars,
                ),
                n_samples=result.n_samples,
                tokens_prompt=result.tokens_prompt,
                tokens_completion=result.tokens_completion,
                n_calls=result.n_calls,
                latency_s=round(latency, 3),
                elicitation=elicitation_id,
                subset=cfg.data.subset,
                sample_stats=list(result.sample_stats),
                meta=_analysis_meta(example),
                extra=_with_trace_provenance(
                    result, example, cfg.runtime.trace_policy
                ),
                finished_at=utc_now_iso(),
                error=error,
            )
            writer.write(record)
            guard.record(latency)
            n_done += 1
            tokens_prompt += result.tokens_prompt
            tokens_completion += result.tokens_completion

            if cfg.runtime.log_every and n_done % cfg.runtime.log_every == 0:
                log.info(
                    "%s ans=%s",
                    guard.progress_line(n_already + n_done, n_total),
                    truncate(str(result.final_answer), 40),
                )
            if n_done % max(1, cfg.runtime.flush_every) == 0:
                _checkpoint_state(
                    state,
                    session,
                    guard,
                    cfg,
                    n_completed=n_already + n_done,
                    n_session_done=n_done,
                    n_errors=n_err,
                    tokens_prompt=tokens_prompt,
                    tokens_completion=tokens_completion,
                    status="running",
                )
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        stop_reason = "interrupted"
        log.warning("interrupted; state has been written")
    finally:
        writer.close()
        stopper.restore()
        if owns_backend:
            try:
                backend.close()
            except Exception:  # noqa: BLE001
                log.debug("backend close failed", exc_info=True)

    n_completed_total = n_already + n_done
    session.stop_reason = stop_reason
    _checkpoint_state(
        state,
        session,
        guard,
        cfg,
        n_completed=n_completed_total,
        n_session_done=n_done,
        n_errors=n_err,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        status=("complete" if n_completed_total >= n_total else f"stopped_{stop_reason}"),
    )
    state.end_session(session)
    state.save(cfg.manifest_path)

    grading_summary: Optional[Dict[str, Any]] = None
    should_grade = cfg.runtime.grade_after_run if grade is None else grade
    if should_grade and n_completed_total:
        try:
            from .grading import grade_file

            grading_summary = grade_file(cfg.results_path, cfg.graded_path)
        except Exception:  # noqa: BLE001 - grading is always re-runnable offline
            log.exception(
                "grading pass failed; raw generations are safe in %s. Re-grade later "
                "with: python -m src.grading --run-dir %s",
                cfg.results_path,
                cfg.run_dir,
            )

    summary = RunSummary(
        run_dir=run_dir,
        config_hash=cfg.config_hash,
        n_total=n_total,
        n_completed_total=n_completed_total,
        n_completed_this_session=n_done,
        n_errors_this_session=n_err,
        n_remaining=max(0, n_total - n_completed_total),
        stop_reason=stop_reason,
        elapsed_seconds=guard.elapsed_seconds,
        gpu_hours_this_session=guard.gpu_hours,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        resume_command=resume_command,
        grading=grading_summary,
        timing=guard.summary(n_remaining=max(0, n_total - n_completed_total)),
    )
    print(summary.report())
    return summary


#: Item metadata copied into every result record. Deliberately a whitelist: the
#: analysis needs `template_id` (GSM-Symbolic bootstraps over templates, because
#: instances of one template are not independent) and the tier tag, and copying
#: the whole meta dict into a million records would waste disk on provenance the
#: manifest already holds.
ANALYSIS_META_KEYS = (
    "template_id",
    "tier",
    "subset",
    "domain",
    "level",
    "subject",
    "paraphrased",
    "orig_index",
)


def _analysis_meta(example: Example) -> Dict[str, Any]:
    return {
        k: example.meta[k]
        for k in ANALYSIS_META_KEYS
        if k in example.meta and example.meta[k] is not None
    }


#: Marker inserted where a trace was cut, so a reader can never mistake a
#: truncated chain for a chain the model actually ended there.
TRUNCATION_MARKER = "\n[... trace truncated by runtime.trace_policy ...]\n"

TRACE_POLICIES = ("full", "truncate", "drop")


def apply_trace_policy(
    traces: Sequence[str], policy: str = "full", max_chars: int = 400
) -> List[str]:
    """Shrink stored chains of thought to fit Kaggle's 20 GB working directory.

    The full matrix generates on the order of a million chains, so keeping every
    chain verbatim is a real disk risk (see `src.budget.estimate_disk`). Both ends
    of a chain are kept because both carry information the analysis or a manual
    audit needs: the opening states the approach, the closing carries the final
    answer that extraction reads.

    This only ever touches *text*. Per-sample extracted answers and per-sample
    token counts are stored separately and are never affected, which is why
    truncation cannot change a single number in the paper.
    """
    policy = str(policy or "full").lower()
    if policy not in TRACE_POLICIES:
        raise ValueError(
            f"unknown trace_policy {policy!r}; expected one of {TRACE_POLICIES}"
        )
    if policy == "full":
        return list(traces)
    if policy == "drop":
        return []
    keep = max(0, int(max_chars))
    out: List[str] = []
    for text in traces:
        text = str(text)
        if len(text) <= 2 * keep + len(TRUNCATION_MARKER):
            out.append(text)
        else:
            out.append(text[:keep] + TRUNCATION_MARKER + text[-keep:])
    return out


def _with_trace_provenance(
    result: StrategyResult, example: Example, policy: str
) -> Dict[str, Any]:
    """Ensure per-sample answers survive any trace policy.

    Under `truncate` or `drop` the raw chains can no longer be re-parsed, so the
    per-sample extracted answers must already be in the record. Strategies that
    vote (which is all of the method's strategies) put them in `extra` anyway;
    this back-fills for the ones that do not, so that grading stays possible
    without re-running inference — which is the whole point of separating the
    grading pass from generation.
    """
    extra = dict(result.extra or {})
    policy = str(policy or "full").lower()
    extra["trace_policy"] = policy
    if policy == "full":
        return extra
    if "sample_answers" not in extra and result.reasoning_traces:
        from .answers import extract_answer

        extra["sample_answers"] = [
            extract_answer(str(t), example.answer_type, example.choices)
            for t in result.reasoning_traces
        ]
    return extra


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort dataclass/object to dict, for provenance in the manifest."""
    import dataclasses

    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    for attr in ("as_dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return dict(fn())
    return {"repr": repr(obj)}


def _checkpoint_state(
    state: RunState,
    session: Any,
    guard: TimeGuard,
    cfg: RunConfig,
    n_completed: int,
    n_session_done: int,
    n_errors: int,
    tokens_prompt: int,
    tokens_completion: int,
    status: str,
) -> None:
    """Refresh the manifest so an abrupt kill still leaves accurate progress."""
    session.n_completed = n_session_done
    session.n_errors = n_errors
    session.elapsed_seconds = round(guard.elapsed_seconds, 1)
    session.gpu_hours = round(guard.gpu_hours, 4)
    session.tokens_prompt = tokens_prompt
    session.tokens_completion = tokens_completion
    state.n_completed = n_completed
    state.n_errors = n_errors
    state.status = status
    if state.sessions:
        from dataclasses import asdict

        state.sessions[-1] = asdict(session)
    state.save(cfg.manifest_path)


# ------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.runner",
        description=(
            "Run or resume one experiment cell. Re-running the same command resumes: "
            "completed examples are skipped."
        ),
    )
    add_config_args(p)
    p.add_argument(
        "--no-grade",
        dest="grade",
        action="store_false",
        default=None,
        help="skip the post-run grading pass (grade later with src.grading)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="load the dataset, print the plan and the resume state, then exit",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="print the fully resolved config (including the hash) and exit",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list registered datasets and strategies, then exit",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "log_level", None) or "INFO")

    if args.list:
        from .datasets_ import available_datasets
        from .strategies import available_strategies

        print("datasets  :", ", ".join(available_datasets()))
        print("strategies:", ", ".join(available_strategies()))
        return 0

    cfg = config_from_args(args)
    overrides = _effective_overrides(args)

    if args.print_config:
        resolved = cfg.run_dir / "config.resolved.yaml"
        cfg.save(resolved)
        print(cfg.describe())
        print(f"\nwrote resolved config to {resolved}")
        return 0

    if args.dry_run:
        examples, uids = plan_examples(cfg)
        done = completed_uids(cfg.results_path)
        pending = [u for u in uids if u not in done]
        print(cfg.describe())
        print(
            f"\nplan: {len(examples)} examples, {len(examples) - len(pending)} already "
            f"complete, {len(pending)} pending"
        )
        if examples:
            print(f"first example id={examples[0].id} uid={uids[0]}")
            print(f"  question: {truncate(examples[0].question, 200)}")
            print(f"  gold    : {examples[0].gold_answer}")
        print(f"\nresume command: {build_resume_command(args.config, overrides)}")
        return 0

    summary = run(cfg, config_path=args.config, overrides=overrides, grade=args.grade)
    return 0 if summary.is_complete else 2


def _effective_overrides(args: argparse.Namespace) -> List[str]:
    """The override list to embed in the resume command."""
    from .config import _SHORTHAND_MAP

    out = list(getattr(args, "overrides", []) or [])
    for attr, dotted in _SHORTHAND_MAP.items():
        val = getattr(args, attr, None)
        # max_examples is per-session, never carried into a resume command.
        if val is not None and attr != "max_examples":
            out.append(f"{dotted}={val}")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
