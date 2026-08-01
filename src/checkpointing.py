"""Resumability: deterministic result IDs, append-only JSONL, run manifest.

This is the most important module in the harness. Kaggle sessions are killed
abruptly at 9-12h, so the invariants are:

1. Every (model, strategy, dataset, example index, seed, config) tuple maps to a
   deterministic `uid`. Restarting a run recomputes the same uids.
2. Results are appended to a line-buffered JSONL and fsync'd every N examples,
   so a kill loses at most N-1 examples (and usually zero, because line
   buffering hands each record to the OS immediately).
3. On start, the runner reads the existing JSONL, collects completed uids, and
   skips them. Work is therefore never duplicated and never lost.
4. A JSON manifest records config, progress, cumulative elapsed time and GPU
   hours, per-session history, and a copy-pasteable resume command.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .types import ResultRecord
from .utils import iter_jsonl, read_json, stable_hash, utc_now_iso, write_json_atomic

log = logging.getLogger(__name__)

#: Bump when the *record schema* changes incompatibly (not the grader).
RECORD_SCHEMA_VERSION = 1


def make_uid(
    model: str,
    strategy: str,
    dataset: str,
    index: int,
    seed: int,
    config_hash: str,
    length: int = 16,
) -> str:
    """Deterministic per-example result key.

    Deliberately built from the *semantic* coordinates of one measurement so the
    same uid is produced on any machine, in any session, in any order.
    """
    return stable_hash(
        {
            "model": model,
            "strategy": strategy,
            "dataset": dataset,
            "index": int(index),
            "seed": int(seed),
            "config_hash": config_hash,
            "schema": RECORD_SCHEMA_VERSION,
        },
        length=length,
    )


class JsonlAppender:
    """Append-only writer with configurable fsync cadence.

    The file is opened line-buffered, so each `write` reaches the OS
    immediately; `fsync` every `flush_every` records forces it to physical
    storage. `flush_every=1` is the paranoid setting (slower on Kaggle's disk).

    A `.gz` path is written as gzip. Appending to a gzip file produces a second
    *member* rather than corrupting the first, and the gzip format defines a
    concatenation of members as a valid stream, so resumability is unaffected —
    which is what makes compression a safe answer to the 20 GB `/kaggle/working`
    limit rather than a trade against crash safety. Each record is written as its
    own flushed member so a killed session loses at most the record in flight.
    """

    def __init__(self, path: str | os.PathLike, flush_every: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self.gzipped = self.path.name.endswith(".gz")
        if self.gzipped:
            import gzip
            import io

            self._raw = self.path.open("ab")
            self._fh = io.TextIOWrapper(
                gzip.GzipFile(fileobj=self._raw, mode="ab"),
                encoding="utf-8",
                newline="\n",
            )
            self._since_sync = 0
            self.n_written = 0
            return
        self._raw = None
        needs_newline = self._ends_without_newline()
        self._fh = self.path.open("a", encoding="utf-8", buffering=1, newline="\n")
        if needs_newline:
            # A kill mid-write can leave a partial line with no trailing newline.
            # Appending straight onto it would fuse the torn record with the next
            # one and lose BOTH, so start a fresh line first. The torn fragment is
            # then simply skipped as malformed by the tolerant reader.
            log.warning(
                "%s does not end with a newline (a previous session was probably "
                "killed mid-write); starting a new line before appending",
                self.path,
            )
            self._fh.write("\n")
        self._since_sync = 0
        self.n_written = 0

    def _ends_without_newline(self) -> bool:
        try:
            if not self.path.exists() or self.path.stat().st_size == 0:
                return False
            with self.path.open("rb") as f:
                f.seek(-1, os.SEEK_END)
                return f.read(1) not in (b"\n", b"\r")
        except OSError:  # pragma: no cover - unreadable file
            return False

    def write(self, obj: Dict[str, Any] | ResultRecord) -> None:
        if isinstance(obj, ResultRecord):
            obj = obj.to_dict()
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self.n_written += 1
        self._since_sync += 1
        if self._since_sync >= self.flush_every:
            self.sync()

    def sync(self) -> None:
        """Flush Python + OS buffers to disk."""
        try:
            self._fh.flush()
            if self.gzipped:
                # Close the current gzip member so the bytes on disk are a
                # complete, readable stream, then start a fresh one. Without this a
                # killed session would leave an unterminated deflate stream and
                # lose the whole file rather than just the record in flight.
                self._fh.close()
                self._raw.flush()  # type: ignore[union-attr]
                os.fsync(self._raw.fileno())  # type: ignore[union-attr]
                self._open_gzip_member()
            else:
                os.fsync(self._fh.fileno())
        except (OSError, ValueError):  # closed handle or fs without fsync
            log.debug("fsync failed for %s", self.path, exc_info=True)
        self._since_sync = 0

    def _open_gzip_member(self) -> None:
        import gzip
        import io

        self._fh = io.TextIOWrapper(
            gzip.GzipFile(fileobj=self._raw, mode="ab"), encoding="utf-8", newline="\n"
        )

    def close(self) -> None:
        if self._fh.closed and (self._raw is None or self._raw.closed):
            return
        if not self._fh.closed:
            self.sync()
            self._fh.close()
        if self._raw is not None and not self._raw.closed:
            self._raw.close()

    def __enter__(self) -> "JsonlAppender":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def load_records(
    path: str | os.PathLike, dedupe: bool = True
) -> List[Dict[str, Any]]:
    """Read a results JSONL, tolerating a truncated tail.

    With `dedupe`, later records win for a given uid: re-running an example
    (e.g. after a crash mid-write) is harmless.
    """
    records = list(iter_jsonl(path, tolerant=True))
    if not dedupe:
        return records
    by_uid: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for r in records:
        uid = r.get("uid")
        if uid is None:
            uid = f"__anon_{len(order)}"
            r["uid"] = uid
        if uid not in by_uid:
            order.append(uid)
        by_uid[uid] = r
    return [by_uid[u] for u in order]


def completed_uids(
    path: str | os.PathLike, include_errors: bool = False
) -> Set[str]:
    """uids already present in the results file.

    By default records that captured an exception are *not* treated as complete,
    so a transient failure is retried on the next session. Pass
    `include_errors=True` to treat them as done (useful when a subset of
    examples reliably crashes and you want the run to finish).
    """
    out: Set[str] = set()
    for r in iter_jsonl(path, tolerant=True):
        uid = r.get("uid")
        if not uid:
            continue
        if r.get("error") and not include_errors:
            continue
        out.add(uid)
    return out


@dataclass
class SessionRecord:
    """One Kaggle session's contribution to a (possibly multi-session) run."""

    started_at: str
    ended_at: Optional[str] = None
    n_completed: int = 0
    n_errors: int = 0
    elapsed_seconds: float = 0.0
    gpu_hours: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    stop_reason: str = "unknown"
    hardware: Dict[str, Any] = field(default_factory=dict)
    backend: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    """The manifest: progress + provenance for one run directory."""

    run_name: str = ""
    config_hash: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = RECORD_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    #: Total examples targeted by this run (after subsample/shard/max-examples).
    n_total: int = 0
    n_completed: int = 0
    n_errors: int = 0
    elapsed_seconds: float = 0.0
    gpu_hours: float = 0.0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    #: "running" | "complete" | "stopped_time_budget" | "stopped_max_examples"
    #: | "interrupted" | "error"
    status: str = "new"
    resume_command: str = ""
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | os.PathLike) -> Optional["RunState"]:
        raw = read_json(path, default=None)
        if not isinstance(raw, dict):
            return None
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | os.PathLike) -> None:
        self.updated_at = utc_now_iso()
        write_json_atomic(path, asdict(self))

    # --------------------------------------------------------------- sessions
    def begin_session(
        self, hardware: Optional[Dict[str, Any]] = None, backend: Optional[Dict[str, Any]] = None
    ) -> SessionRecord:
        sess = SessionRecord(
            started_at=utc_now_iso(), hardware=hardware or {}, backend=backend or {}
        )
        self.sessions.append(asdict(sess))
        self.status = "running"
        return sess

    def end_session(self, sess: SessionRecord) -> None:
        sess.ended_at = utc_now_iso()
        if self.sessions:
            self.sessions[-1] = asdict(sess)
        self.elapsed_seconds += sess.elapsed_seconds
        self.gpu_hours += sess.gpu_hours
        self.tokens_prompt += sess.tokens_prompt
        self.tokens_completion += sess.tokens_completion

    @property
    def n_remaining(self) -> int:
        return max(0, self.n_total - self.n_completed)

    @property
    def is_complete(self) -> bool:
        return self.n_total > 0 and self.n_completed >= self.n_total

    def config_drift(self, current: Dict[str, Any]) -> List[str]:
        """Dotted paths where the current config differs from the stored one.

        Semantic drift is impossible (it changes `config_hash`, hence the run
        directory), so anything reported here is a placement/perf knob. Surfaced
        as a warning for reproducibility bookkeeping.
        """
        drift: List[str] = []

        def walk(a: Any, b: Any, prefix: str = "") -> None:
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    walk(a.get(k), b.get(k), f"{prefix}{k}." if not prefix else f"{prefix}{k}.")
                return
            if a != b:
                drift.append(prefix.rstrip("."))

        walk(self.config, current)
        return [d for d in drift if d]


def build_resume_command(
    config_path: Optional[str],
    overrides: Sequence[str] = (),
    module: str = "src.runner",
) -> str:
    """The exact command to continue this run in the next session."""
    parts = [f"python -m {module}"]
    if config_path:
        parts.append(f"--config {config_path}")
    for o in overrides:
        parts.append(f"--set {o}")
    return " ".join(parts)


def verify_no_duplicates(path: str | os.PathLike) -> Dict[str, int]:
    """Return uids that appear more than once (should always be empty)."""
    counts: Dict[str, int] = {}
    for r in iter_jsonl(path, tolerant=True):
        uid = r.get("uid")
        if uid:
            counts[uid] = counts.get(uid, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}
