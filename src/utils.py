"""Small dependency-light helpers used across the harness."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, TypeVar

T = TypeVar("T")

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Idempotent root logger setup that plays nicely inside notebooks."""
    root = logging.getLogger()
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(lvl)
    for h in root.handlers:
        h.setLevel(lvl)
        with contextlib.suppress(Exception):
            h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    if not root.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        h.setLevel(lvl)
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        root.addHandler(h)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(obj: Any, length: int = 16) -> str:
    """Deterministic hash of any JSON-able object.

    Uses sorted keys and a stable string coercion so the value does not depend
    on dict insertion order or Python version.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=_coerce)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _coerce(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(map(str, o))
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def seed_everything(seed: int) -> None:
    """Seed python/numpy/torch as available. Safe to call without torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    with contextlib.suppress(Exception):
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    with contextlib.suppress(Exception):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def write_json_atomic(
    path: str | os.PathLike, obj: Any, retries: int = 5, retry_delay: float = 0.05
) -> None:
    """Write JSON such that a kill mid-write cannot corrupt the existing file.

    The replace is retried because on Windows an antivirus or search indexer can
    hold a freshly created file open for a few milliseconds, which surfaces as an
    intermittent PermissionError. Losing the manifest to that would mean losing
    the record of a run's progress, so it is worth the retry loop.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, default=_coerce)
            f.flush()
            os.fsync(f.fileno())
        last_error: Optional[BaseException] = None
        for attempt in range(max(1, retries)):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:  # pragma: no cover - platform specific
                last_error = exc
                time.sleep(retry_delay * (attempt + 1))
        # Last resort: a non-atomic direct write still beats losing the state.
        logging.getLogger(__name__).warning(
            "atomic replace of %s failed after %d attempts (%s); writing in place",
            path,
            retries,
            last_error,
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False, default=_coerce)
            f.flush()
            os.fsync(f.fileno())
    finally:
        with contextlib.suppress(OSError):
            if os.path.exists(tmp):
                os.unlink(tmp)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def open_maybe_gzip(path: str | os.PathLike, mode: str = "rt") -> Any:
    """Open a text file, transparently handling a `.gz` suffix.

    Results files may be gzipped to stay inside Kaggle's 20 GB working directory,
    and every reader in the harness goes through here so the choice is invisible
    to callers.
    """
    p = Path(path)
    if p.name.endswith(".gz"):
        import gzip

        return gzip.open(p, mode, encoding="utf-8")
    return p.open(mode, encoding="utf-8")


def iter_jsonl(path: str | os.PathLike, tolerant: bool = True) -> Iterator[Dict[str, Any]]:
    """Stream a JSONL file, skipping a truncated final line.

    An abrupt session kill can leave a partial line even with frequent fsync,
    so tolerant reading is the default everywhere in this project. A truncated
    gzip member raises inside the decompressor rather than at the JSON layer, so
    that is caught too.
    """
    p = Path(path)
    if not p.exists():
        return
    try:
        f = open_maybe_gzip(p, "rt")
    except OSError:  # pragma: no cover - unreadable file
        logging.getLogger(__name__).warning("could not open %s", p, exc_info=True)
        return
    with f:
        for lineno, line in enumerate(_tolerant_lines(f, p, tolerant), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if tolerant:
                    logging.getLogger(__name__).warning(
                        "skipping malformed JSONL line %d in %s (likely truncated by an "
                        "abrupt session kill)",
                        lineno,
                        p,
                    )
                    continue
                raise


def _tolerant_lines(handle: Any, path: Path, tolerant: bool) -> Iterator[str]:
    """Yield lines, surviving a truncated trailing gzip member."""
    while True:
        try:
            line = handle.readline()
        except (EOFError, OSError) as exc:
            if not tolerant:
                raise
            logging.getLogger(__name__).warning(
                "%s ends in a truncated compressed block (%s); the records read so "
                "far are intact",
                path,
                exc,
            )
            return
        if not line:
            return
        yield line


def chunked(seq: Sequence[T], size: int) -> Iterator[List[T]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def human_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Timer:
    """Context manager measuring wall-clock seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0


def truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def deep_get(d: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def unique_preserving_order(items: Iterable[T]) -> List[T]:
    seen: set = set()
    out: List[T] = []
    for it in items:
        key = it if isinstance(it, (str, int, float, bool, type(None))) else repr(it)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def optional_import(module: str) -> Optional[Any]:
    """Import a module, returning None if unavailable (no traceback noise)."""
    try:
        import importlib

        return importlib.import_module(module)
    except Exception:  # pragma: no cover - depends on environment
        return None
