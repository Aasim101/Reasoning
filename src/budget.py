"""Wall-clock guard and GPU-hour accounting.

Kaggle's free tier gives ~30 GPU-hours/week and kills a session at 12h. Wasted
compute is the scarcest resource in this project, so the runner asks a
`TimeGuard` for permission before every example and stops cleanly at an example
boundary while there is still time to write state and print a resume command.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from .utils import human_time

log = logging.getLogger(__name__)

#: Kaggle hard limits (verified against kaggle.com/docs/notebooks, July 2026).
KAGGLE_SESSION_LIMIT_HOURS = 12.0
KAGGLE_WEEKLY_GPU_HOURS = 30.0


class TimeGuard:
    """Monitors elapsed wall-clock and decides when to stop.

    The stop rule is deliberately conservative: stop if the time left is less
    than `reserve + safety_factor * (recent per-example duration)`. Using a
    recent-window mean rather than the whole-run mean matters because
    per-example cost drifts (longer questions later in a dataset, growing KV
    cache, thermal throttling on T4s).
    """

    def __init__(
        self,
        budget_hours: float = 8.0,
        reserve_minutes: float = 10.0,
        n_gpus: int = 0,
        gpu_hour_multiplier: float = 1.0,
        window: int = 20,
        safety_factor: float = 1.5,
        prior_elapsed_seconds: float = 0.0,
    ) -> None:
        self.budget_seconds = float(budget_hours) * 3600.0
        self.reserve_seconds = max(0.0, float(reserve_minutes) * 60.0)
        self.n_gpus = int(n_gpus)
        self.gpu_hour_multiplier = float(gpu_hour_multiplier)
        self.safety_factor = float(safety_factor)
        self.prior_elapsed_seconds = float(prior_elapsed_seconds)
        self._durations: Deque[float] = deque(maxlen=max(1, int(window)))
        self._all_durations: List[float] = []
        self._t0: Optional[float] = None
        self.stop_reason: Optional[str] = None
        if budget_hours > KAGGLE_SESSION_LIMIT_HOURS:
            log.warning(
                "time_budget_hours=%.1f exceeds the Kaggle session limit of %.0fh; "
                "the session will be killed before the budget expires",
                budget_hours,
                KAGGLE_SESSION_LIMIT_HOURS,
            )

    # ------------------------------------------------------------------ control
    def start(self) -> "TimeGuard":
        self._t0 = time.monotonic()
        return self

    @property
    def started(self) -> bool:
        return self._t0 is not None

    @property
    def elapsed_seconds(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.monotonic() - self._t0

    @property
    def remaining_seconds(self) -> float:
        return self.budget_seconds - self.elapsed_seconds

    @property
    def gpu_hours(self) -> float:
        """GPU-hours consumed by *this session*, for the weekly quota."""
        return self.elapsed_seconds / 3600.0 * self.gpu_hour_multiplier

    def record(self, duration_seconds: float) -> None:
        """Report how long one example took."""
        self._durations.append(max(0.0, float(duration_seconds)))
        self._all_durations.append(max(0.0, float(duration_seconds)))

    # ----------------------------------------------------------------- queries
    @property
    def mean_recent_seconds(self) -> Optional[float]:
        if not self._durations:
            return None
        return sum(self._durations) / len(self._durations)

    @property
    def mean_seconds(self) -> Optional[float]:
        if not self._all_durations:
            return None
        return sum(self._all_durations) / len(self._all_durations)

    def projected_next_seconds(self) -> float:
        """Pessimistic estimate of the next example's duration."""
        recent = self.mean_recent_seconds
        if recent is None:
            # Before any measurement, reserve alone must cover the first example.
            return 0.0
        return max(recent, max(self._durations)) if self._durations else recent

    def should_stop(self) -> bool:
        """True when the next example probably will not fit in the budget."""
        if self._t0 is None:
            return False
        needed = self.reserve_seconds + self.safety_factor * self.projected_next_seconds()
        if self.remaining_seconds <= needed:
            self.stop_reason = "time_budget"
            return True
        return False

    def eta_seconds(self, n_remaining: int) -> Optional[float]:
        m = self.mean_recent_seconds
        if m is None:
            return None
        return m * max(0, int(n_remaining))

    def can_finish(self, n_remaining: int) -> Optional[bool]:
        eta = self.eta_seconds(n_remaining)
        if eta is None:
            return None
        return eta + self.reserve_seconds <= self.remaining_seconds

    def sessions_needed(self, n_remaining: int) -> Optional[float]:
        """How many more sessions of this budget the remaining work needs."""
        eta = self.eta_seconds(n_remaining)
        if eta is None or self.budget_seconds <= self.reserve_seconds:
            return None
        usable = self.budget_seconds - self.reserve_seconds
        return eta / usable

    # ----------------------------------------------------------------- reports
    def summary(self, n_remaining: int = 0) -> Dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "elapsed_human": human_time(self.elapsed_seconds),
            "budget_hours": round(self.budget_seconds / 3600.0, 3),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "gpu_hours_this_session": round(self.gpu_hours, 3),
            "n_gpus": self.n_gpus,
            "mean_seconds_per_example": (
                round(self.mean_seconds, 2) if self.mean_seconds is not None else None
            ),
            "mean_recent_seconds_per_example": (
                round(self.mean_recent_seconds, 2)
                if self.mean_recent_seconds is not None
                else None
            ),
            "eta_remaining_human": (
                human_time(self.eta_seconds(n_remaining))
                if self.eta_seconds(n_remaining) is not None
                else None
            ),
            "sessions_needed": (
                round(self.sessions_needed(n_remaining), 2)
                if self.sessions_needed(n_remaining) is not None
                else None
            ),
            "stop_reason": self.stop_reason,
        }

    def progress_line(self, n_done: int, n_total: int) -> str:
        eta = self.eta_seconds(max(0, n_total - n_done))
        rate = self.mean_recent_seconds
        parts = [
            f"[{n_done}/{n_total}]",
            f"elapsed={human_time(self.elapsed_seconds)}",
            f"budget_left={human_time(max(0.0, self.remaining_seconds))}",
        ]
        if rate is not None:
            parts.append(f"s/ex={rate:.1f}")
        parts.append(f"eta={human_time(eta)}" if eta is not None else "eta=?")
        return " ".join(parts)


@dataclass
class BudgetLedger:
    """Cross-session GPU-hour bookkeeping against the weekly quota.

    Kaggle bills session wall-clock time against the weekly accelerator quota;
    `gpu_hour_multiplier` in the config lets you account 2x T4 pessimistically.
    """

    weekly_quota_hours: float = KAGGLE_WEEKLY_GPU_HOURS
    used_hours: float = 0.0
    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, hours: float, note: str = "") -> None:
        self.used_hours += float(hours)
        self.entries.append(
            {"label": label, "hours": round(float(hours), 3), "note": note}
        )

    @property
    def remaining_hours(self) -> float:
        return max(0.0, self.weekly_quota_hours - self.used_hours)

    def report(self) -> str:
        lines = [
            f"GPU-hour ledger: {self.used_hours:.2f}h used / "
            f"{self.weekly_quota_hours:.0f}h weekly quota "
            f"({self.remaining_hours:.2f}h left)"
        ]
        for e in self.entries:
            note = f"  # {e['note']}" if e["note"] else ""
            lines.append(f"  {e['hours']:>6.2f}h  {e['label']}{note}")
        return "\n".join(lines)


def estimate_run_hours(
    n_examples: int,
    seconds_per_example: float,
    n_strategies: int = 1,
    n_seeds: int = 1,
    overhead_minutes: float = 15.0,
) -> float:
    """Plan a batch of experiment cells before spending quota on them."""
    total = n_examples * seconds_per_example * n_strategies * n_seeds
    return (total + overhead_minutes * 60.0) / 3600.0
