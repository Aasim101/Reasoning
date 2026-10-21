"""Pluggable reasoning strategies.

Importing this package registers every built-in strategy by name. Configs select
one with `strategy: {name: ..., params: {...}}`.
"""

from __future__ import annotations

import logging

from .base import (
    STRATEGY_REGISTRY,
    ReasoningStrategy,
    available_strategies,
    build_strategy,
    register_strategy,
)

log = logging.getLogger(__name__)

#: Built-in strategy modules, imported for their registration side effects.
_BUILTIN_MODULES = (
    "direct",
    "cot",
    "self_consistency",
    "best_of_n",
    "self_refine",
    "configured_sampling",
)

for _mod in _BUILTIN_MODULES:
    try:
        __import__(f"{__name__}.{_mod}", fromlist=["*"])
    except Exception as exc:  # pragma: no cover - surfaces a broken plugin early
        log.error("failed to import built-in strategy module %r: %s", _mod, exc)
        raise

__all__ = [
    "STRATEGY_REGISTRY",
    "ReasoningStrategy",
    "available_strategies",
    "build_strategy",
    "register_strategy",
]
