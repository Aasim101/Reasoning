"""The reasoning-strategy plugin interface and its registry.

Adding a new reasoning method is a single ~50-line file:

```python
from ..types import Example, StrategyResult
from .base import ReasoningStrategy, register_strategy


@register_strategy("my_method")
class MyMethod(ReasoningStrategy):
    def __init__(self, k: int = 4, **kw):
        super().__init__(**kw)
        self.k = k

    def run(self, example, backend, params):
        prompt = self.user_prompt(example, instruction="Think step by step.")
        groups = backend.generate([prompt], self.gen(params, n=self.k, temperature=0.7))
        traces = [c.text for c in groups[0]]
        answer, info = self.vote(traces, example)
        tp, tc = self.tally(groups)
        return StrategyResult(
            final_answer=answer,
            reasoning_traces=traces,
            n_samples=self.k,
            tokens_prompt=tp,
            tokens_completion=tc,
            n_calls=1,
            extra=info,
        )
```

Then select it from YAML with `strategy: {name: my_method, params: {k: 8}}`.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from ..generation import GenerationBackend
from ..types import ChatMessage, Completion, Example, GenParams, Prompt, StrategyResult
from ..utils import stable_hash

log = logging.getLogger(__name__)

STRATEGY_REGISTRY: Dict[str, Type["ReasoningStrategy"]] = {}


def register_strategy(
    name: str, *aliases: str
) -> Callable[[Type["ReasoningStrategy"]], Type["ReasoningStrategy"]]:
    """Class decorator adding a strategy to the name -> class registry."""

    def deco(cls: Type["ReasoningStrategy"]) -> Type["ReasoningStrategy"]:
        for key in (name, *aliases):
            existing = STRATEGY_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"strategy name {key!r} already registered by {existing.__name__}"
                )
            STRATEGY_REGISTRY[key] = cls
        cls.name = name
        return cls

    return deco


class ReasoningStrategy(ABC):
    """Turns one `Example` into one `StrategyResult` using a backend.

    Subclasses implement `run` and may use the helpers below. Everything a
    strategy needs to be reproducible is derived from `example.id` and
    `params.seed`, so no strategy should hold mutable state across examples.
    """

    name: str = "base"
    #: Human-readable one-liner used in tables and logs.
    description: str = ""
    #: Generation overrides always applied by this strategy (e.g. temperature).
    default_gen: Dict[str, Any] = {}
    #: Whether this strategy needs per-token logprobs from the backend.
    requires_logprobs: bool = False

    def __init__(self, **params: Any) -> None:
        #: Recorded verbatim in the manifest for provenance.
        self.params: Dict[str, Any] = dict(params)
        unexpected = set(params)
        if unexpected:
            log.debug("%s received extra params: %s", type(self).__name__, sorted(unexpected))

    # ------------------------------------------------------------------ contract
    @abstractmethod
    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        """Produce a final answer (and traces) for `example`."""

    # ------------------------------------------------------------------- helpers
    def gen(self, params: GenParams, **overrides: Any) -> GenParams:
        """Apply `default_gen` then explicit overrides to the run's GenParams."""
        merged = dict(self.default_gen)
        merged.update(overrides)
        if self.requires_logprobs:
            merged.setdefault("logprobs", True)
        return params.with_(**merged) if merged else params

    def user_prompt(
        self,
        example: Example,
        instruction: Optional[str] = None,
        system: Optional[str] = None,
        few_shot: Sequence[Tuple[str, str]] = (),
        hint: Optional[str] = None,
    ) -> Prompt:
        """Build a chat prompt for an example.

        Delegates question formatting (multiple-choice option rendering, answer
        format instructions) to `src.prompts` so every strategy asks in the same
        way and answer extraction stays reliable.

        If the strategy was given a `style` param (a `PromptStyle` dict), it
        wins: that is how a declarative prompt configuration overrides a
        strategy's built-in wording without subclassing it.
        """
        from .. import prompts as P

        style = self.params.get("style")
        if style is not None and instruction is None and system is None and not few_shot:
            return P.PromptStyle.from_dict(style).build(
                example, model_id=str(self.params.get("model_id") or "")
            )
        return P.build_prompt(
            example,
            instruction=instruction,
            system=system,
            few_shot=few_shot,
            hint=hint,
        )

    def extract(self, text: str, example: Example) -> Optional[str]:
        """Extract a final answer from a completion (grader-consistent)."""
        from ..answers import extract_answer

        return extract_answer(text, answer_type=example.answer_type, choices=example.choices)

    def vote(
        self, traces: Sequence[str], example: Example
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Equivalence-aware majority vote over extracted answers."""
        from ..answers import majority_vote

        answers = [self.extract(t, example) for t in traces]
        winner, info = majority_vote(
            answers, answer_type=example.answer_type, choices=example.choices
        )
        info = dict(info)
        info["sample_answers"] = answers
        return winner, info

    @staticmethod
    def per_sample_stats(
        groups: Sequence[Sequence[Completion]],
        example: Optional["Example"] = None,
        traces: Optional[Sequence[str]] = None,
        rendered_prompt_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Flatten completion groups into per-sample accounting records."""
        from ..answers import extract_with_method

        stats: List[Dict[str, Any]] = []
        flat_traces = list(traces or [])
        idx = 0
        for group in groups:
            for c in group:
                row = dict(c.stats())
                row["truncated"] = (c.finish_reason or "").lower() in (
                    "length",
                    "max_tokens",
                )
                if rendered_prompt_hash:
                    row["rendered_prompt_hash"] = rendered_prompt_hash
                if example is not None and idx < len(flat_traces):
                    ans, method, failed = extract_with_method(
                        str(flat_traces[idx]), example.answer_type, example.choices
                    )
                    row["extraction_method"] = method
                    row["extraction_failed"] = failed
                stats.append(row)
                idx += 1
        return stats

    @staticmethod
    def tally(groups: Sequence[Sequence[Completion]]) -> Tuple[int, int]:
        """Token accounting for a list of completion groups.

        Prompt tokens are counted **once per group**, not once per sample,
        because a batched n-sample call encodes the prompt a single time. This
        keeps "tokens per correct answer" honest for self-consistency.
        """
        tokens_prompt = 0
        tokens_completion = 0
        for group in groups:
            if not group:
                continue
            tokens_prompt += max(c.tokens_prompt for c in group)
            tokens_completion += sum(c.tokens_completion for c in group)
        return tokens_prompt, tokens_completion

    def rng(self, example: Example, params: GenParams) -> random.Random:
        """Deterministic per-example RNG (never use the global `random`)."""
        key = stable_hash(
            {"ex": example.id, "seed": params.seed, "strategy": self.name}, length=16
        )
        return random.Random(int(key, 16))

    def sample_seed(self, example: Example, params: GenParams, tag: str = "") -> int:
        """A stable sub-seed for a specific sampling call within one example."""
        key = stable_hash(
            {"ex": example.id, "seed": params.seed, "strategy": self.name, "tag": tag},
            length=8,
        )
        return int(key, 16) % (2**31 - 1)

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "class": type(self).__name__,
            "description": self.description or (type(self).__doc__ or "").strip().split("\n")[0],
            "params": self.params,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, params={self.params!r})"


def build_strategy(name: str, **params: Any) -> ReasoningStrategy:
    """Instantiate a registered strategy, or a `module.path:ClassName` target.

    The `module:Class` form lets an externally-defined method (for example the
    novel method described in docs/METHOD_SPEC.md) be used without editing this
    package.
    """
    if ":" in name:
        import importlib

        mod_name, _, cls_name = name.partition(":")
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        if not issubclass(cls, ReasoningStrategy):
            raise TypeError(f"{name} is not a ReasoningStrategy subclass")
        return cls(**params)
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}. "
            "You can also pass 'my_pkg.my_module:MyClass'."
        )
    return STRATEGY_REGISTRY[name](**params)


def available_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY)
