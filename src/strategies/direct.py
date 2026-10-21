"""Direct answering: no chain of thought.

The floor everything else is measured against. Without it there is no way to say
how much of a model's accuracy comes from reasoning tokens rather than from
pattern-matching the answer, and reviewers ask exactly that.
"""

from __future__ import annotations

from typing import Any

from ..generation import GenerationBackend
from ..prompts import DIRECT_INSTRUCTION, get_few_shot
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy


@register_strategy("direct")
class DirectAnswer(ReasoningStrategy):
    """Ask for the final answer only, greedily, with a tight token cap."""

    description = "Direct answer, no chain of thought"
    default_gen = {"temperature": 0.0}

    def __init__(self, max_new_tokens: int = 32, n_shots: int = 0, **kw: Any) -> None:
        super().__init__(max_new_tokens=max_new_tokens, n_shots=n_shots, **kw)
        self.max_new_tokens = int(max_new_tokens)
        self.n_shots = int(n_shots)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        few_shot = []
        if self.n_shots > 0:
            # Strip the reasoning from the exemplars: a "direct" baseline that
            # showed worked solutions would not be a no-CoT condition at all.
            few_shot = [
                (q, a.strip().split("\n")[-1]) for q, a in get_few_shot(example, self.n_shots)
            ]
        prompt = self.user_prompt(
            example, instruction=DIRECT_INSTRUCTION, few_shot=few_shot
        )
        gen = self.gen(
            params,
            n=1,
            max_new_tokens=min(self.max_new_tokens, params.max_new_tokens),
            seed=self.sample_seed(example, params, tag="direct"),
        )
        groups = backend.generate([prompt], gen)
        text = groups[0][0].text
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=self.extract(text, example),
            reasoning_traces=[text],
            n_samples=1,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=1,
            sample_stats=self.per_sample_stats(groups),
            extra={"n_shots": self.n_shots, "max_new_tokens": gen.max_new_tokens},
        )
