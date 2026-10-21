"""Chain-of-thought baselines: zero-shot and few-shot.

`cot_zeroshot` is the reference baseline that significance tests compare
against, so it is deliberately the plainest possible implementation: one greedy
sample, one prompt, no tricks.
"""

from __future__ import annotations

from typing import Any

from ..generation import GenerationBackend
from ..prompts import COT_INSTRUCTION, get_few_shot
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy


@register_strategy("cot_zeroshot")
class ZeroShotCoT(ReasoningStrategy):
    """"Think step by step", one greedy sample."""

    description = "Zero-shot chain of thought"
    default_gen = {"temperature": 0.0}

    def __init__(self, instruction: str = COT_INSTRUCTION, **kw: Any) -> None:
        super().__init__(instruction=instruction, **kw)
        self.instruction = instruction

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        prompt = self.user_prompt(example, instruction=self.instruction)
        gen = self.gen(
            params, n=1, seed=self.sample_seed(example, params, tag="cot")
        )
        groups = backend.generate([prompt], gen)
        completion = groups[0][0]
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=self.extract(completion.text, example),
            reasoning_traces=[completion.text],
            n_samples=1,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=1,
            sample_stats=self.per_sample_stats(groups),
            extra={
                "n_shots": 0,
                # A truncated trace usually means the answer never got stated;
                # tracking it separates "wrong" from "ran out of tokens".
                "truncated": completion.finish_reason == "length",
            },
        )


@register_strategy("cot_fewshot")
class FewShotCoT(ReasoningStrategy):
    """Chain of thought with worked exemplars prepended as chat turns."""

    description = "Few-shot chain of thought"
    default_gen = {"temperature": 0.0}

    def __init__(
        self,
        n_shots: int = 4,
        shot_order: str = "forward",
        instruction: str = COT_INSTRUCTION,
        **kw: Any,
    ) -> None:
        super().__init__(
            n_shots=n_shots, shot_order=shot_order, instruction=instruction, **kw
        )
        self.n_shots = int(n_shots)
        self.shot_order = shot_order
        self.instruction = instruction

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        few_shot = get_few_shot(example, n=self.n_shots, order=self.shot_order)
        prompt = self.user_prompt(
            example, instruction=self.instruction, few_shot=few_shot
        )
        gen = self.gen(
            params, n=1, seed=self.sample_seed(example, params, tag="cot_fewshot")
        )
        groups = backend.generate([prompt], gen)
        completion = groups[0][0]
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=self.extract(completion.text, example),
            reasoning_traces=[completion.text],
            n_samples=1,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=1,
            sample_stats=self.per_sample_stats(groups),
            extra={
                "n_shots": len(few_shot),
                "shot_order": self.shot_order,
                "truncated": completion.finish_reason == "length",
            },
        )
