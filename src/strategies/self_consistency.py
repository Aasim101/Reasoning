"""Self-consistency: sample k chains, take the equivalence-aware majority vote.

The k samples are requested in a single call with `n=k` so the prompt is
prefilled once and the KV cache is shared. That is a large throughput win on a
T4 and it is also why `tally` counts prompt tokens once per call rather than once
per sample: billing the prompt k times would overstate the cost of this method by
a factor of k in the tokens-per-correct-answer metric.
"""

from __future__ import annotations

from typing import Any, List

from ..generation import Completion, GenerationBackend
from ..prompts import COT_INSTRUCTION, get_few_shot
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy


@register_strategy("self_consistency", "majority_vote")
class SelfConsistency(ReasoningStrategy):
    """Majority vote over k sampled chains of thought."""

    description = "Self-consistency (majority vote over k samples)"

    def __init__(
        self,
        k: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        n_shots: int = 0,
        batch_calls: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(
            k=k,
            temperature=temperature,
            top_p=top_p,
            n_shots=n_shots,
            batch_calls=batch_calls,
            **kw,
        )
        if int(k) < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.n_shots = int(n_shots)
        self.batch_calls = bool(batch_calls)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        few_shot = get_few_shot(example, self.n_shots) if self.n_shots else []
        prompt = self.user_prompt(
            example, instruction=COT_INSTRUCTION, few_shot=few_shot
        )
        base = self.gen(
            params, temperature=self.temperature, top_p=self.top_p
        )

        groups: List[List[Completion]] = []
        n_calls = 0
        if self.batch_calls:
            groups = backend.generate(
                [prompt], base.with_(n=self.k, seed=self.sample_seed(example, params, "sc"))
            )
            n_calls = 1
        else:
            # Fallback for a backend that cannot return n>1: k separate calls with
            # distinct seeds, so the samples are still independent.
            for i in range(self.k):
                out = backend.generate(
                    [prompt],
                    base.with_(n=1, seed=self.sample_seed(example, params, f"sc{i}")),
                )
                groups.append(out[0])
                n_calls += 1
            groups = [groups[i] for i in range(len(groups))]

        traces = [c.text for group in groups for c in group]
        answer, info = self.vote(traces, example)
        tokens_prompt, tokens_completion = self.tally(groups)
        extra = dict(info)
        extra.update(
            {
                "k": self.k,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "n_shots": len(few_shot),
                "n_truncated": sum(
                    1 for g in groups for c in g if c.finish_reason == "length"
                ),
            }
        )
        return StrategyResult(
            final_answer=answer,
            reasoning_traces=traces,
            n_samples=len(traces),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=n_calls,
            sample_stats=self.per_sample_stats(groups),
            extra=extra,
        )
