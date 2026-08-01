"""Template for a new reasoning method. Copy, rename, edit — that is the whole job.

This file is intentionally outside `src/` so it is not auto-registered. Two ways
to use it:

1. Drop your file into `src/strategies/`, add its module name to
   `_BUILTIN_MODULES` in `src/strategies/__init__.py`, and select it by the name
   you passed to `@register_strategy`:

       strategy: {name: my_method, params: {k: 8}}

2. Keep it anywhere importable and reference it by path, with no edits to the
   package at all:

       strategy: {name: "examples.method_template:TemplateMethod", params: {k: 8}}

Everything the harness needs is in the returned `StrategyResult`. Resumability,
the time guard, token accounting, grading, metrics, CIs, figures and tables all
work automatically — none of them know what your method does.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.generation import GenerationBackend
from src.strategies.base import ReasoningStrategy, register_strategy
from src.types import Example, GenParams, StrategyResult


@register_strategy("template_method")
class TemplateMethod(ReasoningStrategy):
    """Sample k candidate solutions, then keep the one the model itself prefers.

    A stand-in for a real method: the point is the shape of the code, not the
    idea. Note what it does *not* have to do: no file IO, no timing, no grading,
    no seeding boilerplate.
    """

    description = "k-sample draft then self-selection (template)"

    def __init__(self, k: int = 4, temperature: float = 0.7, **kw: Any) -> None:
        super().__init__(k=k, temperature=temperature, **kw)
        self.k = int(k)
        self.temperature = float(temperature)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        # 1. Propose. One batched call returns k samples for the same prompt.
        prompt = self.user_prompt(example, instruction="Think step by step.")
        propose = self.gen(
            params,
            n=self.k,
            temperature=self.temperature,
            seed=self.sample_seed(example, params, tag="propose"),
        )
        groups = backend.generate([prompt], propose)
        traces = [c.text for c in groups[0]]

        # 2. Select. Any selection rule goes here: a verifier model, a learned
        #    scorer, a majority vote, a search. This one asks the model.
        listing = "\n\n".join(
            f"Candidate {i + 1}:\n{t}" for i, t in enumerate(traces)
        )
        judge_prompt = self.user_prompt(
            example,
            instruction=(
                f"{listing}\n\nWhich candidate is correct? Reply with the final "
                "answer only, in the required format."
            ),
        )
        judge = self.gen(
            params, n=1, temperature=0.0, seed=self.sample_seed(example, params, tag="judge")
        )
        judge_groups = backend.generate([judge_prompt], judge)
        verdict = judge_groups[0][0].text

        # 3. Report. `extra` is free-form but must be JSON-serialisable; the
        #    analysis code can mine it later without re-running inference.
        answer: Optional[str] = self.extract(verdict, example)
        if answer is None:
            answer, _ = self.vote(traces, example)
        tokens_prompt, tokens_completion = self.tally(groups + judge_groups)
        extra: Dict[str, Any] = {
            "k": self.k,
            "candidate_answers": [self.extract(t, example) for t in traces],
            "judge_text": verdict,
            "fell_back_to_vote": self.extract(verdict, example) is None,
        }
        return StrategyResult(
            final_answer=answer,
            # Put every sampled solution here: grading extracts a per-sample
            # answer from each one, which is what pass@k is computed from.
            reasoning_traces=traces,
            n_samples=self.k,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=2,
            extra=extra,
        )
