"""Best-of-N: sample N candidates, keep the highest-scoring one.

The scorer is the interesting part and is therefore a plug-in point in its own
right: `SCORER_REGISTRY` maps a name to a callable, and a `scorer` value
containing a colon is imported as `module:function`. A new verifier or reward
model can be dropped in from outside this package with no edits here.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..generation import Completion, GenerationBackend
from ..prompts import COT_INSTRUCTION
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy

log = logging.getLogger(__name__)

#: (example, text, completion, backend) -> score; higher is better.
Scorer = Callable[[Example, str, Completion, GenerationBackend], float]

SCORER_REGISTRY: Dict[str, Scorer] = {}


def register_scorer(name: str) -> Callable[[Scorer], Scorer]:
    def deco(fn: Scorer) -> Scorer:
        if name in SCORER_REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        SCORER_REGISTRY[name] = fn
        return fn

    return deco


def available_scorers() -> List[str]:
    return sorted(SCORER_REGISTRY)


@register_scorer("logprob")
def score_mean_logprob(
    example: Example, text: str, completion: Completion, backend: GenerationBackend
) -> float:
    """Mean token logprob: the standard length-normalised likelihood baseline.

    Summed logprob would systematically prefer short answers, which conflates
    "confident" with "brief".
    """
    mean = completion.mean_logprob
    if mean is not None:
        return float(mean)
    if completion.cumulative_logprob is not None and completion.tokens_completion:
        return float(completion.cumulative_logprob) / completion.tokens_completion
    return 0.0


@register_scorer("length")
def score_shorter_is_better(
    example: Example, text: str, completion: Completion, backend: GenerationBackend
) -> float:
    """A deliberately trivial baseline: shorter solutions score higher.

    Worth reporting, because on some benchmarks it is embarrassingly competitive
    with likelihood reranking, which is useful context for any claim that a
    scorer "works".
    """
    return -float(completion.tokens_completion or len(text) / 4)


_RATING_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/\s*10)?")

_JUDGE_RUBRIC = (
    "Rate the candidate solution above for correctness on a scale from 0 to 10, "
    "where 10 means certainly correct and 0 means certainly wrong. Reply with the "
    "number only."
)


@register_scorer("llm_judge")
def score_llm_judge(
    example: Example, text: str, completion: Completion, backend: GenerationBackend
) -> float:
    """Ask the model to rate its own candidate 0-10.

    Costs an extra generation per candidate, so best-of-N with this scorer is
    roughly twice the tokens of best-of-N with `logprob`. Report it on the
    accuracy-versus-tokens axis, not per sample.
    """
    prompt = [
        {"role": "system", "content": "You are a strict grader."},
        {
            "role": "user",
            "content": (
                f"Problem:\n{example.question}\n\nCandidate solution:\n{text}\n\n"
                f"{_JUDGE_RUBRIC}"
            ),
        },
    ]
    try:
        judged = backend.generate_one(
            prompt, GenParams(max_new_tokens=8, temperature=0.0, n=1)
        )
        m = _RATING_RE.search(judged.text)
        if m:
            return max(0.0, min(10.0, float(m.group(1))))
    except Exception:  # noqa: BLE001 - a judge failure must not lose the example
        log.debug("llm_judge scorer failed", exc_info=True)
    return 0.0


def resolve_scorer(spec: str) -> Scorer:
    """Look up a registered scorer, or import a `module:function` target."""
    if ":" in spec:
        module_name, _, fn_name = spec.partition(":")
        fn = getattr(importlib.import_module(module_name), fn_name)
        if not callable(fn):
            raise TypeError(f"{spec} is not callable")
        return fn  # type: ignore[return-value]
    if spec not in SCORER_REGISTRY:
        raise KeyError(
            f"unknown scorer {spec!r}. Available: {available_scorers()}. "
            "You can also pass 'my_module:my_function'."
        )
    return SCORER_REGISTRY[spec]


@register_strategy("best_of_n")
class BestOfN(ReasoningStrategy):
    """Sample n candidates, then select the argmax under a pluggable scorer."""

    description = "Best-of-N with a pluggable scorer"

    def __init__(
        self,
        n: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.95,
        scorer: str = "logprob",
        **kw: Any,
    ) -> None:
        super().__init__(n=n, temperature=temperature, top_p=top_p, scorer=scorer, **kw)
        if int(n) < 1:
            raise ValueError("n must be >= 1")
        self.n = int(n)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.scorer_name = scorer
        self.scorer = resolve_scorer(scorer)
        # Likelihood reranking is meaningless without logprobs, so ask for them.
        self.requires_logprobs = scorer == "logprob"

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        prompt = self.user_prompt(example, instruction=COT_INSTRUCTION)
        gen = self.gen(
            params,
            n=self.n,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.sample_seed(example, params, tag="bon"),
        )
        groups = backend.generate([prompt], gen)
        candidates = groups[0]
        traces = [c.text for c in candidates]

        scores: List[float] = []
        for text, completion in zip(traces, candidates):
            try:
                scores.append(float(self.scorer(example, text, completion, backend)))
            except Exception:  # noqa: BLE001 - a broken scorer must not lose the run
                log.exception("scorer %r failed; scoring this candidate 0", self.scorer_name)
                scores.append(float("-inf"))

        best_index = max(range(len(scores)), key=lambda i: scores[i]) if scores else 0
        answers = [self.extract(t, example) for t in traces]
        n_calls = 1 + (self.n if self.scorer_name == "llm_judge" else 0)
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=answers[best_index] if answers else None,
            reasoning_traces=traces,
            n_samples=len(traces),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=n_calls,
            sample_stats=self.per_sample_stats(groups),
            extra={
                "n": self.n,
                "scorer": self.scorer_name,
                "scores": [None if s == float("-inf") else round(s, 6) for s in scores],
                "best_index": best_index,
                "best_score": (
                    None if not scores or scores[best_index] == float("-inf")
                    else round(scores[best_index], 6)
                ),
                "sample_answers": answers,
                "temperature": self.temperature,
            },
        )
