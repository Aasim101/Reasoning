"""The method's workhorse: draw `N` chains under one elicitation configuration.

This is the only strategy that generates the paper's sampling corpus. Every
analysis — the variance decomposition, the transfer correlations, the hard-subset
overlap, the mode-reordering taxonomy and the configuration-diversified voting
intervention — is computed offline from what this strategy records, with no
further generation. That is the property that makes the paper affordable, so this
strategy's job is to record *enough*, once.

Two design points that a reviewer would otherwise catch:

* Decoding temperature comes from the run's `GenParams`, never from a strategy
  default. Temperature is elicitation axis `c5`; a strategy that quietly forced
  its own value would collapse that axis. This is why the method does not reuse
  `self_consistency`, whose constructor owns a temperature.
* The per-sample extracted answers and per-sample completion-token counts are
  recorded for every chain. The answers are what the modal ceiling and the
  reordering analysis need; the token counts are what makes the budget-matched
  CDV comparison honest, because configurations differ in mean chain length and
  matching on sample count instead of tokens would be a subtle cheat.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..generation import Completion, GenerationBackend
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy


@register_strategy("configured_sampling", "cdv_corpus")
class ConfiguredSampling(ReasoningStrategy):
    """Sample `n_samples` chains at the run's decoding settings, then vote."""

    description = "N-sample corpus draw under one elicitation configuration"

    def __init__(
        self,
        n_samples: int = 24,
        batch_calls: bool = True,
        #: Injected by `RunConfig.strategy_params()`; recorded for provenance.
        elicitation_id: Optional[str] = None,
        **kw: Any,
    ) -> None:
        super().__init__(
            n_samples=n_samples,
            batch_calls=batch_calls,
            elicitation_id=elicitation_id,
            **kw,
        )
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1")
        self.n_samples = int(n_samples)
        self.batch_calls = bool(batch_calls)
        self.elicitation_id = elicitation_id

    def _prompt(self, example: Example) -> Any:
        """Elicitation style when present, otherwise a plain zero-shot CoT ask."""
        if self.params.get("style"):
            return self.user_prompt(example)
        from ..prompts import COT_INSTRUCTION

        return self.user_prompt(example, instruction=COT_INSTRUCTION)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        prompt = self._prompt(example)
        groups: List[List[Completion]] = []
        n_calls = 0
        if self.batch_calls:
            # One call with n=N so the prompt is prefilled once and the KV cache is
            # shared across the N samples. METHOD_SPEC section 4.2 marks this as
            # mandatory, and it is the single largest throughput win available.
            groups = backend.generate(
                [prompt],
                params.with_(
                    n=self.n_samples, seed=self.sample_seed(example, params, "corpus")
                ),
            )
            n_calls = 1
        else:
            for i in range(self.n_samples):
                out = backend.generate(
                    [prompt],
                    params.with_(
                        n=1, seed=self.sample_seed(example, params, f"corpus{i}")
                    ),
                )
                groups.append(out[0])
                n_calls += 1

        from ..prompts import prompt_digest, render_prompt_text

        traces = [c.text for group in groups for c in group]
        answer, info = self.vote(traces, example)
        tokens_prompt, tokens_completion = self.tally(groups)

        model_id = str(self.params.get("model_id") or "")
        rendered, token_ids = render_prompt_text(
            prompt, model_id=model_id or None
        )
        prompt_meta = prompt_digest(rendered)
        sample_stats = self.per_sample_stats(
            groups,
            example=example,
            traces=traces,
            rendered_prompt_hash=prompt_meta["rendered_prompt_hash"],
        )

        extra: Dict[str, Any] = dict(info)
        extra.update(
            {
                "elicitation_id": self.elicitation_id,
                "n_requested": self.n_samples,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "rendered_prompt_hash": prompt_meta["rendered_prompt_hash"],
                "prompt_length_chars": prompt_meta["prompt_length_chars"],
                "rendered_prompt": prompt_meta.get("rendered_prompt"),
                "rendered_token_ids": token_ids,
                "n_truncated": sum(
                    1 for g in groups for c in g if c.finish_reason == "length"
                ),
                # Extraction failure is its own answer class and must be counted,
                # never dropped: an inflated failure class biases the modal ceiling
                # downward (METHOD_SPEC section 4.3).
                "n_extraction_failures": int(info.get("n_none") or 0),
                "top_logprobs_available": any(
                    c.top_logprobs for g in groups for c in g
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
            sample_stats=sample_stats,
            extra=extra,
        )


@register_strategy("greedy_pass")
class GreedyPass(ReasoningStrategy):
    """One greedy chain (baseline B2), for comparability with the literature.

    Temperature is forced to 0 here on purpose: B2 *is* the greedy pass, so it is
    the one place a strategy is entitled to override the decoding axis. The
    elicitation's prompt style is still honoured, so B2 exists per configuration.
    """

    description = "Greedy decoding, one sample (baseline B2)"

    def __init__(self, elicitation_id: Optional[str] = None, **kw: Any) -> None:
        super().__init__(elicitation_id=elicitation_id, **kw)
        self.elicitation_id = elicitation_id

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        if self.params.get("style"):
            prompt = self.user_prompt(example)
        else:
            from ..prompts import COT_INSTRUCTION

            prompt = self.user_prompt(example, instruction=COT_INSTRUCTION)
        groups = backend.generate([prompt], params.with_(n=1, temperature=0.0, top_p=1.0))
        traces = [c.text for group in groups for c in group]
        answer = self.extract(traces[0], example) if traces else None
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=answer,
            reasoning_traces=traces,
            n_samples=len(traces),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=1,
            sample_stats=self.per_sample_stats(groups),
            extra={
                "elicitation_id": self.elicitation_id,
                "greedy": True,
                "sample_answers": [answer],
            },
        )
