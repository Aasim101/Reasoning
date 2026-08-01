"""Self-refine and self-verify: sequential test-time compute.

These are the dependent-call counterpart to self-consistency's parallel samples.
Each round must wait for the previous one, so GPU utilisation is lower and
per-example wall-clock is higher for the same token count - worth remembering
when reading the accuracy-versus-tokens curves, and worth budgeting for.

The draft is fed back in the *assistant* role via `prompts.continue_chat` rather
than pasted into a user turn, because instruct-tuned models behave differently
when asked to critique "your answer" versus "this text".
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..generation import Completion, GenerationBackend
from ..prompts import COT_INSTRUCTION, REFINE_INSTRUCTION, VERIFY_INSTRUCTION, continue_chat
from ..types import Example, GenParams, StrategyResult
from .base import ReasoningStrategy, register_strategy

_VERDICT_RE = re.compile(r"verdict\s*[:\-]?\s*(correct|incorrect)", re.IGNORECASE)


def parse_verdict(text: str) -> Optional[bool]:
    """True = verified correct, False = flagged incorrect, None = unparseable.

    None is kept distinct from False on purpose: "the verifier did not answer" is
    a different failure from "the verifier said no", and conflating them would
    hide a broken prompt behind a plausible-looking refinement rate.
    """
    matches = _VERDICT_RE.findall(text or "")
    if matches:
        return matches[-1].lower() == "correct"
    lowered = (text or "").lower()
    if "incorrect" in lowered or "is wrong" in lowered:
        return False
    if "correct" in lowered:
        return True
    return None


@register_strategy("self_refine")
class SelfRefine(ReasoningStrategy):
    """Draft, then verify and revise for up to `n_rounds` rounds."""

    description = "Self-refine (draft, verify, revise)"

    def __init__(
        self,
        n_rounds: int = 1,
        verify: bool = True,
        stop_when_verified: bool = True,
        temperature: float = 0.0,
        **kw: Any,
    ) -> None:
        super().__init__(
            n_rounds=n_rounds,
            verify=verify,
            stop_when_verified=stop_when_verified,
            temperature=temperature,
            **kw,
        )
        self.n_rounds = max(0, int(n_rounds))
        self.verify = bool(verify)
        self.stop_when_verified = bool(stop_when_verified)
        self.temperature = float(temperature)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        base = self.gen(params, n=1, temperature=self.temperature)
        prompt = self.user_prompt(example, instruction=COT_INSTRUCTION)

        groups: List[List[Completion]] = []
        drafts: List[str] = []
        verdicts: List[Optional[bool]] = []
        verdict_texts: List[str] = []
        n_calls = 0

        out = backend.generate(
            [prompt], base.with_(seed=self.sample_seed(example, params, "draft"))
        )
        groups.append(out[0])
        drafts.append(out[0][0].text)
        n_calls += 1

        rounds_used = 0
        for r in range(self.n_rounds):
            verdict: Optional[bool] = None
            if self.verify:
                verify_prompt = continue_chat(prompt, drafts[-1], VERIFY_INSTRUCTION)
                vout = backend.generate(
                    [verify_prompt],
                    base.with_(
                        max_new_tokens=min(256, base.max_new_tokens),
                        seed=self.sample_seed(example, params, f"verify{r}"),
                    ),
                )
                groups.append(vout[0])
                n_calls += 1
                verdict_text = vout[0][0].text
                verdict = parse_verdict(verdict_text)
                verdicts.append(verdict)
                verdict_texts.append(verdict_text)
                if verdict is True and self.stop_when_verified:
                    break

            refine_prompt = continue_chat(prompt, drafts[-1], REFINE_INSTRUCTION)
            rout = backend.generate(
                [refine_prompt],
                base.with_(seed=self.sample_seed(example, params, f"refine{r}")),
            )
            groups.append(rout[0])
            drafts.append(rout[0][0].text)
            n_calls += 1
            rounds_used += 1

        # Prefer the latest draft that actually yields an answer: a refinement
        # that rambles past the token limit must not destroy a good draft.
        answers = [self.extract(d, example) for d in drafts]
        final = next((a for a in reversed(answers) if a is not None), None)
        tokens_prompt, tokens_completion = self.tally(groups)
        extra: Dict[str, Any] = {
            "n_rounds_requested": self.n_rounds,
            "n_rounds_used": rounds_used,
            "verify": self.verify,
            "verdicts": verdicts,
            "verdict_texts": verdict_texts,
            "draft_answers": answers,
            "answer_changed": len(answers) > 1 and answers[0] != answers[-1],
            "stopped_early": self.verify and bool(verdicts) and verdicts[-1] is True,
        }
        return StrategyResult(
            final_answer=final,
            # Only the drafts are traces: the verifier's output is not a candidate
            # solution and must not be graded as one (it would corrupt pass@k).
            reasoning_traces=drafts,
            n_samples=len(drafts),
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=n_calls,
            sample_stats=self.per_sample_stats([[g[0]] for g in groups[: len(drafts)]]),
            extra=extra,
        )


@register_strategy("self_verify")
class SelfVerify(ReasoningStrategy):
    """Draft and verify, but never revise.

    Isolates the verifier: if `self_refine` beats this, the gain comes from
    revision; if this matches it, the gain was just the extra verification pass.
    """

    description = "Self-verification only (no revision)"

    def __init__(self, temperature: float = 0.0, **kw: Any) -> None:
        super().__init__(temperature=temperature, **kw)
        self.temperature = float(temperature)

    def run(
        self, example: Example, backend: GenerationBackend, params: GenParams
    ) -> StrategyResult:
        base = self.gen(params, n=1, temperature=self.temperature)
        prompt = self.user_prompt(example, instruction=COT_INSTRUCTION)
        out = backend.generate(
            [prompt], base.with_(seed=self.sample_seed(example, params, "draft"))
        )
        draft = out[0][0].text

        verify_prompt = continue_chat(prompt, draft, VERIFY_INSTRUCTION)
        vout = backend.generate(
            [verify_prompt],
            base.with_(
                max_new_tokens=min(256, base.max_new_tokens),
                seed=self.sample_seed(example, params, "verify"),
            ),
        )
        verdict_text = vout[0][0].text
        groups = [out[0], vout[0]]
        tokens_prompt, tokens_completion = self.tally(groups)
        return StrategyResult(
            final_answer=self.extract(draft, example),
            reasoning_traces=[draft],
            n_samples=1,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            n_calls=2,
            sample_stats=self.per_sample_stats([out[0]]),
            extra={
                "verdict": parse_verdict(verdict_text),
                "verdict_text": verdict_text,
            },
        )
