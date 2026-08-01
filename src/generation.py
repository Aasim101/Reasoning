"""Generation backend interface, batching/stop-sequence plumbing, MockBackend.

Every reasoning strategy talks *only* to `GenerationBackend.generate`, so the
same strategy code runs against transformers, vLLM, or the CPU-only
`MockBackend`. Concrete GPU backends live in `src/models.py`.

Token accounting is enforced here (in the base class) rather than trusted to
each backend, because token cost per correct answer is a headline metric.
"""

from __future__ import annotations

import logging
import math
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .types import ChatMessage, Completion, Example, GenParams, Prompt
from .utils import chunked, stable_hash

log = logging.getLogger(__name__)


@dataclass
class BackendStats:
    """Cumulative counters; the runner logs these and stores them in state."""

    n_calls: int = 0
    n_prompts: int = 0
    n_completions: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    generate_seconds: float = 0.0
    oom_retries: int = 0

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["tokens_total"] = self.tokens_prompt + self.tokens_completion
        if self.generate_seconds > 0:
            d["completion_tokens_per_s"] = round(
                self.tokens_completion / self.generate_seconds, 2
            )
        return d


class GenerationBackend(ABC):
    """Batched text generation with a uniform contract.

    Subclasses implement `_generate`, which receives at most `batch_size`
    prompts and must return one list of `params.n` completions per prompt, in
    order. The public `generate` adds chunking, stop-sequence trimming, token
    accounting, and OOM backoff.
    """

    name: str = "base"
    supports_logprobs: bool = False
    #: Whether the backend natively applies a chat template to message lists.
    supports_chat: bool = True

    def __init__(self, batch_size: int = 8, min_batch_size: int = 1) -> None:
        self.batch_size = max(1, int(batch_size))
        self.min_batch_size = max(1, int(min_batch_size))
        self.stats = BackendStats()

    # ------------------------------------------------------------------ public
    def generate(
        self, prompts: Sequence[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        if not prompts:
            return []
        if params.n < 1:
            raise ValueError("GenParams.n must be >= 1")
        out: List[List[Completion]] = []
        t0 = time.perf_counter()
        for batch in chunked(list(prompts), self.batch_size):
            out.extend(self._generate_with_backoff(batch, params))
        self.stats.generate_seconds += time.perf_counter() - t0
        self.stats.n_calls += 1
        self.stats.n_prompts += len(prompts)
        for group in out:
            self.stats.n_completions += len(group)
            for c in group:
                self.stats.tokens_prompt += c.tokens_prompt
                self.stats.tokens_completion += c.tokens_completion
        return out

    def generate_one(self, prompt: Prompt, params: GenParams) -> Completion:
        """Convenience for single-prompt, single-sample calls."""
        groups = self.generate([prompt], params.with_(n=1))
        return groups[0][0]

    # -------------------------------------------------------------- subclasses
    @abstractmethod
    def _generate(
        self, prompts: List[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        """Generate for one batch. Must return len(prompts) groups of n items."""

    def count_tokens(self, text: str) -> int:
        """Token count. Subclasses with a tokenizer should override."""
        return approx_token_count(text)

    def render(self, prompt: Prompt) -> str:
        """Render a prompt to the exact string the model sees (for logging)."""
        return prompt if isinstance(prompt, str) else messages_to_text(prompt)

    @property
    def info(self) -> Dict[str, Any]:
        """Backend/hardware provenance, recorded in the run manifest."""
        return {"backend": self.name, "batch_size": self.batch_size}

    def close(self) -> None:
        """Release GPU memory. Safe to call more than once."""

    def __enter__(self) -> "GenerationBackend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # --------------------------------------------------------------- internals
    def _generate_with_backoff(
        self, batch: List[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        """Halve the batch on CUDA OOM instead of losing the whole session."""
        size = len(batch)
        while True:
            try:
                groups = self._generate(batch[:size], params)
                validate_groups(groups, expected_prompts=len(batch[:size]), n=params.n)
                rest_groups: List[List[Completion]] = []
                if size < len(batch):
                    rest_groups = self._generate_with_backoff(batch[size:], params)
                return [postprocess_group(g, params) for g in groups] + rest_groups
            except Exception as exc:  # noqa: BLE001 - we re-raise unless OOM
                if not self._is_oom(exc) or size <= self.min_batch_size:
                    raise
                size = max(self.min_batch_size, size // 2)
                self.stats.oom_retries += 1
                log.warning(
                    "backend OOM (%s); retrying with batch_size=%d", type(exc).__name__, size
                )
                self._empty_cache()

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg or type(exc).__name__ == "OutOfMemoryError"

    @staticmethod
    def _empty_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch optional on CPU dev boxes
            pass


# ------------------------------------------------------------------- utilities
def approx_token_count(text: str) -> int:
    """Tokenizer-free estimate (~4 chars/token) for mock and fallback paths."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def messages_to_text(messages: Sequence[ChatMessage]) -> str:
    """Generic chat rendering used when no tokenizer chat template exists."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def apply_stop(text: str, stop: Iterable[str]) -> tuple[str, bool]:
    """Truncate at the earliest stop string. Returns (text, was_truncated)."""
    cut = len(text)
    hit = False
    for s in stop or ():
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and idx < cut:
            cut = idx
            hit = True
    return (text[:cut], hit) if hit else (text, False)


def postprocess_group(group: List[Completion], params: GenParams) -> List[Completion]:
    """Apply stop sequences and fill in derived logprob fields."""
    for c in group:
        if params.stop:
            trimmed, hit = apply_stop(c.text, params.stop)
            if hit:
                c.text = trimmed
                c.finish_reason = "stop"
        if c.logprobs and c.cumulative_logprob is None:
            c.cumulative_logprob = float(sum(c.logprobs))
    return group


def validate_groups(groups: Sequence[Sequence[Completion]], expected_prompts: int, n: int) -> None:
    if len(groups) != expected_prompts:
        raise RuntimeError(
            f"backend returned {len(groups)} completion groups for {expected_prompts} prompts"
        )
    for i, g in enumerate(groups):
        if len(g) != n:
            raise RuntimeError(
                f"backend returned {len(g)} samples for prompt {i}, expected n={n}"
            )


# ------------------------------------------------------------------ MockBackend
_STEP_TEMPLATES = (
    "First, I identify the quantities that the question gives me.",
    "Next, I set up the relationship between those quantities.",
    "Then I carry out the arithmetic carefully, one operation at a time.",
    "I double-check the intermediate value before committing to it.",
    "Finally I state the result in the requested format.",
)


class MockBackend(GenerationBackend):
    """Deterministic, CPU-only, model-free backend for tests and dry runs.

    It fabricates a chain-of-thought and a final answer. If gold answers have
    been registered via `register_golds`, it emits the gold answer with
    probability `accuracy` and a plausible distractor otherwise, which makes
    end-to-end accuracy/metric tests meaningful without a model.

    Output is a deterministic function of (prompt, sample index, seed), so
    resume tests produce byte-identical records across restarts.
    """

    name = "mock"
    supports_logprobs = True

    def __init__(
        self,
        batch_size: int = 8,
        accuracy: float = 0.6,
        seed: int = 0,
        latency_s: float = 0.0,
        n_steps: int = 3,
    ) -> None:
        super().__init__(batch_size=batch_size)
        self.accuracy = float(accuracy)
        self.seed = int(seed)
        self.latency_s = float(latency_s)
        self.n_steps = int(n_steps)
        #: (question, gold_answer, answer_type, choices) sorted longest-first.
        self._golds: List[tuple[str, str, str, Optional[List[str]]]] = []

    # -- test/dry-run hook: the runner calls this when it exists on a backend.
    def register_golds(self, examples: Iterable[Example]) -> None:
        for ex in examples:
            q = " ".join(ex.question.split())
            if q:
                self._golds.append((q, ex.gold_answer, ex.answer_type, ex.choices))
        self._golds.sort(key=lambda t: -len(t[0]))

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "batch_size": self.batch_size,
            "accuracy": self.accuracy,
            "seed": self.seed,
            "n_registered_golds": len(self._golds),
        }

    def _lookup_gold(self, text: str) -> Optional[tuple[str, str, Optional[List[str]]]]:
        norm = " ".join(text.split())
        for q, gold, atype, choices in self._golds:
            if q and q in norm:
                return gold, atype, choices
        return None

    def _generate(
        self, prompts: List[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        if self.latency_s:
            time.sleep(self.latency_s * len(prompts))
        groups: List[List[Completion]] = []
        for prompt in prompts:
            text = self.render(prompt)
            n_prompt_tokens = self.count_tokens(text)
            group: List[Completion] = []
            for i in range(params.n):
                body = self._one_sample(text, i, params)
                # Emulate a max_new_tokens cutoff so truncation paths get tested.
                budget_chars = params.max_new_tokens * 4
                finish = "stop"
                if len(body) > budget_chars:
                    body = body[:budget_chars]
                    finish = "length"
                n_completion_tokens = min(
                    self.count_tokens(body), params.max_new_tokens
                )
                rng = self._rng(text, i, params)
                group.append(
                    Completion(
                        text=body,
                        tokens_prompt=n_prompt_tokens,
                        tokens_completion=n_completion_tokens,
                        finish_reason=finish,
                        logprobs=(
                            [-abs(rng.gauss(0.35, 0.2)) for _ in range(n_completion_tokens)]
                            if params.logprobs
                            else None
                        ),
                    )
                )
            groups.append(group)
        return groups

    def _rng(self, text: str, sample_idx: int, params: GenParams) -> random.Random:
        key = stable_hash(
            {
                "text": text,
                # Greedy decoding must be identical across the n samples, so the
                # sample index is dropped from the key.
                "i": 0 if params.greedy else sample_idx,
                "seed": self.seed,
                "call_seed": params.seed,
                "temperature": 0.0 if params.greedy else params.temperature,
            },
            length=16,
        )
        return random.Random(int(key, 16))

    def _one_sample(self, text: str, sample_idx: int, params: GenParams) -> str:
        rng = self._rng(text, sample_idx, params)
        found = self._lookup_gold(text)
        if found is None:
            answer = str(rng.randint(0, 99))
        else:
            gold, atype, choices = found
            # Correctness is probabilistic even when greedy: a mock that were
            # always right when temperature=0 would make every greedy strategy
            # score 1.0 and hide bugs in the metrics.
            correct = rng.random() < self.accuracy
            answer = gold if correct else _distractor(gold, atype, choices, rng)

        answer_line = rf"The answer is \boxed{{{answer}}}."
        wants_cot = "step by step" in text.lower()
        if not wants_cot:
            return answer_line
        steps = list(_STEP_TEMPLATES[: max(1, self.n_steps)])
        if not params.greedy:
            rng.shuffle(steps)
        lines = [
            "Let's think step by step.",
            *(f"Step {i + 1}: {s}" for i, s in enumerate(steps)),
            answer_line,
        ]
        body = "\n".join(lines)
        # If the caller's token budget cannot fit the reasoning, emit just the
        # answer instead of a trace that gets truncated before stating one - a
        # real model asked for a short answer would do the same.
        if len(body) > params.max_new_tokens * 4:
            return answer_line
        return body


def _distractor(
    gold: str, answer_type: str, choices: Optional[List[str]], rng: random.Random
) -> str:
    """A wrong-but-plausible answer, so mock accuracy is not trivially 1.0."""
    gold = (gold or "").strip()
    if answer_type == "mc":
        letters = [chr(ord("A") + i) for i in range(max(2, len(choices or []) or 4))]
        options = [c for c in letters if c != gold.upper()] or ["Z"]
        return rng.choice(options)
    if answer_type == "bool":
        return "False" if gold.lower() in {"true", "yes", "1"} else "True"
    m = re.fullmatch(r"-?\d+(?:\.\d+)?", gold.replace(",", ""))
    if m:
        val = float(m.group(0))
        delta = rng.choice([1, -1, 2, 10])
        out = val + delta
        return str(int(out)) if float(out).is_integer() else f"{out:g}"
    return f"{gold}{rng.choice(['x', ' + 1', '/2'])}" if gold else "42"
