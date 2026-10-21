"""Core data contracts shared by every module in the harness.

These types are the *stable* interface. Datasets produce `Example`s, backends
consume `Prompt`s and produce `Completion`s, and strategies turn an `Example`
plus a backend into a `StrategyResult`. A new reasoning method only needs to
implement `ReasoningStrategy.run` and return a `StrategyResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Union

# A prompt is either a raw string (completion-style) or a list of chat messages
# in OpenAI format: [{"role": "user", "content": "..."}].
ChatMessage = Dict[str, str]
Prompt = Union[str, List[ChatMessage]]

#: Answer types drive answer extraction and equivalence checking.
ANSWER_TYPES = ("math", "mc", "text", "bool")


@dataclass
class Example:
    """One benchmark item in the unified schema.

    `id` is the dataset-local stable identifier (e.g. "gsm8k/test/42"). The
    per-run deterministic result key is derived separately in `checkpointing`.
    """

    id: str
    question: str
    gold_answer: str
    gold_reasoning: Optional[str] = None
    #: Ordered option strings for multiple-choice items (without letter labels).
    choices: Optional[List[str]] = None
    #: One of ANSWER_TYPES; selects the grading path.
    answer_type: str = "math"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"answer_type must be one of {ANSWER_TYPES}, got {self.answer_type!r}"
            )
        self.gold_answer = str(self.gold_answer)

    @property
    def choice_letters(self) -> List[str]:
        n = len(self.choices or [])
        return [chr(ord("A") + i) for i in range(n)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Example":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class GenParams:
    """Sampling parameters. Backends must honour every field or raise."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    #: Number of independent samples per prompt.
    n: int = 1
    #: Strings that terminate generation; the stop text is stripped from output.
    stop: Sequence[str] = ()
    #: Per-call seed. Backends should make sampling reproducible given this.
    seed: Optional[int] = None
    #: Per-token logprobs of the chosen tokens. `True`/`1` returns the chosen
    #: token's logprob; an int > 1 additionally requests that many top
    #: alternatives per position (needed for confidence/entropy-style scorers).
    #: Backends fill `Completion.top_logprobs` best-effort; use `n_logprobs`.
    logprobs: bool | int = False
    repetition_penalty: float = 1.0

    def with_(self, **kw: Any) -> "GenParams":
        """Return a copy with overrides (strategies vary temperature/n a lot)."""
        d = asdict(self)
        d.update(kw)
        return GenParams(**d)

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    @property
    def n_logprobs(self) -> int:
        """0 = off, 1 = chosen token only, k > 1 = also k top alternatives."""
        if not self.logprobs:
            return 0
        return max(1, int(self.logprobs))


@dataclass
class Completion:
    """One sampled continuation plus the token accounting we bill against."""

    text: str
    tokens_prompt: int = 0
    tokens_completion: int = 0
    #: "stop" | "length" | "eos" | "error"
    finish_reason: str = "stop"
    #: Per-token logprobs of the sampled tokens, when requested.
    logprobs: Optional[List[float]] = None
    #: Sum of `logprobs`; used by best-of-N scorers.
    cumulative_logprob: Optional[float] = None
    #: Optional per-position {token: logprob} of the top alternatives, when
    #: `GenParams.n_logprobs > 1`. Best-effort: not every backend supplies it.
    top_logprobs: Optional[List[Dict[str, float]]] = None

    @property
    def mean_logprob(self) -> Optional[float]:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)

    def stats(self) -> Dict[str, Any]:
        """Compact per-sample accounting, cheap enough to store for every sample.

        Written into `StrategyResult.sample_stats` so that offline analysis can
        do token-matched comparisons (accuracy vs *tokens*, not vs sample count)
        and confidence-weighted voting without re-running inference.
        """
        return {
            "tokens_completion": self.tokens_completion,
            "tokens_prompt": self.tokens_prompt,
            "finish_reason": self.finish_reason,
            "mean_logprob": (
                round(self.mean_logprob, 6) if self.mean_logprob is not None else None
            ),
            "cumulative_logprob": (
                round(self.cumulative_logprob, 6)
                if self.cumulative_logprob is not None
                else None
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyResult:
    """What a reasoning strategy returns for a single example.

    `extra` is the escape hatch for method-specific diagnostics (per-sample
    answers, verifier scores, search tree stats, ...). It must be
    JSON-serialisable: it is written verbatim into the results JSONL.
    """

    final_answer: Optional[str]
    reasoning_traces: List[str] = field(default_factory=list)
    n_samples: int = 1
    tokens_prompt: int = 0
    tokens_completion: int = 0
    #: Number of backend generate() round-trips; a proxy for latency cost.
    n_calls: int = 0
    #: Per-sample accounting, aligned index-for-index with `reasoning_traces`
    #: (see `Completion.stats`). Enables budget-matched (equal output tokens)
    #: method comparisons and confidence-weighted voting offline. Strategies
    #: should populate it via `ReasoningStrategy.per_sample_stats`.
    sample_stats: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def tokens_total(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResultRecord:
    """One line of the append-only results JSONL (raw, *ungraded*).

    Grading is a separate cached pass so that fixing the grader never costs GPU
    time. Nothing in this record depends on the grader.
    """

    uid: str
    example_id: str
    dataset: str
    model: str
    strategy: str
    seed: int
    config_hash: str
    index: int
    question: str
    gold_answer: str
    answer_type: str
    final_answer: Optional[str]
    reasoning_traces: List[str]
    n_samples: int
    tokens_prompt: int
    tokens_completion: int
    n_calls: int
    latency_s: float
    #: Elicitation-configuration id (`c0`..`c6`), or None when the factor is
    #: unused. This is the crossed factor the paper's variance decomposition is
    #: about, so it is a first-class column rather than something buried in
    #: `extra`: every analysis groups on (item, model, config, seed).
    elicitation: Optional[str] = None
    #: Dataset config/subset. GSM-Symbolic `main` and `p2` are different cells of
    #: the distribution-shift arm and must never be pooled by accident.
    subset: Optional[str] = None
    #: Per-sample token/logprob accounting, aligned with `reasoning_traces`.
    sample_stats: List[Dict[str, Any]] = field(default_factory=list)
    #: Whitelisted item metadata needed by the analysis: `template_id` (so
    #: GSM-Symbolic bootstraps resample templates rather than instances, which are
    #: not independent), `tier`, and difficulty labels.
    meta: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    #: ISO-8601 UTC timestamp of completion.
    finished_at: str = ""
    #: Set when the example failed; `final_answer` is then None.
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResultRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
