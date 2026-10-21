"""Strategy tests against MockBackend: no GPU, no torch, no network."""

from __future__ import annotations

import json
from typing import Any, List

import pytest

from src.config import GenerationConfig, ModelConfig
from src.generation import GenerationBackend, MockBackend
from src.models import (
    HardwareInfo,
    bitsandbytes_available,
    build_backend,
    detect_hardware,
    estimate_model_vram_gb,
    vllm_available,
)
from src.strategies import available_strategies, build_strategy
from src.strategies.base import ReasoningStrategy
from src.strategies.best_of_n import SCORER_REGISTRY, register_scorer, resolve_scorer
from src.strategies.self_refine import parse_verdict
from src.types import Completion, Example, GenParams, StrategyResult


def base_params(**kw: Any) -> GenParams:
    defaults = dict(max_new_tokens=128, temperature=0.0, n=1, seed=1234, logprobs=True)
    defaults.update(kw)
    return GenParams(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def backend(toy_examples) -> MockBackend:
    mock = MockBackend(batch_size=4, accuracy=0.7, seed=5)
    mock.register_golds(toy_examples)
    return mock


# ------------------------------------------------------------------- registry
def test_expected_strategies_are_registered():
    names = available_strategies()
    for expected in (
        "direct",
        "cot_zeroshot",
        "cot_fewshot",
        "self_consistency",
        "best_of_n",
        "self_refine",
        "self_verify",
    ):
        assert expected in names


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_every_strategy_is_instantiable_and_describable(name: str):
    strategy = build_strategy(name)
    assert isinstance(strategy, ReasoningStrategy)
    described = strategy.describe()
    assert described["name"] and described["class"]
    json.dumps(described)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError):
        build_strategy("no_such_strategy")


def test_strategy_can_be_loaded_by_import_path():
    strategy = build_strategy("examples.method_template:TemplateMethod", k=2)
    assert isinstance(strategy, ReasoningStrategy)
    assert strategy.params["k"] == 2


# -------------------------------------------------------------------- results
@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_every_strategy_returns_a_wellformed_result(
    name: str, backend: MockBackend, toy_examples
):
    strategy = build_strategy(name)
    for example in toy_examples:
        result = strategy.run(example, backend, base_params())
        assert isinstance(result, StrategyResult)
        assert result.reasoning_traces, f"{name} produced no traces"
        assert all(isinstance(t, str) for t in result.reasoning_traces)
        assert result.n_samples >= 1
        assert result.tokens_completion > 0, f"{name} reported no completion tokens"
        assert result.tokens_prompt > 0
        assert result.n_calls >= 1
        # extra is written verbatim into JSONL, so it must serialise.
        json.dumps(result.extra)
        json.dumps(result.sample_stats)
        assert len(result.sample_stats) == len(result.reasoning_traces)


@pytest.mark.parametrize("name", sorted(available_strategies()))
def test_strategies_are_deterministic_given_a_seed(
    name: str, backend: MockBackend, toy_examples
):
    example = toy_examples[0]
    first = build_strategy(name).run(example, backend, base_params())
    second = build_strategy(name).run(example, backend, base_params())
    assert first.final_answer == second.final_answer
    assert first.reasoning_traces == second.reasoning_traces


def test_a_different_seed_changes_sampled_output(backend: MockBackend, toy_examples):
    strategy = build_strategy("self_consistency", k=6, temperature=0.8)
    a = strategy.run(toy_examples[0], backend, base_params(seed=1))
    b = strategy.run(toy_examples[0], backend, base_params(seed=2))
    assert a.reasoning_traces != b.reasoning_traces


# ------------------------------------------------------------ self-consistency
def test_self_consistency_sample_count_and_vote_info(backend: MockBackend, toy_examples):
    strategy = build_strategy("self_consistency", k=5, temperature=0.8)
    result = strategy.run(toy_examples[0], backend, base_params())
    assert result.n_samples == 5
    assert len(result.reasoning_traces) == 5
    assert len(result.extra["sample_answers"]) == 5
    assert result.extra["k"] == 5
    assert sum(result.extra["vote_counts"].values()) <= 5
    assert result.n_calls == 1, "k samples must come from one batched call"


def test_self_consistency_unbatched_fallback_matches_sample_count(
    backend: MockBackend, toy_examples
):
    strategy = build_strategy("self_consistency", k=3, temperature=0.8, batch_calls=False)
    result = strategy.run(toy_examples[0], backend, base_params())
    assert result.n_samples == 3
    assert result.n_calls == 3


def test_token_accounting_counts_the_prompt_once_per_call(
    backend: MockBackend, toy_examples
):
    """Billing the prompt k times would overstate self-consistency's cost by k."""
    k = 8
    single = build_strategy("cot_zeroshot").run(toy_examples[0], backend, base_params())
    many = build_strategy("self_consistency", k=k, temperature=0.8).run(
        toy_examples[0], backend, base_params()
    )
    assert many.tokens_prompt < 2 * single.tokens_prompt
    assert many.tokens_completion > 2 * single.tokens_completion


def test_tally_helper_directly():
    group = [
        Completion(text="a", tokens_prompt=10, tokens_completion=5),
        Completion(text="b", tokens_prompt=10, tokens_completion=7),
    ]
    prompt, completion = ReasoningStrategy.tally([group])
    assert prompt == 10, "prompt tokens are counted once per group"
    assert completion == 12


def test_per_sample_stats_helper():
    group = [
        Completion(text="a", tokens_completion=5, logprobs=[-0.5, -0.5]),
        Completion(text="b", tokens_completion=7, finish_reason="length"),
    ]
    stats = ReasoningStrategy.per_sample_stats([group])
    assert len(stats) == 2
    assert stats[0]["mean_logprob"] == pytest.approx(-0.5)
    assert stats[1]["finish_reason"] == "length"
    assert stats[1]["mean_logprob"] is None


# ------------------------------------------------------------------ best-of-N
def test_best_of_n_records_scores_and_picks_argmax(backend: MockBackend, toy_examples):
    strategy = build_strategy("best_of_n", n=4, temperature=0.8, scorer="logprob")
    result = strategy.run(toy_examples[0], backend, base_params())
    scores = result.extra["scores"]
    assert len(scores) == 4
    assert result.extra["best_index"] == max(range(4), key=lambda i: scores[i])
    assert result.extra["scorer"] == "logprob"
    assert len(result.reasoning_traces) == 4


def test_best_of_n_length_scorer_prefers_shorter(backend: MockBackend, toy_examples):
    strategy = build_strategy("best_of_n", n=4, temperature=0.8, scorer="length")
    result = strategy.run(toy_examples[0], backend, base_params())
    stats = result.sample_stats
    best = result.extra["best_index"]
    shortest = min(range(len(stats)), key=lambda i: stats[i]["tokens_completion"])
    assert stats[best]["tokens_completion"] == stats[shortest]["tokens_completion"]


def test_best_of_n_accepts_a_pluggable_scorer(backend: MockBackend, toy_examples):
    """The extension point a novel verifier would use."""

    @register_scorer("test_prefers_last")
    def _prefers_last(example, text, completion, backend_) -> float:  # noqa: ANN001
        return float(len(text))

    try:
        assert "test_prefers_last" in SCORER_REGISTRY
        strategy = build_strategy(
            "best_of_n", n=3, temperature=0.8, scorer="test_prefers_last"
        )
        result = strategy.run(toy_examples[0], backend, base_params())
        scores = result.extra["scores"]
        assert result.extra["best_index"] == max(range(3), key=lambda i: scores[i])
    finally:
        SCORER_REGISTRY.pop("test_prefers_last", None)


def test_best_of_n_scorer_by_import_path():
    scorer = resolve_scorer("src.strategies.best_of_n:score_mean_logprob")
    assert callable(scorer)
    with pytest.raises(KeyError):
        resolve_scorer("nope_not_a_scorer")


def test_best_of_n_survives_a_broken_scorer(backend: MockBackend, toy_examples):
    @register_scorer("test_explodes")
    def _explodes(example, text, completion, backend_) -> float:  # noqa: ANN001
        raise RuntimeError("boom")

    try:
        strategy = build_strategy("best_of_n", n=2, temperature=0.8, scorer="test_explodes")
        result = strategy.run(toy_examples[0], backend, base_params())
        assert result.extra["scores"] == [None, None]
    finally:
        SCORER_REGISTRY.pop("test_explodes", None)


# ----------------------------------------------------------------- self-refine
def test_self_refine_respects_rounds_and_records_verdicts(
    backend: MockBackend, toy_examples
):
    strategy = build_strategy("self_refine", n_rounds=2, verify=True, stop_when_verified=False)
    result = strategy.run(toy_examples[0], backend, base_params())
    assert result.extra["n_rounds_requested"] == 2
    assert result.extra["n_rounds_used"] == 2
    assert len(result.extra["verdicts"]) == 2
    assert len(result.reasoning_traces) == 3, "one draft plus two revisions"
    assert "answer_changed" in result.extra


def test_self_refine_zero_rounds_is_just_a_draft(backend: MockBackend, toy_examples):
    result = build_strategy("self_refine", n_rounds=0).run(
        toy_examples[0], backend, base_params()
    )
    assert len(result.reasoning_traces) == 1
    assert result.extra["n_rounds_used"] == 0


def test_self_verify_does_not_revise(backend: MockBackend, toy_examples):
    result = build_strategy("self_verify").run(toy_examples[0], backend, base_params())
    assert len(result.reasoning_traces) == 1
    assert result.n_calls == 2, "one draft plus one verification"
    assert "verdict" in result.extra


@pytest.mark.parametrize(
    "text,expected",
    [
        ("VERDICT: CORRECT because ...", True),
        ("verdict: incorrect", False),
        ("This looks correct to me", True),
        ("The step is wrong", False),
        ("no opinion at all", None),
    ],
)
def test_parse_verdict(text: str, expected: Any):
    assert parse_verdict(text) is expected


# --------------------------------------------------------------- direct / CoT
def test_direct_is_cheaper_than_cot(backend: MockBackend, toy_examples):
    example = toy_examples[0]
    direct = build_strategy("direct").run(example, backend, base_params())
    cot = build_strategy("cot_zeroshot").run(example, backend, base_params())
    assert direct.tokens_completion < cot.tokens_completion


def test_fewshot_adds_exemplars(backend: MockBackend, toy_examples):
    example = toy_examples[0]
    zero = build_strategy("cot_zeroshot").run(example, backend, base_params())
    few = build_strategy("cot_fewshot", n_shots=4).run(example, backend, base_params())
    assert few.extra["n_shots"] == 4
    assert few.tokens_prompt > zero.tokens_prompt


def test_prompt_style_overrides_wording(backend: MockBackend, toy_examples):
    """The declarative prompt-configuration hook used for elicitation axes."""
    strategy = build_strategy(
        "cot_zeroshot",
        style={
            "name": "c1",
            "system": "You are an expert mathematician.",
            "instruction": "Solve the problem step by step.",
            "hint": r"End your response with \boxed{}.",
        },
    )
    result = strategy.run(toy_examples[0], backend, base_params())
    assert result.reasoning_traces
    rendered = backend.render(
        strategy.user_prompt(toy_examples[0])
    )
    assert "expert mathematician" in rendered


# -------------------------------------------------------------------- backends
def test_build_backend_mock(toy_examples):
    backend = build_backend(
        ModelConfig(backend="mock", mock_accuracy=0.5), GenerationConfig(batch_size=3),
        examples=toy_examples,
    )
    assert isinstance(backend, MockBackend)
    assert backend.batch_size == 3
    assert backend.info["n_registered_golds"] == len(toy_examples)


def test_build_backend_rejects_unknown():
    with pytest.raises(ValueError):
        build_backend(ModelConfig(backend="nonsense"))


def test_build_backend_vllm_error_names_the_fallback():
    if vllm_available():  # pragma: no cover - depends on the environment
        pytest.skip("vLLM is installed here, so the error path cannot be exercised")
    with pytest.raises(RuntimeError, match="hf"):
        build_backend(ModelConfig(backend="vllm"))


def test_detect_hardware_works_without_torch():
    info = detect_hardware()
    assert isinstance(info, HardwareInfo)
    assert isinstance(info.cuda_available, bool)
    assert isinstance(info.n_gpus, int)
    assert info.cpu_count >= 1
    json.dumps(info.as_dict())
    assert info.summary()
    if not info.cuda_available:
        assert info.supports_bf16 is False
        assert info.n_gpus == 0


def test_availability_probes_never_raise():
    assert isinstance(vllm_available(), bool)
    assert isinstance(bitsandbytes_available(), bool)


def test_estimate_model_vram():
    # A 7B fp16 model is ~14 GB: too big for one 16 GB T4 once the KV cache is
    # added, which is exactly the decision this helper informs.
    assert 13.0 < estimate_model_vram_gb(7.0, "float16") < 15.0
    assert estimate_model_vram_gb(7.0, "float32") > estimate_model_vram_gb(7.0, "float16")
    assert estimate_model_vram_gb(7.0, "float16", load_in_4bit=True) < 5.0


def test_resolve_dtype_downgrades_bf16_without_torch():
    torch = pytest.importorskip("torch")
    from src.models import resolve_dtype

    hw = HardwareInfo(cuda_available=True, supports_bf16=False, gpu_capabilities=["75"])
    assert resolve_dtype("bfloat16", hw) is torch.float16
    assert resolve_dtype("float16", hw) is torch.float16
    hw_ampere = HardwareInfo(cuda_available=True, supports_bf16=True)
    assert resolve_dtype("bfloat16", hw_ampere) is torch.bfloat16
    with pytest.raises(ValueError):
        resolve_dtype("int4", hw)


# ---------------------------------------------------------------- mock backend
def test_mock_backend_contract(toy_examples):
    mock = MockBackend(batch_size=2, accuracy=1.0, seed=0)
    mock.register_golds(toy_examples)
    params = GenParams(max_new_tokens=64, temperature=0.7, n=3, logprobs=True)
    prompts = [
        [{"role": "user", "content": f"{ex.question} Think step by step."}]
        for ex in toy_examples
    ]
    groups = mock.generate(prompts, params)
    assert len(groups) == len(prompts)
    assert all(len(g) == 3 for g in groups)
    for group in groups:
        for completion in group:
            assert completion.tokens_completion > 0
            assert completion.logprobs and len(completion.logprobs) == completion.tokens_completion
    assert mock.stats.n_completions == 3 * len(prompts)


def test_mock_backend_greedy_is_identical_across_samples(toy_examples):
    mock = MockBackend(accuracy=0.5, seed=1)
    mock.register_golds(toy_examples)
    group = mock.generate([toy_examples[0].question], GenParams(n=4, temperature=0.0))[0]
    assert len({c.text for c in group}) == 1, "greedy decoding must not vary"


def test_mock_backend_accuracy_is_neither_zero_nor_one(toy_examples):
    """Guards the test suite itself: a mock that is always right hides bugs."""
    mock = MockBackend(accuracy=0.5, seed=3)
    mock.register_golds(toy_examples)
    strategy = build_strategy("cot_zeroshot")
    from src.answers import answers_equivalent

    verdicts = [
        answers_equivalent(
            strategy.run(ex, mock, base_params()).final_answer,
            ex.gold_answer,
            ex.answer_type,
            ex.choices,
        )
        for ex in toy_examples * 4
    ]
    assert 0 < sum(verdicts) < len(verdicts)


def test_backend_batching_and_oom_backoff():
    class FlakyBackend(GenerationBackend):
        """Raises a CUDA-style OOM until the batch is small enough."""

        name = "flaky"

        def __init__(self) -> None:
            super().__init__(batch_size=8)
            self.calls: List[int] = []

        def _generate(self, prompts, params):  # noqa: ANN001
            self.calls.append(len(prompts))
            if len(prompts) > 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
            return [
                [Completion(text="ok", tokens_completion=1) for _ in range(params.n)]
                for _ in prompts
            ]

    backend = FlakyBackend()
    groups = backend.generate(["a"] * 8, GenParams(n=1))
    assert len(groups) == 8
    assert backend.stats.oom_retries >= 2
    assert max(backend.calls) == 8 and min(backend.calls) <= 2


def test_backend_validates_group_shape():
    class BadBackend(GenerationBackend):
        name = "bad"

        def _generate(self, prompts, params):  # noqa: ANN001
            return [[Completion(text="only one")]]  # wrong: n=2 requested

    with pytest.raises(RuntimeError, match="samples for prompt"):
        BadBackend().generate(["a"], GenParams(n=2))


def test_stop_sequences_are_trimmed():
    class EchoBackend(GenerationBackend):
        name = "echo"

        def _generate(self, prompts, params):  # noqa: ANN001
            return [
                [Completion(text="answer 42\n\nQuestion: next", tokens_completion=9)]
                for _ in prompts
            ]

    groups = EchoBackend().generate(["x"], GenParams(n=1, stop=["\n\nQuestion:"]))
    assert groups[0][0].text == "answer 42"
    assert groups[0][0].finish_reason == "stop"
