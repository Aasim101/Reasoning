"""Tests for the real `transformers` path, using a tiny locally-built model.

No download and no GPU: a 2-layer randomly-initialised Llama plus a BPE
tokenizer trained on a few lines of text is enough to exercise every part of
`HFBackend` that would otherwise stay untested until the first GPU session -
left padding, chat templating, prompt/completion token accounting, decoding only
the new tokens, logprob gathering with n>1, stop sequences, and the
`dtype`/`torch_dtype` kwarg difference between transformers 4.x and 5.x.

The model's *outputs* are gibberish, which is fine: what is under test is the
plumbing, not the reasoning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.config import DataConfig, GenerationConfig, ModelConfig  # noqa: E402
from src.models import HFBackend, build_backend, detect_hardware, resolve_dtype  # noqa: E402
from src.types import GenParams  # noqa: E402

CORPUS = [
    "What is 12 + 30? Let's think step by step.",
    "The answer is 42. The final answer is 7.",
    "A pack holds 8 pencils. How many pencils are in 3 packs?",
    "Which planet is closest to the Sun? Options: Venus Mercury Mars Earth",
    "Step 1: identify the quantities. Step 2: apply the operation.",
    "You are a careful, concise reasoning assistant.",
    "Think step by step, then state the final answer.",
    "system user assistant boxed frac sqrt answer question problem solve",
    "0 1 2 3 4 5 6 7 8 9 10 20 30 42 100 1000",
]

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n"
    "{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal but genuine transformers checkpoint on disk."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = tmp_path_factory.mktemp("tiny_model")

    # ByteLevel with the full initial alphabet guarantees no <unk> for any input.
    backing = Tokenizer(models.BPE())
    backing.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backing.decoder = decoders.ByteLevel()
    backing.train_from_iterator(
        CORPUS,
        trainers.BpeTrainer(
            vocab_size=500,
            special_tokens=["<pad>", "<s>", "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backing,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        chat_template=CHAT_TEMPLATE,
    )
    tokenizer.save_pretrained(out)

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).save_pretrained(out)
    return out


@pytest.fixture(scope="module")
def hf_backend(tiny_model_dir: Path) -> HFBackend:
    cfg = ModelConfig(
        name_or_path=str(tiny_model_dir),
        backend="hf",
        dtype="float32",  # CPU has no usable fp16 kernels
        use_chat_template=True,
    )
    backend = HFBackend(cfg, batch_size=4)
    yield backend
    backend.close()


# ------------------------------------------------------------------ loading
def test_backend_loads_and_reports_provenance(hf_backend: HFBackend):
    info = hf_backend.info
    assert info["backend"] == "hf"
    assert info["n_params"] > 0
    assert info["use_chat_template"] is True
    assert "float32" in info["dtype"]
    # Left padding is mandatory for batched decoder-only generation.
    assert hf_backend.tokenizer.padding_side == "left"
    assert hf_backend.tokenizer.pad_token_id is not None


def test_build_backend_returns_hf(tiny_model_dir: Path):
    backend = build_backend(
        ModelConfig(name_or_path=str(tiny_model_dir), backend="hf", dtype="float32"),
        GenerationConfig(batch_size=2),
    )
    try:
        assert isinstance(backend, HFBackend)
        assert backend.batch_size == 2
    finally:
        backend.close()


def test_dtype_kwarg_is_introspected(tiny_model_dir: Path):
    """transformers 5.x renamed `torch_dtype` to `dtype`; both must work."""
    import inspect

    from transformers import AutoModelForCausalLM

    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    has_var_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    assert "dtype" in params or "torch_dtype" in params or has_var_kwargs
    backend = HFBackend(
        ModelConfig(name_or_path=str(tiny_model_dir), backend="hf", dtype="float32")
    )
    try:
        assert next(backend.model.parameters()).dtype is torch.float32
    finally:
        backend.close()


# ---------------------------------------------------------------- generation
def test_greedy_generation_accounting(hf_backend: HFBackend):
    prompt = [{"role": "user", "content": "What is 12 + 30?"}]
    groups = hf_backend.generate([prompt], GenParams(max_new_tokens=8, temperature=0.0, n=1))
    assert len(groups) == 1 and len(groups[0]) == 1
    completion = groups[0][0]
    assert isinstance(completion.text, str)
    assert completion.tokens_prompt > 0
    assert 0 < completion.tokens_completion <= 8
    assert completion.finish_reason in {"stop", "length"}


def test_chat_template_is_applied(hf_backend: HFBackend):
    rendered = hf_backend.render(
        [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "USER_MARKER"},
        ]
    )
    assert "SYSTEM_MARKER" in rendered and "USER_MARKER" in rendered
    assert "<|assistant|>" in rendered, "generation prompt must be appended"


def test_raw_string_prompt_passes_through(hf_backend: HFBackend):
    assert hf_backend.render("literal prompt") == "literal prompt"
    groups = hf_backend.generate(["literal prompt"], GenParams(max_new_tokens=4, n=1))
    assert groups[0][0].tokens_prompt > 0


def test_multiple_samples_per_prompt(hf_backend: HFBackend):
    params = GenParams(max_new_tokens=8, temperature=0.9, top_p=0.95, n=3, seed=7)
    groups = hf_backend.generate([[{"role": "user", "content": "count"}]], params)
    assert len(groups[0]) == 3
    prompt_tokens = {c.tokens_prompt for c in groups[0]}
    assert len(prompt_tokens) == 1, "samples of one prompt share its prompt length"


def test_batched_prompts_of_different_lengths(hf_backend: HFBackend):
    """Left padding must not corrupt the shorter prompts in a batch."""
    prompts = [
        [{"role": "user", "content": "short"}],
        [{"role": "user", "content": "a considerably longer question " * 4}],
        [{"role": "user", "content": "medium length question here"}],
    ]
    groups = hf_backend.generate(prompts, GenParams(max_new_tokens=6, n=1))
    assert len(groups) == 3
    counts = [g[0].tokens_prompt for g in groups]
    # Padding is excluded, so the reported lengths must actually differ.
    assert counts[1] > counts[2] > counts[0]
    assert all(g[0].tokens_completion > 0 for g in groups)


def test_logprobs_are_gathered_for_multiple_samples(hf_backend: HFBackend):
    params = GenParams(max_new_tokens=6, temperature=0.8, n=3, seed=11, logprobs=True)
    groups = hf_backend.generate([[{"role": "user", "content": "logprob test"}]], params)
    for completion in groups[0]:
        assert completion.logprobs, "logprobs were requested but not returned"
        assert len(completion.logprobs) == completion.tokens_completion
        assert all(lp <= 0.0 for lp in completion.logprobs)
        assert completion.mean_logprob is not None
        assert completion.cumulative_logprob == pytest.approx(sum(completion.logprobs))


def test_sampling_is_reproducible_given_a_seed(hf_backend: HFBackend):
    prompt = [{"role": "user", "content": "reproducible"}]
    params = GenParams(max_new_tokens=8, temperature=1.0, n=2, seed=123)
    first = hf_backend.generate([prompt], params)
    second = hf_backend.generate([prompt], params)
    assert [c.text for c in first[0]] == [c.text for c in second[0]]
    other = hf_backend.generate([prompt], params.with_(seed=456))
    assert [c.text for c in first[0]] != [c.text for c in other[0]]


def test_greedy_is_deterministic(hf_backend: HFBackend):
    prompt = [{"role": "user", "content": "greedy"}]
    params = GenParams(max_new_tokens=8, temperature=0.0, n=1)
    a = hf_backend.generate([prompt], params)[0][0].text
    b = hf_backend.generate([prompt], params)[0][0].text
    assert a == b


def test_max_new_tokens_is_respected(hf_backend: HFBackend):
    for limit in (1, 4, 12):
        groups = hf_backend.generate(
            [[{"role": "user", "content": "limit"}]],
            GenParams(max_new_tokens=limit, temperature=0.0, n=1),
        )
        assert groups[0][0].tokens_completion <= limit


def test_stop_sequences_do_not_break_generation(hf_backend: HFBackend):
    """Stop handling must be safe even when the stop string never appears."""
    groups = hf_backend.generate(
        [[{"role": "user", "content": "stopping"}]],
        GenParams(max_new_tokens=8, temperature=0.0, n=1, stop=["\n\nQuestion:"]),
    )
    assert groups[0][0].tokens_completion > 0
    assert "\n\nQuestion:" not in groups[0][0].text


def test_count_tokens_uses_the_real_tokenizer(hf_backend: HFBackend):
    assert hf_backend.count_tokens("") == 0
    short = hf_backend.count_tokens("hello")
    long = hf_backend.count_tokens("hello " * 20)
    assert 0 < short < long


def test_greedy_with_n_greater_than_one(hf_backend: HFBackend):
    """transformers rejects num_return_sequences>1 without sampling.

    Greedy decoding is deterministic, so the backend generates once and
    replicates: the contract (n completions per prompt) still holds, the samples
    are identical, and they must be independent objects.
    """
    groups = hf_backend.generate(
        [[{"role": "user", "content": "greedy many"}]],
        GenParams(max_new_tokens=6, temperature=0.0, n=3),
    )
    assert len(groups[0]) == 3
    assert len({c.text for c in groups[0]}) == 1
    groups[0][0].text = "mutated"
    assert groups[0][1].text != "mutated", "replicated samples must not be aliased"


def test_backend_stats_accumulate(hf_backend: HFBackend):
    before = hf_backend.stats.tokens_completion
    hf_backend.generate(
        [[{"role": "user", "content": "stats"}]],
        GenParams(max_new_tokens=4, temperature=0.8, n=2),
    )
    assert hf_backend.stats.tokens_completion > before
    assert hf_backend.stats.n_completions >= 2


# ------------------------------------------------------------------- dtypes
def test_resolve_dtype_downgrades_bf16_on_turing_and_pascal():
    """T4 (sm75) and P100 (sm60) have no bf16: the request must be downgraded."""
    from src.models import HardwareInfo

    turing = HardwareInfo(cuda_available=True, supports_bf16=False, gpu_capabilities=["75"])
    assert resolve_dtype("bfloat16", turing) is torch.float16
    assert resolve_dtype("bf16", turing) is torch.float16
    assert resolve_dtype("float16", turing) is torch.float16
    assert resolve_dtype("auto", turing) is torch.float16

    ampere = HardwareInfo(cuda_available=True, supports_bf16=True, gpu_capabilities=["80"])
    assert resolve_dtype("bfloat16", ampere) is torch.bfloat16
    assert resolve_dtype("auto", ampere) is torch.bfloat16

    cpu = HardwareInfo(cuda_available=False)
    assert resolve_dtype("auto", cpu) is torch.float32
    assert resolve_dtype("float16", cpu) is torch.float32, "fp16 on CPU is pointless"

    with pytest.raises(ValueError):
        resolve_dtype("int4", cpu)


def test_detect_hardware_with_torch_present():
    info = detect_hardware()
    assert info.torch_version == torch.__version__
    assert isinstance(info.cuda_available, bool)
    if not info.cuda_available:
        assert info.supports_bf16 is False


# --------------------------------------------------------------- end to end
def test_runner_end_to_end_with_a_real_model(tiny_model_dir: Path, tmp_path: Path):
    """The whole pipeline against a genuine transformers model, on CPU."""
    from src.checkpointing import load_records
    from src.config import load_config
    from src.runner import run

    repo_root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        str(repo_root / "configs" / "base.yaml"),
        [
            f"model.name_or_path={tiny_model_dir.as_posix()}",
            "model.backend=hf",
            "model.dtype=float32",
            "data.name=toy",
            "data.subset=null",
            "data.subsample=3",
            "generation.max_new_tokens=8",
            "generation.batch_size=2",
            "strategy.name=cot_zeroshot",
            f"runtime.out_dir={tmp_path.as_posix()}",
            "runtime.time_budget_hours=1.0",
            "runtime.log_every=100",
        ],
    )
    summary = run(cfg)
    assert summary.is_complete
    assert summary.n_completed_this_session == 3
    assert summary.n_errors_this_session == 0
    assert summary.tokens_completion > 0

    records = load_records(cfg.results_path)
    assert len(records) == 3
    for record in records:
        assert record["error"] is None
        assert record["reasoning_traces"]
        assert record["tokens_prompt"] > 0
        assert record["tokens_completion"] > 0
        assert record["sample_stats"]
    # A random-weights model will not be right, but grading must still run.
    assert cfg.graded_path.exists()
    assert summary.grading is not None
    assert 0.0 <= summary.grading["accuracy"] <= 1.0
