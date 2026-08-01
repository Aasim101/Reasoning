"""Hardware detection and the concrete generation backends.

Everything heavy (torch, transformers, vllm) is imported lazily inside functions
so this module is importable - and testable - on a laptop with none of them
installed. That is what lets the whole harness be exercised on CPU with
`MockBackend`.

Kaggle-specific constraints encoded here:

* T4 is compute capability 7.5 and P100 is 6.0. **Neither supports bf16**, so
  fp16 is the default and a bf16 request is downgraded with a warning rather than
  failing three hours into a run.
* `device_map="auto"` shards a model across both T4s via accelerate; that is how
  a 7B fp16 model (~15 GB of weights) gets enough room for a KV cache.
* vLLM is optional. The transformers path is the default and must always work,
  because a vLLM install failure in the Kaggle image must not cost a session.
"""

from __future__ import annotations

import gc
import inspect
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .config import GenerationConfig, ModelConfig, RunConfig
from .generation import (
    GenerationBackend,
    MockBackend,
    approx_token_count,
    messages_to_text,
)
from .types import Completion, Example, GenParams, Prompt

log = logging.getLogger(__name__)

#: Compute capability at which bf16 tensor cores appear (Ampere).
BF16_MIN_CAPABILITY = (8, 0)


# ------------------------------------------------------------------- hardware
@dataclass
class HardwareInfo:
    """What we are actually running on. Recorded in every run manifest."""

    torch_version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    n_gpus: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_capabilities: List[str] = field(default_factory=list)
    per_gpu_vram_gb: List[float] = field(default_factory=list)
    per_gpu_free_gb: List[float] = field(default_factory=list)
    total_vram_gb: float = 0.0
    supports_bf16: bool = False
    cpu_count: int = 0
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    is_kaggle: bool = False
    driver: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        if not self.cuda_available:
            return (
                f"CPU only (torch={self.torch_version or 'absent'}, "
                f"{self.cpu_count} cores, {self.ram_gb:.0f} GB RAM)"
            )
        gpus = ", ".join(
            f"{name} {vram:.0f}GB sm{cap}"
            for name, vram, cap in zip(
                self.gpu_names, self.per_gpu_vram_gb, self.gpu_capabilities
            )
        )
        return (
            f"{self.n_gpus}x GPU [{gpus}] total {self.total_vram_gb:.0f} GB VRAM, "
            f"bf16={'yes' if self.supports_bf16 else 'NO'}, "
            f"cuda={self.cuda_version}, torch={self.torch_version}"
        )


def detect_hardware() -> HardwareInfo:
    """Probe the machine. Safe to call with torch absent."""
    info = HardwareInfo()
    info.is_kaggle = os.path.isdir("/kaggle") or bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
    )
    try:
        info.cpu_count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        info.cpu_count = os.cpu_count() or 0
    try:
        import psutil  # type: ignore

        info.ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        try:
            info.ram_gb = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 1
            )
        except (ValueError, AttributeError, OSError):
            info.ram_gb = 0.0
    try:
        info.disk_free_gb = round(
            shutil.disk_usage("/kaggle/working" if info.is_kaggle else ".").free / 1024**3, 1
        )
    except OSError:
        info.disk_free_gb = 0.0

    try:
        import torch
    except ImportError:
        return info

    info.torch_version = torch.__version__
    info.cuda_version = getattr(torch.version, "cuda", None)
    try:
        info.cuda_available = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver problems surface here
        info.cuda_available = False
    if not info.cuda_available:
        return info

    info.n_gpus = torch.cuda.device_count()
    capabilities = []
    for i in range(info.n_gpus):
        props = torch.cuda.get_device_properties(i)
        info.gpu_names.append(props.name)
        capability = (props.major, props.minor)
        capabilities.append(capability)
        info.gpu_capabilities.append(f"{props.major}{props.minor}")
        info.per_gpu_vram_gb.append(round(props.total_memory / 1024**3, 2))
        try:
            free, _total = torch.cuda.mem_get_info(i)
            info.per_gpu_free_gb.append(round(free / 1024**3, 2))
        except Exception:
            info.per_gpu_free_gb.append(float("nan"))
    info.total_vram_gb = round(sum(info.per_gpu_vram_gb), 2)
    # A machine is bf16-capable only if EVERY visible GPU is, since a sharded
    # model runs layers on all of them.
    info.supports_bf16 = bool(capabilities) and all(
        c >= BF16_MIN_CAPABILITY for c in capabilities
    )
    return info


def log_hardware(info: Optional[HardwareInfo] = None) -> HardwareInfo:
    """Detect (if needed) and log the hardware. Returns the info."""
    info = info or detect_hardware()
    log.info("hardware: %s", info.summary())
    if info.is_kaggle:
        log.info(
            "kaggle session: %.1f GB free disk, %d CPU cores, %.0f GB RAM",
            info.disk_free_gb,
            info.cpu_count,
            info.ram_gb,
        )
    if info.cuda_available and not info.supports_bf16:
        log.info(
            "no bf16 support on this GPU (compute capability %s) - fp16 will be used",
            ",".join(info.gpu_capabilities),
        )
    return info


_DTYPE_ALIASES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "f16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "f32": "float32",
    "auto": "auto",
}


def resolve_dtype(name: str, hw: Optional[HardwareInfo] = None) -> Any:
    """Map a dtype name to a torch dtype, downgrading bf16 when unsupported."""
    import torch

    key = _DTYPE_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {sorted(set(_DTYPE_ALIASES))}"
        )
    hw = hw or detect_hardware()
    if key == "auto":
        if not hw.cuda_available:
            return torch.float32
        key = "bfloat16" if hw.supports_bf16 else "float16"
    if key == "bfloat16" and hw.cuda_available and not hw.supports_bf16:
        log.warning(
            "bfloat16 requested but this GPU (compute capability %s) has no bf16 "
            "tensor cores - falling back to float16. T4 is sm75 and P100 is sm60; "
            "bf16 on either is unsupported or very slow.",
            ",".join(hw.gpu_capabilities) or "unknown",
        )
        key = "float16"
    if key == "float16" and not hw.cuda_available:
        log.warning("float16 on CPU is extremely slow; using float32 instead")
        key = "float32"
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[key]


def vllm_available() -> bool:
    """Is vLLM importable? Never raises, never imports torch eagerly."""
    try:
        import importlib.util

        return importlib.util.find_spec("vllm") is not None
    except Exception:  # pragma: no cover
        return False


def bitsandbytes_available() -> bool:
    """Is bitsandbytes importable? Never raises."""
    try:
        import importlib.util

        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # pragma: no cover
        return False


_BYTES_PER_PARAM = {"float32": 4.0, "float16": 2.0, "bfloat16": 2.0}


def estimate_model_vram_gb(
    n_params_b: float, dtype: str = "float16", load_in_4bit: bool = False
) -> float:
    """Rough weights-only VRAM estimate, for planning before spending quota.

    Excludes the KV cache and activations, which for a long-context batch can add
    several GB - so treat a number close to 16 GB as "will not fit on one T4".
    """
    per_param = 0.55 if load_in_4bit else _BYTES_PER_PARAM.get(
        _DTYPE_ALIASES.get(str(dtype).lower(), "float16"), 2.0
    )
    return round(n_params_b * 1e9 * per_param / 1024**3, 2)


def _supported_kwargs(fn: Any, candidates: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the kwargs a function actually accepts.

    `from_pretrained` renamed `torch_dtype` to `dtype` in transformers 5.x, and
    Kaggle's image version is not under our control, so the kwarg is chosen by
    introspection instead of by guessing.
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(candidates)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in params}


# ------------------------------------------------------------------- HF backend
class HFBackend(GenerationBackend):
    """HuggingFace `transformers` generation. The default and the fallback."""

    name = "hf"
    supports_logprobs = True

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int = 8,
        hw: Optional[HardwareInfo] = None,
    ) -> None:
        super().__init__(batch_size=batch_size)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.cfg = cfg
        self.hw = hw or detect_hardware()
        self.dtype = resolve_dtype(cfg.dtype, self.hw)
        path = cfg.resolved_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=cfg.trust_remote_code,
            revision=cfg.revision,
            # Decoder-only batched generation REQUIRES left padding: with right
            # padding the model continues from pad tokens and output is garbage.
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": cfg.trust_remote_code,
            "revision": cfg.revision,
            "low_cpu_mem_usage": True,
        }
        if cfg.load_in_4bit:
            load_kwargs["quantization_config"] = self._quantization_config()
        else:
            load_kwargs["dtype"] = self.dtype
        if self.hw.cuda_available:
            load_kwargs["device_map"] = cfg.device_map
        if cfg.attn_implementation:
            load_kwargs["attn_implementation"] = cfg.attn_implementation

        # `dtype` vs the older `torch_dtype`, plus any kwarg this version dropped.
        resolved = _supported_kwargs(AutoModelForCausalLM.from_pretrained, load_kwargs)
        if "dtype" in load_kwargs and "dtype" not in resolved:
            resolved["torch_dtype"] = load_kwargs["dtype"]
        dropped = set(load_kwargs) - set(resolved) - {"dtype"}
        if dropped:
            log.warning(
                "this transformers version does not accept %s; continuing without it",
                sorted(dropped),
            )

        log.info(
            "loading %s (dtype=%s, 4bit=%s, device_map=%s)",
            path,
            self.dtype,
            cfg.load_in_4bit,
            cfg.device_map if self.hw.cuda_available else "cpu",
        )
        self.model = AutoModelForCausalLM.from_pretrained(path, **resolved)
        self.model.eval()
        if not self.hw.cuda_available:
            self.model.to("cpu")
        self._device = next(self.model.parameters()).device
        self._n_params = sum(p.numel() for p in self.model.parameters())
        log.info(
            "loaded %s: %.2fB parameters on %s",
            path,
            self._n_params / 1e9,
            self._device,
        )

    def _quantization_config(self) -> Any:
        from transformers import BitsAndBytesConfig

        if not bitsandbytes_available():
            raise RuntimeError(
                "load_in_4bit=true requires bitsandbytes, which is NOT preinstalled "
                "on Kaggle. Install it with `pip install 'bitsandbytes>=0.50.0'`, or "
                "run in fp16 instead (a 7B fp16 model fits across 2x T4 with "
                "device_map=auto)."
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=self.cfg.bnb_quant_type,
            bnb_4bit_use_double_quant=self.cfg.bnb_double_quant,
            # fp16 compute: on Turing/Pascal bf16 compute is not an option.
            bnb_4bit_compute_dtype=resolve_dtype(self.cfg.bnb_compute_dtype, self.hw),
        )

    # ------------------------------------------------------------------ prompts
    def render(self, prompt: Prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        if self.cfg.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Some templates reject a system role. Merge it into the first
                # user turn rather than dropping the instruction entirely.
                merged = _merge_system_into_user(prompt)
                try:
                    return self.tokenizer.apply_chat_template(
                        merged, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    log.warning(
                        "chat template failed for this model; falling back to a "
                        "generic role-tagged rendering",
                        exc_info=True,
                    )
        return messages_to_text(prompt)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.cfg.resolved_path,
            "revision": self.cfg.revision,
            "dtype": str(self.dtype),
            "load_in_4bit": self.cfg.load_in_4bit,
            "quant_type": self.cfg.bnb_quant_type if self.cfg.load_in_4bit else None,
            "device_map": self.cfg.device_map,
            "device": str(self._device),
            "n_params": self._n_params,
            "batch_size": self.batch_size,
            "use_chat_template": bool(
                self.cfg.use_chat_template and getattr(self.tokenizer, "chat_template", None)
            ),
            "hardware": self.hw.summary(),
        }

    # --------------------------------------------------------------- generation
    def _generate(
        self, prompts: List[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        torch = self._torch
        texts = [self.render(p) for p in prompts]
        # Chat templates already contain BOS; adding specials again would double it.
        templated = all(not isinstance(p, str) for p in prompts)
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=not templated,
        ).to(self._device)
        prompt_len = int(enc["input_ids"].shape[1])
        prompt_token_counts = [int(x) for x in enc["attention_mask"].sum(dim=1).tolist()]

        # Greedy decoding is deterministic, so n samples would be n identical
        # strings - and transformers rejects num_return_sequences>1 without
        # sampling outright. Generate once and replicate, which is both correct
        # and n times cheaper.
        effective_n = 1 if params.greedy else int(params.n)
        if params.greedy and params.n > 1:
            log.debug(
                "greedy request for n=%d: generating once and replicating "
                "(set temperature > 0 for genuinely independent samples)",
                params.n,
            )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(params.max_new_tokens),
            "num_return_sequences": effective_n,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
        }
        if params.greedy:
            # Passing temperature/top_p with do_sample=False makes transformers
            # emit warnings and, in some versions, raise.
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(params.temperature),
                top_p=float(params.top_p),
            )
            if params.top_k and params.top_k > 0:
                gen_kwargs["top_k"] = int(params.top_k)
            if params.repetition_penalty and params.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = float(params.repetition_penalty)
            if params.seed is not None:
                torch.manual_seed(int(params.seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(params.seed))

        want_logprobs = params.n_logprobs > 0
        if want_logprobs:
            gen_kwargs["output_scores"] = True

        stopper = self._build_stopping_criteria(params, prompt_len)
        if stopper is not None:
            gen_kwargs["stopping_criteria"] = stopper

        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kwargs)

        sequences = out.sequences
        new_tokens = sequences[:, prompt_len:]
        scores = getattr(out, "scores", None) if want_logprobs else None
        eos_id = self.tokenizer.eos_token_id

        groups: List[List[Completion]] = []
        for prompt_index in range(len(prompts)):
            group: List[Completion] = []
            for sample_index in range(effective_n):
                row = prompt_index * effective_n + sample_index
                tokens = new_tokens[row]
                length = _completion_length(tokens, eos_id)
                text = self.tokenizer.decode(
                    tokens[:length], skip_special_tokens=True
                )
                finish = "stop" if length < tokens.shape[0] else "length"
                logprobs = (
                    self._gather_logprobs(scores, row, tokens, length)
                    if scores is not None
                    else None
                )
                group.append(
                    Completion(
                        text=text,
                        tokens_prompt=prompt_token_counts[prompt_index],
                        tokens_completion=int(length),
                        finish_reason=finish,
                        logprobs=logprobs,
                    )
                )
            while len(group) < params.n:
                # Independent copies: the base class trims stop sequences in
                # place, so sharing one object across samples would be a bug.
                group.append(replace(group[0]))
            groups.append(group)
        return groups

    def _gather_logprobs(
        self, scores: Sequence[Any], row: int, tokens: Any, length: int
    ) -> Optional[List[float]]:
        """Log-softmax of the chosen token at each step, for one output row.

        `scores` is a tuple over generation steps, each of shape
        [batch * num_return_sequences, vocab], so the row index already accounts
        for multiple samples per prompt. Failures degrade to None rather than
        killing a run that has otherwise produced good generations.
        """
        torch = self._torch
        try:
            out: List[float] = []
            with torch.inference_mode():
                for step in range(min(length, len(scores))):
                    step_scores = scores[step]
                    if step_scores is None or row >= step_scores.shape[0]:
                        break
                    logprob = torch.log_softmax(
                        step_scores[row].float(), dim=-1
                    )[int(tokens[step])]
                    value = float(logprob)
                    # Sampling post-processors (top-p) set filtered logits to
                    # -inf; that is not a usable score.
                    out.append(value if value > -1e30 else -100.0)
            return out or None
        except Exception:  # noqa: BLE001 - logprobs are optional, generations are not
            log.warning("failed to gather logprobs; continuing without them", exc_info=True)
            return None

    def _build_stopping_criteria(self, params: GenParams, prompt_len: int) -> Any:
        """Halt early when every sequence has emitted a stop string.

        Correctness does not depend on this: the base class trims stop sequences
        from the text afterwards. This only saves tokens, which on a 30 GPU-hour
        budget is worth the small complexity.
        """
        if not params.stop:
            return None
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except ImportError:  # pragma: no cover
            return None

        tokenizer = self.tokenizer
        stops = [s for s in params.stop if s]
        longest = max((len(s) for s in stops), default=0)

        class _StopOnStrings(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                generated = input_ids[:, prompt_len:]
                if generated.shape[1] == 0:
                    return False
                # Decode only a short tail: decoding the whole batch every step
                # would cost more than the tokens it saves.
                tail_tokens = generated[:, -(longest + 16) :]
                tails = tokenizer.batch_decode(tail_tokens, skip_special_tokens=True)
                return all(any(s in tail for s in stops) for tail in tails)

        return StoppingCriteriaList([_StopOnStrings()])

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is not None:
            del self.model
        gc.collect()
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _completion_length(tokens: Any, eos_id: Optional[int]) -> int:
    """Number of generated tokens up to and including the first EOS.

    Needed because `generate` pads finished sequences, and pad_token often *is*
    eos_token, so "count the non-pad tokens" would be wrong.
    """
    total = int(tokens.shape[0])
    if eos_id is None:
        return total
    nonzero = (tokens == eos_id).nonzero()
    if nonzero.numel() == 0:
        return total
    return int(nonzero[0].item()) + 1


def _merge_system_into_user(messages: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    if system_parts and rest:
        rest[0]["content"] = "\n\n".join([*system_parts, rest[0].get("content", "")]).strip()
    return rest


# ----------------------------------------------------------------- vLLM backend
class VLLMBackend(GenerationBackend):
    """vLLM offline batch generation: much faster, entirely optional.

    Notes for the Kaggle free tier:
    * `tensor_parallel_size=2` uses both T4s but needs contiguous VRAM and often
      `enforce_eager=True` on Turing; if it fails, prefer one process per GPU
      over debugging TP, since this workload is embarrassingly parallel.
    * FlashAttention-2 needs sm80+, so vLLM falls back to XFormers on T4. Do not
      force the flash backend.
    * The V1 engine does not support bitsandbytes, so 4-bit implies the HF path.
    """

    name = "vllm"
    supports_logprobs = True

    def __init__(
        self,
        cfg: ModelConfig,
        batch_size: int = 64,
        hw: Optional[HardwareInfo] = None,
    ) -> None:
        # vLLM does its own continuous batching, so hand it everything at once.
        super().__init__(batch_size=max(64, batch_size))
        from vllm import LLM  # type: ignore

        self.cfg = cfg
        self.hw = hw or detect_hardware()
        if cfg.load_in_4bit:
            raise RuntimeError(
                "4-bit via bitsandbytes is not supported by the vLLM V1 engine; use "
                "--backend hf for 4-bit, or an AWQ/GPTQ checkpoint with vLLM "
                "(note: Marlin kernels need sm80+, so a T4 must use the plain "
                "awq/gptq kernels)."
            )
        dtype = "float16" if not self.hw.supports_bf16 else _DTYPE_ALIASES.get(
            cfg.dtype.lower(), "float16"
        )
        if _DTYPE_ALIASES.get(cfg.dtype.lower()) == "bfloat16" and not self.hw.supports_bf16:
            log.warning("bf16 unsupported on this GPU; using float16 for vLLM")

        llm_kwargs: Dict[str, Any] = {
            "model": cfg.resolved_path,
            "dtype": dtype,
            "tensor_parallel_size": int(cfg.tensor_parallel_size),
            "gpu_memory_utilization": float(cfg.gpu_memory_utilization),
            "trust_remote_code": cfg.trust_remote_code,
            "enforce_eager": bool(cfg.enforce_eager),
            "seed": 0,
        }
        if cfg.max_model_len:
            llm_kwargs["max_model_len"] = int(cfg.max_model_len)
        if cfg.revision:
            llm_kwargs["revision"] = cfg.revision
        log.info("loading vLLM engine: %s", llm_kwargs)
        self.llm = LLM(**_supported_kwargs(LLM.__init__, llm_kwargs))
        try:
            self.tokenizer = self.llm.get_tokenizer()
        except Exception:  # pragma: no cover
            self.tokenizer = None

    def render(self, prompt: Prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        tok = self.tokenizer
        if self.cfg.use_chat_template and tok is not None and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                try:
                    return tok.apply_chat_template(
                        _merge_system_into_user(prompt),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    log.warning("vLLM chat template failed; using generic rendering")
        return messages_to_text(prompt)

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None or not text:
            return approx_token_count(text)
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "model": self.cfg.resolved_path,
            "dtype": self.cfg.dtype,
            "tensor_parallel_size": self.cfg.tensor_parallel_size,
            "gpu_memory_utilization": self.cfg.gpu_memory_utilization,
            "max_model_len": self.cfg.max_model_len,
            "enforce_eager": self.cfg.enforce_eager,
            "batch_size": self.batch_size,
            "hardware": self.hw.summary(),
        }

    def _generate(
        self, prompts: List[Prompt], params: GenParams
    ) -> List[List[Completion]]:
        from vllm import SamplingParams  # type: ignore

        sampling_kwargs: Dict[str, Any] = {
            "n": int(params.n),
            "max_tokens": int(params.max_new_tokens),
            "temperature": 0.0 if params.greedy else float(params.temperature),
            "top_p": 1.0 if params.greedy else float(params.top_p),
            "stop": list(params.stop) or None,
            "seed": params.seed,
            "repetition_penalty": float(params.repetition_penalty),
        }
        if not params.greedy and params.top_k and params.top_k > 0:
            sampling_kwargs["top_k"] = int(params.top_k)
        if params.n_logprobs > 0:
            sampling_kwargs["logprobs"] = int(params.n_logprobs)
        sampling = SamplingParams(**_supported_kwargs(SamplingParams.__init__, sampling_kwargs))

        texts = [self.render(p) for p in prompts]
        outputs = self.llm.generate(texts, sampling)

        groups: List[List[Completion]] = []
        for out in outputs:
            n_prompt = len(getattr(out, "prompt_token_ids", []) or [])
            group: List[Completion] = []
            for candidate in out.outputs:
                group.append(
                    Completion(
                        text=candidate.text,
                        tokens_prompt=n_prompt,
                        tokens_completion=len(getattr(candidate, "token_ids", []) or []),
                        finish_reason=str(getattr(candidate, "finish_reason", "stop") or "stop"),
                        logprobs=_vllm_logprobs(candidate),
                        cumulative_logprob=getattr(candidate, "cumulative_logprob", None),
                    )
                )
            # A truncated engine response would silently misalign samples, so
            # pad defensively rather than letting validate_groups raise.
            while len(group) < params.n:
                group.append(
                    Completion(text="", tokens_prompt=n_prompt, finish_reason="error")
                )
            groups.append(group[: params.n])
        return groups

    def close(self) -> None:
        if getattr(self, "llm", None) is not None:
            del self.llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover
            pass


def _vllm_logprobs(candidate: Any) -> Optional[List[float]]:
    """Per-token logprobs from a vLLM output, defensively.

    The structure of `CompletionOutput.logprobs` has changed across vLLM
    versions (dict of Logprob objects, dict of floats, or absent), so every
    shape is probed and anything unrecognised yields None.
    """
    raw = getattr(candidate, "logprobs", None)
    if not raw:
        return None
    token_ids = list(getattr(candidate, "token_ids", []) or [])
    out: List[float] = []
    try:
        for i, entry in enumerate(raw):
            if entry is None:
                continue
            if isinstance(entry, dict):
                chosen = token_ids[i] if i < len(token_ids) else None
                value = entry.get(chosen) if chosen is not None else None
                if value is None:
                    value = next(iter(entry.values()), None)
                out.append(float(getattr(value, "logprob", value)))
            else:
                out.append(float(getattr(entry, "logprob", entry)))
        return out or None
    except Exception:  # noqa: BLE001
        log.debug("could not parse vLLM logprobs", exc_info=True)
        return None


# ---------------------------------------------------------------------- factory
def build_backend(
    cfg: Union[RunConfig, ModelConfig],
    generation: Optional[GenerationConfig] = None,
    examples: Optional[Iterable[Example]] = None,
) -> GenerationBackend:
    """Construct the backend named by the config.

    | backend | result                                                          |
    |---------|-----------------------------------------------------------------|
    | mock    | MockBackend: CPU, no model, deterministic. Powers smoke tests.   |
    | hf      | HFBackend (transformers). The default and always-available path. |
    | vllm    | VLLMBackend; raises a clear error naming `hf` if vLLM is absent.  |
    | auto    | vLLM if importable and not 4-bit, else HFBackend with a reason.   |
    """
    if isinstance(cfg, RunConfig):
        model_cfg = cfg.model
        generation = generation or cfg.generation
        seed = cfg.runtime.seed
    else:
        model_cfg = cfg
        seed = 0

    batch_size = generation.batch_size if generation else 8
    choice = (model_cfg.backend or "hf").lower()

    if choice == "mock":
        backend: GenerationBackend = MockBackend(
            batch_size=batch_size,
            accuracy=model_cfg.mock_accuracy,
            seed=seed,
            latency_s=model_cfg.mock_latency_s,
        )
        if examples is not None and hasattr(backend, "register_golds"):
            backend.register_golds(examples)  # type: ignore[attr-defined]
        log.info("backend: mock (no model loaded, CPU only)")
        return backend

    hw = detect_hardware()

    if choice == "auto":
        if model_cfg.load_in_4bit:
            log.info("backend auto -> hf (4-bit is not supported by the vLLM V1 engine)")
            choice = "hf"
        elif vllm_available():
            log.info("backend auto -> vllm (importable and no 4-bit requested)")
            choice = "vllm"
        else:
            log.info("backend auto -> hf (vLLM is not installed)")
            choice = "hf"

    if choice == "vllm":
        if not vllm_available():
            raise RuntimeError(
                "backend='vllm' was requested but vLLM is not importable. Install it "
                "(`pip install vllm`) or use `--backend hf`, which is the default and "
                "needs nothing beyond transformers. The project never requires vLLM."
            )
        try:
            return VLLMBackend(model_cfg, batch_size=batch_size, hw=hw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"vLLM failed to start ({type(exc).__name__}: {exc}). Retry with "
                "`--backend hf`; on 2x T4 also try "
                "`--set model.enforce_eager=true` and "
                "`--set model.tensor_parallel_size=2`."
            ) from exc

    if choice != "hf":
        raise ValueError(
            f"unknown backend {model_cfg.backend!r}; expected mock, hf, vllm or auto"
        )
    return HFBackend(model_cfg, batch_size=batch_size, hw=hw)
