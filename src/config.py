"""Dataclass-based configuration: YAML load + CLI override + identity hashing.

Design notes
------------
* One YAML file per "experiment cell" (see `configs/`).
* `RunConfig.config_hash` is the *semantic* identity of a run. It deliberately
  excludes performance-only knobs (batch size, device map, tensor parallel size,
  backend choice, wall-clock budget) so that a run started on 2x T4 with vLLM
  can be resumed on a single P100 with plain transformers without invalidating
  completed work. Fields that genuinely change the output distribution (model
  name/revision, dtype, quantization, sampling params, dataset subsample,
  strategy params) *are* included.
* Any drift in excluded fields is still recorded in the run manifest and warned
  about, so it is visible in the paper's artifacts.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import GenParams
from .utils import deep_set, optional_import, stable_hash

log = logging.getLogger(__name__)

#: Dotted paths excluded from `config_hash` (performance / placement only).
#:
#: `runtime` is excluded wholesale, and that is load-bearing for the method: it
#: means `runtime.seed` does not enter the hash, so the same elicitation
#: configuration run at seeds 0/1/2 is *one* configuration measured three times.
#: That is exactly the same-configuration independent-seed replicate arm the
#: noise correction depends on. The seed still enters the per-example `uid`, so
#: the three replicates never collide.
HASH_EXCLUDE: Tuple[str, ...] = (
    "model.backend",
    "model.device_map",
    "model.tensor_parallel_size",
    "model.gpu_memory_utilization",
    "model.max_model_len",
    "model.attn_implementation",
    "model.local_path",
    "model.enforce_eager",
    "model.mock_accuracy",
    "model.mock_latency_s",
    "generation.batch_size",
    "runtime",
)


@dataclass
class ModelConfig:
    #: HF repo id, or a local directory (e.g. an attached Kaggle Dataset path).
    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    #: "auto" | "hf" | "vllm" | "mock". "auto" prefers vLLM, falls back to hf.
    backend: str = "hf"
    #: T4/P100 are pre-Ampere: bf16 has no tensor-core support, so fp16 default.
    dtype: str = "float16"
    load_in_4bit: bool = False
    bnb_quant_type: str = "nf4"
    bnb_double_quant: bool = True
    #: 4-bit compute dtype; keep fp16 on Turing/Pascal.
    bnb_compute_dtype: str = "float16"
    #: "auto" shards across all visible GPUs (the 2x T4 case); "cuda:0" pins.
    device_map: str = "auto"
    trust_remote_code: bool = False
    revision: Optional[str] = None
    #: Overrides `name_or_path` when set; used for offline / Kaggle Dataset runs.
    local_path: Optional[str] = None
    #: Apply the tokenizer chat template (instruct models) vs raw completion.
    use_chat_template: bool = True
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    attn_implementation: Optional[str] = None
    #: MockBackend only: fraction of samples that carry the gold answer.
    mock_accuracy: float = 0.6
    mock_latency_s: float = 0.0

    @property
    def resolved_path(self) -> str:
        return self.local_path or self.name_or_path

    @property
    def short_name(self) -> str:
        return Path(self.name_or_path.rstrip("/")).name


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: List[str] = field(default_factory=list)
    logprobs: bool = False
    #: Prompts per backend call. Perf-only; excluded from the config hash.
    batch_size: int = 8

    def to_gen_params(self, **overrides: Any) -> GenParams:
        p = GenParams(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=tuple(self.stop),
            logprobs=self.logprobs,
        )
        return p.with_(**overrides) if overrides else p


@dataclass
class DataConfig:
    #: Registered loader name, see `src/datasets_.py` DATASET_REGISTRY.
    name: str = "gsm8k"
    split: str = "test"
    #: HF dataset config / subset name (e.g. BBH subset, "main" for GSM8K).
    subset: Optional[str] = None
    #: Deterministically subsample to this many examples (None = full split).
    subsample: Optional[int] = 200
    subsample_seed: int = 1234
    #: "legacy" = `random.Random(subsample_seed).sample`, the harness default.
    #: "spec"   = the METHOD_SPEC section 2 protocol: one fixed permutation from
    #:            `numpy.random.default_rng(20260729)`, Tier A takes the first
    #:            `subsample`, Tier B the next `subsample`. Drawing both batches
    #:            from one permutation is what lets the two tiers be pooled with
    #:            no reweighting, so the method configs all use "spec".
    subsample_protocol: str = "legacy"
    #: "A" | "B" | "AB". Only meaningful with `subsample_protocol: spec`.
    tier: str = "A"
    #: Where the selected indices are persisted, for the run manifest.
    item_ids_dir: Optional[str] = "data"
    #: JSON map {example_id: paraphrased_question} applied before the run, used
    #: by the `c6` arm. Produced by `scripts/make_paraphrases.py`.
    paraphrase_file: Optional[str] = None
    #: Local directory for offline loading (Kaggle Dataset mount).
    local_dir: Optional[str] = None
    #: Optional inclusive index range applied *after* subsampling, for sharding.
    shard_index: int = 0
    num_shards: int = 1


@dataclass
class ElicitationConfig:
    """Selects the crossed elicitation-configuration factor for this run.

    `id: null` (the default) means "no elicitation factor": the strategy's own
    prompt wording and the run's own decoding parameters are used unchanged, which
    is the behaviour every non-method config in `configs/` relies on. Setting an
    id makes the configuration a first-class factor: it supplies the prompt style
    and the decoding temperature, is recorded in every result record, and enters
    the config hash.
    """

    id: Optional[str] = None
    #: One-off axis overrides (e.g. `{temperature: 1.0}`); the resolved id is
    #: suffixed so the derived cell is never confused with the registry entry.
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    #: Registered strategy name, see `src/strategies/__init__.py`.
    name: str = "cot_zeroshot"
    #: Strategy-specific keyword arguments, passed to its constructor.
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeConfig:
    out_dir: str = "results"
    #: Human-readable run label; the output directory is
    #: `<out_dir>/<run_name>__<config_hash>` so resume is automatic.
    run_name: Optional[str] = None
    seed: int = 0
    #: Wall-clock budget in hours. Kaggle kills sessions at 12h; default 8.0
    #: leaves room for install, model download, and a clean shutdown.
    time_budget_hours: float = 8.0
    #: Extra safety margin subtracted from the budget before stopping.
    reserve_minutes: float = 10.0
    #: fsync the results file every N completed examples.
    flush_every: int = 10
    #: Hard cap on examples processed this session (after resume skipping).
    max_examples: Optional[int] = None
    log_level: str = "INFO"
    #: Multiplier for GPU-hour accounting. Kaggle appears to bill session
    #: wall-clock hours regardless of GPU count, so 1.0; set 2.0 to be
    #: pessimistic about the 2x T4 configuration.
    gpu_hour_multiplier: float = 1.0
    #: Continue past a per-example exception, recording `error` in the record.
    continue_on_error: bool = True
    #: Write a graded copy of results at the end of the run.
    grade_after_run: bool = True
    #: Emit progress logs every N examples.
    log_every: int = 1
    #: What to keep of each sampled chain of thought. `/kaggle/working` is 20 GB,
    #: and the full matrix at N=24 generates ~1M chains, so this is a real
    #: constraint rather than tidiness. See `src/budget.py:estimate_disk`.
    #:   "full"     - keep every chain verbatim
    #:   "truncate" - keep the first/last `trace_max_chars` of each chain
    #:   "drop"     - keep no chain text at all
    #: Per-sample extracted answers and per-sample token counts are preserved
    #: under every policy, because those are what the analysis actually reads.
    trace_policy: str = "full"
    #: Characters kept from each end of a truncated trace.
    trace_max_chars: int = 400
    #: Append to `results.jsonl.gz` instead of `results.jsonl`. Gzip members
    #: concatenate, so appending and resuming still work unchanged.
    compress_results: bool = False


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    elicitation: ElicitationConfig = field(default_factory=ElicitationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    #: Free-form provenance recorded in the manifest (paper cell name, notes).
    tags: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- elicitation
    def resolved_elicitation(self) -> Optional[Any]:
        """The `Elicitation` this run uses, or None when the factor is unused."""
        if not self.elicitation.id:
            return None
        from .elicitation import get_elicitation

        return get_elicitation(str(self.elicitation.id)).with_overrides(
            self.elicitation.overrides
        )

    @property
    def elicitation_id(self) -> Optional[str]:
        elicitation = self.resolved_elicitation()
        return None if elicitation is None else elicitation.id

    def gen_params(self) -> GenParams:
        """The run's sampling parameters, with the elicitation axis applied.

        Temperature and top_p are a configuration axis (`c5`), so the resolved
        configuration overrides `generation.*`. Everything else — token limits,
        stop strings, logprobs — stays under `generation.*` because it is a cost
        or instrumentation knob, not part of the elicitation.
        """
        elicitation = self.resolved_elicitation()
        if elicitation is None:
            return self.generation.to_gen_params()
        return self.generation.to_gen_params(**elicitation.gen_overrides())

    def strategy_params(self) -> Dict[str, Any]:
        """Strategy kwargs with the elicitation's prompt style composed in.

        Composition, not replacement: the strategy still owns how many chains it
        draws and how it aggregates them. The configuration only owns the wording.
        An explicit `strategy.params.style` in YAML loses to the configuration,
        because the configuration is the factor being varied.
        """
        params = dict(self.strategy.params or {})
        elicitation = self.resolved_elicitation()
        if elicitation is None:
            return params
        if "style" in params:
            log.warning(
                "strategy.params.style is overridden by elicitation %s; remove it "
                "from the YAML to silence this",
                elicitation.id,
            )
        params["style"] = elicitation.prompt_style().to_dict()
        params["elicitation_id"] = elicitation.id
        params["model_id"] = self.model.name_or_path
        return params

    # ---------------------------------------------------------------- identity
    def semantic_dict(self) -> Dict[str, Any]:
        """The config subset that defines run identity."""
        d = asdict(self)
        d.pop("tags", None)
        for dotted in HASH_EXCLUDE:
            parts = dotted.split(".")
            cur: Any = d
            for p in parts[:-1]:
                cur = cur.get(p) if isinstance(cur, dict) else None
                if cur is None:
                    break
            if isinstance(cur, dict):
                cur.pop(parts[-1], None)
        # The *resolved* configuration, not just its id: editing c1's persona text
        # must produce a new run rather than silently mixing two wordings into one
        # results file. Absent when the factor is unused, so legacy configs keep
        # their existing hashes.
        elicitation = self.resolved_elicitation()
        if elicitation is not None:
            d["elicitation"] = elicitation.as_dict()
        else:
            d.pop("elicitation", None)
        return d

    @property
    def config_hash(self) -> str:
        return stable_hash(self.semantic_dict(), length=12)

    @property
    def run_name(self) -> str:
        if self.runtime.run_name:
            return sanitize_path_component(self.runtime.run_name)
        parts = [self.data.name, self.model.short_name, self.strategy.name]
        elicitation_id = self.elicitation_id
        if elicitation_id:
            # Configuration and seed go into the directory name so that the cells
            # of the crossed design -- including the same-configuration seed
            # replicates -- land in separate, individually resumable run dirs.
            # The dataset subset joins them because GSM-Symbolic `main` and `p2`
            # are different cells of the distribution-shift arm.
            if self.data.subset:
                parts.insert(1, str(self.data.subset))
            parts.append(elicitation_id)
            parts.append(f"s{self.runtime.seed}")
        return sanitize_path_component("__".join(parts))

    @property
    def run_dir(self) -> Path:
        return Path(self.runtime.out_dir) / f"{self.run_name}__{self.config_hash}"

    @property
    def results_path(self) -> Path:
        suffix = ".jsonl.gz" if self.runtime.compress_results else ".jsonl"
        return self.run_dir / f"results{suffix}"

    @property
    def graded_path(self) -> Path:
        suffix = ".jsonl.gz" if self.runtime.compress_results else ".jsonl"
        return self.run_dir / f"graded{suffix}"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    # ------------------------------------------------------------ (de)serialise
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunConfig":
        d = copy.deepcopy(d or {})
        unknown_sections = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown_sections:
            raise ValueError(
                f"unknown config section(s): {sorted(unknown_sections)}; "
                f"expected {sorted(f.name for f in dataclasses.fields(cls))}"
            )
        return cls(
            model=_build(ModelConfig, d.get("model")),
            generation=_build(GenerationConfig, d.get("generation")),
            data=_build(DataConfig, d.get("data")),
            strategy=_build(StrategyConfig, d.get("strategy")),
            elicitation=_build(ElicitationConfig, d.get("elicitation")),
            runtime=_build(RuntimeConfig, d.get("runtime")),
            tags=d.get("tags") or {},
        )

    def save(self, path: str | Path) -> None:
        yaml = optional_import("yaml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if yaml is None:  # pragma: no cover - pyyaml is a hard requirement
            import json

            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        else:
            path.write_text(
                yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
            )

    def describe(self) -> str:
        elicitation = self.resolved_elicitation()
        params = self.gen_params()
        lines = [
            f"run={self.run_name} hash={self.config_hash}",
            f"  model      : {self.model.resolved_path} "
            f"[{self.model.backend}, {self.model.dtype}"
            f"{', 4bit-' + self.model.bnb_quant_type if self.model.load_in_4bit else ''}]",
            f"  dataset    : {self.data.name} split={self.data.split} "
            f"subset={self.data.subset} subsample={self.data.subsample} "
            f"(protocol {self.data.subsample_protocol}, tier {self.data.tier}, "
            f"seed {self.data.subsample_seed})",
            f"  strategy   : {self.strategy.name} {self.strategy.params}",
        ]
        if elicitation is not None:
            lines.append(
                f"  elicitation: {elicitation.id} axis={elicitation.axis} "
                f"system={'<default>' if elicitation.system is None else (elicitation.system or '<none>')!r} "
                f"shots={elicitation.n_shots}/{elicitation.shot_order} "
                f"{'[SEPARATE ARM]' if elicitation.separate_arm else ''}"
            )
        lines += [
            f"  generation : max_new_tokens={params.max_new_tokens} "
            f"T={params.temperature} top_p={params.top_p} "
            f"batch={self.generation.batch_size}",
            f"  runtime    : seed={self.runtime.seed} "
            f"budget={self.runtime.time_budget_hours}h "
            f"flush_every={self.runtime.flush_every} "
            f"traces={self.runtime.trace_policy} out={self.run_dir}",
        ]
        return "\n".join(lines)


def sanitize_path_component(name: str) -> str:
    """Make a string safe as a single directory name on every platform.

    Needed because a strategy may be referenced as `my_module:MyClass`, and ':'
    is not a legal filename character on Windows.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-.")
    return cleaned or "run"


def _build(cls: Any, d: Optional[Dict[str, Any]]) -> Any:
    d = dict(d or {})
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}; valid keys are "
            f"{sorted(valid)}"
        )
    return cls(**d)


# --------------------------------------------------------------------- loading
def load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml = optional_import("yaml")
    if yaml is None:  # pragma: no cover
        raise RuntimeError("pyyaml is required to read configs (pip install pyyaml)")
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge; `override` wins. Lists are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Sequence[str]] = None,
    base: Optional[Dict[str, Any]] = None,
) -> RunConfig:
    """Load YAML (optionally chaining a `defaults:` file) then apply overrides.

    `overrides` are dotted `key=value` strings, e.g.
    `["generation.temperature=0.8", "strategy.params.k=8"]`.
    """
    data: Dict[str, Any] = dict(base or {})
    if path is not None:
        raw = load_yaml(path)
        parent = raw.pop("defaults", None)
        if parent:
            parent_path = (Path(path).parent / parent).resolve()
            data = merge_dicts(data, load_yaml(parent_path))
        data = merge_dicts(data, raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, _, value = item.partition("=")
        deep_set(data, key.strip(), parse_scalar(value.strip()))
    return RunConfig.from_dict(data)


def parse_scalar(text: str) -> Any:
    """Parse a CLI value as YAML if possible, else keep the raw string."""
    yaml = optional_import("yaml")
    if yaml is None:  # pragma: no cover
        return text
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


# ------------------------------------------------------------------------- CLI
def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the standard config/override flags to an argparse parser."""
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. --set generation.temperature=0.7)",
    )
    # Shorthands for the knobs used constantly on Kaggle.
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["auto", "hf", "vllm", "mock"])
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument(
        "--elicitation",
        type=str,
        default=None,
        help="elicitation configuration id (c0..c6); see src/elicitation.py",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subsample", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=None)
    parser.add_argument("--log-level", type=str, default=None)
    return parser


_SHORTHAND_MAP = {
    "model": "model.name_or_path",
    "backend": "model.backend",
    "dataset": "data.name",
    "strategy": "strategy.name",
    "elicitation": "elicitation.id",
    "seed": "runtime.seed",
    "subsample": "data.subsample",
    "max_examples": "runtime.max_examples",
    "max_new_tokens": "generation.max_new_tokens",
    "time_budget_hours": "runtime.time_budget_hours",
    "out_dir": "runtime.out_dir",
    "run_name": "runtime.run_name",
    "flush_every": "runtime.flush_every",
    "batch_size": "generation.batch_size",
    "load_in_4bit": "model.load_in_4bit",
    "log_level": "runtime.log_level",
}


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Build a RunConfig from parsed args (shorthands become dotted overrides)."""
    overrides: List[str] = list(getattr(args, "overrides", []) or [])
    for attr, dotted in _SHORTHAND_MAP.items():
        val = getattr(args, attr, None)
        if val is not None:
            overrides.append(f"{dotted}={val}")
    return load_config(getattr(args, "config", None), overrides)
