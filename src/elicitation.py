"""Elicitation configurations: the crossed experimental factor `c`.

An **elicitation configuration** is a semantics-preserving perturbation of
everything about *how* a question is asked, holding the question's mathematical
content fixed. METHOD_SPEC section 3 names six of them plus a separately-reported
paraphrase arm; they are defined here as data, not as code branches, so that a
run is fully described by `(model, benchmark, configuration, strategy, seed)`.

Design contract
---------------
* A configuration **composes with** a strategy rather than replacing it. The
  configuration supplies a `PromptStyle` (system prompt, instruction wording,
  answer-format wording, few-shot count and order) plus decoding overrides
  (temperature, top_p); the strategy still decides how many chains to draw and
  how to aggregate them. So `self_consistency` under `c1` and `cot_zeroshot`
  under `c1` are both meaningful cells.
* The configuration's *resolved* content — not just its id — enters
  `RunConfig.config_hash`, so editing `c1`'s persona text creates a new run
  instead of silently contaminating an existing one.
* The seed is deliberately **not** part of the configuration or the hash. That is
  what makes the same-configuration independent-seed replicate arm expressible:
  `c0` at seeds 0/1/2 is one configuration measured three times, which is the
  noise null the whole variance correction depends on.
* `c6` (paraphrase) is flagged `separate_arm=True` because it rewrites the
  problem statement and therefore cannot be claimed to be information-preserving.
  Every aggregation in `src/analysis` filters on that flag rather than on the id.

Axis coverage (four semantics-preserving axes plus the paraphrase arm):

| id      | axis                        | what moves                          |
|---------|-----------------------------|-------------------------------------|
| `c0`    | reference                   | nothing (the baseline cell)         |
| `c1`    | system-prompt persona       | system prompt                       |
| `c2`    | system-prompt minimality    | system prompt removed               |
| `c3`    | answer-format instruction   | final-answer wording                |
| `c4`    | few-shot exemplar order     | 4 exemplars, reversed               |
| `c4a`   | few-shot exemplar order     | the same 4 exemplars, forward       |
| `c5`    | decoding temperature        | T 0.8 -> 1.0, top_p 0.95 -> 1.0     |
| `c5lo`  | decoding temperature        | T 0.8 -> 0.6 (needed by baseline B4)|
| `c6`    | question paraphrase         | the problem statement itself        |
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: METHOD_SPEC section 3 `c0` wording, quoted verbatim so the spec and the code
#: cannot drift apart.
SPEC_SYSTEM_NEUTRAL = "You are a helpful assistant."
SPEC_SYSTEM_EXPERT = "You are an expert mathematician. Be rigorous."
SPEC_INSTRUCTION = "Solve the problem step by step."
SPEC_HINT_ANSWER_PREFIX = "Put your final answer after 'Answer: '."
SPEC_HINT_BOXED = r"End your response with \boxed{}."


@dataclass(frozen=True)
class Elicitation:
    """One configuration `c`. Immutable: cells are compared, never mutated."""

    id: str
    axis: str
    #: `None` means "use the harness default system prompt"; `""` means
    #: "emit no system turn at all", which is a distinct experimental condition.
    system: Optional[str] = SPEC_SYSTEM_NEUTRAL
    instruction: Optional[str] = SPEC_INSTRUCTION
    #: Overrides the per-answer-type answer-format wording when set.
    hint: Optional[str] = SPEC_HINT_ANSWER_PREFIX
    n_shots: int = 0
    shot_order: str = "forward"
    shot_pool: Optional[str] = None
    temperature: float = 0.8
    top_p: float = 0.95
    #: Rewrites the question text (only `c6`).
    paraphrase: bool = False
    #: Reported separately from the `c0`-`c5` family throughout the analysis.
    separate_arm: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.shot_order not in ("forward", "reverse"):
            raise ValueError(
                f"{self.id}: shot_order must be 'forward' or 'reverse', "
                f"got {self.shot_order!r}"
            )
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError(f"{self.id}: temperature {self.temperature} out of range")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError(f"{self.id}: top_p {self.top_p} out of range")
        if self.n_shots < 0:
            raise ValueError(f"{self.id}: n_shots must be >= 0")
        if self.paraphrase and not self.separate_arm:
            raise ValueError(
                f"{self.id}: a paraphrasing configuration must set separate_arm=True; "
                "it rewrites the problem statement and cannot be pooled with the "
                "semantics-preserving axes"
            )

    # ------------------------------------------------------------------ exports
    def prompt_style(self) -> Any:
        """The `PromptStyle` this configuration implies (imported lazily)."""
        from .prompts import PromptStyle

        return PromptStyle(
            name=self.id,
            system=self.system,
            instruction=self.instruction,
            hint=self.hint,
            include_hint=self.hint is not None,
            n_shots=self.n_shots,
            shot_order=self.shot_order,
            shot_pool=self.shot_pool,
        )

    def gen_overrides(self) -> Dict[str, Any]:
        """Decoding overrides applied on top of the run's `GenerationConfig`."""
        return {"temperature": float(self.temperature), "top_p": float(self.top_p)}

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: Optional[Dict[str, Any]]) -> "Elicitation":
        """A copy with explicit YAML overrides applied.

        Escape hatch for a one-off ablation ("c1 but at T=1.0") that does not
        deserve its own registry entry. The result carries a derived id so the two
        cells can never be confused in a table.
        """
        overrides = {k: v for k, v in dict(overrides or {}).items() if v is not None}
        if not overrides:
            return self
        valid = {f.name for f in fields(self)} - {"id"}
        unknown = set(overrides) - valid
        if unknown:
            raise ValueError(
                f"unknown elicitation override(s): {sorted(unknown)}; "
                f"valid keys are {sorted(valid)}"
            )
        suffix = "+".join(f"{k}={overrides[k]!r}" for k in sorted(overrides))
        return replace(self, **overrides, id=f"{self.id}[{suffix}]")


#: The configurations named in METHOD_SPEC section 3, in table order.
ELICITATIONS: Tuple[Elicitation, ...] = (
    Elicitation(
        id="c0",
        axis="reference",
        notes="Reference cell. Default chat template, neutral system prompt.",
    ),
    Elicitation(
        id="c1",
        axis="system_persona",
        system=SPEC_SYSTEM_EXPERT,
        notes="System-prompt persona; everything else as c0.",
    ),
    Elicitation(
        id="c2",
        axis="system_minimality",
        system="",
        notes=(
            "Empty system prompt. For Qwen2.5 and Llama-3.2 the harness forces a "
            "genuinely empty system turn (see prompts.resolve_system_for_model); "
            "rendered prompts are logged per (model, config)."
        ),
    ),
    Elicitation(
        id="c3",
        axis="answer_format",
        hint=SPEC_HINT_BOXED,
        separate_arm=True,
        notes=(
            "Answer-format instruction only; \\boxed{} instead of 'Answer: '. "
            "Demoted from the core configuration factor (not parse-invariant); "
            "reported on a separate axis alongside c4 and c6."
        ),
    ),
    Elicitation(
        id="c4",
        axis="fewshot_order",
        n_shots=4,
        shot_order="reverse",
        separate_arm=True,
        notes=(
            "4-shot CoT with the exemplars in reverse order. Compare against c4a, "
            "which holds the exemplar set fixed and only changes the order. "
            "Demoted from core: compares zero-shot reference to few-shot (task info)."
        ),
    ),
    Elicitation(
        id="c4a",
        axis="fewshot_order",
        n_shots=4,
        shot_order="forward",
        notes=(
            "Same four exemplars as c4 in forward order. Used only in the ordering "
            "sub-analysis, per METHOD_SPEC section 3."
        ),
    ),
    Elicitation(
        id="c5",
        axis="temperature",
        temperature=1.0,
        top_p=1.0,
        separate_arm=True,
        notes=(
            "Decoding temperature 1.0, top_p 1.0. Removed from the core configuration "
            "factor; used only in baseline B4 (temperature-diversified SC)."
        ),
    ),
    Elicitation(
        id="c5lo",
        axis="temperature",
        temperature=0.6,
        top_p=0.95,
        notes=(
            "Temperature 0.6. NOT in the METHOD_SPEC section 3 table, but baseline "
            "B4 (temperature-diversified self-consistency, T in {0.6, 0.8, 1.0}) "
            "cannot be computed without it: section 3 supplies only 0.8 (c0) and "
            "1.0 (c5). Excluded from the primary c0-c5 configuration family so it "
            "cannot inflate the configuration factor; used only for B4."
        ),
    ),
    Elicitation(
        id="c6",
        axis="paraphrase",
        paraphrase=True,
        separate_arm=True,
        notes=(
            "Question paraphrased once offline with a numeric-preservation gate "
            "(see src/paraphrase.py). Reported separately from c0-c5 because it is "
            "the only axis that rewrites the problem statement."
        ),
    ),
)

ELICITATION_REGISTRY: Dict[str, Elicitation] = {e.id: e for e in ELICITATIONS}

#: Core semantics-preserving configurations for the crossed design (adversarial
#: review §6.3). Headline variance, transfer and hard-subset statistics use this
#: set. `c3`, `c4`, `c5` and `c6` are reported on separate axes.
PRIMARY_CONFIG_IDS: Tuple[str, ...] = ("c0", "c1", "c2")

#: Configurations demoted from the core factor but still run and reported.
SEPARATE_AXIS_CONFIG_IDS: Tuple[str, ...] = ("c3", "c4", "c5", "c6")

#: The reference configuration; the seed-replicate noise null lives here.
REFERENCE_CONFIG_ID = "c0"

#: Non-reference configuration with additional seed replicates (review §6.3).
REPLICATE_CONFIG_C1 = "c1"

#: Configurations CDV spreads over by default (C_use = 4; Tier B O2 arm c0–c3).
DEFAULT_CDV_CONFIG_IDS: Tuple[str, ...] = ("c0", "c1", "c2", "c3")

#: Baseline B4 isolates "diversity per se": same prompt, three temperatures.
TEMPERATURE_ARM_IDS: Tuple[str, ...] = ("c5lo", "c0", "c5")

#: Tier B deep-N arm: multi-configuration sampling at N=64 (review §6.3 O2).
DEEP_N_CONFIG_IDS: Tuple[str, ...] = ("c0", "c1", "c2", "c3")
DEEP_N_SAMPLES = 64

#: The ordering sub-analysis pair (same exemplars, order flipped).
ORDER_PAIR_IDS: Tuple[str, str] = ("c4a", "c4")

#: The paraphrase arm. Reported separately throughout because it is the one axis
#: that rewrites the problem statement, so an effect there is not evidence about
#: semantics-preserving elicitation. Baseline B5 (Self-Para-Consistency) votes over
#: it, which is the only place it is pooled with `c0` -- and that is the published
#: method being reproduced, not a claim of ours.
PARAPHRASE_CONFIG_ID = "c6"


def get_elicitation(config_id: str) -> Elicitation:
    if config_id not in ELICITATION_REGISTRY:
        raise KeyError(
            f"unknown elicitation configuration {config_id!r}. "
            f"Available: {sorted(ELICITATION_REGISTRY)}"
        )
    return ELICITATION_REGISTRY[config_id]


def available_elicitations() -> List[str]:
    return [e.id for e in ELICITATIONS]


def primary_configs() -> List[Elicitation]:
    return [ELICITATION_REGISTRY[i] for i in PRIMARY_CONFIG_IDS]


def is_separate_arm(config_id: Optional[str]) -> bool:
    """True for configurations that must not be pooled with `c0`-`c5`."""
    if not config_id:
        return False
    base = str(config_id).split("[")[0]
    spec = ELICITATION_REGISTRY.get(base)
    return bool(spec.separate_arm) if spec else False


def axis_of(config_id: Optional[str]) -> str:
    base = str(config_id or "").split("[")[0]
    spec = ELICITATION_REGISTRY.get(base)
    return spec.axis if spec else "unknown"


# ----------------------------------------------------------- design bookkeeping
@dataclass
class ReplicateArm:
    """The same-configuration independent-seed arm, made explicit.

    METHOD_SPEC section 3 sets 3 seeds in Tier A on all three Qwen models, and 5
    seeds in Tier B for the 1.5B/3B while the 7B stays at 3. The gap analysis is
    blunt that this arm is never cuttable, because the seed replicates are the
    only estimate of the sampling-noise floor and of the reliability `r_mm` that
    divides the headline `rho_disatt`. `n_pairs` is the quantity that actually
    matters: 3 seeds give 3 independent pairs, 5 give 10.
    """

    config_id: str = REFERENCE_CONFIG_ID
    seeds: Tuple[int, ...] = (0, 1, 2)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    @property
    def n_pairs(self) -> int:
        n = self.n_seeds
        return n * (n - 1) // 2

    def validate(self) -> None:
        if self.n_seeds < 2:
            raise ValueError(
                "the seed-replicate arm needs at least 2 seeds: with one seed there "
                "is no noise-floor estimate and no reliability denominator, so the "
                "variance decomposition and rho_disatt are both unsupportable "
                "(METHOD_SPEC section 8.5 forbids cutting this arm)"
            )
        if len(set(self.seeds)) != self.n_seeds:
            raise ValueError(f"replicate seeds must be distinct, got {self.seeds}")


TIER_A_REPLICATES = ReplicateArm(seeds=(0, 1, 2))
TIER_B_REPLICATES = ReplicateArm(seeds=(0, 1, 2, 3, 4))

#: Seed replicates at c1 for transportability of r_mm (review §6.3).
TIER_B_C1_REPLICATES = ReplicateArm(config_id=REPLICATE_CONFIG_C1, seeds=(0, 1, 2))


@dataclass
class DesignCell:
    """One point of the crossed design, for matrix planning and cost estimates."""

    model: str
    dataset: str
    subset: Optional[str]
    config_id: str
    seed: int
    n_items: int
    n_samples: int
    tier: str = "A"
    label: str = ""

    @property
    def total_samples(self) -> int:
        return self.n_items * self.n_samples

    def key(self) -> Tuple[Any, ...]:
        return (self.model, self.dataset, self.subset, self.config_id, self.seed)


def expand_design(
    models: Sequence[str],
    datasets: Sequence[Tuple[str, Optional[str], int]],
    config_ids: Sequence[str] = PRIMARY_CONFIG_IDS,
    seeds: Sequence[int] = (0,),
    n_samples: int = 24,
    tier: str = "A",
    label: str = "",
) -> List[DesignCell]:
    """The fully crossed cell list for one experiment block.

    Fully crossed is not a stylistic preference here: the variance decomposition
    in `src/analysis/variance.py` is a balanced method-of-moments fit, so a
    missing (item, model, config) cell silently biases the components. The matrix
    driver checks completeness against this expansion before analysis.
    """
    for config_id in config_ids:
        get_elicitation(config_id)  # fail fast on a typo in a YAML matrix
    cells: List[DesignCell] = []
    for model in models:
        for dataset, subset, n_items in datasets:
            for config_id in config_ids:
                for seed in seeds:
                    cells.append(
                        DesignCell(
                            model=model,
                            dataset=dataset,
                            subset=subset,
                            config_id=config_id,
                            seed=int(seed),
                            n_items=int(n_items),
                            n_samples=int(n_samples),
                            tier=tier,
                            label=label,
                        )
                    )
    return cells


def describe_registry() -> str:
    """A printable table of the configurations; used by the notebook's setup cell."""
    lines = [
        f"{'id':<6} {'axis':<20} {'T':>4} {'top_p':>6} {'shots':>6} {'order':<8} "
        f"{'arm':<9} system",
        "-" * 96,
    ]
    for e in ELICITATIONS:
        system = "<default>" if e.system is None else (e.system or "<none>")
        arm = "separate" if e.separate_arm else (
            "primary" if e.id in PRIMARY_CONFIG_IDS else "auxiliary"
        )
        lines.append(
            f"{e.id:<6} {e.axis:<20} {e.temperature:>4.1f} {e.top_p:>6.2f} "
            f"{e.n_shots:>6} {e.shot_order:<8} {arm:<9} {system[:34]}"
        )
    return "\n".join(lines)
