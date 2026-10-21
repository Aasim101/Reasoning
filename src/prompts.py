"""Prompt construction shared by all strategies.

Keeping prompt formatting in one place is what makes answer extraction reliable:
every strategy asks the model for the same answer format, so `src/answers.py`
only has to understand a small number of shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import ChatMessage, Example, Prompt

DEFAULT_SYSTEM = (
    "You are a careful, concise reasoning assistant. You always finish your "
    "reply with the final answer in the requested format."
)

COT_INSTRUCTION = "Think step by step, then state the final answer."
DIRECT_INSTRUCTION = (
    "Answer immediately with the final answer only. Do not explain your reasoning."
)
REFINE_INSTRUCTION = (
    "Review the draft answer above. Point out any error you find, then give the "
    "corrected final answer."
)
VERIFY_INSTRUCTION = (
    "Check the candidate solution above step by step. Reply with VERDICT: CORRECT "
    "or VERDICT: INCORRECT, followed by one sentence of justification."
)

#: How the final answer must be formatted, per answer type.
_FORMAT_HINTS: Dict[str, str] = {
    "math": r"End your reply with the final answer inside \boxed{}, e.g. \boxed{42}.",
    "mc": (
        "End your reply with the single letter of the correct option in the form "
        r"\boxed{A}."
    ),
    "bool": r"End your reply with \boxed{True} or \boxed{False}.",
    "text": r"End your reply with the final answer inside \boxed{}.",
}


def answer_format_hint(example: Example, override: Optional[str] = None) -> str:
    """The answer-format instruction, or an explicit override.

    The override exists because the answer-format wording is itself an
    experimental axis: two prompts that differ only in how they ask for the
    final answer can change accuracy, and a study of prompt sensitivity needs to
    vary it from config without editing code.
    """
    if override is not None:
        return override
    return _FORMAT_HINTS.get(example.answer_type, _FORMAT_HINTS["text"])


def format_choices(example: Example) -> str:
    if not example.choices:
        return ""
    lines = [
        f"{letter}. {text}"
        for letter, text in zip(example.choice_letters, example.choices)
    ]
    return "Options:\n" + "\n".join(lines)


def format_question(
    example: Example, include_hint: bool = True, hint: Optional[str] = None
) -> str:
    """The question block: stem, options (if any), and the answer format hint."""
    parts = [example.question.strip()]
    choices = format_choices(example)
    if choices:
        parts.append(choices)
    if include_hint:
        parts.append(answer_format_hint(example, hint))
    return "\n\n".join(p for p in parts if p)


def model_family_from_id(model_id: str) -> str:
    """Coarse pretraining lineage from a HuggingFace model id."""
    text = str(model_id or "").lower()
    if "qwen" in text:
        return "qwen"
    if "llama" in text or "meta-llama" in text:
        return "llama"
    if "phi" in text:
        return "phi"
    if "mistral" in text:
        return "mistral"
    return text.split("/")[0] if "/" in text else "unknown"


def resolve_system_for_model(
    system: Optional[str], model_id: str, config_id: Optional[str] = None
) -> Optional[str]:
    """Apply model-family overrides for elicitation configurations.

    Qwen2.5's chat template injects a default system persona when none is given;
  Llama-3 handles an absent system differently. For `c2` we force a genuinely
    empty system turn on both families so the configuration factor is the same
    manipulation across the cross-family replication.
    """
    base = str(config_id or "").split("[")[0]
    if base == "c2" and system == "":
        family = model_family_from_id(model_id)
        if family in ("qwen", "llama"):
            return ""  # emit no system turn; logged at render time
    return system


def render_prompt_text(
    prompt: Prompt,
    tokenizer: Any = None,
    model_id: Optional[str] = None,
) -> Tuple[str, Optional[List[int]]]:
    """Render a chat prompt to the string the model sees, plus token ids if possible.

    Returns `(rendered_text, token_ids_or_none)`. When no tokenizer is available
    the text is a best-effort concatenation of message roles.
    """
    if isinstance(prompt, str):
        text = prompt
        ids = None
        if tokenizer is not None:
            try:
                ids = tokenizer.encode(text, add_special_tokens=True)
            except Exception:
                ids = None
        return text, ids
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            ids = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
            return str(rendered), list(ids) if ids is not None else None
        except Exception:
            pass
    parts = []
    for msg in prompt:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<{role}>\n{content}")
    return "\n\n".join(parts), None


def prompt_digest(text: str, max_chars: int = 4096) -> Dict[str, Any]:
    """Compact prompt provenance for per-sample logging."""
    from .utils import stable_hash

    body = str(text or "")
    return {
        "rendered_prompt_hash": stable_hash(body, length=16),
        "prompt_length_chars": len(body),
        "rendered_prompt": body if len(body) <= max_chars else body[:max_chars] + "…",
    }


def build_prompt(
    example: Example,
    instruction: Optional[str] = None,
    system: Optional[str] = None,
    few_shot: Sequence[Tuple[str, str]] = (),
    include_hint: bool = True,
    hint: Optional[str] = None,
) -> Prompt:
    """Assemble a chat prompt.

    Few-shot exemplars are rendered as alternating user/assistant turns, which
    works with instruct-tuned chat templates and keeps the final turn's role
    correct for generation.

    `system=""` produces no system turn at all (some templates reject an empty
    one, and "no system prompt" is a meaningful experimental condition distinct
    from "the default system prompt").
    """
    messages: List[ChatMessage] = []
    system_text = DEFAULT_SYSTEM if system is None else system
    if system_text:
        messages.append({"role": "system", "content": system_text})
    for shot_q, shot_a in few_shot:
        messages.append({"role": "user", "content": shot_q.strip()})
        messages.append({"role": "assistant", "content": shot_a.strip()})
    user = format_question(example, include_hint=include_hint, hint=hint)
    if instruction:
        user = f"{user}\n\n{instruction.strip()}"
    messages.append({"role": "user", "content": user})
    return messages


def continue_chat(prompt: Prompt, assistant_text: str, follow_up: str) -> Prompt:
    """Extend a chat prompt with a model turn plus a new user turn.

    Used by multi-turn strategies (self-refine, self-verification) so the model
    sees its own draft in the correct role.
    """
    if isinstance(prompt, str):
        return f"{prompt}{assistant_text}\n\n{follow_up}"
    return [
        *prompt,
        {"role": "assistant", "content": assistant_text},
        {"role": "user", "content": follow_up},
    ]


# ---------------------------------------------------------------- few-shot pool
#: Compact 4-shot chain-of-thought exemplars for grade-school / competition math.
#: Hand-written (not copied from a benchmark split) so they can never leak test
#: items. Answers use the same \boxed{} format the graders expect.
MATH_FEWSHOT: List[Tuple[str, str]] = [
    (
        "A shop sells pencils in packs of 8. Mia buys 3 packs and gives 5 pencils "
        "to her brother. How many pencils does she have left?",
        "Mia starts with 3 packs of 8 pencils, so 3 * 8 = 24 pencils.\n"
        "She gives away 5, leaving 24 - 5 = 19 pencils.\n"
        r"The final answer is \boxed{19}.",
    ),
    (
        "A train travels 60 km in 45 minutes. At the same speed, how many "
        "kilometres does it travel in 2 hours?",
        "45 minutes is 0.75 hours, so the speed is 60 / 0.75 = 80 km per hour.\n"
        "In 2 hours it covers 80 * 2 = 160 km.\n"
        r"The final answer is \boxed{160}.",
    ),
    (
        "What is the value of x if 3(x - 4) = 2x + 1?",
        "Expanding the left side gives 3x - 12 = 2x + 1.\n"
        "Subtracting 2x from both sides gives x - 12 = 1.\n"
        "Adding 12 to both sides gives x = 13.\n"
        r"The final answer is \boxed{13}.",
    ),
    (
        "A rectangle has area 48 and width 6. What is its perimeter?",
        "The length is 48 / 6 = 8.\n"
        "The perimeter is 2 * (8 + 6) = 28.\n"
        r"The final answer is \boxed{28}.",
    ),
]

#: Multiple-choice exemplars (science/commonsense style, letter answers).
MC_FEWSHOT: List[Tuple[str, str]] = [
    (
        "Which object is the best conductor of electricity?\n\nOptions:\n"
        "A. a rubber band\nB. a copper wire\nC. a glass rod\nD. a wooden spoon\n\n"
        r"End your reply with the single letter of the correct option in the form \boxed{A}.",
        "Conductors let electric charge move freely. Rubber, glass and wood are "
        "insulators, while copper is a metal with mobile electrons.\n"
        r"The final answer is \boxed{B}.",
    ),
    (
        "A plant is placed in a dark room for a week. Which process is most "
        "directly reduced?\n\nOptions:\nA. respiration\nB. transpiration\n"
        "C. photosynthesis\nD. germination\n\n"
        r"End your reply with the single letter of the correct option in the form \boxed{A}.",
        "Photosynthesis requires light energy to convert carbon dioxide and water "
        "into sugars, so removing light reduces it most directly.\n"
        r"The final answer is \boxed{C}.",
    ),
]

#: Logical / multi-hop deduction exemplars (BBH, MuSR style).
LOGIC_FEWSHOT: List[Tuple[str, str]] = [
    (
        "All maple trees in the park are older than every oak in the park. Tree T "
        "is an oak in the park and tree S is a maple in the park. Is S older "
        "than T?\n\n"
        r"End your reply with the final answer inside \boxed{}, e.g. \boxed{42}.",
        "Every maple is older than every oak in the park. S is a maple and T is "
        "an oak, so S must be older than T.\n"
        r"The final answer is \boxed{Yes}.",
    ),
    (
        "Anna left the office before Ben. Ben left before Chen. Who left first?\n\n"
        r"End your reply with the final answer inside \boxed{}, e.g. \boxed{42}.",
        "Anna is before Ben, and Ben is before Chen, so the order is Anna, Ben, "
        "Chen.\n"
        r"The final answer is \boxed{Anna}.",
    ),
]

#: Dataset family -> exemplar pool. Selection is by `Example.answer_type` unless
#: a dataset name is given explicitly, so new datasets get sensible defaults.
FEWSHOT_POOLS: Dict[str, List[Tuple[str, str]]] = {
    "math": MATH_FEWSHOT,
    "mc": MC_FEWSHOT,
    "logic": LOGIC_FEWSHOT,
    "bool": LOGIC_FEWSHOT,
    "text": LOGIC_FEWSHOT,
}


def get_few_shot(
    example: Example,
    n: int = 4,
    pool: Optional[str] = None,
    order: str = "forward",
) -> List[Tuple[str, str]]:
    """Pick `n` exemplars appropriate to an example's answer type.

    Exemplars are a fixed prefix, never randomly shuffled, so few-shot prompts
    are identical across runs and seeds and `config_hash` stays meaningful.
    `order="reverse"` is supported because exemplar *ordering* is a known
    sensitivity axis worth varying deliberately (and reproducibly) from config.
    """
    key = pool or example.meta.get("fewshot_pool") or example.answer_type
    shots = FEWSHOT_POOLS.get(str(key), MATH_FEWSHOT)
    if n <= 0:
        return []
    chosen = list(shots[:n])
    if order == "reverse":
        chosen.reverse()
    elif order != "forward":
        raise ValueError(f"order must be 'forward' or 'reverse', got {order!r}")
    return chosen


@dataclass
class PromptStyle:
    """A named, fully-declarative prompt configuration.

    Everything that can be varied about *how* a question is asked, without
    changing the question, lives here. This makes an elicitation-configuration
    axis expressible from YAML with no new code:

        strategy:
          name: self_consistency
          params:
            k: 24
            style: {name: c1, system: "You are an expert mathematician.",
                    instruction: "Solve the problem step by step.",
                    hint: "End your response with \\boxed{}."}

    A strategy accepts a `style` dict, calls `PromptStyle.from_dict`, and uses
    `style.build(example)`. Because the dict lands in `strategy.params`, it is
    part of the run's config hash: two styles are two different runs, which is
    exactly the required behaviour.
    """

    name: str = "default"
    system: Optional[str] = None
    instruction: Optional[str] = COT_INSTRUCTION
    #: Overrides the per-answer-type answer-format hint when set.
    hint: Optional[str] = None
    include_hint: bool = True
    n_shots: int = 0
    shot_order: str = "forward"
    shot_pool: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "PromptStyle":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown PromptStyle key(s): {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def build(self, example: Example, model_id: Optional[str] = None) -> Prompt:
        few_shot = get_few_shot(
            example, n=self.n_shots, pool=self.shot_pool, order=self.shot_order
        )
        system = resolve_system_for_model(self.system, model_id or "", self.name)
        return build_prompt(
            example,
            instruction=self.instruction,
            system=system,
            few_shot=few_shot,
            include_hint=self.include_hint,
            hint=self.hint,
        )
