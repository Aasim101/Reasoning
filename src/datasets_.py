"""Benchmark loaders, unified into one schema.

Every loader returns `list[Example]` with `{id, question, gold_answer,
gold_reasoning?, choices?, answer_type, meta}`, so strategies, graders and
metrics never contain per-dataset special cases.

Dataset id verification (checked 2026-07-29; see `DatasetSpec.verified`):

| name          | HF id                    | config          | split          | verified |
|---------------|--------------------------|-----------------|----------------|----------|
| toy           | built in                 | -               | test           | yes      |
| gsm8k         | openai/gsm8k             | main            | test           | yes      |
| math500       | HuggingFaceH4/MATH-500   | -               | test           | yes      |
| aime          | Maxwell-Jia/AIME_2024    | -               | train          | yes      |
|               | math-ai/aime25 (2025)    | -               | test/train     | partial  |
| bbh           | lukaemon/bbh             | <task>          | test           | yes      |
| musr          | TAUR-Lab/MuSR            | default         | <domain>       | yes      |
| arc_challenge | allenai/ai2_arc          | ARC-Challenge   | test           | yes      |
| gpqa_diamond  | Idavidrein/gpqa          | gpqa_diamond    | train          | yes*     |
| gsm_symbolic  | apple/GSM-Symbolic       | main/p1/p2      | test           | NO       |

*GPQA resolves but is **gated**: accept the terms on the dataset page and set
`HF_TOKEN`. `gsm_symbolic` could not be verified from this machine (the HF API
is unreachable here) and is marked unverified in code.

Column names are looked up from a list of candidates rather than hard-coded, so a
harmless upstream rename degrades into a clear error instead of a wrong gold
answer.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .config import DataConfig
from .types import Example
from .utils import stable_hash

log = logging.getLogger(__name__)

LoaderFn = Callable[[DataConfig, "DatasetSpec"], List[Example]]


@dataclass
class DatasetSpec:
    """Provenance for one registered benchmark, recorded in the paper's tables."""

    name: str
    hf_id: str
    default_split: str = "test"
    default_subset: Optional[str] = None
    answer_type: str = "math"
    #: True only if the id/config/split were confirmed against the live dataset
    #: page. Anything False must be treated as suspect until a run proves it.
    verified: bool = False
    gated: bool = False
    subsets: Sequence[str] = ()
    notes: str = ""
    loader: Optional[LoaderFn] = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "loader"}
        d["subsets"] = list(self.subsets)
        return d


DATASET_REGISTRY: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> Callable[[LoaderFn], LoaderFn]:
    """Decorator attaching a loader function to a `DatasetSpec`."""

    def deco(fn: LoaderFn) -> LoaderFn:
        spec.loader = fn
        if spec.name in DATASET_REGISTRY:
            raise ValueError(f"dataset {spec.name!r} is already registered")
        DATASET_REGISTRY[spec.name] = spec
        return fn

    return deco


def available_datasets() -> List[str]:
    return sorted(DATASET_REGISTRY)


def dataset_info(name: str) -> Dict[str, Any]:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"unknown dataset {name!r}. Available: {available_datasets()}")
    return DATASET_REGISTRY[name].as_dict()


# ------------------------------------------------------------------- HF plumbing
def _require_datasets() -> Any:
    try:
        import datasets  # type: ignore

        return datasets
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "the `datasets` package is required for this loader "
            "(pip install 'datasets>=3.0'). The built-in `toy` dataset needs no "
            "dependencies and is what the smoke test uses."
        ) from exc


def _load_hf_split(spec: DatasetSpec, cfg: DataConfig) -> Any:
    """Load one split, honouring `local_dir` and offline mode."""
    subset = cfg.subset if cfg.subset is not None else spec.default_subset
    split = cfg.split or spec.default_split

    if cfg.local_dir:
        local = _load_local(Path(cfg.local_dir), split)
        if local is not None:
            return local

    datasets = _require_datasets()
    offline = os.environ.get("HF_HUB_OFFLINE") or os.environ.get("HF_DATASETS_OFFLINE")
    try:
        return datasets.load_dataset(spec.hf_id, subset, split=split)
    except Exception as exc:  # noqa: BLE001 - turn library errors into advice
        message = str(exc)
        lowered = message.lower()
        if spec.gated or "gated" in lowered or "401" in lowered or "403" in lowered:
            raise RuntimeError(
                f"{spec.hf_id} looks gated or unauthorised. Accept the terms at "
                f"https://huggingface.co/datasets/{spec.hf_id} with your HF account, "
                "then expose a token as the HF_TOKEN environment variable "
                "(on Kaggle: Add-ons -> Secrets). Original error: " + message
            ) from exc
        if offline:
            raise RuntimeError(
                f"offline mode is on (HF_HUB_OFFLINE) and {spec.hf_id} is not in the "
                "local cache. Prefetch it with `python scripts/prefetch_assets.py "
                f"--datasets {spec.name}` in a session with internet, attach the "
                "result as a Kaggle Dataset, and pass --set data.local_dir=<path>. "
                "Original error: " + message
            ) from exc
        raise RuntimeError(
            f"failed to load {spec.hf_id} (config={subset!r}, split={split!r}). "
            f"verified={spec.verified}. If the id or split has changed upstream, fix "
            f"the DatasetSpec in src/datasets_.py. Original error: {message}"
        ) from exc


def _load_local(root: Path, split: str) -> Optional[Any]:
    """Load a prefetched dataset directory. Returns None if nothing usable."""
    jsonl = root / "examples.jsonl"
    if jsonl.exists():
        log.info("loading prefetched harness-schema examples from %s", jsonl)
        # Kept as dicts so the row-shaped code path below is identical to the
        # online one.
        return {"__examples__": [e.to_dict() for e in load_examples_jsonl(jsonl)]}

    datasets = _require_datasets()
    for candidate in (root / "arrow", root):
        if (candidate / "dataset_info.json").exists() or (
            candidate / "dataset_dict.json"
        ).exists():
            log.info("loading dataset from disk: %s", candidate)
            loaded = datasets.load_from_disk(str(candidate))
            if hasattr(loaded, "keys") and split in loaded:
                return loaded[split]
            return loaded
    log.warning("no prefetched dataset found under %s; falling back to the hub", root)
    return None


def _first_key(row: Dict[str, Any], candidates: Sequence[str], where: str) -> str:
    for key in candidates:
        if key in row:
            return key
    raise RuntimeError(
        f"{where}: none of the expected columns {list(candidates)} are present. "
        f"Found {sorted(row)}. The upstream schema probably changed; update the "
        "loader in src/datasets_.py."
    )


def _rows(dataset: Any) -> List[Dict[str, Any]]:
    if isinstance(dataset, dict) and "__examples__" in dataset:
        return dataset["__examples__"]
    return [dict(r) for r in dataset]


def _mc_letters(n: int) -> List[str]:
    return [chr(ord("A") + i) for i in range(n)]


# ----------------------------------------------------------------------- loaders
TOY_ITEMS: List[Dict[str, Any]] = [
    {"q": "What is 12 + 30?", "a": "42"},
    {"q": "A pack holds 8 pencils. How many pencils are in 3 packs?", "a": "24"},
    {"q": "Sam has 15 apples and gives away 6. How many are left?", "a": "9"},
    {"q": "What is 7 times 6?", "a": "42"},
    {"q": "A train travels 60 km in 2 hours. What is its speed in km per hour?", "a": "30"},
    {"q": "What is 100 divided by 4?", "a": "25"},
    {"q": "If x + 5 = 12, what is x?", "a": "7"},
    {"q": "What is one half of 9?", "a": "4.5"},
    {"q": "A rectangle is 6 by 4. What is its area?", "a": "24"},
    {"q": "A rectangle has area 48 and width 6. What is its perimeter?", "a": "28"},
    {"q": "What is 3 to the power of 3?", "a": "27"},
    {"q": "Solve for y: 3y = 21.", "a": "7"},
    {"q": "What is the value of 2/4 as a fraction in lowest terms?", "a": "1/2"},
    {"q": "What is the sum of the first 5 positive integers?", "a": "15"},
    {"q": "A shirt costs 20 dollars and is discounted by 25 percent. What is the new price in dollars?", "a": "15"},
    {"q": "What is the square root of 81?", "a": "9"},
    {
        "q": "Which planet is closest to the Sun?",
        "a": "B",
        "choices": ["Venus", "Mercury", "Mars", "Earth"],
        "type": "mc",
    },
    {
        "q": "Which object is the best conductor of electricity?",
        "a": "C",
        "choices": ["a rubber band", "a glass rod", "a copper wire", "a wooden spoon"],
        "type": "mc",
    },
    {
        "q": "What gas do plants primarily absorb during photosynthesis?",
        "a": "A",
        "choices": ["carbon dioxide", "oxygen", "nitrogen", "helium"],
        "type": "mc",
    },
    {
        "q": "Which of these is a prime number?",
        "a": "D",
        "choices": ["9", "15", "21", "13"],
        "type": "mc",
    },
    {"q": "Is 7 a prime number?", "a": "True", "type": "bool"},
    {"q": "Is 12 divisible by 5?", "a": "False", "type": "bool"},
    {
        "q": "Anna left before Ben, and Ben left before Chen. Who left first?",
        "a": "Anna",
        "type": "text",
    },
    {
        "q": "All maples in the park are older than every oak in the park. S is a maple and T is an oak. Is S older than T?",
        "a": "Yes",
        "type": "text",
    },
]


@register_dataset(
    DatasetSpec(
        name="toy",
        hf_id="(built-in)",
        default_split="test",
        answer_type="math",
        verified=True,
        notes=(
            "Synthetic, offline, dependency-free. Powers the smoke test and CI: it "
            "must never require the network or the datasets package."
        ),
    )
)
def _load_toy(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    out: List[Example] = []
    for i, item in enumerate(TOY_ITEMS):
        answer_type = item.get("type", "math")
        out.append(
            Example(
                id=f"toy/{cfg.split or 'test'}/{i}",
                question=item["q"],
                gold_answer=item["a"],
                choices=item.get("choices"),
                answer_type=answer_type,
                meta={
                    "orig_index": i,
                    "dataset": "toy",
                    "fewshot_pool": "math" if answer_type == "math" else answer_type,
                },
            )
        )
    return out


@register_dataset(
    DatasetSpec(
        name="gsm8k",
        hf_id="openai/gsm8k",
        default_subset="main",
        default_split="test",
        answer_type="math",
        verified=True,
        notes="Gold answer follows '#### '. 1319 test items.",
    )
)
def _load_gsm8k(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("question", "problem"), "gsm8k")
        ak = _first_key(row, ("answer", "solution"), "gsm8k")
        solution = str(row[ak])
        gold = solution.split("####")[-1].strip() if "####" in solution else solution.strip()
        out.append(
            Example(
                id=f"gsm8k/{cfg.split}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=gold.replace(",", ""),
                gold_reasoning=solution.split("####")[0].strip() or None,
                answer_type="math",
                meta={"orig_index": i, "dataset": "gsm8k"},
            )
        )
    return out


@register_dataset(
    DatasetSpec(
        name="math500",
        hf_id="HuggingFaceH4/MATH-500",
        default_split="test",
        answer_type="math",
        verified=True,
        notes="500 items sampled from MATH; columns problem/solution/answer/level/subject.",
    )
)
def _load_math500(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("problem", "question"), "math500")
        ak = _first_key(row, ("answer", "solution"), "math500")
        out.append(
            Example(
                id=f"math500/{cfg.split}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=str(row[ak]).strip(),
                gold_reasoning=str(row.get("solution") or "") or None,
                answer_type="math",
                meta={
                    "orig_index": i,
                    "dataset": "math500",
                    "level": row.get("level"),
                    "subject": row.get("subject") or row.get("type"),
                },
            )
        )
    return out


#: AIME year -> (hf id, split, verified). Only 30 problems per year, so these
#: runs are cheap but their confidence intervals are very wide.
AIME_SOURCES: Dict[str, Dict[str, Any]] = {
    "2024": {"hf_id": "Maxwell-Jia/AIME_2024", "split": "train", "verified": True},
    "2025": {"hf_id": "math-ai/aime25", "split": "test", "verified": False},
}


@register_dataset(
    DatasetSpec(
        name="aime",
        hf_id="Maxwell-Jia/AIME_2024",
        default_subset="2024",
        default_split="train",
        answer_type="math",
        verified=True,
        subsets=tuple(AIME_SOURCES),
        notes=(
            "`subset` selects the year. 2024 (Maxwell-Jia/AIME_2024, 30 rows, split "
            "'train', columns ID/Problem/Solution/Answer) is verified. 2025 "
            "(math-ai/aime25) resolves but its column names and split are NOT "
            "verified from here; the loader probes candidates and will raise a clear "
            "error if the schema differs."
        ),
    )
)
def _load_aime(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    year = str(cfg.subset or spec.default_subset)
    if year not in AIME_SOURCES:
        raise ValueError(
            f"unknown AIME year {year!r}; available: {sorted(AIME_SOURCES)}. "
            "Add an entry to AIME_SOURCES in src/datasets_.py for a new year."
        )
    source = AIME_SOURCES[year]
    year_spec = DatasetSpec(
        name=spec.name,
        hf_id=source["hf_id"],
        default_split=source["split"],
        default_subset=None,
        answer_type="math",
        verified=bool(source["verified"]),
    )
    # The per-year split name wins over the generic default.
    year_cfg = DataConfig(**{**cfg.__dict__, "subset": None, "split": source["split"]})
    rows = _rows(_load_hf_split(year_spec, year_cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("Problem", "problem", "question", "Question"), f"aime/{year}")
        ak = _first_key(row, ("Answer", "answer", "solution"), f"aime/{year}")
        out.append(
            Example(
                id=f"aime{year}/{source['split']}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=str(row[ak]).strip(),
                gold_reasoning=str(row.get("Solution") or row.get("solution") or "") or None,
                answer_type="math",
                meta={
                    "orig_index": i,
                    "dataset": "aime",
                    "year": year,
                    "problem_id": row.get("ID") or row.get("id"),
                },
            )
        )
    return out


#: BBH subsets exposed by default. `lukaemon/bbh` uses the task name as the HF
#: config and puts everything in the "test" split (250 items each).
BBH_SUBSETS: Dict[str, str] = {
    "date_understanding": "mc",
    "logical_deduction_seven_objects": "mc",
    "logical_deduction_three_objects": "mc",
    "tracking_shuffled_objects_seven_objects": "mc",
    "penguins_in_a_table": "mc",
    "causal_judgement": "text",
    "navigate": "text",
    "sports_understanding": "text",
    "boolean_expressions": "text",
    "web_of_lies": "text",
    "object_counting": "math",
    "multistep_arithmetic_two": "math",
}


@register_dataset(
    DatasetSpec(
        name="bbh",
        hf_id="lukaemon/bbh",
        default_subset="date_understanding",
        default_split="test",
        answer_type="mc",
        verified=True,
        subsets=tuple(BBH_SUBSETS),
        notes=(
            "One HF config per task, split 'test', columns input/target, 250 items "
            "each. Targets are '(A)'-style for the multiple-choice tasks and free "
            "text otherwise, so answer_type is set per subset. Note the alternative "
            "id maveriq/bigbenchhard uses the 'train' split instead."
        ),
    )
)
def _load_bbh(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    subset = str(cfg.subset or spec.default_subset)
    if subset not in BBH_SUBSETS:
        log.warning(
            "BBH subset %r is not in the curated list %s; assuming free-text answers",
            subset,
            sorted(BBH_SUBSETS),
        )
    answer_type = BBH_SUBSETS.get(subset, "text")
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)

    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("input", "question", "problem"), f"bbh/{subset}")
        ak = _first_key(row, ("target", "answer"), f"bbh/{subset}")
        question = str(row[qk]).strip()
        target = str(row[ak]).strip()
        choices = _parse_bbh_options(question)
        gold = target
        if answer_type == "mc":
            m = re.fullmatch(r"\(([A-Z])\)", target)
            if m:
                gold = m.group(1)
            elif choices:
                letter = None
                for j, choice in enumerate(choices):
                    if choice.strip().lower() == target.lower():
                        letter = _mc_letters(len(choices))[j]
                        break
                gold = letter or target
        out.append(
            Example(
                id=f"bbh/{subset}/{i}",
                question=question,
                gold_answer=gold,
                choices=choices if answer_type == "mc" else None,
                answer_type=answer_type if (choices or answer_type != "mc") else "text",
                meta={"orig_index": i, "dataset": "bbh", "subset": subset,
                      "raw_target": target, "fewshot_pool": "logic"},
            )
        )
    return out


def _parse_bbh_options(question: str) -> Optional[List[str]]:
    """Pull the inline 'Options:\n(A) ...' block out of a BBH prompt."""
    found = re.findall(r"^\s*\(([A-Z])\)\s*(.+)$", question, re.MULTILINE)
    if len(found) < 2:
        return None
    return [text.strip() for _letter, text in found]


MUSR_DOMAINS = ("murder_mysteries", "object_placements", "team_allocation")


@register_dataset(
    DatasetSpec(
        name="musr",
        hf_id="TAUR-Lab/MuSR",
        default_subset="default",
        default_split="murder_mysteries",
        answer_type="mc",
        verified=True,
        subsets=MUSR_DOMAINS,
        notes=(
            "The three domains are SPLITS, not configs: config is 'default' and "
            "split is one of murder_mysteries (250) / object_placements (256) / "
            "team_allocation (250). Columns: narrative, question, choices (a string "
            "repr of a python list), answer_index. Long narratives, so prompt tokens "
            "dominate the cost here."
        ),
    )
)
def _load_musr(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    # Accept the domain in either `subset` or `split`, since it is natural to
    # write it in either field, and map it onto the split MuSR actually uses.
    domain = cfg.split if cfg.split in MUSR_DOMAINS else None
    if domain is None and cfg.subset in MUSR_DOMAINS:
        domain = cfg.subset
    domain = domain or spec.default_split
    musr_cfg = DataConfig(**{**cfg.__dict__, "subset": "default", "split": domain})
    rows = _rows(_load_hf_split(spec, musr_cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)

    out: List[Example] = []
    for i, row in enumerate(rows):
        nk = _first_key(row, ("narrative", "context", "story"), "musr")
        qk = _first_key(row, ("question",), "musr")
        ck = _first_key(row, ("choices", "options"), "musr")
        ik = _first_key(row, ("answer_index", "label", "answer"), "musr")
        choices = _coerce_choices(row[ck])
        try:
            index = int(row[ik])
        except (TypeError, ValueError):
            index = -1
        letters = _mc_letters(len(choices) or 1)
        gold = letters[index] if 0 <= index < len(letters) else str(row[ik])
        out.append(
            Example(
                id=f"musr/{domain}/{i}",
                question=f"{str(row[nk]).strip()}\n\n{str(row[qk]).strip()}",
                gold_answer=gold,
                choices=choices or None,
                answer_type="mc" if choices else "text",
                meta={"orig_index": i, "dataset": "musr", "domain": domain,
                      "fewshot_pool": "logic"},
            )
        )
    return out


def _coerce_choices(value: Any) -> List[str]:
    """MuSR stores choices as the *string* repr of a python list."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value or "").strip()
    if not text:
        return []
    for parser in (ast.literal_eval, json.loads):
        try:
            parsed = parser(text)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except Exception:  # noqa: BLE001 - try the next parser
            continue
    return [p.strip() for p in text.split("|")] if "|" in text else [text]


@register_dataset(
    DatasetSpec(
        name="arc_challenge",
        hf_id="allenai/ai2_arc",
        default_subset="ARC-Challenge",
        default_split="test",
        answer_type="mc",
        verified=True,
        subsets=("ARC-Challenge", "ARC-Easy"),
        notes=(
            "1172 test items. Columns question / choices{text,label} / answerKey. "
            "Labels are usually A-D but a few items use 1-4, which is remapped."
        ),
    )
)
def _load_arc(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("question",), "arc")
        raw_choices = row.get("choices") or {}
        if isinstance(raw_choices, dict):
            texts = [str(t) for t in raw_choices.get("text", [])]
            labels = [str(l) for l in raw_choices.get("label", [])]
        else:
            texts = [str(c) for c in raw_choices]
            labels = []
        key = str(row.get("answerKey") or row.get("answer") or "").strip()
        letters = _mc_letters(len(texts))
        gold = key.upper()
        if labels and key in labels:
            gold = letters[labels.index(key)]
        elif key.isdigit() and 1 <= int(key) <= len(letters):
            gold = letters[int(key) - 1]
        out.append(
            Example(
                id=f"arc_challenge/{cfg.split}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=gold,
                choices=texts,
                answer_type="mc",
                meta={"orig_index": i, "dataset": "arc_challenge", "raw_key": key},
            )
        )
    return out


@register_dataset(
    DatasetSpec(
        name="gpqa_diamond",
        hf_id="Idavidrein/gpqa",
        default_subset="gpqa_diamond",
        default_split="train",
        answer_type="mc",
        verified=True,
        gated=True,
        subsets=("gpqa_diamond", "gpqa_main", "gpqa_extended"),
        notes=(
            "GATED: accept the terms on the dataset page and set HF_TOKEN. Ships a "
            "single split named 'train' (198 diamond items). The correct and "
            "incorrect answers live in separate columns, so options are shuffled "
            "deterministically per item to avoid a fixed-position bias."
        ),
    )
)
def _load_gpqa(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("Question", "question"), "gpqa")
        ck = _first_key(row, ("Correct Answer", "correct_answer"), "gpqa")
        distractors = [
            str(row[key]).strip()
            for key in (
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            )
            if row.get(key) is not None
        ]
        correct = str(row[ck]).strip()
        options = [correct, *distractors]
        # Deterministic per-item shuffle: the correct answer must not always be
        # option A, but the ordering must be identical on every rerun and every
        # machine (so the config hash and the uid stay meaningful).
        order = list(range(len(options)))
        random.Random(int(stable_hash({"gpqa": i, "q": correct}, 8), 16)).shuffle(order)
        shuffled = [options[j] for j in order]
        gold = _mc_letters(len(shuffled))[shuffled.index(correct)]
        out.append(
            Example(
                id=f"gpqa_diamond/{cfg.split}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=gold,
                choices=shuffled,
                answer_type="mc",
                meta={
                    "orig_index": i,
                    "dataset": "gpqa_diamond",
                    "domain": row.get("High-level domain") or row.get("Subdomain"),
                    "option_order": order,
                },
            )
        )
    return out


@register_dataset(
    DatasetSpec(
        name="gsm_symbolic",
        hf_id="apple/GSM-Symbolic",
        default_subset="main",
        default_split="test",
        answer_type="math",
        verified=False,
        subsets=("main", "p1", "p2"),
        notes=(
            "UNVERIFIED: the HF API was unreachable from the development machine, so "
            "the config and split names are assumed by analogy with GSM8K. Confirm "
            "with `python -m src.runner --config ... --dry-run` before spending "
            "quota. Templated GSM8K variants, useful as a contamination-resistant "
            "replicate; `template_id` is recorded in meta so analyses can bootstrap "
            "over templates rather than instances (instances from one template are "
            "not independent)."
        ),
    )
)
def _load_gsm_symbolic(cfg: DataConfig, spec: DatasetSpec) -> List[Example]:
    rows = _rows(_load_hf_split(spec, cfg))
    if rows and "gold_answer" in rows[0]:
        return _from_harness_rows(rows, spec, cfg)
    subset = str(cfg.subset or spec.default_subset)
    out: List[Example] = []
    for i, row in enumerate(rows):
        qk = _first_key(row, ("question", "problem"), "gsm_symbolic")
        ak = _first_key(row, ("answer", "solution", "final_answer"), "gsm_symbolic")
        solution = str(row[ak])
        gold = solution.split("####")[-1].strip() if "####" in solution else solution.strip()
        out.append(
            Example(
                id=f"gsm_symbolic/{subset}/{i}",
                question=str(row[qk]).strip(),
                gold_answer=gold.replace(",", ""),
                gold_reasoning=solution.split("####")[0].strip() or None,
                answer_type="math",
                meta={
                    "orig_index": i,
                    "dataset": "gsm_symbolic",
                    "subset": subset,
                    "template_id": row.get("template_id")
                    or row.get("instance_id")
                    or row.get("id"),
                },
            )
        )
    return out


def _from_harness_rows(
    rows: Sequence[Dict[str, Any]], spec: DatasetSpec, cfg: DataConfig
) -> List[Example]:
    """Rebuild Examples from prefetched harness-schema JSONL rows."""
    return [Example.from_dict(r) for r in rows]


# ------------------------------------------------------- subsampling and sharding
#: METHOD_SPEC section 2: the one fixed permutation both tiers are drawn from.
SPEC_PERMUTATION_SEED = 20260729


def spec_item_indices(
    n_rows: int, n_per_tier: int, tier: str = "A"
) -> List[int]:
    """The METHOD_SPEC section 2 subsampling protocol.

    Sort by dataset index, draw one fixed permutation from
    `numpy.random.default_rng(20260729)`, then take the first `n_per_tier` for
    Tier A and the next `n_per_tier` for Tier B. Drawing both batches from a
    single permutation is the point: the Tier B items are a fresh independent
    sample from the same population, so the two batches pool with no reweighting
    and the expansion cannot be a biased top-up.

    Returned indices are sorted ascending so iteration order follows the dataset.
    """
    import numpy as np

    tier = str(tier).upper()
    if tier not in ("A", "B", "AB"):
        raise ValueError(f"tier must be 'A', 'B' or 'AB', got {tier!r}")
    permutation = np.random.default_rng(SPEC_PERMUTATION_SEED).permutation(n_rows)
    take = int(n_per_tier)
    if take <= 0:
        raise ValueError("n_per_tier must be positive under the spec protocol")
    if tier == "A":
        chosen = permutation[:take]
    elif tier == "B":
        chosen = permutation[take : 2 * take]
    else:
        chosen = permutation[: 2 * take]
    if len(chosen) < take and tier != "AB":
        log.warning(
            "tier %s asked for %d items but the split only supplies %d after the "
            "permutation; the design is no longer balanced against the other tier",
            tier,
            take,
            len(chosen),
        )
    return sorted(int(i) for i in chosen)


def subsample_spec(
    examples: List[Example], n: Optional[int], tier: str = "A"
) -> List[Example]:
    """Subsample under the spec protocol, tagging each example with its tier."""
    if n is None or n <= 0:
        return list(examples)
    indices = spec_item_indices(len(examples), n, tier)
    chosen: List[Example] = []
    for position in indices:
        ex = examples[position]
        ex.meta = {
            **ex.meta,
            "subsample_protocol": "spec",
            "subsample_permutation_seed": SPEC_PERMUTATION_SEED,
            "subsample_n": n,
            "subsample_position": position,
            "tier": str(tier).upper(),
        }
        chosen.append(ex)
    return chosen


def write_item_ids(
    examples: Sequence[Example],
    dataset: str,
    subset: Optional[str],
    tier: str,
    out_dir: str | os.PathLike,
) -> Path:
    """Persist the selected item ids, tagged by tier, for the run manifest.

    METHOD_SPEC section 2 requires this file: every model/configuration cell
    within a tier must use the *identical* item set, and the variance
    decomposition is only valid on a fully crossed design. Having the ids on disk
    turns "did every cell see the same items?" into a file comparison instead of
    an act of faith.
    """
    from .utils import write_json_atomic

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{dataset}{'_' + subset if subset else ''}")
    path = Path(out_dir) / f"item_ids_{slug}.json"
    payload = {
        "dataset": dataset,
        "subset": subset,
        "tier": str(tier).upper(),
        "permutation_seed": SPEC_PERMUTATION_SEED,
        "n_items": len(examples),
        "example_ids": [e.id for e in examples],
        "orig_indices": [int(e.meta.get("orig_index", -1)) for e in examples],
        "index_sha": stable_hash(
            [int(e.meta.get("orig_index", -1)) for e in examples], 12
        ),
    }
    write_json_atomic(path, payload)
    return path


def apply_paraphrases(
    examples: List[Example], path: str | os.PathLike
) -> List[Example]:
    """Replace question text from a paraphrase map, for the `c6` arm.

    Items with no accepted paraphrase are **dropped**, not silently left in their
    original wording: leaving them would make `c6` a mixture of paraphrased and
    original items and quietly bias the arm towards `c0`. The exclusion list is
    what METHOD_SPEC section 3 asks to be reported.
    """
    from .utils import read_json

    payload = read_json(path, default=None)
    if payload is None:
        raise FileNotFoundError(
            f"paraphrase file {path} not found. Generate it first with "
            "`python scripts/make_paraphrases.py --config <cfg>`; the c6 arm cannot "
            "run without it."
        )
    mapping = payload.get("paraphrases", payload) or {}
    kept: List[Example] = []
    excluded: List[str] = []
    for ex in examples:
        text = mapping.get(ex.id)
        if not text:
            excluded.append(ex.id)
            continue
        ex.question = str(text)
        ex.meta = {**ex.meta, "paraphrased": True, "original_question_sha": stable_hash(
            ex.meta.get("original_question", ""), 8
        )}
        kept.append(ex)
    if excluded:
        log.warning(
            "paraphrase arm excludes %d/%d item(s) with no accepted paraphrase "
            "(reported in the c6 exclusion list): %s",
            len(excluded),
            len(examples),
            ", ".join(excluded[:8]) + (" ..." if len(excluded) > 8 else ""),
        )
    if not kept:
        raise RuntimeError(
            f"every item was excluded by the paraphrase gate in {path}; the c6 arm "
            "has nothing to run"
        )
    return kept


def subsample(examples: List[Example], n: Optional[int], seed: int) -> List[Example]:
    """Deterministically take `n` examples, preserving dataset order.

    `random.Random(seed).sample` on the index range is used rather than numpy so
    the selection cannot change with a numpy version bump - the subsample is part
    of the experiment's identity and must be reproducible years later. Indices
    are sorted so iteration order matches the dataset, and the selection is
    recorded in each example's meta.
    """
    if n is None or n >= len(examples) or n <= 0:
        return list(examples)
    indices = sorted(random.Random(seed).sample(range(len(examples)), n))
    chosen = []
    for position in indices:
        ex = examples[position]
        ex.meta = {
            **ex.meta,
            "subsample_seed": seed,
            "subsample_n": n,
            "subsample_position": position,
        }
        chosen.append(ex)
    return chosen


def shard(examples: List[Example], shard_index: int, num_shards: int) -> List[Example]:
    """Strided split, so every shard sees a similar difficulty mix."""
    if num_shards <= 1:
        return list(examples)
    if not 0 <= shard_index < num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    out = []
    for i, ex in enumerate(examples):
        if i % num_shards == shard_index:
            ex.meta = {**ex.meta, "shard_index": shard_index, "num_shards": num_shards}
            out.append(ex)
    return out


def load_dataset_examples(cfg: DataConfig) -> List[Example]:
    """The runner's entry point: load, subsample, shard, and log provenance."""
    if cfg.name not in DATASET_REGISTRY:
        raise KeyError(
            f"unknown dataset {cfg.name!r}. Available: {available_datasets()}"
        )
    spec = DATASET_REGISTRY[cfg.name]
    if spec.loader is None:  # pragma: no cover - registration guarantees this
        raise RuntimeError(f"dataset {cfg.name!r} has no loader")
    if not spec.verified:
        log.warning(
            "dataset %r (%s) is marked UNVERIFIED: %s",
            spec.name,
            spec.hf_id,
            spec.notes.split(".")[0],
        )

    examples = spec.loader(cfg, spec)
    n_loaded = len(examples)
    protocol = str(getattr(cfg, "subsample_protocol", "legacy")).lower()
    if protocol == "spec":
        examples = subsample_spec(examples, cfg.subsample, getattr(cfg, "tier", "A"))
    elif protocol == "legacy":
        examples = subsample(examples, cfg.subsample, cfg.subsample_seed)
    else:
        raise ValueError(
            f"unknown subsample_protocol {protocol!r}; expected 'legacy' or 'spec'"
        )
    n_subsampled = len(examples)
    examples = shard(examples, cfg.shard_index, cfg.num_shards)

    paraphrase_file = getattr(cfg, "paraphrase_file", None)
    if paraphrase_file:
        examples = apply_paraphrases(examples, paraphrase_file)

    index_sha = stable_hash([e.meta.get("orig_index", i) for i, e in enumerate(examples)], 12)
    if protocol == "spec" and getattr(cfg, "item_ids_dir", None):
        try:
            write_item_ids(
                examples, cfg.name, cfg.subset, getattr(cfg, "tier", "A"), cfg.item_ids_dir
            )
        except OSError:  # noqa: BLE001 - a read-only working dir must not kill a run
            log.warning("could not persist item ids to %s", cfg.item_ids_dir, exc_info=True)
    log.info(
        "dataset %s (%s) split=%s subset=%s: loaded %d -> subsampled %d "
        "(protocol=%s tier=%s n=%s seed=%s) -> shard %d/%d = %d examples "
        "[index_sha=%s]",
        cfg.name,
        spec.hf_id,
        cfg.split,
        cfg.subset,
        n_loaded,
        n_subsampled,
        protocol,
        getattr(cfg, "tier", "A"),
        cfg.subsample,
        cfg.subsample_seed,
        cfg.shard_index,
        cfg.num_shards,
        len(examples),
        index_sha,
    )
    return examples


# ------------------------------------------------------------- offline round-trip
def cache_dir_for(name: str, root: str | os.PathLike) -> Path:
    """Canonical per-dataset directory used by scripts/prefetch_assets.py."""
    return Path(root) / re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def save_examples_jsonl(examples: Iterable[Example], path: str | os.PathLike) -> Path:
    """Persist Examples as JSONL: small, diffable, and loadable without `datasets`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_examples_jsonl(path: str | os.PathLike) -> List[Example]:
    out: List[Example] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Example.from_dict(json.loads(line)))
    return out
