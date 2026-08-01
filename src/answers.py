"""Answer extraction and equivalence checking.

This module is load-bearing for the whole project: a weak extractor silently
understates every method's accuracy, and the failure is invisible in aggregate
numbers. It is therefore deliberately layered and separately testable, and it is
used by *both* the strategies (to pick a final answer) and the grader (to judge
it), so a strategy can never be graded against a different notion of "answer"
than the one it produced.

Equivalence order (cheap and strict first):
  1. exact match after normalisation
  2. numeric match with tolerance (handles fractions, percents, trailing zeros)
  3. unordered set/tuple match for list-valued answers
  4. symbolic match via `math_verify`, else `sympy`, else nothing

`grader_backend_name()` reports which symbolic layer is actually installed, and
it is recorded in every graded record so the paper can state how answers were
judged.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

#: Bump on any behaviour change: `grade_file` re-grades automatically when it
#: differs from the version stored in graded.jsonl. Cheap, because grading never
#: needs the GPU.
GRADER_VERSION = "1.0.0"

#: Symbolic inputs longer than this are not handed to sympy. Long strings are
#: where sympy parsing pathologically hangs, and a hang in the grading pass of a
#: 200-example run is a wasted session.
MAX_SYMBOLIC_CHARS = 200

_LETTERS = "ABCDEFGHIJ"

#: Units and trailing words stripped from math answers before comparison. Kept
#: as an explicit list rather than a generic "strip trailing letters" rule,
#: which would corrupt algebraic answers such as "2x" or "3n".
_UNITS = (
    "square centimeters", "square centimetres", "square meters", "square metres",
    "square inches", "square feet", "cubic centimeters", "cubic meters",
    "centimeters", "centimetres", "millimeters", "millimetres", "kilometers",
    "kilometres", "kilograms", "milligrams", "milliliters", "millilitres",
    "liters", "litres", "meters", "metres", "inches", "inch", "feet", "foot",
    "yards", "yard", "miles", "mile", "grams", "gram", "pounds", "pound",
    "ounces", "ounce", "tons", "ton", "seconds", "second", "minutes", "minute",
    "hours", "hour", "days", "day", "weeks", "week", "months", "month",
    "years", "year", "degrees", "degree", "radians", "radian", "dollars",
    "dollar", "cents", "cent", "percent", "people", "students", "children",
    "boys", "girls", "men", "women", "books", "apples", "oranges", "eggs",
    "cups", "cup", "times", "units", "unit", "cm", "mm", "km", "kg", "mg",
    "ml", "lb", "lbs", "oz", "ft", "hrs", "hr", "min", "sec", "sq",
)

_BOOL_TRUE = {"true", "yes", "correct", "valid", "1", "t", "y"}
_BOOL_FALSE = {"false", "no", "incorrect", "invalid", "0", "f", "n"}

#: A number with optional sign, thousands separators, decimals and exponent.
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"
)

#: "the answer is X" and friends. An explicit separator (is/:/=) is REQUIRED:
#: without it, prose like "no answer at all" matches and yields "at all".
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer|result|solution)"
    r"\s*(?:\*\*|__|\*)?\s*(?::|=|\bis\b|\bare\b)\s*(?:\*\*|__|\*)?\s*",
    re.IGNORECASE,
)

#: Where an answer ends: a newline, or a full stop that closes the value.
_TAIL_END_RE = re.compile(r"\n|(?<=[\d\}\)])\s*\.\s|\.\s+[A-Z]")


# --------------------------------------------------------------------- backends
def _math_verify():
    try:
        import math_verify  # type: ignore

        return math_verify
    except Exception:
        return None


def _sympy():
    try:
        import sympy  # type: ignore

        return sympy
    except Exception:
        return None


def grader_backend_name() -> str:
    """Which symbolic equivalence backend is active: report this in the paper."""
    if _math_verify() is not None:
        return "math_verify"
    if _sympy() is not None:
        return "sympy"
    return "regex"


# -------------------------------------------------------------------- extraction
def _find_balanced(text: str, start: int) -> Optional[Tuple[str, int]]:
    """Read a `{...}` group at `start`, respecting nesting. Returns (body, end)."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    # Unbalanced (truncated generation): take the rest, which is usually right.
    return text[start + 1 :], len(text)


def extract_boxed(text: str, macros: Sequence[str] = (r"\boxed", r"\fbox")) -> Optional[str]:
    """The content of the LAST \\boxed{...} / \\fbox{...} in the text.

    The last one matters because models often restate the answer at the end, and
    because a chain of thought may box intermediate results.
    """
    best: Optional[str] = None
    for macro in macros:
        idx = 0
        while True:
            found = text.find(macro, idx)
            if found == -1:
                break
            after = found + len(macro)
            while after < len(text) and text[after] in " \t":
                after += 1
            if after < len(text) and text[after] == "{":
                got = _find_balanced(text, after)
                if got is not None:
                    best = got[0]
                    idx = got[1]
                    continue
            # `\boxed 42` with no braces: take the next token.
            m = re.match(r"([^\s$,.;]+)", text[after:])
            if m:
                best = m.group(1)
            idx = after + 1
    return best.strip() if best is not None else None


def _extract_after_phrase(text: str) -> Optional[str]:
    """The value following the LAST 'the answer is'-style phrase.

    Only the phrase itself is matched, never the tail, so that every occurrence
    is found: a greedy `.+` would let the first match swallow the rest of the
    text and hide the model's final restatement of its answer.
    """
    best: Optional[str] = None
    for m in _ANSWER_PHRASE_RE.finditer(text):
        tail = text[m.end() :].strip()
        if not tail:
            continue
        tail = _TAIL_END_RE.split(tail, maxsplit=1)[0]
        tail = tail.strip().strip("*_ \t")
        if tail:
            best = tail
    return best.rstrip(".,;:").strip() if best else None


def _looks_like_math_answer(value: str) -> bool:
    """Reject prose that merely followed the word "answer".

    A mathematical answer contains a digit, or LaTeX, or is a single short token
    (a bare variable). "at all" is none of those.
    """
    if not value:
        return False
    if any(ch.isdigit() for ch in value) or "\\" in value:
        return True
    return len(value) <= 12 and not any(ch.isspace() for ch in value)


def _extract_hash_marker(text: str) -> Optional[str]:
    """GSM8K gold format: `#### 42`."""
    matches = re.findall(r"####\s*(.+)", text)
    return matches[-1].strip() if matches else None


def _extract_last_number(text: str) -> Optional[str]:
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1] if numbers else None


def _letter_from_choices(value: str, choices: Optional[Sequence[str]]) -> Optional[str]:
    """Map a full option *text* back to its letter, for verbose answers."""
    if not choices or not value:
        return None
    needle = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    if not needle:
        return None
    for i, choice in enumerate(choices):
        hay = re.sub(r"[^a-z0-9]+", " ", str(choice).lower()).strip()
        if hay and (hay == needle or (len(hay) > 3 and hay in needle)):
            return _LETTERS[i]
    return None


def _extract_mc(text: str, choices: Optional[Sequence[str]]) -> Optional[str]:
    """A multiple-choice letter, from the many shapes models emit."""
    n = len(choices) if choices else len(_LETTERS)
    valid = _LETTERS[: max(1, min(n, len(_LETTERS)))]
    cls = f"[{valid}]"

    boxed = extract_boxed(text)
    if boxed:
        m = re.search(rf"\b({cls})\b", boxed.upper())
        if m:
            return m.group(1)
        letter = _letter_from_choices(boxed, choices)
        if letter:
            return letter

    phrase = _extract_after_phrase(text)
    if phrase:
        m = re.match(rf"^\W*({cls})\b", phrase.upper())
        if m:
            return m.group(1)
        letter = _letter_from_choices(phrase, choices)
        if letter:
            return letter

    # "(B)", "B.", "option B", "choice B:" - take the last occurrence.
    patterns = (
        rf"\(({cls})\)",
        rf"\b(?:option|choice|answer)\s*[:\-]?\s*({cls})\b",
        rf"(?:^|\n)\s*({cls})[.):]",
        rf"\b({cls})\b(?=\s*(?:$|\n|\.))",
    )
    for pat in patterns:
        found = re.findall(pat, text, re.MULTILINE | re.IGNORECASE)
        if found:
            return found[-1].upper()
    return None


def _extract_bool(text: str) -> Optional[str]:
    for candidate in (extract_boxed(text), _extract_after_phrase(text), text):
        if not candidate:
            continue
        words = re.findall(r"[A-Za-z]+", candidate.lower())
        for w in words if candidate is text else reversed(words):
            if w in _BOOL_TRUE:
                return "True"
            if w in _BOOL_FALSE:
                return "False"
    return None


def extract_answer(
    text: str,
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Extract the final answer from a model response. Never raises.

    Returns None when nothing plausible is present. None is a meaningful value:
    extraction failure is its own outcome class and must be counted rather than
    silently dropped, because dropping it inflates accuracy.
    """
    if not text or not str(text).strip():
        return None
    text = str(text)

    try:
        if answer_type == "mc":
            return _extract_mc(text, choices)
        if answer_type == "bool":
            return _extract_bool(text)

        for extractor in (extract_boxed, _extract_after_phrase, _extract_hash_marker):
            value = extractor(text)
            if not value:
                continue
            cleaned = value.strip().strip("$ \t")
            if not cleaned:
                continue
            # A phrase match on prose ("no answer at all") must not become the
            # answer; fall through to the last-number heuristic instead.
            if (
                answer_type == "math"
                and extractor is _extract_after_phrase
                and not _looks_like_math_answer(cleaned)
            ):
                continue
            return cleaned
        if answer_type == "math":
            return _extract_last_number(text)
        return None
    except Exception:  # noqa: BLE001 - extraction must never break a run
        log.debug("answer extraction failed", exc_info=True)
        return None


def extract_answer_alt(
    text: str,
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Second independent extractor: rule-based numeric / LaTeX path.

    Differs from the primary path by preferring the last number and simple
    `\\boxed{}` parsing without math-verify, giving a different failure profile
    for the extraction-audit confound control.
    """
    if not text or not str(text).strip():
        return None
    text = str(text)
    try:
        if answer_type == "mc":
            return _extract_mc(text, choices)
        if answer_type == "bool":
            return _extract_bool(text)
        boxed = extract_boxed(text)
        if boxed:
            return boxed.strip().strip("$")
        num = _extract_last_number(text)
        if num is not None:
            return num
        phrase = _extract_after_phrase(text)
        if phrase and _looks_like_math_answer(phrase):
            return phrase.strip()
        return None
    except Exception:
        return None


def extract_with_method(
    text: str,
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
    method: str = "primary",
) -> Tuple[Optional[str], str, bool]:
    """Extract an answer and report method + failure flag."""
    if method == "alt":
        ans = extract_answer_alt(text, answer_type, choices)
        return ans, "rule_numeric_latex", ans is None
    ans = extract_answer(text, answer_type, choices)
    return ans, grader_backend_name(), ans is None


# ----------------------------------------------------------------- normalisation
def _strip_latex_wrappers(s: str) -> str:
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\$", "").replace("\\%", "%").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\(?:text|mbox|textrm|mathrm|textbf)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:d|t)frac", r"\\frac", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)
    s = s.replace("\\ ", " ").replace("$", "")
    s = re.sub(r"\\+$", "", s)
    return s


def _strip_units(s: str) -> str:
    low = s.lower()
    for unit in _UNITS:  # longest-first ordering is built into _UNITS
        if low.endswith(unit):
            candidate = s[: len(s) - len(unit)].strip().rstrip("\\").strip()
            # Only strip when what remains still looks like a value, so "cm" is
            # removed from "5 cm" but "x" is not removed from "2x".
            if candidate and _to_float(candidate) is not None:
                return candidate
    return s


def _canonical_number_string(value: float) -> str:
    """A canonical string so 2.50, 2.5 and 5/2 collapse to the same key."""
    if value == 0:  # kills the "-0" vs "0" distinction
        return "0"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.12g}"


def normalize_answer(answer: str, answer_type: str = "math") -> str:
    """Canonical form used for exact matching and for vote clustering."""
    if answer is None:
        return ""
    s = str(answer).strip()
    if not s:
        return ""

    if answer_type == "mc":
        m = re.search(r"[A-Ja-j]", s)
        return m.group(0).upper() if m else s.strip().upper()
    if answer_type == "bool":
        low = s.strip().lower().strip(".")
        if low in _BOOL_TRUE:
            return "True"
        if low in _BOOL_FALSE:
            return "False"
        return s.strip()

    s = _strip_latex_wrappers(s)
    s = s.replace("\u2212", "-").replace("\u00d7", "*")
    s = re.sub(r"\s+", " ", s).strip()
    # Only TRAILING punctuation may be stripped: a leading "." is a decimal
    # point (".50" must stay 0.5, not become 50).
    s = s.rstrip(".,;: \t")
    s = _strip_units(s)
    s = s.lstrip("+")
    # Thousands separators only between digit groups, never a list comma.
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)

    number = _to_float(s)
    if number is not None:
        return _canonical_number_string(number)

    s = s.replace(" ", "")
    return s


# ---------------------------------------------------------------- numeric parsing
def _to_float(s: str) -> Optional[float]:
    """Parse a scalar answer to a float, or None. Handles fractions and percents.

    A percent sign is *stripped*, not divided by 100: benchmark gold answers
    write "50" where a model writes "50\\%", and treating those as different
    would be a grading bug, not strictness.
    """
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    t = _strip_latex_wrappers(t).strip()
    t = t.replace("%", "").replace("\u2212", "-").strip()
    t = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", t)
    t = t.lstrip("+").strip()
    if not t:
        return None

    # \frac{a}{b}
    m = re.fullmatch(r"(-?)\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", t)
    if m:
        num, den = _to_float(m.group(2)), _to_float(m.group(3))
        if num is not None and den not in (None, 0):
            return -num / den if m.group(1) else num / den
        return None

    # Mixed number: "2 1/2"
    m = re.fullmatch(r"(-?)(\d+)\s+(\d+)\s*/\s*(\d+)", t)
    if m:
        whole, num, den = float(m.group(2)), float(m.group(3)), float(m.group(4))
        if den:
            value = whole + num / den
            return -value if m.group(1) else value
        return None

    # Plain fraction: "3/4"
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", t)
    if m:
        den = float(m.group(2))
        return float(m.group(1)) / den if den else None

    try:
        return float(t)
    except ValueError:
        return None


def _split_list(s: str) -> Optional[List[str]]:
    """Split "1, 2" / "(1,2)" / "{1,2}" into parts, respecting nesting."""
    t = str(s).strip()
    if not t:
        return None
    if len(t) >= 2 and t[0] in "([{" and t[-1] in ")]}":
        t = t[1:-1]
    parts: List[str] = []
    depth = 0
    current = ""
    for ch in t:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else None


# ------------------------------------------------------------------- equivalence
def _numeric_equal(a: str, b: str) -> Optional[bool]:
    fa, fb = _to_float(a), _to_float(b)
    if fa is None or fb is None:
        return None
    return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-9)


def _set_equal(a: str, b: str) -> Optional[bool]:
    la, lb = _split_list(a), _split_list(b)
    if la is None or lb is None:
        return None
    if len(la) != len(lb):
        return False
    norm_a = sorted(normalize_answer(x) for x in la)
    norm_b = sorted(normalize_answer(x) for x in lb)
    return norm_a == norm_b


def _symbolic_equal(a: str, b: str) -> Optional[bool]:
    """Symbolic equivalence, guarded so a pathological input cannot hang grading."""
    if len(a) > MAX_SYMBOLIC_CHARS or len(b) > MAX_SYMBOLIC_CHARS:
        return None

    mv = _math_verify()
    if mv is not None:
        try:
            parse, verify = mv.parse, mv.verify
            gold, pred = parse(f"${b}$"), parse(f"${a}$")
            if gold and pred:
                return bool(verify(gold, pred))
        except Exception:  # noqa: BLE001 - fall through to sympy
            log.debug("math_verify failed on %r vs %r", a, b, exc_info=True)

    sp = _sympy()
    if sp is None:
        return None
    try:
        ea, eb = _sympify(sp, a), _sympify(sp, b)
        if ea is None or eb is None:
            return None
        if bool(sp.simplify(ea - eb) == 0):
            return True
        # An exact form compared against a rounded decimal (\frac{\sqrt{2}}{2}
        # vs 0.7071...) will never simplify to exactly 0, so fall back to a
        # numeric evaluation. The tolerance is tight enough that 1/3 and 0.333
        # still count as different answers.
        fa, fb = complex(sp.N(ea)), complex(sp.N(eb))
        if fa.imag or fb.imag:
            return abs(fa - fb) <= 1e-6 * max(1.0, abs(fa), abs(fb))
        return math.isclose(fa.real, fb.real, rel_tol=1e-6, abs_tol=1e-9)
    except Exception:  # noqa: BLE001 - sympy raises a wide variety of errors
        log.debug("sympy comparison failed on %r vs %r", a, b, exc_info=True)
        return None


def _latex_to_sympy(text: str) -> str:
    """Rewrite the small LaTeX subset that benchmark answers actually use.

    `sympy.parsing.latex.parse_latex` needs the optional antlr4 runtime, which
    is absent more often than not (including on a stock Kaggle image), so this
    textual rewrite is the path that actually runs. It covers \\frac, \\sqrt and
    the common constants, which is the overwhelming majority of MATH-style
    answers.
    """
    out = text
    for _ in range(4):  # nested \frac needs a few passes
        new = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", out)
        new = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{([^{}]+)\}", r"((\2)**(1/(\1)))", new)
        new = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", new)
        if new == out:
            break
        out = new
    out = out.replace("\\pi", "pi").replace("\\infty", "oo")
    out = re.sub(r"\\(?:log|ln)\s*\{?([^{}\s]+)\}?", r"log(\1)", out)
    out = re.sub(r"\\(sin|cos|tan|sec|csc|cot|exp)\b", r"\1", out)
    out = out.replace("{", "(").replace("}", ")")
    return out


def _sympify(sp: Any, s: str) -> Any:
    text = _strip_latex_wrappers(str(s)).strip()
    if not text:
        return None
    if "\\" in text:
        try:
            from sympy.parsing.latex import parse_latex  # type: ignore

            return parse_latex(text)
        except Exception:
            pass
    # `parse_expr` with implicit multiplication is what makes "2sqrt(3)" and
    # "2x" parse at all; plain `sympify` rejects both, and benchmark answers are
    # full of them.
    from sympy.parsing.sympy_parser import (  # type: ignore
        convert_xor,
        implicit_multiplication_application,
        standard_transformations,
    )

    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )
    return sp.parsing.sympy_parser.parse_expr(
        _latex_to_sympy(text), transformations=transformations
    )


def answers_equivalent(
    pred: Optional[str],
    gold: str,
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
) -> bool:
    """Is `pred` the same answer as `gold`? Never raises."""
    if pred is None:
        return False
    try:
        if answer_type in ("mc", "bool"):
            np_, ng = normalize_answer(pred, answer_type), normalize_answer(gold, answer_type)
            if np_ and np_ == ng:
                return True
            if answer_type == "mc" and choices:
                letter = _letter_from_choices(str(pred), choices)
                return bool(letter and letter == ng)
            return False

        np_, ng = normalize_answer(pred, answer_type), normalize_answer(gold, answer_type)
        if not np_ and not ng:
            return True
        if np_ == ng:
            return True

        for layer in (_numeric_equal, _set_equal, _symbolic_equal):
            verdict = layer(np_, ng)
            if verdict is True:
                return True
            if verdict is False and layer is _numeric_equal:
                # Both sides are numbers and they differ: no later layer can
                # make them equal, so stop before paying for sympy.
                return False
        return False
    except Exception:  # noqa: BLE001 - grading must never crash a run
        log.debug("equivalence check failed for %r vs %r", pred, gold, exc_info=True)
        return False


# ----------------------------------------------------------------- majority vote
def majority_vote(
    answers: Sequence[Optional[str]],
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Equivalence-aware plurality vote.

    Clustering by equivalence rather than by raw string is what makes the vote
    correct: "0.5", ".50" and "1/2" are one candidate, not three, and counting
    them separately would understate self-consistency.

    Ties break by (highest count, then first appearance) - deterministically,
    never randomly, so a rerun reproduces the same answer.
    """
    clusters: List[Dict[str, Any]] = []
    n_none = 0

    for raw in answers:
        if raw is None or not str(raw).strip():
            n_none += 1
            continue
        value = str(raw)
        placed = False
        for cluster in clusters:
            if answers_equivalent(value, cluster["key"], answer_type, choices):
                cluster["count"] += 1
                cluster["members"].append(value)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "key": normalize_answer(value, answer_type) or value,
                    "representative": value,
                    "count": 1,
                    "members": [value],
                }
            )

    n_valid = sum(c["count"] for c in clusters)
    info: Dict[str, Any] = {
        "vote_counts": {c["key"]: c["count"] for c in clusters},
        "clusters": {c["key"]: c["members"] for c in clusters},
        "n_valid": n_valid,
        "n_none": n_none,
        "n_total": len(answers),
        "top_count": 0,
        "tie": False,
        "n_candidates": len(clusters),
    }
    if not clusters:
        return None, info

    counts = sorted((c["count"] for c in clusters), reverse=True)
    top = counts[0]
    runner_up = counts[1] if len(counts) > 1 else 0
    leaders = [c for c in clusters if c["count"] == top]
    info["top_count"] = top
    info["second_count"] = runner_up
    # The top-1/top-2 frequency gap, normalised by the number of samples. This is
    # the "margin" the mechanism analysis is built on: a configuration change can
    # only move the mode where two classes hold comparable mass, so the paper's
    # prediction is that reorderings concentrate at small margin.
    info["margin"] = (top - runner_up) / len(answers) if answers else 0.0
    info["tie"] = len(leaders) > 1
    # `leaders` preserves insertion order, so [0] is the first-seen winner.
    return leaders[0]["representative"], info


# --------------------------------------------------------- canonical answer classes
#: The extraction-failure class. METHOD_SPEC section 4.3 writes it as the symbol
#: bottom; spelled out here because it has to survive a cp1252 Windows console, a
#: CSV round-trip and a LaTeX table without becoming mojibake.
EXTRACTION_FAILURE_CLASS = "<unparsed>"


def canonical_classes(
    answers: Sequence[Optional[str]],
    answer_type: str = "math",
    choices: Optional[Sequence[str]] = None,
) -> List[str]:
    """Label each answer with its equivalence-class key.

    Two answers get the same label iff the grader judges them equivalent, so the
    labels define the classes over which the mode, the modal ceiling and the
    mode-transition taxonomy are computed. Unparseable answers all land in
    `EXTRACTION_FAILURE_CLASS`, which is a real class and not a dropped sample:
    dropping them would silently raise the apparent modal-hit rate.

    Clustering is greedy in input order and therefore deterministic, and it is the
    same procedure `majority_vote` uses, so a class label and a vote can never
    disagree.
    """
    keys: List[str] = []
    representatives: List[str] = []
    for raw in answers:
        if raw is None or not str(raw).strip():
            keys.append(EXTRACTION_FAILURE_CLASS)
            continue
        value = str(raw)
        label: Optional[str] = None
        for representative in representatives:
            if answers_equivalent(value, representative, answer_type, choices):
                label = representative
                break
        if label is None:
            label = normalize_answer(value, answer_type) or value
            representatives.append(label)
        keys.append(label)
    return keys
