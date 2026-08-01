"""Answer extraction and equivalence tests.

The non-equivalence cases matter as much as the equivalence ones: a grader that
says yes to everything looks great in aggregate and is worthless.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from src.answers import (
    GRADER_VERSION,
    answers_equivalent,
    extract_answer,
    extract_boxed,
    grader_backend_name,
    majority_vote,
    normalize_answer,
)


# ------------------------------------------------------------------- extraction
@pytest.mark.parametrize(
    "text,expected",
    [
        (r"the result is \boxed{42}", "42"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"nested \boxed{\frac{\sqrt{2}}{2}} here", r"\frac{\sqrt{2}}{2}"),
        (r"first \boxed{1} then \boxed{2}", "2"),
        (r"$\boxed{7}$", "7"),
        (r"\boxed 99", "99"),
        (r"\fbox{13}", "13"),
        ("no box here", None),
    ],
)
def test_extract_boxed(text: str, expected: Optional[str]):
    assert extract_boxed(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is 42.", "42"),
        ("the answer is 42", "42"),
        ("**Answer:** 17", "17"),
        ("Final answer: -3", "-3"),
        ("answer = 8", "8"),
        ("First the answer is 1. Later the answer is 2.", "2"),
    ],
)
def test_extract_answer_phrases(text: str, expected: str):
    assert extract_answer(text, "math") == expected


def test_extract_gsm8k_hash_marker():
    assert extract_answer("Some working out\n#### 18", "math") == "18"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("So we get 12 apples and then 5 oranges", "5"),
        ("the total is 1,234 units", "1,234"),
        ("that costs $19.99 in the end", "19.99"),
        ("about -7 degrees", "-7"),
        ("roughly 1.5e3 joules", "1.5e3"),
    ],
)
def test_extract_last_number_fallback(text: str, expected: str):
    assert extract_answer(text, "math") == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "no numbers and no answer at all",
    ],
)
def test_extract_returns_none_on_garbage(text: str):
    assert extract_answer(text, "math") is None


CHOICES = ["Venus", "Mercury", "Mars", "Earth"]


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"\boxed{B}", "B"),
        ("Answer: C", "C"),
        ("The answer is (D)", "D"),
        ("I pick option B here", "B"),
        ("**Answer:** A", "A"),
        ("After thinking, the answer is Mercury", "B"),
        ("clearly Mars is right, so the answer is Mars", "C"),
        ("nothing relevant", None),
    ],
)
def test_extract_multiple_choice(text: str, expected: Optional[str]):
    assert extract_answer(text, "mc", CHOICES) == expected


def test_mc_letters_beyond_the_choice_count_are_not_invented():
    # Only 4 options exist, so "E" must not be returned as a valid letter.
    assert extract_answer("Answer: E", "mc", CHOICES) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Yes, definitely.", "True"),
        ("No, that is wrong.", "False"),
        (r"\boxed{True}", "True"),
        ("The answer is false", "False"),
    ],
)
def test_extract_bool(text: str, expected: str):
    assert extract_answer(text, "bool") == expected


def test_extraction_never_raises_on_weird_input():
    for text in (r"\boxed{", r"\boxed{{{", "\x00\x01", r"\frac{}{}", "$" * 500):
        extract_answer(text, "math")
        extract_answer(text, "mc", CHOICES)
        extract_answer(text, "bool")


# ---------------------------------------------------------------- normalisation
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2.50", "2.5"),
        (".50", "0.5"),
        ("+5", "5"),
        ("-0", "0"),
        ("1,234", "1234"),
        ("5 cm", "5"),
        ("12 apples", "12"),
        ("50%", "50"),
        (r"$42$", "42"),
        ("42.", "42"),
        (r"\frac{1}{2}", "0.5"),
        ("1/2", "0.5"),
    ],
)
def test_normalize_math(raw: str, expected: str):
    assert normalize_answer(raw, "math") == expected


def test_normalize_mc_and_bool():
    assert normalize_answer("b", "mc") == "B"
    assert normalize_answer("(C)", "mc") == "C"
    assert normalize_answer("yes", "bool") == "True"
    assert normalize_answer("NO", "bool") == "False"


# ------------------------------------------------------------------ equivalence
@pytest.mark.parametrize(
    "pred,gold",
    [
        ("42", "42"),
        ("1/2", "0.5"),
        (".50", "0.5"),
        ("2.50", "2.5"),
        ("5 cm", "5"),
        ("12 apples", "12"),
        ("-0", "0"),
        ("+5", "5"),
        ("50%", "50"),
        ("1,000", "1000"),
        (r"\frac{3}{4}", "0.75"),
        (r"\dfrac{3}{4}", "0.75"),
        ("2 1/2", "2.5"),
        (r"\sqrt{4}", "2"),
        ("1, 2", "2, 1"),
        ("(1,2)", "(2,1)"),
        (r"\text{5}", "5"),
        ("x+1", "1+x"),
    ],
)
def test_equivalent(pred: str, gold: str):
    assert answers_equivalent(pred, gold, "math") is True


@pytest.mark.parametrize(
    "pred,gold",
    [
        ("3", "4"),
        ("1/2", "1/3"),
        ("2", "-2"),
        ("0.333", "1/3"),
        (r"\sqrt{2}", "1.41"),
        ("1, 2", "1, 3"),
        ("1, 2", "1, 2, 3"),
        ("2x", "2"),
        ("42", "43"),
        ("", "5"),
    ],
)
def test_not_equivalent(pred: str, gold: str):
    assert answers_equivalent(pred, gold, "math") is False


def test_none_prediction_is_never_correct():
    assert answers_equivalent(None, "42", "math") is False
    assert answers_equivalent(None, "A", "mc", CHOICES) is False


def test_mc_equivalence():
    assert answers_equivalent("b", "B", "mc", CHOICES) is True
    assert answers_equivalent("A", "B", "mc", CHOICES) is False
    # Option text is accepted where the gold is a letter.
    assert answers_equivalent("Mercury", "B", "mc", CHOICES) is True


def test_bool_equivalence():
    assert answers_equivalent("yes", "True", "bool") is True
    assert answers_equivalent("no", "True", "bool") is False


def test_grader_backend_is_reported():
    assert grader_backend_name() in {"math_verify", "sympy", "regex"}
    assert GRADER_VERSION


# ---------------------------------------------------------------- majority vote
def test_majority_vote_clusters_equivalent_answers():
    winner, info = majority_vote(["0.5", "1/2", ".50", "3", "3"], "math")
    # Three equivalent forms of one half beat two threes.
    assert answers_equivalent(winner, "0.5", "math")
    assert info["top_count"] == 3
    assert info["n_candidates"] == 2
    assert info["tie"] is False
    assert info["n_valid"] == 5
    assert info["n_none"] == 0


def test_majority_vote_ignores_none_but_counts_it():
    winner, info = majority_vote(["7", None, "7", None], "math")
    assert winner == "7"
    assert info["n_none"] == 2
    assert info["n_valid"] == 2
    assert info["n_total"] == 4


def test_majority_vote_all_none():
    winner, info = majority_vote([None, None], "math")
    assert winner is None
    assert info["n_valid"] == 0
    assert info["vote_counts"] == {}


def test_majority_vote_empty():
    winner, info = majority_vote([], "math")
    assert winner is None
    assert info["top_count"] == 0


def test_majority_vote_tie_breaks_deterministically_by_first_seen():
    for _ in range(5):
        winner, info = majority_vote(["5", "9"], "math")
        assert winner == "5"
        assert info["tie"] is True


def test_majority_vote_mc():
    winner, info = majority_vote(["A", "B", "b", None], "mc", CHOICES)
    assert winner in {"B", "b"}
    assert info["top_count"] == 2
