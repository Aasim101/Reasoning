"""Metric and statistics tests.

`pass_at_k` is checked against brute-force enumeration of every size-k subset,
and McNemar against hand-computed exact binomial p-values. These are the numbers
a reviewer will recompute, so they are pinned to ground truth rather than to
whatever the implementation happens to return.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from src.metrics import (
    accuracy,
    accuracy_vs_compute,
    aggregate_seeds,
    bootstrap_ci,
    compare_strategies,
    holm_bonferroni,
    majority_accuracy_curve,
    majority_vote_at_k,
    mcnemar,
    paired_bootstrap,
    pass_at_k,
    pass_at_k_curve,
    pass_at_k_records,
    summarize_run,
    token_cost,
    tokens_per_correct,
    wilson_ci,
)


# ------------------------------------------------------------------- fixtures
def make_record(
    index: int,
    is_correct: bool,
    n_samples: int = 4,
    n_correct_samples: int = 2,
    tokens_completion: int = 100,
    strategy: str = "cot_zeroshot",
    seed: int = 0,
    dataset: str = "toy",
    answer_type: str = "math",
) -> Dict[str, Any]:
    """A synthetic graded record, so these tests need no other module."""
    sample_correct = [i < n_correct_samples for i in range(n_samples)]
    sample_answers: List[Optional[str]] = [
        "42" if ok else str(100 + i) for i, ok in enumerate(sample_correct)
    ]
    return {
        "uid": f"uid{index}-{strategy}",
        "example_id": f"{dataset}/test/{index}",
        "index": index,
        "dataset": dataset,
        "model": "mock/tiny",
        "strategy": strategy,
        "seed": seed,
        "config_hash": "abc123",
        "gold_answer": "42",
        "answer_type": answer_type,
        "final_answer": "42" if is_correct else "999",
        "reasoning_traces": ["t"] * n_samples,
        "n_samples": n_samples,
        "n_samples_graded": n_samples,
        "n_correct_samples": n_correct_samples,
        "sample_answers": sample_answers,
        "sample_correct": sample_correct,
        "is_correct": is_correct,
        "vote_correct": is_correct,
        "vote_answer": "42" if is_correct else "999",
        "pred_answer": "42" if is_correct else "999",
        "tokens_prompt": 20,
        "tokens_completion": tokens_completion,
        "n_calls": 1,
        "latency_s": 0.5,
        "error": None,
        "grader_version": "1.0.0",
        "grader_backend": "sympy",
    }


# --------------------------------------------------------------------- pass@k
def brute_force_pass_at_k(n: int, c: int, k: int) -> float:
    """Exact fraction of size-k subsets containing at least one of c successes."""
    items = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    hits = sum(1 for subset in subsets if any(items[i] for i in subset))
    return hits / len(subsets)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_pass_at_k_matches_brute_force(n: int):
    for c in range(0, n + 1):
        for k in range(1, n + 1):
            expected = brute_force_pass_at_k(n, c, k)
            assert pass_at_k(n, c, k) == pytest.approx(expected, abs=1e-12), (
                f"n={n} c={c} k={k}"
            )


def test_pass_at_k_edge_cases():
    assert pass_at_k(10, 0, 3) == 0.0
    assert pass_at_k(10, 10, 3) == 1.0
    assert pass_at_k(8, 2, 1) == pytest.approx(2 / 8)
    assert pass_at_k(8, 3, 8) == 1.0
    # Closed form check against the binomial coefficient definition.
    assert pass_at_k(10, 4, 3) == pytest.approx(
        1 - math.comb(6, 3) / math.comb(10, 3)
    )


@pytest.mark.parametrize(
    "n,c,k", [(0, 0, 1), (5, 6, 1), (5, -1, 1), (5, 1, 0), (5, 1, 6)]
)
def test_pass_at_k_rejects_invalid_input(n: int, c: int, k: int):
    with pytest.raises(ValueError):
        pass_at_k(n, c, k)


def test_pass_at_k_records_skips_short_records():
    records = [
        make_record(0, True, n_samples=4, n_correct_samples=2),
        make_record(1, True, n_samples=1, n_correct_samples=1),
    ]
    # k=4 is only computable for the first record.
    assert pass_at_k_records(records, 4) == pytest.approx(pass_at_k(4, 2, 4))
    assert math.isnan(pass_at_k_records(records, 8))


def test_pass_at_k_curve_reports_eligible_counts():
    records = [make_record(i, True, n_samples=4, n_correct_samples=1) for i in range(3)]
    curve = pass_at_k_curve(records, [1, 2, 4, 8])
    assert curve["ks"] == [1, 2, 4]
    assert curve["n_records"] == [3, 3, 3]
    assert curve["values"][0] <= curve["values"][-1], "pass@k is nondecreasing in k"


# ---------------------------------------------------------------------- CIs
def test_wilson_ci_contains_estimate_and_narrows_with_n():
    low, high = wilson_ci(5, 10)
    assert low < 0.5 < high
    wide = wilson_ci(5, 10)
    narrow = wilson_ci(500, 1000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_ci_degenerate():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    low, high = wilson_ci(10, 10)
    assert high == pytest.approx(1.0) and low < 1.0
    low, high = wilson_ci(0, 10)
    assert low == pytest.approx(0.0) and high > 0.0


def test_bootstrap_ci_reproducible_and_contains_point():
    values = [1.0] * 30 + [0.0] * 20
    low, high, point = bootstrap_ci(values, n_bootstrap=2000, seed=42)
    assert point == pytest.approx(0.6)
    assert low < point < high
    assert bootstrap_ci(values, n_bootstrap=2000, seed=42) == (low, high, point)
    assert bootstrap_ci(values, n_bootstrap=2000, seed=7) != (low, high, point)


def test_bootstrap_ci_degenerate():
    assert bootstrap_ci([], n_bootstrap=100) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([1.0], n_bootstrap=100) == (1.0, 1.0, 1.0)
    low, high, point = bootstrap_ci([1.0] * 20, n_bootstrap=500, seed=0)
    assert (low, high, point) == (1.0, 1.0, 1.0)


# ------------------------------------------------------------------- McNemar
def test_mcnemar_exact_hand_computed():
    """b01=1, b10=8: two-sided exact binomial p = 2 * sum_{i<=1} C(9,i) / 2^9."""
    a = [True] * 8 + [False] * 1 + [True] * 5
    b = [False] * 8 + [True] * 1 + [True] * 5
    result = mcnemar(a, b, exact=True)
    assert result["b10"] == 8
    assert result["b01"] == 1
    expected = 2 * (math.comb(9, 0) + math.comb(9, 1)) / 2**9
    assert result["p_value"] == pytest.approx(expected)
    assert result["method"] == "exact_binomial"


def test_mcnemar_is_symmetric_in_p():
    a = [True, True, False, False, True, False, True, True]
    b = [False, True, True, False, True, True, False, True]
    assert mcnemar(a, b)["p_value"] == pytest.approx(mcnemar(b, a)["p_value"])


def test_mcnemar_no_difference_gives_p_one():
    a = [True, False, True, False]
    assert mcnemar(a, list(a))["p_value"] == 1.0
    assert mcnemar(a, list(a))["n_discordant"] == 0
    # Balanced discordance is also p = 1.
    balanced = mcnemar([True, False], [False, True])
    assert balanced["b01"] == 1 and balanced["b10"] == 1
    assert balanced["p_value"] == pytest.approx(1.0)


def test_mcnemar_chi2_branch_hand_computed():
    # b01 = 30, b10 = 10: chi2 = (|30-10|-1)^2 / 40 = 361/40 = 9.025
    a = [False] * 30 + [True] * 10
    b = [True] * 30 + [False] * 10
    result = mcnemar(a, b, exact=False, correction=True)
    assert result["statistic"] == pytest.approx(361 / 40)
    assert result["p_value"] == pytest.approx(math.erfc(math.sqrt((361 / 40) / 2)))
    assert result["p_value"] < 0.01
    assert "chi2" in result["method"]
    uncorrected = mcnemar(a, b, exact=False, correction=False)
    assert uncorrected["statistic"] == pytest.approx(400 / 40)


def test_mcnemar_falls_back_to_exact_for_small_samples():
    a = [True, False, False]
    b = [False, True, False]
    assert mcnemar(a, b, exact=False)["method"] == "exact_binomial"


def test_mcnemar_length_mismatch_raises():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# ----------------------------------------------------------- paired bootstrap
def test_paired_bootstrap_identical_inputs():
    flags = [1, 0, 1, 1, 0] * 6
    result = paired_bootstrap(flags, list(flags), n_bootstrap=1000, seed=0)
    assert result["diff"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["ci_low"] == result["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_detects_a_large_difference():
    a = [0] * 40
    b = [1] * 40
    result = paired_bootstrap(a, b, n_bootstrap=2000, seed=0)
    assert result["diff"] == pytest.approx(1.0)
    assert result["p_value"] <= 2.0 / 2000 + 1e-12
    assert result["ci_low"] > 0.5


def test_paired_bootstrap_is_reproducible_and_records_provenance():
    a = [1, 0, 1, 0, 1, 1, 0, 0]
    b = [1, 1, 1, 0, 1, 0, 1, 0]
    first = paired_bootstrap(a, b, n_bootstrap=500, seed=3)
    second = paired_bootstrap(a, b, n_bootstrap=500, seed=3)
    assert first == second
    assert first["seed"] == 3 and first["n_bootstrap"] == 500
    assert paired_bootstrap(a, b, n_bootstrap=500, seed=4) != first


def test_paired_bootstrap_p_value_floor():
    result = paired_bootstrap([0] * 20, [1] * 20, n_bootstrap=100, seed=0)
    assert result["p_value"] >= 1.0 / 100


def test_paired_bootstrap_empty_and_mismatch():
    assert paired_bootstrap([], [], n_bootstrap=10)["n_pairs"] == 0
    with pytest.raises(ValueError):
        paired_bootstrap([1], [1, 0])


# --------------------------------------------------------- compare_strategies
def test_compare_strategies_pairs_on_shuffled_order():
    import random

    a = [make_record(i, i % 2 == 0, strategy="base") for i in range(20)]
    b = [make_record(i, i % 3 != 0, strategy="new") for i in range(20)]
    random.Random(0).shuffle(b)

    result = compare_strategies(a, b, n_bootstrap=500, label_a="base", label_b="new")
    assert result["n_paired"] == 20
    assert result["n_only_a"] == 0 and result["n_only_b"] == 0
    assert result["accuracy_a"] == pytest.approx(0.5)
    assert result["accuracy_b"] == pytest.approx(sum(1 for i in range(20) if i % 3) / 20)
    assert result["delta"] == pytest.approx(result["accuracy_b"] - result["accuracy_a"])
    assert result["mcnemar"]["n_pairs"] == 20
    assert result["paired_bootstrap"]["n_pairs"] == 20


def test_compare_strategies_refuses_disjoint_sets():
    a = [make_record(i, True, strategy="base") for i in range(5)]
    b = [make_record(i, True, strategy="new") for i in range(100, 105)]
    result = compare_strategies(a, b)
    assert result["n_paired"] == 0
    assert result["n_only_a"] == 5 and result["n_only_b"] == 5
    assert "mcnemar" not in result
    with pytest.raises(ValueError):
        compare_strategies(a, b, strict=True)


def test_compare_strategies_reports_partial_overlap():
    a = [make_record(i, True, strategy="base") for i in range(10)]
    b = [make_record(i, True, strategy="new") for i in range(5, 15)]
    result = compare_strategies(a, b, n_bootstrap=100)
    assert result["n_paired"] == 5
    assert result["n_only_a"] == 5
    assert result["n_only_b"] == 5


# ----------------------------------------------------------------- basic stats
def test_accuracy_and_empty_input():
    assert accuracy([]) == 0.0
    records = [make_record(0, True), make_record(1, False)]
    assert accuracy(records) == pytest.approx(0.5)


def test_errored_records_count_as_incorrect():
    bad = make_record(0, False)
    bad["error"] = "RuntimeError: boom"
    assert accuracy([bad]) == 0.0
    assert summarize_run([bad])["n_errors"] == 1


def test_token_cost_and_tokens_per_correct():
    records = [
        make_record(0, True, tokens_completion=100),
        make_record(1, False, tokens_completion=300),
    ]
    costs = token_cost(records)
    assert costs["total_completion"] == 400
    assert costs["mean_completion"] == pytest.approx(200.0)
    assert costs["total_prompt"] == 40
    assert tokens_per_correct(records) == pytest.approx(400.0)

    assert token_cost([])["n"] == 0
    assert tokens_per_correct([]) == float("inf")
    assert tokens_per_correct([make_record(0, False)]) == float("inf")


def test_accuracy_vs_compute():
    out = accuracy_vs_compute(
        {
            "a": [make_record(0, True, tokens_completion=100)],
            "b": [make_record(0, False, tokens_completion=400)],
            "empty": [],
        }
    )
    assert set(out["labels"]) == {"a", "b"}
    assert out["y"][out["labels"].index("a")] == 1.0
    assert out["x"][out["labels"].index("b")] == pytest.approx(400.0)


# ---------------------------------------------------------- majority vote @ k
def test_majority_vote_at_k_basic():
    answers = ["42", "42", "7"]
    correct = [True, True, False]
    assert majority_vote_at_k(correct, answers, 3) is True
    assert majority_vote_at_k(correct, answers, 5) is None, "too few samples"
    # Only the last two: one "42" and one "7" ties, first-seen wins -> correct.
    assert majority_vote_at_k(correct, answers, 2, indices=[1, 2]) is True


def test_majority_vote_at_k_all_none():
    assert majority_vote_at_k([False, False], [None, None], 2) is None


def test_majority_accuracy_curve_is_bounded_and_reproducible():
    records = [
        make_record(i, True, n_samples=8, n_correct_samples=6 if i % 2 == 0 else 2)
        for i in range(12)
    ]
    curve = majority_accuracy_curve(records, [1, 2, 4, 8], n_bootstrap=200, seed=0)
    assert curve["ks"] == [1, 2, 4, 8]
    assert all(0.0 <= v <= 1.0 for v in curve["values"])
    assert curve["seed"] == 0 and curve["n_bootstrap"] == 200
    assert all(low <= value <= high for low, value, high
               in zip(curve["ci_low"], curve["values"], curve["ci_high"]))
    again = majority_accuracy_curve(records, [1, 2, 4, 8], n_bootstrap=200, seed=0)
    assert again["values"] == curve["values"]


def test_majority_accuracy_curve_rises_when_samples_mostly_correct():
    records = [make_record(i, True, n_samples=8, n_correct_samples=6) for i in range(10)]
    curve = majority_accuracy_curve(records, [1, 8], seed=0)
    assert curve["values"][-1] >= curve["values"][0]


def test_majority_accuracy_curve_empty():
    assert majority_accuracy_curve([], [1, 2])["ks"] == []


# ------------------------------------------------------------------ summaries
def test_summarize_run():
    records = [make_record(i, i < 6, n_samples=4, n_correct_samples=2) for i in range(10)]
    summary = summarize_run(records)
    assert summary["n"] == 10
    assert summary["n_correct"] == 6
    assert summary["accuracy"] == pytest.approx(0.6)
    assert summary["ci_low"] < 0.6 < summary["ci_high"]
    assert summary["ci_method"] == "wilson"
    assert summary["dataset"] == "toy"
    assert summary["max_samples"] == 4
    assert summary["tokens_mean_completion"] == pytest.approx(100.0)
    assert "pass_at_1" in summary and "pass_at_max" in summary
    assert summarize_run([])["n"] == 0


def test_aggregate_seeds():
    summaries = [
        summarize_run([make_record(i, i < k, seed=s) for i in range(10)])
        for s, k in enumerate([5, 6, 7])
    ]
    agg = aggregate_seeds(summaries)
    assert agg["n_seeds"] == 3
    assert agg["accuracy_mean"] == pytest.approx(0.6)
    assert agg["accuracy_std"] > 0
    assert agg["accuracy_min"] == pytest.approx(0.5)
    assert agg["accuracy_max"] == pytest.approx(0.7)
    assert agg["accuracy_ci_low"] <= agg["accuracy_mean"] <= agg["accuracy_ci_high"]
    assert aggregate_seeds([])["n_seeds"] == 0


def test_holm_bonferroni_is_monotone_and_conservative():
    result = holm_bonferroni([0.001, 0.02, 0.04], alpha=0.05)
    assert result["family_size"] == 3
    adjusted = result["p_adjusted"]
    assert adjusted[0] >= 0.001 * 3 - 1e-12
    assert adjusted == sorted(adjusted), "step-down values must be nondecreasing"
    assert all(p <= 1.0 for p in adjusted)


# --------------------------------------------------- figures and tables smoke
def test_figures_and_tables_are_produced(tmp_path: Path):
    """Catches rcParam and font errors on machines with no LaTeX fonts."""
    from src.analysis import figures, tables

    summaries = []
    for si, strategy in enumerate(("cot_zeroshot", "self_consistency")):
        records = [
            make_record(i, i < 5 + si, n_samples=4, n_correct_samples=2, strategy=strategy)
            for i in range(10)
        ]
        summary = summarize_run(records)
        summary["majority_curve"] = majority_accuracy_curve(records, [1, 2, 4], n_bootstrap=100)
        summary["pass_curve"] = pass_at_k_curve(records, [1, 2, 4])
        summary["elapsed_seconds"] = 3600.0
        summary["gpu_hours"] = 1.0
        summaries.append(summary)

    curves = {s["strategy"]: s["majority_curve"] for s in summaries}
    pass_curves = {s["strategy"]: s["pass_curve"] for s in summaries}
    written = figures.write_all(summaries, tmp_path, curves, pass_curves)
    assert written
    for path in written:
        assert path.suffix == ".pdf"
        assert path.stat().st_size > 1000, f"{path} looks empty"

    comparisons = [
        compare_strategies(
            [make_record(i, i < 5, strategy="cot_zeroshot") for i in range(10)],
            [make_record(i, i < 8, strategy="self_consistency") for i in range(10)],
            n_bootstrap=200,
            label_a="cot_zeroshot",
            label_b="self_consistency",
        )
    ]
    comparisons[0]["dataset"] = "toy"
    tex_paths = tables.write_all(summaries, comparisons, tmp_path, baseline="cot_zeroshot")
    assert tex_paths
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        assert path.suffix == ".tex"
        assert r"\toprule" in text
        assert r"\bottomrule" in text
        assert r"\begin{table}" in text and r"\end{table}" in text
        assert r"booktabs" in text


def test_latex_escaping():
    from src.analysis.tables import escape

    assert escape("Qwen2.5_7B") == r"Qwen2.5\_7B"
    assert escape("50%") == r"50\%"
    assert escape("a&b") == r"a\&b"
    assert escape("#1") == r"\#1"
