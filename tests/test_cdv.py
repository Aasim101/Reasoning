"""Configuration-diversified voting at matched token budget (METHOD_SPEC 5.6).

The claims under test, in order of how much damage a bug would do:

1. **No method exceeds its token budget, ever.** If one did, the headline comparison
   would be between methods spending different amounts of compute, which is the
   failure mode the matched-budget rule exists to prevent.
2. **Matching on tokens is not the same as matching on samples.** Asserted directly,
   because if the two coincided on this corpus the tests would not be checking the
   thing that makes the comparison honest.
3. **No free lunch.** With identical configurations, CDV must not beat
   self-consistency: any gain there would be an artefact of the re-partitioning.
4. **A gain when configurations really do reorder modes.** The intervention has to
   work on data where the mechanism is present.
"""

from __future__ import annotations

import pytest

from src.analysis.cdv import (
    DEFAULT_BUDGET_MULTIPLES,
    build_item_pools,
    build_methods,
    budget_curves,
    cdv_summary,
    compare_to_sc,
    evaluate_method,
    greedy_accuracy,
    token_budgets,
    transferred_difficulty_allocation,
)
from src.analysis.corpus import Corpus
from tests.synthetic import CORRECT_CLASS as G
from tests.synthetic import make_cell, make_corpus

np = pytest.importorskip("numpy")

CONFIGS = ("c0", "c1", "c2", "c3")


def _mixed_length_corpus(n_items=40, n=24, seed=0, reorder=True):
    """A corpus where configurations differ in both chain length and modal answer.

    `c0` and `c3` favour the wrong class, `c1` and `c4` the right one, and the chain
    lengths differ by more than 2x across configurations -- so token matching and
    sample matching genuinely disagree.
    """
    rng = np.random.default_rng(seed)
    lengths = {"c0": (80, 120), "c1": (180, 260), "c2": (100, 150), "c3": (90, 130)}
    if reorder:
        p_by_config = {"c0": 0.35, "c1": 0.62, "c2": 0.40, "c3": 0.58}
    else:
        p_by_config = {c: 0.45 for c in CONFIGS}
    cells = []
    for i in range(n_items):
        for config in CONFIGS:
            p = p_by_config[config]
            labels = [G if rng.random() < p else "w1" for _ in range(n)]
            low, high = lengths[config]
            cells.append(
                make_cell(
                    f"i{i:03d}", "m0", config, labels,
                    tokens=list(rng.integers(low, high, n)),
                )
            )
    return make_corpus(cells)


@pytest.fixture(scope="module")
def pools_and_methods():
    corpus = _mixed_length_corpus(seed=1)
    pools = build_item_pools(corpus, "synth", None, "m0")
    methods = build_methods(pools[0].configs())
    return corpus, pools, methods


# -------------------------------------------------------------- budget conservation
def test_no_method_ever_exceeds_its_token_budget(pools_and_methods):
    """The non-negotiable invariant. Checked at every budget for every method."""
    _corpus, pools, methods = pools_and_methods
    grid = token_budgets(pools, methods)
    for multiple, budget in zip(grid["multiples"], grid["budgets"]):
        for method in methods:
            point = evaluate_method(method, pools, budget, float(multiple), n_repeats=20)
            assert point.mean_tokens_used <= budget + 1e-6, (
                f"{method.name} at {multiple}x spent {point.mean_tokens_used:.1f} of "
                f"{budget:.1f} tokens"
            )


def test_every_method_faces_the_identical_budget(pools_and_methods):
    """One scalar conversion for all methods: a budget point is the same token count."""
    _corpus, pools, methods = pools_and_methods
    grid = token_budgets(pools, methods)
    budget = grid["budgets"][-1]
    points = [
        evaluate_method(m, pools, budget, 1.0, n_repeats=10) for m in methods
    ]
    assert len({p.budget_tokens for p in points}) == 1


def test_budget_scale_comes_from_the_reference_configuration(pools_and_methods):
    """Scaling off `c0` is what keeps the top budget point spendable by SC."""
    _corpus, pools, methods = pools_and_methods
    grid = token_budgets(pools, methods)
    assert grid["scale_configs"] == ["c0"]
    assert grid["multiples"] == list(DEFAULT_BUDGET_MULTIPLES), (
        "the full grid should survive when scaled off the reference configuration"
    )
    assert grid["dropped_budgets"] == []


def test_pool_exhaustion_is_reported_not_hidden(pools_and_methods):
    """At the top budget self-consistency runs out of chains, and must say so.

    The top budget is `N` times the mean chain cost, so roughly half the items -- the
    ones whose chains ran longer than average -- cannot fit all `N`. That is exactly
    why the rate is reported instead of assumed to be zero.
    """
    _corpus, pools, methods = pools_and_methods
    grid = token_budgets(pools, methods)
    sc = next(m for m in methods if m.name == "sc")
    top = evaluate_method(sc, pools, grid["budgets"][-1], 24.0, n_repeats=20)
    bottom = evaluate_method(sc, pools, grid["budgets"][0], 1.0, n_repeats=20)
    assert top.pool_exhausted_rate > 0.3
    assert bottom.pool_exhausted_rate < 0.05
    assert top.mean_chains_used > bottom.mean_chains_used


def test_token_matching_and_sample_matching_disagree(pools_and_methods):
    """The subtle cheat this design avoids, demonstrated on the actual corpus.

    At one shared token budget, self-consistency on the terse reference configuration
    buys visibly more chains than CDV does spreading over configurations that include
    a verbose one. Matching on sample count instead would have handed CDV extra
    tokens for free.
    """
    _corpus, pools, methods = pools_and_methods
    grid = token_budgets(pools, methods)
    budget = grid["budgets"][3]
    sc = evaluate_method(
        next(m for m in methods if m.name == "sc"), pools, budget, 8.0, n_repeats=40
    )
    cdv = evaluate_method(
        next(m for m in methods if m.name == "cdv"), pools, budget, 8.0, n_repeats=40
    )
    assert sc.mean_chains_used > cdv.mean_chains_used * 1.05
    # ...while the token spend is closely matched, which is the point.
    assert abs(sc.mean_tokens_used - cdv.mean_tokens_used) / budget < 0.15


def test_repartitioning_never_invents_chains(pools_and_methods):
    """Every drawn chain must come from the corpus: no method may exceed the pool."""
    _corpus, pools, methods = pools_and_methods
    huge = 10 ** 9
    for method in methods:
        point = evaluate_method(method, pools, huge, 999.0, n_repeats=5)
        available = sum(
            pools[0].n_available(c) for c in method.configs if c in pools[0].classes
        )
        assert point.mean_chains_used <= available + 1e-9
        capacity = float(np.mean([p.capacity_tokens(method.configs) for p in pools]))
        assert point.mean_tokens_used <= capacity + 1e-6


# ------------------------------------------------------------------- no free lunch
def test_cdv_does_not_beat_sc_when_configurations_are_identical():
    """Identical configurations carry no diversity, so there is nothing to exploit."""
    corpus = _mixed_length_corpus(n_items=60, seed=4, reorder=False)
    points, _warnings = budget_curves(corpus, multiples=(4, 8), n_repeats=40)
    comparisons = compare_to_sc(points, n_bootstrap=300)
    cdv_rows = [c for c in comparisons if c["method"] == "cdv"]
    assert cdv_rows
    for row in cdv_rows:
        assert row["delta"] < 0.05, (
            "CDV gained on configurations with no modal difference, which would mean "
            "the re-partitioning itself is producing the effect"
        )
        assert not row["significant_holm"]


def test_cdv_gains_when_configurations_reorder_modes():
    corpus = _mixed_length_corpus(n_items=80, n=16, seed=5, reorder=True)
    points, _warnings = budget_curves(corpus, multiples=(4, 8, 16), n_repeats=40)
    comparisons = compare_to_sc(points, n_bootstrap=300)
    summary = cdv_summary(points, comparisons)
    assert summary["cdv_beats_sc"] is True
    assert summary["cdv_minus_sc_at_largest"] > 0.02
    assert "falsification_hook" in summary


def test_oracle_configuration_selection_bounds_cdv():
    """B11 is an upper bound for any configuration-selection policy, CDV included."""
    corpus = _mixed_length_corpus(n_items=60, seed=6, reorder=True)
    points, _warnings = budget_curves(corpus, multiples=(8,), n_repeats=40)
    by_method = {p.method: p.accuracy for p in points}
    assert by_method["oracle_config"] >= by_method["cdv"] - 1e-9
    assert by_method["oracle_config"] >= by_method["sc"] - 1e-9


def test_adaptive_cdv_spends_less_when_configurations_agree():
    """The whole point of the two-stage version: agreement should cost less."""
    agree = _mixed_length_corpus(n_items=40, seed=7, reorder=False)
    pools = build_item_pools(agree, "synth", None, "m0")
    methods = build_methods(pools[0].configs())
    grid = token_budgets(pools, methods)
    budget = grid["budgets"][-1]
    cdv = evaluate_method(
        next(m for m in methods if m.name == "cdv"), pools, budget, 24.0, n_repeats=40
    )
    adaptive = evaluate_method(
        next(m for m in methods if m.name == "adaptive_cdv"), pools, budget, 24.0,
        n_repeats=40,
    )
    assert adaptive.mean_tokens_used < cdv.mean_tokens_used


def test_comparison_reports_the_token_match_error(pools_and_methods):
    """A reviewer must be able to check the matching, so it is a reported column."""
    corpus, _pools, _methods = pools_and_methods
    points, _warnings = budget_curves(corpus, multiples=(8,), n_repeats=20)
    comparisons = compare_to_sc(points, n_bootstrap=200)
    assert comparisons
    for row in comparisons:
        assert "token_match_error" in row
        assert row["family_size"] == len(
            [c for c in comparisons if c["budget_multiple"] == row["budget_multiple"]]
        )
        assert 0.0 <= row["p_holm"] <= 1.0


def test_missing_token_counts_are_flagged_loudly():
    """Budget matching silently degrades to sample matching without token counts.

    `make_cell` with zero tokens exercises the same path as a corpus graded from
    records that never carried `sample_stats`.
    """
    cells = [
        make_cell(f"i{i}", "m0", config, [G] * 4 + ["w1"] * 4, tokens=0)
        for i in range(20)
        for config in CONFIGS
    ]
    _points, warnings = budget_curves(make_corpus(cells), multiples=(4,), n_repeats=5)
    assert any("no completion-token counts" in w for w in warnings)


# ---------------------------------------------------------------- allocation and B2
def test_transferred_allocation_reports_both_arms_and_validity():
    corpus = _mixed_length_corpus(n_items=60, n=24, seed=8, reorder=True)
    result = transferred_difficulty_allocation(
        corpus, "synth", None, "m0", target_config="c1", n_repeats=30
    )
    for key in (
        "accuracy_uniform", "accuracy_transferred", "accuracy_aware",
        "transfer_degradation", "hard_set_jaccard_source_vs_target", "valid",
    ):
        assert key in result
    assert 0.0 <= result["hard_set_jaccard_source_vs_target"] <= 1.0
    # The three arms must spend the same tokens; that is the matched-budget rule.
    spends = [result[f"tokens_{a}"] for a in ("uniform", "transferred", "aware")]
    assert max(spends) - min(spends) < 0.05 * max(spends)


def test_allocation_rejects_an_unknown_rule():
    corpus = _mixed_length_corpus(n_items=20, seed=9)
    with pytest.raises(ValueError, match="uncertainty"):
        transferred_difficulty_allocation(
            corpus, "synth", None, "m0", rule="vibes", n_repeats=2
        )


def test_seed_pooled_matches_sc_when_only_c0_differs():
    """Identical components at c0: seed-pooled voting must equal SC."""
    corpus = _mixed_length_corpus(n_items=40, seed=10, reorder=False)
    pools = build_item_pools(corpus, "synth", None, "m0")
    methods = build_methods(pools[0].configs())
    grid = token_budgets(pools, methods)
    budget = grid["budgets"][2]
    sc = evaluate_method(
        next(m for m in methods if m.name == "sc"), pools, budget, 8.0, n_repeats=30
    )
    pooled = evaluate_method(
        next(m for m in methods if m.name == "seed_pooled_sc"),
        pools,
        budget,
        8.0,
        n_repeats=30,
    )
    assert pooled.accuracy == pytest.approx(sc.accuracy, abs=0.02)
    assert pooled.mean_tokens_used == pytest.approx(sc.mean_tokens_used, rel=0.05)


def test_cdv_gain_decomposition_reports_modal_mass_split():
    from src.analysis.cdv import cdv_gain_decomposition

    corpus = _mixed_length_corpus(n_items=50, seed=11, reorder=True)
    row = cdv_gain_decomposition(corpus, "synth", None, "m0")
    assert row["n_items"] == 50
    assert row["new_modal_mass"] + row["configuration_selection"] == row["cdv_wins_over_sc"]
    assert 0.0 <= row["fraction_new_modal_mass"] <= 1.0
    assert "pi_mode_sc" in row and "pi_mode_mixture" in row


def test_greedy_pass_is_read_from_the_auxiliary_arm():
    """B2 lives in `Corpus.aux` so its N=1 cells cannot pollute the crossed design."""
    corpus = Corpus()
    for i in range(10):
        cell = make_cell(f"i{i}", "m0", "c0", [G] if i % 2 == 0 else ["w1"], tokens=50)
        cell.strategy = "greedy_pass"
        corpus.aux[cell.aux_key()] = cell
    rows = greedy_accuracy(corpus)
    assert len(rows) == 1
    assert rows[0]["baseline"] == "B2"
    assert rows[0]["greedy_accuracy"] == pytest.approx(0.5)
    assert rows[0]["n_items"] == 10
    assert len(corpus.cells) == 0
