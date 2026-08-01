"""Mode transitions and modal ceilings (METHOD_SPEC 5.5).

The taxonomy is countable, so most of these tests assert exact counts on
hand-specified answer classes. The one statistical test is the mechanism claim:
reorderings must concentrate on small-margin items, and the check is that the
detector fires on data built to have that property and stays quiet on data built
without it.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.modes import (
    SMALL_MARGIN_THRESHOLD,
    TRANSITION_KINDS,
    ceiling_rows,
    ceiling_spread,
    classify_transition,
    reorder_rate_by_margin_bin,
    transition_records,
    transition_rows,
    transition_summary,
    union_ceiling,
)
from src.answers import EXTRACTION_FAILURE_CLASS
from tests.synthetic import CORRECT_CLASS as G
from tests.synthetic import make_cell, make_corpus

np = pytest.importorskip("numpy")


# ------------------------------------------------------------------- truth table
@pytest.mark.parametrize(
    "mode_a,ok_a,mode_b,ok_b,kind,reordered",
    [
        ("x", False, "x", False, "benign", False),
        (G, True, G, True, "benign", False),
        ("x", False, "y", False, "benign", True),      # changed, still wrong
        ("x", False, G, True, "corrective", True),
        (G, True, "x", False, "destructive", True),
    ],
)
def test_classify_transition_truth_table(mode_a, ok_a, mode_b, ok_b, kind, reordered):
    verdict = classify_transition(mode_a, ok_a, mode_b, ok_b)
    assert verdict["kind"] == kind
    assert verdict["reordered"] is reordered


def test_unparsed_mode_is_flagged():
    verdict = classify_transition(EXTRACTION_FAILURE_CLASS, False, G, True)
    assert verdict["unparsed_involved"] is True
    assert verdict["kind"] == "corrective"


# ------------------------------------------------------------------ exact counts
@pytest.fixture
def taxonomy_corpus():
    """Four items, one of each transition outcome, with controlled margins."""
    cells = [
        # corrective: wrong mode (3-2) becomes right (3-2)
        make_cell("i0", "m0", "c0", ["w1"] * 3 + [G] * 2),
        make_cell("i0", "m0", "c1", [G] * 3 + ["w1"] * 2),
        # stable and unanimous: margin 1.0, cannot reorder
        make_cell("i1", "m0", "c0", [G] * 5),
        make_cell("i1", "m0", "c1", [G] * 5),
        # destructive
        make_cell("i2", "m0", "c0", [G] * 3 + ["w2"] * 2),
        make_cell("i2", "m0", "c1", ["w2"] * 4 + [G]),
        # benign reorder between two wrong classes
        make_cell("i3", "m0", "c0", ["w1"] * 3 + ["w2"] * 2),
        make_cell("i3", "m0", "c1", ["w2"] * 3 + ["w1"] * 2),
    ]
    return make_corpus(cells)


def test_taxonomy_counts_are_exact(taxonomy_corpus):
    row = transition_rows(taxonomy_corpus, configs=["c0", "c1"], n_bootstrap=100)[0]
    assert row.n_items == 4
    assert row.reorder_rate == pytest.approx(0.75)
    assert (row.n_corrective, row.n_destructive, row.n_benign) == (1, 1, 2)
    assert row.net_corrective_rate == pytest.approx(0.0)


def test_taxonomy_kinds_sum_to_the_item_count(taxonomy_corpus):
    """A table of the three kinds must account for every item, reordered or not."""
    row = transition_rows(taxonomy_corpus, configs=["c0", "c1"], n_bootstrap=0)[0]
    counts = {"benign": row.n_benign, "corrective": row.n_corrective,
              "destructive": row.n_destructive}
    assert set(counts) == set(TRANSITION_KINDS)
    assert sum(counts.values()) == row.n_items


def test_unanimous_cells_never_reorder(taxonomy_corpus):
    records = transition_records(taxonomy_corpus, configs=["c0", "c1"])
    only = next(iter(records.values()))
    unanimous = [r for r in only if r["margin"] == 1.0]
    assert unanimous
    assert not any(r["reordered"] for r in unanimous)


def test_source_margin_is_the_covariate_not_the_minimum(taxonomy_corpus):
    """The predictor must come from the source configuration alone.

    Using a function of both configurations' margins would let the outcome leak into
    the predictor and make the concentration test circular.
    """
    records = next(iter(transition_records(taxonomy_corpus, configs=["c0", "c1"]).values()))
    by_item = {r["item_id"]: r for r in records}
    # i2's c0 margin is (3-2)/5 = 0.2; its c1 margin is (4-1)/5 = 0.6.
    assert by_item["i2"]["margin"] == pytest.approx(0.2)
    assert by_item["i2"]["margin_b"] == pytest.approx(0.6)


def test_margin_bins_are_monotone_on_the_taxonomy_fixture(taxonomy_corpus):
    records = next(iter(transition_records(taxonomy_corpus, configs=["c0", "c1"]).values()))
    bins = reorder_rate_by_margin_bin(records)
    assert sum(b["n_items"] for b in bins) == 4
    low = [b for b in bins if b["margin_high"] <= 0.25 and b["n_items"]]
    top = [b for b in bins if b["margin_high"] == 1.0 and b["n_items"]]
    assert low and top
    assert low[0]["reorder_rate"] > top[0]["reorder_rate"]


# ---------------------------------------------------------- ceilings and coverage
def test_pass_at_n_is_never_below_pi_mode():
    """A structural inequality: if the mode is right, some sample is right."""
    rng = np.random.default_rng(0)
    cells = []
    for i in range(60):
        labels = list(rng.choice([G, "w1", "w2"], size=16, p=[0.4, 0.35, 0.25]))
        cells.append(make_cell(f"i{i:02d}", "m0", "c0", labels))
    rows = ceiling_rows(make_corpus(cells), n_bootstrap=100)
    assert len(rows) == 1
    assert rows[0].pass_at_n >= rows[0].pi_mode - 1e-12
    assert rows[0].identifiability_gap >= -1e-12


def test_union_ceiling_is_at_least_the_best_single_configuration():
    rng = np.random.default_rng(1)
    cells = []
    for i in range(50):
        for config, p in (("c0", 0.4), ("c1", 0.5), ("c2", 0.45)):
            labels = list(rng.choice([G, "w1"], size=12, p=[p, 1 - p]))
            cells.append(make_cell(f"i{i:02d}", "m0", config, labels))
    result = union_ceiling(make_corpus(cells), "synth", None, "m0", ["c0", "c1", "c2"])
    assert result["union_ceiling"] >= result["best_single_config_pi_mode"]
    assert result["oracle_headroom_over_best_config"] >= 0.0
    assert result["n_items"] == 50


def test_unparsed_mode_counts_as_incorrect_and_is_reported():
    """A cell whose plurality is an unparseable answer must not flatter the ceiling."""
    cells = [
        make_cell("i0", "m0", "c0", [EXTRACTION_FAILURE_CLASS] * 4 + [G] * 2),
        make_cell("i1", "m0", "c0", [G] * 5 + ["w1"]),
    ]
    row = ceiling_rows(make_corpus(cells), n_bootstrap=0)[0]
    assert row.pi_mode == pytest.approx(0.5)
    assert row.unparsed_mode_rate == pytest.approx(0.5)
    assert row.extraction_failure_rate == pytest.approx(4 / 12)


def test_ceiling_spread_evaluates_p4():
    cells = []
    rng = np.random.default_rng(2)
    for i in range(80):
        for config, p in (("c0", 0.35), ("c1", 0.60)):
            labels = list(rng.choice([G, "w1"], size=12, p=[p, 1 - p]))
            cells.append(make_cell(f"i{i:02d}", "m0", config, labels))
    rows = ceiling_rows(make_corpus(cells), n_bootstrap=100)
    spread = ceiling_spread(rows)[0]
    assert spread["prediction"] == "P4"
    assert spread["pi_mode_spread_points"] > 3.0
    assert spread["supported"] is True
    assert spread["argmax_config"] == "c1"


# ----------------------------------------------------- the mechanism, statistically
def _margin_structured_corpus(reorder_only_small_margin: bool, seed: int):
    """Items across the margin range; reorderings injected by design.

    When `reorder_only_small_margin` is True, a configuration change flips the mode
    only where the top two classes are within one sample of each other. When False,
    flips are scattered uniformly over margins, which is the null the detector must
    stay quiet on.
    """
    rng = np.random.default_rng(seed)
    n = 24
    cells = []
    for i in range(200):
        top = int(rng.integers(13, 25))         # winner's count, 13..24
        second = n - top
        margin = (top - second) / n
        a_labels = [G] * top + ["w1"] * second
        should_flip = (
            margin <= SMALL_MARGIN_THRESHOLD
            if reorder_only_small_margin
            else bool(rng.random() < 0.35)
        )
        if should_flip:
            b_labels = ["w1"] * top + [G] * second
        else:
            b_labels = list(a_labels)
        cells.append(make_cell(f"i{i:03d}", "m0", "c0", a_labels))
        cells.append(make_cell(f"i{i:03d}", "m0", "c1", b_labels))
    return make_corpus(cells)


def test_mechanism_detected_when_reorderings_are_margin_concentrated():
    rows = transition_rows(
        _margin_structured_corpus(True, 5), configs=["c0", "c1"], n_bootstrap=200
    )
    row = rows[0]
    assert row.mean_margin_reordered < row.mean_margin_stable
    assert row.small_margin_reorder_rate > row.large_margin_reorder_rate
    assert row.margin_test["p_value"] < 0.01
    assert row.margin_test["effect_size_auc"] < 0.5
    assert "SMALLER" in row.margin_test["direction"]
    summary = transition_summary(rows)
    assert summary["mechanism_supported"] is True
    assert summary["n_pairs_significant"] == 1


def test_mechanism_not_claimed_when_reorderings_are_margin_independent():
    rows = transition_rows(
        _margin_structured_corpus(False, 6), configs=["c0", "c1"], n_bootstrap=200
    )
    row = rows[0]
    assert row.margin_test["p_value"] > 0.05
    assert row.margin_test["effect_size_auc"] == pytest.approx(0.5, abs=0.12)
    assert transition_summary(rows)["n_pairs_significant"] == 0


def test_reorder_quartile_rates_decrease_under_the_mechanism():
    rows = transition_rows(
        _margin_structured_corpus(True, 7), configs=["c0", "c1"], n_bootstrap=0
    )
    rates = [r for r in rows[0].reorder_rate_by_margin_quartile if math.isfinite(r)]
    assert len(rates) >= 3
    assert rates[0] > rates[-1]
