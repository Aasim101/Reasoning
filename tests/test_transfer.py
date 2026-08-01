"""Difficulty transfer and hard-subset overlap: arithmetic, then null calibration.

Two things have to be right for section 5.3-5.4 to mean anything.

**The disattenuation must recover a known latent correlation.** The tests below
inject two configurations whose *latent* difficulty vectors have a known Spearman
correlation, add binomial sampling noise, and check that the raw correlation is
attenuated while the disattenuated one comes back to the injected value. That is a
stronger claim than "the division is implemented correctly", and it is the claim
the paper makes.

**The Jaccard statistic must be calibrated against its own null.** When two
configurations share an identical latent difficulty, `J_config` must land on
`J_seed` and the permutation test must *not* fire. A statistic that reports excess
instability on data with no configuration effect would produce the paper's headline
result out of noise alone, so this is the single most important test in the file.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.transfer import (
    HARD_QUANTILE,
    boundary_tie_fraction,
    hard_subset,
    hard_subset_overlap,
    measure_reliability,
    model_family,
    transfer_summary,
    transfer_table,
)
from src.metrics import jaccard, spearman
from synthetic import corpus_from_probs

np = pytest.importorskip("numpy")


# ---------------------------------------------------------------- latent builders
def _correlated_latents(n_items, rho, rng, spread=1.0, centre=0.0):
    """Two latent logit vectors with population Spearman approximately `rho`.

    Built through a bivariate normal, whose Spearman correlation is
    `(6/pi) * arcsin(rho_pearson / 2)`. The test inverts that relation rather than
    assuming Spearman equals Pearson, because at rho = 0.8 the two differ by about
    0.02 and the tolerances here are tighter than that.
    """
    rho_pearson = 2.0 * math.sin(math.pi * rho / 6.0)
    cov = [[1.0, rho_pearson], [rho_pearson, 1.0]]
    draws = rng.multivariate_normal([0.0, 0.0], cov, size=n_items)
    z = centre + spread * draws
    return z[:, 0], z[:, 1]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------ disattenuation maths
def test_disattenuation_is_the_classical_correction():
    """`rho_raw / r_mm` equals `rho_raw / sqrt(r*r)`: the spec's form is classical."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.6 * x + rng.normal(size=200)
    rho = spearman(x, y)
    r_mm = 0.9
    assert rho / r_mm == pytest.approx(rho / math.sqrt(r_mm * r_mm))


def test_disattenuated_correlation_recovers_the_latent_correlation():
    """The point of the correction: attenuated in, true value out.

    Two configurations with an injected latent Spearman of 0.80, observed at N=24 so
    the sampling noise is substantial. The raw correlation must fall clearly short of
    0.80, and dividing by the measured reliability must bring it back.
    """
    rng = np.random.default_rng(20260729)
    n_items = 600
    rho_true = 0.80
    z0, z1 = _correlated_latents(n_items, rho_true, rng, spread=1.1)
    p0, p1 = _sigmoid(z0), _sigmoid(z1)
    corpus = corpus_from_probs(
        {
            ("qwen/m0", "c0", 0): p0,
            ("qwen/m0", "c0", 1): p0,   # independent replicate: same latent
            ("qwen/m0", "c0", 2): p0,
            ("qwen/m0", "c1", 0): p1,
        },
        n_samples=24,
        seed=5,
    )
    rows = transfer_table(
        corpus, configs=["c0", "c1"], n_bootstrap=300, bootstrap_seed=1
    )
    config_rows = [r for r in rows if r.kind == "config"]
    assert len(config_rows) == 1
    row = config_rows[0]

    assert row.rho_raw < rho_true - 0.05, "sampling noise should attenuate the raw rho"
    assert row.reliability < 1.0, "reliability must be below 1 at finite N"
    assert row.rho_disatt == pytest.approx(rho_true, abs=0.06), (
        f"rho_raw={row.rho_raw:.3f} r_mm={row.reliability:.3f} "
        f"rho_disatt={row.rho_disatt:.3f}, injected {rho_true}"
    )
    assert row.rho_disatt > row.rho_raw


def test_disattenuation_recovers_a_perfect_latent_correlation():
    """Identical latents: the corrected transfer must be ~1.0 even though raw is not.

    This is the case that would otherwise be misread as a configuration effect. If
    the correction did not return ~1.0 here, every reported `rho_disatt` would be
    biased downward and the paper's Q2 answer would be an artefact.
    """
    rng = np.random.default_rng(11)
    p = _sigmoid(rng.normal(0.0, 1.2, 600))
    corpus = corpus_from_probs(
        {
            ("qwen/m0", "c0", 0): p,
            ("qwen/m0", "c0", 1): p,
            ("qwen/m0", "c0", 2): p,
            ("qwen/m0", "c1", 0): p,
        },
        n_samples=24,
        seed=9,
    )
    rows = transfer_table(corpus, configs=["c0", "c1"], n_bootstrap=200)
    row = next(r for r in rows if r.kind == "config")
    assert row.rho_raw < 0.95
    assert row.rho_disatt == pytest.approx(1.0, abs=0.06)


def test_reliability_rises_with_more_samples_per_cell():
    """`r_mm` is a function of the sampling noise, so it must improve with `N`."""
    rng = np.random.default_rng(3)
    p = _sigmoid(rng.normal(0.0, 1.0, 400))
    measured = {}
    for n_samples in (8, 24, 96):
        corpus = corpus_from_probs(
            {("qwen/m0", "c0", s): p for s in (0, 1, 2)}, n_samples=n_samples, seed=17
        )
        reliability = measure_reliability(
            corpus, "synth", None, "qwen/m0", n_bootstrap=100
        )
        measured[n_samples] = reliability.r_mm
    assert measured[8] < measured[24] < measured[96]
    assert measured[96] > 0.9


def test_reliability_needs_two_seeds_and_says_so(caplog):
    rng = np.random.default_rng(1)
    p = _sigmoid(rng.normal(size=50))
    corpus = corpus_from_probs({("qwen/m0", "c0", 0): p}, n_samples=24)
    with caplog.at_level("WARNING"):
        assert measure_reliability(corpus, "synth", None, "qwen/m0") is None
    assert any("has no denominator" in r.getMessage() for r in caplog.records)


def test_model_pairs_use_the_square_root_denominator():
    """Reliabilities differ across models, so the geometric mean is the right scale."""
    rng = np.random.default_rng(4)
    p_a = _sigmoid(rng.normal(0.0, 1.0, 300))
    p_b = _sigmoid(rng.normal(0.0, 1.0, 300))
    corpus = corpus_from_probs(
        {
            ("qwen/a", "c0", 0): p_a,
            ("qwen/a", "c0", 1): p_a,
            ("meta-llama/b", "c0", 0): p_b,
            ("meta-llama/b", "c0", 1): p_b,
        },
        n_samples=24,
        seed=2,
    )
    rows = transfer_table(corpus, configs=["c0"], n_bootstrap=100)
    cross = [r for r in rows if r.kind in ("model", "family")]
    assert len(cross) == 1
    assert cross[0].kind == "family"
    assert "sqrt" in cross[0].reliability_note


def test_model_family_splits_on_the_org_prefix():
    assert model_family("Qwen/Qwen2.5-3B-Instruct") == "Qwen"
    assert model_family("meta-llama/Llama-3.2-3B-Instruct") == "meta-llama"
    assert model_family("Qwen/Qwen2.5-7B") != model_family("meta-llama/Llama-3.1-8B")
    assert model_family("mock-small") == "mock"


def test_transfer_summary_evaluates_p2():
    rng = np.random.default_rng(6)
    z0, z1 = _correlated_latents(500, 0.55, rng, spread=1.1)
    corpus = corpus_from_probs(
        {
            ("qwen/m0", "c0", 0): _sigmoid(z0),
            ("qwen/m0", "c0", 1): _sigmoid(z0),
            ("qwen/m0", "c0", 2): _sigmoid(z0),
            ("qwen/m0", "c1", 0): _sigmoid(z1),
        },
        n_samples=48,
        seed=8,
    )
    rows = transfer_table(corpus, configs=["c0", "c1"], n_bootstrap=200)
    summary = transfer_summary(rows, kind="config")
    assert summary["n_pairs"] == 1
    assert summary["rho_disatt_below_085"] is True
    assert summary["prediction"] == "P2"


# ------------------------------------------------------------------- Jaccard maths
def test_jaccard_closed_form():
    assert jaccard([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert jaccard([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(2 / 6)
    assert jaccard([1, 2], [3, 4]) == pytest.approx(0.0)
    assert math.isnan(jaccard([], []))
    # A duplicate cannot inflate the overlap: these are sets.
    assert jaccard([1, 1, 2], [1, 2]) == pytest.approx(1.0)


def test_hard_subset_size_and_ordering():
    items = [f"i{i:02d}" for i in range(40)]
    p_hat = [i / 40 for i in range(40)]
    subset = hard_subset(items, p_hat, HARD_QUANTILE)
    assert len(subset) == 10
    assert set(subset) == set(items[:10])


def test_hard_subset_tie_break_is_stable_across_columns():
    """A tie-break that varied by configuration would manufacture instability.

    Every item is tied at `p_hat = 0`, so the subset is entirely determined by the
    tie-break. Two identical columns must therefore produce Jaccard exactly 1.0.
    """
    items = [f"i{i:02d}" for i in range(40)]
    flat = [0.0] * 40
    first = hard_subset(items, flat)
    second = hard_subset(items, list(flat))
    assert first == second
    assert jaccard(first, second) == pytest.approx(1.0)
    # And it does not depend on the order the items arrive in.
    order = list(np.random.default_rng(0).permutation(40))
    shuffled = hard_subset([items[i] for i in order], [flat[i] for i in order])
    assert sorted(shuffled) == sorted(first)


def test_boundary_tie_fraction_flags_a_saturated_benchmark():
    saturated = [0.0] * 30 + [i / 10 for i in range(1, 11)]
    assert boundary_tie_fraction(saturated) > 0.5
    spread = [i / 40 for i in range(40)]
    assert boundary_tie_fraction(spread) == pytest.approx(1 / 40)


# --------------------------------------------------------- the seed-pair null itself
@pytest.fixture(scope="module")
def null_overlap():
    """Configurations with an *identical* latent difficulty: no real effect at all."""
    rng = np.random.default_rng(777)
    p = _sigmoid(rng.normal(0.0, 1.2, 400))
    corpus = corpus_from_probs(
        {
            ("qwen/m0", "c0", 0): p,
            ("qwen/m0", "c0", 1): p,
            ("qwen/m0", "c0", 2): p,
            ("qwen/m0", "c1", 0): p,
            ("qwen/m0", "c2", 0): p,
            ("qwen/m0", "c3", 0): p,
        },
        n_samples=24,
        seed=31,
    )
    return hard_subset_overlap(
        corpus, "synth", None, "qwen/m0", configs=["c0", "c1", "c2", "c3"],
        n_bootstrap=300, n_permutations=300,
    )


def test_null_j_config_matches_the_seed_null(null_overlap):
    """No configuration effect means no excess instability. The calibration test."""
    result = null_overlap
    assert result is not None
    assert 0.0 < result.j_config < 1.0
    assert 0.0 < result.j_seed < 1.0
    assert result.excess == pytest.approx(0.0, abs=0.06), (
        f"J_config={result.j_config:.3f} J_seed={result.j_seed:.3f} on data with "
        "identical latent difficulty across configurations"
    )


def test_null_reestimation_test_does_not_fire(null_overlap):
    assert null_overlap.null_test["available"] is True
    assert null_overlap.null_test["p_value"] > 0.05


def test_reestimation_null_centre_agrees_with_the_seed_null(null_overlap):
    """Internal consistency: two routes to the same null must agree.

    `J_seed` measures re-estimation overlap from real independent seeds;
    `null_test["null_mean"]` simulates it binomially. They estimate the same
    quantity, so a disagreement would mean one of them is wrong.
    """
    assert null_overlap.null_test["null_mean"] == pytest.approx(
        null_overlap.j_seed, abs=0.06
    )


def test_spec_permutation_test_is_uninformative_by_construction(null_overlap):
    """Documents the METHOD_SPEC 5.4 step 5 defect so it cannot be misread."""
    perm = null_overlap.spec_permutation
    assert perm["uninformative_by_construction"] is True
    assert perm["null_mean_j_config"] == pytest.approx(
        perm["observed_j_config"], abs=0.03
    )


def test_corrected_interaction_permutation_is_not_flagged_uninformative(null_overlap):
    perm = null_overlap.permutation
    assert perm.get("interaction_test") is True
    assert perm.get("uninformative_by_construction") is False


def test_neither_overlap_reaches_one_at_finite_n(null_overlap):
    """Re-estimation noise alone breaks overlap, which is why 1.0 is the wrong null."""
    assert null_overlap.j_seed < 0.95
    assert null_overlap.j_config < 0.95


@pytest.fixture(scope="module")
def real_effect_overlap():
    """Configurations that genuinely reorder difficulty: the excess must appear."""
    rng = np.random.default_rng(888)
    n_items = 400
    base = rng.normal(0.0, 1.2, n_items)
    latents = {}
    for i, config in enumerate(["c0", "c1", "c2", "c3"]):
        # Each configuration gets its own independent item x config perturbation of
        # comparable size to the item main effect, which is the regime the paper
        # claims to find.
        latents[config] = base + rng.normal(0.0, 1.0, n_items)
    probs = {("qwen/m0", "c0", s): _sigmoid(latents["c0"]) for s in (0, 1, 2)}
    for config in ("c1", "c2", "c3"):
        probs[("qwen/m0", config, 0)] = _sigmoid(latents[config])
    corpus = corpus_from_probs(probs, n_samples=24, seed=42)
    return hard_subset_overlap(
        corpus, "synth", None, "qwen/m0", configs=["c0", "c1", "c2", "c3"],
        n_bootstrap=300, n_permutations=300,
    )


def test_real_configuration_effect_produces_excess_instability(real_effect_overlap):
    result = real_effect_overlap
    assert result.j_config < result.j_seed
    assert result.excess > 0.10
    assert result.excess_ci[0] > 0.0


def test_real_configuration_effect_is_detected_by_the_reestimation_null(real_effect_overlap):
    """The test that has power: J_config falls below what re-estimation explains."""
    assert real_effect_overlap.null_test["available"] is True
    assert real_effect_overlap.null_test["p_value"] < 0.05
    assert real_effect_overlap.j_config < real_effect_overlap.null_test["null_q05"]


def test_spec_permutation_test_misses_a_real_configuration_effect(real_effect_overlap):
    perm = real_effect_overlap.spec_permutation
    assert perm["p_value"] > 0.05


def test_overlap_reports_its_own_geometry(real_effect_overlap):
    row = real_effect_overlap.as_dict()
    assert row["n_config_pairs"] == 6
    assert row["n_seed_pairs"] == 3
    assert row["n_items"] == 400
    assert row["quantile"] == HARD_QUANTILE
    assert math.isfinite(row["boundary_tie_fraction"])
