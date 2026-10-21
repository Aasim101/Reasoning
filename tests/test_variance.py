"""Variance decomposition: closed-form identities first, then synthetic recovery.

The decomposition is the paper's headline number and nothing about it can be
checked by inspection, so these tests are ordered from "provable" to "statistical":

1. On an exactly additive design with no noise, the method-of-moments estimator
   must return the design's own component variances to floating-point precision.
2. Subtracting a noise floor must move the residual by exactly the floor and leave
   every other component untouched -- the identity the module's docstring claims.
3. The Haldane sampling variance must be ~2.0 and flat in `N` at a saturated cell,
   which is the arithmetic behind the spec's decision to buy items over samples.
4. Only then: recovery of injected components from binomial draws, where the
   tolerance has to accommodate real estimator bias.
"""

from __future__ import annotations

import math

import pytest

from src.analysis.variance import (
    COMPONENT_NAMES,
    components_from_ms,
    mean_squares,
    measure_noise_floor,
    shares_from_components,
    variance_decomposition,
)
from src.metrics import haldane_logit, haldane_logit_var, is_saturated
from synthetic import cell_from_counts, make_corpus, synthetic_design

np = pytest.importorskip("numpy")


# ------------------------------------------------------------------ closed form
def _additive_design(n_items=7, n_models=3, n_configs=4, seed=0):
    """`z` built to be exactly additive, with the component variances returned."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.1, n_items)
    b = rng.normal(0, 0.7, n_models)
    g = rng.normal(0, 0.4, n_configs)
    a, b, g = a - a.mean(), b - b.mean(), g - g.mean()
    z = 0.3 + a[:, None, None] + b[None, :, None] + g[None, None, :]
    truth = {
        "item": float(np.var(a, ddof=1)),
        "model": float(np.var(b, ddof=1)),
        "config": float(np.var(g, ddof=1)),
        "item_model": 0.0,
        "item_config": 0.0,
        "model_config": 0.0,
        "residual": 0.0,
    }
    return z, truth


def test_purely_additive_design_recovered_exactly():
    """No noise, no interaction: the estimator is an identity, so demand precision."""
    z, truth = _additive_design()
    fit = mean_squares(z)
    dims = fit["dims"]
    components = components_from_ms(
        {name: float(fit["ms"][name]) for name in COMPONENT_NAMES},
        dims["n_items"],
        dims["n_models"],
        dims["n_configs"],
    )
    for name in COMPONENT_NAMES:
        assert components[name] == pytest.approx(truth[name], abs=1e-9), name


def test_pure_interaction_lands_entirely_in_one_sum_of_squares():
    """A double-centred item x config effect puts all of the SS in `item_config`.

    Asserted on sums of squares rather than on the method-of-moments components: a
    double-centred effect is a *fixed* interaction with zero marginal means, and the
    moment estimator, which assumes the marginals of a random interaction are
    nonzero, correctly returns a small negative item component for it. The SS
    partition is the part that is exact arithmetic.
    """
    rng = np.random.default_rng(11)
    n_items, n_models, n_configs = 9, 3, 5
    ag = rng.normal(0, 0.8, (n_items, n_configs))
    ag = ag - ag.mean(axis=0, keepdims=True) - ag.mean(axis=1, keepdims=True) + ag.mean()
    z = np.repeat(ag[:, None, :], n_models, axis=1)
    fit = mean_squares(z)
    assert float(fit["ss"]["item_config"]) == pytest.approx(
        n_models * float((ag**2).sum()), rel=1e-9
    )
    for name in ("item", "model", "config", "item_model", "model_config", "residual"):
        assert float(fit["ss"][name]) == pytest.approx(0.0, abs=1e-9), name


def test_sums_of_squares_partition_the_total():
    """The ANOVA identity: the seven SS must add up to the total SS, exactly."""
    z = np.random.default_rng(4).normal(size=(13, 4, 3))
    fit = mean_squares(z)
    total = float(((z - z.mean()) ** 2).sum())
    assert sum(float(fit["ss"][name]) for name in COMPONENT_NAMES) == pytest.approx(
        total, rel=1e-9
    )
    assert sum(fit["df"][name] for name in COMPONENT_NAMES) == z.size - 1


def test_noise_floor_subtraction_only_moves_the_residual():
    """The identity the module relies on: the floor cancels out of every contrast.

    Each structural component is a contrast of mean squares with coefficients
    summing to zero, so subtracting a constant from all of them changes nothing but
    the residual. If this ever fails, the shares are being corrected twice.
    """
    z, _ = _additive_design(seed=5)
    z = z + np.random.default_rng(2).normal(0, 0.4, z.shape)
    fit = mean_squares(z)
    dims = fit["dims"]
    ms = {name: float(fit["ms"][name]) for name in COMPONENT_NAMES}
    args = (dims["n_items"], dims["n_models"], dims["n_configs"])

    uncorrected = components_from_ms(ms, *args, noise_floor=0.0)
    floor = 0.137
    corrected = components_from_ms(ms, *args, noise_floor=floor)

    assert corrected["residual"] == pytest.approx(uncorrected["residual"] - floor, abs=1e-12)
    for name in COMPONENT_NAMES:
        if name == "residual":
            continue
        assert corrected[name] == pytest.approx(uncorrected[name], abs=1e-12), name


def test_shares_are_taken_over_corrected_total():
    """Correcting the floor must *raise* every structural share, not just the residual."""
    z, _ = _additive_design(seed=8)
    z = z + np.random.default_rng(3).normal(0, 0.5, z.shape)
    fit = mean_squares(z)
    dims = fit["dims"]
    ms = {name: float(fit["ms"][name]) for name in COMPONENT_NAMES}
    args = (dims["n_items"], dims["n_models"], dims["n_configs"])
    raw, _ = shares_from_components(components_from_ms(ms, *args, noise_floor=0.0))
    corrected, _ = shares_from_components(components_from_ms(ms, *args, noise_floor=0.2))
    assert float(corrected["item"]) > float(raw["item"])
    assert float(corrected["residual"]) < float(raw["residual"])
    assert sum(float(corrected[n]) for n in COMPONENT_NAMES) == pytest.approx(1.0)


def test_clamping_is_reported_not_hidden():
    negative = {name: -0.05 for name in COMPONENT_NAMES}
    negative["item"] = 1.0
    shares, info = shares_from_components(negative, clamp_negative=True)
    assert float(shares["item"]) == pytest.approx(1.0)
    assert bool(np.asarray(info["clamped"]["model"]).any())
    assert not bool(np.asarray(info["clamped"]["item"]).any())


# ----------------------------------------------------- the saturated-cell property
@pytest.mark.parametrize("n", [8, 24, 96, 1000])
def test_saturated_sampling_variance_is_two_and_flat_in_n(n):
    """`1/(k+1/2) + 1/(N-k+1/2)` at `k=0` is ~2.0 whatever `N` is.

    This is why saturated cells dominate the noise floor and why more samples
    cannot rescue them -- the budget decision in METHOD_SPEC 8.4 rests on it.
    """
    assert haldane_logit_var(0, n) == pytest.approx(2.0, abs=0.15)
    assert haldane_logit_var(n, n) == pytest.approx(2.0, abs=0.15)
    assert is_saturated(0, n) and is_saturated(n, n)


def test_interior_sampling_variance_halves_when_n_doubles():
    """The contrast to the above: an undecided cell *does* respond to more samples."""
    small = haldane_logit_var(12, 24)
    large = haldane_logit_var(24, 48)
    assert large == pytest.approx(small / 2.0, rel=0.05)
    assert not is_saturated(12, 24)


def test_saturated_floor_dominates_a_mixed_cell_set():
    """A design that is half decided cannot get its floor below roughly half of 2.0."""
    interior = [haldane_logit_var(12, 24) for _ in range(50)]
    decided = [haldane_logit_var(0, 24) for _ in range(50)]
    floor = float(np.mean(interior + decided))
    assert floor > 0.9
    # Ten times the samples barely moves it, because only the interior half shrinks.
    interior_big = [haldane_logit_var(120, 240) for _ in range(50)]
    floor_big = float(np.mean(interior_big + [haldane_logit_var(0, 240) for _ in range(50)]))
    assert floor_big > 0.9 * floor


def test_haldane_logit_is_finite_at_the_boundary():
    assert math.isfinite(haldane_logit(0, 24))
    assert math.isfinite(haldane_logit(24, 24))
    assert haldane_logit(0, 24) == pytest.approx(-haldane_logit(24, 24))


# ------------------------------------------------------------- measured noise floor
def test_measured_noise_floor_matches_injected_variance():
    """Seed replicates drawn around a known `p` recover the analytic floor.

    The replicate floor and the analytic binomial floor are two independent routes
    to the same quantity, so agreement between them on data generated as independent
    Bernoulli draws is a real check on both.
    """
    corpus, _ = synthetic_design(
        n_items=250, models=("m0", "m1"), configs=("c0", "c1"), n_samples=24,
        sd_item=0.8, sd_model=0.3, sd_config=0.2, sd_item_model=0.4,
        sd_item_config=0.5, n_seeds=4, seed=7, clip=2.5,
    )
    floor = measure_noise_floor(corpus, "synth", None)
    assert floor.n_seeds == 4
    assert floor.n_pairs == 6
    assert floor.used == "replicate"
    ratio = floor.replicate / floor.analytic
    assert 0.75 < ratio < 1.35, f"replicate/analytic = {ratio:.3f}"


def test_noise_floor_falls_back_and_warns_without_replicates(caplog):
    corpus, _ = synthetic_design(
        n_items=40, models=("m0", "m1"), configs=("c0", "c1"), n_seeds=1, seed=1
    )
    with caplog.at_level("WARNING"):
        floor = measure_noise_floor(corpus, "synth", None)
    assert floor.used == "analytic"
    assert math.isfinite(floor.value)
    assert any("no seed replicates" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------- synthetic recovery
@pytest.fixture(scope="module")
def recovery_fit():
    """A design with injected components, fitted from binomial counts.

    Deliberately kept away from the boundary (moderate effect sizes, no clipping,
    `N=64`): the recovery target is the estimator's behaviour, and a saturated design
    would instead be testing how the Haldane correction compresses extreme logits,
    which the censoring tests cover separately.
    """
    corpus, truth = synthetic_design(
        n_items=400,
        models=("m0", "m1", "m2"),
        configs=("c0", "c1", "c2", "c3"),
        n_samples=64,
        sd_item=0.7,
        sd_model=0.3,
        sd_config=0.2,
        sd_item_model=0.35,
        sd_item_config=0.45,
        n_seeds=3,
        seed=424242,
        clip=None,
    )
    result = variance_decomposition(corpus, "synth", None, n_bootstrap=200)
    return result, truth


def test_recovers_the_components_of_the_noiseless_logit_array(recovery_fit):
    """Fitting noisy counts must reproduce a noiseless fit of the same latent array.

    This is the sharp form of the recovery claim. The comparison target is
    `estimator_shares()` -- the components of the realised latent logits -- not the
    population variances the effects were drawn from, so any gap is estimator error
    from the binomial sampling and the noise correction, which is exactly what is
    under test.
    """
    result, truth = recovery_fit
    target = truth.estimator_shares()
    for name, expected in target.items():
        assert result.shares[name] == pytest.approx(expected, abs=0.05), (
            f"{name}: got {result.shares[name]:.3f}, noiseless fit gives {expected:.3f}"
        )


def test_item_indexed_components_match_the_population_variances(recovery_fit):
    """The item, item x model and item x config components recover their true values.

    These are the components the paper's claims rest on, and they are estimated
    across 400 item levels, so they are well determined. Compared as *variances*
    rather than shares, because a share also depends on the poorly determined main
    effects below.
    """
    result, truth = recovery_fit
    for name in ("item", "item_model", "item_config"):
        expected = truth.components[name]
        assert result.components[name] == pytest.approx(expected, rel=0.20), (
            f"{name}: got {result.components[name]:.3f}, injected {expected:.3f}"
        )


def test_model_and_config_main_effects_are_weakly_determined_by_few_levels():
    """A caveat the paper must state, asserted so it cannot be forgotten.

    The model main-effect variance is estimated from as many levels as there are
    models -- three here, four in Tier A. The sampling distribution of a variance on
    three levels is enormously wide, so a model or configuration *main effect* share
    is not a precise quantity, whatever its point estimate looks like. The claims that
    matter (P1, about item x configuration) are indexed by item and do not inherit
    this problem. This test documents the asymmetry by showing the main-effect
    estimate missing its population value by a wide factor while the item-indexed
    components stay tight.
    """
    corpus, truth = synthetic_design(
        n_items=400, models=("m0", "m1", "m2"), configs=("c0", "c1", "c2", "c3"),
        n_samples=64, sd_item=0.7, sd_model=0.3, sd_config=0.2,
        sd_item_model=0.35, sd_item_config=0.45, n_seeds=3, seed=424242, clip=None,
    )
    noiseless = truth.estimator_components()
    # The realised three model levels happen to be far more spread than their
    # population sd implies; the estimator is right and the population value is
    # simply not identifiable from three draws.
    assert noiseless["model"] > 2.0 * truth.components["model"]
    assert noiseless["item"] == pytest.approx(truth.components["item"], rel=0.25)
    assert corpus is not None


def test_recovery_preserves_component_ordering(recovery_fit):
    result, truth = recovery_fit
    target = truth.estimator_shares()
    injected = [n for n in truth.shares]
    ranked_est = sorted(injected, key=lambda k: -result.shares[k])
    ranked_true = sorted(injected, key=lambda k: -target[k])
    assert ranked_est == ranked_true


def test_uninjected_components_stay_near_zero(recovery_fit):
    """No model x config and no three-way were injected, so both must be small."""
    result, _ = recovery_fit
    assert result.shares["model_config"] < 0.03
    assert result.shares["residual"] < 0.06


def test_bootstrap_intervals_cover_the_truth(recovery_fit):
    result, truth = recovery_fit
    target = truth.estimator_shares()
    for name, expected in target.items():
        low, high = result.shares_ci[name]
        assert low - 0.03 <= expected <= high + 0.03, (
            f"{name}: target {expected:.3f} outside [{low:.3f}, {high:.3f}]"
        )


def test_uncorrected_fit_understates_structural_shares():
    """Skipping the noise correction is not conservative -- it hides the interaction.

    Fit the same corpus with the floor forced to zero and confirm every structural
    share shrinks. This is the concrete reason the correction is not optional.
    """
    from src.analysis.variance import NoiseFloor

    corpus, _truth = synthetic_design(
        n_items=250, n_samples=16, sd_item=0.8, sd_item_config=0.6, n_seeds=3, seed=99
    )
    corrected = variance_decomposition(corpus, "synth", None, n_bootstrap=0)
    zeroed = variance_decomposition(
        corpus, "synth", None, n_bootstrap=0,
        noise_floor=NoiseFloor(replicate=0.0, analytic=0.0, used="replicate"),
    )
    assert zeroed.shares["item_config"] < corrected.shares["item_config"]
    assert zeroed.shares["item"] < corrected.shares["item"]
    assert zeroed.shares["residual"] > corrected.shares["residual"]


# ------------------------------------------------------------------- censoring
def test_items_saturated_everywhere_are_censored_and_counted():
    """Always-wrong and always-right items are excluded, and the rate is reported."""
    cells = []
    models, configs = ("m0", "m1"), ("c0", "c1", "c2")
    for i in range(30):
        for model in models:
            for config in configs:
                if i < 5:
                    k = 0          # never solved: saturated in every cell
                elif i < 8:
                    k = 24         # always solved: saturated at the other boundary
                else:
                    k = 6 + (i % 7) + (2 if config == "c1" else 0)
                cells.append(cell_from_counts(f"i{i:02d}", model, config, k, 24))
                if config == "c0":
                    cells.append(
                        cell_from_counts(f"i{i:02d}", model, config, max(0, k - 1), 24, seed=1)
                    )
    corpus = make_corpus(cells)
    result = variance_decomposition(corpus, "synth", None, n_bootstrap=0)

    assert result.censoring["n_items_before_censoring"] == 30
    assert result.censoring["n_items_all_zero"] == 5
    assert result.censoring["n_items_all_one"] == 3
    assert result.censoring["n_items_all_saturated"] == 8
    assert result.censoring["censoring_rate"] == pytest.approx(8 / 30)
    assert result.n_items == 22


def test_censoring_can_be_disabled_for_a_sensitivity_check():
    cells = []
    for i in range(20):
        for model in ("m0", "m1"):
            for config in ("c0", "c1", "c2"):
                k = 0 if i < 4 else 5 + (i % 9)
                cells.append(cell_from_counts(f"i{i:02d}", model, config, k, 24))
    corpus = make_corpus(cells)
    kept = variance_decomposition(
        corpus, "synth", None, n_bootstrap=0, exclude_all_saturated=False
    )
    dropped = variance_decomposition(corpus, "synth", None, n_bootstrap=0)
    assert kept.n_items == 20
    assert dropped.n_items == 16
    assert kept.censoring["excluded"] is False


def test_all_saturated_benchmark_raises_a_useful_error():
    cells = [
        cell_from_counts(f"i{i}", model, config, 24, 24)
        for i in range(12)
        for model in ("m0", "m1")
        for config in ("c0", "c1")
    ]
    with pytest.raises(ValueError, match="survive censoring"):
        variance_decomposition(make_corpus(cells), "synth", None, n_bootstrap=0)


def test_incomplete_design_is_restricted_to_the_crossed_intersection():
    """A missing cell biases a balanced fit, so those items must be dropped loudly."""
    cells = []
    for i in range(24):
        for model in ("m0", "m1"):
            for config in ("c0", "c1", "c2"):
                if i >= 20 and config == "c2":
                    continue  # four items never ran under c2
                cells.append(cell_from_counts(f"i{i:02d}", model, config, 4 + (i % 11), 24))
    corpus = make_corpus(cells)
    report = corpus.design_report("synth", None, ("m0", "m1"), ("c0", "c1", "c2"))
    assert report["n_items_union"] == 24
    assert report["n_items_crossed"] == 20
    assert report["n_items_dropped"] == 4
    assert report["balanced"] is False
    assert variance_decomposition(corpus, "synth", None, n_bootstrap=0).n_items == 20


def test_single_level_factor_is_rejected():
    """One configuration means its main effect and interactions are unidentifiable."""
    z = np.random.default_rng(0).normal(size=(10, 3, 1))
    with pytest.raises(ValueError, match="at least 2 levels"):
        mean_squares(z)
