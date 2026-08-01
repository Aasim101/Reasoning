"""GLMM primary estimator: synthetic recovery and parametric-bootstrap null."""

from __future__ import annotations

import pytest

from src.analysis.glmm import (
    glmm_decomposition,
    parametric_bootstrap_null,
)
from tests.synthetic import synthetic_design

np = pytest.importorskip("numpy")
statsmodels = pytest.importorskip("statsmodels")


@pytest.fixture(scope="module")
def glmm_fit():
    corpus, truth = synthetic_design(
        n_items=24,
        models=("m0", "m1"),
        configs=("c0", "c1", "c2"),
        n_samples=16,
        sd_item=0.7,
        sd_model=0.3,
        sd_config=0.2,
        sd_item_model=0.3,
        sd_item_config=0.45,
        n_seeds=3,
        seed=20260729,
        clip=None,
    )
    result = glmm_decomposition(
        corpus, "synth", None, n_bootstrap=3, bootstrap_seed=1
    )
    return result, truth


def test_glmm_recovers_injected_item_config_share(glmm_fit):
    result, truth = glmm_fit
    target = truth.estimator_shares().get("item_config", 0.0)
    assert result.converged
    assert result.shares["item_config"] == pytest.approx(target, abs=0.12)


def test_glmm_item_config_exceeds_null_calibration(glmm_fit):
    result, _ = glmm_fit
    corpus, _ = synthetic_design(
        n_items=40,
        models=("m0",),
        configs=("c0", "c1", "c2"),
        n_samples=16,
        sd_item_config=0.5,
        n_seeds=2,
        seed=3,
    )
    null = parametric_bootstrap_null(
        corpus, "synth", None, n_draws=5, draw_seed=9, use_glmm=True
    )
    assert null["n_glmm_draws"] >= 3
    assert result.shares["item_config"] > null["glmm_item_config_null_q95"]


def test_parametric_null_centres_near_zero_on_h0():
    corpus, _ = synthetic_design(
        n_items=36,
        models=("m0", "m1"),
        configs=("c0", "c1", "c2"),
        n_samples=16,
        sd_item=0.5,
        sd_item_config=0.0,
        sd_item_model=0.0,
        sd_config=0.0,
        n_seeds=3,
        seed=11,
    )
    null = parametric_bootstrap_null(
        corpus, "synth", None, n_draws=5, draw_seed=4
    )
    assert null["n_glmm_draws"] >= 3
    assert null["glmm_item_config_null_mean"] == pytest.approx(0.0, abs=0.15)
    if null["n_glmm_draws"] >= 10:
        assert null["passes_calibration"] is True
