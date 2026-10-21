"""The artefact contract: exact filenames, and a build that survives thin data.

`paper/main.tex` cites thirteen-plus artefacts by literal filename, so a rename is
a silent failure -- LaTeX emits a missing-float warning and the paper still
compiles. These tests pin the names and assert the build produces them from a
synthetic corpus, end to end, with plotting included.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.paper import (
    FIGURE_FILENAMES,
    TABLE_FILENAMES,
    PaperConfig,
    analyse,
    build,
    write_csv,
)
from synthetic import CORRECT_CLASS as G
from synthetic import make_cell, make_corpus

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

#: `c3` is in the corpus but is a separate arm: CDV spreads over it, the crossed
#: variance design must not.
CONFIGS = ("c0", "c1", "c2", "c3")
PRIMARY = ("c0", "c1", "c2")
MODELS = ("Qwen/tiny-1.5B", "meta-llama/tiny-3B")


#: Two benchmarks so the distribution-shift figure has something to compare; the
#: second gets a larger item x config term, which is the shift the paper predicts.
BENCHMARKS = (("gsm_symbolic", "main", 0.7), ("gsm_symbolic", "p2", 1.1))


def _paper_corpus(n_items=60, n=12, seed=0):
    """Small but fully crossed: two benchmarks, two models, three configs, seeds at c0."""
    rng = np.random.default_rng(seed)
    cells = []
    for dataset, subset, sd_interaction in BENCHMARKS:
        base = rng.uniform(-1.5, 1.5, n_items)
        for i in range(n_items):
            for m_index, model in enumerate(MODELS):
                for config in CONFIGS:
                    z = base[i] + 0.3 * m_index + rng.normal(0, sd_interaction)
                    p = 1.0 / (1.0 + np.exp(-z))
                    seeds = (0, 1, 2) if config == "c0" else (0,)
                    for s in seeds:
                        labels = [
                            G if rng.random() < p else ("w1" if rng.random() < 0.6 else "w2")
                            for _ in range(n)
                        ]
                        cells.append(
                            make_cell(
                                f"item/{i:03d}", model, config, labels, seed=s,
                                dataset=dataset, subset=subset,
                                tokens=list(rng.integers(60, 140, n)),
                            )
                        )
    return make_corpus(cells)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    corpus = _paper_corpus()
    cfg = PaperConfig(
        n_bootstrap=120, n_repeats=10, budget_multiples=(1, 4, 12),
        table_multiples=(4, 12), allocation_target="c1",
    )
    analysis = analyse(corpus, cfg)
    out_dir = tmp_path_factory.mktemp("paper")
    return analysis, build(analysis, out_dir), out_dir


# ------------------------------------------------------------------- the contract
def test_filename_constants_match_the_spec():
    """Byte-for-byte, because main.tex cites these strings."""
    assert FIGURE_FILENAMES == {
        "F1": "fig1_variance_components.pdf",
        "F2": "fig2_transfer_scatter.pdf",
        "F3": "fig3_hard_subset_overlap.pdf",
        "F4": "fig4_ceiling_vs_coverage.pdf",
        "F5": "fig5_cdv_vs_sc.pdf",
        "F6": "fig6_reordering.pdf",
        "F7": "fig7_shift.pdf",
        "F8": "fig8_precision_control.pdf",
    }
    assert TABLE_FILENAMES == {
        "T1": "tab1_setup.csv",
        "T2": "tab2_main_accuracy.csv",
        "T3": "tab3_variance_components.csv",
        "T4": "tab4_transfer.csv",
        "T5": "tab5_method_comparison.csv",
        "T6": "tab6_downstream.csv",
        "T7": "tab7_precision_control.csv",
    }


def test_figures_are_pdf_and_tables_are_csv():
    assert all(name.endswith(".pdf") for name in FIGURE_FILENAMES.values())
    assert all(name.endswith(".csv") for name in TABLE_FILENAMES.values())


def test_all_single_precision_artefacts_are_written(built):
    """Everything except the precision control, which needs a second corpus."""
    _analysis, artefacts, out_dir = built
    for key, name in FIGURE_FILENAMES.items():
        if key == "F8":
            continue
        assert key in artefacts.written, f"{key} skipped: {artefacts.skipped.get(key)}"
        assert (out_dir / name).exists()
        assert (out_dir / name).stat().st_size > 1000
    for key, name in TABLE_FILENAMES.items():
        if key == "T7":
            continue
        assert key in artefacts.written, f"{key} skipped: {artefacts.skipped.get(key)}"
        assert (out_dir / name).exists()


def test_precision_control_is_skipped_with_an_actionable_reason(built):
    _analysis, artefacts, _out_dir = built
    for key in ("F8", "T7"):
        assert key not in artefacts.written
        assert "--precision-results-dir" in artefacts.skipped[key]


def test_artefact_manifest_records_the_run(built):
    _analysis, _artefacts, out_dir = built
    manifest = json.loads((out_dir / "artefacts.json").read_text(encoding="utf-8"))
    assert manifest["n_expected"] == 15
    assert manifest["expected_figures"] == FIGURE_FILENAMES
    assert set(manifest["written"]) | set(manifest["skipped"]) == set(
        FIGURE_FILENAMES
    ) | set(TABLE_FILENAMES)
    assert manifest["generated_at"].endswith("Z") or "T" in manifest["generated_at"]


def test_predictions_are_written_including_verdicts(built):
    _analysis, _artefacts, out_dir = built
    text = (out_dir / "predictions.csv").read_text(encoding="utf-8")
    assert "prediction" in text.splitlines()[0]
    assert "P1" in text
    for tag in ("P2", "P3", "P4", "P5"):
        assert tag in text
    assert "F3_primary" in text


# ---------------------------------------------------------------- table contents
def test_tab3_reports_censoring_and_the_noise_floor(built):
    _analysis, _artefacts, out_dir = built
    text = (out_dir / "tab3_variance_components.csv").read_text(encoding="utf-8")
    header = text.splitlines()[0]
    for column in ("component", "share", "share_ci_low", "noise_floor", "censoring_rate"):
        assert column in header
    assert "item_config" in text
    assert "DESIGN" in text


def test_tab4_carries_reliability_and_both_overlap_nulls(built):
    _analysis, _artefacts, out_dir = built
    text = (out_dir / "tab4_transfer.csv").read_text(encoding="utf-8")
    assert "r_mm" in text
    assert "rho_disatt" in text
    assert "j_seed_null" in text
    assert "null_p" in text
    assert "spec_permutation_uninformative" in text


def test_tab5_reports_realised_tokens_next_to_accuracy(built):
    """The matched-budget claim has to be auditable from the table alone."""
    _analysis, _artefacts, out_dir = built
    header = (out_dir / "tab5_method_comparison.csv").read_text(encoding="utf-8").splitlines()[0]
    for column in ("budget_tokens", "mean_tokens_used", "accuracy", "method"):
        assert column in header


def test_tab1_totals_the_corpus_even_without_manifests(built):
    _analysis, _artefacts, out_dir = built
    text = (out_dir / "tab1_setup.csv").read_text(encoding="utf-8")
    assert "TOTAL" in text
    assert "NOISE_FLOOR" in text


def test_analysis_bundle_is_populated(built):
    analysis, _artefacts, _out_dir = built
    assert analysis.glmm or analysis.variance
    assert analysis.transfer
    assert analysis.overlaps
    assert analysis.ceilings
    assert analysis.transitions
    assert analysis.cdv_points
    assert analysis.comparisons
    assert analysis.allocations
    # The crossed design uses the primary family only; c3 is present in the corpus
    # and available to CDV, but pooling it into the configuration factor would mix a
    # parse-format change into a claim about semantics-preserving elicitation.
    assert analysis.configs == list(PRIMARY)
    assert "c3" in analysis.corpus.configs(include_separate_arms=True)
    assert len(analysis.corpus.benchmarks()) == 2


# ------------------------------------------------------------------ robustness
def test_build_survives_a_corpus_too_thin_for_most_artefacts(tmp_path):
    """A dead session leaves a partial corpus; that must cost artefacts, not the run."""
    cells = [
        make_cell(f"i{i}", "m0", "c0", [G] * 3 + ["w1"] * 3, tokens=100)
        for i in range(5)
    ]
    analysis = analyse(make_corpus(cells), PaperConfig(n_bootstrap=20, n_repeats=3))
    artefacts = build(analysis, tmp_path / "thin")
    assert artefacts.skipped, "a single-configuration corpus cannot support F1"
    assert "F1" in artefacts.skipped
    assert (tmp_path / "thin" / "artefacts.json").exists()
    assert "[skip]" in artefacts.report()


def test_build_reports_every_expected_artefact_in_its_report(built):
    _analysis, artefacts, _out_dir = built
    report = artefacts.report()
    for name in list(FIGURE_FILENAMES.values()) + list(TABLE_FILENAMES.values()):
        assert name in report


# --------------------------------------------------------------------- csv writer
def test_write_csv_handles_ragged_rows_and_specials(tmp_path):
    path = write_csv(
        [
            {"a": 1, "b": float("nan")},
            {"a": 2, "c": float("inf")},
            {"a": 3, "b": True, "d": [1, 2]},
        ],
        tmp_path / "x.csv",
    )
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    assert lines[0] == "a,b,c,d"
    assert lines[1] == "1,,,"        # nan becomes an empty cell, not the string "nan"
    assert lines[2] == "2,,inf,"
    assert lines[3] == "3,true,,1;2"
