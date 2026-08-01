"""End-to-end pipeline tests on CPU with MockBackend: no GPU, no network.

These are the tests that would have caught every integration bug in this
project: config -> dataset -> strategy -> JSONL -> grading -> metrics ->
figures/tables, plus the CLI surface and the external-plugin path that a novel
method will use.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.checkpointing import RunState, load_records
from src.config import load_config
from src.runner import main as runner_main
from src.runner import run
from src.strategies import available_strategies, build_strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = str(REPO_ROOT / "configs" / "base.yaml")
SMOKE_CONFIG = str(REPO_ROOT / "configs" / "smoke_mock.yaml")


def mock_cfg(tmp_path: Path, strategy: str = "cot_zeroshot", **over: object):
    overrides = [
        "model.backend=mock",
        "model.name_or_path=mock/tiny",
        "model.mock_accuracy=0.7",
        "data.name=toy",
        "data.subset=null",
        "data.subsample=null",
        "generation.max_new_tokens=64",
        "generation.batch_size=4",
        "generation.logprobs=true",
        f"strategy.name={strategy}",
        f"runtime.out_dir={tmp_path.as_posix()}",
        "runtime.log_every=100",
        "runtime.time_budget_hours=1.0",
    ]
    overrides += [f"{k}={v}" for k, v in over.items()]
    return load_config(BASE_CONFIG, overrides)


# ------------------------------------------------------------------ full pipeline
def test_full_run_writes_all_artifacts(tmp_path: Path):
    cfg = mock_cfg(tmp_path)
    summary = run(cfg)

    assert summary.is_complete
    assert cfg.results_path.exists()
    assert cfg.manifest_path.exists()
    assert cfg.graded_path.exists(), "grade_after_run should produce graded.jsonl"

    records = load_records(cfg.results_path)
    assert len(records) == summary.n_total
    for r in records:
        assert r["dataset"] == "toy"
        assert r["strategy"] == "cot_zeroshot"
        assert r["question"] and r["gold_answer"]
        assert r["tokens_completion"] > 0
        assert r["n_calls"] >= 1
        assert r["latency_s"] >= 0
        assert isinstance(r["reasoning_traces"], list) and r["reasoning_traces"]
        json.dumps(r)  # every record must round-trip as JSON

    assert summary.tokens_completion > 0
    assert summary.grading is not None
    assert 0.0 <= summary.grading["accuracy"] <= 1.0

    state = RunState.load(cfg.manifest_path)
    assert state is not None
    assert state.status == "complete"
    assert state.config["strategy"]["name"] == "cot_zeroshot"
    assert state.sessions and state.sessions[-1]["backend"]["backend"] == "mock"


def test_mock_accuracy_is_neither_zero_nor_one(tmp_path: Path):
    """Guards against a grader that always passes or always fails."""
    cfg = mock_cfg(tmp_path, strategy="self_consistency", **{"strategy.params.k": 4})
    cfg.model.mock_accuracy = 0.6
    summary = run(cfg)
    acc = summary.grading["accuracy"]
    assert 0.0 < acc < 1.0, f"suspicious end-to-end accuracy {acc}"


@pytest.mark.parametrize("strategy", sorted(available_strategies()))
def test_every_registered_strategy_runs_through_the_runner(
    strategy: str, tmp_path: Path
):
    cfg = mock_cfg(tmp_path / strategy, strategy=strategy)
    cfg.runtime.max_examples = 3
    summary = run(cfg)
    assert summary.n_completed_this_session == 3
    assert summary.n_errors_this_session == 0, f"{strategy} raised on a toy example"
    records = load_records(cfg.results_path)
    assert all(r["error"] is None for r in records)
    assert all(r["reasoning_traces"] for r in records)


def test_external_plugin_strategy_loads_by_path(tmp_path: Path):
    """The extension point a novel method uses: no edits to src/ required."""
    strategy = build_strategy("examples.method_template:TemplateMethod", k=3)
    assert strategy.params["k"] == 3

    cfg = mock_cfg(tmp_path, strategy="examples.method_template:TemplateMethod")
    cfg.strategy.params = {"k": 3}
    cfg.runtime.max_examples = 4
    summary = run(cfg)
    assert summary.n_completed_this_session == 4
    assert summary.n_errors_this_session == 0
    records = load_records(cfg.results_path)
    assert all(len(r["reasoning_traces"]) == 3 for r in records)
    assert all("candidate_answers" in r["extra"] for r in records)


# ---------------------------------------------------------------------- grading
def test_grading_is_separate_cached_and_rerunnable(tmp_path: Path):
    from src.grading import grade_file, load_graded

    cfg = mock_cfg(tmp_path, **{"runtime.grade_after_run": "false"})
    run(cfg)
    assert not cfg.graded_path.exists(), "grading must be opt-out of the run loop"

    first = grade_file(cfg.results_path, cfg.graded_path)
    assert first["n_graded_now"] > 0
    graded = load_graded(cfg.graded_path)
    assert len(graded) == first["n_total"]

    # Re-running must be a cheap no-op: this is what makes a grader fix free.
    second = grade_file(cfg.results_path, cfg.graded_path)
    assert second["n_graded_now"] == 0
    assert second["n_skipped"] == first["n_total"]
    assert second["accuracy"] == pytest.approx(first["accuracy"])

    # force=True regrades everything without touching raw generations.
    raw_before = cfg.results_path.read_bytes()
    forced = grade_file(cfg.results_path, cfg.graded_path, force=True)
    assert forced["n_graded_now"] == first["n_total"]
    assert cfg.results_path.read_bytes() == raw_before, "results.jsonl is append-only"

    for g in load_graded(cfg.graded_path):
        assert isinstance(g["is_correct"], bool)
        assert isinstance(g["sample_correct"], list)
        assert len(g["sample_correct"]) == len(g["reasoning_traces"])
        assert g["n_correct_samples"] == sum(g["sample_correct"])
        assert g["grader_version"] and g["grader_backend"]


def test_grading_survives_a_record_with_an_error(tmp_path: Path):
    from src.checkpointing import JsonlAppender
    from src.grading import grade_record

    bad = {
        "uid": "x",
        "gold_answer": "42",
        "answer_type": "math",
        "final_answer": None,
        "reasoning_traces": [],
        "error": "RuntimeError: boom",
    }
    graded = grade_record(bad)
    assert graded["is_correct"] is False
    assert graded["n_samples_graded"] == 0


# --------------------------------------------------------------------- analysis
def test_aggregate_emits_summary_figures_and_tables(tmp_path: Path):
    from src.analysis.aggregate import main as aggregate_main

    for strategy in ("cot_zeroshot", "self_consistency"):
        run(mock_cfg(tmp_path / "results", strategy=strategy))

    out_dir = tmp_path / "paper_assets"
    rc = aggregate_main(
        [
            "--results-dir",
            str(tmp_path / "results"),
            "--out-dir",
            str(out_dir),
            "--figures",
            "--tables",
            "--baseline",
            "cot_zeroshot",
        ]
    )
    assert rc == 0

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summaries) >= 2
    assert (out_dir / "summary.csv").exists()

    pdfs = list(out_dir.rglob("*.pdf"))
    assert pdfs, "no figures were produced"
    assert all(p.stat().st_size > 1000 for p in pdfs)

    texs = list(out_dir.rglob("*.tex"))
    assert texs, "no LaTeX tables were produced"
    assert any("\\toprule" in p.read_text(encoding="utf-8") for p in texs)


# -------------------------------------------------------------------------- CLI
def test_cli_list_and_dry_run(tmp_path: Path, capsys):
    assert runner_main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "toy" in out and "cot_zeroshot" in out

    rc = runner_main(
        [
            "--config",
            SMOKE_CONFIG,
            "--dry-run",
            "--set",
            f"runtime.out_dir={tmp_path.as_posix()}",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "plan:" in out and "resume command:" in out


def test_smoke_config_runs_as_a_subprocess(tmp_path: Path):
    """The exact command the Kaggle notebook runs first, in a clean process."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.runner",
            "--config",
            SMOKE_CONFIG,
            "--set",
            f"runtime.out_dir={tmp_path.as_posix()}",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RUN COMPLETE" in proc.stdout
    assert list(tmp_path.rglob("results.jsonl")), "no results were written"
