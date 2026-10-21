"""Resumability tests: the property the whole project depends on.

A Kaggle session dies abruptly. These tests assert that after a kill and a
restart, every example is completed exactly once: nothing lost, nothing
duplicated, and the record for an example is identical whichever session
produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.checkpointing import (
    JsonlAppender,
    RunState,
    build_resume_command,
    completed_uids,
    load_records,
    make_uid,
    verify_no_duplicates,
)
from src.config import RunConfig, load_config
from src.runner import plan_examples, run

#: Tests must not depend on the working directory.
BASE_CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")


# --------------------------------------------------------------------- uid rules
def test_make_uid_is_deterministic_and_component_sensitive():
    base = dict(
        model="m",
        strategy="s",
        dataset="d",
        index=3,
        seed=0,
        config_hash="abc",
    )
    uid = make_uid(**base)
    assert uid == make_uid(**base), "uid must be stable across calls"
    assert len(uid) == 16

    for field, changed in [
        ("model", "m2"),
        ("strategy", "s2"),
        ("dataset", "d2"),
        ("index", 4),
        ("seed", 1),
        ("config_hash", "abd"),
    ]:
        other = dict(base)
        other[field] = changed
        assert make_uid(**other) != uid, f"uid must change when {field} changes"


def test_make_uid_ignores_int_str_ambiguity():
    a = make_uid(model="m", strategy="s", dataset="d", index=3, seed=0, config_hash="h")
    b = make_uid(model="m", strategy="s", dataset="d", index="3", seed="0", config_hash="h")
    assert a == b, "index/seed are coerced to int, so 3 and '3' must agree"


# ------------------------------------------------------------------ jsonl writer
def test_appender_writes_and_survives_truncation(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    with JsonlAppender(path, flush_every=2) as w:
        for i in range(5):
            w.write({"uid": f"u{i}", "value": i})
    assert len(load_records(path)) == 5

    # Emulate an abrupt kill in the middle of writing line 6.
    with path.open("a", encoding="utf-8") as f:
        f.write('{"uid": "u5", "value": 5')

    records = load_records(path)
    assert len(records) == 5, "a truncated final line must be skipped, not crash"
    assert [r["uid"] for r in records] == [f"u{i}" for i in range(5)]

    # Appending after the damaged line still works and stays readable.
    with JsonlAppender(path, flush_every=1) as w:
        w.write({"uid": "u6", "value": 6})
    uids = [r["uid"] for r in load_records(path)]
    assert "u6" in uids and "u5" not in uids


def test_load_records_dedupes_keeping_last(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    with JsonlAppender(path, flush_every=1) as w:
        w.write({"uid": "a", "v": 1})
        w.write({"uid": "b", "v": 2})
        w.write({"uid": "a", "v": 99})
    records = load_records(path, dedupe=True)
    assert len(records) == 2
    assert {r["uid"]: r["v"] for r in records} == {"a": 99, "b": 2}
    assert load_records(path, dedupe=False).__len__() == 3
    assert verify_no_duplicates(path) == {"a": 2}


def test_completed_uids_excludes_errors_by_default(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    with JsonlAppender(path, flush_every=1) as w:
        w.write({"uid": "ok", "error": None})
        w.write({"uid": "bad", "error": "RuntimeError: boom"})
    assert completed_uids(path) == {"ok"}, "failed examples are retried next session"
    assert completed_uids(path, include_errors=True) == {"ok", "bad"}


def test_run_state_roundtrip_and_sessions(tmp_path: Path):
    path = tmp_path / "manifest.json"
    state = RunState(run_name="r", config_hash="h", config={"a": 1}, n_total=10)
    sess = state.begin_session(hardware={"n_gpus": 2}, backend={"backend": "mock"})
    sess.n_completed = 4
    sess.elapsed_seconds = 12.5
    sess.gpu_hours = 0.5
    state.end_session(sess)
    state.n_completed = 4
    state.save(path)

    loaded = RunState.load(path)
    assert loaded is not None
    assert loaded.n_total == 10 and loaded.n_completed == 4
    assert loaded.n_remaining == 6 and not loaded.is_complete
    assert loaded.elapsed_seconds == pytest.approx(12.5)
    assert loaded.gpu_hours == pytest.approx(0.5)
    assert loaded.sessions[0]["hardware"]["n_gpus"] == 2
    assert RunState.load(tmp_path / "missing.json") is None


def test_config_drift_reports_only_changed_paths():
    state = RunState(config={"model": {"backend": "hf", "dtype": "float16"}})
    drift = state.config_drift({"model": {"backend": "vllm", "dtype": "float16"}})
    assert any("backend" in d for d in drift)
    assert not any("dtype" in d for d in drift)


def test_build_resume_command():
    cmd = build_resume_command("configs/x.yaml", ["runtime.seed=1", "data.subsample=50"])
    assert cmd.startswith("python -m src.runner --config configs/x.yaml")
    assert "--set runtime.seed=1" in cmd and "--set data.subsample=50" in cmd


# --------------------------------------------------------- end-to-end resumption
def _mock_cfg(tmp_path: Path, **runtime: object) -> RunConfig:
    """A toy-dataset, mock-backend config writing into tmp_path."""
    overrides = [
        "model.backend=mock",
        "model.name_or_path=mock/tiny",
        "data.name=toy",
        "data.subsample=null",
        "data.subset=null",
        "generation.max_new_tokens=64",
        "generation.batch_size=4",
        "strategy.name=cot_zeroshot",
        f"runtime.out_dir={tmp_path.as_posix()}",
        "runtime.grade_after_run=false",
        "runtime.log_every=100",
        "runtime.time_budget_hours=1.0",
    ]
    overrides += [f"runtime.{k}={v}" for k, v in runtime.items()]
    return load_config(BASE_CONFIG, overrides)


def test_resume_completes_every_example_exactly_once(tmp_path: Path):
    """Kill and restart repeatedly; assert no lost and no duplicated work."""
    cfg = _mock_cfg(tmp_path, max_examples=5)
    examples, all_uids = plan_examples(cfg)
    n_total = len(examples)
    assert n_total > 10, "the toy dataset should have enough items to interrupt"

    first = run(cfg, config_path=BASE_CONFIG)
    assert first.n_completed_this_session == 5
    assert first.n_remaining == n_total - 5
    assert not first.is_complete
    assert first.stop_reason in {"completed", "time_budget"}
    assert "python -m src.runner" in first.resume_command

    # Emulate an abrupt kill: a half-written final line, as fsync cannot prevent
    # a partial record if the machine dies mid-write.
    with cfg.results_path.open("a", encoding="utf-8") as f:
        f.write('{"uid": "torn", "final_answer": "4')

    # Second session with a smaller cap, to exercise repeated resumption.
    second = run(_mock_cfg(tmp_path, max_examples=5), config_path=BASE_CONFIG)
    assert second.n_completed_this_session == 5
    assert second.n_completed_total == 10

    # Final session: no cap, should finish the run.
    final = run(_mock_cfg(tmp_path), config_path=BASE_CONFIG)
    assert final.is_complete
    assert final.n_completed_total == n_total
    assert final.n_remaining == 0

    raw = load_records(cfg.results_path, dedupe=False)
    assert verify_no_duplicates(cfg.results_path) == {}, "no example may run twice"
    assert len(raw) == n_total, "no example may be lost or repeated"
    assert {r["uid"] for r in raw} == set(all_uids)
    assert not any(r.get("error") for r in raw)

    state = RunState.load(cfg.manifest_path)
    assert state is not None
    assert state.n_completed == n_total
    assert state.status == "complete"
    assert len(state.sessions) == 3, "each session must be recorded in the manifest"


def test_resume_is_a_noop_when_already_complete(tmp_path: Path):
    cfg = _mock_cfg(tmp_path)
    first = run(cfg)
    again = run(_mock_cfg(tmp_path))
    assert again.n_completed_this_session == 0
    assert again.stop_reason == "nothing_to_do"
    assert again.is_complete
    assert again.n_completed_total == first.n_completed_total
    assert len(load_records(cfg.results_path, dedupe=False)) == first.n_total


def test_records_are_identical_across_sessions(tmp_path: Path):
    """The same example must produce the same record in any session."""
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    run(_mock_cfg(a_dir, max_examples=3))
    run(_mock_cfg(b_dir))

    a = {r["uid"]: r for r in load_records(_mock_cfg(a_dir).results_path)}
    b = {r["uid"]: r for r in load_records(_mock_cfg(b_dir).results_path)}
    shared = set(a) & set(b)
    assert len(shared) == 3
    volatile = {"finished_at", "latency_s"}
    for uid in shared:
        ra = {k: v for k, v in a[uid].items() if k not in volatile}
        rb = {k: v for k, v in b[uid].items() if k not in volatile}
        assert ra == rb, f"record for {uid} differs between sessions"


def test_changing_a_semantic_field_starts_a_new_run(tmp_path: Path):
    """A different temperature is a different measurement, so a different dir."""
    cfg_a = _mock_cfg(tmp_path)
    cfg_b = load_config(
        BASE_CONFIG,
        [
            "model.backend=mock",
            "model.name_or_path=mock/tiny",
            "data.name=toy",
            "data.subsample=null",
            "data.subset=null",
            "generation.max_new_tokens=64",
            "generation.temperature=0.9",
            "strategy.name=cot_zeroshot",
            f"runtime.out_dir={tmp_path.as_posix()}",
            "runtime.grade_after_run=false",
        ],
    )
    assert cfg_a.config_hash != cfg_b.config_hash
    assert cfg_a.run_dir != cfg_b.run_dir


def test_changing_a_perf_field_resumes_the_same_run(tmp_path: Path):
    """Switching batch size or backend must not orphan completed work."""
    cfg_a = _mock_cfg(tmp_path)
    cfg_b = _mock_cfg(tmp_path)
    cfg_b.generation.batch_size = cfg_a.generation.batch_size * 2
    cfg_b.model.device_map = "cuda:0"
    cfg_b.model.tensor_parallel_size = 2
    cfg_b.runtime.time_budget_hours = 2.0
    assert cfg_a.config_hash == cfg_b.config_hash
    assert cfg_a.run_dir == cfg_b.run_dir


def test_time_guard_stops_cleanly_mid_run(tmp_path: Path):
    """A tiny budget must stop at an example boundary and leave a resumable run."""
    cfg = _mock_cfg(tmp_path)
    cfg.model.mock_latency_s = 0.1
    cfg.runtime.time_budget_hours = 1.0 / 3600.0  # one second
    cfg.runtime.reserve_minutes = 0.0

    summary = run(cfg)
    assert summary.stop_reason == "time_budget"
    assert 0 < summary.n_completed_total < summary.n_total
    assert summary.n_remaining > 0
    assert not summary.is_complete
    assert "python -m src.runner" in summary.resume_command

    state = RunState.load(cfg.manifest_path)
    assert state is not None and state.status.startswith("stopped_")
    assert state.n_completed == summary.n_completed_total

    # And the run is resumable to completion with a real budget.
    cfg2 = _mock_cfg(tmp_path)
    final = run(cfg2)
    assert final.is_complete
    assert verify_no_duplicates(cfg.results_path) == {}
