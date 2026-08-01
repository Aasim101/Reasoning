"""Dataset loader tests. Offline: only the built-in `toy` set is exercised.

The network-dependent loaders are covered by `--dry-run` on a real machine; here
we assert the parts that must hold regardless of the upstream data: registry
hygiene, deterministic subsampling, sharding, and the offline JSONL round trip.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import DataConfig
from src.datasets_ import (
    DATASET_REGISTRY,
    available_datasets,
    cache_dir_for,
    dataset_info,
    load_dataset_examples,
    load_examples_jsonl,
    save_examples_jsonl,
    shard,
    subsample,
)
from src.types import ANSWER_TYPES

REQUIRES_NETWORK = pytest.mark.skipif(
    not os.environ.get("HARNESS_ALLOW_NETWORK"),
    reason="set HARNESS_ALLOW_NETWORK=1 to run loaders that download data",
)


def toy_cfg(**kw: object) -> DataConfig:
    defaults = dict(name="toy", split="test", subset=None, subsample=None)
    defaults.update(kw)
    return DataConfig(**defaults)  # type: ignore[arg-type]


# ------------------------------------------------------------------- registry
def test_expected_datasets_are_registered():
    names = available_datasets()
    for expected in (
        "toy",
        "gsm8k",
        "math500",
        "aime",
        "bbh",
        "musr",
        "arc_challenge",
        "gpqa_diamond",
    ):
        assert expected in names


def test_every_spec_is_well_formed():
    for name, spec in DATASET_REGISTRY.items():
        assert spec.name == name
        assert spec.hf_id, f"{name} has no source id"
        assert spec.default_split, f"{name} has no default split"
        assert spec.answer_type in ANSWER_TYPES
        assert isinstance(spec.verified, bool)
        assert spec.loader is not None
        # An unverified id must say why, so nobody trusts it by accident.
        if not spec.verified:
            assert spec.notes, f"{name} is unverified but undocumented"


def test_dataset_info_round_trips():
    info = dataset_info("gsm8k")
    assert info["hf_id"] == "openai/gsm8k"
    assert info["default_subset"] == "main"
    assert info["default_split"] == "test"
    assert info["verified"] is True
    with pytest.raises(KeyError):
        dataset_info("does_not_exist")


def test_gated_dataset_is_flagged():
    assert DATASET_REGISTRY["gpqa_diamond"].gated is True


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        load_dataset_examples(toy_cfg(name="nope"))


# ---------------------------------------------------------------- toy loader
def test_toy_loader_schema():
    examples = load_dataset_examples(toy_cfg())
    assert len(examples) >= 20
    assert len({e.id for e in examples}) == len(examples), "ids must be unique"
    types = {e.answer_type for e in examples}
    assert {"math", "mc", "bool"} <= types, "toy set must exercise several answer types"
    for i, ex in enumerate(examples):
        assert ex.question.strip()
        assert str(ex.gold_answer).strip()
        assert ex.answer_type in ANSWER_TYPES
        assert ex.meta["orig_index"] == i
        if ex.answer_type == "mc":
            assert ex.choices and len(ex.choices) >= 2
            assert ex.gold_answer in ex.choice_letters


def test_toy_loader_needs_no_network_or_datasets_package():
    # conftest sets HF_HUB_OFFLINE, so a hidden download attempt would fail here.
    assert load_dataset_examples(toy_cfg())


# --------------------------------------------------------------- subsampling
def test_subsample_is_deterministic_and_a_subset():
    full = load_dataset_examples(toy_cfg())
    a = load_dataset_examples(toy_cfg(subsample=8, subsample_seed=1))
    b = load_dataset_examples(toy_cfg(subsample=8, subsample_seed=1))
    c = load_dataset_examples(toy_cfg(subsample=8, subsample_seed=2))

    assert len(a) == 8
    assert [e.id for e in a] == [e.id for e in b], "same seed must give the same sample"
    assert [e.id for e in a] != [e.id for e in c], "different seed must differ"
    assert {e.id for e in a} <= {e.id for e in full}


def test_subsample_preserves_dataset_order_and_records_provenance():
    sampled = load_dataset_examples(toy_cfg(subsample=6, subsample_seed=7))
    indices = [e.meta["orig_index"] for e in sampled]
    assert indices == sorted(indices), "iteration order must follow the dataset"
    for ex in sampled:
        assert ex.meta["subsample_seed"] == 7
        assert ex.meta["subsample_n"] == 6


def test_subsample_no_op_cases():
    full = load_dataset_examples(toy_cfg())
    assert len(subsample(full, None, 0)) == len(full)
    assert len(subsample(full, 10**6, 0)) == len(full)
    assert len(subsample(full, 0, 0)) == len(full)


# ------------------------------------------------------------------- sharding
def test_shard_partitions_without_overlap_or_loss():
    full = load_dataset_examples(toy_cfg())
    shards = [shard(list(full), i, 3) for i in range(3)]
    ids = [e.id for s in shards for e in s]
    assert len(ids) == len(full)
    assert set(ids) == {e.id for e in full}
    assert len(set(ids)) == len(ids), "no example may appear in two shards"
    for i, s in enumerate(shards):
        for ex in s:
            assert ex.meta["shard_index"] == i
            assert ex.meta["num_shards"] == 3


def test_shard_single_is_identity_and_bad_index_raises():
    full = load_dataset_examples(toy_cfg())
    assert [e.id for e in shard(list(full), 0, 1)] == [e.id for e in full]
    with pytest.raises(ValueError):
        shard(list(full), 3, 3)


def test_load_dataset_examples_applies_subsample_then_shard():
    examples = load_dataset_examples(
        toy_cfg(subsample=12, subsample_seed=3, shard_index=1, num_shards=2)
    )
    assert len(examples) == 6
    assert all(e.meta["subsample_n"] == 12 for e in examples)
    assert all(e.meta["shard_index"] == 1 for e in examples)


# --------------------------------------------------------- offline round trip
def test_jsonl_round_trip(tmp_path: Path):
    examples = load_dataset_examples(toy_cfg())
    path = save_examples_jsonl(examples, tmp_path / "examples.jsonl")
    assert path.exists()
    restored = load_examples_jsonl(path)
    assert [e.to_dict() for e in restored] == [e.to_dict() for e in examples]


def test_local_dir_loads_prefetched_jsonl(tmp_path: Path):
    """The offline path must yield the same Examples as the online one."""
    examples = load_dataset_examples(toy_cfg())
    cache = cache_dir_for("gsm8k", tmp_path)
    cache.mkdir(parents=True, exist_ok=True)
    save_examples_jsonl(examples, cache / "examples.jsonl")

    loaded = load_dataset_examples(
        DataConfig(name="gsm8k", split="test", subset="main", subsample=None,
                   local_dir=str(cache))
    )
    assert [e.id for e in loaded] == [e.id for e in examples]
    assert [e.gold_answer for e in loaded] == [e.gold_answer for e in examples]


def test_cache_dir_for_sanitises_names(tmp_path: Path):
    assert cache_dir_for("gsm8k", tmp_path).name == "gsm8k"
    assert "/" not in cache_dir_for("some/name", tmp_path).name


# ------------------------------------------------------ network-only smoke test
@REQUIRES_NETWORK
def test_gsm8k_downloads_and_parses():  # pragma: no cover - opt-in only
    examples = load_dataset_examples(
        DataConfig(name="gsm8k", split="test", subset="main", subsample=5)
    )
    assert len(examples) == 5
    assert all(e.gold_answer and "####" not in e.gold_answer for e in examples)
