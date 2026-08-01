"""Pytest configuration: make the repo root importable and keep tests offline.

Tests must never touch the network or a GPU. Forcing the HF offline flags here
means an accidental `load_dataset` in a test fails fast and loudly instead of
silently downloading several GB on someone's laptop or in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Opt-in escape hatch for the (skipped by default) network tests.
if not os.environ.get("HARNESS_ALLOW_NETWORK"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Headless plotting; must be set before matplotlib.pyplot is imported anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")


import pytest  # noqa: E402  (import after sys.path surgery)


@pytest.fixture
def toy_examples():
    """A few inline Examples covering each answer type, no dataset dependency."""
    from src.types import Example

    return [
        Example(
            id="fixture/0",
            question="What is 12 + 30?",
            gold_answer="42",
            answer_type="math",
        ),
        Example(
            id="fixture/1",
            question="What is one half of 5?",
            gold_answer="2.5",
            answer_type="math",
        ),
        Example(
            id="fixture/2",
            question="Which planet is closest to the Sun?",
            gold_answer="B",
            choices=["Venus", "Mercury", "Mars", "Earth"],
            answer_type="mc",
        ),
        Example(
            id="fixture/3",
            question="Is 7 a prime number?",
            gold_answer="True",
            answer_type="bool",
        ),
    ]


@pytest.fixture
def mock_backend(toy_examples):
    """A deterministic MockBackend that knows the fixture gold answers."""
    from src.generation import MockBackend

    backend = MockBackend(batch_size=4, accuracy=0.75, seed=7)
    backend.register_golds(toy_examples)
    return backend
