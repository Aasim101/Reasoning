"""Offline analysis: results JSONL -> figures, LaTeX tables, summary files.

Nothing here touches a GPU or the run loop, which is the point: plotting inside a
run loop means a matplotlib error can destroy hours of generation, and it means
you cannot re-style a figure without re-running the experiment.

Entry point:

    python -m src.analysis.aggregate --results-dir results --out-dir paper_assets \\
        --figures --tables
"""

from __future__ import annotations

__all__ = ["aggregate", "figures", "tables"]
