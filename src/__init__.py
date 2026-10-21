"""Method-agnostic experimental harness for LLM reasoning research.

Layout
------
`config`        dataclass config, YAML load, CLI override, semantic run hashing
`types`         the shared data contracts (Example, Completion, StrategyResult, ...)
`datasets_`     benchmark loaders in one unified schema
`prompts`       prompt construction shared by all strategies
`generation`    the GenerationBackend interface + MockBackend (CPU, model-free)
`models`        hardware detection, HF transformers backend, optional vLLM backend
`strategies`    pluggable reasoning strategies (the extension point for a method)
`answers`       answer extraction + equivalence checking
`grading`       separate, cached grading pass over raw generations
`metrics`       accuracy, pass@k, bootstrap CIs, paired significance tests
`runner`        the experiment driver (resumable, time-budgeted)
`checkpointing` deterministic result ids, append-only JSONL, run manifest
`budget`        wall-clock guard and GPU-hour accounting
`analysis`      results -> LaTeX tables and matplotlib figures

Nothing here implements a specific novel reasoning method: a method is a single
`ReasoningStrategy` subclass registered by name (see `strategies/base.py`).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
