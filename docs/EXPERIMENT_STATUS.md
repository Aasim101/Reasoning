# Experiment status (19 Oct 2026)

Target venue: **ICML 2027** (abstract ~mid-Jan 2027, paper ~late Jan 2027 AoE — confirm on official CFP).

## Corpus inventory

| Metric | Value |
|---|---|
| Run directories | **66** |
| Complete | **66 / 66** |
| Graded | **66 / 66** |
| Graded JSONL size | ~572 MB |
| Reported GPU-hours (sum of manifests) | ~342 h (inflated by session accounting; wall-clock on Kaggle was multi-week) |

## Checklist vs plan

| Block | Status |
|---|---|
| Phase 1: 1.5B / 3B × GSM8K+MATH × c0–c2 | **DONE** |
| Phase 1: 7B × MATH × c0–c2 | **DONE** |
| Phase 1: 7B × GSM8K × c1,c2 | **DONE** |
| Phase 1: **7B x GSM8K x c0 seed0** | **DONE** (maj@24 = 94.5%) |
| Phase 2: c0 seeds 1,2 for 1.5/3/7B × GSM8K+MATH | **DONE** |
| E3a: Llama-3.2-3B × GSM8K+MATH × c0–c2 | **DONE** |
| E4: GSM-Symbolic main+p2 × 3 Qwen × c0–c2 | **DONE** |
| PC: 4-bit 1.5B+3B MATH × c0–c2 | **DONE** |
| O2: 3B MATH N=64 × c0–c3 | **DONE** |
| O2b: 3B MATH c1 seeds 1,2 | **DONE** |

## Only missing GPU experiment

**None.** The 7B x GSM8K x c0 x seed0 cell is in the bundle (maj@24 = 94.5%).

## OverflowError during runs

Cause: `human_time()` did `int(seconds)` when an ETA was non-finite → `OverflowError: cannot convert float infinity to integer`. Caught by the runner’s per-example error path, so the run continued.

**Fixed** in `src/utils.py` (`human_time` now returns `"?"` on NaN/Inf). Harmless for completed data; re-download the fixed `src/utils.py` before future sessions.

## Analysis

- `scripts/quick_summary.py` → `paper_artefacts/quick_summary.json` (done; draft numbers).
- **Split first** (required for F8/T7). On Kaggle Linux the old Windows-only `mklink` path produced `fp16=0 4bit=0` — use the updated `scripts/split_precision.py`:

```bash
python scripts/split_precision.py --src results_bundle
# expect fp16≈61 4bit≈6; if both 0, wrong --src / cwd
```

- Full `python -m src.analysis.paper ...` produces F1–F8 / T1–T7. Use:

```bash
python -u -m src.analysis.paper \
  --results-dir results_fp16 \
  --precision-results-dir results_4bit \
  --out-dir paper_artefacts \
  --n-bootstrap 1000 \
  --n-repeats 50 \
  --no-png
```

Bypass (no precision figs): `--results-dir results_bundle` and omit `--precision-results-dir`.
For camera-ready, re-run with `--n-bootstrap 10000`.

## Preliminary maj@24 accuracies (seed0, fp16)

| Model | Dataset | c0 | c1 | c2 | range |
|---|---|---:|---:|---:|---:|
| 1.5B | GSM8K | 75.0 | 76.0 | 68.5 | **7.5 pp** |
| 1.5B | MATH-500 | 56.5 | 54.0 | 55.5 | 2.5 pp |
| 3B | GSM8K | 91.0 | 90.5 | 91.0 | 0.5 pp |
| 3B | MATH-500 | 74.5 | 74.0 | 71.5 | 3.0 pp |
| 7B | GSM8K | 94.5 | 93.5 | 88.0 | **6.5 pp** |
| 7B | MATH-500 | 81.0 | 80.5 | 81.5 | 1.0 pp |
| Llama-3.2-3B | GSM8K | 89.5 | 88.5 | 87.0 | 2.5 pp |
| Llama-3.2-3B | MATH-500 | 59.0 | 59.5 | 58.5 | 1.0 pp |

GSM-Symbolic shows larger spreads (e.g. 7B main: 88→74 = **14 pp**). 4-bit PC drops MATH-500 accuracy ~6–14 pp vs fp16 (expected; gate P7 still needs GLMM share comparison from full analysis).
