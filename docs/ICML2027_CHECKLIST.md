# ICML 2027 bulletproof checklist

**Estimated deadlines (confirm on icml.cc CFP):** abstract ~16–23 Jan 2027 AoE; full paper ~22–29 Jan 2027 AoE. Location: South America.

## Must do before submission

### A. One missing GPU run (~2–4 h)
- [x] `Qwen2.5-7B` × GSM8K × `configured_sampling` × **c0** × **seed 0** × N=24 × T=0.8  
  (Not the same as `gsm_symbolic__...7B...c0` — that is a different benchmark.)

### B. Full analysis (CPU, GPU off)
On Kaggle **with accelerator OFF** (or locally). From the **repo root** (where `results_bundle/` lives):

```bash
# 1) Split precision (symlinks on Linux/Kaggle; junctions/copy on Windows)
#    If you see fp16=0 4bit=0, --src was wrong or cwd was wrong.
python scripts/split_precision.py --src results_bundle
# Expect: fp16=61 4bit=6  (or similar; fp16 must be >0)

# Bypass if split still fails: analyse the whole bundle (skips F8/T7 precision figs)
#   python -u -m src.analysis.paper --results-dir results_bundle --out-dir paper_artefacts --n-bootstrap 200 --no-png

# 2a) Draft artefacts (minutes; skips slow VB-GLMM)
python -u -m src.analysis.paper \
  --results-dir results_fp16 \
  --precision-results-dir results_4bit \
  --out-dir paper_artefacts \
  --n-bootstrap 0 --n-repeats 20 --skip-glmm --no-png

# 2b) Camera-ready (hours; GLMM + bootstrap)
python -u -m src.analysis.paper \
  --results-dir results_fp16 \
  --precision-results-dir results_4bit \
  --out-dir paper_artefacts \
  --n-bootstrap 1000 --n-repeats 50 --no-png
```

**Why `fp16=0` happened:** the old splitter used Windows `mklink` only, so on Kaggle Linux it created empty `results_fp16/` / `results_4bit/`. Pull the updated `scripts/split_precision.py`, delete those empty dirs, re-run with `--src results_bundle` (or `/kaggle/input/.../results_bundle` if the zip is a dataset).

Why a prior full run took 3+ hours: default bootstrap is **10_000**. Use `-u` for live logs. Do **not** enable GPU for this.

### C. Paper finalize after artefacts exist
- [x] Draft `paper_artefacts/` (F1–F8, T1–T7) built with `--skip-glmm --n-bootstrap 0`
- [x] Results/Analysis/Conclusion filled from `tab2` + `predictions.csv` (verified)
- [x] ICML-format `paper/main.tex` (`icml2027.sty` from 2026 kit); anonymous; see `paper/SUBMISSION.md`
- [x] Figures in `paper/figures/`; related work + secondary figs in appendix
- [ ] Camera-ready analysis: no `--skip-glmm`, `--n-bootstrap 1000+`, refresh PDFs
- [ ] Compile (Overleaf); confirm main body ≤ 8 pages
- [ ] Replace style with official ICML 2027 kit when released

### D. Scientific honesty (encoded in `main.tex`)
- [x] P5/F3 CDV negative reported; Q1/Q2 kept where supported
- [x] Do not claim CDV wins
- [x] P7 unsupported in draft (missing 4-bit `ρ_disatt`)
- [x] O2 N=64 vs N=24: loader keeps larger; table footnoted

### E. OverflowError
- [x] Fixed in `src/utils.py` (`human_time` safe on Inf/NaN)
- [ ] Pull fixed file into your Kaggle harness before any new GPU session

## Strength of current corpus

| Claim | Data ready? |
|---|---|
| π_mode varies across configs (Q1) | **Yes** (strong on Symbolic / 1.5B GSM8K) |
| Cross-family (Llama) | **Yes** |
| Seed noise floor (E2) | **Yes** |
| GSM-Symbolic shift (E4) | **Yes** |
| Precision control (PC) | **Yes** |
| Deep-N O2 for plateau | **Yes** |
| CDV raises plateau (Q3) | Data yes; **preliminary result negative** — need F5 |
| 7B GSM8K fully crossed | **Yes** — c0 seed0 present (maj@24 = 94.5%) |

## Recommended narrative for ICML

Lead with **Q1**: reported ceilings are model×prompt properties (Table in Results; Symbolic 10–14 pp).  
Mechanism **Q2** with mode reordering figures.  
**Q3** as a sharp pre-registered test — accept either outcome.  
Cite BrittleBench/TEE as premise, not competitors.
