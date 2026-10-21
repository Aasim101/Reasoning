# ICML submission checklist (this paper)

**Venue:** ICML 2027 (estimated abstract ~22–23 Jan 2027 AoE, paper ~28–29 Jan 2027 AoE — confirm on [icml.cc](https://icml.cc)).  
**Style:** `icml2027.sty` is the official ICML 2026 kit with year strings updated (2027 kit not published as of Oct 2026). When `icml2027.zip` appears on icml.cc, replace `paper/icml2027.sty` / `.bst` with the official files and recompile.

## What is already done

- [x] Full graded corpus (incl. 7B×GSM8K×c0)
- [x] Draft artefacts F1–F8 / T1–T7 (`paper_artefacts/` and `paper/figures/`)
- [x] `main.tex` completed: Setup, Results (numbers from `tab2`), Analysis, Conclusion, Impact, Reproducibility
- [x] Anonymous ICML preamble; no `\todo` / `\placeholder`
- [x] Honest negatives: P5/CDV fails; P1 unsupported; P7 incomplete without 4-bit `ρ_disatt`
- [x] Related work moved to appendix; secondary figures in appendix for page budget

## You must still run / do

### 1) Camera-ready analysis (CPU, hours; no GPU)

From repo root, with `results_bundle` present:

```bash
python scripts/split_precision.py --src results_bundle
# expect fp16≈61 4bit≈6

python -u -m src.analysis.paper \
  --results-dir results_fp16 \
  --precision-results-dir results_4bit \
  --out-dir paper_artefacts \
  --n-bootstrap 1000 \
  --n-repeats 50 \
  --no-png
# omit --skip-glmm for the real GLMM (slow)

cp paper_artefacts/fig*.pdf paper/figures/
```

Then re-check any CI numbers that come from GLMM / bootstrap CIs. Draft claims that only need maj@$N$ / moment shares are already locked to `tab2`/`tab3`.

### 2) Compile PDF (Overleaf recommended)

Zip `paper/` (`main.tex`, `references.bib`, `icml2027.sty`, `icml2027.bst`, `fancyhdr.sty`, `algorithm*.sty`, `figures/`, `_related_appendix.tex`) → Overleaf → pdfLaTeX → compile twice after BibTeX.

**Hard constraint:** main body ≤ **8 pages** (refs + appendix unlimited). If over, shrink floats or move Table~\ref{tab:main-acc} notes.

### 3) Before OpenReview upload

- [ ] Confirm official ICML 2027 CFP dates and swap style if a new kit is released
- [ ] Main body ≤ 8 pages in the compiled PDF
- [ ] No deanonymizing links / repo URLs in the PDF (code as anonymous supplement)
- [ ] Abstract in OpenReview matches the PDF
- [ ] Optional: replace draft artefacts with camera-ready run from (1)

## Number source of truth

All headline accuracies in Table 1 come from `paper_artefacts/tab2_main_accuracy.csv` (seed 0).  
Qwen2.5-3B×MATH-500 uses **N=64** (O2) because the loader keeps the larger duplicate over N=24 — this is footnoted in the table.
