# Building the paper (ICML 2027)

`main.tex` uses the ICML style (`icml2027.sty`, adapted from the official ICML 2026 kit until the 2027 kit is released). See **`SUBMISSION.md`** for the submission checklist and what you still need to run.

## Overleaf (recommended)

1. Zip the `paper/` directory (include `figures/`, `_related_appendix.tex`, `icml2027.sty`, `icml2027.bst`, `fancyhdr.sty`, `algorithm.sty`, `algorithmic.sty`).
2. Upload to Overleaf; set main document to `main.tex`; compiler **pdfLaTeX**.
3. Recompile after BibTeX if citations show `[?]`.

## Local build

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or: `pdflatex` → `bibtex` → `pdflatex` ×2.

## Figures

PDFs live in `figures/` (`fig1_…`–`fig8_…`), copied from `../paper_artefacts/`. After a camera-ready analysis rerun, copy new PDFs into `figures/`.

## Blind vs camera-ready

- Submission (default): `\usepackage{icml2027}`
- Camera-ready: `\usepackage[accepted]{icml2027}` and fill real authors in the `icmlauthorlist` block
