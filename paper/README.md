# Building the paper

`main.tex` compiles standalone with the stock `article` class and stock CTAN
packages. No conference style file is required, and none is vendored here.

## Zero-install option: Overleaf

1. Zip the `paper/` directory (`main.tex`, `references.bib`, `figures/`).
2. On Overleaf choose **New Project → Upload Project** and pick the zip.
3. Set the main document to `main.tex` (Menu → Main document).
4. Set the compiler to **pdfLaTeX** (Menu → Compiler). The document uses
   `inputenc`/`fontenc`, so pdfLaTeX is correct; XeLaTeX and LuaLaTeX will also
   work but will warn about `inputenc`.
5. Overleaf runs BibTeX automatically. If citations render as `[?]`, hit
   Recompile once more — the first pass has no `.bbl` yet.

## Local build

The dependency-free way, which is what CI runs:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`latexmk` figures out how many passes are needed and when to call BibTeX. If
you do not have it, the explicit sequence is:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex   # writes main.aux with \citation lines
bibtex   main                                # resolves them against references.bib
pdflatex -interaction=nonstopmode main.tex   # pulls in main.bbl
pdflatex -interaction=nonstopmode main.tex   # settles cross-references
```

Three `pdflatex` passes are genuinely needed: the first collects citation keys,
the second inserts the bibliography (which shifts page numbers), and the third
settles `\ref` and `\cite` numbering. Stopping at two leaves stale numbers.

To clean up: `latexmk -C`, or delete `*.aux *.bbl *.blg *.log *.out *.fdb_latexmk *.fls`.

### Required TeX packages

All are in TeX Live's `scheme-medium` or MiKTeX's default set: `geometry`,
`inputenc`, `fontenc`, `times` (psnfss), `amsmath`, `amssymb`, `amsthm`,
`graphicx`, `booktabs`, `multirow`, `xcolor`, `url`, `natbib`, `hyperref`,
`algorithm` (float), `algpseudocode` (algorithmicx).

On Ubuntu: `sudo apt-get install texlive-latex-recommended texlive-latex-extra
texlive-fonts-recommended texlive-science latexmk`.

## Swapping in a conference style

The header comment block at the top of `main.tex` carries the per-venue
instructions. The general recipe:

1. Download the style file from the venue's author kit into `paper/`
   (`neurips_2026.sty`, `iclr2027_conference.sty` + `.bst`, `icml2026.sty`, or
   `acl.sty` + `acl_natbib.bst`).
2. Edit the preamble as directed in the header comment.
3. **Delete the `\usepackage[margin=1in]{geometry}` line.** Every conference
   style sets its own page geometry, and leaving `geometry` in will silently
   override it and produce a non-compliant layout — this is the most common way
   to get desk-rejected on format.
4. ACL/EMNLP additionally want `\bibliographystyle{acl_natbib}` instead of
   `plainnat`, and their style loads `natbib` itself, so drop our
   `\usepackage[numbers,sort&compress]{natbib}` line to avoid an option clash.
5. Everything from `\begin{document}` onward is portable and needs no change.

## Figures

`\includegraphics` is not yet used anywhere: the Results and Analysis sections
carry `\todo` markers naming the artefacts instead. The analysis pipeline emits
them into `paper/figures/` under the exact filenames listed in
`docs/METHOD_SPEC.md` §9 (`fig1_variance_components.pdf` through
`fig8_precision_control.pdf`, and `tab1_setup.csv` through
`tab7_precision_control.csv`). When a
figure lands, replace the corresponding `\todo` with:

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig1_variance_components.pdf}
  \caption{...}
  \label{fig:variance}
\end{figure}
```

Keep the filenames byte-identical to the spec; the `\todo` markers are the
contract between this document and the harness.

## Continuous integration

`.github/workflows/paper.yml` builds `main.tex` on every push or pull request
that touches `paper/`, and uploads the resulting PDF as a build artefact. It
fails the build on LaTeX errors and prints any undefined-citation or
undefined-reference warnings, which is the check that matters most here — a
missing bib key is silent in the PDF but fatal at submission.
