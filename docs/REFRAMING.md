# Reframing Document (post-adversarial review)

**Date:** 29 July 2026  
**Trigger:** `docs/ADVERSARIAL_REVIEW.md` — verified overlap with BrittleBench (arXiv:2603.13285), TEE (arXiv:2604.11581), and arXiv:2606.19636.  
**Status:** binding. Do not run experiments under the old headline claim.

---

## 1. Old claim → new claim

### Old headline (retired)

> Per-item reasoning difficulty is substantially a model × elicitation-configuration interaction; we decompose variance of per-item success rate and show prompt-attributable share exceeds item×model share.

**Why it is retired.** BrittleBench (March 2026) already decomposes performance variance into task difficulty vs. semantics-preserving prompt sensitivity on an item × perturbation matrix and reports the prompt share at roughly **half**. TEE (April 2026) fits crossed REML with an explicit item × prompt interaction. arXiv:2606.19636 establishes that sampling-derived difficulty labels are unstable and corrupt RL filters and curricula. Competing on “item difficulty is substantially prompt-attributable” is a reject, not a contribution.

### New headline (binding)

> **The self-consistency ceiling (π_mode) is not a model constant — it is configuration-relative.** Configuration changes reorder the top competing answer classes on small-margin items; spreading a fixed token budget across configurations raises the modal plateau that i.i.d. resampling structurally cannot move.

### What survives as the delta

| Established premise (cite as foundation) | Our extension |
|---|---|
| BrittleBench, TEE: per-item difficulty is prompt-attributable; variance decomposes across item and prompt facets | **Stochastic generation:** inference variance does not vanish; binomial noise must be separated from interaction (GLMM, not moment subtraction) |
| FormatSpread, Mizrahi et al., PromptEval: aggregate / template-level prompt sensitivity | **Modes:** nobody measures whether sensitivity reaches the **modal answer**, where the selection bottleneck lives |
| Bay & Yearick, Brown et al.: self-consistency plateaus at π_mode | **Configuration axis untested:** Bay & Yearick never vary elicitation; π_mode may be a model–prompt property |
| arXiv:2606.19636: unstable difficulty labels break downstream consumers | **Intervention:** CDV exploits configuration diversity at zero extra GPU cost; we decompose gain into new modal mass vs. configuration selection |
| TEE: item × prompt inflates aggregate CI (averages away at benchmark scale) | **Per-item estimand:** adaptive TTC, curricula, and hard subsets consume difficulty **one item at a time** — it does not average away |

### Estimand shift

- **Was:** aggregate variance share of item × configuration (competing with BrittleBench).
- **Is:** per-item π_mode(c), reorder rate, CDV plateau vs. single-config SC plateau — the objects downstream methods actually consume.

---

## 2. Artefact reorder (§4.3 of adversarial review)

Same corpus; different narrative order. **Lead figures move up; variance decomposition becomes supporting measurement.**

| Priority | Artefact | Role under new framing |
|---|---|---|
| **Primary** | **F4** (`fig4_ceiling_vs_coverage.pdf`) | π_mode varies across configurations; identifiability gap is configuration-relative |
| **Primary** | **F5** (`fig5_cdv_vs_sc.pdf`) | CDV plateau vs. SC plateau at matched tokens; primary test of mechanism |
| **Primary** | **F6** (`fig6_reordering.pdf`) | Corrective vs. destructive mode transitions; mechanism figure |
| Supporting | F1, T3 | GLMM variance components (item × configuration share); calibrated against parametric-bootstrap null |
| Supporting | F2, F3, T4 | Disattenuated transfer and hard-subset stability |
| Secondary | F7 | GSM-Symbolic distribution shift (GSM8K demoted) |
| Robustness | F8, T7 | Precision control |

### Results section order in `paper/main.tex`

1. Modal ceilings across configurations (F4) — **was §5.3**
2. Configuration-diversified voting (F5) — **was §5.4**
3. Mode reordering mechanism (F6) — **was Analysis**
4. Variance decomposition and transfer (F1–F3) — **was §5.1–5.2**
5. Downstream applied consequence (T6) — demote curriculum-RL to Discussion implication

---

## 3. Prediction and falsification rewrite

Each condition is **individually decisive for its sub-claim** (not “all three must hold”).

| ID | Sub-claim | Condition | Primary cell |
|---|---|---|---|
| **F3** | Mechanism / intervention | Plug-in π_mode(mixture) ≤ π_mode(c0) at matched budget **and** maj@n curves do not separate | 3B × MATH-500 |
| **F4** | π_mode is configuration-relative | max_c π_mode − min_c π_mode ≤ 3 pp | 3B × MATH-500 |
| **F1** | Item × configuration variance non-trivial | GLMM item×config share ≤ 5% **or** not above parametric-bootstrap null 95th percentile | 3B × MATH-500 |
| **F2** | Hard subset unstable beyond noise | J_config ≥ J_seed − 0.02 (relative; not absolute J ≥ 0.90) | 3B × MATH-500 |

**P4, P5 are primary pre-registered predictions.** P1 is secondary (supporting measurement). P2, P3 supporting. P6 applied. P7 precision gate unchanged.

**F3 primary test:** pre-register **plug-in π_mode comparison** (π_mode from N=24 at c0 vs. π_mode from pooled 3×24 at CDV) as primary; token-matched maj@n curves as secondary (small-budget regime explicitly labelled).

**Multiplicity:** one primary cell per prediction family; Holm–Bonferroni within declared families; secondary cells reported descriptively with aggregation rule stated.

**5–10% dead zone (F1/P1):** if GLMM item×config share ∈ [5%, 10%], report as “small but non-zero; below BrittleBench-scale prompt share on MC benchmarks; consistent with MATH-500 behaving more like MathQA/ARC (difficulty-dominated).”

---

## 4. Experiment matrix changes (summary)

See `docs/METHOD_SPEC.md` §8 for normative detail.

| Change | Rationale | GPU effect |
|---|---|---|
| **Drop c5** from core factor | Circular for mode-reordering claim; already in B4 | **≈ −1.7 h** |
| **Core = c0, c1, c2** | Semantics-preserving axes only; no parse/format or few-shot confounds in headline grid | Main grid 6 → 3 configs |
| **Demote c3, c4/c4a/c4b** to separate axes | Parse confound (c3); task-information confound (c4) | 0 h if not generated in Tier A |
| **Redirect O2** | 3B, MATH-500, c0–c3, N: 24→64 (not 7B c0-only) | ≈ 0 h net |
| **Add O2b** | 2 seed replicates at c1, 3B, MATH-500 | **≈ +0.5 h** |
| **Demote GSM8K** to secondary replicate | Saturation + contamination | 0 h (analysis) |
| **Lead MATH-500** in prose and primary cells | Right benchmark for decomposition + plateau | 0 h |

**Revised tier totals (estimated):** Tier A ≈ **12.5 h** (was 18.8 h); Tier B ≈ **16.1 h** (was 22.6 h); two-week paper-critical total ≈ **28.6 h** + pilots/slack ≈ **32 h** (was ≈ 46 h).

---

## 5. Analysis changes (zero GPU)

- **GLMM primary;** moment decomposition → appendix robustness.
- **Parametric-bootstrap null** on synthetic data before any GPU run.
- **Interaction-specific permutation test** (centre per configuration before permuting).
- **Oracle-best-single-configuration baseline** (B12) + CDV gain decomposition (new modal mass vs. selection).
- **Extraction audit:** two extractors, stratified manual audit, ⊥-dropped and integer-only sensitivities.
- **Truncation controls:** log `finish_reason`, rendered prompt, token IDs; report per-cell truncation rates.
- **Define π_mode self-containedly** in paper; reduce Bay & Yearick to one gap-statement citation.
- **IRT comparison** pre-registered; cite PromptEval as antecedent.

---

## 6. Citation obligations

Must cite as foundations (not competitors): BrittleBench, TEE, arXiv:2606.19636, arXiv:2607.13304, CyclicJudge, PromptEval, Mizrahi et al. TACL 2024, Alzahrani et al. ACL 2024.

**Open:** `wang2025measuring` (BrittleBench internal cite) — unresolved; see `docs/CITATION_AUDIT.md`.

**Verified this pass:** arXiv:2606.28661 (Bay & Yearick) exists; authors Bay & Yearick; preprint status unchanged.
