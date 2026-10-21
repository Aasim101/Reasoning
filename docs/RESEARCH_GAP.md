# Research Gap Analysis

**Date of analysis:** 29 July 2026 (revised post-adversarial review)  
**Constraint envelope:** ~25–30 GPU-hours total on Kaggle free tier (2×T4 16 GB, or 1×P100 16 GB), sessions ≤ 9 h, inference-time methods on open models ≤ 8 B fp16 / ≤ 14 B 4-bit. Revised matrix: **Tier A ≈ 12.5 h, Tier B ≈ 16.1 h, paper-critical ≈ 28.6 h (~32 h with pilots)**.  
**Binding reframing:** see `docs/REFRAMING.md`. The headline claim is **not** variance decomposition of item difficulty (BrittleBench, TEE). It is **configuration-relativity of π_mode**.

---

## 0. Why the novelty audit dominated — and what changed in July 2026

LLM reasoning is saturated. A hostile pre-execution review (July 2026) verified three papers that **kill Contribution 1 as originally written**:

| Paper | What it already establishes |
|---|---|
| **BrittleBench** (arXiv:2603.13285) | Item × perturbation variance decomposition; semantics-preserving prompt share ≈ **50%**; persona and paraphrase axes overlap our c1/c6 |
| **TEE** (arXiv:2604.11581) | Crossed REML with item × prompt interaction; D-study; strictly better estimator than method-of-moments |
| **arXiv:2606.19636** | Sampling-derived difficulty labels unstable; 10–29% of “hardest” stratum reachable under alternate regime; breaks RL filters and curricula |

**We do not compete on “prompt-attributable difficulty is large.”** We cite these as **foundations** and ask the question they leave open for **sampled chain-of-thought on free-generation math benchmarks**: does prompt sensitivity reach the **mode** — the object that caps self-consistency — and can spreading budget across configurations **raise** that ceiling?

---

## 1. Candidate gaps considered (historical)

The original audit enumerated ten candidates; six were killed by 2025–2026 literature (see prior version). Candidate **C8** (“is per-item difficulty a model × elicitation interaction?”) was chosen but is now **reframed** as C8′:

| # | Candidate | Verdict |
|---|---|---|
| **C8′** | **Is π_mode configuration-relative, and can configuration-diversified voting raise the plateau i.i.d. resampling cannot?** | **CHOSEN (reframed).** Variance decomposition and transfer statistics are **supporting measurement**, not the headline. |
| C8 (original) | Item difficulty as model × configuration interaction | **Demoted.** BrittleBench + TEE establish the phenomenon at scale (deterministic / aggregate estimands). Our delta is stochastic axis + modes + intervention. |

---

## 2. Chosen contribution

### 2.1 One-sentence claim

> **The self-consistency ceiling (π_mode) is not a model constant — it is configuration-relative.** Semantics-preserving elicitation changes reorder the top competing answer classes on small-margin items; π_mode moves across configurations; spreading a fixed token budget across configurations raises the modal plateau that i.i.d. resampling structurally cannot move.

### 2.2 Why this survives the audit

Three literatures assume a stable per-item success object; we test the **mode** under configuration change:

1. **Modal-ceiling / test-time scaling.** Plurality voting converges to the mode; selection accuracy is capped at π_mode regardless of budget \citep{brown2024monkeys,bay2026modal,blendasc2025,certifiedsc2025,samplecomplexity2025}. **Nobody has varied elicitation configuration** while measuring π_mode on reasoning benchmarks with sampled CoT. Bay & Yearick's five-session control shows same-configuration run-to-run variance ≈ 0 (ρ_w ≈ 0.0007) but **never changes the prompt.**

2. **Prompt sensitivity (now established at scale).** BrittleBench decomposes item vs. prompt variance on log-prob-scored benchmarks; TEE fits item × prompt by REML. **Their inference variance is zero by construction** (deterministic scoring). For generative math with N=24 samples per cell, binomial noise is the methodological problem — we extend their decomposition to that setting with a crossed binomial GLMM, but that is **supporting**, not the headline.

3. **Downstream consumers of per-item difficulty.** Adaptive TTC, IRT routing, hard-subset curation consume **per-item** parameters one at a time — they do not average away item × prompt interaction the way TEE's aggregate CI does \citep{snell2024scaling,atlas2025,irtrouter2025}. arXiv:2606.19636 shows labels are unstable along the stochastic axis; we show instability along the **configuration** axis reaches the **mode** and is exploitable via CDV.

### 2.3 Nearest prior work and the precise delta

| Nearest prior work | What it establishes | Delta of this paper |
|---|---|---|
| **BrittleBench** (arXiv:2603.13285) | Item × perturbation variance; prompt share ≈ 50%; deterministic inference | **Stochastic axis:** sampled CoT, non-zero inference variance; **modes:** π_mode, reorder rate, identifiability gap per configuration; **intervention:** CDV |
| **TEE** (arXiv:2604.11581) | Crossed REML; item × prompt; D-study on aggregate θ̂ | **Per-item estimand** for consumers that do not average; free-generation math terrain |
| **arXiv:2606.19636** | pass@k labels conflate hard vs. unreached; fix-set Jaccard | **Configuration** axis (not activation grafting); mode-level consequence |
| **arXiv:2607.13304** | G-theory facet decomposition; repeats are cheap facet | Same allocation logic for **configurations vs. i.i.d. repeats** at the **mode**; different outcome and domain |
| **PromptEval** (arXiv:2405.17202) | IRT over item × template matrix | We add **configuration facet** to reasoning CoT; compare unidimensional vs. config-specific item difficulty |
| **Bay & Yearick** (arXiv:2606.28661) | π_mode, identifiability gap, correlation ceiling | **Configuration factor missing** in their decomposition; we measure π_mode(m,c) fresh |
| **Sclar et al. FormatSpread** | Aggregate format sensitivity | Item-level **mode reordering** and CDV plateau — strictly stronger object |
| **Self-Para-Consistency / SCoP** | Paraphrase voting baselines | **Measurement + mechanism** on non-paraphrase axes (c0–c2); CDV discriminating prediction vs. temperature SC |

### 2.4 What would falsify the claim

Each condition is **individually decisive** for its sub-claim (see `docs/REFRAMING.md` §3):

| ID | Sub-claim | Falsification condition | Primary cell |
|---|---|---|---|
| **F3** | CDV raises plateau (mechanism) | Plug-in π_mode(CDV) ≤ π_mode(c0) at matched budget; curves coincide at large B_s | 3B × MATH-500 |
| **F4** | π_mode is configuration-relative | max π_mode − min π_mode ≤ 3 pp across core configs | 3B × MATH-500 |
| **F1** | Item × configuration structure (supporting) | GLMM interaction share ≤ 5% and below parametric-bootstrap null | 3B × MATH-500 |
| **F2** | Hard subset instability (supporting) | J_config ≥ J_seed − 0.02 (relative, not J ≥ 0.90 absolute) | 3B × MATH-500 |

**F3 is the primary falsification condition** under the reframed paper. A clean negative on F4/F3 — “π_mode is stable across configurations; reported ceilings are model properties” — is a publishable negative result about a live claim in the modal-ceiling literature.

### 2.5 Risks, honestly stated

- **Risk 1 — “BrittleBench got there first.”** Mitigated by reframing: cite as premise; lead with π_mode and CDV; variance share is supporting evidence calibrated against a synthetic null.
- **Risk 2 — parse/truncation confounds.** c3 demoted; dual extractors, extraction audit, truncation logging pre-registered (see METHOD_SPEC §3–§4).
- **Risk 3 — F3 unmeasurable at N=24.** O2 redirects deep-N to multi-config arm on 3B MATH-500; plug-in π_mode comparison pre-registered as primary F3 test.
- **Risk 4 — Bay & Yearick preprint dependence.** Define π_mode from first principles; one citation for the “they never varied configuration” gap; verify arXiv:2606.28661 (done this pass).
- **Risk 5 — GSM8K saturation.** MATH-500 leads; GSM8K secondary; GSM-Symbolic contamination-resistant replicate.

### 2.6 Why it fits the compute budget

Single sampling corpus; CDV is re-partitioning only. Revised matrix is **cheaper** than the original (~32 h vs. ~46 h) while enabling the sharpest test (F3/P5).
