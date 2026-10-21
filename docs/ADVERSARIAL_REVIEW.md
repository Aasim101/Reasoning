# Adversarial Pre-Execution Review

**Reviewer role:** hostile but technically fair NeurIPS/ICLR/ICML/ACL reviewer.
**Date:** 29 July 2026.
**Materials reviewed:** `docs/RESEARCH_GAP.md`, `docs/LITERATURE_REVIEW.md`, `docs/METHOD_SPEC.md`, `docs/CITATION_AUDIT.md`, `paper/main.tex`.
**Purpose:** find reasons to reject *before* ~46 GPU-hours of a scarce free-tier quota are spent.

**Bottom line up front.** I found a paper the authors do not cite that already performs the headline variance decomposition — separating item difficulty from semantics-preserving prompt sensitivity — at larger scale, in March 2026, and reports the prompt-attributable share at roughly **half** of total variance. That is **BrittleBench, arXiv:2603.13285**. It does not kill the whole paper, but it kills Contribution 1 as written and forces a reframing. Two further 2026 papers (**arXiv:2604.11581**, **arXiv:2606.19636**) occupy the generalizability-theory framing and the "unstable difficulty labels break RL filters and curricula" framing respectively. Separately, the design as specified **cannot test its own sharpest prediction (P5/F3)**, because at `N=24` self-consistency has not reached its plateau on MATH-500, so "CDV raises the plateau" is unmeasurable at matched budget. Both problems are fixable at zero or near-zero net GPU cost. **Do not run as designed.**

---

## 1. Novelty attack

### 1.1 The kill: BrittleBench (arXiv:2603.13285)

**A. Romanou, M. Ibrahim, C. Ross, C. Shaib, K. Oktar, S. J. Bell, A. Ovalle, J. Dodge, A. Bosselut, K. Sinha, A. Williams. "BrittleBench: Quantifying LLM Robustness via Prompt Sensitivity." arXiv:2603.13285, March 2026.** (Verified: abstract and full body fetched from `arxiv.org/pdf/2603.13285`; independently corroborated by arXiv:2604.11581, which cites it by ID and summarises it identically. Author list is strong — Meta AI / EPFL / AI2 lineage.)

What it does, in its own words:

> "we introduce a theoretical framework for quantifying model sensitivity to prompt variants, or brittleness, that can enable us to **disentangle data-induced difficulty from prompt-related variability**."
> "Our methodology estimates model variability by **decomposing observed performance variance into components attributable to task difficulty and prompt sensitivity**."
> "we find that semantics-preserving input perturbations can **account for up to half of the performance variance** for a given model."

The formal apparatus is the law of total variance applied to binary correctness `Y = Y(D, P, R)` over items `D`, perturbation conditions `P`, and inference runs `R`:

```
Var(Y) = E_{D,P}[Var_R(Y | D,P)]        <- inference variance
       + Var_D( E_P[Y | D] )             <- "data difficulty variance"
       + E_D[ Var_P(Y | D) ]             <- "brittleness" = item x perturbation
```

Estimated from an item × perturbation binary outcome matrix, pooled to model-level and benchmark-level "brittleness scores" `Π_m`, `Π_b`.

**This is the paper under review's Q1, with the same estimand, the same interpretation, and the opposite of a null result.** Their perturbation taxonomy explicitly includes **persona insertion** ("context augmentation involves the addition of information such as personas, emotional phrases, or explanatory paraphrasing") and **paraphrase** — i.e. this project's `c1` and `c6`. They frame it as a **construct-validity** question in exactly the project's terms: "whether observed difficulty arises from inherent task complexity or from incidental, structural and/or formatting artifacts in the prompt design." They report the result holds across scale (4B to 70B plus commercial), and they even run a test-time-intervention arm (whether CoT mitigates brittleness).

**Verdict: this is a kill on Contribution 1 and on the paper's one-sentence claim as currently stated.** The abstract sentence "We test that assumption directly… we decompose the variance of per-item success rate" describes work published four months ago. A reviewer who knows this paper will open with it, and the Related Work section's current framing — that the open question is "whether the movement is *noise* … or *structure*" — will read as unaware of the literature, which is fatal for a paper whose selling point is a rigorous novelty audit.

**What BrittleBench does *not* do, and what therefore survives:**

1. **The stochastic axis vanishes by construction in their design.** They state: "In our evaluation setup, inference is deterministic, since we are using an evaluation framework (eval-harness) that assesses performance based on the output log-prob of the model; for any fixed pair `(D,P)`, the model performance is fully determined and therefore the inference variance term vanishes." They never had to separate sampling noise from interaction, because there is none. For **sampled chain-of-thought on generative reasoning benchmarks**, the noise floor *is* the whole methodological problem. That is a real gap, and it is this project's genuine methodological contribution — but it is a contribution about *measurement technique for a known phenomenon*, not a new phenomenon.
2. **No modes, no self-consistency, no ceiling.** Nothing in BrittleBench touches the answer-class distribution, the modal-hit rate, or the identifiability gap. The mode-reordering mechanism and the `π_mode`-is-configuration-relative claim are untouched.
3. **No intervention that exploits the effect.** Their test-time arm asks whether CoT *mitigates* brittleness (it does, by 0.41 percentage points — negligibly). Nobody there proposes spending budget across configurations to *exploit* it. CDV survives.
4. **No rank-transfer or hard-subset statistic.** They report variance shares, not disattenuated rank correlation or bottom-quartile Jaccard.
5. **Benchmarks are mostly MMLU/GPQA/ARC/TruthfulQA/LogiQA/MathQA** — multiple-choice and log-prob-scored, not free-generation math. GSM8K/MATH-500/GSM-Symbolic with `math-verify` canonicalisation is genuinely different terrain.

**Also note their number is bad news for P1.** BrittleBench finds the prompt-attributable share is *around half* on multiple-choice benchmarks and *lower* on MathQA and ARC, which are "largely driven by item difficulty." The project's P1 threshold is a modest 10%. So either the effect is large (in which case BrittleBench got there first and P1 is unsurprising) or math-reasoning items behave like MathQA/ARC (in which case P1 may fail). Neither branch is comfortable, and the second is not currently in the pre-registered negative-result path in a form that survives the overlap.

### 1.2 The generalizability-theory framing is also taken

**arXiv:2604.11581, "Decomposing and Reducing Hidden Measurement Error in LLM Evaluation Pipelines" / "Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking" (Total Evaluation Error, TEE), April 2026.** (Verified: v3 and v6 HTML fetched in full.)

This is a crossed random-effects (generalizability-theory) decomposition of LLM evaluation with **item, prompt-variant, temperature and model facets and all two-way interactions**, fit by **REML**, plus a **decision study (D-study)**:

```
Y_ivhm^(r) = μ + α_i + φ_v + τ_h + λ_m
           + (αφ)_iv + (ατ)_ih + (φτ)_vh + (αλ)_im + (φλ)_vm
           + ε_ivhm + ρ_ivhm^(r)
```

Note `(αφ)_iv` — the item × prompt interaction — is an explicitly named, separately estimated variance component with its own expected-mean-square row and its own D-study denominator. They report it on MMLU (200 items, factorial over prompt variants and replications): within-category item heterogeneity 35.0%, model design-sensitivity 25.0%, prompt main effect 14.7%, prompt × SUT 13.8%, item × SUT 5.3%; and item × prompt `σ²_αφ` = 2.5% of `Var(θ̂)` in their summary table. They also include a replicate facet `ρ` that *is* sampling noise, handled exactly by the mixed model rather than by subtraction.

Most pointedly, they name this project's hypothesis as an assumption of their own model and dismiss it as second-order:

> "the model assumes item difficulty is independent of prompt sensitivity (`α_i ⊥ (αφ)_iv`), yet **ambiguous items may be more prompt-sensitive**. … Section E.2, Scenario 2 shows that even strong dependence produces ≤2% D-study bias."

**Consequences.** (a) The claim in `RESEARCH_GAP.md` §2.2 that psychometric evaluation "never stress-tested [difficulty] against elicitation configuration as a crossed factor" is **false** as of April 2026. (b) The obvious defensive reframing — "this is a generalizability-theory / G-study + D-study treatment of reasoning benchmarks" — is **pre-empted**, so do not reach for it. (c) TEE's statistical machinery is strictly better than the method-of-moments-with-clamping in `METHOD_SPEC` §5.2, and its Figure 7 supplies external evidence that REML interaction components are <5% biased above ~1 000 observations. The project will have 3 600 cells per dataset. **Adopt their estimator.**

**What survives.** TEE's estimand is `Var(θ̂)` — the variance of the *aggregate benchmark score* — and every interaction enters divided by the number of levels (`σ²_αφ / N'V'`). Its entire message is that item × prompt is a *nuisance that averages away* and inflates your CI. This project's estimand is the **per-item parameter itself**, which is what adaptive TTC, curricula, hard-subset curation and IRT item banks consume, and which **does not average away** because those consumers use it one item at a time. That distinction is real, sharp, and defensible — but it is currently nowhere in the paper, and it must become the opening move of §related-prompt.

Two further G-theory-in-LLM-eval papers to cite and not be scooped by: **arXiv:2603.01865 (CyclicJudge)**, a G-theory decomposition into scenario/generation/judge/residual for LLM-as-judge; and **arXiv:2607.13304**, a crossed random-effects decomposition over resampling / prompt-paraphrase / model / language facets with a D-study allocation rule. The latter is from **this month** and its central structural argument is uncomfortably close to this project's: that **resampling is the facet a fixed budget already suppresses through averaging**, so repeats buy less per query than paraphrases or models. That is the allocation-theoretic sibling of "i.i.d. resampling cannot move the plateau." Different outcome (brand sentiment), no modes, no reasoning — but the *shape* of the argument is published.

### 1.3 The downstream framing is also partly taken

**arXiv:2606.19636, "Hard or Just Unreached? Diagnosing the Sampling Blind Spot in Math-Reasoning Difficulty Estimation," June 2026.** Four open-weight models (3B/8B/12B) × three reasoning benchmarks (GSM8K, MATH, MMLU-Pro), twelve (model, benchmark) cells. Finding: of items no sampling seed solves at `k=6`, 10–29% are reached by a deterministic regime at matched compute. Conclusion, in their words:

> "sampling-derived difficulty annotations conflate 'hard' with 'unreached on the stochastic axis' on a non-trivial fraction of their hardest stratum, which is what **RL filters, curricula, and pass@k-based labels** build on."

They also use **cross-condition fix-set Jaccard** as an instability statistic. This is *the same downstream argument* — per-item difficulty labels are unstable, and this corrupts curricula, RL data filters and hardness buckets — established on the same benchmarks, along a different axis (stochastic vs. deterministic decoding rather than prompt configuration). It does not kill the project, but it means "we are the first to point out that difficulty-conditioned pipelines consume an unstable label" is no longer available, and the curriculum-RL consequence in particular is now second-mover.

### 1.4 Other must-cite omissions

| Paper | Why it must be cited | Verified |
|---|---|---|
| **BrittleBench, arXiv:2603.13285** | Kills Contribution 1; nearest prior work by a wide margin | Body fetched |
| **TEE, arXiv:2604.11581** | Crossed variance decomposition with item × prompt, REML, D-study | Body fetched |
| **arXiv:2606.19636** | Difficulty labels unstable; breaks RL filters / curricula / hardness buckets | Abstract + body excerpt fetched |
| **arXiv:2607.13304** | G-theory facet decomposition; "repeats are the cheap facet" argument | Body fetched |
| **arXiv:2603.01865 (CyclicJudge)** | G-theory variance decomposition in LLM eval | Abstract + body excerpt fetched |
| **PromptEval, arXiv:2405.17202, NeurIPS 2024** (Polo, Ashury-Tahan, et al.) | **Serious omission.** Fits an IRT model over the joint (example, prompt-template) response matrix — `Y_ij` for item `i`, template `j` — precisely to estimate the *distribution* of performance across 100 prompt templates on MMLU/BBH/LMentry, "borrowing strength across prompts and examples." It is the direct psychometric antecedent of this design and it is absent from a 94-entry bibliography that contains four other IRT-for-LLM papers. A reviewer will notice. | Abstract + body fetched |
| **Mizrahi et al., "State of What Art? A Call for Multi-Prompt LLM Evaluation," TACL 2024** | The canonical multi-prompt-evaluation call to arms; standard citation in this space | Not re-verified this session — confirm before citing |
| **Alzahrani et al. 2024** (answer-choice reordering shifts MMLU rankings by up to 8 positions) | Cited by TEE; standard evidence for elicitation-driven ranking instability | Via TEE's reference list |
| **`wang2025measuring`** | BrittleBench cites this as the prior source of its decomposition approach ("similarly to wang2025measuring"). I could not resolve the full reference. **Chase it — it may be closer still.** | Unresolved |

### 1.5 Novelty verdict

**Serious overlap, amounting to a kill of the headline claim as written; not a kill of the project.** The phenomenon (per-item difficulty is substantially prompt-attributable) is published. The methodology (crossed variance components with a prompt facet) is published, better executed. The downstream argument (unstable difficulty labels corrupt curricula and RL filters) is published. What is *not* published is the intersection of the stochastic axis with the selection bottleneck: that the **modal answer**, and hence `π_mode`, and hence the reported reasoning boundary of a model, is configuration-relative — and that this is exploitable.

**Nearest surviving reframing (detailed in §4.3): make `π_mode` the subject of the paper, not item difficulty.**

---

## 2. Statistical attack

### 2.1 The variance decomposition on the Haldane logit

**(a) The censoring bound is a function of `N`, so the headline variance share is not a property of the models.** This is the objection I would put first among the statistical ones, and I do not think the authors have noticed it.

The Haldane–Anscombe logit `z = log((k+½)/(N−k+½))` is **bounded**: at `k=0` and `k=N` it takes the values `∓log(2N+1)`. At `N=24` that is `±3.892`; at `N=48`, `±4.575`; at `N=64`, `±4.860`. Every saturated cell — and on GSM8K with the 7B there will be many — is pinned at the bound rather than at its true latent difficulty. Two consequences:

- The **item main-effect variance `σ²_α` is mechanically inflated** by the saturated fraction and grows with `N`, because the bound grows with `N`. The item × configuration *share*, being a ratio with `σ²_α` in the denominator, therefore **mechanically shrinks as `N` rises**, with no change in any underlying quantity.
- The variance share is consequently **not comparable across cells with different `N`**. This directly breaks the Tier B `O2` arm (7B, MATH-500, `N: 24 → 64`) as a comparison, and it breaks comparability with any future study at different `N`.

There is no fix within the moment-based approach. There is a clean fix outside it (§2.1c).

**(b) The noise floor is not transportable across configurations, and the subtraction is knife-edge.** `METHOD_SPEC` §5.2 step 3 estimates `σ²_samp` from **seed replicates at `c0` only**, then subtracts "the expected binomial contribution from every mean square." But the sampling variance of `z` is severely heteroscedastic in `p`: the plug-in `v = 1/(k+½) + 1/(N−k+½)` equals **0.160** at `k=12` and **2.041** at `k=0` — a factor of **12.8**. Configurations differ in accuracy (that is the premise of the paper), so they differ in their saturated fraction, so they differ in their true noise floor. A floor estimated at `c0` is systematically wrong for `c1…c5`, and the sign of the error depends on whether the configuration pushes items toward or away from the boundary.

Now the magnitude. The estimand is a *difference*: `σ̂²_αγ = MS_αγ − floor`. For an interior cell at `p ≈ 0.5`, `σ²_samp ≈ 0.16`. A per-item configuration-induced shift of `Δp = 0.05` corresponds to `Δlogit ≈ 0.20`, i.e. `σ²_αγ ≈ 0.04`. So the subtraction removes ~80% of `MS_αγ` and keeps ~20%. **A 10% error in the floor is a 40% error in the estimand. A 20% error wipes it out.** Given a 12.8× heteroscedasticity range and a floor measured in only one configuration, a 10–20% mis-transport is not a worst case, it is the expected case.

The design is well powered against *sampling* variability and completely unprotected against *floor bias*. The power calculation in §2.6 makes this precise.

**(c) Yes, this should be a GLMM, and yes, the moment approach biases the interaction. Make the GLMM the headline.** `METHOD_SPEC` §5.2 step 6 lists the Bayesian binomial GLMM as "SECONDARY (optional, if runtime allows)… a confirmation, not the headline." **That is backwards and it is the single cheapest high-value fix available.** Fit

```
k_imc ~ Binomial(N, p_imc)
logit(p_imc) = μ + a_i + b_m + g_c + (ab)_im + (ag)_ic + (bg)_mc
```

with crossed random effects, by REML (`lme4::glmer`) or NUTS (`numpyro`). This:

- handles the binomial noise **exactly, by construction** — no floor estimate, no subtraction, no transportability assumption, no negative components, no clamping;
- handles saturated cells by **partial pooling** rather than by pinning them at `±log(2N+1)`, so the components no longer depend on `N`;
- yields the disattenuated transfer correlation **directly and exactly** as `σ²_a / (σ²_a + σ²_ag)`, replacing the ad hoc Spearman ratio of §5.3 with a model-based quantity;
- costs **CPU minutes** on 3 600 cells and zero GPU time;
- is what TEE (arXiv:2604.11581) already does, so a reviewer will ask why this paper did not.

The moment-based ANOVA should be demoted to a robustness appendix. Note also that with one observation per `(i,m,c)` cell, the three-way `item × model × configuration` term is confounded with the residual in the moment approach — and that three-way term is arguably *the* component named by the paper's own title ("a model × elicitation interaction"). The GLMM can carry it explicitly.

**(d) Clamping is not neutral; it inflates the surviving shares.** §5.2 step 3 clamps negative components to zero and reports the clamping rate. But shares are computed as a fraction of the *total corrected* variance. Clamping a component to zero removes mass from the denominator, which **raises the reported share of every surviving component**, including `item × configuration`. Reporting the clamping rate does not correct this. The GLMM eliminates it (variance components are non-negative by parameterisation).

**(e) Credit where due: floor uncertainty *is* propagated, mostly.** The reviewer's concern that the floor's uncertainty is not propagated is largely unfounded. §5.2 step 5 bootstraps items and "refit[s] steps 1–4 each time," and step 3 is inside that loop, so the floor is re-estimated per bootstrap replicate. Furthermore, the floor is pooled over 200 items × 3 models × 2 df ≈ 1 200 degrees of freedom, so its *sampling* precision is not the problem. The problem is its **bias and non-transportability** (§2.1b), which no amount of bootstrapping over items will reveal. The one residual gap: the bootstrap does not resample the seed dimension, so the (small) 3-seed contribution to floor uncertainty is omitted. Minor.

### 2.2 Saturated cells and the exclusion rule — the reviewer's lead objection, evaluated

**Verdict: the objection as stated is *not* fatal, but only because the exclusion rule is nearly vacuous — and that fact is itself a worse problem.**

The rule (Risk 4, `RESEARCH_GAP.md` §2.5) drops items with `p̂ = 0` in **all** cells. In Tier A a Qwen-ladder item has 3 models × 6 configurations = **18 cells**. For an item to be excluded, all 18 must be saturated. `P(k=N) = p^24`, so `P(excluded) = p^{432}`:

| true `p` | `P(k=24)` per cell | `P(all 18 cells saturated)` |
|---|---|---|
| 0.90 | 0.080 | ~1 × 10⁻²⁰ |
| 0.99 | 0.786 | 0.013 |
| 0.997 | 0.931 | 0.28 |
| 0.999 | 0.976 | 0.65 |

So the rule bites **only** on items with true `p ≳ 0.997` or `≲ 0.003`. The censoring rate will be near zero, the excluded set will be a negligible sliver, and **the selection-toward-mid-difficulty bias the reviewer fears is essentially absent.** Selection on the noise cannot inflate the interaction term when almost nothing is selected. So: **not fatal.** State this in the paper with exactly this arithmetic — it is a complete, cheap rebuttal and it will neutralise the objection on first contact.

**But the reason the rebuttal works is damning in a different way.** Because the rule excludes almost nothing, the retained sample is *dominated* by cells at or near `k = 0` or `k = N`, which are precisely the cells where (i) `z` is censored at `±log(2N+1)` so real variation in `p` is invisible, (ii) the plug-in sampling variance is 2.04 rather than 0.16, so the noise floor is at its maximum and its estimate at its worst, and (iii) no mode reordering is possible, since a mode can only move if two answer classes hold non-trivial mass.

That produces the objection I would actually lead with as a reviewer, which is the *steel* version of the one posed:

> Your `item × configuration` variance is necessarily concentrated on mid-difficulty items, because those are the only items where the outcome is not censored. Mid-difficulty items are also mechanically the most variable under any noise model. So your headline share may be a restatement of "items near `p = 0.5` move around more," which is arithmetic, not a finding about elicitation.

**Three analyses rebut it, all at zero GPU cost, and all three should be pre-registered now:**

1. **A parametric-bootstrap null calibration.** This is the most important missing analysis in the whole design. Simulate under `H₀`: take each item's *observed marginal* `p̄_i` (pooled across configurations), impose **no** item × configuration interaction, draw `k ~ Binomial(24, p̄_i)` for every cell, and run **the entire pipeline** — Haldane transform, moment decomposition, floor subtraction, clamping, exclusion rule — on the synthetic data. Report the null distribution of the item × configuration share. This converts P1 from an arbitrary 10% threshold into a calibrated test, and it directly answers "is the share mechanically inflated?" with a number. It also exposes any pipeline bias from clamping and censoring in one shot. **Cost: CPU only. Do this before spending any GPU time — it will tell you whether the design can detect what you expect, using nothing but simulated data.**
2. **Stratify by marginal difficulty.** Report the interaction share, `ρ_disatt` and `J_config` within strata of `p̄_i` (say quintiles). If the effect exists only in the mid-difficulty stratum, say so and interpret it honestly; if it persists across strata, the objection dies.
3. **Report on the raw proportion scale as well as the logit scale.** The censoring artefact is a property of the logit transform. If the qualitative conclusion holds on both, the transform is not doing the work. (TEE reports on the probability scale and flags the logit question as an open limitation; this paper is better positioned on that axis and should say so.)

### 2.3 Disattenuation

**Applied correctly in form.** `r_mm` is the mean pairwise correlation between two independent single measurements, i.e. a test–retest reliability, so the correct classical denominator `√(r_xx · r_yy)` reduces to `r_mm`. Credit — this is a place where a lot of papers get it wrong and this one does not.

**Four problems, in descending severity:**

1. **`r_mm > 0.95` will be an artefact of saturation, not evidence of good measurement.** Test–retest reliability of `p̂` is near-perfect for items at `p ≈ 0` or `1` (they reproduce exactly) and poor for interior items. With a saturation-dominated sample, P2's "`r_mm` above 0.95" is close to guaranteed and **tells us nothing about the reliability of the estimates that carry the claim**. Meanwhile `ρ_raw` across configurations is depressed mainly by interior items. So the headline ratio has a denominator dominated by saturated items and a numerator driven by interior ones. **Report `r_mm` and `ρ_disatt` within difficulty strata**, and stop treating `r_mm > 0.95` as a validity check. A reviewer should and will attack P2 on exactly this.
2. **Reliability is measured in `c0` only, and is assumed transportable.** Seed replicates exist at `c0` and nowhere else, so `r_mm(c1) … r_mm(c5)` are unmeasured, and by the heteroscedasticity argument of §2.1b they will differ. Correcting `ρ_raw(c0, c3)` by `r_mm(c0)` alone is a systematic mis-correction of unknown sign. **This is the one place where I recommend spending new GPU time** (§6.3): add seed replicates at one non-reference configuration.
3. **Spearman disattenuation is not theoretically licensed.** The attenuation formula is a classical-test-theory result for Pearson correlations with additive independent errors. Rank correlations do not satisfy it. Either do the correction on Pearson correlations of `z` (with Fisher-`z` intervals) and report Spearman descriptively, or — better — take the latent correlation `σ²_a / (σ²_a + σ²_ag)` from the GLMM, which needs no correction at all.
4. **Values above 1 are possible whenever `ρ_raw > r_mm`,** which will happen for some configuration pairs by chance even under the paper's own predictions. Pre-register the handling (report as-is, do not truncate, and do not average truncated values — truncation would bias the headline mean downward, i.e. *toward* the hypothesis). Also: for cross-model pairs (§5.3 step 5) the denominator should be `√(r_m · r_m')`, not a single `r_mm`; the spec does not say which is used.

### 2.4 The Jaccard / seed-pair null

**The null is the right *idea* and is misspecified in two concrete ways.**

1. **The permutation test tests the wrong null.** §5.4 step 5 "shuffles the config label within item." Under that permutation, both the configuration **main effect** and the interaction are destroyed, so the test rejects if *either* is non-zero. Since a configuration main effect is essentially certain (some prompts are just better), the test will reject regardless of whether any interaction exists. **This is a clean specification error and a cheap fix:** centre each configuration to its own mean (or rank items *within* configuration) before permuting, so the null is exchangeability of the residuals and the test is about the interaction alone.
2. **Bottom-quartile membership is determined by tie-breaking, and the tie mass is configuration-dependent.** On a hard benchmark a large block of items sits at `p̂ = 0`. If the 25th percentile falls inside that block — which it will for MATH-500 on the 1.5B — then membership in `H(m,c)` is decided by arbitrary tie-breaking, and **both** `J_config` and `J_seed` become measurements of tie-breaking noise. Worse, a configuration that lowers accuracy grows the tie mass, so the two Jaccards are not comparable. Rank-based quartiles are invariant to monotone shifts, which protects against uniform difficulty shifts (good), but **not** against items collapsing into the boundary tie block (bad). **Fix, now, and pre-register it:** report the number of items tied at the quartile cut per cell; use a deterministic tie rule; and report a tie-robust variant (e.g. Jaccard over the bottom-quartile set with all tied-at-cut items either wholly included or wholly excluded, giving an interval rather than a point).
3. On the local-density concern specifically: the reviewer is right that membership noise depends on the density near the cut, but for a *fixed* quantile the seed-pair null and the configuration-pair statistic face the same density, so this largely cancels — **except** through the tie mechanism above, which is where the real damage is. Judgement: the null is repairable, cheaply, and is not fatal.

### 2.5 Multiple comparisons

**Under-controlled, and the pre-registration makes it worse rather than better.**

`METHOD_SPEC` §7.2 applies Holm–Bonferroni "across the family of baseline comparisons within each table." That covers **T5** and nothing else. Uncontrolled families include:

- **P1–P6 across cells.** Each prediction is evaluated on up to 3 models × (GSM8K, MATH-500, GSM-Symbolic `main`, GSM-Symbolic `p2`) ≈ 6–12 cells. That is roughly 50–70 primary tests with no adjustment.
- **`ρ_disatt` across 15 configuration pairs × 3 models × 2 datasets ≈ 90 correlations** (T4).
- **`J_config` permutation tests** over the same pair set.
- The precision control (P7) adds another paired family.

And the wording of P1 makes it a **disjunction**: "exceeds the item×model share **on at least one dataset**." A prediction satisfied by 1 of 6 cells, with no multiplicity control, is not a pre-registered test; it is a search. **Fix now, before data:** define one primary cell per prediction (I would nominate 3B × MATH-500), state the aggregation rule across the remaining cells explicitly, declare the family sizes, and apply Holm within each declared family. This costs nothing and it is the difference between "pre-registered" and "pre-registered-looking."

### 2.6 Power for the interaction term — rough calculation

Take one model, `I = 200` items × `C = 6` configurations, one observation per cell.

- `df(MS_αγ) = (I−1)(C−1) = 199 × 5 = 995`.
- `E[MS_αγ] = σ²_samp + σ²_αγ`.
- `SD(MS_αγ) ≈ E[MS_αγ] · √(2/df) = 4.5% of E[MS_αγ]`.

Take an interior-dominated regime, `σ²_samp ≈ 0.16`, and a configuration-induced per-item logit shift with SD 0.20 (i.e. `Δp ≈ 0.05` at `p = 0.5`), so `σ²_αγ ≈ 0.04`:

- `E[MS_αγ] ≈ 0.20`, `SD ≈ 0.009`.
- `σ̂²_αγ = 0.04 ± 0.009` from sampling variability alone → `z ≈ 4.4`.

**So the design is *not* underpowered against sampling variability for the interaction term.** 995 degrees of freedom is ample, and item expansion to 400 (Tier B) is a luxury rather than a necessity for this component. §8.4's reasoning ("items are the best buy") is correct but solves a problem the design does not have.

**The binding constraint is bias, not power.** The same arithmetic shows that a 10% error in the noise floor (0.016 absolute) is **1.8×** the sampling SD and **40%** of the estimand, and a 25% floor error erases the estimate entirely. Floor bias does not shrink with more items, more seeds or more bootstrap replicates. This is why the GLMM recommendation (§2.1c) matters more than any amount of Tier B expansion: it removes the floor from the estimator rather than trying to estimate it better.

Caveat on the effect size: if the true per-item configuration shift is `Δp ≈ 0.02` rather than 0.05, then `σ²_αγ ≈ 0.006`, which is **4%** of the floor — undetectable by subtraction at any item count, and detectable by a GLMM only marginally. **The parametric-bootstrap calibration in §2.2(1), plus a minimum-detectable-effect curve, should be run on synthetic data before any GPU hour is spent.** It is the cheapest possible way to find out whether this experiment can answer its own question.

Note the counter-current: the same interior-regime assumption that makes power adequate is contradicted by the saturation-dominated reality of GSM8K. On GSM8K with `σ²_samp ≈ 1.5`, `σ²_αγ ≈ 0.04` is 2.7% of the floor and the estimate is hopeless. **GSM8K is the wrong benchmark for the headline decomposition and MATH-500 is the right one.** The spec currently treats them symmetrically.

---

## 3. Confound attack

Ordered by how much of the headline they could manufacture on their own.

### 3.1 Answer extraction and canonicalisation — serious, and the design does not currently defeat it

**Threat.** `c3` changes the final-answer instruction from `"Put your final answer after 'Answer: '"` to `"End your response with \boxed{}"`. `math-verify` is built around `\boxed{}`. If extraction reliability differs between the two — in **either** direction — then:

- the `⊥` (extraction-failure) rate differs by configuration;
- `⊥` is counted as wrong (§4.3), so `p̂` differs by configuration **for items whose reasoning did not change at all**;
- that is a genuine item × configuration interaction in the data, manufactured entirely by the parser, concentrated on items whose answers are formatting-awkward (fractions, radicals, intervals, multi-part answers) — i.e. **structured, item-specific, and reproducible across seeds**, which is exactly the signature the paper offers as evidence of real structure. The seed-replicate null **cannot** detect it, because parsing failure is deterministic given the text.
- `⊥` is a legitimate answer class, so it can become the **modal** class, which corrupts `π_mode`, the reorder rate, and the corrective/destructive split;
- **CDV pools across configurations, which dilutes any single configuration's `⊥` mass.** So CDV's entire advantage over single-configuration SC could be a parsing artefact.

**Does the design defeat it? No.** §4.3 reports the per-cell extraction-failure rate and flags cells above 5%. That is monitoring, not control, and a 4%-vs-1% differential is well under the flag threshold while being more than large enough to produce the predicted effects. This is the most under-defended part of the design.

**Required, all at zero GPU cost, all to be pre-registered before the run:**

1. **Two independent extractors** (e.g. `math-verify` plus a rule-based numeric/LaTeX extractor with a different failure profile). Report per-configuration agreement. Run the entire headline pipeline under both and report both. If the conclusion is extractor-dependent, the paper has no result.
2. **A stratified manual/LLM audit** of ~100 samples per configuration, balanced across parse-success and `⊥`, estimating per-configuration extraction precision and recall. Report as a table. This is the single most persuasive thing that can be done here and it needs no GPU.
3. **Three pre-registered sensitivity analyses:** (i) `⊥` dropped rather than scored wrong; (ii) analysis restricted to items with zero `⊥` in every cell; (iii) analysis restricted to items whose gold answer is a bare integer (where both extractors are near-perfect). The headline must survive all three.
4. **Demote `c3` to a separately reported axis, alongside `c6`.** The paper already isolates `c6` because it is not strictly semantics-preserving; `c3` should be isolated because it is not *parse*-invariant. Show the headline holds on `{c0, c1, c2, c4}` and report `c3` beside it. This is free and it removes the objection from the critical path.

### 3.2 Length, truncation and stop sequences — serious and under-instrumented

Three distinct mechanisms, none currently controlled:

- **`max_tokens` truncation.** `c1` ("You are an expert mathematician. Be rigorous.") is a persona explicitly selected to induce longer, more careful chains. On MATH-500 `max_tokens = 1024`. A configuration that lengthens chains truncates more often; truncated chains yield no parseable answer → `⊥` → scored wrong. Truncation concentrates on **hard items** (long solutions), so this manufactures an item × configuration interaction that is item-specific, structured and seed-stable. Same signature problem as §3.1.
- **`max_model_len = 2048`.** `c4` is 4-shot. A 4-shot MATH prompt plausibly runs 600–900 tokens; 900 + 1024 = 1 924, leaving under 130 tokens of headroom. Generation will be silently clipped by the context limit in some cells and not others.
- **Stop sequences.** `stop = ["\n\nProblem", "\n\nQuestion"]`. These strings are highly likely to appear in the 4-shot exemplar formatting for `c4`, and can appear spontaneously inside chains. Stop-string firing is therefore configuration-dependent truncation by another route.

**Required (zero GPU):** log and report `finish_reason` per sample; report per-cell rates of length-truncation, stop-string termination, and context-limit clipping; pre-register a sensitivity analysis excluding truncated samples; and pre-register a rule that any configuration whose truncation rate exceeds `c0`'s by more than (say) 2 percentage points is reported separately. Also: verify at smoke-test time that the longest 4-shot prompt plus `max_tokens` fits inside `max_model_len` for every dataset, and raise `max_model_len` to 3072 for `c4` if it does not.

There is a second-order effect worth stating in the paper: because comparisons are matched on **output tokens**, a verbose configuration contributes **fewer votes per token** to the CDV pool. CDV is therefore not a uniform `1/C` mixture; its composition is determined by chain length. This must be reported (effective votes per configuration at each budget), not assumed away.

### 3.3 `c4` is not a semantics-preserving axis

`METHOD_SPEC` §3 asserts "no axis carries task information." **`c4` violates this.** It is defined as a 4-shot CoT prompt in reverse exemplar order relative to `c4a`, but the main grid compares it against `c0`, which is **zero-shot**. So the `c0 ↔ c4` contrast conflates *exemplar order* with *the presence of four worked solutions* — and worked solutions are task information by any reading. BrittleBench independently reports that "few-shot prompting substantially improves baseline performance but often amplifies sensitivity to prompt-level perturbations," so this axis is known to behave differently in kind.

**Fix (choose one):** put **both** `c4` and `c4a` in the main grid so that ordering is the contrast and shot-count is held fixed; or demote `c4` to a separately reported axis with `c3` and `c6`. The first costs a configuration's worth of GPU time; the second is free. Either is acceptable; the status quo is not, because the paper's central design argument is that the axes carry no task information.

### 3.4 Temperature (`c5`) is circular and should be moved

The mechanism claim is that configuration changes reorder the top answer classes. Temperature **is** a direct reparameterisation of the answer distribution — flattening it, shrinking top-two margins, and moving items into the reordering-susceptible zone. Including it as a configuration and then reporting that "elicitation configuration reorders modes" is close to circular, and it is the same argument the authors themselves make in §1.5 for *rejecting* 4-bit quantisation ("quantisation … shrinks answer-class margins, which moves items *into* the susceptible zone … a confound that is both directional and concentrated on the critical subpopulation is the one we should not adopt voluntarily"). **That argument applies verbatim to `c5`, and the spec does not notice.**

It is also internally inconsistent: `c5` inflates the item × configuration term in Algorithm 1, but CDV's default `C_use = 4` ("first `C_use` configs") **excludes** `c5`, and `T = 1.0` simultaneously appears inside baseline B4 (temperature-diversified SC), which is supposed to be the *control* that isolates "diversity per se."

**Recommendation: remove `c5` from the core configuration factor.** Either report it separately like `c6`/`c3`, or fold it entirely into B4 where it belongs. This is **budget-positive** — see §6.3.

### 3.5 Chat-template and tokenizer artefacts

**`c2` ("empty system prompt") may not exist as specified.** Qwen2.5's chat template inserts a default system message (`"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`) when none is supplied. If so, `c2` is not "no persona" but "Qwen's own persona" — a different manipulation than advertised, and one whose *content* differs from the neutral `c0` prompt in a way that is model-specific. Llama-3's template handles an absent system message differently again. Consequence: **the configuration factor is not the same object across model families**, which contaminates `model × configuration`, the three-way term, and the cross-family replication that carries the "not Qwen-specific" claim.

**Required (zero GPU):** persist the **fully rendered prompt string and its token IDs** for every (model, configuration) pair; diff them; report the rendered templates in an appendix; and if the default-system-prompt injection occurs, either force a genuinely empty system turn or redefine `c2` honestly as "family-default system prompt."

### 3.6 Contamination × configuration

The defence in §2 of `METHOD_SPEC` — "a uniform contamination-induced accuracy inflation cancels" — is the wrong defence, and the paper's own §1.5 explains why: contamination is **item-specific**, not uniform. If memorised GSM8K items are retrieved reliably under `c0` but the retrieval cue is disrupted by a persona or format change, that produces an item × configuration interaction driven by memorisation rather than reasoning. The GSM-Symbolic arm is the right control; the fix is to **lead with MATH-500 and GSM-Symbolic and demote GSM8K to a secondary replicate**, and to state the interaction-with-memorisation risk plainly rather than asserting cancellation. (This aligns with §2.6 above: GSM8K is also the wrong benchmark statistically.)

### 3.7 Degenerate outputs, refusals, repetition

Currently these are silently absorbed into "wrong." A "be rigorous" persona plausibly changes rates of hedging, non-committal answers, and repetition loops. **Required (zero GPU, post-hoc from `raw_text`):** per-cell rates of empty output, refusal/abstention, high `n`-gram repetition, and answer-absent-but-untruncated responses, with a pre-registered sensitivity analysis. This is one table and it forecloses a whole class of reviewer questions.

### 3.8 Summary of confounds

Threats §3.1 and §3.2 are each, on their own, **sufficient to produce the headline result with no genuine reasoning change**, and both have the same signature the paper offers as proof of genuine structure (item-specific, reproducible across seeds). The seed-replicate null does not touch either. **This is the second reason not to run as designed.** All required fixes cost zero GPU time; they cost code and pre-registration, which is precisely why they must happen now.

---

## 4. Framing attack

### 4.1 Does the defence land?

**Partially, and not enough — and it now lands considerably less well than the authors think, because it is aimed at the wrong paper.**

The Related Work subsection §related-prompt is a genuinely good piece of scholarship. It concedes Sclar et al.'s cross-model format finding explicitly, names it as "real prior art for the existence of such interactions," and articulates a three-axis delta: item-level rather than aggregate; noise-corrected rather than confounded with binomial error; connected to a selection mechanism. Against **FormatSpread**, that delta holds.

But FormatSpread is no longer the nearest prior work. **BrittleBench (§1.1) already does the item-level variance decomposition separating item difficulty from semantics-preserving prompt sensitivity, and reports the prompt share at roughly half.** Two of the three delta axes evaporate against it:

- *"item-level rather than aggregate"* — BrittleBench's decomposition is built on an item × perturbation outcome matrix. Gone.
- *"noise-corrected"* — survives, but shrinks to a technical qualification: BrittleBench's inference-variance term is zero *by construction* because it scores by log-probability. So the honest delta is "we extend a published decomposition to the stochastic-generation setting, where the noise term does not vanish." That is a methods contribution, and a good one, but it is not a headline.
- *"connected to a selection mechanism"* — survives fully, and is now the **only** load-bearing axis.

**The blunt version.** As currently framed, a well-read reviewer reads this as: *a more careful re-measurement of a phenomenon established four months ago, using a weaker statistical estimator than the concurrent generalizability-theory papers, plus a prompt-ensemble voting method whose gain is explicable as ordinary ensembling and possibly as differential answer parsing.* That is a **reject on novelty** at NeurIPS/ICLR/ICML. The paper's greatest strength — an unusually rigorous, self-critical novelty audit — becomes its greatest liability the moment a reviewer produces a paper the audit missed, because the audit itself invited the standard.

The "prompts matter, we knew that" risk the authors identified is real but **misidentified**. The lethal version is not "we knew prompts matter"; it is "**we knew per-item difficulty is substantially prompt-attributable, it was measured at scale in March 2026, and the number was about 50%.**"

### 4.2 What is genuinely, defensibly new

Strip away everything that overlaps and this remains:

1. **`π_mode` is configuration-relative.** The self-consistency plateau — the modal-hit rate — is not a model constant. Bay & Yearick's own five-session control shows same-configuration run-to-run variance is ~0 (`ρ_w ≈ 0.0007`); they never vary the configuration. **This is the untested step in their decomposition, and the authors correctly identified it.** It is untouched by BrittleBench (no sampling, no modes), by TEE (aggregate CI), and by arXiv:2606.19636 (stochastic-vs-deterministic axis, not configuration).
2. **The mechanism: configuration changes reorder answer classes on small-margin items,** with a corrective/destructive split and a prediction about *which* items are susceptible. Nobody has measured this.
3. **The corollary that follows only from (1) and (2):** if the mode is configuration-relative, then a "reasoning boundary," a "hard subset," and a `π_mode` are properties of a **model–prompt pair**, and spreading budget across configurations accesses accuracy that i.i.d. resampling structurally cannot. CDV is the demonstration.
4. **The per-item estimand distinction from TEE.** TEE shows item × prompt variance inflates your *aggregate* CI and divides it by `N'V'`. This paper's point is that adaptive TTC, curricula, hard-subset curation and IRT item banks consume the per-item parameter **one item at a time**, so it does *not* average away. That is a sharp, correct, and currently unstated distinction.

### 4.3 The strongest available reframing — zero new GPU time

**Make `π_mode` the subject of the paper. Demote the variance decomposition to supporting measurement.**

Working title: **"The Self-Consistency Ceiling Is Not a Model Constant: Modal Answers Are a Model × Elicitation Property."**

The reframed argument, in order:

> Recent analyses bound self-consistency by a modal ceiling `π_mode` and treat it as a property of a model on a benchmark. We show it is a property of a **model–prompt pair**. Configuration changes reorder the top answer classes on small-margin items; `π_mode` moves by X points across semantics-preserving configurations; the "hard subset" and the reported reasoning boundary move with it. Because reordering is the one operation i.i.d. resampling cannot perform, spreading a fixed token budget across configurations raises the plateau rather than reaching it sooner. Prompt sensitivity of item difficulty is established (BrittleBench, TEE); what was not established is that it reaches the **mode**, which is where the selection bottleneck lives, and which is what determines whether a reported ceiling is a fact about a model.

Why this is the right move:

- **It requires no new experiments.** It is a reordering of the same artefacts: `F4` (`π_mode` across configurations), `F5` (CDV vs. SC), `F6` (reordering) lead; `F1`/`F2`/`F3` (variance components, transfer, hard subsets) become the supporting measurement section.
- **It converts the overlap from a competitor into a foundation.** BrittleBench and TEE become the cited establishment of the premise, which the paper then extends to the object nobody has examined. A paper that *builds on* a March 2026 result is safe; a paper that *re-derives* it is not.
- **It sits in the most active theoretical conversation in the field** (modal ceiling, sharpening, the RLVR pass@k debate) rather than in the evaluation-hygiene conversation, which is now crowded with three concurrent G-theory papers.
- **P4 and P5 become the primary predictions**, and they are the two most discriminating ones. The 10%-threshold P1 — the prediction most at risk from both the overlap and the censoring artefact — becomes secondary, which is exactly where a fragile prediction belongs.
- **It survives a null result better.** "`π_mode` is stable across configurations, so reported ceilings are model properties after all, licensing the modal-ceiling literature's framing" is a clean, publishable negative result about a specific live claim. "Item difficulty transfers after all" is a negative result that now merely contradicts BrittleBench without the scale to win that argument.

**Do not** reframe as a generalizability-theory / D-study paper. That was the obvious second option and TEE (§1.2) has taken it, with better statistics and a 13-benchmark survey.

---

## 5. Weakest-link audit

### 5.1 Leaning on Bay & Yearick (arXiv:2606.28661)

**Risk assessment: moderate, currently under-managed, and cheaply mitigated.**

The `CITATION_AUDIT` is admirably candid: two-author preprint, equal contribution, affiliation "PhD, University of Illinois at Urbana-Champaign," gmail correspondence, not peer reviewed. **I have not independently verified this paper's existence or contents in this session** — a dedicated verification pass was dispatched and its result is not yet in hand. I am therefore recording this as an **open verification item**, not as a finding either way. Given that the paper is cited in the abstract, introduction, related work and method, and that the recommended reframing (§4.3) makes it *more* load-bearing rather than less, this must be closed before submission.

The structural risk is real regardless of whether the paper checks out:

- A reviewer who cannot find or does not trust a preprint that four sections depend on will discount the framing.
- If `π_mode`, the "identifiability gap," and the correlation ceiling are all named with reference to a single unreviewed source, the paper inherits its fate.

**Mitigations, all free:**

1. **Define `π_mode` and the identifiability gap self-containedly**, from first principles, in one or two lines. Both are elementary: plurality voting converges to the mode, so selection accuracy is capped at the modal-hit rate; the gap is `pass@N − π_mode`. Cite Bay & Yearick for priority, not for the definition. The paper measures both itself, so it does not need the source to be right — but it currently *reads* as though it does.
2. **Anchor each claim on a peer-reviewed co-citation.** Brown et al. (*Large Language Monkeys*) established the plateau empirically; arXiv:2510.17472 and arXiv:2511.12309 give mode-convergence and stopping theory; arXiv:2506.05295 gives the `Θ(1/Δ²)` top-two-margin separation. The modal-ceiling *concept* is over-determined by the literature. Bay & Yearick should be one citation among several, not the spine.
3. **Reduce four load-bearing citations to one or two**, and mark the preprint status inline at first use.

Note the one place where the dependence is *unavoidable and fine*: the observation that Bay & Yearick estimate `ρ_b` and `ρ_w` **without ever varying the configuration** is the project's central gap statement. That is a claim about what a specific paper *omits*, and it necessarily cites that paper. Keep it; it is the strongest sentence in the gap analysis.

### 5.2 Falsification conditions F1–F3

**F1 — "item main effect ≥ 80% of explainable variance, item×configuration ≤ 5%."** Poorly specified and possibly satisfiable for the wrong reason. On the censored Haldane logit with a saturation-heavy sample, `σ²_α` is mechanically inflated (§2.1a), so a share this high is *plausible even when a practically important interaction exists*. Worse, F1's `≤ 5%` and P1's `> 10%` leave an undeclared dead zone at 5–10% where neither the prediction nor the falsification fires — and that is exactly where the effect is most likely to land given TEE's 2.5% and BrittleBench's benchmark heterogeneity. **Declare the 5–10% outcome and what it means, now.**

**F2 — "hard subset retains ≥ 90% membership under configuration B, beyond what re-estimation noise predicts."** **Unsatisfiable as an absolute threshold, and ambiguous as a relative one — so it is not currently a falsification condition at all.** Arithmetic: `J = P(both) / (2q − P(both))`. Setting `J = 0.90` at `q = 0.25` gives `P(both) = 0.2368`, i.e. **94.7% of the bottom quartile must coincide.** At a quartile boundary with `N = 24` and `r_mm ≈ 0.95`, `J_seed` itself will not reach 0.90 — and since `J_config ≤ J_seed` necessarily, F2 can never fire. If instead the intent is relative (`J_config ≥ 0.9 · J_seed`), then say so; note that this sits within 0.02 of P3's threshold (`J_config ≤ J_seed − 0.10`), which is fine but means the ambiguity is a live analytic degree of freedom. **Resolve the wording before seeing data or F2 is post-hoc by construction.**

**F3 — "configuration-diversified voting fails to raise `π_mode` over i.i.d. SC at matched token budget."** **This is a genuinely good, sharply discriminating falsification condition** — it separates variance reduction from mode reordering, and it is the paper's best methodological instinct. Credit. **But the design as specified cannot evaluate it (§5.4).**

**The structural problem with the set.** `RESEARCH_GAP` §2.4 states the claim "dies if **all** of the following hold." A claim requiring three simultaneous failures to be falsified is not falsifiable in practice, and a reviewer will say so. **Fix:** make each condition individually decisive for its own sub-claim — F1 falsifies the variance claim, F2 the hard-subset claim, F3 the mechanism claim — and state which sub-claim's failure kills the paper versus which merely trims it. Under the §4.3 reframing, **F3 becomes the primary falsification condition**, which is the right outcome since it is the only well-posed one.

### 5.3 Are the downstream consequences demonstrated or asserted?

| Claimed consequence | Status | Cheapest demonstration | GPU cost |
|---|---|---|---|
| **Curated hard subsets** | **Demonstrated.** `J_config` vs. `J_seed` (Alg. 3) is a direct measurement. | — | 0 |
| **Adaptive test-time compute** | **Demonstrated, but only against a strawman.** B9 allocates budget in `c1` using difficulty estimated in `c0`. That is the authors' own allocator, not a published method. | **Re-implement Adaptive-Consistency's Dirichlet stopping rule and ESC's unanimity window offline on the corpus** — both are pure post-hoc rules over a sample sequence — and show that a stopping threshold calibrated in `c0` mis-stops in `c1` (excess samples spent, and accuracy lost at matched tokens). This names two published methods by name and is a re-analysis, not a run. **Do this; it is the single highest-value free addition to the applied claim.** | 0 |
| **IRT-style psychometric evaluation** | **Asserted; computable.** Currently a bare `\todo` in Analysis with no pre-registered statistic. | Fit a unidimensional 2PL to the item × (model, configuration) response matrix; compare against a model with configuration-specific item difficulties; report the fit difference and the variance share the unidimensional model absorbs as noise. **Pre-register the model, the fit statistic and the comparison now**, or it is post-hoc. Cite PromptEval (arXiv:2405.17202) as the antecedent that already fits an item × template IRT model — and be careful, because a reviewer may ask why PromptEval's pIRT is not simply the right tool here. | 0 |
| **Curriculum RL** | **Purely asserted, and not demonstrable within scope.** No RL is run (§1.6, correctly). | There is no honest free demonstration. The best available proxy: bin items into 5 difficulty buckets (as a curriculum would) and report the fraction reassigned to a different bucket across configurations, plus Kendall's τ on the ordering. That is a re-labelling of existing numbers. **But it is a proxy, not a demonstration — so drop curriculum RL from the abstract's claim list and mark it as a stated implication in Discussion.** arXiv:2606.19636 has already made the curriculum point along a different axis, so there is little to gain by overclaiming it. | 0 |

**Overall:** two of four consequences are genuinely measured, one is measurable and must be specified now, one should be demoted from claim to implication. That is a respectable position, and it is better than the abstract currently implies.

### 5.4 Is CDV's gain apples-to-apples? — and a design flaw that blocks F3

**On the mixture objection: the comparison is fair as an engineering claim, and under-identified as a mechanism claim.**

The reviewer's objection is that CDV takes the mode of a mixture `(1/C) Σ_c P_c(a | i)` while SC takes the mode of a single `P_{c0}`, so they are different estimands. As an *engineering* comparison this is not a problem: both are inference procedures with the same token cost, and "which procedure is more accurate at matched tokens" is a well-posed question. Credit the authors for enforcing token matching, which is stricter than most of the literature.

As a *mechanism* claim it is under-identified, because two hypotheses predict CDV > SC:

- **H_reorder** (the paper's): configurations reorder modes, and the mixture's mode is more often correct.
- **H_ensemble** (the boring one): a mixture beats its average component whenever components' errors are imperfectly correlated. This is ordinary bagging. It requires only that `P_c` differ across `c` — no interesting reordering, no small-margin subpopulation, no new mechanism.

P5 (CDV beats SC, temperature-diversified SC does not) discriminates *prompt* diversity from *temperature* diversity, which is worth having. It does **not** discriminate H_reorder from H_ensemble, because a critic will say the prompt mixture simply has less correlated components than the temperature mixture — which is H_ensemble with a stronger manipulation.

**Three free additions that do identify the mechanism:**

1. **An "oracle best single configuration" baseline.** This is missing from B1–B11 and its absence is damaging. B11 is oracle *per-item* configuration selection; there is no baseline for "pick the single globally best configuration and run SC in it." If CDV beats the *mean* single configuration but not the *best* single configuration, then the finding is "don't pick a bad prompt," which is much weaker and is essentially FormatSpread's advice. **Add this baseline. It is one line of analysis code and it is the comparison a reviewer will demand.**
2. **Decompose CDV's gain by category.** Split the items CDV gets right into (i) items where **no** single configuration's mode was correct — genuinely new modal mass, the only category that supports H_reorder — and (ii) items where **some** configuration's mode was correct — which is configuration selection, explicable by H_ensemble. Report the split. If category (i) is empty, H_reorder is dead and the paper must say so.
3. **A seed-pooled voting control.** Pool across the `c0` seed replicates at matched tokens. This is a mixture with *identical* components, so it must equal SC. It validates that the pooling machinery introduces no bias, and it is free (the replicates exist).

**Now the design flaw, which is more serious than the mixture objection.**

To show CDV's *plateau* exceeds SC's plateau, SC must have reached its plateau. It has not. With `N = 24` per cell and `C_use = 4`, CDV can be evaluated up to 96 pooled samples, while **SC is capped at 24**. At matched budget the only overlapping range is ≤ 24 total samples — where CDV draws **6 per configuration**, so each configuration's modal estimate is based on 6 draws. That is precisely the small-budget regime that `METHOD_SPEC` §5.6's own falsification hook declares uninformative ("the paper's Analysis section is required to show the large-`B_s` behaviour, not just the small-`B_s` regime where any variance reduction looks good"). And on MATH-500 the SC plateau is reported to arrive around `n ≈ 64`, well beyond 24.

**So F3 and P5 — the sharpest test in the design, and the primary falsification condition under the recommended reframing — cannot be evaluated by the experiment matrix as written.** Tier B's `O2` arm raises `N` to 64 but **only at `c0`, only on the 7B**, so CDV cannot be extended to match it.

Two fixes, and I recommend both:

- **Free:** compare *plug-in plateau estimates* rather than empirical `maj@n` curves. `π_mode(c0)` estimated from `N = 24` is an estimate of the `n → ∞` plurality limit; `π_mode(mixture)` estimated from the pooled 96 is the corresponding estimate for CDV. Comparing these two is legitimate and costs nothing, but it is a *different analysis* from an accuracy-vs-tokens curve and must be pre-registered as the **primary** test of F3, with the token-matched curve as secondary and explicitly labelled as small-budget.
- **Cheap, and I would fund it:** redirect `O2` from deep sampling at `c0` alone to deep sampling across `c0`–`c3` (§6.3). Same cost, and it is the difference between being able and unable to test the paper's sharpest prediction.

A final note on §8.4. Its argument that "raising `N` is a trap" is correct **for Algorithm 1** and **wrong for Algorithm 5**. `N` is exactly what the plateau claim needs, because the plateau is an asymptotic-in-`N` object. The spec over-generalises a valid point into a global rule and thereby starves its own best experiment.

---

## 6. Verdict

### 6.1 Simulated review

**Summary.** The paper asks whether per-item difficulty in LLM reasoning benchmarks is a property of the item or a model × elicitation-configuration interaction. It builds a fully crossed item × model × configuration design with independent-seed replicates, decomposes the variance of the Haldane-corrected empirical logit of per-item success rate with the binomial sampling floor subtracted, measures disattenuated difficulty transfer and bottom-quartile hard-subset stability against a seed-resampling null, and proposes a mechanism — configuration changes reorder the top competing answer classes — together with a zero-additional-cost intervention, configuration-diversified voting, predicted to raise the self-consistency plateau rather than merely reach it sooner. Six predictions are pre-registered with a stated negative-result path.

**Strengths.**

1. The **experimental hygiene is well above the norm** for this literature: a measured noise floor rather than an assumed one; matched **output-token** budgets rather than matched sample counts; item-clustered bootstrap with template-level resampling for GSM-Symbolic; extraction failure treated as a class rather than discarded; pre-registered predictions with a declared negative-result path.
2. **F3 / P5 is a genuinely discriminating pre-registered prediction.** Separating "diversity reduces variance" from "diversity moves the mode" by whether the *plateau* or only the *rate* changes is the right test, precisely stated in advance. This is the best idea in the document set.
3. The **gap identification against Bay & Yearick is exactly right**: they condition on a per-problem latent success rate, they show within-configuration run-to-run variance is ~0, and they never vary the configuration. That is a real and well-chosen hole.
4. The **compute plan is unusually honest** — a backend decision cascade rather than an assumption, a KV-cache fit table computed rather than guessed, a pre-costed fallback branch that is *cheaper* than the primary so that no budget pressure corrupts the scientific choice, and a precision control that pre-authorises the fallback before the fact. The reasoning in §1.5 for preferring fp16 (quantisation shrinks exactly the top-two margin the mechanism turns on) is a genuinely sophisticated piece of confound reasoning.
5. **Disattenuation is set up correctly in form** — the reliability denominator is a test–retest correlation between two single measurements, which is the right quantity. Many papers get this wrong.

**Weaknesses, ordered by severity.**

1. **[FATAL as framed] The headline measurement claim is published.** BrittleBench (arXiv:2603.13285, March 2026) decomposes performance variance into item-difficulty and semantics-preserving-prompt-sensitivity components, on an item × perturbation outcome matrix, over frontier and open-weight models and many benchmarks, and reports the prompt share at roughly half. TEE (arXiv:2604.11581) fits the crossed item/prompt/temperature/model random-effects model with an explicit item × prompt component by REML plus a D-study. arXiv:2606.19636 establishes that per-item math-reasoning difficulty labels are unstable and that this corrupts RL filters, curricula and hardness buckets. None is cited. The novelty audit that is offered as a deliverable missed the three nearest papers.
2. **[FATAL as designed] The sharpest prediction cannot be tested.** At `N = 24`, single-configuration self-consistency has not plateaued on MATH-500 (the plateau is reported near `n ≈ 64`), so "CDV raises the plateau" is unmeasurable at matched budget; CDV's reachable budget is 4× SC's, and the only overlapping regime is the small-budget one the spec itself declares uninformative. Tier B's deep-`N` arm is at `c0` only and cannot support the comparison.
3. **[MUST FIX] Two confounds can each manufacture the headline unaided, and both share its diagnostic signature.** Configuration-dependent answer extraction — `c3` switches to `\boxed{}`, the format `math-verify` is built around, and `⊥` is scored wrong and can become the modal class — and configuration-dependent truncation, where a "be rigorous" persona lengthens chains into `max_tokens = 1024` on the items with the longest solutions. Both are item-specific and reproducible across seeds, so the seed-replicate null cannot detect either. Reporting failure rates and flagging cells above 5% is monitoring, not control.
4. **[MUST FIX] The inferential core is the wrong estimator.** The moment-based decomposition subtracts a noise floor estimated in one configuration from mean squares in all configurations, when the floor is heteroscedastic by a factor of 12.8 across the `p` range; the estimand is a residual after removing ~80% of the mean square, so a 10% floor error is a 40% error in the answer; negative components are clamped, which inflates the shares of survivors; and the Haldane logit is censored at `±log(2N+1)`, so the reported variance shares are partly a function of `N` and are not comparable across cells with different `N`. The crossed-random-effects binomial GLMM that fixes all four is listed as optional and secondary.
5. **[MUST FIX] Pre-registration is looser than it appears.** F1–F3 kill the claim only in conjunction. F2's `≥ 0.90` Jaccard threshold requires ~94.7% coincidence of the bottom quartile and is unreachable given the seed-pair ceiling. P1 is satisfiable by 1 of ~6 cells with no multiplicity control, and F1/P1 leave an undeclared 5–10% dead zone. Holm correction is applied within one table only, against roughly 50–70 uncorrected primary tests.
6. **[MUST FIX] Two configuration axes are not what the design claims.** `c4` compares a 4-shot prompt against a zero-shot reference, so the axis carries task information, contradicting the stated design principle. `c5` (temperature) is a direct reparameterisation of the answer distribution, which makes it near-circular for a mechanism claim about distributional reordering — by the authors' own argument for rejecting 4-bit quantisation — and it is simultaneously excluded from CDV and embedded in the B4 control.
7. **[MUST FIX] CDV's gain is not identified as a mechanism.** No "oracle best single configuration" baseline exists, so the finding may reduce to "avoid a bad prompt." The gain is not decomposed into new modal mass versus configuration selection.
8. **[GRUMBLE] Statistical framing is over-claimed on generality.** With six configurations there are five degrees of freedom on the configuration facet; bootstrapping items does not license generalisation over configurations. Limitations calls the estimates "lower bounds," which they are not — they are conditional on a convenience sample of six, and could be higher or lower.
9. **[GRUMBLE] GSM8K is the wrong headline benchmark**, both statistically (saturation drives the noise floor to ~1.5, against an expected `σ²_αγ ≈ 0.04`) and for contamination (whose effect is item-specific, not the uniform inflation the spec's cancellation argument assumes).
10. **[GRUMBLE] Framing depends on an unreviewed two-author preprint in four places**, with `π_mode` and the identifiability gap defined by reference to it rather than from first principles.
11. **[GRUMBLE] Curriculum RL is asserted, not demonstrated**, and no RL is run. It should be an implication, not a claim.
12. **[GRUMBLE] `c2` may not exist as specified**; Qwen2.5's chat template injects a default system persona when none is given, and Llama-3's does not, so the configuration factor may not be the same object across the two families that carry the cross-family replication.

**Questions to authors.**

1. How does your contribution differ from BrittleBench (arXiv:2603.13285), which decomposes performance variance into task-difficulty and semantics-preserving-prompt-sensitivity components and reports the latter at roughly half?
2. Why a moment-based decomposition with floor subtraction and clamping rather than the crossed binomial GLMM, given that TEE (arXiv:2604.11581) fits exactly this model by REML and reports interaction components with <5% bias above ~1 000 observations?
3. Your variance shares are computed on a logit censored at `±log(2N+1)`. How do you establish that the item × configuration share is not partly a function of `N`, and how do you compare cells at `N = 24` with the `N = 64` arm?
4. `c3` switches to `\boxed{}`, which is the format your equivalence checker is built around, and `⊥` is scored as wrong. What is the per-configuration extraction precision, and does the headline survive under a second independent extractor, under `⊥`-dropped scoring, and restricted to bare-integer gold answers?
5. What is the per-configuration truncation rate at `max_tokens`, and does the "be rigorous" persona lengthen chains? What fraction of `c1`'s accuracy difference from `c0` is attributable to truncation rather than reasoning?
6. `π_mode` is an asymptotic-in-`N` quantity. At `N = 24`, has single-configuration self-consistency plateaued on MATH-500? If not, on what basis is P5 evaluated?
7. Does CDV beat the *best single* configuration at matched tokens, or only the average one? What fraction of its gain comes from items where no single configuration's mode was correct?
8. Under your permutation test the configuration main effect and the interaction are destroyed together. How do you attribute rejection to the interaction?
9. F2 requires ~94.7% coincidence of the bottom quartile. Given `J_seed < 0.90` at `r_mm ≈ 0.95`, under what circumstances can F2 fire?
10. What is your rendered prompt for `c2` on Qwen2.5 and on Llama-3.2? Are they the same manipulation?

**Score.**

- **As currently framed and designed: ICLR 4 — reject, marginally below the acceptance threshold. Confidence 4.** The overlap with BrittleBench and TEE is dispositive for the headline claim, and the inability to evaluate F3 removes the paper's best result.
- **With the §4.3 reframing, the GLMM as the primary estimator, the parsing and truncation controls, and the multi-configuration deep-`N` arm: ICLR 6 — marginally above the acceptance threshold, possibly 7 if the mode-reordering result is clean and the corrective/destructive net is positive. Confidence 4.** This is a good, careful, honest study of the right object; it is currently pointed at the wrong claim.

### 6.2 The single most important change before any GPU time is spent

**Reframe the paper around `π_mode` — the configuration-relativity of the self-consistency ceiling — instead of around item difficulty, and cite BrittleBench and TEE as the established premise rather than competing for it.**

This costs zero GPU hours, uses the same corpus and the same figures in a different order, converts the two most damaging overlaps from competitors into foundations, promotes the two most discriminating predictions (P4, P5) to primary, and demotes the prediction most at risk from both the overlap and the censoring artefact (P1) to supporting evidence. It is the only change that alters the paper's fate rather than its polish.

Two changes are close seconds, and both must also happen before the run because they determine what gets logged and analysed:

- **Make the crossed binomial GLMM the primary estimator** (§2.1c) and the moment decomposition the appendix robustness check. Cost: CPU minutes.
- **Run the parametric-bootstrap null calibration on synthetic data now** (§2.2, item 1), before any GPU hour. It will tell you whether this design can detect the effect you expect and whether your pipeline returns a non-zero interaction share under a strict null. If the minimum detectable effect exceeds a plausible effect size, you have learned that for free, which is the best possible outcome of this review.

### 6.3 Experiment matrix: add and drop

**Drop — and this pays for the additions:**

| Change | Rationale | Saving |
|---|---|---|
| **Remove `c5` (temperature) from the core configuration factor.** Report separately, or fold into B4 where `T ∈ {0.6, 0.8, 1.0}` already lives. | Near-circular for a mechanism claim about distributional reordering, by the authors' own §1.5 argument against 4-bit; already excluded from CDV's default `C_use = 4`; already inside the B4 control. | `E1a/b/c`: 1/6 of 7.9 h ≈ **1.3 h**; `PC`: 1/6 of 2.4 h ≈ **0.4 h**. Total ≈ **1.7 h** |
| **Demote GSM8K from headline to secondary replicate** (keep the cells; change what leads). | Saturation drives `σ²_samp` to ~1.5 against an expected `σ²_αγ ≈ 0.04`; contamination is item-specific, so the cancellation argument fails. | 0 h (analysis change) |
| **Demote `c3` and `c4` to separately reported axes** alongside `c6`, unless `c4a` is added. | `c3` is not parse-invariant; `c4` carries task information. | 0 h (analysis change) |

**Add:**

| Change | Rationale | Cost |
|---|---|---|
| **Replace Tier B `O2` with a multi-configuration deep-`N` arm.** Currently: 7B, MATH-500, `c0` only, 400 items, `N: 24 → 64` (1.7 h). Instead: **3B, MATH-500, `c0`–`c3`, 200 items, `N: 24 → 64`** — 4 × 200 × 40 × 550 ≈ 17.6 M tokens at ~2 800 tok/s ≈ **1.75 h**. | **This is the difference between being able and unable to test F3/P5**, the paper's sharpest prediction and (under the reframing) its primary falsification condition. `O2` as specified cannot support CDV because it raises `N` in one configuration only. | **≈ 0 h net** (redirection) |
| **Seed replicates at one non-reference configuration.** Two extra seeds at `c3` (or `c1`), 3B, MATH-500, 200 items, `N = 24`. | `r_mm` is currently measured **only at `c0`** and assumed transportable, but reliability is `p`-dependent and configurations differ in accuracy. This is the denominator of the headline `ρ_disatt`. Without it, the disattenuation is an untested assumption. | ≈ **0.4 h** |
| **Verify `max_model_len` headroom for `c4` in the smoke test**; raise to 3072 for few-shot cells if the longest 4-shot prompt plus `max_tokens` does not fit. | Silent context clipping is a configuration-dependent confound. | ≈ 0 h |
| **Log `finish_reason`, rendered prompt string and token IDs** for every sample and cell. | Required for the truncation and chat-template controls (§3.2, §3.5). Cannot be recovered after the fact. | 0 h |
| **Analysis-only additions (zero GPU):** parametric-bootstrap null calibration; crossed binomial GLMM; second independent extractor with agreement table; stratified extraction audit (~100 samples/config); `⊥`-dropped and integer-only sensitivity analyses; degenerate-output/refusal/repetition rate table; per-configuration truncation rates; oracle-best-single-configuration baseline; CDV gain decomposition (new modal mass vs. configuration selection); seed-pooled voting control; difficulty-stratified `ρ_disatt`, `r_mm` and `J_config`; tie-mass reporting at the quartile cut; corrected within-item permutation test; offline Adaptive-Consistency and ESC mis-stopping demonstration; pre-registered IRT fit comparison. | Each closes a specific objection in §2–§5. All are code and pre-registration, not GPU. | 0 h |

**Net GPU effect: approximately −1.3 h.** The critique makes the plan cheaper, not more expensive. There is no budget argument against any of it.

### 6.4 Severity triage

**Fatal — do not run as designed.**

1. The headline claim overlaps BrittleBench (arXiv:2603.13285) and TEE (arXiv:2604.11581) to the point of pre-emption. **Fatal to the framing, not to the corpus** — fixed by the §4.3 reframing at zero cost, but it must be fixed *before* the run, because the reframing changes which predictions are primary and therefore what must be logged and pre-registered.
2. F3/P5 cannot be evaluated at `N = 24` with `C_use = 4` at matched budget. **Fatal to the paper's best result** — fixed by redirecting `O2` (§6.3) at roughly zero net cost, plus pre-registering the plug-in-plateau comparison as primary.

**Must fix, cheap to fix now.**

3. Configuration-dependent answer extraction (`c3` / `⊥` / `math-verify`). Two extractors, stratified audit, three sensitivity analyses, demote `c3`. Zero GPU.
4. Configuration-dependent truncation and stop-string firing. Log `finish_reason`; report rates; sensitivity analysis; check `c4` context headroom. Zero GPU.
5. Moment-based decomposition with a transported, heteroscedastic noise floor, clamping, and an `N`-dependent censoring bound. Promote the crossed binomial GLMM to primary. Zero GPU.
6. No null calibration for the interaction share. Run the parametric bootstrap on synthetic data **before** the GPU run. Zero GPU, and it may save the whole 46 hours.
7. Pre-registration looseness: F1–F3 in conjunction; F2 unsatisfiable; P1 disjunctive; 5–10% dead zone; multiplicity uncontrolled outside T5. Rewrite now. Zero GPU.
8. `c4` carries task information; `c5` is circular. Demote or restructure. Budget-positive.
9. CDV mechanism under-identified: add the oracle-best-single-configuration baseline and the gain decomposition. Zero GPU.
10. Permutation test targets the joint null rather than the interaction; quartile ties unhandled. Zero GPU.
11. `r_mm` measured only at `c0`. Add seed replicates at one more configuration, ≈ 0.4 h.
12. `c2` may be Qwen's default persona rather than an empty system prompt; log rendered prompts. Zero GPU.
13. Missing citations: BrittleBench, TEE, arXiv:2606.19636, arXiv:2607.13304, arXiv:2603.01865, PromptEval (arXiv:2405.17202), Mizrahi et al. TACL 2024, Alzahrani et al. 2024. Chase `wang2025measuring`. Zero GPU.
14. Bay & Yearick over-reliance: define `π_mode` and the identifiability gap self-containedly; co-cite peer-reviewed sources; reduce from four load-bearing uses to one or two. **Also: independently verify arXiv:2606.28661 before submission** — this review did not close that item. Zero GPU.

**Reviewer will grumble; acceptable.**

15. Six configurations give five degrees of freedom on the configuration facet, so nothing generalises over configurations. Correct the "lower bound" wording in Limitations to "conditional on our six configurations."
16. GSM8K contamination and saturation. Demote to secondary; state the item-specific-contamination risk honestly.
17. Curriculum RL asserted, not demonstrated. Move to Discussion as an implication; report the difficulty-bucket reassignment proxy.
18. Model scale ceiling at 7B–8B. Already handled by the ladder-plus-family argument, which is the right answer.
19. Spearman disattenuation is not classically licensed. Superseded automatically if the GLMM becomes primary.
20. Tier B item expansion (200 → 400) is a luxury for the interaction term, which already has 995 degrees of freedom. Keep it — it helps the transfer and hard-subset statistics — but do not describe it as necessary for the variance components.

---

## 7. Verification status of this review

**Verified by direct fetch this session:** arXiv:2603.13285 (BrittleBench — abstract and body, including the variance-decomposition equations, the perturbation taxonomy, the brittleness-score definitions, the ~50% headline, and the test-time/CoT arm); arXiv:2604.11581 (TEE — abstract, model equation, D-study formulas, expected-mean-square table, MMLU decomposition, the `α_i ⊥ (αφ)_iv` assumption passage, and the REML convergence note); arXiv:2603.01865 (CyclicJudge — abstract and decomposition passage); arXiv:2607.13304 (facet decomposition, D-study allocation, the "repeats are the cheap facet" argument); arXiv:2606.19636 (abstract and key body passages); arXiv:2405.17202 (PromptEval — abstract and body). BrittleBench's identity and headline are additionally corroborated by TEE's reference list, which cites it by ID with a matching summary.

**Not verified this session, flagged as open:** arXiv:2606.28661 (Bay & Yearick). A dedicated verification pass covering this and the other 2026 IDs in `RESEARCH_GAP.md` and `CITATION_AUDIT.md` was dispatched and had not returned when this document was written. **Close this before submission.** Nothing in this review's substantive findings depends on Bay & Yearick being real or correct; the §4.3 reframing does depend on it being a citable object, so it is on the critical path for the reframing rather than for the critique.

**Not verified, cited from the reviewed documents' own claims:** Mizrahi et al. TACL 2024; Alzahrani et al. 2024; `wang2025measuring`; and all pre-2026 IDs already audited in `docs/CITATION_AUDIT.md`.

**Arithmetic in this review** (censoring bounds `±log(2N+1)`; the 12.8× heteroscedasticity ratio between `k=0` and `k=12` at `N=24`; the `p^{432}` exclusion-rule table; `df = 995` and the `√(2/df) = 4.5%` relative standard error; the `J = 0.90 ⇒ 94.7%` coincidence requirement; the token and hour costs in §6.3) was computed here and is reproducible from the stated inputs. It should be re-checked independently before being quoted back to a reviewer, since it is load-bearing for several of the recommendations.
