# METHOD SPECIFICATION

**Project:** Is the self-consistency ceiling (π_mode) a model constant, or a model × elicitation property—and can spreading budget across configurations raise the plateau i.i.d. resampling cannot?
**Reframing (binding):** see `docs/REFRAMING.md`. BrittleBench/TEE are cited as foundations; headline is π_mode + CDV, not variance-decomposition novelty.
**Audience:** the code agent building the experimental harness.
**Status of this document:** normative. Where a value is given as a default, use it unless a listed contingency fires.
**Compute envelope:** Kaggle free tier, 2× NVIDIA T4 16 GB (preferred) or 1× P100 16 GB, ~29 GB RAM, ~57 GB disk, internet ON.
**Quota model (verified):** the ~30 h weekly allowance is billed by **session wall-clock time with an accelerator enabled**, *not* per GPU device — "GPU T4 × 2" for one hour costs one hour. Session cap is 12 h; plan against 9 h. The allowance resets weekly. Quota is consumed whenever the accelerator is enabled **even if idle**, so all debugging happens with the accelerator off (§4.1.4). Consequences for parallelism are in §4.1.3 and for the budget in §8.

**Backend risk (unresolved from outside Kaggle):** vLLM has effectively deprecated Turing/sm_75, which is what a T4 is. The spec does not assume vLLM works; §4.1.1 specifies a smoke test and a five-step fallback cascade that the first session must run before anything else. §1.5 sets the precision policy (fp16 default, uniform 4-bit as a pre-authorised fallback) and §8.3.1 pre-costs the fallback branch.

> **Verification legend.** ✅ = I fetched the HuggingFace page and confirmed it resolves. ⚠️ = plausible but the code agent MUST confirm at runtime before relying on it, and fall back as directed. Do not silently substitute a different ID.

---

## 0. One-paragraph summary of what we are building

For a grid of (item × model × elicitation-configuration) cells we sample `N` chains-of-thought, extract and canonicalise the final answer, and record a success count. Everything else — **per-configuration modal ceilings π_mode**, mode-reordering rates, configuration-diversified voting (CDV), and supporting measurement (crossed binomial GLMM variance decomposition, disattenuated transfer, hard-subset Jaccard) — is computed **offline from that single sampling corpus**. There is no training and no additional generation for CDV. Budget GPU time for generation; budget CPU time for analysis (GLMM, parametric-bootstrap null, extraction audit).

---

## 1. Models

### 1.0 Per-rung VRAM fit (the table everything else depends on)

Architecture figures below are **verified** from each model's `config.json` / model card. KV-cache size is computed, not guessed:

```
KV bytes per token = 2 (K and V) x n_layers x n_kv_heads x head_dim x 2 bytes (fp16)
```

| Model | Params | Layers | Q heads | KV heads | head_dim | KV/token | Verified |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.54 B | 28 | 12 | 2 | 128 | **28 KiB** | ✅ model card |
| `Qwen/Qwen2.5-3B-Instruct` | 3.09 B | 36 | 16 | 2 | 128 | **36 KiB** | ✅ model card |
| `Qwen/Qwen2.5-7B-Instruct` | 7.61 B | 28 | 28 | 4 | 128 | **56 KiB** | ✅ `config.json` |
| `meta-llama/Llama-3.2-3B-Instruct` | 3.21 B | 28 | 24 | 8 | 128 | **112 KiB** | ✅ `config.json` |
| `meta-llama/Llama-3.1-8B-Instruct` | 8.03 B | 32 | 32 | 8 | 128 | **128 KiB** | ✅ config |

**Note the Llama-3.2-3B trap.** It has 8 KV heads against Qwen2.5-3B's 2, so despite near-identical parameter counts its KV cache is **4× larger per token**. This is not a footnote — it is what caps its batch size and therefore its throughput, and the original budget missed it.

Fit is evaluated against a **14.4 GiB** budget (`gpu_memory_utilization=0.90` of a 16 GB T4), at `max_model_len=2048` and a target of **48 concurrent sequences** (2 prompts in flight × `n=24`), plus ~0.5–0.8 GiB for activations and CUDA graphs. All figures GiB.

| Rung | fp16 weights | KV @ 48×2048 | Overhead | Total | Fits one T4? |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 2.87 | 2.63 | 0.5 | **6.0** | ✅ large margin |
| Qwen2.5-3B | 5.75 | 3.38 | 0.6 | **9.7** | ✅ comfortable |
| Llama-3.2-3B | 5.98 | 10.50 | 0.6 | **17.1** | ❌ at 48 seqs — ✅ at 24 seqs (**11.8**) |
| Qwen2.5-7B | 14.17 | 5.25 | 0.8 | **20.2** | ❌ **no** |
| Llama-3.1-8B | 14.95 | 12.00 | 0.8 | **27.8** | ❌ **no** |

And the two-card / quantised variants:

| Configuration | Per-card total | Verdict |
|---|---|---|
| Qwen2.5-7B fp16, TP-2 (weights 7.09, KV halved to 28 KiB/tok → 2.63) | **10.5** | ✅ comfortable headroom |
| Qwen2.5-7B AWQ 4-bit, single card (weights ≈5.6 incl. fp16 `lm_head`) | **11.65** | ✅ fits, could raise concurrency to 64 |
| Llama-3.1-8B fp16, TP-2 (weights 7.48, KV 64 KiB/tok → 6.00) | **14.3** | ⚠️ within 0.1 GiB of budget — **marginal** |

**Conclusions.** Three rungs (1.5B, 3B, Llama-3.2-3B) fit one card in fp16 and get genuine DP-2. Qwen2.5-7B needs either TP-2 or 4-bit. Llama-3.1-8B is marginal even under TP-2 and is **demoted to Tier C** (§1.2).

### 1.1 Primary size ladder (Qwen2.5 instruct family) — CORE

| Role | HF ID | Params | Default precision | Mode |
|---|---|---|---|---|
| Small | `Qwen/Qwen2.5-1.5B-Instruct` | 1.54 B | fp16 | DP-2 |
| Medium | `Qwen/Qwen2.5-3B-Instruct` | 3.09 B | fp16 | DP-2 |
| Large | `Qwen/Qwen2.5-7B-Instruct` | 7.61 B | fp16 | **TP-2** (see §1.5 for the fallback) |

All three are `Qwen2ForCausalLM`; the 7B's `config.json` was written by `transformers 4.43.1`, which matters for §4.1.

If the ladder must shrink, drop the **3B** — keeping 1.5B and 7B preserves the widest span. Never drop the 7B unless step S2(e) of the §4.1.1 cascade fires.

### 1.2 Cross-family control — CORE (revised)

| Role | HF ID | Params | Precision | Tier | Verified |
|---|---|---|---|---|---|
| Family-2 | `meta-llama/Llama-3.2-3B-Instruct` | 3.21 B | fp16, DP-2 @ ≤24 seqs | **A** | ✅ config; ⚠️ gated repo |
| Family-2 large | `meta-llama/Llama-3.1-8B-Instruct` | 8.03 B | fp16, TP-2 | **C** (demoted) | ✅ config; ⚠️ gated |

**Why Llama-3.1-8B moved to Tier C.** Under TP-2 it lands at 14.3 GiB/card against a 14.4 GiB budget — a 0.7 % margin, which will fail the moment vLLM's profiler allocates slightly differently or a prompt runs long. Its scientific job is to show the effect is not Qwen-specific, and **Llama-3.2-3B already does that job**. What the 8B additionally buys is a cross-family *size trend*, which is a nice-to-have, not a load-bearing claim. Demoting it removes one of the two problem cells at almost no scientific cost, and it removes the second TP-2 dependency.

**Contingency if the Llama gate blocks the download.** Substitutes must be compatible with the possibly-pinned `transformers==4.46.3` (§4.1), which rules out most of the previously listed options:

| Substitute | Arch | Min `transformers` | Compatible with the pin? |
|---|---|---|---|
| `microsoft/Phi-3.5-mini-instruct` (3.8 B) ⚠️ | `Phi3ForCausalLM` | 4.41 | ✅ **preferred substitute** |
| `mistralai/Mistral-7B-Instruct-v0.3` ⚠️ | `MistralForCausalLM` | 4.34 | ✅ but 14.5 GB fp16 — needs 4-bit |
| ~~`microsoft/Phi-4-mini-instruct`~~ | Phi3 variant | ≥ 4.48 | ❌ **removed** |
| ~~`google/gemma-3-4b-it`~~ | `Gemma3ForCausalLM` | ≥ 4.50 | ❌ **removed** |
| ~~`mistralai/Ministral-8B-Instruct-2410`~~ | Mistral variant | ~4.46, borderline | ❌ **removed**, too risky |

The scientific requirement is only that family 2 has a **different pretraining lineage** from Qwen; record which was used in T1.

### 1.3 Long-CoT arm — TIER C

| Role | HF ID | Min `transformers` | Status |
|---|---|---|---|
| Distilled long-CoT | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 4.37 (Qwen2 arch) | ✅ compatible with the pin |
| ~~Hybrid thinking~~ | ~~`Qwen/Qwen3-4B` / `Qwen3-4B-Thinking-2507`~~ | **≥ 4.51.0** | ❌ **DROPPED** — see below |

**Qwen3 is incompatible with the pinned stack.** Qwen3 support landed in `transformers 4.51.0`; earlier versions raise `KeyError: 'qwen3'` at config load. If the §4.1 smoke test finds that a *modern* vLLM works on T4, the pin is unnecessary and Qwen3-4B can be restored. If the pin is required, the hybrid-thinking configuration axis is unavailable and the long-CoT arm reduces to the R1 distill alone. This is a Tier C loss only.

### 1.4 `transformers==4.46.3` compatibility — consolidated verdict

The pin only applies if the §4.1.1 smoke test lands on S0b. **Verdict: the pin does not constrain the core ladder at all.** It costs us only Tier C's hybrid-thinking arm and two of the three ungated Llama substitutes.

| Model | Architecture | Min `transformers` | Under the pin |
|---|---|---|---|
| Qwen2.5-1.5B / 3B / 7B-Instruct | `Qwen2ForCausalLM` | 4.37 | ✅ (configs written by 4.43.1) |
| Qwen2.5-\*-Instruct-AWQ | `Qwen2ForCausalLM` + AWQ | 4.37 | ✅ (AWQ configs written by 4.46.1) |
| Llama-3.2-3B-Instruct | `LlamaForCausalLM`, llama3 RoPE | **4.43.0** | ✅ |
| Llama-3.1-8B-Instruct | `LlamaForCausalLM`, llama3 RoPE | **4.43.0** | ✅ |
| DeepSeek-R1-Distill-Qwen-1.5B | `Qwen2ForCausalLM` | 4.37 | ✅ |
| Phi-3.5-mini-instruct (substitute) | `Phi3ForCausalLM` | 4.41 | ✅ |
| **Qwen3-4B / Qwen3-4B-Thinking** | `Qwen3ForCausalLM` | **4.51.0** | ❌ `KeyError: 'qwen3'` |
| Phi-4-mini-instruct | Phi3 variant | ≥ 4.48 | ❌ |
| Gemma-3-4b-it | `Gemma3ForCausalLM` | ≥ 4.50 | ❌ |

The margin is comfortable rather than lucky: every core model predates November 2024, and `transformers 4.46.3` is a November 2024 release. **The code agent must still assert this at runtime** — log `transformers.__version__` and the resolved `model_type` for every model before the first generation, and fail loudly rather than silently falling back to a remote-code path.

### 1.5 Numerical precision: policy, and why fp16 is the default

**Policy: fp16 is the default and the reported headline. 4-bit is a pre-authorised fallback that, if it fires, applies uniformly to the entire ladder — never to one rung.**

The engineering case for going 4-bit everywhere is genuinely strong: every rung would fit one card, every cell would get DP-2, the 7B would actually run *faster* than under fp16 TP-2 (§8.0), and it would sidestep the vLLM-on-Turing problem entirely. The objection is not that quantising only the top rung confounds size with precision — that objection does dissolve under a uniform ladder, correctly. The objection is narrower and, we think, decisive.

**4-bit quantisation perturbs the exact quantity the mechanism claim is about.** Our mechanism (§5.5) is that configuration changes reorder the modal answer on items whose **top-two answer-class margin** is small. Quantisation raises perplexity, which flattens next-token distributions, which shrinks answer-class margins, which moves items *into* the susceptible zone. So a 4-bit ladder would be expected to show **more** mode reordering and a **larger** item × configuration interaction than an fp16 ladder — an inflation in the direction of our hypothesis.

This is worth contrasting with the contamination argument in §2, where we argue that a uniform accuracy inflation cancels because our claim is relative. **That argument does not transfer here.** Contamination shifts `p̂` roughly uniformly across items; quantisation shifts it *differentially*, concentrated on precisely the small-margin subpopulation that carries the mechanism. A confound that is both directional and concentrated on the critical subpopulation is the one we should not adopt voluntarily when an alternative exists.

**But we test it rather than merely avoid it.** The precision control (PC) below runs in Tier A regardless of which branch fires, and it does double duty: it is a robustness result in its own right, and it pre-authorises the 4-bit fallback so that, if we need it, the licence was earned before the fact rather than argued after it.

#### Precision control (PC) — Tier A, runs unconditionally

Run the **full main grid at 4-bit AWQ** for `Qwen2.5-1.5B-Instruct-AWQ` and `Qwen2.5-3B-Instruct-AWQ` ⚠️ on MATH-500 only, 200 items, all **3 core** configurations (c0–c2), `N=24`, seed 0. The fp16 counterpart already exists from E1a/E1b, so this yields an exactly paired comparison. Compute the **entire analysis pipeline** at both precisions: variance components, `ρ_disatt`, `r_mm`, `J_config`, `π_mode`, the reorder rate, and the top-two margin distribution.

Report as Figure F8 (`fig8_precision_control.pdf`) and Table T7 (`tab7_precision_control.csv`).

Why 1.5B and 3B are the *right* place for this control, not a compromise: quantisation damage is relatively **worse** in smaller models, which have less parameter redundancy to absorb it. Demonstrating invariance at 1.5B is therefore evidence *a fortiori* for the 7B. The extrapolation runs in the favourable direction, which is unusual and worth stating explicitly in the paper.

**Pre-registered prediction P7** (added to §10): the item × configuration variance share differs by **< 3 percentage points** between fp16 and 4-bit, and `ρ_disatt` differs by **< 0.05**, on both rungs.

- **P7 supported** → the 4-bit branch is licensed. If the 7B cannot run at fp16, re-run the whole ladder at 4-bit and report PC as the control.
- **P7 refuted** → **4-bit is off the table entirely.** If the 7B cannot run at fp16, drop the 7B rung and report a 1.5B/3B/Llama-3.2-3B ladder, stating why. We commit to this in advance so the precision decision cannot become a convenience.

Note the asymmetry this creates, deliberately: the fallback branch is *cheaper* than the primary (§8.3.1), so there is no budget pressure pushing us toward it. We prefer fp16 on scientific grounds while knowing it is the slower option.

### 1.6 Explicitly out of scope

No PRM, no reward model, no fine-tuning, no LoRA. The paper's claims do not require them and the budget does not permit them.

---

## 2. Datasets

| HF dataset ID | Split | Full size | Subsample | Verified | Why |
|---|---|---|---|---|---|
| `HuggingFaceH4/MATH-500` | `test` | **500** ✅ | **200** → 400 | ✅ (page fetched: 500 rows) | **Headline benchmark** — mid/hard range, right noise floor for GLMM interaction, SC plateau near n≈64 |
| `openai/gsm8k` (config `main`) | `test` | 1319 | **200** → 400 | ⚠️ | **Secondary replicate** — saturation inflates noise floor; contamination is item-specific, not uniformly cancelling |
| `apple/GSM-Symbolic` (configs `main`, `p2`) | see card | templated | **150** → 300 per variant | ⚠️ | Distribution-shift arm; template-matched variants let us test whether the interaction structure survives perturbation |

The first number is the Tier A item count (§8.1); the second is the Tier B expansion (§8.2). MATH-500 has only 500 items in total, so 400 is 80 % of the corpus and is the practical maximum.

**Subsampling protocol (must be deterministic and recorded).** Sort items by their dataset index, then draw a fixed pseudo-random permutation with `numpy.random.default_rng(20260729)`. **Tier A takes the first 200 of that permutation; Tier B takes the next 200.** Persist the selected indices, tagged by tier, to `data/item_ids_{dataset}.json` and commit that file to the run manifest. Drawing both batches from one permutation means the Tier B items are a fresh independent sample from the same population, so the two batches can simply be pooled — no reweighting, and no risk of the expansion being a biased top-up. Every model/configuration cell within a tier must use the **identical** item set; the entire variance decomposition depends on a fully crossed design.

**Why these counts.** Items are the bootstrap resampling unit, so every confidence interval scales as `1/sqrt(I)`. At 200 items the item-clustered bootstrap CI on an accuracy difference is roughly ±0.05 at 95 %, adequate for the effect sizes we expect (hard-subset overlap changes of 10–40 points); at 400 it is ≈29 % tighter. Item count is the most efficient axis to expand, and **it must be expanded before `N`** — see §8.4 for the arithmetic showing why raising `N` buys much less than it appears to.

**Contamination.** GSM8K is known to be contaminated (Zhang et al.'s GSM1k work). Contamination is **item-specific**, not uniform — a persona or format change can disrupt memorisation cues and manufacture item×configuration interaction. **Lead with MATH-500 and GSM-Symbolic**; report GSM8K as a secondary replicate and state the memorisation×configuration risk plainly. GSM-Symbolic remains the contamination-resistant arm.

---

## 3. Elicitation configurations

A **configuration** `c` is a semantics-preserving perturbation of everything except the item's mathematical content. Configurations split into a **core factor** (used in the main crossed grid and CDV) and **separate axes** (reported alongside, not in the headline pipeline).

### 3.1 Core factor (3 levels) — used in E1, CDV default `C_use=3`

| ID | Axis | Setting |
|---|---|---|
| `c0` | *reference* | Default chat template, neutral system prompt (`"You are a helpful assistant."`), zero-shot CoT instruction `"Solve the problem step by step. Put your final answer after 'Answer: '."`, `T=0.8`, `top_p=0.95` |
| `c1` | System-prompt persona | System prompt replaced with `"You are an expert mathematician. Be rigorous."` — everything else as `c0` |
| `c2` | System-prompt minimality | **Rendered per model family** (see §3.3) — not assumed identical across Qwen and Llama |

**Dropped from core:** `c5` (temperature) — circular for mode-reordering claims; folded into baseline B4 only (~1.7 h saved vs.\ the pre-review 6-config grid). **Demoted to separate axes:** `c3` (parse/format), `c4`/`c4a`/`c4b` (few-shot and exemplar-order contrasts).

### 3.2 Separate axes (not in headline GLMM / F1; reported in appendix tables)

| ID | Axis | Setting | Why separate |
|---|---|---|---|
| `c3` | Answer-format instruction | Final-answer instruction → `"End your response with \boxed{}."` | Not parse-invariant w.r.t. `math-verify` |
| `c4` / `c4a` / `c4b` | Few-shot and order | 4-shot CoT (`c4` vs.\ `c0`); forward (`c4a`) vs.\ reverse (`c4b`) exemplar order at fixed shot count | Carries task information (worked solutions) |
| `c5` | Decoding temperature | As `c0` but `T=1.0`, `top_p=1.0` | Reparameterises answer distribution; used in B4 only |
| `c6` | Question paraphrase | Item text paraphrased once offline (7B, `T=0.3`) with numeric-preservation check | Rewrites problem statement |

`c6` supports baseline B5 (Self-Para-Consistency). Run `c6` in Tier B (O3) as today.

**Paraphrase validity gate for `c6`.** Reject and regenerate (max 3 attempts) any paraphrase whose numeric literal multiset differs from the original. Log rejection rate.

### 3.3 `c2` rendered prompts (mandatory smoke-test deliverable)

Persist the **fully rendered prompt string and token IDs** for every (model, configuration) pair. Diff and report in appendix.

| Model family | `c2` implementation | Rationale |
|---|---|---|
| **Qwen2.5** | If the chat template injects the default system message (`"You are Qwen, created by Alibaba Cloud…"`) when none is supplied, either (a) force an explicit empty system turn in the API, or (b) redefine `c2` in T1 as `"family-default system prompt"` — do not label it "empty" if it is not | Qwen default-system injection makes `c2` ≠ "no persona" |
| **Llama-3.2** | Omit system message if the template allows; else explicit minimal system string documented in T1 | Cross-family `c2` is not the same manipulation unless rendered strings are audited |

If rendered `c2` differs materially from the intended axis, report `model × configuration` three-way sensitivity with `c2` excluded from the headline core set `{c0,c1}`.

### 3.4 Seeds

**Tier A:** `c0` with seeds `{0,1,2}` on all three Qwen models (noise floor + `r_mm` at reference config).  
**Tier B:** seeds 3→5 at `c0` for 1.5B/3B on all 400 items; **plus O2b:** two extra seeds at **`c1`**, 3B, MATH-500, 200 items, `N=24` (reliability transportability).  
All other core configurations use seed `0` only.

---

## 4. Generation

### 4.1 Engine, and the Turing (sm_75) problem

**The single largest execution risk in this project is that vLLM may not run on a T4 at all.** T4 is compute capability 7.5 (Turing). Recent vLLM releases have effectively deprecated it, and there are two distinct documented failure classes:

1. **CUTLASS DSL cannot detect sm_75** during engine init, producing `-arch=compute_ is an unsupported option`. Reported against vLLM 0.15.1. `--enforce-eager`, `VLLM_USE_V1=0` and `TORCH_CUDA_ARCH_LIST=7.5` are all reported *not* to work around it.
2. **Triton shared-memory failures on Turing** (64 KB shared memory per SM) for large head dimensions.

Separately, FlashAttention-2 requires sm_80+, so it never runs on T4; XFORMERS is the attention backend. Marlin INT4 kernels also require sm_80+, so any 4-bit path must select the plain `awq` (GEMM) or `gptq` kernel.

The known-working Kaggle recipe pins **`vllm==0.7.3` with `transformers==4.46.3`**, uses XFORMERS, and runs **two single-GPU replicas with round-robin load balancing** rather than tensor parallelism. That last detail is a warning: it suggests TP-2 inside a Kaggle notebook is itself a friction point, independent of the sm_75 issue.

**We cannot resolve any of this from outside Kaggle. The first session must settle it empirically.** The spec therefore does not commit to a backend; it commits to a decision procedure.

#### 4.1.1 Smoke test and backend cascade (normative, ~30 min, accelerator ON)

Run this before any production cell. Record every outcome in `runs/_backend_probe.json` and reproduce it in T1 — which branch fired is a reportable fact about the experiment, not an implementation detail.

**S0 — Does vLLM initialise at all?**

```
S0a. Try the vLLM already present in the Kaggle image, with current transformers.
     LLM(Qwen2.5-1.5B-Instruct, dtype="float16", tensor_parallel_size=1,
         gpu_memory_utilization=0.90, max_model_len=2048)
     Generate n=8 for 4 prompts. PASS => record the version; NO PIN NEEDED.
     (This is the good branch: Tier C's Qwen3-4B stays available.)

S0b. If S0a fails with a CUTLASS/sm_75 or Triton shared-memory error:
     pip install vllm==0.7.3 transformers==4.46.3, restart the runtime, retry S0a.
     PASS => record PINNED MODE. Qwen3-4B and the Phi-4/Gemma-3 substitutes
     are now unavailable (see 1.2, 1.3); the CORE LADDER IS UNAFFECTED.

S0c. If S0b also fails => vLLM is unusable. Go to S3.
```

**S1 — Is DP-2 real?** Launch two single-GPU vLLM processes (`CUDA_VISIBLE_DEVICES=0` and `=1`), each generating half of a 200-prompt list, and compare aggregate tokens/s against one process on the full list.

- Speedup **≥ 1.7×** → DP-2 confirmed. This is the mode for 1.5B, 3B and Llama-3.2-3B.
- Speedup **< 1.7×** → the sharding harness is at fault, not the hardware. Fix the harness. Do **not** re-plan the budget around a bad measurement.

**S2 — The 7B strategy, tried in this order.** Stop at the first that passes.

```
(a) fp16 TP-2:  LLM(Qwen2.5-7B-Instruct, dtype="float16", tensor_parallel_size=2)
    PREFERRED. Expect ~10.5 GiB/card (1.0). If init succeeds and a
    4-prompt x n=8 generation completes, measure tok/s and proceed.

(b) AWQ 4-bit, single card, DP-2:  Qwen/Qwen2.5-7B-Instruct-AWQ
    VERIFIED TO EXIST: quant_method="awq", version="gemm", bits=4,
    group_size=128. The "gemm" version is the sm_75-compatible kernel.
    MUST verify in the vLLM startup log that the selected method is
    `awq` and NOT `awq_marlin`. If the log says marlin, abort - it will
    fail on Turing.
    Firing this branch REQUIRES the whole ladder to move to 4-bit (1.5),
    and requires P7 to have been supported by the precision control.

(c) bitsandbytes NF4, single card, DP-2:
    quantization="bitsandbytes", load_format="bitsandbytes".
    Slower than AWQ on Turing (on-the-fly dequant, weaker kernels).
    Same uniform-ladder and P7 requirements as (b).

(d) HF transformers + NF4, single card, DP-2 by process.
    No continuous batching, no paged KV. Expect ~3-5x slower than vLLM.

(e) LAST RESORT: drop the 7B rung entirely and report a
    1.5B / 3B / Llama-3.2-3B ladder, stating the reason in Limitations.
    This costs the paper its size trend, so exhaust (a)-(d) first.
```

**S3 — vLLM unusable, HF fallback for everything.** `transformers` + `accelerate`, one single-GPU process per card, manual batching, DP-2 by process. Expect 3–5× slower than vLLM across the board. If S3 fires, run **Tier A only**, keep items at 200, and drop configurations `c4` and `c5`. Report the reduced design honestly rather than overrunning the quota.

**Explicitly forbidden: `device_map="auto"` across both GPUs for the 7B/8B.** That is naive *pipeline* parallelism — only one GPU computes at a time, there is no continuous batching, and throughput is roughly single-card. At an estimated ~200–330 tok/s it would turn E1c from 4.4 h into **19–31 h**, which neither fits the budget nor a 9 h session. If DP-2 is impossible, use one GPU and accept single-card throughput; do not spread a model across cards with `accelerate`.

#### 4.1.2 Engine settings

```python
LLM(
    model=MODEL_ID,
    dtype="float16",              # T4 (sm_75) has no bf16 tensor cores
    gpu_memory_utilization=0.90,  # 14.4 GiB budget; see 1.0
    max_model_len=2048,           # 4096 for the Tier C long-CoT arm
    tensor_parallel_size=1,       # 2 ONLY for the fp16 7B via S2(a)
    max_num_seqs=48,              # 24 for Llama-3.2-3B (KV-bound; see 1.0)
    swap_space=4,
    seed=SEED,
)
```

- Do **not** force `VLLM_ATTENTION_BACKEND=FLASH_ATTN`; let it fall back to XFORMERS.
- `max_num_seqs` is not a free parameter. It is set by the §1.0 fit table, and Llama-3.2-3B genuinely needs the lower value.

#### 4.1.3 Parallelism policy (normative)

Kaggle bills by **session wall-clock time with an accelerator enabled**, not per device, so the second GPU is free throughput and the default must be to use it.

| Model | Fits one T4? | Policy | Why |
|---|---|---|---|
| Qwen2.5-1.5B | ✅ 6.0 GiB | **DP-2** | Two replicas, no communication |
| Qwen2.5-3B | ✅ 9.7 GiB | **DP-2** | Two replicas, no communication |
| Llama-3.2-3B | ✅ 11.8 GiB @ 24 seqs | **DP-2, `max_num_seqs=24`** | 8 KV heads cap concurrency |
| Qwen2.5-7B | ❌ 20.2 GiB | **TP-2** (fallback: 4-bit DP-2) | Weights alone exceed budget |
| Llama-3.1-8B | ❌ 27.8 GiB | Tier C only | Marginal even under TP-2 |

**DP-2** means one single-GPU vLLM process per card via `CUDA_VISIBLE_DEVICES`, prompt list sharded in half, Parquet outputs concatenated. No cross-GPU traffic, so throughput is close to 2× one card. Never use TP-2 for a model that fits on one card — it adds all-reduce for no memory benefit.

**TP-2** is used only where fp16 weights exceed one card. Kaggle's 2×T4 have no NVLink, so the all-reduce crosses PCIe. At 48 sequences the per-step all-reduce for the 7B is ~19 MB, roughly 2 ms against a ~34 ms step — about 5 % overhead, which is tolerable. TP-2 is therefore **not** 2× one card; it is a way to make the model fit, and §8.0 prices it accordingly.

#### 4.1.4 Quota hygiene (operational, non-negotiable)

Quota is consumed whenever the accelerator is **enabled**, whether or not the GPU is doing work. Therefore:

- All code development, data-loading, prompt-template debugging, answer-extraction testing and analysis/plotting must be done with the **accelerator switched off**. Analysis reads only the persisted Parquet corpus and needs no GPU.
- Turn the accelerator on only to run a generation cell that is already known to work end-to-end on CPU with a stub model.
- Never leave a session idle with the accelerator on. Checkpoint after every cell (§4.4) so a killed session costs at most one cell.
- Log accumulated quota use after every session; the budget in §8 has no room for accidental idle burn.

### 4.2 Sampling parameters

| Parameter | Default | Notes |
|---|---|---|
| `N` (samples per cell) | **24** | Enough for a stable `p̂` (binomial SE ≤ 0.10) and to see the SC plateau; see §8 for the budget arithmetic |
| `temperature` | 0.8 | Except `c5` (1.0) |
| `top_p` | 0.95 | Except `c5` (1.0) |
| `top_k` | −1 (off) | |
| `max_tokens` | 512 (GSM8K, GSM-Symbolic), 1024 (MATH-500), 4096 (long-CoT arm) | |
| `stop` | `["\n\nProblem", "\n\nQuestion"]` | Prevents run-on into a hallucinated next problem |
| `logprobs` | 5 on the final-answer token span | Needed for the self-certainty baseline (B6); costs nothing extra |
| `n` | 24 | Use vLLM's `n` so the prompt is prefilled once and the KV cache is shared across the 24 samples — this is a large throughput win and **must** be used |

### 4.3 Answer extraction and canonicalisation

This step is load-bearing: the nearest prior work (Bay & Yearick 2026) could not report self-consistency on MATH because their numeric extractor could not parse boxed answers, and that is exactly the failure we must not repeat.

- **MATH-500 and GSM-Symbolic:** use the `math-verify` library (HuggingFace) for both extraction and equivalence. Treat two answers as the same candidate iff `math_verify` judges them equivalent; this defines the equivalence classes over which the mode is computed.
- **GSM8K:** extract the final integer after the answer marker; fall back to the last number in the response.
- **Extraction failure** is its own answer class `⊥` and must be counted, never dropped. Report the per-cell extraction-failure rate; if it exceeds 5 % in any cell, flag it in the results table, because an inflated `⊥` class biases the modal ceiling downward.
- Persist, for every sample: `item_id, model, config, seed, sample_idx, raw_text, extracted_answer, canonical_class, is_correct, n_output_tokens, mean_logprob, finish_reason, rendered_prompt, rendered_prompt_token_ids`.
- **Truncation / stop controls:** report per-cell rates of `length` truncation, stop-string termination, and context-limit clipping. Pre-register sensitivity excluding truncated samples. Flag any configuration whose truncation rate exceeds `c0`'s by >2 pp. For separate-axis few-shot cells (`c4`/`c4a`/`c4b`), verify longest prompt + `max_tokens` fits in `max_model_len`; raise to **3072** if not.
- **Dual extractors:** run headline pipeline under `math-verify` and a rule-based numeric/LaTeX extractor; report per-configuration agreement. Stratified manual/LLM audit (~100 samples/config). Sensitivity analyses: (i) `⊥` dropped, (ii) zero-`⊥` items only, (iii) bare-integer gold answers only.

### 4.4 Output artefact (the sampling corpus)

One Parquet file per (model, dataset, config), schema as in §4.3, written to `runs/{model}/{dataset}/{config}.parquet`. Everything downstream reads only these files. **Checkpoint after every cell** — Kaggle sessions die.

---

## 5. Algorithms

### 5.1 Core estimands

**π_mode (self-contained definition).** For item `i`, model `m`, configuration `c`, let `a*` be the plurality answer class over `N` samples. Then

```
π_mode(i,m,c) = 1[ a* is correct ]
π_mode(m,c)   = mean_i π_mode(i,m,c)    # modal-hit rate = SC plateau
```

The **identifiability gap** is `pass@N − π_mode`. Increasing `N` estimates the mode more precisely but cannot change which class is modal. Cite Bay & Yearick for priority on the terminology; do not depend on their paper for the definition.

For variance modelling, let `k` be correct samples out of `N`, `p̂ = k/N`, and `z = log((k+½)/(N−k+½))` (Haldane–Anscombe). **Primary inference uses a crossed binomial GLMM on `k ~ Binomial(N, p)`**; logit-scale moment decomposition is **appendix robustness only** (see §5.2b).

### 5.2 Algorithm 1 — Crossed binomial GLMM (PRIMARY)

```
INPUT : k[i,m,c] correct of N trials per cell (seed 0); seed replicates at c0 (and c1 for O2b)
OUTPUT: variance components, item×config share, latent transfer correlation

MODEL (primary):
  k_imc ~ Binomial(N, p_imc)
  logit(p_imc) = μ + a_i + b_m + g_c + (ab)_im + (ag)_ic + (bg)_mc + ε_imc
  crossed random effects; fit by REML (lme4::glmer) or NUTS (numpyro)

REPORT:
  - variance shares on probability/logit scale (lead with probability scale)
  - latent difficulty correlation: σ²_a / (σ²_a + σ²_ag)  [replaces Spearman disattenuation]
  - item-clustered bootstrap over items, B = 10000

PARAMETRIC-BOOTSTRAP NULL (run on CPU before any GPU generation):
  For each item, take pooled marginal p̄_i; draw k ~ Binomial(N, p̄_i) with NO interaction;
  run full pipeline; record null distribution of item×config share.
  P1 is "supported" only if observed share exceeds null 95th percentile (primary cell).
```

### 5.2b Algorithm 1b — Moment decomposition (APPENDIX robustness)

```
INPUT : z[i,m,c] ; seed replicates at c0
OUTPUT: method-of-moments shares with floor subtraction (legacy spec)

Demoted from headline. Report alongside GLMM; do not use for primary F1/P1.
Known issues: Haldane censoring at ±log(2N+1); transported noise floor; clamping bias.
Also report raw proportion-scale shares.
```

### 5.3 Algorithm 2 — Disattenuated difficulty transfer

The raw correlation between `p̂` in two configurations is attenuated by sampling noise. The seed replicates let us remove that attenuation, which is what makes the comparison honest.

```
INPUT : p̂[i,m,c] ; p̂[i,m,c0,s] for s in S_m
OUTPUT: rho_config (disattenuated), rho_seed (the ceiling)

1. reliability r_mm(c) = mean pairwise Spearman correlation of p̂[.,m,c,s] over seed pairs at config c.
   Primary: r_mm(c0); O2b supplies r_mm(c1) for transportability check.
2. For each config pair (c, c'): rho_raw = Spearman(p̂[.,m,c], p̂[.,m,c']);
   rho_disatt = rho_raw / sqrt(r_mm(c) * r_mm(c'))  [cross-config pairs use both configs' reliabilities].
3. Report rho_disatt within marginal-difficulty quintiles (saturation artefact guard).
4. GLMM latent correlation σ²_a / (σ²_a + σ²_ag) is the preferred headline transfer statistic.
```

### 5.4 Algorithm 3 — Hard-subset stability

```
INPUT : p̂[i,m,c]
PARAM : quantile q = 0.25 (bottom quartile = "hard")
OUTPUT: J_config, J_seed, and the excess-instability statistic

1. H(m,c) = set of items in the bottom q-quantile of p̂[.,m,c]
2. J_config = mean over config pairs of Jaccard( H(m,c), H(m,c') )
3. J_seed   = mean over seed pairs  of Jaccard( H(m,c0,s), H(m,c0,s') )
             # this is the NULL: instability attributable to sampling alone
4. Excess instability = J_seed - J_config.
   Report as: "X% of the hard subset defined under one configuration is
   not hard under another, over and above resampling noise."
5. Significance: **interaction-specific permutation test** — centre each item's p̂ within configuration (or rank within config), then shuffle config labels within item, 10000 permutations, one-sided.
6. Report tie mass at quartile cut per cell; deterministic tie rule; tie-robust Jaccard interval (all tied-at-cut in or out).
```

### 5.5 Algorithm 4 — Modal ceiling and mode reordering

```
INPUT : the per-sample canonical answer classes
OUTPUT: pi_mode(m,c), the reordering rate, and the union ceiling

1. mode(i,m,c) = most frequent canonical class among the N samples
   (ties broken by first occurrence; log the tie rate)
2. pi_mode(m,c) = fraction of items where mode(i,m,c) is correct
   -> this is the self-consistency plateau for that cell
3. REORDER RATE = fraction of items where mode(i,m,c) != mode(i,m,c')
   for a config pair; split into
     - benign      : both wrong or both right
     - corrective  : wrong -> right
     - destructive : right -> wrong
4. UNION CEILING  = fraction of items where SOME config's mode is correct
   (an oracle upper bound for any configuration-selection policy)
5. Report pi_mode(m,c) alongside coverage pass@N(m,c) so the
   identifiability gap is visible per cell.
```

### 5.6 Algorithm 5 — Configuration-Diversified Voting (CDV), the intervention

**Costs zero additional GPU time**: it re-partitions samples that already exist.

```
INPUT : sampling corpus; total sample budget B_s (matched across methods)
PARAM : C_use = 3 (default: c0, c1, c2 — all core configs)
        allocation = "uniform"   (default)
OUTPUT: predicted answer per item

CDV(i, B_s, C_use):
    per_cfg = floor(B_s / C_use)
    pool = []
    for c in first C_use configs (fixed order, no per-item selection):
        pool += sample WITHOUT replacement per_cfg chains from cell (i,m,c)
    return most frequent canonical class in pool

ADAPTIVE-CDV(i, B_s, C_use, n0=4):
    # two-stage: detect cross-config disagreement cheaply, then commit
    probe = for each c in C_use: n0 samples -> modal class m_c
    if all m_c agree:
        return that class                       # spend nothing further
    else:
        spend the remaining budget uniformly across the disagreeing configs
        return most frequent canonical class in the pooled set
```

**Matched-budget rule (non-negotiable).** Every method comparison is at equal **total generated output tokens**, not equal sample count, because configurations differ in mean chain length. Compute the token cost from the logged `n_output_tokens` and report accuracy-vs-tokens curves, not accuracy-vs-N curves. Report accuracy-vs-N as a secondary panel only.

**Falsification hook (F3 — PRIMARY).** Pre-register two tests:
1. **Primary:** plug-in π_mode — compare π_mode(mixture from pooled 3×N samples) vs. π_mode(c0 from N samples) at matched token budget.
2. **Secondary:** maj@n and pass@n vs. tokens curves; label small-budget overlap (≤24 tokens) explicitly as underpowered for plateau comparison.

**CDV mechanism identification (zero GPU):**
- **B12:** oracle **best single configuration** globally (highest π_mode); CDV must beat mean config, comparison to B12 is decisive vs. H_ensemble.
- **Gain decomposition:** split CDV-won items into (i) no single config's mode correct [supports H_reorder] vs. (ii) some config's mode correct [selection only].
- **Seed-pooled c0 control:** pool c0 seed replicates at matched tokens; must equal SC (validates machinery).

---

## 6. Baselines

All are computed offline from the corpus except where noted.

| ID | Baseline | Notes |
|---|---|---|
| B1 | avg@1 (mean single-sample accuracy) | The honest single-sample number; preferred over greedy for variance reasons |
| B2 | Greedy decoding (`T=0`) | One extra cheap pass per (model, dataset, config); include for comparability with the literature |
| B3 | Self-consistency, single config, `N ∈ {1,2,4,8,16,24}` | The primary baseline |
| B4 | Temperature-diversified SC (same prompt, `T ∈ {0.6,0.8,1.0}`) | Isolates "diversity per se" from "configuration diversity" — an essential ablation |
| B5 | Self-Para-Consistency (vote over `c6` paraphrases) | The nearest published intervention |
| B6 | Self-certainty-weighted vote | Weight each chain by KL(answer-token distribution ‖ uniform); free from logged logprobs |
| B7 | CISC-style verbalised-confidence weighted vote | Requires appending a confidence request; adds ~10 tokens/sample. Include only if budget permits |
| B8 | Adaptive-Consistency / ESC early stopping | The efficiency baseline; evaluate on the accuracy-vs-tokens curve |
| B9 | **Transferred-difficulty allocation** | Estimate per-item difficulty in `c0`, then allocate a non-uniform budget in `c1`. This is the baseline whose degradation is our applied claim |
| B10 | Oracle pass@N (coverage) | Upper bound |
| B11 | Oracle per-item configuration selection | Upper bound for CDV |
| B12 | Oracle **best single configuration** (global max π_mode) | CDV vs. B12 discriminates ensemble from "pick a good prompt" |

---

## 7. Metrics and statistics

### 7.1 Metrics

| Metric | Definition |
|---|---|
| `avg@1` | mean over items of `p̂` |
| `pass@N` | unbiased Chen et al. estimator, `1 − C(N−k, n)/C(N, n)`, evaluated at `n ≤ N` |
| `maj@n` | accuracy of the plurality answer over `n` samples, averaged over 200 random subsets of size `n` drawn without replacement |
| `π_mode` | fraction of items whose modal class (over all `N`) is correct = the SC plateau |
| identifiability gap | `pass@N − π_mode` |
| `ρ_disatt` | disattenuated Spearman difficulty transfer (§5.3) |
| `J_config`, `J_seed` | hard-subset Jaccard (§5.4) |
| variance shares | §5.2 |
| reorder rate (corrective / destructive) | §5.5 |
| tokens | total output tokens, the x-axis for all budget-matched comparisons |

### 7.2 Significance testing — required, not optional

- **Primary uncertainty:** nonparametric bootstrap resampling **items** (the clustering unit), `B = 10000`, BCa intervals. For GSM-Symbolic, resample **templates**, not instances, because instances from one template are not independent.
- **Paired comparisons between methods on the same items:** paired bootstrap on the per-item difference plus **McNemar's exact test** on the discordant pairs. Report both the effect size with CI and the p-value; lead with the effect size.
- **Seeds:** independent sampling seeds for the `c0` cells power the noise correction — 3 per Qwen model in Tier A, 5 for the 1.5 B and 3 B in Tier B (§3). Headline accuracies in the main table are reported as mean ± bootstrap CI, with the across-seed standard deviation given in a footnote.
- **Multiplicity:** declare families explicitly:
  - **Primary predictions P4, P5:** one cell each (3B × MATH-500); no disjunction.
  - **P1, P2, P3:** primary cell 3B × MATH-500; secondary cells reported with stated aggregation (median + range).
  - **T5 baselines:** Holm–Bonferroni within table.
  - **ρ_disatt pairs, J_config tests:** Holm within each (model, dataset) family; report family size.
- **Never** report a bare accuracy difference without an interval. The 2025 reproducibility literature (seed variance on AIME-scale evaluations) makes this the fastest way to lose a reviewer.

---

## 8. Experiment matrix and GPU budget

*(Adversarial review §6.3 experiment-matrix changes are normative here.)*

### 8.0 How the budget is counted

Kaggle bills **session wall-clock hours with an accelerator enabled**, capped at 12 h per session (plan against 9 h), with a ~30 h weekly allowance that **resets weekly**. Every number in this section is therefore a session hour on a 2×T4 instance, and it is the number that is charged. Parallelism policy is in §4.1.3.

#### Throughput model — all figures are ESTIMATES with stated bands

These are **derived, not measured.** Decode on a T4 is bound by weight-memory bandwidth, so:

```
decode steps/s  ~=  0.65 x 320 GB/s / (bytes of weights read per pass)
tokens/s        ~=  steps/s x concurrent sequences        (capped by compute)
```

The 0.65 is an assumed achieved fraction of peak bandwidth for vLLM on Turing. **Every number below inherits that assumption and must be replaced by a measurement from the §4.1.1 smoke test before the full run is committed.**

| Rung | Mode | Weights read/pass | Concurrency | Planning tok/s | Plausible band |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | fp16 DP-2 | 3.08 GB/card | 48 | **5 000** | 3 500 – 6 500 |
| Qwen2.5-3B | fp16 DP-2 | 6.18 GB/card | 48 | **2 800** | 2 000 – 3 500 |
| Llama-3.2-3B | fp16 DP-2 | 6.42 GB/card | **24** | **1 600** | 1 200 – 2 200 |
| Qwen2.5-7B | fp16 TP-2 | 7.09 GB/card | 48 | **1 400** | 900 – 1 700 |
| *Qwen2.5-7B* | *AWQ 4-bit DP-2* | *5.6 GB/card* | *48* | *2 400* | *2 100 – 2 900* |
| *Qwen2.5-1.5B* | *AWQ 4-bit DP-2* | *1.4 GB/card* | *48* | *4 500* | *3 000 – 6 000* |
| *Qwen2.5-3B* | *AWQ 4-bit DP-2* | *2.3 GB/card* | *48* | *3 100* | *2 200 – 4 000* |

Italic rows are the 4-bit fallback branch (§1.5), priced but not planned.

**Two corrections to the previous version of this table.**

*Llama-3.2-3B was overpriced by ~1.75×.* The old table gave it the same throughput as Qwen2.5-3B on the grounds of similar parameter count. But its 8 KV heads make its KV cache 4× fatter per token (§1.0), which caps concurrency at 24 instead of 48 and roughly halves throughput. Corrected from ~2 800 to ~1 600 tok/s. This adds ~1.1 h to E3a.

*4-bit is not automatically faster per token.* At these sizes AWQ dequantises to fp16 and runs a normal GEMM, so at large batch it approaches fp16 compute cost while reading fewer weight bytes. The gain is real but modest, and at 1.5B it is roughly a wash because those cells are scheduler-overhead-bound rather than bandwidth-bound. **The reason 4-bit helps at the 7B is not the arithmetic — it is that it converts a TP-2 cell into a DP-2 cell.** Note the consequence: the 7B's 4-bit fallback (~2 400 tok/s) is *faster* than its fp16 primary (~1 400 tok/s). We are choosing the slower option deliberately, on the grounds in §1.5.

Mean output length assumed: GSM8K 220 tok, MATH-500 550 tok, GSM-Symbolic 250 tok, long-CoT 2 000 tok.

### 8.1 Tier A — the paper (Week 1)

Tier A alone produces **every figure and table in §9**. It is the must-run tier; nothing here is negotiable.

| # | Cell | Grid | Est. tokens | Est. session hours |
|---|---|---|---|---|
| S | **Backend smoke test and throughput pilot** (§4.1.1) | — | — | **0.5 h** |
| E1a | Main grid, 1.5 B | 2 datasets × **3 core cfg** × 200 items × 24 | 11.1 M | **0.6 h** |
| E1b | Main grid, 3 B | same | 11.1 M | **1.1 h** |
| E1c | Main grid, 7 B | same | 11.1 M | **2.2 h** |
| E2 | Seed-replicate null (`c0`, +2 seeds → 3 total), all 3 Qwen | 2 datasets × 200 × 24 × 2 | 14.8 M | **1.7 h** |
| E3a | Cross-family, Llama-3.2-3B | 2 datasets × 3 cfg × 200 × 24 | 11.1 M | **1.9 h** |
| E4 | Distribution shift, GSM-Symbolic (`main`,`p2`), 3 Qwen | 2 variants × 3 cfg × 150 × 24 | 16.2 M | **1.9 h** |
| E5 | Greedy passes (B2), all models/datasets/configs | 1 sample per cell, `T=0` | 1.8 M | **0.3 h** |
| **PC** | **Precision control** (§1.5): 1.5 B + 3 B at AWQ 4-bit | MATH-500, **3 core cfg**, 200 items × 24 | 15.8 M | **1.2 h** |
| | | | **TIER A TOTAL** | **≈ 12.5 h** |

(Generation cells alone are 12.0 h; smoke test adds 0.5 h.)

**Execution order:** S → E1a → E1b → PC → decide → E1c → E2 → E3a → E4 → E5.

Changes from pre-review: core factor 6 → **3** configs (c0–c2 only; −c3, −c4/c4a, −c5); GSM8K demoted in analysis; **≈ −6.3 h** vs.\ prior Tier A (18.8 h).

Largest single cell is E1c at 2.2 h. **Week 1 consumes ≈ 15 h** of a 30 h allowance (incl. ~2.5 h slack).

### 8.2 Tier B — statistical strengthening (Week 2)

Everything in Tier B buys **precision on the existing claims**, not new claims. Rationale for the allocation is in §8.4.

| # | Cell | What it buys | Est. session hours |
|---|---|---|---|
| E1a′/b′/c′ | Main grid on **second 200 items**, all 3 Qwen | **3.9 h** |
| E2′ | Seed-replicate null on new 200 items | **1.7 h** |
| E2″ | Seeds 3 → 5 at `c0` for 1.5 B and 3 B, 400 items | **1.5 h** |
| E3a′ | Cross-family Llama-3.2-3B, new 200 items | **1.9 h** |
| E4′ | GSM-Symbolic 150 → 300 per variant | **1.9 h** |
| O3→core | Paraphrase arm `c6`, Qwen ladder, 400 items | **2.0 h** |
| **O2** | **Deep sampling:** 3 B, MATH-500, **c0–c3**, 200 items, **N: 24 → 64** | **1.75 h** |
| **O2b** | **Seed replicates at c1:** 3 B, MATH-500, 200 items, +2 seeds, N=24 | **0.5 h** |
| E5′ | Greedy top-up on new items | **0.3 h** |
| | **TIER B TOTAL** | **≈ 16.1 h** |

**Week 2 consumes ≈ 18 h** including slack. Paper-critical generation across two weeks is **≈ 28.6 h**, or **≈ 32 h** with pilots and restarts (down from ≈ 46 h pre-review; net **≈ −0.2 h** after O2/O2b vs.\ c5 drop).

**O2 rationale:** F3/P5 require SC to reach its plateau and CDV to pool deep-N across configurations. Prior O2 (7B, c0-only) could not support CDV comparison. Redirected arm: 3B × MATH-500 × {c0,c1,c2,c3} × (64−24) extra samples × 200 items.

### 8.6 Experiment matrix amendments (adversarial review §6.3)

Normative changelog applied in this pass. **Do not run the pre-review 6-config grid.**

| Change | Rationale | GPU effect |
|---|---|---|
| **Drop `c5` from core** | Circular for mode-reordering claims; already in B4 | **≈ −1.7 h** |
| **Core = `c0`, `c1`, `c2`** | Semantics-preserving axes only; CDV `C_use=3` | Main grid 6 → 3 configs |
| **Demote `c3`, `c4`/`c4a`/`c4b`** | Parse confound (`c3`); task-information confound (`c4`) | 0 h in Tier A (not generated) |
| **Redirect O2** | 3B, MATH-500, `c0`–`c3`, `N: 24→64` (not 7B `c0`-only) | ≈ 0 h net (was 1.75 h) |
| **Add O2b** | +2 seeds at **`c1`**, 3B, MATH-500 (reliability transportability) | **≈ +0.5 h** |
| **Demote GSM8K** | Saturation + item-specific contamination | 0 h (analysis only) |
| **GLMM primary** | Moment decomposition → appendix | 0 h (CPU) |
| **Pre-register plug-in π_mode** | Primary F3/P5 test (curves secondary) | 0 h |

**Net vs.\ pre-review matrix:** dropping `c5` and demoting `c3`/`c4` from Tier A saves **≈ 6.3 h** on Tier A alone; O2 redirect + O2b add **≈ +0.2 h** net on Tier B. Paper-critical total **≈ 28.6 h** (Tier A **12.5 h** + Tier B **16.1 h**), or **≈ 32 h** with pilots/slack (down from **≈ 46 h**).

**O2 token arithmetic (recomputed 29 Jul 2026):** $4 \times 200 \times 40 \times 550 = 17.6$ M tokens at 2 800 tok/s → **1.75 h**. **O2b:** $2 \times 200 \times 24 \times 550 = 5.28$ M tokens → **0.52 h** at 2 800 tok/s (budgeted **0.4–0.5 h**).

### 8.3 Tier C — genuinely optional (Week 3, only if Tiers A and B are clean)

| # | Cell | Est. session hours |
|---|---|---|
| O1 | Long-CoT arm: R1-Distill-Qwen-1.5B, MATH-500, 4 cfg, 150 items, N=16, 4096 tok | **1.1 h** |
| E3b | Cross-family large, Llama-3.1-8B fp16 TP-2 — **only if S2(a) succeeded** | **3.2 h** |
| O4 | 14 B 4-bit spot check (`Qwen/Qwen2.5-14B-Instruct` AWQ ⚠️) on MATH-500, 4 cfg, 150 items, N=16 | **2.5 h** |
| | **TIER C TOTAL** | **≈ 6.8 h** |

O1 halved because the Qwen3-4B hybrid-thinking arm is dropped under the pinned stack (§1.3). Tier C adds **model diversity**; it is last on purpose (§8.4).

### 8.3.1 The 4-bit fallback branch, pre-costed

If S2(a) fails and S2(b) fires, the **entire ladder** moves to 4-bit and Tier A is re-run. Cost:

| Cell | fp16 (planned) | 4-bit (fallback) |
|---|---|---|
| E1a 1.5 B | 1.3 h | 1.4 h |
| E1b 3 B | 2.2 h | 2.0 h |
| E1c 7 B | 4.4 h | **2.6 h** |
| E2 seeds | 2.6 h | 2.0 h |
| E3a Llama-3.2-3B ⚠️ | 2.6 h | 2.4 h |
| E4 GSM-Symbolic | 2.5 h | 1.9 h |
| E5 greedy | 0.3 h | 0.3 h |
| PC | 2.4 h | — (already run; it *is* the fp16 comparison) |
| **Total** | **18.3 h** | **12.6 h** |

Two things follow. First, **the fallback is cheaper than the primary**, so no budget pressure pushes us toward it — the fp16 preference in §1.5 costs us nothing we need. Second, the worst realistic case is discovering the failure after part of Tier A is already spent: fp16 E1a + E1b + E2 + PC (8.5 h) followed by a full 4-bit Tier A (12.6 h) is 21.1 h, which still fits Week 1 plus a small bite of Week 2. The branch is affordable even when taken late.

⚠️ One gap the code agent must close during the smoke test: there is no official Qwen-published AWQ build of `meta-llama/Llama-3.2-3B-Instruct`. If the 4-bit branch fires, that rung needs either a community AWQ repo (verify it exists and loads) or bitsandbytes NF4. Its 2.4 h estimate assumes the latter and is the least certain figure in this table.

### 8.4 Why the headroom goes to items and seeds, not models

The headline estimand is a **difference**: a raw mean square minus an estimated binomial noise floor. Its precision is governed by three things, and they are not equally cheap to improve.

**1. Items are the best buy, by a wide margin.** Items are the bootstrap resampling unit (§7.2), so every confidence interval in F1, T3 and T4 scales as $1/\sqrt{I}$. Going 200 → 400 narrows them all by 29 % for a proportional cost. Nothing else in the budget has that property: the model factor has 5 levels and the **core** configuration factor has 3, so neither is the binding constraint on the item × configuration component, which is estimated across $I \times C$ cells and bootstrapped over $I$.

**2. Raising N is necessary for plateau claims (Algorithm 5), not for Algorithm 1 alone.** O2 exists specifically so F3 can compare asymptotic π_mode objects. For saturated GSM8K cells, raising N remains a trap for variance decomposition; MATH-500 is the headline benchmark.

**3. Seeds are the second-best buy, because of where they sit in the formula.** The seed replicates do double duty: they estimate the noise floor $\sigma^2_{\text{samp}}$ (which is also available analytically, so the replicates mainly *validate* it) and they estimate the reliability $r_{mm}$, which is the **denominator** of the headline $\rho_{\text{disatt}} = \rho_{\text{raw}} / r_{mm}$. A noisy denominator is a direct threat to the Q2 claim and the first thing a reviewer will probe. Three seeds give only 3 independent pairs; five give 10. Tier B buys this on the 1.5 B and 3 B, where it is cheap, and leaves the 7 B at 3 seeds, where it is not.

**4. More configurations is third.** It would raise the degrees of freedom of the item × configuration term and would somewhat blunt the "configuration space is a sample, not a census" limitation. But the bootstrap CI is driven by items, not configurations, so it buys less precision per hour than item expansion. Revisit only if Tier B finishes under budget.

**5. More models is last, and this is a deliberate disagreement with the intuition that a bigger ladder is more convincing.** A fifth model adds one level to a factor that already has four, does not tighten any item-level interval, and contributes nothing at all to the item × configuration component that carries the claim. The reviewer's "does this scale?" question is answered by a *trend* across 1.5 B → 3 B → 7 B plus a cross-family replication, which Tier A already delivers; a fourth Qwen would not change the shape of that trend. Tier C exists so that the long-CoT, 8 B and 14 B arms are available as a robustness note, not because they are load-bearing.

This is also why demoting Llama-3.1-8B out of Tier A (§1.2) is cheap: it was the *second* level of the cross-family factor, and the first level already carries the "not Qwen-specific" argument. What it cost us — a cross-family size trend — is exactly the nice-to-have this paragraph is about.

**Summary of the recommendation.** Spend the headroom on **items first (200 → 400), seeds second (3 → 5 where cheap)**, and treat additional models as a Tier-C nicety. This matches the reasoning that the whole claim rests on separating real interaction variance from sampling noise — with the one correction that increasing $N$, which looks like the direct way to reduce sampling noise, is the least effective option available.

### 8.5 If the plan must be cut

The budget does not need cutting under the current quota rules, and the sequencing above already degrades gracefully: **drop whole tiers from the bottom.** Tier A alone is a complete paper.

**The realistic reason to cut is not the quota — it is the backend.** If the §4.1.1 cascade lands on S3 (vLLM unusable, HF `generate` for everything), throughput falls 3–5× and Tier A's 18.3 h becomes 55–90 h. That does not fit anything. The prescribed response is in S3: run **Tier A only**, keep items at 200, drop configurations `c4` and `c5` (6 → 4), and report the reduced design. Cut order thereafter: E4 → E3a → PC. Do **not** cut E2, the seed-replicate null — without it the noise correction, and therefore the central claim, is unsupportable. Do **not** reduce `N` below 24; §8.4 point 2 explains why that saves less than it appears to while costing the mode-reordering analysis its resolution.

> *Footnote, for the case where Kaggle changes policy.* Were the quota ever charged per **device**-hour rather than per session hour, Tier A would cost ~35 device-hours and would no longer fit a single week. In that event, run Tier A across two weekly windows rather than shrinking it.

---

## 9. Figures and tables the paper needs

The code agent should emit exactly these artefacts, as both a vector PDF (for LaTeX) and a CSV/JSON of the underlying numbers.

### Figures (presentation order matches reframed paper)

| Name | File | Content | Priority |
|---|---|---|---|
| **F4** | `fig4_ceiling_vs_coverage.pdf` | Per model size: `maj@n`, `pass@n` vs. tokens; **π_mode horizontal per configuration** | **Primary** |
| **F5** | `fig5_cdv_vs_sc.pdf` | CDV vs. SC vs. B4 vs. B12; plug-in π_mode comparison annotated | **Primary** |
| **F6** | `fig6_reordering.pdf` | Mode transitions: benign / corrective / destructive | **Primary** |
| **F1** | `fig1_variance_components.pdf` | GLMM variance shares (moment decomposition in appendix) | Supporting |
| **F2** | `fig2_transfer_scatter.pdf` | Transfer scatter + seed ceiling panel | Supporting |
| **F3** | `fig3_hard_subset_overlap.pdf` | J_config vs. J_seed with interaction permutation p-values | Supporting |
| **F7** | `fig7_shift.pdf` | MATH-500 vs. GSM-Symbolic (GSM8K secondary panel optional) | Secondary |
| **F8** | `fig8_precision_control.pdf` | fp16 vs. 4-bit robustness | Robustness |

### Tables

| Name | File | Content |
|---|---|---|
| **T1** | `tab1_setup.csv` | Models, datasets, item counts, configs, `N`, seeds, total tokens, measured wall-clock. Reproducibility table. |
| **T2** | `tab2_main_accuracy.csv` | Per (model, dataset, config): avg@1, greedy, maj@24, `π_mode`, pass@24, extraction-failure rate. All with bootstrap CIs. |
| **T3** | `tab3_variance_components.csv` | Numeric backing for F1, with CIs and the clamping report. |
| **T4** | `tab4_transfer.csv` | `ρ_raw`, `ρ_disatt`, reliability `r_mm`, for config-pairs / model-pairs / family-pairs. |
| **T5** | `tab5_method_comparison.csv` | B1–B12 and CDV at 3 token budgets; CDV gain decomposition columns; Holm within table |
| **T6** | `tab6_downstream.csv` | Transferred-difficulty allocation (B9) vs. configuration-aware allocation, at matched tokens — the applied consequence. |
| **T7** | `tab7_precision_control.csv` | Numeric backing for F8: variance shares, `ρ_disatt`, `r_mm`, `J_config`, `π_mode` and reorder rate at fp16 and 4-bit, with paired-bootstrap differences and the P7 verdict. |

T1 must additionally record **which backend branch fired** in the §4.1.1 cascade — the vLLM version, whether the pin was needed, the attention backend, the quantisation method string reported by the engine, and the parallelism mode per model. If the 4-bit branch fired, T1 must say so and the Limitations section must cite F8.

### Reporting requirements

Every figure and table must be regenerable by a single script from the Parquet corpus, with no manual steps, and must record the git commit and the run manifest hash in its metadata. `\todo` markers in the paper are keyed to these exact filenames.

---

## 10. Pre-registered predictions

**Primary cell for all headline tests:** Qwen2.5-3B × MATH-500 unless noted. See `docs/REFRAMING.md` for falsification logic.

| ID | Tier | Prediction |
|---|---|---|
| **P4** | **Primary** | `π_mode` varies by **> 3 accuracy points** across core configurations for a fixed (model, dataset). |
| **P5** | **Primary** | **Plug-in π_mode(CDV) > π_mode(c0)** at matched token budget; temperature-diversified SC (B4) does **not** exceed π_mode(c0). Maj@n curves secondary. |
| P1 | Supporting | GLMM item×configuration share exceeds **parametric-bootstrap null 95th percentile**; exceeds item×model share on MATH-500. Secondary cells: median across ladder. **5–10% zone:** report explicitly, do not treat as null. |
| P2 | Supporting | `ρ_disatt` (or GLMM latent correlation) for config-pairs **< 0.85** on primary cell; report `r_mm` by difficulty quintile (not as standalone validity gate). |
| P3 | Supporting | `J_config` **≤ J_seed − 0.10** (interaction permutation p < 0.05 after Holm within family). |
| P6 | Applied | Transferred-difficulty allocation (B9) loses to configuration-aware allocation at matched tokens. Offline Adaptive-Consistency / ESC mis-stopping demo required (zero GPU). |
| P7 | Gate | Item×configuration share differs **< 3 pp** fp16 vs. 4-bit; `ρ_disatt` differs **< 0.05** on 1.5B/3B MATH-500. |

### Falsification conditions (individually decisive)

| ID | Sub-claim | Condition |
|---|---|---|
| **F3** | CDV / mechanism | Plug-in π_mode(CDV) ≤ π_mode(c0); B12 beats CDV; gain decomposition category (i) empty |
| **F4** | π_mode stability | max π_mode − min π_mode ≤ 3 pp |
| **F1** | Variance structure | GLMM item×config ≤ 5% **and** below bootstrap null |
| **F2** | Hard subset | J_config ≥ J_seed − 0.02 (relative; **not** J ≥ 0.90 absolute) |

If **F3/F4** fail: negative result — π_mode is a model property; modal-ceiling framing licensed. If **F1/F2** fail but F3/F4 hold: mechanism paper without strong transfer claim. Report all paths in advance.
