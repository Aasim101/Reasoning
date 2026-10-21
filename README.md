# A resumable harness for LLM reasoning experiments on a free-tier GPU budget

Method-agnostic experimental infrastructure for a paper on LLM reasoning. It runs
reasoning strategies over benchmarks, survives abrupt session kills, and turns
raw generations into publication figures and LaTeX tables.

It deliberately contains **no novel method**. A new reasoning strategy is one
~50-line file (see [Adding a reasoning method](#adding-a-reasoning-method)).
Everything else — resumability, batching, time budgeting, grading, statistics,
figures — already works and does not need to know what your method does.

Designed against **Kaggle Notebooks, free tier** (verified 2026-07-29):

| Resource | Value |
|---|---|
| Accelerator | 2x NVIDIA T4 16 GB (Turing, sm75) **or** 1x P100 16 GB (sm60) |
| bf16 | **Not supported on either.** Default dtype is fp16; a bf16 request is downgraded with a warning. |
| Session limit | 12 h hard kill (GPU/CPU). Default run budget: **8.0 h**. |
| Weekly quota | ~30 GPU-hours, resets weekly |
| RAM / CPU | 29 GB / 4 cores |
| Disk | **20 GB** auto-saved `/kaggle/working`; extra ephemeral scratch outside it |
| Internet | may be on or off — both are supported |

---

## Contents

- [Install](#install)
- [Smoke test first](#smoke-test-first)
- [Run one experiment cell](#run-one-experiment-cell)
- [Resuming across sessions](#resuming-across-sessions)
- [Grading is a separate pass](#grading-is-a-separate-pass)
- [Figures and tables](#figures-and-tables)
- [Adding a reasoning method](#adding-a-reasoning-method)
- [Datasets](#datasets)
- [Model backends](#model-backends)
- [Offline / no-internet sessions](#offline--no-internet-sessions)
- [GPU-hour budget plan](#gpu-hour-budget-plan)
- [Tests](#tests)
- [Layout](#layout)
- [Design decisions worth knowing](#design-decisions-worth-knowing)

---

## Install

**Local development (CPU, no model):** everything except real inference works.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux
pip install pyyaml numpy matplotlib sympy pytest
```

**Kaggle:** open `notebooks/kaggle_runner.ipynb` and run it top to bottom. It
installs only what Kaggle is missing. Do **not** `pip install -U torch numpy
transformers` there — Kaggle's torch is compiled against its numpy, and
replacing either produces `numpy.dtype size changed, may indicate binary
incompatibility` and costs you a session.

**Full install:** `pip install -r requirements.txt`. The file documents which
packages Kaggle preinstalls versus which you must add (`bitsandbytes`,
`math-verify`, optionally `vllm`).

---

## Smoke test first

Run this before anything that costs quota. It exercises the entire pipeline —
dataset, prompting, strategy, JSONL, resume bookkeeping, grading, metrics — with
`MockBackend`, so no model, no network, no GPU:

```bash
python -m src.runner --config configs/smoke_mock.yaml
```

Then optionally validate the real `transformers` path with a 135M model:

```bash
python -m src.runner --config configs/smoke_tiny_model.yaml
```

`python -m src.runner --list` prints the registered datasets and strategies.

**On Kaggle, run CPU validation before enabling the GPU** (uses zero quota):

```bash
python scripts/validate_cpu.py
```

This loads every benchmark in the matrix, checks splits/columns, and grades a few
gold answers. Fix any `FAIL` rows before spending GPU hours.

---

## Run one experiment cell

One YAML config in `configs/` = one experiment cell = one row of the results
table. Configs inherit from `configs/base.yaml` via `defaults:`, so a cell only
states what it changes.

```bash
python -m src.runner --config configs/gsm8k_cot_zeroshot.yaml \
    --model Qwen/Qwen2.5-7B-Instruct \
    --subsample 200 --seed 0 \
    --time-budget-hours 8.0 \
    --out-dir results
```

Any field is overridable from the CLI with dotted paths:

```bash
--set strategy.params.k=16 --set model.load_in_4bit=true --set data.subset=main
```

Useful flags: `--dry-run` (load the data, print the plan and what remains, spend
nothing), `--print-config` (resolve and dump the config plus its hash),
`--max-examples N` (cap this session), `--max-new-tokens N`, `--no-grade`.

Output lands in `results/<run_name>__<config_hash>/`:

```
results.jsonl    append-only raw generations, one line per example (never rewritten)
graded.jsonl     cached grading pass over results.jsonl
manifest.json    config, progress, per-session history, elapsed time, GPU-hours,
                 and the exact resume command
```

---

## Resuming across sessions

**The start command and the resume command are the same command.** Run it again
and it continues.

How it works:

1. Every example's result is keyed by a deterministic
   `uid = hash(model, strategy, dataset, example index, seed, config_hash)`.
2. Results are appended to a line-buffered `results.jsonl` and `fsync`'d every
   `runtime.flush_every` examples (default 10). Line buffering means a killed
   *process* loses nothing; the fsync bounds what a killed *machine* can lose to
   at most 9 examples.
3. On start, the runner reads `results.jsonl`, collects finished uids, and skips
   them. A truncated final line (the machine died mid-write) is detected and
   discarded, and that example is simply redone.
4. `manifest.json` records progress, cumulative wall-clock and GPU-hours,
   per-session history, and the resume command.

Examples recorded with an `error` are **not** treated as complete, so a
transient failure is retried next session. Pass `include_errors=True` to
`completed_uids` (or just let them retry) if some example crashes reliably.

### The wall-clock guard

`runtime.time_budget_hours` (default **8.0**) is a soft budget below the 12 h
kill. Before each example the runner asks: is the remaining budget more than
`reserve_minutes` plus 1.5x the recent per-example time? A recent-window mean is
used rather than a whole-run mean because per-example cost drifts (longer
questions later in a split, growing KV cache, T4 thermal throttling). When the
answer is no it stops **at an example boundary**, writes the manifest, and
prints:

```
==============================================================================
RUN INCOMPLETE  (time_budget)
==============================================================================
  progress           : 137/200 examples (63 remaining)
  this session       : 137 done, 0 errors, 7h51m12s wall-clock
  throughput         : 3.42 s/example, eta for remainder 3m35s
  sessions needed    : ~0.1 more at this budget

  RESUME IN THE NEXT SESSION WITH EXACTLY THIS COMMAND:
    python -m src.runner --config configs/gsm8k_cot_zeroshot.yaml --set ...
```

`SIGINT`/`SIGTERM` are also caught and turned into a clean stop at the next
example boundary, so manually stopping a notebook still leaves a consistent run.

### Carrying a run between Kaggle sessions

1. Zip the run directory (cell 11 of the notebook) — raw JSONL is usually a few MB.
2. Download `results_bundle.zip` from the notebook output.
3. Next session: upload it as a Kaggle Dataset (or attach this notebook's output
   as an input), unzip into `/kaggle/working/results`.
4. Re-run the same command.

Because the directory name embeds the config hash, a resumed run can only append
to the matching experiment. Changing a **semantic** setting (model, dtype,
quantization, sampling params, subsample, strategy params) creates a *new*
directory rather than corrupting the old one. Changing a **performance** setting
(batch size, `device_map`, `tensor_parallel_size`, backend, time budget) keeps
the same directory, so you can move a run from 2x T4 to a P100, or from vLLM to
transformers, without losing work. Such drift is logged and recorded in the
manifest.

---

## Grading is a separate pass

Fixing the answer checker must never cost GPU time. It doesn't:

```bash
# Grade or re-grade a single run, or a whole results tree
python -m src.grading --run-dir results/gsm8k__Qwen2.5-7B-Instruct__cot_zeroshot__ab12cd34ef56
python -m src.grading --results-root results --force
```

`results.jsonl` is append-only and never rewritten. Grading reads it, extracts
answers from **every** reasoning trace (so `pass@k` is computable offline for any
strategy), checks equivalence, and writes `graded.jsonl`. Records already graded
with the current `GRADER_VERSION` are skipped; bumping that constant
automatically triggers a re-grade.

Answer extraction handles `\boxed{}` (including nested braces), `\fbox{}`,
"the answer is X", GSM8K's `#### X`, multiple-choice letters in several shapes,
and a last-number fallback. Equivalence is layered: normalised string match,
then numeric match with tolerance, then symbolic comparison via `math_verify` or
`sympy`, then set/tuple comparison. The active backend is reported by
`grader_backend_name()` and stored in every graded record, so the paper can state
exactly how answers were judged.

---

## Figures and tables

```bash
python -m src.analysis.aggregate --results-dir results --out-dir paper_assets \
    --figures --tables --baseline cot_zeroshot
```

Produces `summary.json`, `summary.csv`, `comparisons.json`, PDF figures
(accuracy with CIs, accuracy-vs-compute, majority-vote-vs-k, pass@k, tokens per
correct answer) and `booktabs` LaTeX tables ready to `\input{}`. Figures are
matplotlib only (no seaborn), with embedded editable fonts (`pdf.fonttype=42`),
colourblind-safe colours, and distinct markers so they survive greyscale
printing. Plotting is never done inside the run loop.

Statistics are first-class: bootstrap and Wilson confidence intervals, and
paired significance tests (McNemar and a paired bootstrap) computed on
identical examples with explicit pairing and recorded seeds. `pass@k` uses the
unbiased estimator, tested against brute-force enumeration.

---

## Adding a reasoning method

Copy `examples/method_template.py`, edit, done. The whole interface is one
method:

```python
from src.strategies.base import ReasoningStrategy, register_strategy
from src.types import StrategyResult


@register_strategy("my_method")
class MyMethod(ReasoningStrategy):
    def __init__(self, k: int = 4, **kw):
        super().__init__(**kw)
        self.k = k

    def run(self, example, backend, params) -> StrategyResult:
        prompt = self.user_prompt(example, instruction="Think step by step.")
        groups = backend.generate([prompt], self.gen(params, n=self.k, temperature=0.7))
        traces = [c.text for c in groups[0]]
        answer, info = self.vote(traces, example)
        tp, tc = self.tally(groups)
        return StrategyResult(
            final_answer=answer, reasoning_traces=traces, n_samples=self.k,
            tokens_prompt=tp, tokens_completion=tc, n_calls=1, extra=info,
        )
```

Select it from YAML by registered name, or by import path with no edits to the
package at all:

```yaml
strategy:
  name: "my_pkg.my_module:MyMethod"
  params: {k: 8}
```

Helpers available on the base class: `gen()` (override sampling params),
`user_prompt()` (consistent prompt + answer-format instruction),
`extract()`/`vote()` (grader-consistent answer extraction and
equivalence-aware majority voting), `tally()` (token accounting that counts
prompt tokens once per batched call, not once per sample), `rng()` and
`sample_seed()` (reproducible per-example randomness).

Put every sampled solution in `reasoning_traces`: grading extracts a per-sample
answer from each one, which is what `pass@k` and the majority-vote-vs-k curves
are computed from. `extra` is free-form but must be JSON-serialisable.

Baselines already implemented: `direct` (no CoT), `cot_zeroshot`, `cot_fewshot`,
`self_consistency` (majority vote over k samples), `best_of_n` (with a pluggable
scorer hook, including `"module:function"` scorers), `self_refine` and
`self_verify`.

### Varying *how* the question is asked

Prompt wording is an experimental axis, not a constant, so it is configurable
without new code. Pass a `style` dict to any strategy and it overrides the
system prompt, the instruction, the answer-format wording, and the few-shot count
and order:

```yaml
strategy:
  name: self_consistency
  params:
    k: 24
    style:
      name: c1
      system: "You are an expert mathematician. Be rigorous."
      instruction: "Solve the problem step by step."
      hint: "End your response with \\boxed{}."
      n_shots: 4
      shot_order: reverse
```

`style` lands in `strategy.params`, so it is part of the config hash: two styles
are two different runs with two different output directories, which is what you
want when comparing elicitation configurations. `system: ""` omits the system
turn entirely, which is a distinct condition from the default prompt.

### Per-sample accounting

`StrategyResult.sample_stats` (populated via `self.per_sample_stats(groups)`)
records per-sample completion tokens, finish reason and mean logprob, aligned
with `reasoning_traces`. It is written to every record, which is what makes
**budget-matched** comparisons (equal output tokens rather than equal sample
count) and confidence-weighted voting computable offline, without re-running
inference. Populate it in any new strategy.

---

## Datasets

All loaders return the same schema: `{id, question, gold_answer,
gold_reasoning?, choices?, answer_type, meta}` with `answer_type` in
`{math, mc, bool, text}`. Multiple-choice gold answers are letters.

| Name | HF id | config | split | Verified | Notes |
|---|---|---|---|---|---|
| `toy` | built in | – | `test` | yes | 24 synthetic items; powers the smoke test and CI, needs no network |
| `gsm8k` | `openai/gsm8k` | `main` | `test` | yes | gold follows `#### ` |
| `math500` | `HuggingFaceH4/MATH-500` | – | `test` | yes | 500 items |
| `aime` | `Maxwell-Jia/AIME_2024` | – | `train` | yes | `subset` selects the year; only 30 items |
| `aime` (2025) | `math-ai/aime25` | – | `test` | **partial** | id resolves; column names not verified |
| `bbh` | `lukaemon/bbh` | task name | `test` | yes | 250 items per task; `answer_type` per subset |
| `musr` | `TAUR-Lab/MuSR` | `default` | domain name | yes | domains are **splits**, not configs |
| `arc_challenge` | `allenai/ai2_arc` | `ARC-Challenge` | `test` | yes | multiple choice |
| `gpqa_diamond` | `Idavidrein/gpqa` | `gpqa_diamond` | `train` | yes | **gated**: accept the terms, set `HF_TOKEN`; options shuffled deterministically |
| `gsm_symbolic` | `apple/GSM-Symbolic` | `main`/`p1`/`p2` | `test` | **NO** | assumed by analogy with GSM8K; confirm with `--dry-run` |

Verification was done by reading the dataset pages, because the HuggingFace API
was unreachable from the development machine. `src/datasets_.py` carries the
`verified` flag and the reasoning per dataset, and `load_dataset_examples` logs a
warning for any unverified loader. **Run `--dry-run` on a new dataset before
spending quota on it** — it loads the data, prints the plan, and exits.

Column names are looked up from a list of candidates rather than hard-coded, so
an upstream rename produces a clear error naming the loader to fix instead of a
silently wrong gold answer.

Subsampling is mandatory on this budget and is made reproducible: indices are
drawn with `random.Random(subsample_seed).sample(...)`, sorted, and recorded in
each example's `meta` and in the config hash. The original dataset row index is
preserved in `meta["orig_index"]`, and it — not the position in the subsample —
is what the result uid is built from.

---

## Model backends

Selected by `model.backend`:

| Value | Behaviour |
|---|---|
| `hf` | **Default.** HuggingFace `transformers`. Always works; the project never hard-depends on vLLM. |
| `vllm` | vLLM for throughput. Errors clearly if vLLM is unavailable. |
| `auto` | vLLM if importable and not 4-bit, else `hf` with a logged reason. |
| `mock` | `MockBackend`: templated deterministic outputs, CPU, no model. Powers the smoke test and all CI. |

Supported loading paths: fp16 on a single GPU; `device_map="auto"` sharded
across 2x T4 via accelerate; 4-bit NF4 through bitsandbytes with fp16 compute.
GPU count, names, compute capability and free VRAM are detected and logged every
session. `transformers` renamed `from_pretrained(torch_dtype=)` to `dtype=` in
5.x, so the loader introspects the signature and passes whichever exists
(verified against transformers 5.14.1).

Two failure modes are handled rather than left to bite you mid-run:

- **CUDA OOM** does not end a session: the base backend halves the batch and
  retries, down to a single prompt.
- **Greedy sampling with n > 1** is rejected outright by `transformers`
  (`num_return_sequences>1` requires sampling or beams). Since greedy decoding is
  deterministic, the backend generates once and replicates, which honours the
  `n`-completions contract and is n times cheaper.

---

## Offline / no-internet sessions

Once, with internet on:

```bash
python scripts/prefetch_assets.py \
    --model Qwen/Qwen2.5-7B-Instruct --revision main \
    --datasets gsm8k math500 arc_challenge \
    --out /kaggle/working/assets --zip
```

Upload the bundle as a Kaggle Dataset, attach it, then:

```bash
export HF_HUB_OFFLINE=1
python -m src.runner --config configs/gsm8k_cot_zeroshot.yaml \
    --set model.local_path=/kaggle/input/<slug>/models/Qwen2.5-7B-Instruct \
    --set data.local_dir=/kaggle/input/<slug>/datasets/gsm8k
```

Datasets are cached as harness-schema JSONL, so the offline path produces exactly
the same `Example` objects as the online path. Keep the HF cache out of
`/kaggle/working` (`HF_HOME=/kaggle/temp/hf`): that directory is capped at 20 GB
and is uploaded as notebook output.

---

## GPU-hour budget plan

~30 GPU-hours per week is roughly **three 8-hour sessions**. A workable plan for
a paper's worth of results with a 7B model:

| Phase | Cost | Notes |
|---|---|---|
| Mock + tiny-model smoke tests | ~0 h | Do these every session. Free. |
| Throughput calibration | 0.2 h | Run 20 examples, read s/example, then plan. |
| Baselines: 4 strategies x 2 datasets x 200 examples | ~8-12 h | Single-sample strategies are cheap. |
| Self-consistency k=8 (1 dataset) | ~6-8 h | Costs ~8x the completion tokens of CoT. |
| Your method | ~6-8 h | Reserve at least one full session. |
| Extra seeds for CIs | ~4 h | Reviewers ask for this; budget it up front. |

Budget rules that follow from the numbers:

- Reserve one session per week for re-runs. Something always breaks.
- Prefer more examples on a cheap strategy over few examples on an expensive one:
  a 200-example subsample already gives a ~±7 point 95% CI, and 100 examples
  gives ~±10. Small benchmarks like GPQA-Diamond (198 items) and AIME (~30) can
  not support claims about small improvements without multiple seeds and a
  paired test.
- `src/budget.py:estimate_run_hours` plans a batch of cells before you spend on
  them, and `BudgetLedger` tracks cumulative hours against the weekly quota. The
  manifest records GPU-hours per session, and notebook cell 11 sums them.
- Grading, metrics, figures and tables cost **zero** GPU time. Iterate there
  freely.

---

## Tests

```bash
python -m pytest tests -q
```

Everything runs on CPU with no model and no network:

- `tests/test_answers.py` — answer extraction and equivalence, including cases
  that must **not** match, so the grader cannot be trivially permissive.
- `tests/test_datasets.py` — the `toy` loader, deterministic subsampling,
  sharding, JSONL round-trip.
- `tests/test_strategies.py` — every registered strategy against `MockBackend`,
  determinism, token accounting, the pluggable best-of-N scorer.
- `tests/test_metrics.py` — `pass@k` against brute-force enumeration of subsets,
  McNemar against hand-computed exact binomial p-values, bootstrap
  reproducibility, plus a smoke test that a real PDF and `.tex` are produced.
- `tests/test_checkpointing.py` — **kill and restart mid-run**: repeated
  interrupted sessions, a deliberately torn JSONL line, and assertions that every
  example completes exactly once with byte-identical records across sessions;
  plus the time guard stopping cleanly and remaining resumable.
- `tests/test_e2e.py` — the full pipeline through `MockBackend`, grading
  caching, external plugin loading, the aggregate CLI, and the smoke config in a
  clean subprocess.
- `tests/test_hf_backend.py` — the real `transformers` path against a tiny
  2-layer model built locally with random weights (no download, no GPU): left
  padding with mixed-length batches, chat templating, exact prompt/completion
  token counts, logprob gathering with n>1, seeded reproducibility, the
  greedy-with-n>1 case, and the bf16-to-fp16 downgrade on sm75/sm60. Skips
  automatically when torch is absent.

Current status: run `python -m pytest tests -q` locally; GLMM tests require
`statsmodels` (`pip install statsmodels`).

---

## Layout

```
requirements.txt          pinned deps + which ones Kaggle preinstalls
configs/                  one YAML per experiment cell; base.yaml holds defaults
src/
  config.py               dataclass config, YAML load, CLI override, config hash
  types.py                Example / GenParams / Completion / StrategyResult / ResultRecord
  datasets_.py            benchmark loaders in a unified schema
  prompts.py              prompt construction shared by all strategies
  generation.py           GenerationBackend ABC, batching, stop sequences, MockBackend
  models.py               hardware detection, HF backend, optional vLLM backend
  strategies/             pluggable reasoning strategies + registry (the extension point)
  answers.py              answer extraction + equivalence checking
  grading.py              separate cached grading pass
  metrics.py              accuracy, pass@k, token cost, bootstrap CIs, paired tests
  runner.py               the experiment driver
  checkpointing.py        deterministic ids, append-only JSONL, run manifest
  budget.py               wall-clock guard, GPU-hour accounting
  analysis/               aggregate -> LaTeX tables + matplotlib figures
scripts/prefetch_assets.py   download models/datasets for offline use
notebooks/kaggle_runner.ipynb the notebook you actually run
examples/method_template.py   copy this to add a method
tests/
```

---

## Relationship to `docs/METHOD_SPEC.md`

The method specification is written by a separate effort and this harness does
not implement it. The mapping, so the two fit together without changes here:

| The spec needs | The harness provides |
|---|---|
| A fully crossed item x model x configuration grid | One run per cell; `uid` includes model, strategy, dataset, index and seed, so cells never collide and each resumes independently |
| Elicitation configurations c0-c2 (core) + c3-c6 (separate arms) | `elicitation.id` in YAML; core=`c0,c1,c2`; `c3`,`c4`,`c5` demoted per adversarial review §6.3 |
| N samples per cell sharing one prefill | `GenParams.n`, honoured by both backends; `tally` counts the prompt once |
| Seed replicates for a noise floor | `--seed`; the seed is part of the uid, so seeds are separate resumable runs |
| Per-sample answer, token count and logprob | `sample_stats` plus grading's `sample_answers` / `sample_correct` on every record |
| Extraction failure as its own class | `final_answer=None` is preserved, counted, and reported as `extraction_failure_rate` (with a warning above 5%) |
| Budget-matched (equal-token) comparisons | per-sample completion tokens in `sample_stats`; `accuracy_vs_compute` and `tokens_per_correct` |
| `pass@N`, `maj@n`, bootstrap CIs, McNemar, Holm | `src/metrics.py`, all tested against closed forms |
| Offline analysis over a stored corpus | `graded.jsonl` plus `src/analysis/` |

Two deliberate deviations, both defensible but worth knowing:

1. **JSONL, not Parquet.** The spec asks for one Parquet file per cell. Parquet
   cannot be safely appended to, and append-only writing is what makes a run
   survive an abrupt kill. The JSONL is the source of truth; converting a
   finished run to Parquet is a two-line pandas call if a downstream tool wants
   it.
2. **Session budget defaults to 8 h, not 9 h.** Kaggle's documented cap is 12 h
   for GPU sessions (9 h applies to TPU), so 8 h leaves room for install, model
   download and a clean shutdown. Raise `runtime.time_budget_hours` if you want
   to use more of the window.

## Design decisions worth knowing

- **The work unit is one example.** Nothing is ever half-recorded, which is what
  makes "skip what is done" a correct resume strategy rather than a heuristic.
- **Run identity excludes performance knobs.** Batch size, device map, tensor
  parallel size, backend and the time budget do not change the config hash, so
  hardware changes between sessions do not orphan completed work. Everything that
  changes the output distribution does change it.
- **Grading is downstream of inference, always.** The expensive artifact is the
  raw generation; the cheap artifact is the judgement about it.
- **Token accounting counts prompt tokens once per batched call**, not once per
  sample, so "tokens per correct answer" does not silently overstate the cost of
  self-consistency by a factor of k.
- **fp16 is the default, not bf16.** T4 (sm75) and P100 (sm60) have no bf16
  tensor-core support. A bf16 request is downgraded with a warning rather than
  failing at hour three.
- **MockBackend is a first-class citizen.** It makes the entire pipeline testable
  on a laptop with no GPU, which is why CI can assert end-to-end accuracy is
  neither 0 nor 1.
