"""Synthetic corpora with known ground truth, shared by the analysis tests.

The statistical layer's whole job is to recover quantities that cannot be checked
by eye, so the tests need data whose answer is known in advance. Two generators:

* `make_corpus` -- hand-specified answer classes per cell. Used where the expected
  value is exact and countable (mode transitions, Jaccard overlaps, token-matched
  re-partitioning).
* `synthetic_design` -- draws cells from a logit-additive truth with named variance
  components, then samples binomially at a given `N`. Used to check that the
  variance decomposition recovers the components it was given, including the
  sampling-noise floor, which is the one number in the pipeline that is otherwise
  impossible to validate against anything but itself.

A helper module rather than fixtures in `conftest.py` because several test files
need the generators with different parameters, and parameterising a fixture that
takes eight arguments is worse than calling a function.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.analysis.corpus import Cell, Corpus, SampleRow

#: Answer-class labels that grade as wrong; distinct so a "wrong" mode is a real
#: competing class rather than a single lumped bucket.
WRONG_CLASSES = ("w1", "w2", "w3", "w4")
CORRECT_CLASS = "gold"

DEFAULT_TOKENS = 100


def make_cell(
    item_id: str,
    model: str,
    config: str,
    classes: Sequence[str],
    seed: int = 0,
    dataset: str = "synth",
    subset: Optional[str] = None,
    tokens: Sequence[int] | int = DEFAULT_TOKENS,
    template_id: Optional[str] = None,
) -> Cell:
    """A cell whose samples emit exactly `classes`, in that order.

    `CORRECT_CLASS` samples are graded correct; every other label is wrong. Sample
    order matters: `Cell.modal_class` breaks ties by first occurrence, so a test
    that wants a deterministic tie controls it by ordering here.
    """
    if isinstance(tokens, int):
        token_list = [tokens] * len(classes)
    else:
        token_list = list(tokens)
        if len(token_list) != len(classes):
            raise ValueError("tokens and classes must have the same length")
    samples = [
        SampleRow(
            item_id=item_id,
            dataset=dataset,
            subset=subset,
            model=model,
            config=config,
            seed=seed,
            sample_idx=i,
            answer=label,
            canonical_class=label,
            is_correct=(label == CORRECT_CLASS),
            tokens_completion=token_list[i],
        )
        for i, label in enumerate(classes)
    ]
    return Cell(
        item_id=item_id,
        dataset=dataset,
        subset=subset,
        model=model,
        config=config,
        seed=seed,
        gold_answer=CORRECT_CLASS,
        answer_type="math",
        samples=samples,
        template_id=template_id,
    )


def make_corpus(cells: Sequence[Cell]) -> Corpus:
    corpus = Corpus()
    for cell in cells:
        corpus.cells[cell.key()] = cell
    return corpus


def cell_from_counts(
    item_id: str,
    model: str,
    config: str,
    k: int,
    n: int,
    seed: int = 0,
    dataset: str = "synth",
    subset: Optional[str] = None,
    wrong_class: str = WRONG_CLASSES[0],
    tokens: int = DEFAULT_TOKENS,
    template_id: Optional[str] = None,
) -> Cell:
    """`k` correct out of `n`, with the wrong mass in a single competing class."""
    classes = [CORRECT_CLASS] * k + [wrong_class] * (n - k)
    return make_cell(
        item_id, model, config, classes, seed=seed, dataset=dataset,
        subset=subset, tokens=tokens, template_id=template_id,
    )


def corpus_from_probs(
    probs: Dict[Tuple[str, str, int], Sequence[float]],
    n_samples: int = 24,
    seed: int = 0,
    dataset: str = "synth",
    subset: Optional[str] = None,
    item_ids: Optional[Sequence[str]] = None,
) -> Corpus:
    """Binomial draws from explicit per-column success probabilities.

    `probs` maps `(model, config, sampling_seed)` to a per-item probability vector,
    all vectors aligned to the same item order. Lets a test state exactly what latent
    structure it is injecting -- a shared latent for the null, correlated latents for
    the disattenuation check -- instead of inferring it from variance parameters.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    lengths = {len(v) for v in probs.values()}
    if len(lengths) != 1:
        raise ValueError(f"all probability vectors must be the same length, got {lengths}")
    n_items = lengths.pop()
    ids = list(item_ids) if item_ids else [f"synth/{i:04d}" for i in range(n_items)]

    cells: List[Cell] = []
    for (model, config, sampling_seed), vector in sorted(probs.items()):
        for i, p in enumerate(vector):
            k = int(rng.binomial(n_samples, float(p)))
            cells.append(
                cell_from_counts(
                    ids[i], model, config, k, n_samples,
                    seed=sampling_seed, dataset=dataset, subset=subset,
                )
            )
    return make_corpus(cells)


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class SyntheticTruth:
    """The parameters a synthetic design was generated from.

    Two notions of "truth" live here and they are not interchangeable:

    * `shares` -- the variances of the Gaussian effects the design was drawn from.
      The population truth.
    * `estimator_shares` -- the components a noiseless fit of the *realised* latent
      logit array yields. This is what a correct estimator should return from noisy
      counts, and it is the right target for a recovery test, because it already
      accounts for finite-sample effect draws and for any clipping. Comparing
      against `shares` instead would conflate estimator error with the sampling
      variability of the effects themselves.
    """

    def __init__(
        self,
        sd_item: float,
        sd_model: float,
        sd_config: float,
        sd_item_model: float,
        sd_item_config: float,
        n_samples: int,
        latent_z: Any = None,
    ) -> None:
        self.sd_item = sd_item
        self.sd_model = sd_model
        self.sd_config = sd_config
        self.sd_item_model = sd_item_model
        self.sd_item_config = sd_item_config
        self.n_samples = n_samples
        #: (items, models, configs) latent logits, in sorted item / given model /
        #: given config order -- the same order `Corpus.counts_matrix` produces.
        self.latent_z = latent_z

    @property
    def components(self) -> Dict[str, float]:
        return {
            "item": self.sd_item**2,
            "model": self.sd_model**2,
            "config": self.sd_config**2,
            "item_model": self.sd_item_model**2,
            "item_config": self.sd_item_config**2,
        }

    @property
    def total(self) -> float:
        return sum(self.components.values())

    @property
    def shares(self) -> Dict[str, float]:
        total = self.total
        return {k: v / total for k, v in self.components.items()}

    def estimator_components(self) -> Dict[str, float]:
        """Components of a noiseless fit of the realised latent logit array."""
        from src.analysis.variance import COMPONENT_NAMES, components_from_ms, mean_squares

        if self.latent_z is None:
            raise ValueError("this design was built without a latent array")
        fit = mean_squares(self.latent_z)
        dims = fit["dims"]
        return components_from_ms(
            {name: float(fit["ms"][name]) for name in COMPONENT_NAMES},
            dims["n_items"],
            dims["n_models"],
            dims["n_configs"],
        )

    def estimator_shares(self) -> Dict[str, float]:
        from src.analysis.variance import shares_from_components

        shares, _ = shares_from_components(self.estimator_components())
        return {k: float(v) for k, v in shares.items()}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SyntheticTruth({self.components}, N={self.n_samples})"


def synthetic_design(
    n_items: int = 200,
    models: Sequence[str] = ("m0", "m1", "m2"),
    configs: Sequence[str] = ("c0", "c1", "c2", "c3"),
    n_samples: int = 24,
    sd_item: float = 1.2,
    sd_model: float = 0.5,
    sd_config: float = 0.3,
    sd_item_model: float = 0.6,
    sd_item_config: float = 0.8,
    intercept: float = 0.0,
    n_seeds: int = 2,
    seed: int = 20260729,
    dataset: str = "synth",
    subset: Optional[str] = None,
    clip: Optional[float] = 4.0,
) -> Tuple[Corpus, SyntheticTruth]:
    """A fully crossed corpus drawn from a known logit-additive truth.

    The latent success rate of cell (i, m, c) is
    `sigmoid(mu + a_i + b_m + g_c + (ab)_im + (ag)_ic)`, and the observed `k` is a
    binomial draw at `n_samples`. So the recoverable signal variance is exactly the
    sum of the squared standard deviations passed in, and everything above that in
    the empirical logit's variance is sampling noise -- which is the property the
    decomposition test asserts.

    `n_seeds` independent replicate draws are produced at the *reference*
    configuration with the same latent probabilities, which is what the noise-floor
    estimator consumes. `clip` bounds the latent logit so a synthetic run does not
    consist mostly of saturated cells (which would test censoring, not recovery).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, sd_item, size=n_items)
    b = rng.normal(0.0, sd_model, size=len(models))
    g = rng.normal(0.0, sd_config, size=len(configs))
    ab = rng.normal(0.0, sd_item_model, size=(n_items, len(models)))
    ag = rng.normal(0.0, sd_item_config, size=(n_items, len(configs)))

    cells: List[Cell] = []
    latent = np.zeros((n_items, len(models), len(configs)), dtype=float)
    for i in range(n_items):
        item_id = f"synth/{i:04d}"
        for mi, model in enumerate(models):
            for ci, config in enumerate(configs):
                z = intercept + a[i] + b[mi] + g[ci] + ab[i, mi] + ag[i, ci]
                if clip is not None:
                    z = float(np.clip(z, -clip, clip))
                latent[i, mi, ci] = z
                p = sigmoid(z)
                # Replicates only where the design calls for them (config c0), so a
                # test that forgets to restrict to seed 0 fails loudly.
                seeds = range(n_seeds) if ci == 0 else range(1)
                for s in seeds:
                    k = int(rng.binomial(n_samples, p))
                    cells.append(
                        cell_from_counts(
                            item_id, model, config, k, n_samples, seed=s,
                            dataset=dataset, subset=subset,
                        )
                    )
    truth = SyntheticTruth(
        sd_item=sd_item,
        sd_model=sd_model,
        sd_config=sd_config,
        sd_item_model=sd_item_model,
        sd_item_config=sd_item_config,
        n_samples=n_samples,
        latent_z=latent,
    )
    return make_corpus(cells), truth
