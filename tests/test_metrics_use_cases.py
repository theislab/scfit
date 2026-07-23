"""Use-case / long-term-vision tests for ``scfit.metrics``.

Where :mod:`tests.test_metrics` pins each metric's *math* against hand-computed answers, this suite pins
the *behaviors the metrics exist to provide* — how the validation harness would use each one to **rank
models** in the single-cell settings the module targets (the regime taxonomy in
``scfit/metrics/_metrics.py``):

* **Distributional (unpaired)** — *perturbation-response prediction*: a model predicts the whole
  *population* of cells under a perturbation and we compare it to the measured population with no
  cell-to-cell correspondence (scGen / CPA / cellflow / scPerturb style). Metrics: ``e-dist``, ``mmd``.
* **Aggregated (mean profile)** — the "R² on the mean expression profile" reported across perturbation
  benchmarks. Metric: ``mean_aggregated_r_squared``.
* **Paired (1-1)** — *gene imputation / denoising / masked-gene reconstruction*: a held-out truth exists
  per cell. Metrics: ``mse``, ``mae`` (reused from ``torchmetrics``).
* **Monitor (single population)** — *representation-collapse / explosion* diagnostics (a VAE whose latent
  collapses, a perturbation model that just predicts the mean). Metric: ``pred_std`` / ``pred_mean_abs``.

Each test is deterministic (seeded) and asserts a *discriminative* property — ordering, invariance, or
targeted sensitivity — i.e. the thing that makes the metric usable for model selection, not merely a
point value.

One subtle fact is pinned deliberately: this :class:`EnergyDistance` uses **squared** Euclidean distances
(the cellflow/scPerturb convention), which collapses algebraically to ``2·‖mean_pred − mean_target‖²``.
So ``e-dist`` here is a *mean-only* statistic — like ``mean_aggregated_r_squared`` it is blind to spread;
**only MMD** sees higher moments. The tests below encode which failure modes each metric can and cannot
catch, so a future change (e.g. switching to non-squared Euclidean) trips a test rather than silently
altering what "e-dist" means.
"""

from __future__ import annotations

import pytest
import torch

from scfit.metrics import (
    METRICS_REGISTRY,
    EnergyDistance,
    MaximumMeanDiscrepancy,
    MeanAggregatedRSquared,
    PredictionDispersion,
)

# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────


def _score(metric, pred: torch.Tensor, target: torch.Tensor | None = None) -> float:
    """One update + compute, as the validation callback would run a single evaluation group."""
    metric.update(pred, target)
    return float(metric.compute())


def _score_by_name(name: str, pred: torch.Tensor, target: torch.Tensor) -> float:
    """Score through the name→factory registry — exactly how the harness selects a metric."""
    return _score(METRICS_REGISTRY[name](), pred, target)


def _centered(n: int, d: int) -> torch.Tensor:
    """A population whose per-feature mean is *exactly* zero, so an added shift is the entire mean.

    Lets a test control the mean and the spread independently: ``_centered(n, d) + s`` has mean ``s`` in
    every feature regardless of ``n``, and ``k * _centered(n, d)`` scales spread while keeping mean 0.
    """
    z = torch.randn(n, d)
    return z - z.mean(dim=0)


# ══ A. Distributional regime — perturbation-response prediction ═══════════════════════════════════════


@pytest.mark.parametrize("name", ["e-dist", "mmd"])
def test_perturbation_prediction_closer_population_scores_lower(name: str):
    """The core model-selection property: a predicted perturbed population whose mean is farther from the
    measured population scores strictly worse. This is what lets the harness pick the better predictor
    when there is no cell-to-cell correspondence to compute a per-cell error against.
    """
    torch.manual_seed(0)
    n, d = 256, 5
    measured = _centered(n, d)  # the real perturbed population (mean 0)
    scores = [_score_by_name(name, _centered(n, d) + shift, measured) for shift in (0.25, 1.0, 2.5)]
    assert scores[0] < scores[1] < scores[2]  # monotone in how far the predicted mean is pushed off


@pytest.mark.parametrize("name", ["e-dist", "mmd"])
def test_distributional_metrics_are_permutation_invariant(name: str):
    """No 1-1 correspondence is assumed, so relabelling cells within a population must not move the score
    — the defining contract of the distributional regime (contrast the paired metrics below).
    """
    torch.manual_seed(1)
    pred, target = torch.randn(64, 6), torch.randn(80, 6)
    base = _score_by_name(name, pred, target)
    shuffled = _score_by_name(name, pred[torch.randperm(64)], target[torch.randperm(80)])
    assert shuffled == pytest.approx(base, abs=1e-4)


def test_energy_distance_is_mean_only_by_squared_euclidean_construction():
    """Pin the (easily-overlooked) consequence of the squared-Euclidean convention: this energy distance
    equals ``2·‖mean_pred − mean_target‖²`` — a function of the population *means* alone. A refactor to
    non-squared Euclidean (a genuine distributional distance) would break this and should be a conscious,
    test-updating decision, not a silent change to what ``e-dist`` measures.
    """
    torch.manual_seed(2)
    for _ in range(5):
        pred = torch.randn(80, 6) + torch.randn(6)  # differ in BOTH mean and spread
        target = 2.0 * torch.randn(110, 6) + torch.randn(6)
        mean_only = 2.0 * float(((pred.mean(0) - target.mean(0)) ** 2).sum())
        assert _score(EnergyDistance(), pred, target) == pytest.approx(mean_only, rel=1e-4)


def test_energy_distance_nonnegative_for_distinct_populations():
    """As a divergence used for ranking it must not reward the wrong model with a negative score; the
    squared-Euclidean form is provably ``2·‖Δμ‖² ≥ 0`` (0 only when the means coincide).
    """
    torch.manual_seed(3)
    for _ in range(30):
        assert _score(EnergyDistance(), torch.randn(50, 6), torch.randn(70, 6)) >= -1e-5


# ══ B. Paired-vs-distributional contrast — why the regime split exists ════════════════════════════════


def test_row_correspondence_matters_only_for_paired_metrics():
    """The single most important distinction driving the regime taxonomy: shuffle the prediction's cell
    order (same population, correspondence destroyed) and the *paired* metric (gene-imputation error)
    jumps from perfect to bad, while the *distributional* metrics stay at their perfect-match fixed point.
    Picking the wrong regime for a task silently mis-scores every model.
    """
    torch.manual_seed(4)
    truth = torch.randn(64, 8)
    shuffled_pred = truth[torch.randperm(64)]  # identical population, wrong per-cell alignment

    # paired (imputation): correspondence broken -> error appears
    assert _score_by_name("mse", truth, truth) == pytest.approx(0.0, abs=1e-7)
    assert _score_by_name("mse", shuffled_pred, truth) > 0.1

    # distributional: same population -> still a perfect match
    assert _score(EnergyDistance(), shuffled_pred, truth) == pytest.approx(0.0, abs=1e-3)
    assert _score(MaximumMeanDiscrepancy(), shuffled_pred, truth) == pytest.approx(0.0, abs=1e-4)


def test_only_mmd_catches_variance_collapse_when_means_match():
    """A classic perturbation-model failure: the prediction reproduces the mean response but collapses the
    biological variance. ``mean_aggregated_r_squared`` (mean profile) and ``e-dist`` (mean-only, see above)
    both call it perfect; **only MMD**, which sees higher moments, flags it. This test is the map of which
    metric to reach for when variance — not the mean — is the thing that must be right.
    """
    torch.manual_seed(5)
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    measured = profile + 1.0 * _centered(512, 5)  # real population spread
    predicted = profile + 4.0 * _centered(512, 5)  # SAME mean profile, 4× the spread

    assert _score(MeanAggregatedRSquared(), predicted, measured) == pytest.approx(1.0, abs=1e-4)  # blind
    assert _score(EnergyDistance(), predicted, measured) == pytest.approx(0.0, abs=1e-3)  # blind (mean-only)
    assert _score(MaximumMeanDiscrepancy(), predicted, measured) > 1e-2  # catches the collapse


# ══ C. Aggregated regime — R² on the mean expression profile ══════════════════════════════════════════


def test_mean_r2_ranks_closer_mean_profile_higher():
    """The benchmark use: R²-of-mean should rank a model whose mean expression profile is closer to the
    measured profile higher (toward 1), so it is usable to select perturbation predictors.
    """
    torch.manual_seed(6)
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    measured = profile + _centered(400, 5)
    good = _score(MeanAggregatedRSquared(), (profile + 0.2) + _centered(400, 5), measured)
    bad = _score(MeanAggregatedRSquared(), (profile + 2.0) + _centered(400, 5), measured)
    assert good > bad
    assert good > 0.9  # a near-perfect mean profile lands close to 1


def test_mean_r2_needs_neither_equal_sizes_nor_correspondence():
    """Its selling point over the paired metrics: the two populations may have different cell counts and
    no alignment — matching *mean profiles* alone drives R²→1.
    """
    torch.manual_seed(7)
    profile = torch.tensor([0.5, -1.0, 2.0, 3.5])
    measured = profile + _centered(300, 4)  # 300 cells
    predicted = profile + 2.0 * _centered(120, 4)  # 120 cells, different spread, same mean
    assert _score(MeanAggregatedRSquared(), predicted, measured) == pytest.approx(1.0, abs=1e-4)


# ══ D. Paired regime — gene imputation / denoising ════════════════════════════════════════════════════


@pytest.mark.parametrize("name", ["mse", "mae"])
def test_gene_imputation_beats_per_gene_mean_baseline(name: str):
    """Gene imputation's sanity baseline is "predict each gene's population mean" (cell-agnostic). A
    cell-aware imputer that recovers per-cell structure must score strictly better than that baseline —
    otherwise the model has learned nothing beyond the marginal.
    """
    torch.manual_seed(8)
    n_cells, n_genes = 200, 20
    truth = torch.randn(n_cells, n_genes)
    cell_aware = truth + 0.1 * torch.randn(n_cells, n_genes)  # recovers per-cell signal
    mean_baseline = truth.mean(dim=0, keepdim=True).repeat(n_cells, 1)  # per-gene mean, ignores the cell
    assert _score_by_name(name, cell_aware, truth) < _score_by_name(name, mean_baseline, truth)


@pytest.mark.parametrize("name", ["mse", "mae"])
def test_gene_imputation_error_grows_with_corruption(name: str):
    """Monotonicity a denoising/reconstruction objective relies on: more corruption in the reconstruction
    → strictly larger paired error, so the metric orders checkpoints by reconstruction quality.
    """
    torch.manual_seed(9)
    n_cells, n_genes = 128, 12
    truth = torch.randn(n_cells, n_genes)
    noise = torch.randn(n_cells, n_genes)
    scores = [_score_by_name(name, truth + level * noise, truth) for level in (0.1, 0.5, 2.0)]
    assert scores[0] < scores[1] < scores[2]


# ══ E. Monitor regime — collapse / explosion diagnostics ══════════════════════════════════════════════


def test_dispersion_flags_representation_collapse():
    """The monitor catches what every population-*comparison* metric misses: a prediction that is
    (near-)constant across cells — latent/posterior collapse. Its across-cell std goes to 0 while a
    healthy prediction retains spread.
    """
    torch.manual_seed(10)
    healthy = torch.randn(128, 10)
    collapsed = torch.full((128, 10), 3.0)  # every cell identical
    assert _score(PredictionDispersion("std"), collapsed) == pytest.approx(0.0, abs=1e-6)
    assert _score(PredictionDispersion("std"), healthy) > 0.5


def test_dispersion_detects_under_dispersion_against_reference():
    """Scored on the prediction and on the reference population, the monitor shows whether the model's
    spread tracks the data's. A model that just predicts the per-gene mean (the "predict-the-mean" cop-out)
    has far lower dispersion than the population it is meant to reproduce.
    """
    torch.manual_seed(11)
    reference = torch.randn(256, 8) * 1.5
    predict_the_mean = reference.mean(dim=0, keepdim=True).repeat(256, 1)
    ref_spread = _score(PredictionDispersion("std"), reference)
    model_spread = _score(PredictionDispersion("std"), predict_the_mean)
    assert model_spread == pytest.approx(0.0, abs=1e-6)
    assert model_spread < ref_spread


def test_mean_abs_monitor_flags_activation_explosion():
    """The complementary failure the ``mean_abs`` statistic exists for: prediction magnitudes blowing up
    (diverged training) produce a large mean-absolute value versus a well-behaved prediction.
    """
    torch.manual_seed(12)
    normal = torch.randn(128, 10)
    exploded = torch.randn(128, 10) * 100.0
    assert _score(PredictionDispersion("mean_abs"), exploded) > 10.0 * _score(PredictionDispersion("mean_abs"), normal)


# ══ F. Harness integration — how the validation loop drives these ═════════════════════════════════════

# The averaging metrics (per-group statistic, then mean over groups); paired mse/mae POOL samples instead,
# so the "== mean of per-group scores" identity below is specific to these.
_AVERAGING = ["e-dist", "mmd", "mean_aggregated_r_squared", "pred_std"]


@pytest.mark.parametrize("name", _AVERAGING)
def test_metric_averages_over_evaluation_groups(name: str):
    """The documented callback pattern: ``update`` once per evaluation group (e.g. per perturbation /
    per cell type), then ``compute`` once. The result must equal the mean of the per-group scores, so
    adding groups re-weights the reported number the obvious way.
    """
    torch.manual_seed(13)
    groups = [(torch.randn(40, 5) + i, torch.randn(50, 5)) for i in range(4)]
    per_group = [_score_by_name(name, pred, target) for pred, target in groups]

    accumulated = METRICS_REGISTRY[name]()
    for pred, target in groups:
        accumulated.update(pred, target)
    assert float(accumulated.compute()) == pytest.approx(sum(per_group) / len(per_group), rel=1e-5)


@pytest.mark.parametrize("name", _AVERAGING)
def test_metric_is_reusable_across_epochs_via_reset(name: str):
    """A single metric instance is reused every validation epoch, so ``reset`` must fully clear the
    accumulated state — after it, the score reflects only the new epoch's group.
    """
    torch.manual_seed(14)
    metric = METRICS_REGISTRY[name]()
    metric.update(torch.randn(30, 5) + 5.0, torch.randn(40, 5))  # "epoch 0" — pollutes the state
    metric.compute()
    metric.reset()

    fresh_pred, fresh_target = torch.randn(30, 5), torch.randn(40, 5)
    metric.update(fresh_pred, fresh_target)
    assert float(metric.compute()) == pytest.approx(_score_by_name(name, fresh_pred, fresh_target), rel=1e-5)


def test_harness_selects_by_name_and_the_better_model_wins_in_each_regime():
    """End-to-end: the harness holds metric *names*, builds them from the registry, and a model that is
    genuinely better under a regime's assumptions wins on that regime's metric. One representative per
    regime, over a shared good/bad pair.
    """
    torch.manual_seed(15)
    profile = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    measured = profile + _centered(300, 5)
    good = profile + 0.2 + _centered(300, 5)  # close in mean profile
    bad = profile + 3.0 + _centered(300, 5)  # far off

    assert _score_by_name("e-dist", good, measured) < _score_by_name("e-dist", bad, measured)  # distributional
    assert _score_by_name("mean_aggregated_r_squared", good, measured) > _score_by_name(
        "mean_aggregated_r_squared", bad, measured
    )  # aggregated
    # paired: give the good model per-cell-aligned truth, the bad model a mean-shifted guess
    assert _score_by_name("mse", measured + 0.1, measured) < _score_by_name("mse", measured + 3.0, measured)
