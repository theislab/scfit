"""Validation metrics, organized by the sample-correspondence they assume.

Three regimes, all sharing the torchmetrics ``Metric`` interface (``update(pred, target)`` + ``compute()``),
so the validation callback stays agnostic to which you plug in — it just runs
``metric.update(predictor.predict(batch), target)``:

* **Distributional (unpaired)** — no 1-1 correspondence; compare the two *populations* using all samples
  (pairwise distances / kernels). :class:`EnergyDistance`, :class:`MaximumMeanDiscrepancy`.
* **Aggregated (unpaired, summary-statistic)** — first collapse each population to a summary *profile*
  ``(n_features,)`` (the "combination logic"), then compare across the shared feature dimension.
  :class:`MeanAggregatedRSquared` (mean profile, R²). Needs neither correspondence nor equal population
  sizes, but only sees what the aggregation keeps, unlike the full distributional metrics. (A general
  pluggable aggregate-then-compare metric is a documented TODO on the class.)
* **Paired (1-1)** — ``pred[i]`` is the prediction for ``target[i]`` (same cell). Used for gene imputation /
  reconstruction / masked-token objectives where a held-out truth exists per sample. These are ordinary
  supervised metrics — *reuse torchmetrics* (``MeanSquaredError``, ``MeanAbsoluteError``,
  ``PearsonCorrCoef``, ...) rather than reimplement here.
* **Monitor (single-population)** — a diagnostic on the prediction alone (``target`` ignored), e.g.
  :class:`PredictionDispersion` for representation collapse/explosion.

The regime is a property of the eval data + predictor + metric, not of the harness.
"""

from collections.abc import Callable
from typing import Any

import torch
from torchmetrics import Metric

__all__ = [
    "EnergyDistance",
    "MaximumMeanDiscrepancy",
    "MeanAggregatedRSquared",
    "PredictionDispersion",
]


def _rbf_kernel_torch(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0):
    """Gaussian/RBF kernel — an internal of :class:`MaximumMeanDiscrepancy` (not part of the public API)."""
    return torch.exp(-gamma * torch.cdist(x1, x2) ** 2)


# TODO(metrics): generalize to a pluggable "aggregate-then-compare" metric —
#   AggregatedMetric(aggregate=mean|median|<callable>, compare=r2|cosine|correlation|<callable>).
#   The aggregation (population -> profile, the "combination logic") and the comparison would both be
#   parameters, so "median profiles by cosine" etc. fall out of one class. Kept mean+R² only for now
#   (only variant in use); add aggregators/comparisons when a second one is actually needed.
class MeanAggregatedRSquared(Metric):
    """R² between the feature-wise **mean profiles** of the two populations.

    Each population is first collapsed to its mean vector ``(n_features,)`` — the cell dimension is
    aggregated away — then R² is computed across features. So it compares *mean profiles*: it needs neither
    1-1 correspondence nor equal population sizes, but it is blind to spread / higher moments (unlike the
    full-distribution :class:`EnergyDistance` / :class:`MaximumMeanDiscrepancy`). Averaged over the
    evaluation groups the callback feeds it.
    """

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("r2_sum", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred_mean = pred.mean(dim=0)
        target_mean = target.mean(dim=0)
        ss_res = torch.sum((target_mean - pred_mean) ** 2)
        # total variance of the target mean-vector across features; clamp guards a (near-)constant target.
        ss_tot = torch.sum((target_mean - target_mean.mean()) ** 2).clamp_min(1e-12)
        self.r2_sum += 1.0 - ss_res / ss_tot
        self.total += 1

    def compute(self) -> Any:
        return self.r2_sum / self.total


class EnergyDistance(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("energy_distance_raw", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # Squared-Euclidean pairwise means (cellflow/scPerturb). cdist gives Euclidean; square it back to
        # sqeuclidean (memory-efficient — the explicit (x-y)**2 broadcast is O(n·m·d) and blows up at 1024²).
        sigma_pred = torch.cdist(pred, pred).square().mean()
        sigma_target = torch.cdist(target, target).square().mean()
        delta = torch.cdist(pred, target).square().mean()
        self.energy_distance_raw += 2.0 * delta - sigma_pred - sigma_target
        self.total += 1

    def compute(self) -> Any:
        return self.energy_distance_raw / self.total


class MaximumMeanDiscrepancy(Metric):
    def __init__(self, gammas: list[float] | None = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if gammas is None:
            self.gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
        else:
            self.gammas = gammas

        self.add_state("mmd_raw", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # computing mmd
        mmds = []
        for gamma in self.gammas:
            xx = _rbf_kernel_torch(pred, pred, gamma=gamma).mean()
            xy = _rbf_kernel_torch(pred, target, gamma=gamma).mean()
            yy = _rbf_kernel_torch(target, target, gamma=gamma).mean()
            mmds.append(xx + yy - 2 * xy)
        self.mmd_raw += torch.nanmean(torch.tensor(mmds))
        self.total += 1

    def compute(self) -> Any:
        return self.mmd_raw / self.total


#: Named single-population statistics for :class:`PredictionDispersion` — ordinary torch reductions, so the
#: metric is "based on existing stats" rather than a bespoke accumulator.
_DISPERSION_STATS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "std": lambda pred: pred.std(dim=0).mean(),  # across-cell std, averaged over features
    "mean_abs": lambda pred: pred.abs().mean(),  # mean magnitude
}


class PredictionDispersion(Metric):
    """Monitor metric — a diagnostic on the PREDICTION alone (``target`` is accepted but ignored).

    Catches representation collapse (``std`` → 0) or explosion that population-*comparison* metrics miss.
    Like the distributional metrics here it averages its per-group statistic over the evaluation groups the
    callback feeds it; the statistic itself is a plain torch reduction (see :data:`_DISPERSION_STATS`), so it
    composes with whatever the predictor emits. Scored on both the prediction and any reference population,
    it shows whether the model's spread tracks the reference's.
    """

    def __init__(self, stat: str = "std", dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if stat not in _DISPERSION_STATS:
            raise ValueError(f"stat must be one of {sorted(_DISPERSION_STATS)}, found {stat!r}.")
        self._stat = stat
        self.add_state("value_sum", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> None:
        self.value_sum += _DISPERSION_STATS[self._stat](pred)
        self.total += 1

    def compute(self) -> Any:
        return self.value_sum / self.total
