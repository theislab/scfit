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

Every metric here is a **mean over the evaluation groups** the callback feeds it: :class:`_MeanOverGroups`
owns the running-sum accumulator + the ``sum/total`` reduction, so each metric implements only its
per-group ``_statistic``.
"""

from collections.abc import Callable

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


class _MeanOverGroups(Metric):
    """Accumulate a per-group scalar statistic and report its mean over the evaluation groups.

    The validation callback calls ``update`` once per evaluation group; this base owns the running sum +
    group count (torchmetrics ``add_state`` with ``dist_reduce_fx="sum"``, so it reduces correctly across
    DDP ranks) and the ``sum / total`` reduction. A subclass implements only ``_statistic(pred, target) ->
    0-dim tensor``. ``target`` is optional so single-population monitors (which ignore it) share the base.
    """

    def __init__(self, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("stat_sum", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def _statistic(self, pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} must implement _statistic(pred, target).")

    def update(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> None:
        self.stat_sum += self._statistic(pred, target)
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.stat_sum / self.total


# TODO(metrics): generalize to a pluggable "aggregate-then-compare" metric —
#   AggregatedMetric(aggregate=mean|median|<callable>, compare=r2|cosine|correlation|<callable>).
#   The aggregation (population -> profile, the "combination logic") and the comparison would both be
#   parameters, so "median profiles by cosine" etc. fall out of one class. Kept mean+R² only for now
#   (only variant in use); add aggregators/comparisons when a second one is actually needed.
class MeanAggregatedRSquared(_MeanOverGroups):
    """R² between the feature-wise **mean profiles** of the two populations.

    Each population is first collapsed to its mean vector ``(n_features,)`` — the cell dimension is
    aggregated away — then R² is computed across features. So it compares *mean profiles*: it needs neither
    1-1 correspondence nor equal population sizes, but it is blind to spread / higher moments (unlike the
    full-distribution :class:`EnergyDistance` / :class:`MaximumMeanDiscrepancy`). Averaged over the
    evaluation groups the callback feeds it.
    """

    def _statistic(self, pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        pred_mean = pred.mean(dim=0)
        target_mean = target.mean(dim=0)
        ss_res = torch.sum((target_mean - pred_mean) ** 2)
        # total variance of the target mean-vector across features; clamp guards a (near-)constant target.
        ss_tot = torch.sum((target_mean - target_mean.mean()) ** 2).clamp_min(1e-12)
        return 1.0 - ss_res / ss_tot


class EnergyDistance(_MeanOverGroups):
    """Energy distance between two populations (unpaired, full-distribution), averaged over groups.

    ``2·E‖x−y‖² − E‖x−x'‖² − E‖y−y'‖²`` over squared-Euclidean pairwise distances — a proper distance
    between distributions that needs no 1-1 correspondence or equal population sizes.
    """

    def _statistic(self, pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        # Squared-Euclidean pairwise means (cellflow/scPerturb). cdist gives Euclidean; square it back to
        # sqeuclidean (memory-efficient — the explicit (x-y)**2 broadcast is O(n·m·d) and blows up at 1024²).
        sigma_pred = torch.cdist(pred, pred).square().mean()
        sigma_target = torch.cdist(target, target).square().mean()
        delta = torch.cdist(pred, target).square().mean()
        return 2.0 * delta - sigma_pred - sigma_target


class MaximumMeanDiscrepancy(_MeanOverGroups):
    """Multi-scale RBF Maximum Mean Discrepancy between two populations, averaged over groups.

    ``MMD² = E k(x,x') + E k(y,y') − 2·E k(x,y)`` averaged over a bandwidth sweep (``gammas``) — a kernel
    two-sample statistic; unpaired, needs no equal population sizes.
    """

    def __init__(self, gammas: list[float] | None = None, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.gammas = [2, 1, 0.5, 0.1, 0.01, 0.005] if gammas is None else gammas

    def _statistic(self, pred: torch.Tensor, target: torch.Tensor | None) -> torch.Tensor:
        mmds = []
        for gamma in self.gammas:
            xx = _rbf_kernel_torch(pred, pred, gamma=gamma).mean()
            xy = _rbf_kernel_torch(pred, target, gamma=gamma).mean()
            yy = _rbf_kernel_torch(target, target, gamma=gamma).mean()
            mmds.append(xx + yy - 2 * xy)
        # torch.stack keeps the reduction on-device (torch.tensor(...) would detach + force a CPU sync,
        # which misbehaves under GPU-resident streaming / preload_to_gpu).
        return torch.nanmean(torch.stack(mmds))


#: Named single-population statistics for :class:`PredictionDispersion` — ordinary torch reductions, so the
#: metric is "based on existing stats" rather than a bespoke accumulator.
_DISPERSION_STATS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "std": lambda pred: pred.std(dim=0).mean(),  # across-cell std, averaged over features
    "mean_abs": lambda pred: pred.abs().mean(),  # mean magnitude
}


class PredictionDispersion(_MeanOverGroups):
    """Monitor metric — a diagnostic on the PREDICTION alone (``target`` is accepted but ignored).

    Catches representation collapse (``std`` → 0) or explosion that population-*comparison* metrics miss.
    Like the distributional metrics here it averages its per-group statistic over the evaluation groups the
    callback feeds it; the statistic itself is a plain torch reduction (see :data:`_DISPERSION_STATS`), so it
    composes with whatever the predictor emits. Scored on both the prediction and any reference population,
    it shows whether the model's spread tracks the reference's.
    """

    def __init__(self, stat: str = "std", dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if stat not in _DISPERSION_STATS:
            raise ValueError(f"stat must be one of {sorted(_DISPERSION_STATS)}, found {stat!r}.")
        self._stat = stat

    def _statistic(self, pred: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        return _DISPERSION_STATS[self._stat](pred)
