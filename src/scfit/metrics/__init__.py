"""Validation metrics + the name→factory registry the harness selects from.

See :mod:`scfit.metrics._metrics` for the regime taxonomy (distributional / paired / monitor). Paired
(1-1) metrics are ordinary supervised metrics reused straight from ``torchmetrics``; only the
distributional and monitor metrics are implemented here.
"""

from functools import partial

from torchmetrics import MeanAbsoluteError, MeanSquaredError

from scfit.metrics._metrics import (
    EnergyDistance,
    MaximumMeanDiscrepancy,
    MeanAggregatedRSquared,
    PredictionDispersion,
)

#: name -> zero-arg factory, grouped by the sample-correspondence the metric assumes:
METRICS_REGISTRY = {
    # distributional (unpaired populations, full samples)
    "e-dist": EnergyDistance,
    "mmd": MaximumMeanDiscrepancy,
    # aggregated (unpaired, summary-statistic: R² of mean profiles)
    "mean_aggregated_r_squared": MeanAggregatedRSquared,
    # paired (1-1) — reused from torchmetrics, for imputation / reconstruction / FM objectives
    "mse": MeanSquaredError,
    "mae": MeanAbsoluteError,
    # monitor (single-population diagnostics)
    "pred_std": partial(PredictionDispersion, "std"),
    "pred_mean_abs": partial(PredictionDispersion, "mean_abs"),
}

__all__ = [
    "EnergyDistance",
    "MaximumMeanDiscrepancy",
    "MeanAggregatedRSquared",
    "PredictionDispersion",
    "METRICS_REGISTRY",
]
