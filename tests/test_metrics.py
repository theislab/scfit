"""Known-answer + registry coverage for every metric in ``scfit.metrics``.

Pure-torch (no data layer / annbatch): pins each metric's math against a hand-computed value and an
identity property, and asserts every name in ``METRICS_REGISTRY`` builds and runs. The loader→metrics
integration (populations off a real ``EvalLoader``) lives in ``tests/data/test_metrics_over_loader.py``.
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

# Every registry name grouped by the correspondence it assumes (mirrors metrics/__init__.py).
_DISTRIBUTIONAL = ("e-dist", "mmd")  # unpaired populations, full samples
_AGGREGATED = ("mean_aggregated_r_squared",)  # unpaired, mean-profile R²
_PAIRED = ("mse", "mae")  # 1-1, reused from torchmetrics
_MONITOR = ("pred_std", "pred_mean_abs")  # single-population diagnostics (target ignored)
_ALL_NAMES = _DISTRIBUTIONAL + _AGGREGATED + _PAIRED + _MONITOR


def test_registry_names_are_exactly_the_documented_set():
    # Guards against a metric being added/renamed without updating the grouped coverage below.
    assert set(METRICS_REGISTRY) == set(_ALL_NAMES)


@pytest.mark.parametrize("name", _ALL_NAMES)
def test_registry_factory_builds_updates_computes(name: str):
    metric = METRICS_REGISTRY[name]()
    pred = torch.randn(16, 5)
    target = torch.randn(16, 5)  # equal size: works for every regime incl. the paired mse/mae
    metric.update(pred, target)
    value = metric.compute()
    assert torch.isfinite(torch.as_tensor(value)), f"{name} produced a non-finite value"


@pytest.mark.parametrize("name", _DISTRIBUTIONAL + _AGGREGATED + _MONITOR)
def test_unpaired_metrics_accept_unequal_population_sizes(name: str):
    # the whole point of the distributional / aggregated / monitor regimes: no 1-1 correspondence,
    # so |pred| need not equal |target| (unlike the paired mse/mae).
    metric = METRICS_REGISTRY[name]()
    metric.update(torch.randn(16, 5), torch.randn(24, 5))
    assert torch.isfinite(metric.compute()).all()


# ── identity properties (pred IS target): the metric's "perfect match" fixed point ──────────────────


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("e-dist", 0.0),  # 2·delta − sigma_pred − sigma_target, all equal ⇒ 0
        ("mmd", 0.0),  # xx + yy − 2xy, all equal ⇒ 0
        ("mean_aggregated_r_squared", 1.0),  # ss_res = 0 ⇒ R² = 1
        ("mse", 0.0),
        ("mae", 0.0),
    ],
)
def test_identity_fixed_point(name: str, expected: float):
    x = torch.randn(32, 7)
    metric = METRICS_REGISTRY[name]()
    metric.update(x, x.clone())
    assert metric.compute().item() == pytest.approx(expected, abs=1e-5)


# ── hand-computed known answers (pin the actual math, not just the fixed point) ──────────────────────


def test_energy_distance_hand_computed():
    # pred = {0, 0}; target = {0, 2} (1-D).
    # sigma_pred = mean pairwise sqdist within pred = 0
    # sigma_target = mean of [[0,4],[4,0]] = 2
    # delta = mean of cdist(pred,target)^2 = mean([[0,4],[0,4]]) = 2
    # energy = 2·2 − 0 − 2 = 2
    pred = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[0.0], [2.0]])
    m = EnergyDistance()
    m.update(pred, target)
    assert m.compute().item() == pytest.approx(2.0, abs=1e-5)


def test_energy_distance_is_symmetric():
    a, b = torch.randn(10, 4), torch.randn(13, 4)
    m1, m2 = EnergyDistance(), EnergyDistance()
    m1.update(a, b)
    m2.update(b, a)
    assert m1.compute().item() == pytest.approx(m2.compute().item(), abs=1e-5)


def test_mean_aggregated_r_squared_hand_computed():
    # pred mean profile = [0,0,0]; target mean profile = [1,2,3].
    # ss_res = 1+4+9 = 14; ss_tot = (1-2)²+(2-2)²+(3-2)² = 2; R² = 1 − 14/2 = −6
    pred = torch.zeros(4, 3)
    target = torch.tensor([1.0, 2.0, 3.0]).repeat(4, 1)
    m = MeanAggregatedRSquared()
    m.update(pred, target)
    assert m.compute().item() == pytest.approx(-6.0, abs=1e-5)


def test_mmd_identity_and_nonnegative():
    x = torch.randn(20, 6)
    same = MaximumMeanDiscrepancy()
    same.update(x, x.clone())
    assert same.compute().item() == pytest.approx(0.0, abs=1e-5)
    # distinct populations ⇒ MMD ≥ 0 (up to float noise)
    diff = MaximumMeanDiscrepancy()
    diff.update(torch.randn(20, 6), torch.randn(20, 6) + 5.0)
    assert diff.compute().item() >= -1e-5


def test_mmd_respects_custom_gammas():
    m = MaximumMeanDiscrepancy(gammas=[0.5])
    assert m.gammas == [0.5]


# ── monitor metrics: statistic on the prediction alone, target ignored ───────────────────────────────


def test_prediction_dispersion_std_matches_torch_std():
    pred = torch.randn(30, 4)
    m = PredictionDispersion("std")
    m.update(pred, target=None)  # target optional / ignored
    assert m.compute().item() == pytest.approx(pred.std(dim=0).mean().item(), abs=1e-5)


def test_prediction_dispersion_mean_abs_matches_torch():
    pred = torch.randn(30, 4)
    m = PredictionDispersion("mean_abs")
    m.update(pred)
    assert m.compute().item() == pytest.approx(pred.abs().mean().item(), abs=1e-5)


def test_prediction_dispersion_ignores_target():
    pred = torch.randn(15, 3)
    with_t, without_t = PredictionDispersion("std"), PredictionDispersion("std")
    with_t.update(pred, target=torch.randn(99, 3))
    without_t.update(pred)
    assert with_t.compute().item() == pytest.approx(without_t.compute().item(), abs=1e-6)


def test_prediction_dispersion_rejects_unknown_stat():
    with pytest.raises(ValueError, match="stat must be one of"):
        PredictionDispersion("variance")


# ── accumulation: compute() averages over the groups the callback feeds ──────────────────────────────


def test_metric_averages_over_multiple_updates():
    m = MeanAggregatedRSquared()
    x = torch.randn(10, 4)
    m.update(x, x.clone())  # R² = 1
    m.update(torch.zeros(4, 3), torch.tensor([1.0, 2.0, 3.0]).repeat(4, 1))  # R² = −6
    assert m.compute().item() == pytest.approx((1.0 + -6.0) / 2, abs=1e-5)


def test_energy_distance_accumulates_then_averages():
    m = EnergyDistance()
    m.update(torch.tensor([[0.0], [0.0]]), torch.tensor([[0.0], [2.0]]))  # 2.0
    m.update(torch.zeros(2, 1), torch.zeros(2, 1))  # 0.0
    assert m.compute().item() == pytest.approx(1.0, abs=1e-5)


def test_all_gammas_default_are_used():
    # sanity: default gamma list is the documented multi-scale set
    assert MaximumMeanDiscrepancy().gammas == [2, 1, 0.5, 0.1, 0.01, 0.005]
