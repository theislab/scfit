"""End-to-end DATA → METRICS: score real ``EvalLoader`` / ``Loader`` populations with every metric.

Ties the streaming layer to the scoring layer — the path the validation callback actually walks: a
control-rooted ``EvalLoader`` yields ``(source, target)`` populations per condition; the metrics turn
those into a scalar. Exercises the binding axes that change what the loader emits (single-context bind,
Loader vs EvalLoader, multi-key nodes → ``*_reps``) and asserts every registry metric that admits
unequal population sizes runs on them, plus the identity fixed points off loader-derived tensors.

Distributional/aggregated/monitor metrics allow unequal ``|source| != |target|``; the paired metrics
(``mse``/``mae``) need 1-1 correspondence, so here they are only checked on the identity (same tensor).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("annbatch")

import torch
from scheme_helpers import LINES, feature_adata, perturbation_scheme

from scfit.data import EvalLoader, Loader, SamplerConfig
from scfit.metrics import METRICS_REGISTRY

DRUGS = ("control", "d1", "d2")
N_GENES = 12
# to="torch" so loader output feeds the (torch) metrics directly, no jax needed.
_CFG = SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4, to="torch")

# metrics that accept two populations of different sizes (unpaired) — scorable on (source, target)
_UNPAIRED = ("e-dist", "mmd", "mean_aggregated_r_squared", "pred_std", "pred_mean_abs")
# identity fixed points (metric(x, x)) — includes the paired metrics (same tensor ⇒ same size)
_IDENTITY = {
    "e-dist": 0.0,
    "mmd": 0.0,
    "mean_aggregated_r_squared": 1.0,
    "mse": 0.0,
    "mae": 0.0,
}


def _adata(n_per_combo: int = 8, *, with_rep: bool = False, seed: int = 0):
    """Perturbation AnnData with a real (random) feature matrix + a control flag, per (line, drug) combo."""
    return feature_adata(LINES, DRUGS, n_per_combo, n_genes=N_GENES, seed=seed, obsm_rep=with_rep)


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


@pytest.mark.parametrize("name", _UNPAIRED)
def test_unpaired_metric_scores_evalloader_populations(name: str):
    """Every unpaired metric turns (source controls, target perturbed) into a finite scalar, condition
    after condition — the exact call the validation callback makes."""
    loader = EvalLoader(perturbation_scheme(_adata()), _CFG, condition_lookup=None)
    metric = METRICS_REGISTRY[name]()
    n = 0
    for out in loader.iter_conditions():  # one batch per control population
        metric.update(_t(out["source"]), _t(out["target"]))
        n += 1
    assert n == len(LINES)  # both control populations scored
    assert torch.isfinite(metric.compute()).all()


@pytest.mark.parametrize(("name", "expected"), list(_IDENTITY.items()))
def test_identity_fixed_point_on_loader_target(name: str, expected: float):
    """Feeding a loader-derived population against itself hits the documented fixed point (0 / R²=1)."""
    loader = EvalLoader(perturbation_scheme(_adata()), _CFG, condition_lookup=None)
    target = _t(next(loader.iter_conditions())["target"])
    metric = METRICS_REGISTRY[name]()
    metric.update(target, target.clone())
    assert metric.compute().item() == pytest.approx(expected, abs=1e-4)


def test_train_loader_target_population_is_scorable():
    """The train-time ``Loader`` (target-rooted) also yields a population the metrics can score."""
    loader = Loader(perturbation_scheme(_adata()), _CFG, condition_lookup=None)
    batch = next(iter(loader))
    target = _t(batch["target"])
    for name in _UNPAIRED:
        metric = METRICS_REGISTRY[name]()
        # score against a control-free reference of the same width; unpaired ⇒ sizes may differ
        metric.update(target, target[: max(1, target.shape[0] // 2)])
        assert torch.isfinite(metric.compute()).all()


def test_metrics_score_aligned_reps_from_multikey_node():
    """A node with >1 key emits ``target_reps`` (aligned reps of the same cells); metrics score a rep
    just as they score the primary stream — covers the multi-key binding axis feeding the scorer."""
    loader = EvalLoader(perturbation_scheme(_adata(with_rep=True), key=("X", "obsm/rep")), _CFG, condition_lookup=None)
    out = next(loader.iter_conditions())
    assert "target_reps" in out and set(out["target_reps"]) == {"X", "obsm/rep"}
    primary = _t(out["target_reps"]["X"])
    rep = _t(out["target_reps"]["obsm/rep"])
    # rep == X here, so the identity fixed point must hold across the two aligned streams
    m = METRICS_REGISTRY["e-dist"]()
    m.update(primary, rep)
    assert m.compute().item() == pytest.approx(0.0, abs=1e-4)
