"""Sampler bookkeeping — epoch length, the run-length rule, and non-contiguous coverage.

The scfit analogs of annbatch's ``test_n_batches`` / ``test_sampling_invariants`` (epoch length),
``test_run_length_error_names_class_labels`` + ``test_zero_weight_class_exempt_from_run_length_rule``
(the ``chunk_size>1`` contiguous-run rule and its zero-weight exemption), and
``test_noncontiguous_class_samples_all_runs`` (a leaf split across runs is fully sampled). Schemes inline.
"""

from __future__ import annotations

from itertools import islice

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

from scheme_helpers import encoded_adata, only_leaf, rep

from scfit.data import Loader, Node, SamplerConfig, Scheme
from scfit.data._schema import uniform

COLS = ("cell_line", "drug")


def _root_only(adata, weights) -> Scheme:
    return Scheme(sources={"data": adata}, nodes={"root": Node("data", COLS, "X", weights)}, root="root", seed=0)


def test_epoch_length_is_root_obs_over_batch_size():
    # root = 2 lines × 3 drugs × 16 = 96 target cells; batch 8 → 12 batches per with-replacement pass
    adata = encoded_adata(("A", "B"), ("d1", "d2", "d3"), n_per_combo=16)
    weights = uniform([(cl, dr) for cl in ("A", "B") for dr in ("d1", "d2", "d3")])
    loader = Loader(_root_only(adata, weights), SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    assert loader._n_batches == 96 // 8
    list(islice(loader, loader._n_batches))  # consume exactly one pass
    assert loader._pos == loader._n_batches  # the pass boundary is real; the next __next__ starts fresh


def test_chunk_run_length_error_names_the_node():
    # a positive-weight leaf with a 2-row run shorter than chunk_size(4): annbatch raises, scfit re-raises
    # with the offending node's name prepended (so the user knows *which* node's runs are too short).
    adata = encoded_adata(("A", "B"), ("d1", "d2"), n_per_combo=2)
    weights = uniform([(cl, dr) for cl in ("A", "B") for dr in ("d1", "d2")])
    with pytest.raises(ValueError, match=r"node 'root'"):
        Loader(_root_only(adata, weights), SamplerConfig(batch_size=4, chunk_size=4, preload_nchunks=4))


def _mixed_runs() -> ad.AnnData:
    # sorted by (cell_line, drug): perturbed leaves in runs of 8 (>= chunk), controls in runs of 2 (< chunk)
    rows: list[tuple[str, str]] = []
    for cl in ("A", "B"):
        rows += [(cl, "control")] * 2 + [(cl, "d1")] * 8 + [(cl, "d2")] * 8
    obs = pd.DataFrame(rows, columns=list(COLS))
    obs.index = obs.index.astype(str)
    x = np.random.default_rng(0).random((len(obs), 4), dtype="float32")
    return ad.AnnData(X=x, obs=obs)


def test_zero_weight_leaf_is_exempt_from_run_length_rule():
    adata, cfg = _mixed_runs(), SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2)
    # excluding the short-run control (weight 0) → its 2-row run is exempt, chunk_size=4 builds fine
    Loader(_root_only(adata, uniform([(cl, dr) for cl in ("A", "B") for dr in ("d1", "d2")])), cfg)
    # giving that same control a positive weight → its short run now violates the rule
    with pytest.raises(ValueError, match=r"node 'root'"):
        Loader(_root_only(adata, uniform([(cl, dr) for cl in ("A", "B") for dr in ("control", "d1", "d2")])), cfg)


def test_noncontiguous_leaf_is_fully_sampled():
    # (A, d1) lives in two separate runs — rows [0:8) and [16:24); over many draws both runs must be hit
    obs = pd.DataFrame([("A", "d1")] * 8 + [("A", "d2")] * 8 + [("A", "d1")] * 8, columns=list(COLS))
    obs.index = obs.index.astype(str)
    x = np.stack(
        [
            np.zeros(24),  # LINE: all A → 0
            (obs["drug"] == "d2").to_numpy().astype(float),  # DRUG: d1 → 0, d2 → 1
            np.zeros(24),  # IS_CONTROL
            np.arange(24),  # ROW_ID
        ],
        axis=1,
    ).astype("float32")
    scheme = _root_only(ad.AnnData(x, obs=obs), uniform([("A", "d1"), ("A", "d2")]))
    loader = Loader(scheme, SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4))
    seen: set[int] = set()
    for b in islice(loader, 400):
        tgt = rep(b, "root")
        if only_leaf(tgt)[1] == 0:  # a (A, d1) batch
            seen.update(tgt[:, 3].astype(int).tolist())  # collect its ROW_IDs
    assert any(r < 8 for r in seen) and any(r >= 16 for r in seen), "both runs of (A, d1) must be sampled"
