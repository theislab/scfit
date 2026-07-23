"""Loader is picklable mid-stream: the live annbatch iterators are dropped, RNG/state is kept."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("annbatch")

from scheme_helpers import feature_adata, perturbation_scheme

from scfit.data import Loader, SamplerConfig


def _loader(seed=0):
    scheme = perturbation_scheme(feature_adata(("A", "B"), ("control", "d1", "d2"), 16, n_genes=4), seed=seed)
    # condition_lookup omitted so the loader is plain-picklable (no closure); state/RNG is what we test here
    return Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))


def test_pickle_mid_stream_and_resume_deterministic():
    loader = _loader()
    it = iter(loader)
    for _ in range(2):  # advance into a pass → live annbatch generators + a wrapped sampler RNG exist
        next(it)
    blob = pickle.dumps(loader)  # would raise "cannot pickle 'generator'" without __getstate__/__reduce__

    # two loaders restored from the same checkpoint must produce identical streams
    la, lb = pickle.loads(blob), pickle.loads(blob)
    a = [next(la)["pert"]["X"] for _ in range(4)]  # target = the root node, keyed by name → rep loc
    b = [next(lb)["pert"]["X"] for _ in range(4)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].shape == (8, 4)
