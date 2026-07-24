"""Loader is picklable mid-stream: the live annbatch iterators are dropped, RNG/state is kept."""

from __future__ import annotations

import pickle

import numpy as np
from scheme_helpers import feature_adata, perturbation_loader


def _loader(seed=0):
    # feature_adata (random matrix — no identity to decode) so we only test state/RNG resumption
    return perturbation_loader(feature_adata(("A", "B"), ("control", "d1", "d2"), 16, n_genes=4), seed=seed)


def test_pickle_mid_stream_and_resume_deterministic():
    loader = _loader()
    it = iter(loader)
    for _ in range(2):  # advance into a pass → live annbatch generators + sampler RNG state exist
        next(it)
    blob = pickle.dumps(loader)  # would raise "cannot pickle 'generator'" without __getstate__

    # two loaders restored from the same checkpoint must produce identical streams
    la, lb = pickle.loads(blob), pickle.loads(blob)
    a = [next(la)["primary"]["X"] for _ in range(4)]  # target = the primary stream, keyed by name → rep loc
    b = [next(lb)["primary"]["X"] for _ in range(4)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert a[0].shape == (8, 4)
