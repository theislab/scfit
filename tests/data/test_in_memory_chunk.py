"""An in-memory node must be configured with ``chunk_size=1`` — the loader refuses to guess it.

A materialized (in-RAM) node gets no benefit from chunked contiguous reads and the run-length rule is
meaningless for it. Rather than silently rewrite the user's ``chunk_size`` (``SamplerConfig`` is
deliberately explicit — no hidden defaults), :class:`~binded.Loader` raises unless it is 1. Set that way,
a matched control child with short runs sits in memory fine — the case that raises annbatch's run-length
error when the same node is *streamed*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("annbatch")

import anndata as ad
from scheme_helpers import LINES, perturbation_scheme

from scfit.data import Loader, SamplerConfig

COLS = ("cell_line", "drug")


def _source() -> ad.AnnData:
    # sorted by (cell_line, drug): perturbed leaves in runs of 8 (>= chunk), controls in runs of 2 (< chunk)
    rows: list[tuple[str, str]] = []
    for cl in LINES:
        rows += [(cl, "control")] * 2
        rows += [(cl, "d1")] * 8
        rows += [(cl, "d2")] * 8
    obs = pd.DataFrame(rows, columns=list(COLS))
    obs.index = obs.index.astype(str)
    x = np.random.default_rng(0).random((len(obs), 4), dtype="float32")
    return ad.AnnData(X=x, obs=obs)


def test_in_memory_node_with_chunk_gt_one_raises():
    # in_memory control + chunk_size>1 → the loader refuses to silently coerce; it raises with guidance.
    cfg = SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2)
    with pytest.raises(ValueError, match=r"in_memory.*chunk_size=1|must use chunk_size=1"):
        Loader(perturbation_scheme(_source(), ctrl_in_memory=True), cfg)


def test_in_memory_node_at_chunk_one_builds_despite_short_runs():
    # the actual guarantee: at chunk_size=1 the short-run (len 2) control sits in memory fine — the case
    # that raises annbatch's run-length error when the same node is streamed (see the test below).
    cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)
    loader = Loader(perturbation_scheme(_source(), ctrl_in_memory=True), cfg)
    assert loader._cfg["ctrl"].chunk_size == 1
    assert isinstance(loader._nodes["ctrl"], ad.AnnData)  # materialized into RAM


def test_streamed_short_run_still_raises():
    # same short control runs, but NOT in memory → annbatch enforces the run-length rule itself.
    source = _source()
    cfg = SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2)
    with pytest.raises(ValueError, match="run|chunk"):
        Loader(perturbation_scheme(source, ctrl_in_memory=False), cfg)
