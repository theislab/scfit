"""An in-memory stream must be configured with ``chunk_size=1`` — the loader refuses to guess it.

A materialized (in-RAM) stream gets no benefit from chunked contiguous reads and the run-length rule is
meaningless for it. Rather than silently rewrite the user's ``chunk_size`` (the sampler kwargs are
deliberately explicit), :class:`~scfit.data.Loader` raises unless it is 1. Set that way, a matched control
with short runs sits in memory fine — the case that raises annbatch's run-length error when *streamed*.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scheme_helpers import LINES, perturbation_loader

from scfit.data import Source

COLS = ("cell_line", "drug")


def _source() -> ad.AnnData:
    # sorted by (cell_line, drug): perturbed labels in runs of 8 (>= chunk), controls in runs of 2 (< chunk)
    rows: list[tuple[str, str]] = []
    for cl in LINES:
        rows += [(cl, "control")] * 2 + [(cl, "d1")] * 8 + [(cl, "d2")] * 8
    obs = pd.DataFrame(rows, columns=list(COLS))
    obs.index = obs.index.astype(str)
    x = np.random.default_rng(0).random((len(obs), 4), dtype="float32")
    return ad.AnnData(X=x, obs=obs)


def test_in_memory_control_with_chunk_gt_one_raises():
    # in_memory control + chunk_size>1 → the loader refuses to silently coerce; it raises with guidance.
    with pytest.raises(ValueError, match=r"in_memory.*chunk_size|must use chunk_size=1"):
        perturbation_loader(_source(), ctrl_in_memory=True, batch_size=8, chunk_size=4, preload_nchunks=2)


def test_in_memory_control_at_chunk_one_builds_despite_short_runs():
    # the actual guarantee: at chunk_size=1 the short-run (len 2) control sits in memory fine — the case
    # that raises annbatch's run-length error when the same stream is streamed (see the test below).
    loader = perturbation_loader(_source(), ctrl_in_memory=True, batch_size=8, chunk_size=1, preload_nchunks=8)
    assert loader._cfg["ctrl"].chunk_size == 1
    assert isinstance(loader._resolved["ctrl"], Source)  # materialized into a RAM-backed Source
    assert isinstance(loader._resolved["ctrl"].adatas[0].X, np.ndarray)


def test_streamed_short_run_still_raises():
    # same short control runs, but NOT in memory → annbatch enforces the run-length rule itself.
    with pytest.raises(ValueError, match="run|chunk"):
        perturbation_loader(_source(), ctrl_in_memory=False, batch_size=8, chunk_size=4, preload_nchunks=2)
