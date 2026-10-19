"""``group_by=()``: every cell is one group, whose leaf is ``()``."""

from __future__ import annotations

import anndata as ad
import numpy as np

from scfit.data import EvalLoader, Loader, Stream


def _adata(start: int, n: int) -> ad.AnnData:
    """``X[:, 0]`` is each cell's id, so a batch shows which cells it read."""
    return ad.AnnData(X=np.arange(start, start + n, dtype=np.float32)[:, None])


def test_training_streams_every_cell_of_every_file_as_one_group():
    loader = Loader(
        {"d": [_adata(0, 8), _adata(8, 8)]},
        primary=Stream("d", group_by=()),
        sampler={"batch_size": 4, "chunk_size": 1, "preload_nchunks": 4},
    )
    seen = set()
    for batch in (next(loader) for _ in range(40)):
        assert batch["leaves"]["primary"] == ()
        seen |= set(np.asarray(batch["primary"]["X"])[:, 0].astype(int).tolist())
    assert seen == set(range(16))


def test_an_ungrouped_link_draws_from_every_cell_in_memory():
    loader = Loader(
        {"d": _adata(0, 8), "ctrl": _adata(100, 8)},
        primary=Stream("d", group_by=()),
        links={"ctrl": Stream("ctrl", group_by=(), in_memory=True)},
        sampler={"batch_size": 4, "chunk_size": 1, "preload_nchunks": 4},
    )
    batch = next(loader)
    assert batch["leaves"]["ctrl"] == ()
    assert (np.asarray(batch["ctrl"]["X"]) >= 100).all()


def test_eval_walks_the_one_group():
    loader = EvalLoader({"d": _adata(0, 6)}, primary=Stream("d", group_by=()))
    batches = list(loader)
    assert len(batches) == 1 and batches[0]["leaf"] == ()
    assert sorted(np.asarray(batches[0]["primary"]["X"])[:, 0].astype(int).tolist()) == list(range(6))
