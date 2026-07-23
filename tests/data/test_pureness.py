"""Batch pureness — every emitted batch (target and each bound source) is a single leaf.

Mirrors annbatch's ``test_batches_are_class_coherent`` one level up: decode each yielded batch and assert
it does not mix conditions, across per-row (``chunk_size=1``) and chunked (``chunk_size>1``) read configs.
The scheme is spelled out inline; only the read config is parametrized.
"""

from __future__ import annotations

from itertools import islice

import pytest

pytest.importorskip("annbatch")

from scheme_helpers import encoded_adata, only_leaf, rep

from scfit.data import Bind, Loader, Node, SamplerConfig, Scheme
from scfit.data._schema import uniform

COLS = ("cell_line", "drug")
PERT = [(cl, dr) for cl in ("A", "B") for dr in ("d1", "d2", "d3")]
CTRL = [(cl, "control") for cl in ("A", "B")]


def _matched_scheme(adata) -> Scheme:
    # root = perturbed combos; ctrl = control combos; ctrl bound to root on cell_line (same-context control)
    return Scheme(
        sources={"data": adata},
        nodes={
            "pert": Node("data", COLS, "X", uniform(PERT)),
            "ctrl": Node("data", COLS, "X", uniform(CTRL)),
        },
        root="pert",
        seed=0,
        binds=(Bind("pert", "ctrl", common=("cell_line",)),),
    )


@pytest.mark.parametrize(
    "cfg",
    [
        SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),  # per-row reads
        SamplerConfig(batch_size=8, chunk_size=4, preload_nchunks=2),  # a batch spans two chunks
        SamplerConfig(batch_size=4, chunk_size=4, preload_nchunks=4),  # one chunk == one batch
    ],
    ids=["chunk1", "chunk4_batch8", "chunk4_batch4"],
)
def test_every_batch_is_leaf_pure(cfg: SamplerConfig):
    # runs of 16 per leaf (>= the largest chunk_size) so chunked reads stay within one leaf
    scheme = _matched_scheme(encoded_adata(("A", "B"), ("control", "d1", "d2", "d3"), n_per_combo=16))
    for batch in islice(Loader(scheme, cfg), 40):
        only_leaf(rep(batch, "pert"))  # target: raises if the batch mixes leaves
        only_leaf(rep(batch, "ctrl"))  # source: matched control, also one leaf
