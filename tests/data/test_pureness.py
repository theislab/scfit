"""Batch pureness — every emitted batch (primary and each linked source) is a single label.

Mirrors annbatch's ``test_batches_are_class_coherent`` one level up: decode each yielded batch and assert
it does not mix conditions, across per-row (``chunk_size=1``) and chunked (``chunk_size>1``) read configs.
Only the read config is parametrized.
"""

from __future__ import annotations

from itertools import islice

import pytest
from scheme_helpers import encoded_adata, only_leaf, perturbation_loader, rep


@pytest.mark.parametrize(
    "read",
    [
        {"batch_size": 8, "chunk_size": 1, "preload_nchunks": 8},  # per-row reads
        {"batch_size": 8, "chunk_size": 4, "preload_nchunks": 2},  # a batch spans two chunks
        {"batch_size": 4, "chunk_size": 4, "preload_nchunks": 4},  # one chunk == one batch
    ],
    ids=["chunk1", "chunk4_batch8", "chunk4_batch4"],
)
def test_every_batch_is_leaf_pure(read: dict):
    # runs of 16 per leaf (>= the largest chunk_size) so chunked reads stay within one leaf
    adata = encoded_adata(("A", "B"), ("control", "d1", "d2", "d3"), n_per_combo=16)
    for batch in islice(perturbation_loader(adata, **read), 40):
        only_leaf(rep(batch, "primary"))  # primary: raises if the batch mixes leaves
        only_leaf(rep(batch, "ctrl"))  # source: matched control, also one leaf
