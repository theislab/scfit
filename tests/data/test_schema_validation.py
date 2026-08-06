"""Structural validation of the ``Stream`` spec + the ``Loader``'s sampler-kwargs resolution.

The Stream cases are data-free ``__init__`` guards. The Loader cases build a tiny in-memory loader to
exercise the resolution rules (a stream's own sampler kwargs win; else the loader's; else error) and the
cross-stream guards (reserved name, match_on ⊆ shared, in_memory ⇒ chunk_size=1, label_lookup coverage).
"""

from __future__ import annotations

import numpy as np
import pytest
from scheme_helpers import KEY, encoded_adata, uniform

from scfit.data import Loader, Stream

ADATA = encoded_adata(("A", "B"), ("d1", "d2"), 8)  # 4 groups × 8 cells (perturbed only — no control needed)
SRC = {KEY: ADATA}  # the sources mapping every Loader case below streams from
COLS = ("cell_line", "drug")
W = uniform([("A", "d1"), ("A", "d2"), ("B", "d1"), ("B", "d2")])
SAMPLER = {"batch_size": 8, "chunk_size": 1, "preload_nchunks": 8}


# ── Stream: shape guards (no Loader built) ───────────────────────────────────────────────────────────
def test_rep_normalizes_to_tuple():
    assert Stream("k", group_by=COLS, reps="X").reps == ("X",)  # a bare loc string becomes a 1-tuple
    assert Stream("k", group_by=COLS, reps=("X", "obsm/rep")).reps == ("X", "obsm/rep")


def test_source_key_must_be_non_empty_string():
    with pytest.raises(ValueError, match="source_key must be a non-empty string"):
        Stream("", group_by=COLS)


@pytest.mark.parametrize(
    ("kw", "exc", "msg"),
    [
        pytest.param({"group_by": ()}, ValueError, "group_by must be non-empty", id="empty_group_by"),
        pytest.param({"group_by": COLS, "reps": 123}, ValueError, "loc strings", id="non_string_rep"),
        pytest.param({"group_by": COLS, "reps": ""}, ValueError, "loc strings", id="empty_rep"),
        pytest.param({"group_by": COLS, "reps": ("X", "")}, ValueError, "loc strings", id="one_empty_rep"),
        pytest.param({"group_by": COLS, "weights": {("A",): 1.0}}, ValueError, "arity", id="weight_arity"),
        pytest.param({"group_by": COLS, "weights": {("A", "d1"): -1.0}}, ValueError, "non-negative", id="neg_weight"),
        pytest.param(
            {"group_by": COLS, "label_lookup": {("A",): {"c": np.zeros((1, 1))}}},
            ValueError,
            "label_lookup key",
            id="label_lookup_arity",
        ),
        # sampler kwargs are all-or-nothing on a Stream
        pytest.param({"group_by": COLS, "batch_size": 8}, ValueError, "all-or-nothing", id="partial_one"),
        pytest.param(
            {"group_by": COLS, "batch_size": 8, "chunk_size": 1}, ValueError, "all-or-nothing", id="partial_two"
        ),
    ],
)
def test_stream_rejects(kw: dict, exc: type[Exception], msg: str):
    with pytest.raises(exc, match=msg):
        Stream("k", **kw)


# ── Loader: sampler-kwargs resolution (the merge's new behavior — previously untested) ─────────────────
def test_stream_sampler_overrides_loader():
    # the primary sets its own batch_size=4; the loader default is 8 → the primary uses ITS OWN (4).
    ld = Loader(
        SRC,
        primary=Stream(KEY, group_by=COLS, weights=W, batch_size=4, chunk_size=1, preload_nchunks=4),
        batch_size=8,
        chunk_size=1,
        preload_nchunks=8,
        seed=0,
    )
    assert ld._cfg["primary"]["batch_size"] == 4
    assert next(iter(ld))["primary"]["X"].shape[0] == 4  # and it really yields 4-row batches


def test_stream_inherits_loader_sampler():
    ld = Loader(SRC, primary=Stream(KEY, group_by=COLS, weights=W), **SAMPLER, seed=0)
    assert ld._cfg["primary"]["batch_size"] == 8


def test_no_sampler_on_either_raises():
    with pytest.raises(ValueError, match="neither the Stream nor the Loader"):
        Loader(SRC, primary=Stream(KEY, group_by=COLS, weights=W), seed=0)


def test_loader_partial_sampler_raises():
    with pytest.raises(ValueError, match="all-or-nothing"):
        Loader(SRC, primary=Stream(KEY, group_by=COLS, weights=W), batch_size=8, seed=0)


def test_source_key_not_in_sources_raises():
    with pytest.raises(ValueError, match="source_key 'missing' not in sources"):
        Loader(SRC, primary=Stream("missing", group_by=COLS, weights=W), **SAMPLER, seed=0)


# ── Loader: cross-stream guards ────────────────────────────────────────────────────────────────────────
def test_reserved_primary_link_name():
    with pytest.raises(ValueError, match="reserved"):
        Loader(
            SRC,
            primary=Stream(KEY, group_by=COLS, weights=W),
            links={"primary": Stream(KEY, group_by=COLS, weights=W, match_on=("cell_line",))},
            **SAMPLER,
        )


def test_match_on_must_be_shared():
    with pytest.raises(ValueError, match="must be ⊆"):
        Loader(
            SRC,
            primary=Stream(KEY, group_by=("cell_line", "drug"), weights=W),
            links={
                "c": Stream(KEY, group_by=("drug",), match_on=("cell_line",), weights=uniform([("d1",), ("d2",)]))
            },
            **SAMPLER,
        )


def test_in_memory_requires_chunk_one():
    with pytest.raises(ValueError, match="chunk_size=1"):
        Loader(
            SRC,
            primary=Stream(KEY, group_by=COLS, weights=W, in_memory=True, batch_size=8, chunk_size=2, preload_nchunks=8),
            seed=0,
        )


def test_label_lookup_must_cover_positive_weight_labels():
    with pytest.raises(ValueError, match="label_lookup misses positive-weight"):
        Loader(
            SRC,
            primary=Stream(KEY, group_by=COLS, weights=W, label_lookup={("A", "d1"): {"c": np.zeros((1, 1))}}),
            **SAMPLER,
            seed=0,
        )
