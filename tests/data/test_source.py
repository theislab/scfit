"""The :class:`Source` abstraction — cross-file unified categoricals, rep-width guard, shared factorization.

A ``source_key`` resolves to one :class:`Source` (a list of AnnData). Its cells form one unified categorical
universe (categories unioned across files, obs concatenated in list order == the backings order), so a leaf
that lives in only one file resolves to that file when sampled. A streamed rep must share ``shape[1]`` across
the files. Streams naming the same key factorize the dataset's obs only once.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scheme_helpers import KEY, encoded_adata, perturbation_streams

from scfit.data import Loader, Source, Stream

READ = {"batch_size": 4, "chunk_size": 1, "preload_nchunks": 4, "to": None}


def _file(cell_types, file_id: int, n: int = 8) -> ad.AnnData:
    """A one-file AnnData: X col 0 encodes the file id (so a batch row reveals which file it came from)."""
    obs = pd.DataFrame({"cell_type": [ct for ct in cell_types for _ in range(n)]})
    x = np.stack([np.full(len(obs), file_id, dtype="float32"), np.arange(len(obs), dtype="float32")], axis=1)
    return ad.AnnData(x, obs=obs)


# ── unified categoricals across files ───────────────────────────────────────────────────────────
def test_cross_file_categoricals_are_unioned():
    # file A has {T, B}; file B has {NK, Mono} — disjoint. The Source unifies them into one string-sorted space.
    src = Source([_file(["T", "B"], 0), _file(["NK", "Mono"], 1)])
    f = src.factorize(("cell_type",))
    assert f.leaves == [("B",), ("Mono",), ("NK",), ("T",)]  # union of both files, string-sorted


def test_leaf_resolves_to_the_file_that_holds_it():
    # sampling a leaf present in only one file must pull rows from that file (the list-order invariant).
    a, b = _file(["T", "B"], file_id=0), _file(["NK", "Mono"], file_id=1)
    loader = Loader({KEY: [a, b]}, primary=Stream(KEY, group_by=("cell_type",), weights={("NK",): 1.0}), **READ)
    for _ in range(8):
        x = np.asarray(next(iter(loader))["primary"]["X"])
        assert (x[:, 0] == 1).all()  # NK lives only in file B (file_id 1)


# ── rep width guard ─────────────────────────────────────────────────────────────────────────────
def test_rep_width_mismatch_raises():
    a = _file(["T"], 0)  # X width 2
    b = ad.AnnData(np.zeros((8, 3), "float32"), obs=pd.DataFrame({"cell_type": ["NK"] * 8}))  # X width 3
    with pytest.raises(ValueError, match=r"inconsistent shape\[1\]"):
        Source([a, b]).rep("X")


def test_shared_obsm_rep_ok_despite_differing_x_width():
    # different raw X widths are fine as long as the STREAMED rep (a shared obsm embedding) is aligned.
    a = _file(["T"], 0)  # X width 2
    b = ad.AnnData(np.zeros((8, 3), "float32"), obs=pd.DataFrame({"cell_type": ["NK"] * 8}))  # X width 3
    a.obsm["emb"] = np.zeros((a.n_obs, 4), "float32")
    b.obsm["emb"] = np.zeros((b.n_obs, 4), "float32")
    src = Source([a, b])
    assert [x.shape[1] for x in src.rep("obsm/emb")] == [4, 4]  # equal-width rep → no error


# ── shared factorization ────────────────────────────────────────────────────────────────────────
def test_shared_source_factorized_once():
    # a primary and its matched control name the same source_key + group_by → the Source factorizes it once.
    adata = encoded_adata(("A", "B"), ("control", "d1", "d2"), 8)
    primary, links = perturbation_streams(adata)
    loader = Loader({KEY: adata}, primary=primary, links=links, batch_size=8, chunk_size=1, preload_nchunks=8)
    assert len(loader._sources[KEY]._leaf_cache) == 1  # one (group_by) entry despite two streams
