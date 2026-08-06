"""Coherence of the matched Stream/Loader: each yielded row decodes to the label it must belong to.

Each cell's ``(cell_line, drug, control)`` is encoded into ``X`` so every row can be decoded: the primary
batch is one perturbed condition, ``batch["labels"]["primary"]`` matches it, and the ``ctrl`` batch is
control cells of the *matched* context (``match_on`` → same cell line). Also: aligned reps read the same
cells, multiple links stay keyed by name, ``match_on=()`` decouples, and per-stream batch sizes differ.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scheme_helpers import (
    CONTROL,
    DRUGS,
    IS_CONTROL,
    KEY,
    LINES,
    ROW_ID,
    codes,
    encoded_adata,
    only_leaf,
    perturbation_labels,
    perturbation_loader,
    perturbation_streams,
    rep,
    uniform,
)

from scfit.data import Loader, Stream
from scfit.data._source import Source

COLS = ("cell_line", "drug")


def test_primary_label_and_context_coherent():
    loader = perturbation_loader(encoded_adata(LINES, DRUGS, 16), label_lookup=perturbation_labels())
    it = iter(loader)
    for _ in range(12):
        b = next(it)
        tgt, src = rep(b, "primary"), rep(b, "ctrl")
        line, drug = only_leaf(tgt)  # primary batch is one perturbed condition
        assert tgt[0, IS_CONTROL] == 0.0, "primary must be perturbed (not control)"
        cond = np.asarray(b["labels"]["primary"]["condition"])
        assert cond[0, 0] == line and cond[0, 1] == drug, "label ≠ primary condition"
        assert (src[:, IS_CONTROL] == 1.0).all(), "ctrl must be control cells"
        assert only_leaf(src)[0] == line, "ctrl context (cell line) ≠ primary context"


def test_reps_from_distinct_stores_are_the_same_cells():
    # obsm/rep is a DISTINCT payload (not a copy of X) still carrying each cell's ROW_ID, so we can prove
    # the "obsm/rep" loader reads its own array AND draws the same cells, in the same order, as "X".
    adata = encoded_adata(LINES, DRUGS, 16)
    rid = adata.X[:, ROW_ID]
    adata.obsm["rep"] = np.stack([rid, -rid], axis=1).astype("float32")
    loader = perturbation_loader(adata, rep=("X", "obsm/rep"), ctrl_rep="X")
    for _ in range(loader.epoch_len):
        reps = next(iter(loader))["primary"]
        x, r = np.asarray(reps["X"]), np.asarray(reps["obsm/rep"])
        np.testing.assert_array_equal(r[:, 0], x[:, ROW_ID])  # same cell per position (row-for-row)
        np.testing.assert_array_equal(r[:, 1], -x[:, ROW_ID])  # and it really is obsm/rep, not X again


def test_source_reps_are_aligned_same_cells():
    # aligned reps apply to the control link too — its reps read the same control cells.
    loader = perturbation_loader(encoded_adata(LINES, DRUGS, 16, obsm_rep=True), rep="X", ctrl_rep=("X", "obsm/rep"))
    reps = next(iter(loader))["ctrl"]
    np.testing.assert_array_equal(np.asarray(reps["X"])[:, ROW_ID], np.asarray(reps["obsm/rep"])[:, ROW_ID])


def test_more_than_two_reps_all_aligned():
    adata = encoded_adata(LINES, DRUGS, 16, obsm_rep=True)  # adds obsm["rep"]
    adata.obsm["rep2"] = adata.X.copy()
    loader = perturbation_loader(adata, rep=("X", "obsm/rep", "obsm/rep2"), ctrl_rep="X")
    reps = next(iter(loader))["primary"]
    x = np.asarray(reps["X"])[:, ROW_ID]
    np.testing.assert_array_equal(np.asarray(reps["obsm/rep"])[:, ROW_ID], x)
    np.testing.assert_array_equal(np.asarray(reps["obsm/rep2"])[:, ROW_ID], x)


def test_multiple_links_are_keyed_by_name():
    # Two links matched on DIFFERENT columns both appear in the batch, keyed by their own names.
    obs = pd.DataFrame(
        [(ln, dr) for ln in ("A", "B") for dr in ("d1", "d2") for _ in range(24)], columns=["line", "drug"]
    )
    lc, dc = codes(("A", "B")), codes(("d1", "d2"))
    x = np.stack([obs["line"].map(lc), obs["drug"].map(dc)], axis=1).astype("float32")  # [line code, drug code]
    adata = ad.AnnData(x, obs=obs)
    loader = Loader(
        {KEY: adata},
        primary=Stream(
            KEY, group_by=("line", "drug"), weights=uniform([(ln, dr) for ln in ("A", "B") for dr in ("d1", "d2")])
        ),
        links={
            "on_line": Stream(KEY, group_by=("line",), match_on=("line",), weights=uniform([("A",), ("B",)])),
            "on_drug": Stream(KEY, group_by=("drug",), match_on=("drug",), weights=uniform([("d1",), ("d2",)])),
        },
        batch_size=8,
        chunk_size=1,
        preload_nchunks=8,
        seed=0,
    )
    b = next(iter(loader))
    assert set(b) == {"primary", "on_line", "on_drug"}  # every stream present, keyed by name (no clobber)
    tgt = rep(b, "primary")
    assert (rep(b, "on_line")[:, 0] == tgt[0, 0]).all()  # on_line matched on the line column
    assert (rep(b, "on_drug")[:, 1] == tgt[0, 1]).all()  # on_drug matched on the drug column


def test_match_on_empty_is_unconditional():
    # match_on=() ⇒ the control is drawn independently, so over many batches its cell line decouples from
    # the primary's — while still being a control group.
    loader = perturbation_loader(encoded_adata(LINES, DRUGS, 16), ctrl_match_on=())
    pairs = set()
    for _ in range(40):
        b = next(iter(loader))
        assert (rep(b, "ctrl")[:, IS_CONTROL] == 1.0).all(), "ctrl must still be a control group"
        pairs.add((only_leaf(rep(b, "primary"))[0], only_leaf(rep(b, "ctrl"))[0]))
    assert any(t != c for t, c in pairs), "match_on=() must let the control decouple from the primary cell line"


def test_per_stream_batch_sizes_differ():
    # the primary and its control can emit different row counts (source rows need not equal target rows).
    adata = encoded_adata(LINES, DRUGS, 16)
    pert = uniform([(cl, dr) for cl in LINES for dr in DRUGS if dr != CONTROL])
    ctrl = uniform([(cl, CONTROL) for cl in LINES])
    loader = Loader(
        {KEY: adata},
        primary=Stream(KEY, group_by=COLS, weights=pert),
        links={
            "ctrl": Stream(
                KEY,
                group_by=COLS,
                match_on=("cell_line",),
                weights=ctrl,
                batch_size=4,
                chunk_size=1,
                preload_nchunks=4,
            )
        },
        batch_size=8,
        chunk_size=1,
        preload_nchunks=8,
        seed=0,
    )
    b = next(iter(loader))
    assert rep(b, "primary").shape[0] == 8 and rep(b, "ctrl").shape[0] == 4


def test_materialize_node_selects_positive_weight_rows():
    from scfit.data._io import leaf_codes, materialize_node

    adata = encoded_adata(LINES, DRUGS, 16)
    _, links = perturbation_streams(adata)
    mem, sub_codes, sub_leaves = materialize_node(adata, links["ctrl"])  # materialize_node takes a Stream now
    assert isinstance(mem, ad.AnnData)
    assert (mem.obs["drug"] == "control").all()  # only positive-weight (control) rows materialized
    assert mem.n_obs == int((adata.obs["drug"] == "control").sum())
    assert (np.asarray(mem.X)[:, IS_CONTROL] == 1.0).all()
    rc, rl = leaf_codes(mem.obs[list(COLS)], COLS)  # returned factorization matches a direct factorize
    assert sub_leaves == rl
    assert np.array_equal(sub_codes, rc)


def test_in_memory_control_materialized():
    loader = perturbation_loader(
        encoded_adata(LINES, DRUGS, 16), ctrl_in_memory=True, label_lookup=perturbation_labels()
    )
    materialized = loader._resolved["ctrl"]  # control materialized into a RAM-backed Source
    assert isinstance(materialized, Source)
    assert isinstance(materialized.adatas[0].X, np.ndarray)  # dense, in-memory (not a backing)
    b = next(iter(loader))
    assert (rep(b, "ctrl")[:, IS_CONTROL] == 1.0).all()  # served from memory, still matched controls
