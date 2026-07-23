"""Condition-coherence of the BoundClassSampler-based Loader — the guarantee the old suite missed.

Each cell's ``(cell_line, drug, control)`` is encoded into ``X`` so every yielded row can be decoded and
checked: the target batch is one perturbed condition, the ``condition`` vector matches that condition,
and the source batch is control cells of the *matched* context (``common=`` → same cell line). This is
what silently broke when binded's rng-wrapping scheduler met annbatch's refactored class draw.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("annbatch")

import anndata as ad
from scheme_helpers import (
    CONTROL,
    DRUG,
    DRUGS,
    IS_CONTROL,
    LINE,
    LINES,
    ROW_ID,
    codes,
    encoded_adata,
    perturbation_scheme,
)

from scfit.data import Bind, Loader, Node, SamplerConfig, Scheme
from scfit.data._schema import uniform

_LINE, _DRUG = codes(LINES), codes(DRUGS)


def _annotations() -> dict:
    # per-node ``{node: {leaf: {realm: array}}}``: annotate the root "pert" node's perturbed leaves.
    return {
        "pert": {
            (cl, dr): {"condition": np.array([[_LINE[cl], _DRUG[dr]]], dtype=np.int64)}
            for cl in LINES
            for dr in DRUGS
            if dr != CONTROL
        }
    }


def test_common_bind_target_condition_and_context_coherent():
    loader = Loader(
        perturbation_scheme(encoded_adata(LINES, DRUGS, 16)),
        SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),
        annotations=_annotations(),
    )
    it = iter(loader)
    for _ in range(12):
        batch = next(it)
        tgt = np.asarray(batch["pert"]["X"])  # target = root node "pert", rep "X" (batch keyed by node name)
        src = np.asarray(batch["ctrl"]["X"])  # source = bound child "ctrl", rep "X"
        cond = np.asarray(batch["annotations"]["pert"]["condition"])

        # target batch is one perturbed condition
        assert len(np.unique(tgt[:, LINE])) == 1, "target batch mixes cell lines"
        assert len(np.unique(tgt[:, DRUG])) == 1, "target batch mixes drugs"
        assert tgt[0, IS_CONTROL] == 0.0, "target must be perturbed (not control)"

        # condition vector matches that exact (cell_line, drug)
        assert np.all(cond[:, 0] == tgt[0, LINE]) and np.all(cond[:, 1] == tgt[0, DRUG]), "condition ≠ target"

        # source batch is control cells of the SAME context (cell line)
        assert np.all(src[:, IS_CONTROL] == 1.0), "source must be control cells"
        assert len(np.unique(src[:, LINE])) == 1 and src[0, LINE] == tgt[0, LINE], "source context ≠ target context"


def test_reps_are_aligned_same_cells():
    # two aligned reps of the target: X and an obsm copy of X. Same sampled rows → identical values.
    scheme = perturbation_scheme(encoded_adata(LINES, DRUGS, 16, obsm_rep=True), key=("X", "obsm/rep"), ctrl_key="X")
    loader = Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    batch = next(iter(loader))
    x = np.asarray(batch["pert"]["X"])
    rep = np.asarray(batch["pert"]["obsm/rep"])
    np.testing.assert_array_equal(x, rep)  # aligned reps must be the same cells


def test_reps_from_distinct_stores_are_the_same_cells():
    # The robust way to test rep alignment: make the second rep a *distinct* store (not a copy of X) that
    # still carries each cell's unique ROW_ID, then check the ids line up PER POSITION across a whole pass.
    # That proves the "obsm/rep" loader reads its own array AND draws the same cells, in the same order, as
    # the "X" loader — the guarantee the per-rep deepcopy'd samplers provide; a desync would break it.
    adata = encoded_adata(LINES, DRUGS, 16)
    rid = adata.X[:, ROW_ID]
    adata.obsm["rep"] = np.stack([rid, -rid], axis=1).astype("float32")  # distinct payload, id recoverable
    scheme = perturbation_scheme(adata, key=("X", "obsm/rep"), ctrl_key="X")
    loader = Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    it = iter(loader)
    for _ in range(loader._n_batches):
        reps = next(it)["pert"]  # the root node's aligned reps {loc: rows}
        x, rep = np.asarray(reps["X"]), np.asarray(reps["obsm/rep"])
        np.testing.assert_array_equal(rep[:, 0], x[:, ROW_ID])  # same cell per position (row-for-row)
        np.testing.assert_array_equal(rep[:, 1], -x[:, ROW_ID])  # and it really is obsm/rep, not X again


def test_source_reps_are_aligned_same_cells():
    # Aligned reps apply to the bound control (source) node too — its reps read the same control cells.
    scheme = perturbation_scheme(encoded_adata(LINES, DRUGS, 16, obsm_rep=True), key="X", ctrl_key=("X", "obsm/rep"))
    loader = Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    reps = next(iter(loader))["ctrl"]  # the bound child's aligned reps {loc: rows}
    np.testing.assert_array_equal(np.asarray(reps["X"])[:, ROW_ID], np.asarray(reps["obsm/rep"])[:, ROW_ID])


def test_more_than_two_reps_all_aligned():
    # deepcopy-per-rep generalizes beyond two: three reps of the target all read the same cells.
    adata = encoded_adata(LINES, DRUGS, 16, obsm_rep=True)  # adds obsm["rep"]
    adata.obsm["rep2"] = adata.X.copy()
    scheme = perturbation_scheme(adata, key=("X", "obsm/rep", "obsm/rep2"), ctrl_key="X")
    loader = Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))
    reps = next(iter(loader))["pert"]  # the root node's aligned reps {loc: rows}
    x = np.asarray(reps["X"])[:, ROW_ID]
    np.testing.assert_array_equal(np.asarray(reps["obsm/rep"])[:, ROW_ID], x)
    np.testing.assert_array_equal(np.asarray(reps["obsm/rep2"])[:, ROW_ID], x)


def test_multiple_bound_sources_are_keyed_by_child_name():
    # Two children bound to the root on DIFFERENT columns must both appear in the batch, keyed by their
    # own node names — they used to clobber a single `out["source"]`. Each is matched on its own bind column.
    import pandas as pd

    obs = pd.DataFrame(
        [(ln, dr) for ln in ("A", "B") for dr in ("d1", "d2") for _ in range(24)], columns=["line", "drug"]
    )
    lc, dc = codes(("A", "B")), codes(("d1", "d2"))
    x = np.stack([obs["line"].map(lc), obs["drug"].map(dc)], axis=1).astype("float32")  # [line code, drug code]
    scheme = Scheme(
        sources={"data": ad.AnnData(x, obs=obs)},
        nodes={
            "root": Node(
                "data", ("line", "drug"), "X", uniform([(ln, dr) for ln in ("A", "B") for dr in ("d1", "d2")])
            ),
            "on_line": Node("data", ("line",), "X", uniform([("A",), ("B",)])),
            "on_drug": Node("data", ("drug",), "X", uniform([("d1",), ("d2",)])),
        },
        root="root",
        binds=(Bind("root", "on_line", ("line",)), Bind("root", "on_drug", ("drug",))),
        seed=0,
    )
    batch = next(iter(Loader(scheme, SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8))))
    assert set(batch) == {"root", "on_line", "on_drug"}  # every node present, keyed by name (no clobber)
    tgt = np.asarray(batch["root"]["X"])
    assert np.all(np.asarray(batch["on_line"]["X"])[:, 0] == tgt[0, 0])  # on_line matched on the line column
    assert np.all(np.asarray(batch["on_drug"]["X"])[:, 1] == tgt[0, 1])  # on_drug matched on the drug column


def test_materialize_node_selects_positive_weight_rows():
    # `materialize_node` reads only a node's positive-weight (here: control) cells into an in-memory AnnData
    from scfit.data._io import leaf_codes, materialize_node

    adata = encoded_adata(LINES, DRUGS, 16)
    cols = ("cell_line", "drug")
    ctrl_node = perturbation_scheme(adata).nodes["ctrl"]
    mem, sub_codes, sub_leaves = materialize_node(adata, ctrl_node)
    assert isinstance(mem, ad.AnnData)
    assert (mem.obs["drug"] == "control").all()  # only positive-weight (control) rows materialized
    assert mem.n_obs == int((adata.obs["drug"] == "control").sum())
    assert (np.asarray(mem.X)[:, IS_CONTROL] == 1.0).all()  # X[:, IS_CONTROL] encodes is_control
    # returned factorization matches a direct factorize of the materialized subset
    rc, rl = leaf_codes(mem.obs[list(cols)], cols)
    assert sub_leaves == rl
    assert np.array_equal(sub_codes, rc)


def test_in_memory_node_materialized():
    # Node.in_memory → the loader materializes that node into RAM (served from memory, not re-read each
    # batch). Per-node `to` and the global `preload_to_gpu` Loader arg are exercised here.
    scheme = perturbation_scheme(encoded_adata(LINES, DRUGS, 16), ctrl_in_memory=True)
    cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8, to="torch")
    dl = Loader(scheme, cfg, annotations=_annotations(), preload_to_gpu=False)
    assert isinstance(dl._nodes["ctrl"], ad.AnnData)  # ctrl node materialized into RAM
    batch = next(iter(dl))
    assert (np.asarray(batch["ctrl"]["X"])[:, IS_CONTROL] == 1.0).all()  # source = bound child "ctrl", rep "X"
