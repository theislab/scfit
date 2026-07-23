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
from scheme_helpers import DRUG, DRUGS, IS_CONTROL, LINE, LINES, codes, encoded_adata, perturbation_scheme

from scfit.data import Loader, SamplerConfig

_LINE, _DRUG = codes(LINES), codes(DRUGS)


def _condition_lookup(leaf: tuple) -> dict[str, np.ndarray]:
    return {"condition": np.array([[_LINE[leaf[0]], _DRUG[leaf[1]]]], dtype=np.int64)}


def test_common_bind_target_condition_and_context_coherent():
    loader = Loader(
        perturbation_scheme(encoded_adata(LINES, DRUGS, 16)),
        SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8),
        condition_lookup=_condition_lookup,
    )
    it = iter(loader)
    for _ in range(12):
        batch = next(it)
        tgt = np.asarray(batch["target"])
        src = np.asarray(batch["source"])
        cond = np.asarray(batch["condition"]["condition"])

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
    x = np.asarray(batch["target_reps"]["X"])
    rep = np.asarray(batch["target_reps"]["obsm/rep"])
    np.testing.assert_array_equal(x, rep)  # aligned reps must be the same cells


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
    # batch). SamplerConfig also carries the user-set `to` / `preload_to_gpu` (exercised here).
    scheme = perturbation_scheme(encoded_adata(LINES, DRUGS, 16), ctrl_in_memory=True)
    cfg = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8, to="torch", preload_to_gpu=False)
    dl = Loader(scheme, cfg, condition_lookup=_condition_lookup)
    assert isinstance(dl._nodes["ctrl"], ad.AnnData)  # ctrl node materialized into RAM
    batch = next(iter(dl))
    assert (np.asarray(batch["source"])[:, IS_CONTROL] == 1.0).all()  # source = matched control cells (from RAM)
