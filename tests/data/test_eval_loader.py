"""``EvalLoader``: control-rooted eval reader (Sequential control inner + BoundClassSampler target).

For each control population (context, e.g. ``cell_line``) the source is **all** its control cells (read in
full via the Sequential inner); the target is a matched perturbed batch (annbatch samples a drug within
the context). The condition is the perturbed leaf the target drew. Cells carry a unique index in ``X`` so
we can assert the source is exactly the context's controls and the target is that context's perturbed cells.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("annbatch")

from scheme_helpers import (
    DRUG,
    IS_CONTROL,
    LINE,
    LINES,
    ROW_ID,
    codes,
    encoded_adata,
    perturbation_scheme,
    row_ids,
    write_zarr,
)

from scfit.data import Bind, EvalLoader, Loader, Node, SamplerConfig, Scheme, split_scheme, uniform

DRUGS = ("control", "d1", "d2", "d3")
_LINE, _DRUG = codes(LINES), codes(DRUGS)
_CFG = SamplerConfig(batch_size=4, chunk_size=1, preload_nchunks=4)


def _adata(n_per_combo: int = 8):
    return encoded_adata(LINES, DRUGS, n_per_combo)


def _condition_lookup(leaf: tuple) -> dict[str, np.ndarray]:
    return {"drug": np.array([[_LINE[leaf[0]], _DRUG[leaf[1]]]], dtype=float)}


def test_source_is_all_controls_of_context_target_matches_condition():
    adata = _adata()
    obs = adata.obs
    loader = EvalLoader(perturbation_scheme(adata), _CFG, _condition_lookup)
    assert set(loader.control_populations) == {("A", "control"), ("B", "control")}

    seen_ctx = []
    for out in loader.iter_conditions():  # one batch per control population
        cl, dr = out["leaf"]
        seen_ctx.append(cl)
        assert dr != "control"  # target is a perturbed condition
        # source = ALL control cells of this cell line (read in full)
        src_truth = set(np.flatnonzero((obs["cell_line"] == cl).to_numpy() & obs["control"].to_numpy()).tolist())
        assert row_ids(out["source"]) == src_truth
        # target cells are perturbed cells of this (cell_line, drug)
        rows = np.asarray(out["target"])
        assert np.all(rows[:, LINE].astype(int) == _LINE[cl])  # same cell line
        assert np.all(rows[:, DRUG].astype(int) == _DRUG[dr])  # the drawn drug
        assert np.all(rows[:, 2] == 0.0)  # not control
        # condition embedding is the drawn perturbed leaf
        np.testing.assert_array_equal(out["condition"]["drug"], [[_LINE[cl], _DRUG[dr]]])
        assert out["condition"]["drug"].dtype == np.float64

    assert set(seen_ctx) == set(LINES)  # each control population once


def test_n_conditions_cycles_control_populations():
    loader = EvalLoader(perturbation_scheme(_adata()), _CFG, _condition_lookup)
    outs = list(loader.iter_conditions(n_conditions=5))
    assert len(outs) == 5  # 2 control populations cycled to 5 batches
    for out in outs:
        assert out["leaf"][1] != "control"


def test_deterministic_across_calls():
    loader = EvalLoader(perturbation_scheme(_adata()), _CFG, _condition_lookup)
    a = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
    b = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
    assert a == b  # same seed ⇒ same drawn conditions each call


def test_train_and_eval_loaders_share_condition_lookup_contract():
    scheme = perturbation_scheme(_adata())

    def lookup(leaf: tuple) -> dict[str, np.ndarray]:
        return {
            "category": np.array([[_DRUG[leaf[1]]]], dtype=np.int32),
            "feature": np.array([[[_LINE[leaf[0]], 1.5]]], dtype=np.float64),
        }

    train_condition = next(iter(Loader(scheme, _CFG, condition_lookup=lookup)))["condition"]
    eval_condition = next(EvalLoader(scheme, _CFG, condition_lookup=lookup).iter_conditions())["condition"]

    assert train_condition["category"].shape == eval_condition["category"].shape == (1, 1)
    assert train_condition["feature"].shape == eval_condition["feature"].shape == (1, 1, 2)
    assert train_condition["category"].dtype == eval_condition["category"].dtype == np.int32
    assert train_condition["feature"].dtype == eval_condition["feature"].dtype == np.float64


def test_reps_aligned_same_cells():
    # two aligned reps of the target (X and an obsm copy) — same sampled rows → identical values.
    adata = encoded_adata(LINES, DRUGS, obsm_rep=True)
    scheme = perturbation_scheme(adata, key=("X", "obsm/rep"), ctrl_key="X")
    out = next(EvalLoader(scheme, _CFG, _condition_lookup).iter_conditions())
    np.testing.assert_array_equal(np.asarray(out["target_reps"]["X"]), np.asarray(out["target_reps"]["obsm/rep"]))


# ── held-out split regression (split_scheme + EvalLoader) ──────────────────────────────────────────
#
# split_scheme restricts only the ROOT (target) node's weights to a split; the bound CHILD (control)
# node's weights are carried through UNCHANGED (see test_split.py::test_controls_carried_through_
# unchanged) -- by design, so match_context columns outside split_by stay fully available. But when
# split_by *is* the bind's own context (as here: cell_line), a `val` scheme's control side still spans
# every cell line the FULL pre-split scheme had, not just the held-out one(s). EvalLoader must not
# schedule a control population whose context has zero matching target leaves in ITS OWN scheme, or
# annbatch's BoundClassSampler raises "... has no drawable run of at least chunk_size ..." on the first
# such (training-only) context -- deterministically, regardless of chunk_size/min_runs_per_leaf.

_HELD_OUT_LINES = ("A", "B", "C")
_HELD_OUT_LINE = codes(_HELD_OUT_LINES)


def _held_out_split():
    """3 cell lines -> split_scheme(split_by=["cell_line"]) holds exactly 1 out to `val`."""
    adata = encoded_adata(_HELD_OUT_LINES, DRUGS)
    scheme = perturbation_scheme(adata)
    splits = split_scheme(scheme, split_by=["cell_line"], ratios={"train": 2 / 3, "val": 1 / 3}, random_state=0)
    return adata, splits


class TestEvalLoaderOnHeldOutSplit:
    def test_control_populations_restricted_to_held_out_context(self):
        """The regression itself: control_populations must match the held-out line(s) exactly, not
        every line with positive (unrestricted) control weight."""
        _adata, splits = _held_out_split()
        val_scheme, train_scheme = splits["val"], splits["train"]

        # sanity-check split_scheme's own documented contract still holds: ctrl weights ARE identical
        # across splits (unrestricted) -- the fix lives in EvalLoader, not in split_scheme.
        assert val_scheme.nodes["ctrl"].weights == train_scheme.nodes["ctrl"].weights

        val_lines = {c[0] for c, w in val_scheme.nodes["pert"].weights.items() if w > 0}
        assert len(val_lines) == 1
        (held_out_line,) = val_lines

        loader = EvalLoader(val_scheme, _CFG, condition_lookup=None)
        assert loader.control_populations == [(held_out_line, "control")]

    def test_no_mixed_batches_and_schedule_is_1to1(self):
        """Every batch is class-coherent (no mixing across cell lines / drugs / control-vs-target),
        and each schedule position's source and target refer to the SAME (context, leaf) -- no drift
        onto a training-only context that would have zero matching target leaves here."""
        adata, splits = _held_out_split()
        val_scheme = splits["val"]
        obs = adata.obs
        val_leaves = {c for c, w in val_scheme.nodes["pert"].weights.items() if w > 0}
        (held_out_line,) = {c[0] for c in val_leaves}

        loader = EvalLoader(val_scheme, _CFG, condition_lookup=None)
        for out in loader.iter_conditions(n_conditions=8):  # cycle several times over the one context
            cl, dr = out["leaf"]
            assert cl == held_out_line
            assert (cl, dr) in val_leaves

            # source: EXACTLY this context's controls -- no other cell line's cells mixed in.
            src_truth = set(np.flatnonzero((obs["cell_line"] == cl).to_numpy() & obs["control"].to_numpy()).tolist())
            assert row_ids(out["source"]) == src_truth

            # target: EXACTLY this (cell_line, drug) leaf's cells -- one class per batch, no blending.
            rows = np.asarray(out["target"])
            assert np.all(rows[:, LINE].astype(int) == _HELD_OUT_LINE[cl])
            assert np.all(rows[:, DRUG].astype(int) == _DRUG[dr])
            assert np.all(rows[:, 2] == 0.0)

    def test_deterministic_across_calls_on_held_out_split(self):
        _adata, splits = _held_out_split()
        loader = EvalLoader(splits["val"], _CFG, condition_lookup=None)
        a = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
        b = [out["leaf"] for out in loader.iter_conditions(n_conditions=6)]
        assert a == b


# ── multi-store reads (regression for scverse/annbatch#256) ─────────────────────────────────────────
#
# Over a source spanning several separately-added stores, the loader's request->buffer index map was a
# gather instead of the inverse-permutation scatter, so once a class-coherent batch's chunks regrouped
# across stores the perturbed TARGET was read from the WRONG cell lines (a cross-context mix) while the
# SOURCE stayed correct -- every held-out condition then scored one cell line's controls against a mixed
# target (catastrophically negative r_squared at real multi-plate scale). Single in-memory adata (the
# tests above) never regroups across stores, so this only surfaces with >=2 stores.

_MS_LINES = ("A", "B", "C", "D")
_MS_DRUGS = ("control", "d1", "d2")
_ML, _MD = codes(_MS_LINES), codes(_MS_DRUGS)
_MS_CFG = SamplerConfig(batch_size=12, chunk_size=4, preload_nchunks=8 * (12 // 4), to=None)


def _write_store(path, *, lines, drugs, sid, n_per=24, base=0):
    """A grouped store over ``lines`` × ``drugs``, X identity-encoded with the shared _ML/_MD vocabulary
    and global row ids offset by ``base`` (so ids stay unique across stores)."""
    adata = encoded_adata(lines, drugs, n_per, line=_ML, drug=_MD, row_id_start=base, index_prefix=f"s{sid}_")
    return write_zarr(adata, path), adata.n_obs


def test_multistore_source_and_target_stay_context_coherent(tmp_path):
    """Over a two-store source, EACH condition's source and target must be exactly the leaf's cell line
    (pre-fix the target was a cross-line mix). Uses chunk_size>1 so batches regroup chunks across stores."""
    p0, n0 = _write_store(tmp_path / "s0.zarr", lines=_MS_LINES, drugs=_MS_DRUGS, sid=0, base=0)
    p1, _ = _write_store(tmp_path / "s1.zarr", lines=_MS_LINES, drugs=_MS_DRUGS, sid=1, base=n0)

    cols = ("cell_line", "drug")
    combos = {(cl, dr) for cl in _MS_LINES for dr in _MS_DRUGS}
    nodes = {
        "pert": Node("data", cols, "X", uniform([c for c in combos if c[1] != "control"])),
        "ctrl": Node("data", cols, "X", uniform([c for c in combos if c[1] == "control"])),
    }
    scheme = Scheme.from_paths(
        sources={"data": [p0, p1]}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),)
    )
    loader = EvalLoader(scheme, _MS_CFG, condition_lookup=None, seed=0)

    n = 0
    for out in loader.iter_conditions(n_conditions=12):  # cycle over all four held-out lines several times
        cl, dr = out["leaf"]
        src, tgt = np.asarray(out["source"]), np.asarray(out["target"])
        assert set(src[:, LINE].astype(int).tolist()) == {_ML[cl]}, "source must be exactly the leaf's cell line"
        assert np.all(src[:, 2] == 1.0), "source must be controls"
        assert set(tgt[:, LINE].astype(int).tolist()) == {_ML[cl]}, "target must be exactly the leaf's cell line"
        assert set(tgt[:, DRUG].astype(int).tolist()) == {_MD[dr]}, "target must be exactly the drawn drug"
        assert np.all(tgt[:, 2] == 0.0), "target must be perturbed"
        n += 1
    assert n == 12


def test_control_population_merges_same_context_across_all_stores(tmp_path):
    """Every store carries all of cell lines A/B/C, yet the bind still merges like-to-like: each condition's
    source and target are the SAME line (A↔A, B↔B, C↔C), and a context's control *population* is the union
    of that line's controls across EVERY store -- the "merge a→a where everyone has abc" case. Row ids are
    offset per store (base = 1000·i) so we can assert the merged source spans all three stores."""
    lines = ("A", "B", "C")
    paths = [
        _write_store(tmp_path / f"s{i}.zarr", lines=lines, drugs=_MS_DRUGS, sid=i, base=1000 * i)[0] for i in range(3)
    ]
    cols = ("cell_line", "drug")
    combos = {(cl, dr) for cl in lines for dr in _MS_DRUGS}
    nodes = {
        "pert": Node("data", cols, "X", uniform([c for c in combos if c[1] != "control"])),
        "ctrl": Node("data", cols, "X", uniform([c for c in combos if c[1] == "control"])),
    }
    scheme = Scheme.from_paths(
        sources={"data": paths}, nodes=nodes, root="pert", seed=0, binds=(Bind("pert", "ctrl", ("cell_line",)),)
    )
    loader = EvalLoader(scheme, _MS_CFG, condition_lookup=None, seed=0)
    assert set(loader.control_populations) == {(cl, "control") for cl in lines}

    seen = set()
    for out in loader.iter_conditions():  # one batch per control population (A, B, C)
        cl, _dr = out["leaf"]
        seen.add(cl)
        src, tgt = np.asarray(out["source"]), np.asarray(out["target"])
        # like-to-like: source (control) and target (perturbed) are the SAME cell line
        assert set(src[:, LINE].astype(int).tolist()) == {_ML[cl]}, f"source not all line {cl}"
        assert set(tgt[:, LINE].astype(int).tolist()) == {_ML[cl]}, f"target not all line {cl}"
        assert np.all(src[:, IS_CONTROL] == 1.0) and np.all(tgt[:, IS_CONTROL] == 0.0)
        # the control population is MERGED across every store: its row ids fall in all three id ranges
        ids = row_ids(out["source"])
        hit = [any(1000 * i <= r < 1000 * i + 1000 for r in ids) for i in range(3)]
        assert all(hit), f"context {cl!r} controls not merged across all stores: {sorted(ids)}"
    assert seen == set(lines)


def test_separate_control_source_matched_by_context(tmp_path):
    """Controls may live in a SEPARATE store from the targets (cellflow's ``control_path``): the ctrl node
    reads its own source and is matched to targets purely by the bind's context (cell_line). Targets span
    two stores to keep the multi-store read path (annbatch#256) covered."""
    # targets: perturbed-only, two stores; controls: a separate consolidated store, all cell lines
    t0, m0 = _write_store(tmp_path / "t0.zarr", lines=_MS_LINES, drugs=("d1", "d2"), sid=0, base=0)
    t1, _ = _write_store(tmp_path / "t1.zarr", lines=_MS_LINES, drugs=("d1", "d2"), sid=1, base=m0)
    cpath, _ = _write_store(tmp_path / "ctrl.zarr", lines=_MS_LINES, drugs=("control",), sid=9, base=10_000)

    pert = Node("data", ("cell_line", "drug"), "X", uniform([(cl, dr) for cl in _MS_LINES for dr in ("d1", "d2")]))
    ctrl = Node("control", ("cell_line",), "X", uniform([(cl,) for cl in _MS_LINES]))
    scheme = Scheme.from_paths(
        sources={"data": [t0, t1], "control": cpath},
        nodes={"pert": pert, "ctrl": ctrl},
        root="pert",
        seed=0,
        binds=(Bind("pert", "ctrl", ("cell_line",)),),
    )
    loader = EvalLoader(scheme, _MS_CFG, condition_lookup=None, seed=0)
    assert set(loader.control_populations) == {(cl,) for cl in _MS_LINES}

    n = 0
    for out in loader.iter_conditions(n_conditions=8):
        cl, dr = out["leaf"]
        src, tgt = np.asarray(out["source"]), np.asarray(out["target"])
        # source came from the SEPARATE control store: right cell line, all controls
        assert set(src[:, LINE].astype(int).tolist()) == {_ML[cl]}
        assert np.all(src[:, 2] == 1.0) and np.all(src[:, ROW_ID] >= 10_000), "source drawn from the control store"
        # target came from the data stores: right cell line + drug, perturbed
        assert set(tgt[:, LINE].astype(int).tolist()) == {_ML[cl]}
        assert set(tgt[:, DRUG].astype(int).tolist()) == {_MD[dr]}
        assert np.all(tgt[:, 2] == 0.0) and np.all(tgt[:, ROW_ID] < 10_000), "target drawn from the data stores"
        n += 1
    assert n == 8
