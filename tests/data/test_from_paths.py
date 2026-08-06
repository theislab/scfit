"""Path sources — ``Loader.from_paths`` opens zarr ``{source_key: path | [paths]}`` backed.

Covers the source-value shapes (single path / list of paths / already-constructed AnnData), the "load only
the reps the streams use" contract (via ``load_backed_adata``), that an annbatch collection root is
rejected (unsupported), and that a path-sourced loader streams matched batches end-to-end.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
from annbatch import DatasetCollection
from scheme_helpers import KEY, perturbation_obs, uniform, write_zarr

from scfit.data import Loader, Stream
from scfit.data._io import load_backed_adata

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2")
COLS = ("cell_line", "drug")
READ = {"batch_size": 16, "chunk_size": 16, "preload_nchunks": 1, "to": None}


def _adata(n_per_combo: int = 16, seed: int = 0) -> ad.AnnData:
    """Sparse X + an obsm rep + a layer + an unreferenced obs column (open must load only what's used)."""
    rng = np.random.default_rng(seed)
    obs = perturbation_obs(LINES, DRUGS, n_per_combo)
    obs["extra"] = "unused"  # an obs column no stream references — must not be required
    a = ad.AnnData(X=sp.csr_matrix(rng.random((len(obs), 5), dtype="float32")), obs=obs)
    a.obsm["emb"] = rng.random((len(obs), 3), dtype="float32")
    a.layers["log1p"] = sp.csr_matrix(rng.random((len(obs), 5), dtype="float32"))
    return a


def _streams():
    combos = {(cl, dr) for cl in LINES for dr in DRUGS}
    pert = uniform([c for c in combos if c[1] != "control"])
    ctrl = uniform([c for c in combos if c[1] == "control"])
    return (
        Stream(KEY, group_by=COLS, reps="X", weights=pert),
        {"ctrl": Stream(KEY, group_by=COLS, reps=("X", "obsm/emb"), match_on=("cell_line",), weights=ctrl)},
    )


def _loader(path_value):
    primary, links = _streams()
    return Loader.from_paths({KEY: path_value}, primary=primary, links=links, seed=0, **READ)


def _open_group(path):
    import zarr

    return zarr.open_group(path, mode="r")


# ── load_backed_adata: only the requested reps/cols are materialized ───────────────────────────
def test_load_backed_adata_reads_only_requested_keys(tmp_path):
    g = _open_group(write_zarr(_adata(), tmp_path / "a.zarr"))
    backed = load_backed_adata(g, keys=("X", "obsm/emb"), cols=COLS)
    assert isinstance(backed.X, ad.abc.CSRDataset)  # sparse rep stays backed (not read into RAM)
    assert list(backed.obsm) == ["emb"]  # obsm/emb requested → present
    assert "log1p" not in backed.layers  # layers/log1p not requested → never touched
    assert list(backed.obs.columns) == list(COLS)  # obs reduced to the requested cols
    np.testing.assert_array_equal(backed.obsm["emb"].shape, (backed.n_obs, 3))


def test_load_backed_adata_layers_and_no_x(tmp_path):
    g = _open_group(write_zarr(_adata(), tmp_path / "a.zarr"))
    backed = load_backed_adata(g, keys=("layers/log1p",), cols=("cell_line",))
    assert backed.X is None  # X not requested
    assert "log1p" in backed.layers  # requested layer present
    assert list(backed.obs.columns) == ["cell_line"]


# ── Loader.from_paths: source-value shapes ─────────────────────────────────────────────────────
def test_single_path_source_opened_backed_and_streams(tmp_path):
    p = write_zarr(_adata(), tmp_path / "a.zarr")
    loader = _loader(p)
    a = loader._sources[KEY].adatas[0]
    assert isinstance(a.X, ad.abc.CSRDataset)  # backed, not in-memory
    assert list(a.obsm) == ["emb"]  # only the reps the streams use (union: X + obsm/emb)
    assert "log1p" not in a.layers
    b = next(iter(loader))
    assert b["primary"]["X"].shape == (16, 5)
    assert b["ctrl"]["X"].shape == (16, 5)  # matched control rows
    assert b["ctrl"]["obsm/emb"].shape == (16, 3)  # aligned obsm rep of the same cells


def test_list_of_paths_source_streams(tmp_path):
    paths = [write_zarr(_adata(seed=i), tmp_path / f"a{i}.zarr") for i in range(2)]
    loader = _loader(paths)
    adatas = loader._sources[KEY].adatas
    assert len(adatas) == 2 and all(isinstance(a, ad.AnnData) for a in adatas)
    assert next(iter(loader))["primary"]["X"].shape == (16, 5)


def test_in_memory_source_passes_through():
    # an already-constructed container is returned unchanged (not re-opened). Tested at the open_source
    # boundary so it doesn't depend on annbatch's in-memory-sparse path (which needs the numba extra).
    from scfit.data._io import open_source

    adata = _adata()
    assert open_source(adata, keys=("X",), cols=COLS) is adata


def test_collection_root_is_unsupported(tmp_path):
    ap = write_zarr(_adata(), tmp_path / "a.zarr")
    cp = str(tmp_path / "coll.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DatasetCollection(cp, mode="a").add_adatas([ap], groupby=list(COLS), shuffle=False)
    with pytest.raises(NotImplementedError, match="DatasetCollection"):
        _loader(cp)
