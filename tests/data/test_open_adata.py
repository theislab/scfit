from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata.acc import A
from scipy import sparse

from scfit.data import open_adata, open_source
from scfit.data._io import is_backed_array


def dense(array) -> np.ndarray:
    """An on-disk or in-memory array as dense numpy."""
    out = array.to_memory() if hasattr(array, "to_memory") else array[...]
    return np.asarray(out.toarray() if sparse.issparse(out) else out)


@pytest.fixture
def adata() -> ad.AnnData:
    rng = np.random.default_rng(0)
    return ad.AnnData(
        X=rng.normal(size=(6, 3)).astype(np.float32),
        obs=pd.DataFrame(
            {"group": pd.Categorical(list("aabbcc")), "value": np.arange(6.0)}, index=[f"c{i}" for i in range(6)]
        ),
        var=pd.DataFrame(index=["m0", "m1", "m2"]),
        obsm={"emb": rng.normal(size=(6, 2))},
        layers={"counts": sparse.random(6, 3, density=0.5, format="csr", random_state=0)},
        uns={"table": "kept", "big_unused": np.ones(100)},
    )


@pytest.fixture(params=[".zarr", ".h5ad"])
def store(request, adata: ad.AnnData, tmp_path: Path) -> Path:
    path = tmp_path / f"data{request.param}"
    adata.write_zarr(path) if request.param == ".zarr" else adata.write_h5ad(path)
    return path


def test_reads_only_the_referenced_parts(store: Path, adata: ad.AnnData):
    opened = open_adata(store, A.X, A.layers["counts"], A.obs["group"], uns=["table"])
    assert list(opened.obs.columns) == ["group"] and list(opened.uns) == ["table"]
    assert "counts" in opened.layers and opened.X is not None and list(opened.obsm) == []
    assert list(opened.var_names) == list(adata.var_names)
    assert list(opened.obs_names) == [str(i) for i in range(6)]  # no index asked for: a RangeIndex
    np.testing.assert_array_equal(opened.obs["group"], adata.obs["group"])
    np.testing.assert_allclose(dense(opened.X), adata.X)
    np.testing.assert_allclose(dense(opened.layers["counts"]), adata.layers["counts"].toarray())


def test_zarr_arrays_stay_on_disk_h5ad_arrays_are_loaded(store: Path):
    opened = open_adata(store, A.X, A.obsm["emb"], A.layers["counts"])
    backed = store.suffix == ".zarr"
    assert all(is_backed_array(a) == backed for a in (opened.X, opened.obsm["emb"], opened.layers["counts"]))


def test_index_on_request(store: Path, adata: ad.AnnData):
    assert list(open_adata(store, A.obs["value"], A.obs.index).obs_names) == list(adata.obs_names)
    assert list(open_adata(store).obs_names) == list(adata.obs_names)  # nothing else sizes it


def test_a_reference_into_an_array_opens_the_array(store: Path, adata: ad.AnnData):
    opened = open_adata(store, A.obsm["emb"][:, 0])
    np.testing.assert_allclose(dense(opened.obsm["emb"]), adata.obsm["emb"])


def test_missing_parts_name_what_the_store_has(store: Path):
    with pytest.raises(KeyError, match=r"obsm\['nope'\] \(has obsm: \['emb'\]\).*obs \['zz'\].*uns \['gone'\]"):
        open_adata(store, A.obsm["nope"], A.obs["zz"], uns=["gone"])


def test_var_references_are_refused(store: Path):
    with pytest.raises(ValueError, match="not var"):
        open_adata(store, A.var["x"])


def test_open_source_reads_h5ad_paths(store: Path, adata: ad.AnnData):
    (opened,) = open_source([store], A.obsm["emb"], A.obs["group"], uns=["table"])
    np.testing.assert_allclose(dense(opened.obsm["emb"]), adata.obsm["emb"])
    assert list(opened.obs.columns) == ["group"] and list(opened.uns) == ["table"]


def test_open_source_names_missing_parts(adata: ad.AnnData, tmp_path: Path):
    adata.write_zarr(tmp_path / "d.zarr")
    with pytest.raises(KeyError, match=r"obsm\['nope'\]"):
        open_source(tmp_path / "d.zarr", A.obsm["nope"])


def test_obs_columns_keep_the_requested_order(store: Path):
    assert list(open_adata(store, A.obs["value"], A.obs["group"]).obs.columns) == ["value", "group"]


def test_the_obs_index_is_not_a_column(store: Path):
    with pytest.raises(KeyError, match="obs"):
        open_adata(store, A.obs["_index"])


def test_skip_leaves_out_what_the_store_lacks(store: Path):
    opened = open_adata(
        store, A.obsm["emb"], A.obsm["nope"], A.obs["group"], A.obs["zz"], uns=["table", "gone"], missing="skip"
    )
    assert list(opened.obsm) == ["emb"] and list(opened.obs.columns) == ["group"] and list(opened.uns) == ["table"]
