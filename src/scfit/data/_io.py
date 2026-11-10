"""Container-agnostic data access for the loader: obs columns, leaf codes, and rep backings.

Every helper works uniformly over an in-memory ``AnnData``, an out-of-core annbatch
``DatasetCollection``, and reps given as :mod:`anndata.acc` references (``A.X``, ``A.layers[k]``,
``A.obsm[k]``), so the loader never branches on the source kind. No cell matrices are touched for
grouping — obs only.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from anndata.acc import AdRef, LayerAcc, MetaAcc

from scfit.data._schema import Container, Rep, _slot, weight_vector

if TYPE_CHECKING:
    from scfit.data._schema import Stream

__all__ = [
    "get_from_container",
    "leaf_codes",
    "materialize_node",
    "obs_columns",
    "open_adata",
    "open_source",
    "is_backed_array",
]


def read_backing(x: zarr.Array | zarr.Group) -> zarr.Array | ad.abc.CSRDataset:
    if hasattr(x, "shape"):
        return x
    from anndata.io import sparse_dataset

    return sparse_dataset(x)


def get_from_container(source: ad.AnnData | list[ad.AnnData], ref: Rep) -> list[zarr.Array | ad.abc.CSRDataset]:
    """The array(s) backing rep ``ref`` (``A.X`` | ``A.layers[k]`` | ``A.obsm[k]``) — one per adata for a list."""
    if isinstance(source, list):
        return [b for a in source for b in get_from_container(a, ref)]
    return [ref.get(source)]


def _put(adata: ad.AnnData, ref: Rep, value) -> None:
    """Place ``value`` at rep ``ref``'s slot of ``adata``, so :func:`get_from_container` finds it again."""
    field, k = _slot(ref)
    if k is None:
        adata.X = value
    else:
        getattr(adata, field)[k] = value


def _path(ref: Rep) -> str:
    """Rep ``ref``'s path inside a zarr / h5ad store (``"X"``, ``"layers/<k>"``, ``"obsm/<k>"``)."""
    field, k = _slot(ref)
    return field if k is None else f"{field}/{k}"


def _read_rows(source: Container, ref: Rep, row_idx: np.ndarray) -> np.ndarray:
    r"""Densely read the global rows ``row_idx`` (ascending, unique) of rep ``ref`` from a source's backings.

    Each maximal run of *consecutive* rows is read as one contiguous slice (``b[start:stop]``), not a
    scattered fancy-index gather — so a leaf-contiguous layout collapses to a few big reads, and fully
    scattered rows degrade to one slice per row (still correct). Slicing reads uniformly across dense zarr,
    backed ``CSRDataset``\\s and in-memory arrays, so no per-backend indexing branch is needed.
    """
    backings = get_from_container(source, ref)  # each has ``.shape`` (sparse groups already wrapped by read_backing)
    offs = np.concatenate([[0], np.cumsum([b.shape[0] for b in backings])]).astype(np.int64)
    parts = []
    for d, b in enumerate(backings):
        lb = row_idx[(row_idx >= offs[d]) & (row_idx < offs[d + 1])] - offs[d]  # ascending within this dataset
        runs = np.split(lb, np.flatnonzero(np.diff(lb) != 1) + 1) if lb.size else ()  # maximal consecutive runs
        for run in runs:
            sub = b[int(run[0]) : int(run[-1]) + 1]  # contiguous slice read
            dense = getattr(sub, "todense", None)
            parts.append(np.asarray(dense()) if dense is not None else np.asarray(sub))
    return np.concatenate(parts, axis=0) if parts else np.empty((0, int(backings[0].shape[1])), dtype=np.float32)


def materialize_node(
    source: Container, stream: Stream, factorization: tuple[np.ndarray, list[tuple]] | None = None
) -> tuple[ad.AnnData, np.ndarray, list[tuple]]:
    """Materialize a stream's selected (positive-weight) cells into an in-memory (dense) ``AnnData``.

    Behind :attr:`~scfit.data.Stream.in_memory`: reads only the positive-weight rows (for each rep) and
    sorts them by ``group_by`` so ``chunk_size > 1`` still reads contiguous runs. Dense and sparse (CSR
    group) alike. Cells must fit host RAM (intended for a small, frequently re-drawn population, e.g.
    matched controls).

    Also returns the subset's ``(codes, leaves)`` — the caller's sampler needs the subset categorical, and
    it's derived from ``factorization`` (the source's cached ``leaf_codes`` over ``group_by``) when given,
    avoiding a re-factorize; omit it for standalone use.
    """
    cols = stream.group_by
    obs = obs_columns(source, cols)
    codes, leaves = factorization if factorization is not None else leaf_codes(obs, cols)
    selected = np.flatnonzero(weight_vector(stream.weights, leaves) > 0)  # leaf codes with positive weight
    row_idx = np.flatnonzero(np.isin(codes, selected))  # ascending global rows of the selected leaves
    reps = {ref: _read_rows(source, ref, row_idx) for ref in stream.reps}  # keyed by reference
    sub_obs = obs.iloc[row_idx].reset_index(drop=True)
    # contiguous runs for chunk>1; with no grouping every row is already in the one run
    order = sub_obs.sort_values(list(cols), kind="stable").index.to_numpy() if cols else np.arange(len(sub_obs))
    sub_obs, reps = sub_obs.iloc[order].reset_index(drop=True), {k: v[order] for k, v in reps.items()}
    sub_obs.index = sub_obs.index.astype(str)
    n_vars = next((v.shape[1] for r, v in reps.items() if isinstance(r, LayerAcc)), 0)  # X / layer width
    adata = ad.AnnData(obs=sub_obs, var=pd.DataFrame(index=pd.RangeIndex(n_vars).astype(str)))
    for ref, v in reps.items():
        _put(adata, ref, v)
    # Subset (codes, leaves) derived from the parent factorization — no re-factorize. Parent ``leaves`` are
    # string-sorted, so the distinct parent codes present (ascending) are already the subset's leaves in that
    # order, and searchsorted compacts parent → subset codes. Byte-identical to leaf_codes(sub_obs, cols).
    sel = codes[row_idx][order]  # parent leaf code per subset row, in the materialized (sorted) order
    present = np.unique(sel)  # ascending == string-sorted order of the parent leaves
    sub_leaves = [leaves[c] for c in present]
    sub_codes = np.searchsorted(present, sel)
    return adata, sub_codes, sub_leaves


def leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations (the grouping over ``cols``).

    ``leaves`` are the unique ``cols`` combinations as ``.to_numpy()``-typed tuples — the same construction
    the ``{combo: weight}`` mappings use, so a leaf hash-equals its weight key — ordered by per-element
    string key; ``codes[i]`` indexes ``leaves``.

    Vectorized: rows are factorized once at C level and remapped onto the string-sorted leaf order. Grouping
    columns are cast to ``category`` first (a pure speed lever — the returned ``(codes, leaves)`` are
    identical either way).
    """
    cols = list(cols)
    if not cols:  # no grouping: every cell is in the one leaf ()
        return np.zeros(len(obs), dtype=np.int64), [()] if len(obs) else []
    sub = obs[cols]
    # leaves keep the source dtypes' `.to_numpy()` element types (loop over the FEW unique rows only).
    leaves = sorted((tuple(r) for r in sub.drop_duplicates().to_numpy()), key=lambda t: tuple(map(str, t)))
    if len(sub) == 0:
        return np.zeros(0, dtype=np.int64), leaves
    # Per-column categorical codes + mixed-radix combine → a raw per-cell group code without materializing
    # ~N object tuples. Cast only non-categorical cols (keeps existing category orders); a column's NaN is
    # code -1, shifted to slot 0 so it owns a distinct combination.
    cats = [sub[c] if isinstance(sub[c].dtype, pd.CategoricalDtype) else sub[c].astype("category") for c in cols]
    ncards = [len(c.cat.categories) + 1 for c in cats]
    prod = 1
    for n in ncards:
        prod *= n
    if prod < (1 << 62):  # O(N) integer combine (no object tuples); guard against int64 overflow
        combined = np.zeros(len(sub), dtype=np.int64)
        mult = 1
        for cat, n in zip(cats, ncards, strict=True):
            combined += (cat.cat.codes.to_numpy().astype(np.int64) + 1) * mult
            mult *= n
        raw = pd.factorize(combined, use_na_sentinel=False)[0]
    else:  # rare: cardinality product overflows int64 → fall back to the object-tuple path
        raw = pd.factorize(
            pd.MultiIndex.from_frame(pd.DataFrame({c: cats[i] for i, c in enumerate(cols)})), use_na_sentinel=False
        )[0]
    # Remap raw group codes → string-sorted leaf codes, keying off each group's representative row read back
    # through `.to_numpy()` (like `leaves`) — so both share the same dtype promotion and NaN→"nan"
    # normalization that the raw factorize `uniques` (keeping per-column dtypes) would not.
    _, first_idx = np.unique(raw, return_index=True)  # first row of each group code g = 0..G-1
    reps = sub.iloc[first_idx].to_numpy()
    key_to_code = {tuple(map(str, lf)): i for i, lf in enumerate(leaves)}
    remap = np.fromiter((key_to_code[tuple(map(str, r))] for r in reps), dtype=np.int64, count=len(reps))
    return remap[raw], leaves


def obs_columns(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """The ``cols`` of ``source``'s obs, without touching any cell matrix.

    A list of AnnData is read as one dataset: their columns are stacked in list order with a fresh positional
    index, and a column categorical in every file gets the union of their categories, so its codes compare
    across files.
    """
    if isinstance(source, list):  # list of AnnData: concatenate obs in list order (= key_backings order)
        frames = [obs_columns(a, cols) for a in source]
        _align_categoricals(frames)
        return pd.concat(frames, ignore_index=True)
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection


def _align_categoricals(frames: list[pd.DataFrame]) -> None:
    """Align each column that is categorical in *every* frame to the union of its categories, in place.

    ``pd.concat`` upcasts categoricals with mismatched categories to ``object``, so per-source obs with
    differing category sets would turn grouping columns into strings and force a re-factorize downstream.
    Setting the union categories first keeps integer codes (faster, lower-memory). No-op for <2 frames or
    columns not categorical everywhere.
    """
    from pandas.api.types import union_categoricals

    if len(frames) < 2:
        return
    for col in frames[0].columns:
        series = [f[col] for f in frames]
        if not all(isinstance(s.dtype, pd.CategoricalDtype) for s in series):
            continue
        union = union_categoricals(series, ignore_order=True).categories
        for f in frames:
            f[col] = f[col].cat.set_categories(union)  # replace the column; `.loc` cannot change a dtype


def _keys(group) -> list[str]:
    return sorted(group.keys()) if group is not None else []


def _has(g, ref: Rep) -> bool:
    field, k = _slot(ref)
    return "X" in g if k is None else k in _keys(g.get(field))


def _obs_cols(g) -> list[str]:
    return list(g["obs"].attrs["column-order"])  # the on-disk dataframe spec lists its columns


def _check_present(where, g, arrays: Sequence[Rep], cols: Sequence[str], uns: Sequence[str]) -> None:
    """Raise one :class:`KeyError` naming every missing rep, obs column and uns key, with what the store has."""
    missing = []
    for ref in arrays:
        if not _has(g, ref):
            field, k = _slot(ref)
            missing.append("X" if k is None else f"{field}[{k!r}] (has {field}: {_keys(g.get(field))})")
    if absent := sorted(set(cols) - set(_obs_cols(g))):
        missing.append(f"obs {absent} (has {sorted(_obs_cols(g))})")
    if absent := sorted(set(uns) - set(_keys(g.get("uns")))):
        missing.append(f"uns {absent} (has {_keys(g.get('uns'))})")
    if missing:
        raise KeyError(f"{where} lacks " + "; ".join(missing))


def _assemble(g, obs: pd.DataFrame, arrays: dict, uns: Sequence[str]) -> ad.AnnData:
    """An AnnData of ``obs``, ``var``'s index, the opened ``arrays`` (``{ref: array}``) and the ``uns`` keys."""
    var = g["var"]
    adata = ad.AnnData(obs=obs, var=pd.DataFrame(index=pd.Index(ad.io.read_elem(var[var.attrs["_index"]])).astype(str)))
    for ref, array in arrays.items():
        _put(adata, ref, array)
    for key in uns:
        adata.uns[key] = ad.io.read_elem(g["uns"][key])
    return adata


def _split_refs(refs) -> tuple[list[Rep], list[str], bool]:
    """``(arrays, obs columns, whether the obs index is asked for)`` from :mod:`anndata.acc` references.

    A reference into an array (``A.obsm["pca"][:, 0]``) stands for the whole array.
    """
    arrays, cols, index = [], [], False
    for ref in refs:
        if isinstance(ref, AdRef) and isinstance(ref.acc, MetaAcc):
            if ref.acc.dim != "obs":
                raise ValueError(f"{ref!r}: only obs columns and the obs index can be opened, not var.")
            if ref.idx is None:
                index = True
            else:
                cols.append(ref.idx)
        else:
            arr = ref.acc if isinstance(ref, AdRef) else ref
            _slot(arr)  # X, a layer or an obsm entry
            arrays.append(arr)
    return list(dict.fromkeys(arrays)), list(dict.fromkeys(cols)), index


def _is_h5ad(path) -> bool:
    return os.path.splitext(os.fspath(path))[1].lower() == ".h5ad"


def _open_store(path):
    """A zarr group, or an h5py file for a ``.h5ad`` path. An annbatch collection root is refused."""
    if _is_h5ad(path):
        import h5py

        return h5py.File(path, "r")
    g = zarr.open_group(path, mode="r")
    if g.attrs.get("encoding-type") == "annbatch-preshuffled":
        raise NotImplementedError(
            f"{path} is an annbatch DatasetCollection root, which open_adata does not support yet."
        )
    return g


def open_adata(
    path: str | os.PathLike, *refs, uns: Sequence[str] = (), missing: Literal["raise", "skip"] = "raise"
) -> ad.AnnData:
    """Open ``path`` with exactly the parts the :mod:`anndata.acc` references name.

    ``A.X``, ``A.layers[k]`` and ``A.obsm[k]`` open arrays; a reference into one, such as
    ``A.obsm["pca"][:, 0]``, opens the whole array. ``A.obs[col]`` reads a column and ``A.obs.index`` the obs
    names; without it the frame gets a ``RangeIndex``. ``uns`` keys are read into memory.

    The arrays of a zarr store stay on disk, wrapped by :func:`read_backing` (a dense array passes through,
    a sparse group becomes a ``CSRDataset``), so the loader streams them and a caller reads only the rows it
    indexes. ``var`` is reduced to its index. A ``.h5ad`` file is read the same way, but its
    arrays are loaded into memory and the file is closed: the loader streams only zarr and backed sparse
    arrays, and an open h5py handle blocks writes to the file and does not survive a forked worker.
    A reference the store lacks raises a :class:`KeyError` naming what it has, or with ``missing="skip"`` is left
    out, for a dataset whose files each hold some of the parts.
    """
    arrays, cols, index = _split_refs(refs)
    g = _open_store(path)
    h5ad = not isinstance(g, zarr.Group)
    try:
        if missing == "skip":
            arrays = [r for r in arrays if _has(g, r)]
            cols = [c for c in cols if c in set(_obs_cols(g))]
            uns = [k for k in uns if k in _keys(g.get("uns"))]
        else:
            _check_present(path, g, arrays, cols, uns)
        # zarr: on disk, wrapped for streaming; h5ad: into memory (dense or sparse) before the file closes
        read = ad.io.read_elem if h5ad else read_backing
        opened = {ref: read(g[_path(ref)]) for ref in arrays}
        obs_group = g["obs"]
        obs = pd.DataFrame({c: ad.io.read_elem(obs_group[c]) for c in cols})
        if index or not (opened or cols):  # nothing else gives the number of cells
            obs.index = pd.Index(ad.io.read_elem(obs_group[obs_group.attrs["_index"]])).astype(str)
        else:
            obs.index = pd.RangeIndex(next(iter(opened.values())).shape[0] if opened else len(obs)).astype(str)
        return _assemble(g, obs, opened, uns)
    finally:
        if h5ad:
            g.close()


def open_source(src, *refs, uns: Sequence[str] = ()) -> Container:
    """Resolve one :class:`~scfit.data.Stream` source value to a :data:`~scfit.data.Container`.

    An in-memory ``AnnData`` or ``DatasetCollection`` passes through unchanged. A path (zarr or ``.h5ad``) is
    opened with :func:`open_adata`, reading only the parts the :mod:`anndata.acc` ``refs`` name and the
    ``uns`` keys; a zarr annbatch collection root (``encoding-type`` ``"annbatch-preshuffled"``) is not yet
    supported. A list of paths becomes a list of AnnData (the loader feeds one backing per adata to
    ``add_datasets``).
    """
    from annbatch import DatasetCollection

    if isinstance(src, ad.AnnData | DatasetCollection):
        return src
    if isinstance(src, str | os.PathLike):
        return open_adata(src, *refs, uns=uns)
    return [a if isinstance(a, ad.AnnData) else open_adata(a, *refs, uns=uns) for a in src]


def is_backed_array(arr) -> bool:
    """True if ``arr`` is an on-disk backing (dense zarr array / backed ``CSRDataset``), not in-memory."""
    import zarr
    from anndata.abc import CSRDataset

    return isinstance(arr, zarr.Array | CSRDataset)
