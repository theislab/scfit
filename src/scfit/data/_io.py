"""Container-agnostic data access for the loader: obs columns, leaf codes, and rep backings.

Every helper works uniformly over an in-memory ``AnnData`` and an out-of-core annbatch
``DatasetCollection``, and over ``X`` / ``obsm`` / ``layers`` representations — so the loader never
branches on the source kind. No cell matrices are touched for grouping (obs only); cells are read
only when a batch is materialized (by annbatch's own loader over the returned backings).
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import zarr

from scfit.data._schema import Container, Node, weight_vector

__all__ = [
    "get_from_container",
    "leaf_codes",
    "load_backed_adata",
    "materialize_node",
    "obs_columns",
    "open_source",
    "is_backed_array",
]


def read_backing(x: zarr.Array | zarr.Group) -> zarr.Array | ad.abc.CSRDataset:
    if hasattr(x, "shape"):
        return x
    from anndata.io import sparse_dataset

    return sparse_dataset(x)


def get_from_container(source: ad.AnnData | list[ad.AnnData], loc: str) -> list[zarr.Array | ad.abc.CSRDataset]:
    """The array(s) backing rep ``loc`` (``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``) — one per adata for a list."""
    if isinstance(source, list):
        return [b for a in source for b in get_from_container(a, loc)]
    if loc == "X":
        return [source.X]
    field, sub = loc.split("/", 1)  # "obsm/<k>" | "layers/<k>"
    return [getattr(source, field)[sub]]


def _read_rows(source: Container, loc: str, row_idx: np.ndarray) -> np.ndarray:
    r"""Densely read the global rows ``row_idx`` (ascending, unique) of rep ``loc`` from a source's backings.

    Each maximal run of *consecutive* rows is read as a **slice** (``b[start:stop]`` — one contiguous I/O)
    rather than a scattered fancy-index gather. ``materialize_node`` selects whole positive-weight leaves,
    so on a source laid out by the node's ``cols`` (the grouping the whole design assumes — see
    ``chunk_size``) the selection collapses to a handful of long runs: a few big slice reads instead of a
    per-row gather. Fully scattered rows degrade to one slice per row — no worse in row count, and still
    correct. Slicing reads uniformly across dense zarr arrays, backed ``CSRDataset``\\s and in-memory
    scipy/numpy, so no per-backend indexing branch is needed.
    """
    backings = get_from_container(source, loc)  # each has ``.shape`` (sparse groups already wrapped by read_backing)
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


# TODO: remove this, this shouldn't be our concern
def materialize_node(
    source: Container, node: Node, factorization: tuple[np.ndarray, list[tuple]] | None = None
) -> tuple[ad.AnnData, np.ndarray, list[tuple]]:
    """Materialize a node's selected (positive-weight) cells into an in-memory (dense) ``AnnData``.

    The general "read this node's cells into RAM" op behind :attr:`~binded.Node.in_memory`: reads only
    the rows whose leaf has positive weight — for each of the node's reps (``keys``) — and sorts them by
    ``cols`` so ``chunk_size > 1`` still reads contiguous runs. Handles dense and sparse (CSR-group)
    backings alike (via :func:`key_backings`). Cells must fit host RAM (the intended use is a small,
    frequently re-drawn population such as matched controls).

    Returns the materialized ``AnnData`` together with *its* ``(codes, leaves)`` factorization — the caller
    (the loader's sampler) needs the subset's categorical, and it is derived here from the source
    factorization rather than re-factorizing the subset obs. Pass ``factorization`` — the source's
    :func:`leaf_codes` over ``node.cols`` — to reuse a cached one (the source obs is then never factorized
    a second time); omit it and it is computed here for standalone use.
    """
    obs = obs_columns(source, node.cols)
    codes, leaves = factorization if factorization is not None else leaf_codes(obs, node.cols)
    selected = np.flatnonzero(weight_vector(node.weights, leaves) > 0)  # leaf codes with positive weight
    row_idx = np.flatnonzero(np.isin(codes, selected))  # ascending global rows of the selected leaves
    reps = {key: _read_rows(source, key, row_idx) for key in node.keys}  # keyed by loc string
    sub_obs = obs.iloc[row_idx].reset_index(drop=True)
    order = sub_obs.sort_values(list(node.cols), kind="stable").index.to_numpy()  # contiguous runs for chunk>1
    sub_obs, reps = sub_obs.iloc[order].reset_index(drop=True), {k: v[order] for k, v in reps.items()}
    adata = ad.AnnData(X=reps["X"], obs=sub_obs) if "X" in reps else ad.AnnData(obs=sub_obs)  # X -> n_var inferred
    for key, v in reps.items():  # place each non-X rep so `get_from_container(adata, key)` finds it again
        if key != "X":
            field, sub = key.split("/", 1)  # "obsm/<k>" | "layers/<k>"
            getattr(adata, field)[sub] = v
    # Subset (codes, leaves) derived from the parent factorization — no re-factorize. Parent ``leaves`` are
    # string-key sorted, so the ascending distinct parent codes present in the subset already give the
    # subset's leaves in that same order, and searchsorted compacts parent codes -> subset codes. This is
    # byte-identical to ``leaf_codes(sub_obs, node.cols)`` (same string-sorted order, same ``.to_numpy()``
    # dtypes, since ``sub_obs`` is a row-subset of the very obs ``leaves`` came from).
    sel = codes[row_idx][order]  # parent leaf code per subset row, in the materialized (sorted) order
    present = np.unique(sel)  # ascending == string-sorted order of the parent leaves
    sub_leaves = [leaves[c] for c in present]
    sub_codes = np.searchsorted(present, sel)
    return adata, sub_codes, sub_leaves


def leaf_codes(obs: pd.DataFrame, cols: Sequence[str]) -> tuple[np.ndarray, list[tuple]]:
    """Per-cell leaf code + the ordered leaf combinations (the grouping over ``cols``).

    ``leaves`` are the unique ``cols`` combinations as ``.to_numpy()``-typed tuples — the *same*
    construction the ``{combo: weight}`` mappings use, so a leaf hash-equals its weight key — ordered by
    per-element string key. ``codes[i]`` indexes ``leaves`` for cell ``i``.

    Vectorized: the rows are factorized once at C level (no per-cell Python loop) and the raw factorize
    codes are remapped onto the string-sorted leaf order. Grouping columns are cast to ``category`` first
    so the factorize hashes small integer codes rather than raw (string) values — a pure speed lever, the
    returned ``(codes, leaves)`` are identical either way (already-categorical columns are left as-is, so
    a column's category order — hence the leaf order — is untouched).
    """
    cols = list(cols)
    sub = obs[cols]
    # leaves keep the source dtypes' `.to_numpy()` element types (loop over the FEW unique rows only).
    leaves = sorted((tuple(r) for r in sub.drop_duplicates().to_numpy()), key=lambda t: tuple(map(str, t)))
    if len(sub) == 0:
        return np.zeros(0, dtype=np.int64), leaves
    # Per-column categorical codes + mixed-radix combine → a raw per-cell group code, WITHOUT materializing
    # ~N object tuples (the old `pd.factorize(pd.MultiIndex.from_frame(...))` spent most of its time in
    # `MultiIndex._values` building N tuples). Cast only non-categorical cols (keeps existing category
    # orders); a column's NaN is code -1, shifted to slot 0 so it owns a distinct combination.
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
    # Remap raw group codes → string-sorted leaf codes. The per-group representative is read back through
    # `.to_numpy()` (like `leaves`), so both sides share the same dtype promotion (e.g. mixed int/float →
    # float) and NaN normalizes to "nan" on both — the raw factorize `uniques` would not (it keeps each
    # column's own dtype), which is why we key off representative rows, not `uniques`.
    _, first_idx = np.unique(raw, return_index=True)  # first row of each group code g = 0..G-1
    reps = sub.iloc[first_idx].to_numpy()
    key_to_code = {tuple(map(str, lf)): i for i, lf in enumerate(leaves)}
    remap = np.fromiter((key_to_code[tuple(map(str, r))] for r in reps), dtype=np.int64, count=len(reps))
    return remap[raw], leaves


def obs_columns(source: Container, cols: Sequence[str]) -> pd.DataFrame:
    """Obs columns from any container (AnnData attr vs DatasetCollection reader vs list) — no cell matrices."""
    if isinstance(source, list):  # list of AnnData: concatenate obs in list order (= key_backings order)
        frames = [obs_columns(a, cols) for a in source]
        _align_categoricals(frames)
        return pd.concat(frames, ignore_index=True)
    if isinstance(source, ad.AnnData):
        return source.obs[list(cols)]
    return source.obs(columns=list(cols))  # DatasetCollection


def _align_categoricals(frames: list[pd.DataFrame]) -> None:
    """Align each column that is categorical in *every* frame to the union of its categories, in place.

    ``pd.concat`` upcasts categoricals with mismatched categories to ``object`` — so concatenating
    per-source obs whose category sets differ (e.g. per-plate drugs/cell-lines) turns the grouping
    columns into strings, and every downstream consumer (:func:`leaf_codes`, the samplers) then
    re-factorizes ~N-cell string columns. Setting the union categories first keeps the concat
    categorical (integer codes), which is both faster and lower-memory. No-op for <2 frames or for
    columns that aren't categorical everywhere. Cheap: obs cols are already in memory; only codes remap.
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
            f[col] = f[col].cat.set_categories(union)


def _read_obs_cols(obs_group, cols: Sequence[str]) -> pd.DataFrame:
    """Read only ``cols`` from a zarr obs group as a pandas frame — no full read, no xarray ``Dataset2D``.

    ``read_elem(obs_group)`` would decode every column; ``read_lazy`` would yield an xarray-backed
    ``Dataset2D``. Instead read each requested column element on its own (each ``read_elem`` reconstructs
    its dtype — categoricals included), so unused obs columns are never touched.

    The obs **index** is deliberately NOT read: it is frequently a large object/string barcode array
    (e.g. Tahoe's ~1.4GB over 90M cells) and nothing downstream uses it — ``leaf_codes`` keys off the
    columns and ``materialize_node`` ``reset_index(drop=True)``. The frame gets a default ``RangeIndex``
    (row position), which is all any consumer relies on. (Pass ``cols=()`` for the full obs incl. index.)
    """
    if not cols:
        return ad.io.read_elem(obs_group)
    return pd.DataFrame({c: ad.io.read_elem(obs_group[c]) for c in cols})


def load_backed_adata(g, *, keys: Sequence[str], cols: Sequence[str] = ()) -> ad.AnnData:
    """Open a zarr adata group as a (backed) AnnData, reading only the reps in ``keys`` and obs ``cols``.

    Each rep loc (``"X"`` | ``"obsm/<k>"`` | ``"layers/<k>"``) is read straight from the zarr group and
    wrapped backed via :func:`read_backing` (dense array passes through; a sparse group becomes a
    ``CSRDataset``), so nothing is pulled into RAM, and placed at the matching slot so
    ``get_from_container`` finds it again. ``var`` is reduced to its index and ``obs`` to ``cols``.
    """
    var = g["var"]
    kw: dict = {
        "obs": _read_obs_cols(g["obs"], cols),
        "var": pd.DataFrame(index=pd.Index(ad.io.read_elem(var[var.attrs.get("_index")]))),
    }
    if "X" in keys:
        kw["X"] = read_backing(g["X"])
    adata = ad.AnnData(**kw)
    for key in keys:
        if key == "X":
            continue
        field, sub = key.split("/", 1)  # "obsm/<k>" | "layers/<k>"
        getattr(adata, field)[sub] = read_backing(g[field][sub])
    return adata


def open_source(src, *, keys: Sequence[str], cols: Sequence[str] = ()) -> Container:
    """Resolve one :class:`~binded.Scheme` source value to a :data:`~binded.Container`.

    An in-memory ``AnnData`` or ``DatasetCollection`` passes through unchanged. A single path is opened as
    zarr and auto-detected: an annbatch collection root (``encoding-type`` ``"annbatch-preshuffled"``)
    becomes a ``DatasetCollection``; otherwise it is a single adata read backed via :func:`load_backed_adata`.
    A list of paths becomes a list of backed AnnData (the loader feeds one backing per adata to
    ``add_datasets``). Only the reps in ``keys`` and obs ``cols`` are read (see :func:`load_backed_adata`).
    """
    import zarr
    from annbatch import DatasetCollection

    if isinstance(src, ad.AnnData | DatasetCollection):
        return src
    if isinstance(src, str | os.PathLike):
        g = zarr.open_group(src, mode="r")
        if g.attrs.get("encoding-type") == "annbatch-preshuffled":
            raise NotImplementedError("DatasetCollection support is not yet implemented in open_source.")
            # return DatasetCollection(src, mode="r")
        return load_backed_adata(g, keys=keys, cols=cols)
    # list/sequence of adata zarr paths (or already-open AnnData) → list of backed AnnData
    return [
        a if isinstance(a, ad.AnnData) else load_backed_adata(zarr.open_group(a, mode="r"), keys=keys, cols=cols)
        for a in src
    ]


def is_backed_array(arr) -> bool:
    """True if ``arr`` is an on-disk backing (dense zarr array / backed ``CSRDataset``), not in-memory."""
    import zarr
    from anndata.abc import CSRDataset

    return isinstance(arr, zarr.Array | CSRDataset)
