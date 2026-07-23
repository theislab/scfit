r"""annbatch adapter — shared plumbing between binded's ``Scheme`` and annbatch's samplers/loaders.

Turns a :class:`~binded.Scheme`'s nodes/sources into annbatch samplers + :class:`~annbatch.Loader`\\s.
:class:`~binded.Loader` (target-rooted, random epoch) sits on top of this layer: resolve a node's
source, build the tuple-labelled categorical the bound sampler matches on, and wire one annbatch
``Loader`` per rep. These primitives are kept source-kind-agnostic and loader-agnostic here (rather than
inlined into the loader) so a second loader — e.g. a control-rooted deterministic eval sweep, added
later — can reuse them by wiring a different sampler topology on top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import DatasetCollection
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not binded's `Loader`)
from annbatch.samplers import BoundClassSampler, ClassSampler

from scfit.data._io import _readable, key_backings, leaf_codes, materialize_node, obs_columns
from scfit.data._schema import Container, Node, SamplerConfig, Scheme, _weight_vector

# annbatch's GPU path (cupy vstack/indexing → the batch is GPU-resident via dlpack) needs cupy. When it's
# absent (CPU-only envs — Mac, CI), fall back so `to="torch"`/`"jax"` still yields a CPU array.
_HAS_CUPY = find_spec("cupy") is not None


def _is_backed(arr) -> bool:
    """True if ``arr`` is an on-disk backing (dense zarr array / backed ``CSRDataset``), not in-memory."""
    import zarr
    from anndata.abc import CSRDataset

    return isinstance(arr, zarr.Array | CSRDataset)


def _group_rep(g, key: str):
    """Rep ``key`` of a collection's zarr group as a readable backing (dense array / wrapped CSR group)."""
    if key == "X":
        return _readable(g["X"])
    field, sub = key.split("/", 1)  # "obsm/X_pca" | "layers/log1p"
    return _readable(g[field][sub])


def _flat_categorical(codes: np.ndarray, leaves: list[tuple]) -> pd.Categorical:
    """A tuple-labelled categorical: per-cell leaf code over ``leaves`` (categories are the leaf tuples).

    Tuple labels (not opaque integer codes) are what let :class:`~annbatch.samplers.BoundClassSampler`
    match a child to its parent by the bind's ``common`` columns — it projects the label by position.
    """
    categories = pd.MultiIndex.from_tuples(leaves).to_flat_index()
    return pd.Categorical.from_codes(codes, categories=categories)


class _SchemeReader:
    """Base for the loader(s): resolves each node's source and factorizes each source's obs ONCE.

    Holds the per-node primitive :class:`~scfit.data.Loader` (and a later eval loader) needs — the resolved
    source (a ``Node.in_memory`` node materialized into RAM) plus its tuple-labelled categorical, weight
    vector and leaf list. A scheme source's obs is read + factorized ONCE per ``(source, cols)`` via
    :meth:`_factorize`'s cache, and that single ``(codes, leaves)`` is reused across every node over the
    same ``(source, cols)`` — a backed node's categorical *and* (through :func:`materialize_node`) an
    ``in_memory`` sibling's row selection — so the (possibly 100M-row) source obs is never factorized twice.
    Holding the cache on the instance is why the loader never threads it through its own methods.
    """

    def __init__(self) -> None:
        # (source name, cols) -> its full-obs (codes, leaves). Construction-only scratch; loaders that
        # pickle drop it (the per-node categoricals already carry their own codes).
        self._leaf_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, list[tuple]]] = {}

    @staticmethod
    def _check_in_memory_chunk(name: str, cfg: SamplerConfig, node: Node) -> None:
        """Reject a chunked config on an ``in_memory`` node (both loaders call this, so the rule lives once).

        A materialized node is read from RAM in one shot, so a ``chunk_size > 1`` sampler over it is a
        contradiction — the run-length rule it implies (each sampled leaf in a contiguous run ≥
        ``chunk_size``) is meaningless. Rather than silently rewrite the user's ``chunk_size`` (a
        ``SamplerConfig`` field is deliberately explicit — no hidden defaults), require it to be 1 and raise
        a clear error otherwise, so the user sets it themselves.
        """
        if node.in_memory and cfg.chunk_size != 1:
            raise ValueError(
                f"node {name!r} sets in_memory=True but chunk_size={cfg.chunk_size}: an in-memory node is "
                "read from RAM in one shot and must use chunk_size=1 (set it explicitly)."
            )

    def _factorize(self, src: Container, source: str, cols: tuple[str, ...]) -> tuple[np.ndarray, list[tuple]]:
        """The source's ``(codes, leaves)`` over ``cols`` — read + factorized once, cached by ``(source, cols)``."""
        ck = (source, cols)
        if ck not in self._leaf_cache:
            self._leaf_cache[ck] = leaf_codes(obs_columns(src, cols), cols)
        return self._leaf_cache[ck]

    def _node_stats(self, src: Container, node: Node) -> tuple[pd.Categorical, np.ndarray, list[tuple]]:
        """Tuple-labelled categorical, normalized weight vector, and leaf list for a node read from ``src``.

        ``src`` is the node's scheme source (not a materialized subset); its factorization is shared via
        :meth:`_factorize`. The categorical and weight vector are cheap and always rebuilt — nodes over the
        same ``(source, cols)`` differ only in ``weights``.
        """
        codes, leaves = self._factorize(src, node.source, node.cols)
        return _flat_categorical(codes, leaves), _weight_vector(node.weights, leaves), leaves

    def _prepare(self, scheme: Scheme, node: Node) -> tuple[Container, pd.Categorical, np.ndarray, list[tuple]]:
        """A node's resolved source + its categorical, weight vector and leaf list (obs only).

        An ``in_memory`` node is materialized into RAM here (positive-weight rows only): the row selection
        reuses the *cached* source factorization rather than re-reading the source obs, and the subset's
        own ``(codes, leaves)`` are derived by :func:`materialize_node` from that same factorization — so no
        obs is factorized more than once. A non-materialized node reads straight from the scheme source.
        """
        src = scheme.sources[node.source]
        if node.in_memory:
            codes, leaves = self._factorize(src, node.source, node.cols)
            src, codes, leaves = materialize_node(src, node, (codes, leaves))
            return src, _flat_categorical(codes, leaves), _weight_vector(node.weights, leaves), leaves
        cats, w, leaves = self._node_stats(src, node)
        return src, cats, w, leaves

    @staticmethod
    def _emit_rep(out: dict, kind: str, reps: dict[str, np.ndarray], node: Node) -> None:
        """Write a node's batch into ``out`` under ``kind`` (``"source"`` | ``"target"``).

        The shared batch-dict schema both loaders emit: ``out[kind]`` is the primary streamed rep
        (``node.keys[0]``), and ``out[f"{kind}_reps"]`` carries the full aligned-rep dict only when the node
        streams more than one key.
        """
        out[kind] = reps[node.keys[0]]
        if len(node.keys) > 1:
            out[f"{kind}_reps"] = reps


def _bind_on(inner_node: Node, bound_node: Node, common: Sequence[str]) -> dict[int, int]:
    """Map inner-node tuple positions → bound-node tuple positions for each shared ``common`` column."""
    return {inner_node.cols.index(c): bound_node.cols.index(c) for c in common}


def _attach(loader: AnnbatchLoader, src: Container, key: str) -> AnnbatchLoader:
    """Feed rep ``key`` of ``src`` to a fresh annbatch ``Loader`` via the source-appropriate entry point.

    Dispatch by source kind, streaming only rep ``key`` with **no obs** — binded owns the class
    labels through the sampler (``classes=``), so annbatch never needs the source's obs:

    * ``DatasetCollection`` → :meth:`~annbatch.Loader.use_collection` (annbatch's own collection API),
      each group's rep loaded as an obs-free ``X`` (the default ``load_adata`` would decode *all* obs);
    * **backed** ``AnnData`` / list of backed ``AnnData`` → the raw rep backings through ``add_datasets``;
    * **in-memory** ``AnnData`` (a user adata, or a materialized ``in_memory`` node such as the matched
      control) → ``add_adatas`` over the rep wrapped as an obs-free ``X``.
    """
    if isinstance(src, DatasetCollection):
        return loader.use_collection(src, load_adata=lambda g: ad.AnnData(X=_group_rep(g, key)))
    backings = key_backings(src, key)
    if isinstance(src, ad.AnnData) and not _is_backed(backings[0]):  # in-memory adata → add_adatas
        return loader.add_adatas([ad.AnnData(X=b) for b in backings])
    return loader.add_datasets(backings)  # backed adata or list of backed adata


def _build_loaders(
    src: Container,
    node: Node,
    cfg: SamplerConfig,
    make_sampler: Callable[[], ClassSampler | BoundClassSampler],
) -> dict[str, AnnbatchLoader]:
    """Per rep (``node.keys``) an annbatch ``Loader`` over its own fresh sampler, fed via :func:`_attach`.

    Reps need separate Loaders (annbatch can't mix feature dims in one loader), each with native chunked
    reads.

    ``to`` (default "torch") + ``preload_to_gpu`` come from the ``SamplerConfig``. ``to="torch"`` yields
    native torch tensors (no host round-trip); ``preload_to_gpu`` keeps the read window on-GPU (needs cupy),
    else it defers the device copy to the step. Auto-selects cupy when ``preload_to_gpu`` is unset.
    """
    preload_to_gpu = cfg.preload_to_gpu if cfg.preload_to_gpu is not None else _HAS_CUPY  # None ⇒ auto
    loaders: dict[str, AnnbatchLoader] = {}
    for key in node.keys:
        base = AnnbatchLoader(
            batch_sampler=make_sampler(), return_index=False, to=cfg.to, preload_to_gpu=preload_to_gpu
        )
        loaders[key] = _attach(base, src, key)
    return loaders
