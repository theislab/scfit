r"""The :class:`Source` abstraction: one dataset (a list of ``AnnData``) plus its factorization cache.

A ``source_key`` in the :class:`~scfit.data.Loader`'s ``sources`` mapping resolves to one :class:`Source`.
Its cells form **one unified categorical universe**: obs is concatenated in list order (== the backings
order fed to annbatch's ``add_datasets``) and each grouping column is unioned to the categories seen across
files, so a leaf that lives in only one file resolves to that file when sampled (see :func:`~scfit.data._io.obs_columns`).

That list order is a **load-bearing invariant** — :meth:`factorize` (obs concat) and :meth:`rep` (backings)
must iterate ``adatas`` in the same order, or a leaf code would point at the wrong file. The factorization
is cached per ``group_by``, so several streams naming this ``source_key`` factorize it only once (a primary
and its matched control over one dataset); an in-memory dataset is now deduped this way too — the old cache
was keyed by zarr path and skipped in-memory sources.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import anndata as ad
import numpy as np
import pandas as pd

from scfit.data._io import get_from_container, is_backed_array, leaf_codes, materialize_node, obs_columns

if TYPE_CHECKING:
    from scfit.data._schema import Stream

__all__ = ["Source"]


class _Factorized(NamedTuple):
    """A source's obs factorized over one ``group_by``.

    Per-cell leaf ``codes``, the ordered ``leaves``, and the tuple-labelled ``pd.Categorical`` the annbatch
    samplers bind on.
    """

    codes: np.ndarray
    leaves: list[tuple]
    cats: pd.Categorical


def _as_list(adatas: ad.AnnData | list[ad.AnnData]) -> list[ad.AnnData]:
    """Normalize a source value to a list — a single ``AnnData`` becomes a one-element list."""
    return [adatas] if isinstance(adatas, ad.AnnData) else list(adatas)


class Source:
    """One dataset — a list of (in-memory or backed) ``AnnData`` — and its per-``group_by`` factorization."""

    def __init__(self, adatas: ad.AnnData | list[ad.AnnData]) -> None:
        self._adatas = _as_list(adatas)
        if not self._adatas:
            raise ValueError("Source needs at least one AnnData.")
        self._leaf_cache: dict[tuple[str, ...], _Factorized] = {}

    @property
    def adatas(self) -> list[ad.AnnData]:
        """The dataset's AnnData, in the order everything else (factorization, backings) relies on."""
        return self._adatas

    def factorize(self, group_by: tuple[str, ...]) -> _Factorized:
        """The dataset's obs factorized over ``group_by`` (cached — computed once per ``group_by``)."""
        gb = tuple(group_by)
        if gb not in self._leaf_cache:
            codes, leaves = leaf_codes(obs_columns(self._adatas, gb), gb)
            cats = pd.Categorical.from_codes(codes, pd.MultiIndex.from_tuples(leaves).to_flat_index())
            self._leaf_cache[gb] = _Factorized(codes, leaves, cats)
        return self._leaf_cache[gb]

    def rep(self, loc: str) -> list:
        """The array(s) backing rep ``loc``, one per AnnData in list order — with a shared-width check.

        A streamed rep is stacked into one batch array by annbatch, so its ``shape[1]`` must agree across the
        dataset's files. Differing raw gene counts are fine as long as the *streamed* rep is aligned (e.g. a
        shared ``obsm`` embedding); a genuine mismatch raises here rather than deep inside annbatch.
        """
        backings = get_from_container(self._adatas, loc)
        widths = {int(b.shape[1]) for b in backings}
        if len(widths) > 1:
            raise ValueError(
                f"rep {loc!r} has inconsistent shape[1] across this source's AnnData ({sorted(widths)}); a "
                "streamed representation must share its feature dimension across files (align to a common "
                "space — e.g. a shared obsm embedding or a common gene panel — or use separate source_keys)."
            )
        return backings

    def in_memory(self, loc: str) -> bool:
        """True if rep ``loc`` is backed by in-memory arrays (→ annbatch ``add_adatas``), not on-disk backings."""
        return not is_backed_array(self.rep(loc)[0])

    def materialize(self, stream: Stream) -> Source:
        """A new :class:`Source` holding this stream's selected (positive-weight) cells in RAM.

        Backs :attr:`~scfit.data.Stream.in_memory`. Reuses this source's factorization (no re-factorize of
        the subset obs) and pre-seeds the derived subset's factorization on the returned source.
        """
        f = self.factorize(tuple(stream.group_by))
        adata, codes, leaves = materialize_node(self._adatas, stream, (f.codes, f.leaves))
        sub = Source([adata])
        cats = pd.Categorical.from_codes(codes, pd.MultiIndex.from_tuples(leaves).to_flat_index())
        sub._leaf_cache[tuple(stream.group_by)] = _Factorized(codes, leaves, cats)
        return sub

    def clear_cache(self) -> None:
        """Drop the factorization cache (the codes are redundant once a Loader has built its samplers)."""
        self._leaf_cache = {}
