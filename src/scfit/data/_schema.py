r"""Public data spec: the :class:`Stream` consumed by :class:`~scfit.data.Loader`.

Everything is a :class:`Stream` — one streamed population over a source, described by the columns it
groups on, the representation(s) it reads, its per-group weights, and (for a matched *partner*) the
columns it shares with the primary. There is no separate node / bind / scheme object: the loader takes
one primary :class:`Stream` plus any number of named partner :class:`Stream`\\s and wires them into
annbatch samplers directly.

A Stream partitions its source's cells into **leaves** (unique combinations of ``group_by``) with a
per-combination :data:`Weights` mapping. A weight of 0 (or a combination absent from the mapping) is
*excluded* — that IS the selection, native to annbatch's ``ClassSampler``. ``weights=None`` is uniform
over every group. See ``README.md`` for the model and the cellflow / sc-flow-tools mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import TypedDict, Unpack

import anndata as ad
import numpy as np
from anndata.acc import RefAcc

# A representation a stream reads. Accepted as a loc string — ``"X"`` | ``"obsm/<k>"`` | ``"varm/<k>"`` |
# ``"layers/<k>"`` — or the equivalent :mod:`anndata.acc` accessor (``A.X`` | ``A.obsm["<k>"]`` |
# ``A.layers["<k>"]``). :class:`Stream` normalizes to the loc string, the canonical locator downstream
# (the batch-dict rep key, and how a rep is read from a source).
RepKey = str | RefAcc


def _rep_loc(key: RepKey) -> str:
    """Normalize a rep key (loc string or :mod:`anndata.acc` accessor) to a loc string."""
    if isinstance(key, str):
        return key
    dim, k = getattr(key, "dim", None), getattr(key, "k", None)  # anndata.acc accessor
    if dim in ("obs", "var"):  # MultiAcc → obsm / varm
        return f"{dim}m/{k}"
    if k is None:  # LayerAcc with no key → X
        return "X"
    return f"layers/{k}"  # LayerAcc with a key → layers/<k>


type Container = ad.AnnData | list[ad.AnnData]
# What a Stream accepts as its source: an in-memory container, or zarr path(s) opened backed (see _io).
type Source = Container | str | PathLike | Sequence[str | PathLike]

# The selection: a mapping {group -> weight} (a group is a ``group_by`` tuple). A group absent / weight 0
# is excluded — that IS the selection. ``None`` means uniform over every group.
Weights = Mapping[tuple, float]

__all__ = ["Container", "SamplerKwargs", "Source", "Stream", "Weights", "weight_vector"]

# The annbatch read parameters, defined once (via ``Unpack[SamplerKwargs]``) for both Stream and Loader.
_SAMPLER_KEYS = ("batch_size", "chunk_size", "preload_nchunks")


class SamplerKwargs(TypedDict, total=False):
    """The annbatch read parameters, shared by :class:`Stream` and :class:`~scfit.data.Loader`.

    ``total=False`` so a caller may pass none (inherit from the other level) or all three; a *partial*
    set is rejected at runtime by :func:`_check_sampler` (all-or-nothing).

    Parameters
    ----------
    batch_size
        Rows per emitted batch for this stream (source and target row counts need not match).
    chunk_size
        annbatch read-slice size. ``1`` ⇒ per-row reads (any layout); ``>1`` ⇒ contiguous chunked reads
        (each sampled leaf must sit in a contiguous run ≥ ``chunk_size``). Must divide ``batch_size``.
    preload_nchunks
        Chunks per annbatch read window; a positive multiple of ``batch_size // chunk_size``.
    """

    batch_size: int
    chunk_size: int
    preload_nchunks: int


def _check_sampler(sampler: Mapping[str, int], where: str) -> None:
    """Validate collected sampler kwargs: known keys only, and all-or-nothing (partial → error)."""
    extra = [k for k in sampler if k not in _SAMPLER_KEYS]
    if extra:
        raise TypeError(f"{where}: unexpected keyword(s) {extra}; sampler kwargs are {list(_SAMPLER_KEYS)}.")
    given = [k for k in _SAMPLER_KEYS if k in sampler]
    if given and len(given) != len(_SAMPLER_KEYS):
        missing = [k for k in _SAMPLER_KEYS if k not in sampler]
        raise ValueError(
            f"{where}: sampler kwargs are all-or-nothing — got {given} but missing {missing} "
            f"(set all of {list(_SAMPLER_KEYS)}, or none to inherit)."
        )


def weight_vector(weights: Weights | None, leaves: Sequence[tuple]) -> np.ndarray:
    """Resolve ``{group: weight}`` to normalized per-leaf weights (→ ``ClassSampler.class_weights``).

    ``weights=None`` means *uniform* over every leaf (the default: each group equally likely).
    """
    if weights is None:
        v = np.ones(len(leaves), dtype=float)
    else:
        v = np.array([float(weights.get(tuple(lf), 0.0)) for lf in leaves], dtype=float)
    s = v.sum()
    if s <= 0:
        raise ValueError("weights resolve to all-zero over these leaves — nothing to sample.")
    return v / s


class Stream:
    r"""One streamed population: a source, its grouping columns, reps, weights, and read parameters.

    The single public unit :class:`~scfit.data.Loader` consumes. A Stream passed in ``partners=`` (with a
    ``match_on``) is matched batch-for-batch to the primary on the shared ``match_on`` values.

    Parameters
    ----------
    source
        An in-memory ``AnnData``, a zarr path, or a list of either — the cells this stream samples. A path
        is opened backed, reading only the reps + columns this stream uses (see :func:`~binded._io.open_source`).
    group_by
        Columns whose unique combinations define the groups sampled (the leaves).
    rep
        Representation location(s) to stream — a loc string (``"X"`` / ``"obsm/<k>"`` / ``"layers/<k>"``)
        or :mod:`anndata.acc` accessor, or a tuple of them for several **aligned** reps of the same cells.
    weights
        ``{group: weight}`` (a group is a ``group_by`` tuple); a group absent or with weight 0 is excluded.
        :obj:`None` (default) is uniform over every group present.
    match_on
        Columns a *partner* stream shares with the primary — set only on a partner, so its group is drawn
        from the same ``match_on`` values as the primary's group each batch. Empty ⇒ unconditional.
    in_memory
        Materialize this stream's selected (positive-weight) cells into RAM once, instead of re-reading the
        source each batch (for a small, frequently re-drawn pool such as a matched control).
    **sampler
        The read parameters :class:`SamplerKwargs` — ``batch_size`` / ``chunk_size`` / ``preload_nchunks``.
        All-or-nothing: pass all three to set them on this stream, or none to inherit the
        :class:`~scfit.data.Loader`'s.
    """

    def __init__(
        self,
        source: Source,
        *,
        group_by: Sequence[str],
        rep: RepKey | Sequence[RepKey] = "X",
        weights: Weights | None = None,
        match_on: Sequence[str] = (),
        in_memory: bool = False,
        **sampler: Unpack[SamplerKwargs],
    ) -> None:
        _check_sampler(sampler, "Stream")
        group_by = tuple(group_by)
        if not group_by:
            raise ValueError("Stream.group_by must be non-empty.")
        reps = rep if isinstance(rep, tuple | list) else (rep,)
        rep_locs = tuple(_rep_loc(r) for r in reps)
        if not rep_locs or any(not r for r in rep_locs):
            raise ValueError("Stream.rep must be one or more non-empty representation locations.")
        if weights is not None:
            for k in weights:
                if len(k) != len(group_by):
                    raise ValueError(f"weight key {k!r} arity != group_by {group_by}.")
            if any(w < 0 for w in weights.values()):
                raise ValueError("Stream.weights must be non-negative.")

        self.source = source
        self.group_by = group_by
        self.rep: tuple[str, ...] = rep_locs
        self.weights = weights
        self.match_on = tuple(match_on)
        self.in_memory = in_memory
        self.sampler: dict[str, int] = dict(sampler)  # {} (inherit) or all three (see _check_sampler)

    def __repr__(self) -> str:
        return (
            f"Stream(group_by={self.group_by}, rep={self.rep}, match_on={self.match_on}, "
            f"in_memory={self.in_memory}, weights={'uniform' if self.weights is None else 'custom'})"
        )
