r"""Public data spec: the :class:`Stream` consumed by :class:`~scfit.data.Loader`.

Everything is a :class:`Stream` — one streamed population over a source, described by the columns it
groups on, the representation(s) it reads, its per-group weights, an optional per-group array lookup, and
(for a matched *link*) the columns it shares with the primary. There is no separate node / bind / scheme
object: the loader takes one primary :class:`Stream` plus any number of named linked :class:`Stream`\\s
and wires them into annbatch samplers directly.

A Stream partitions its source's cells into **leaves** (unique combinations of ``group_by``) with a
per-combination :data:`Weights` mapping. A weight of 0 (or a combination absent from the mapping) is
*excluded* — that IS the selection, native to annbatch's ``ClassSampler``. ``weights=None`` is uniform
over every group. See ``README.md`` for the model and the cellflow / sc-flow-tools mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict, Unpack

import anndata as ad
import numpy as np

type Container = ad.AnnData | list[ad.AnnData]

# The selection: a mapping {group -> weight} (a group is a ``group_by`` tuple). A group absent / weight 0
# is excluded — that IS the selection. ``None`` means uniform over every group.
Weights = Mapping[tuple, float]

__all__ = ["Container", "SamplerKwargs", "Stream", "Weights", "weight_vector"]

# The annbatch read parameters, defined once (via ``Unpack[SamplerKwargs]``) for both Stream and Loader.
_SAMPLER_KEYS = ("batch_size", "chunk_size", "preload_nchunks")

# Reserved name of the root / target (primary) stream — shared by Loader and EvalLoader.
_PRIMARY = "primary"


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

    The single public unit :class:`~scfit.data.Loader` consumes. A Stream passed in ``links=`` (with a
    ``match_on``) is matched batch-for-batch to the primary on the shared ``match_on`` values.

    Parameters
    ----------
    source_key
        Which dataset(s) this stream samples — a key (or a sequence of keys) into the ``sources`` mapping
        given to :class:`~scfit.data.Loader` (``{source_key: list[AnnData]}``). Several streams may name the
        same key (a primary and its matched control over one dataset), and the dataset's obs is factorized
        once and shared across them. Passing **several** keys unifies those datasets into one categorical
        universe (their cells concatenated), so the stream samples across all of them with a single sampler —
        each leaf still resolves to whichever dataset holds it. Unified datasets must share the streamed
        rep's feature dimension (``shape[1]``).
    group_by
        Columns whose unique combinations define the groups sampled (the leaves).
    reps
        Representation location(s) to stream — a loc string ``"X"`` / ``"obsm/<k>"`` / ``"layers/<k>"``, or a
        tuple of them for several **aligned** reps of the same cells.
    weights
        ``{group: weight}`` (a group is a ``group_by`` tuple); a group absent or with weight 0 is excluded.
        :obj:`None` (default) is uniform over every group present.
    label_lookup
        Optional ``{label: {realm: array}}`` — per-label side arrays (e.g. a perturbation encoding), where a
        *label* is a ``group_by`` combination. Each batch surfaces this stream's current label's arrays under
        ``batch["labels"][<stream name>]``. Every positive-weight label must be present (checked at
        :class:`~scfit.data.Loader` construction).
    match_on
        Columns a *linked* stream shares with the primary — set only on a link, so its group is drawn from
        the same ``match_on`` values as the primary's group each batch. Empty ⇒ unconditional.
    in_memory
        Materialize this stream's selected (positive-weight) cells into RAM once, instead of re-reading the
        source each batch (for a small, frequently re-drawn pool such as a matched control).
    **sampler_kwargs
        The read parameters :class:`SamplerKwargs` — ``batch_size`` / ``chunk_size`` / ``preload_nchunks``.
        All-or-nothing: pass all three to set them on this stream, or none to inherit the
        :class:`~scfit.data.Loader`'s.
    """

    def __init__(
        self,
        source_key: str | Sequence[str],
        *,
        group_by: Sequence[str],
        reps: str | Sequence[str] = "X",
        weights: Weights | None = None,
        label_lookup: Mapping[tuple, Mapping[str, np.ndarray]] | None = None,
        match_on: Sequence[str] = (),
        in_memory: bool = False,
        **sampler_kwargs: Unpack[SamplerKwargs],
    ) -> None:
        _check_sampler(sampler_kwargs, "Stream")
        keys = (source_key,) if isinstance(source_key, str) else tuple(source_key)
        if not keys or not all(isinstance(k, str) and k for k in keys):
            raise ValueError("Stream.source_key must be a non-empty string, or a non-empty sequence of them.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Stream.source_key has duplicate keys: {source_key!r}.")
        group_by = tuple(group_by)
        if not group_by:
            raise ValueError("Stream.group_by must be non-empty.")
        reps = (reps,) if isinstance(reps, str) else tuple(reps) if isinstance(reps, tuple | list) else ()
        if not reps or not all(isinstance(r, str) and r for r in reps):
            raise ValueError('Stream.reps must be one or more non-empty loc strings ("X" / "obsm/<k>" / …).')
        if weights is not None:
            for k in weights:
                if len(k) != len(group_by):
                    raise ValueError(f"weight key {k!r} arity != group_by {group_by}.")
            if any(w < 0 for w in weights.values()):
                raise ValueError("Stream.weights must be non-negative.")
        if label_lookup is not None:
            for k in label_lookup:
                if len(k) != len(group_by):
                    raise ValueError(f"label_lookup key {k!r} arity != group_by {group_by}.")

        self.source_key = source_key
        self.source_keys: tuple[str, ...] = keys  # normalized; one, or several to unify over
        self.group_by = group_by
        self.reps: tuple[str, ...] = reps
        self.weights = weights
        self.label_lookup = label_lookup
        self.match_on = tuple(match_on)
        self.in_memory = in_memory
        self.sampler_kwargs: dict[str, int] = dict(sampler_kwargs)  # {} (inherit) or all three (see _check_sampler)


def validate_links(primary: Stream, links: Mapping[str, Stream]) -> None:
    """Reserved-name + ``match_on`` ⊆ shared-columns checks for a primary and its links.

    Shared by :class:`~scfit.data.Loader` and :class:`~scfit.data.EvalLoader`.
    """
    if _PRIMARY in links:
        raise ValueError(f"link name {_PRIMARY!r} is reserved for the primary stream.")
    for name, link in links.items():
        shared = set(primary.group_by) & set(link.group_by)
        if not set(link.match_on) <= shared:
            raise ValueError(
                f"stream {name!r} match_on {link.match_on} must be ⊆ the columns it shares "
                f"with the primary ({sorted(shared)})."
            )
