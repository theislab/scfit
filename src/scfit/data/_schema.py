"""Public data spec: the :class:`Stream` that :class:`~scfit.data.Loader` and
:class:`~scfit.data.EvalLoader` consume, plus the read parameters they share. See ``README.md`` for the
model and the cellflow / sc-flow-tools mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Required, TypedDict

import anndata as ad
import numpy as np

from scfit.params import Default

type Container = ad.AnnData | list[ad.AnnData]

# The selection: {group -> weight} (a group is a ``group_by`` tuple). A group absent, or with weight 0, is
# excluded — that IS the selection, native to annbatch's ClassSampler. ``None`` means uniform over every group.
Weights = Mapping[tuple, float]

__all__ = ["Container", "SamplerParams", "Stream", "Weights", "weight_vector"]

# Reserved name of the root / target (primary) stream — shared by Loader and EvalLoader.
_PRIMARY = "primary"


class SamplerParams(TypedDict, total=False):
    """annbatch's sampler settings for one stream, passed as ``sampler=`` to :class:`~scfit.data.Loader`."""

    batch_size: Required[int]
    """Rows per emitted batch for the stream. Source and target row counts need not match."""
    preload_nchunks: Required[int]
    """Chunks per annbatch read window. annbatch validates the three sizes together."""
    chunk_size: Annotated[int, Default(1)]
    """annbatch read-slice size. ``1`` reads per row (any layout); ``>1`` reads contiguous chunks, so each
    sampled leaf must sit in a contiguous run of at least ``chunk_size``."""


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

    The single public unit :class:`~scfit.data.Loader` consumes. It partitions its source's cells into
    **leaves** — the unique ``group_by`` combinations — and those are what gets weighted, sampled, and
    reported back per batch. A Stream passed in ``links=`` (with a ``match_on``) is matched batch-for-batch
    to the primary on the shared ``match_on`` values.

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
        tuple of them for several **aligned** reps of the same cells. ``()`` is **metadata-only** — no cell
        matrix is read and the stream contributes only its leaf, for a prediction pass over covariate
        combinations with no known target state (:class:`~scfit.data.EvalLoader` only).
    weights
        ``{group: weight}`` (a group is a ``group_by`` tuple); a group absent or with weight 0 is excluded.
        :obj:`None` (default) is uniform over every group present.
    match_on
        Columns a *linked* stream shares with the primary — set only on a link, so its group is drawn from
        the same ``match_on`` values as the primary's group each batch. Empty ⇒ unconditional.
    in_memory
        Materialize this stream's selected (positive-weight) cells into RAM once, instead of re-reading the
        source each batch (for a small, frequently re-drawn pool such as a matched control).
    """

    def __init__(
        self,
        source_key: str | Sequence[str],
        *,
        group_by: Sequence[str],
        reps: str | Sequence[str] = "X",
        weights: Weights | None = None,
        match_on: Sequence[str] = (),
        in_memory: bool = False,
    ) -> None:
        keys = (source_key,) if isinstance(source_key, str) else tuple(source_key)
        if not keys or not all(isinstance(k, str) and k for k in keys):
            raise ValueError("Stream.source_key must be a non-empty string, or a non-empty sequence of them.")
        if len(set(keys)) != len(keys):
            raise ValueError(f"Stream.source_key has duplicate keys: {source_key!r}.")
        group_by = tuple(group_by)
        if not group_by:
            raise ValueError("Stream.group_by must be non-empty.")
        if isinstance(reps, str):
            reps = (reps,)
        elif isinstance(reps, tuple | list):
            reps = tuple(reps)  # `()` is legal: a metadata-only stream (see the `reps` docstring)
        else:
            raise ValueError(f"Stream.reps must be a loc string or a sequence of loc strings (got {reps!r}).")
        if not all(isinstance(r, str) and r for r in reps):
            raise ValueError('Stream.reps entries must be non-empty loc strings ("X" / "obsm/<k>" / …).')
        if weights is not None:
            for k in weights:
                if len(k) != len(group_by):
                    raise ValueError(f"weight key {k!r} arity != group_by {group_by}.")
            if any(w < 0 for w in weights.values()):
                raise ValueError("Stream.weights must be non-negative.")

        self.source_key = source_key
        self.source_keys: tuple[str, ...] = keys  # normalized; one, or several to unify over
        self.group_by = group_by
        self.reps: tuple[str, ...] = reps
        self.weights = weights
        self.match_on = tuple(match_on)
        self.in_memory = in_memory


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
