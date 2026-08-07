"""``EvalLoader`` — the deterministic, full-coverage counterpart to :class:`~scfit.data.Loader`.

It reuses :class:`Loader`'s :class:`~scfit.data.Stream` + :class:`~scfit.data._source.Source` machinery;
only the sampler becomes an ordered leaf walk, so matching, unified sources and reps behave identically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import anndata as ad
import numpy as np

from scfit.data._io import _read_rows
from scfit.data._schema import _PRIMARY, Stream, Weights, validate_links
from scfit.data._source import Source, build_sources, resolve_source

__all__ = ["EvalLoader"]


def _positive(weights: Weights | None) -> set[tuple] | None:
    """The positive-weight leaves as a set — ``None`` when unweighted (keep every leaf)."""
    return None if weights is None else {tuple(k) for k, w in weights.items() if w > 0}


class EvalLoader:
    """Deterministic full-coverage (or deduplicated) matched pass over grouped data.

    Where :class:`~scfit.data.Loader` draws a stochastic per-batch schedule for training, this walks every
    primary **leaf once** in factorization order, yielding all its cells, each matched to its linked source
    by ``match_on``. Every group is covered and surfaced as ``batch["leaf"]`` — what shape inference and
    metrics want.

    Parameters
    ----------
    max_per_group
        Per-group cap that doubles as a dedup knob. :obj:`None` — every cell of every group (full eval
        against real target cells); ``N`` — at most ``N`` per group (all of them when fewer); ``1`` — one
        representative per group, i.e. the unique ``group_by`` combinations (a deduplicated dataset).
    subsample
        Which ``N``: ``"head"`` (first ``N`` in row order), ``"random"`` (a ``seed``-ed, per-seed
        reproducible draw), or a callable ``(rows, n, rng) -> rows``.
    """

    def __init__(
        self,
        sources: Mapping[str, ad.AnnData | list[ad.AnnData]],
        *,
        primary: Stream,
        links: Mapping[str, Stream] | None = None,
        to: str | None = None,
        max_per_group: int | None = None,
        subsample: str | Callable[[np.ndarray, int, np.random.Generator], np.ndarray] = "head",
        seed: int = 0,
    ) -> None:
        self._to = to
        self._max = max_per_group
        self._subsample = subsample
        self._seed = seed
        self._primary = primary
        self._links = dict(links or {})
        validate_links(primary, self._links)
        self._sources = build_sources(sources)
        self._union_cache: dict[tuple[str, ...], Source] = {}

        # Primary: factorize once -> per-cell codes + the ordered leaves (== unique group_by combinations).
        self._psrc = resolve_source(self._sources, primary, self._union_cache)
        self._pf = self._psrc.factorize(tuple(primary.group_by))
        # Optional selection: a primary weight of 0 (or an absent leaf) excludes that group, as for Loader.
        self._wanted = _positive(primary.weights)

        # Each link: bucket its (positive-weight) cells by ``match_on`` value, so a primary leaf resolves its
        # matched rows in O(1); ``match_on=()`` buckets everything under one unconditional key.
        self._lsrc: dict[str, Source] = {}
        self._lrows: dict[str, dict[tuple, np.ndarray]] = {}
        for name, link in self._links.items():
            src = resolve_source(self._sources, link, self._union_cache)
            self._lsrc[name] = src
            lf = src.factorize(tuple(link.group_by))
            keep = _positive(link.weights)
            match_pos = [link.group_by.index(c) for c in link.match_on]
            buckets: dict[tuple, list[np.ndarray]] = {}
            for code, leaf in enumerate(lf.leaves):
                if keep is not None and leaf not in keep:
                    continue
                key = tuple(leaf[p] for p in match_pos)
                buckets.setdefault(key, []).append(np.flatnonzero(lf.codes == code))
            self._lrows[name] = {key: np.sort(np.concatenate(rows)) for key, rows in buckets.items()}

    def _cap(self, rows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Subsample ``rows`` to at most ``max_per_group`` per group, per the ``subsample`` strategy."""
        if self._max is None or len(rows) <= self._max:
            return rows
        if self._subsample == "head":
            return rows[: self._max]
        if self._subsample == "random":
            return np.sort(rng.choice(rows, size=self._max, replace=False))
        return self._subsample(rows, self._max, rng)  # custom callable

    def _read(self, src: Source, reps: tuple[str, ...], rows: np.ndarray) -> dict[str, object]:
        """Read the given global ``rows`` for each rep of a stream (optionally as torch tensors).

        ``reps=()`` reads nothing and returns ``{}`` — a metadata-only stream, contributing just its leaf.
        On the primary that is a prediction pass with no known target state: every covariate combination
        enumerated, with linked streams still supplying real cells (e.g. matched controls) as model input.
        """
        arrays: dict[str, object] = {loc: _read_rows(src.adatas, loc, rows) for loc in reps}
        if self._to == "torch":
            import torch

            arrays = {loc: torch.as_tensor(np.asarray(a)) for loc, a in arrays.items()}
        return arrays

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> Iterator[dict]:
        """Yield one batch per primary leaf: ``{"primary": {loc: rows}, <link>: {...}, "leaf": <tuple>}``."""
        p_pos = {c: i for i, c in enumerate(self._primary.group_by)}
        rng = np.random.default_rng(self._seed)  # reset each pass -> reproducible subsamples for a fixed seed
        for code, leaf in enumerate(self._pf.leaves):
            if self._wanted is not None and leaf not in self._wanted:
                continue
            rows = self._cap(np.flatnonzero(self._pf.codes == code), rng)
            out: dict[str, object] = {_PRIMARY: self._read(self._psrc, self._primary.reps, rows), "leaf": leaf}
            for name, link in self._links.items():
                match_vals = tuple(leaf[p_pos[c]] for c in link.match_on)  # primary leaf's match_on values
                lrows = self._cap(self._lrows[name].get(match_vals, np.empty(0, dtype=int)), rng)
                out[name] = self._read(self._lsrc[name], link.reps, lrows)
            yield out

    def __len__(self) -> int:
        """Number of batches == number of selected primary groups."""
        if self._wanted is None:
            return len(self._pf.leaves)
        return sum(1 for leaf in self._pf.leaves if leaf in self._wanted)
