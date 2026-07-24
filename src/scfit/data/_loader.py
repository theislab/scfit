"""``Loader`` — streams matched batches from one primary :class:`Stream` plus named partner streams.

Each pass is a fresh epoch. The primary draws a per-batch class schedule ∝ its weights via an annbatch
:class:`~annbatch.samplers.ClassSampler`; every partner replays that schedule onto its own cells via a
:class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on its ``match_on`` columns (select via
the partner's weights + project via ``match_on``). A batch is ``{stream name: {rep loc: rows}}`` for the
primary and every partner, plus ``"annotations"``. Every sampler that must agree within a pass (the
schedule oracle, the primary's reps, each partner's inner) is a ``deepcopy`` of one seeded oracle, so they
stay in lockstep, a stream's reps read the same rows, and a pickled loader resumes the same stream.
See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import NamedTuple, Unpack

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not scfit's `Loader`)
from annbatch.samplers import BoundClassSampler, ClassSampler

from scfit.data._io import (
    get_from_container,
    is_backed_array,
    leaf_codes,
    materialize_node,
    obs_columns,
    open_source,
)
from scfit.data._schema import (
    _SAMPLER_KEYS,
    Container,
    SamplerKwargs,
    Stream,
    _check_sampler,
    weight_vector,
)

__all__ = ["Annotations", "Loader"]

# Per-stream annotations: ``{stream name: {leaf: {realm: array}}}`` — from a stream's leaf (its ``group_by``
# combination) to the named arrays for a batch drawn from it (e.g. the perturbation encoding). Keyed by the
# SAME leaf tuples as the stream's ``weights``. Each batch surfaces ``{stream name: the current leaf's
# arrays}`` under ``"annotations"`` for every stream given here.
type Annotations = Mapping[str, Mapping[tuple, Mapping[str, np.ndarray]]]

_PRIMARY = "primary"  # reserved name of the root / target stream


class _Read(NamedTuple):
    """Resolved read parameters for one stream: the merged :class:`SamplerKwargs` + the loader-global backend."""

    batch_size: int
    chunk_size: int
    preload_nchunks: int
    to: str | None


class Loader:
    """Yields ``{stream name: {rep loc: rows}}`` (+ ``"annotations"``) — the primary plus matched partners."""

    def __init__(
        self,
        primary: Stream,
        partners: Mapping[str, Stream] | None = None,
        *,
        seed: int = 0,
        annotations: Annotations | None = None,
        to: str | None = "torch",
        preload_to_gpu: bool = False,
        **sampler: Unpack[SamplerKwargs],
    ) -> None:
        _check_sampler(sampler, "Loader")
        partners = dict(partners or {})
        if _PRIMARY in partners:
            raise ValueError(f"partner name {_PRIMARY!r} is reserved for the primary stream.")
        self._streams: dict[str, Stream] = {_PRIMARY: primary, **partners}
        self._partners: list[str] = list(partners)
        self._annotations = annotations
        self._preload_to_gpu = preload_to_gpu

        # Resolve read config per stream: the stream's own sampler kwargs win; else the loader's; else error.
        self._cfg: dict[str, _Read] = {}
        for name, s in self._streams.items():
            eff = s.sampler or dict(sampler)
            if not eff:
                raise ValueError(
                    f"stream {name!r}: sampler kwargs {list(_SAMPLER_KEYS)} set on neither the Stream nor the Loader."
                )
            cfg = _Read(eff["batch_size"], eff["chunk_size"], eff["preload_nchunks"], to)
            if s.in_memory and cfg.chunk_size != 1:
                raise ValueError(
                    f"stream {name!r} is in_memory but chunk_size={cfg.chunk_size}: an in-memory stream is read "
                    "from RAM in one shot and must use chunk_size=1 (set it explicitly)."
                )
            self._cfg[name] = cfg
        self._root_batch_size = self._cfg[_PRIMARY].batch_size

        # Each partner is matched to the primary on its ``match_on`` columns (⊆ the columns they share).
        primary_cols = self._streams[_PRIMARY].group_by
        for name in self._partners:
            shared = set(primary_cols) & set(self._streams[name].group_by)
            if not set(self._streams[name].match_on) <= shared:
                raise ValueError(
                    f"stream {name!r} match_on {self._streams[name].match_on} must be ⊆ the columns it shares "
                    f"with the primary ({sorted(shared)})."
                )

        # Resolve each stream's source (opening a path backed, reading only the reps + cols it uses). Streams
        # are uniquely named, so each keeps its own resolved source and factorization — no cross-stream de-dup.
        self._sources: dict[str, Container] = {
            name: open_source(s.source, keys=sorted(s.rep), cols=sorted(s.group_by))
            for name, s in self._streams.items()
        }
        # (stream name, cols) -> full-obs (codes, leaves): construction-only scratch, emptied on pickle.
        self._leaf_cache: dict[tuple, tuple[np.ndarray, list[tuple]]] = {}

        # One independent sub-generator per stream (spawned off ``default_rng(seed)``); the samplers deepcopy
        # these rather than advance them, so a stream's oracle/target/partner samplers all start from the
        # identical state and stay in lockstep, and the whole stream is reproducible from the seed.
        rng = np.random.default_rng(seed)
        self._rngs: dict[str, np.random.Generator] = dict(
            zip(sorted(self._streams), rng.spawn(len(self._streams)), strict=True)
        )

        # Per stream: resolve source (materializing an in_memory stream into RAM) + build its tuple-labelled
        # categorical / weight vector / leaf list (obs only — no cell matrices). Streams over the same
        # (source, cols) share one factorization via ``_factorize``, so a big obs is never factorized twice.
        self._resolved: dict[str, Container] = {}
        self._st: dict[str, dict] = {}
        for name, s in self._streams.items():
            src, cats, w, leaves = self._prepare(name, s)
            self._resolved[name] = src
            self._st[name] = {"leaves": leaves, "w": w, "cats": cats}
        self._validate_annotations()  # strict: every named stream's positive-weight leaves must be covered

        # A natural epoch over the primary: its cell count // the primary's batch_size. The primary drives the
        # zip, so every stream draws the same number of batches (its own batch_size ⇒ its own row count).
        n_root_obs = len(self._st[_PRIMARY]["cats"])
        self._n_batches = max(1, n_root_obs // self._root_batch_size)

        # Oracle template: deepcopied into the primary's loader and each partner's inner, so their per-batch
        # class draws agree. All of a stream's reps share the stream's rng, so the (identical) samplers select
        # the same rows every batch — a stream's reps are aligned (same cells).
        self._oracle_sampler = self._new_class_sampler(_PRIMARY)
        self._loaders: dict[str, dict[str, AnnbatchLoader]] = {}
        self._add_stream_loaders(_PRIMARY, deepcopy(self._oracle_sampler))
        for name in self._partners:
            self._add_stream_loaders(name, self._new_bound_sampler(name))

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
        self._pos = 0

    # ── source access ──────────────────────────────────────────────────────
    def _factorize(self, name: str, cols: tuple[str, ...]) -> tuple[np.ndarray, list[tuple]]:
        """The stream source's ``(codes, leaves)`` over ``cols`` — read + factorized once, cached by (stream, cols)."""
        ck = (name, cols)
        if ck not in self._leaf_cache:
            self._leaf_cache[ck] = leaf_codes(obs_columns(self._sources[name], cols), cols)
        return self._leaf_cache[ck]

    def _prepare(self, name: str, s: Stream) -> tuple[Container, pd.Categorical, np.ndarray, list[tuple]]:
        """A stream's resolved source + its categorical, weight vector and leaf list (obs only).

        An ``in_memory`` stream is materialized into RAM here (positive-weight rows only), reusing the cached
        source factorization; a non-materialized stream reads straight from the resolved source.
        """
        codes, leaves = self._factorize(name, s.group_by)
        if s.in_memory:
            src, codes, leaves = materialize_node(self._sources[name], s, (codes, leaves))
        else:
            src = self._sources[name]
        return src, _flat_categorical(codes, leaves), weight_vector(s.weights, leaves), leaves

    # ── build ────────────────────────────────────────────────────────────
    def _new_class_sampler(self, name: str) -> ClassSampler:
        cfg = self._cfg[name]
        try:  # annbatch enforces its own run-length rule for chunk>1; forward with stream context
            return ClassSampler(
                chunk_size=cfg.chunk_size,
                preload_nchunks=cfg.preload_nchunks,
                batch_size=cfg.batch_size,
                classes=self._st[name]["cats"],
                num_samples=self._n_batches * cfg.batch_size,
                class_weights=self._st[name]["w"],
                drop_last=True,
                rng=deepcopy(self._rngs[name]),
            )
        except ValueError as e:
            raise ValueError(f"stream {name!r}: {e}") from e

    def _new_bound_sampler(self, name: str) -> BoundClassSampler:
        # Inner = a copy of the oracle, so the partner draws the same per-batch schedule. Match on the shared
        # ``match_on`` columns; the partner's leaf weights (0 for excluded leaves) go in as the *secondary*
        # ``classes`` so only positive-weight partner leaves are drawn within each matched context.
        cfg, cats = self._cfg[name], self._st[name]["cats"]
        primary, partner = self._streams[_PRIMARY], self._streams[name]
        try:
            return BoundClassSampler(
                deepcopy(self._oracle_sampler),
                cfg.chunk_size,
                cfg.preload_nchunks,
                cfg.batch_size,
                classes_to_bind_on=cats,
                # primary tuple position → partner tuple position, per shared ``match_on`` column
                on={primary.group_by.index(c): partner.group_by.index(c) for c in partner.match_on},
                classes=cats,
                class_weights=self._st[name]["w"],
                rng=deepcopy(self._rngs[name]),
            )
        except ValueError as e:
            raise ValueError(f"stream {name!r}: {e}") from e

    def _add_stream_loaders(self, name: str, sampler: ClassSampler | BoundClassSampler) -> None:
        # one annbatch Loader per rep of the stream, keyed by rep loc; all deepcopy the same sampler.
        src, s, to = self._resolved[name], self._streams[name], self._cfg[name].to
        loaders: dict[str, AnnbatchLoader] = {}
        for loc in s.rep:
            base = AnnbatchLoader(
                batch_sampler=deepcopy(sampler), return_index=False, to=to, preload_to_gpu=self._preload_to_gpu
            )
            backings = get_from_container(src, loc)
            # in-memory adata → add_adatas (obs-free X wrapper); backed adata / list of backed → add_datasets
            loaders[loc] = (
                base.add_adatas([ad.AnnData(X=b) for b in backings])
                if isinstance(src, ad.AnnData) and not is_backed_array(backings[0])
                else base.add_datasets(backings)
            )
        self._loaders[name] = loaders

    def _validate_annotations(self) -> None:
        """Strict coverage: every named stream's positive-weight leaf must have an annotation (no silent gaps)."""
        if self._annotations is None:
            return
        for name, stream_ann in self._annotations.items():
            if name not in self._st:
                raise ValueError(f"annotations reference unknown stream {name!r}; streams are {sorted(self._st)}.")
            leaves, weights = self._st[name]["leaves"], self._st[name]["w"]
            missing = [lf for lf, w in zip(leaves, weights, strict=True) if w > 0 and lf not in stream_ann]
            if missing:
                raise ValueError(f"annotations for stream {name!r} miss positive-weight leaves {missing}.")

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        Every sampler's RNG state is kept, so a reloaded loader resumes the same reproducible stream (the
        next pass) on the next ``__next__``; ``_iters`` is dropped and rebuilt, and the construction-only
        ``_leaf_cache`` is emptied (its codes are already carried by each stream's categorical).
        """
        state = self.__dict__.copy()
        state["_iters"] = None
        state["_leaf_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._iters = None  # force a fresh pass on the next `__next__`, using the restored sampler RNG

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> Loader:
        return self

    def _stream_next(self, name: str) -> tuple[dict, object]:
        """A stream's aligned reps ``{rep loc: rows}`` plus its per-batch ``comb`` (category label).

        Identical samplers pick the same aligned rows for every rep, and share one ``comb`` per batch.
        """
        reps, comb = {}, None
        for loc in self._streams[name].rep:
            batch = next(self._iters[name][loc])
            reps[loc], comb = batch["X"], batch["comb"]
        return reps, comb

    def _leaf_of(self, name: str, comb: object) -> tuple:
        """The stream's own leaf for this batch: the primary's ``comb`` as-is; a partner's is ``comb`` past ``match_on``."""
        leaf = tuple(comb)
        return leaf if name == _PRIMARY else leaf[len(self._streams[name].match_on) :]

    def __next__(self) -> dict[str, dict]:
        if self._iters is None or self._pos >= self._n_batches:
            # Start a fresh pass (first pass, next epoch, or resume after unpickling): (re)build one iterator
            # per stream/rep from the (advancing) sampler RNG, so each pass is a fresh reproducible epoch.
            self._iters = {
                name: {loc: iter(ld) for loc, ld in loaders.items()} for name, loaders in self._loaders.items()
            }
            self._pos = 0
        self._pos += 1

        # One entry per streamed stream, keyed by name: the primary is the target, each partner a source.
        out: dict = {}
        combs: dict = {}
        for name in (_PRIMARY, *self._partners):
            out[name], combs[name] = self._stream_next(name)
        if self._annotations is not None:
            # per named stream, look up its current leaf (from its `comb`) — strict coverage was checked at
            # construction, so every drawn leaf is present.
            out["annotations"] = {
                name: stream_ann[self._leaf_of(name, combs[name])] for name, stream_ann in self._annotations.items()
            }
        return out


def _flat_categorical(codes: np.ndarray, leaves: list[tuple]) -> pd.Categorical:
    """A tuple-labelled categorical: per-cell leaf code over ``leaves`` (categories are the leaf tuples).

    Tuple labels (not opaque integer codes) let :class:`~annbatch.samplers.BoundClassSampler` match a
    partner to the primary by the ``match_on`` columns — it projects the label by position.
    """
    categories = pd.MultiIndex.from_tuples(leaves).to_flat_index()
    return pd.Categorical.from_codes(codes, categories=categories)
