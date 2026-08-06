"""``Loader`` — streams matched batches from one primary :class:`Stream` plus named linked streams.

Each pass is a fresh epoch. The primary draws a per-batch class schedule ∝ its weights via an annbatch
:class:`~annbatch.samplers.ClassSampler`; every link replays that schedule onto its own cells via a
:class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on its ``match_on`` columns (select via
the link's weights + project via ``match_on``). A batch is ``{stream name: {rep loc: rows}}`` for the
primary and every link, plus ``"labels"`` (the current label's arrays for every stream with a
``label_lookup``). Every sampler that must agree within a pass (the schedule oracle, the primary's reps,
each link's inner) is a ``deepcopy`` of one seeded oracle, so they stay in lockstep, a stream's reps read
the same rows, and a pickled loader resumes the same stream.

Streams address their data by ``source_key`` into the ``sources`` mapping; each key resolves to one
:class:`~scfit.data.Source`, which owns that dataset's obs factorization (shared by every stream naming the
key). :meth:`Loader.from_paths` opens zarr path(s) backed — reading only the reps + cols the streams use —
and builds that mapping.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import NamedTuple, Unpack

import anndata as ad
import numpy as np
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not scfit's `Loader`)

# ClassSampler/BoundClassSampler + the loader's `label`/#256 behaviour are fork-only; the shim vendors them
# over stock annbatch==0.2.1 (and patches AnnbatchLoader.__iter__ on import) so scfit stays PyPI-publishable.
from scfit.data._annbatch_shim import BoundClassSampler, ClassSampler
from scfit.data._io import is_backed_array, open_source
from scfit.data._schema import _SAMPLER_KEYS, SamplerKwargs, Stream, _check_sampler, weight_vector
from scfit.data._source import Source

__all__ = ["Loader"]

_PRIMARY = "primary"  # reserved name of the root / target stream


class _SamplerCfg(NamedTuple):
    """Resolved read parameters for one stream: the merged :class:`SamplerKwargs` + the loader-global backend."""

    batch_size: int
    chunk_size: int
    preload_nchunks: int
    to: str | None


class Loader:
    """Yields ``{stream name: {rep loc: rows}}`` (+ ``"labels"``) — the primary plus its matched links."""

    def __init__(
        self,
        sources: Mapping[str, Source | ad.AnnData | list[ad.AnnData]],
        *,
        primary: Stream,
        links: Mapping[str, Stream] | None = None,
        seed: int = 0,
        to: str | None = None,
        preload_to_gpu: bool = False,
        **sampler_kwargs: Unpack[SamplerKwargs],
    ) -> None:
        _check_sampler(sampler_kwargs, "Loader")
        links = dict(links or {})
        if _PRIMARY in links:
            raise ValueError(f"link name {_PRIMARY!r} is reserved for the primary stream.")
        self._streams: dict[str, Stream] = {_PRIMARY: primary, **links}
        self._links: list[str] = list(links)
        self._preload_to_gpu = preload_to_gpu

        # Each source_key resolves to one Source (a dataset + its factorization cache). Streams naming the
        # same key share the Source, so its obs is factorized once (a primary and its matched control).
        self._sources: dict[str, Source] = {
            k: v if isinstance(v, Source) else Source(v) for k, v in dict(sources).items()
        }
        for name, s in self._streams.items():
            if s.source_key not in self._sources:
                raise ValueError(
                    f"stream {name!r}: source_key {s.source_key!r} not in sources {sorted(self._sources)}."
                )

        # Resolve sampler config per stream: the stream's own sampler kwargs win; else the loader's; else error.
        self._cfg: dict[str, _SamplerCfg] = {}
        for name, s in self._streams.items():
            eff = s.sampler_kwargs or dict(sampler_kwargs)
            if not eff:
                raise ValueError(
                    f"stream {name!r}: sampler kwargs {list(_SAMPLER_KEYS)} set on neither the Stream nor the Loader."
                )
            cfg = _SamplerCfg(eff["batch_size"], eff["chunk_size"], eff["preload_nchunks"], to)
            if s.in_memory and cfg.chunk_size != 1:
                raise ValueError(
                    f"stream {name!r} is in_memory but chunk_size={cfg.chunk_size}: an in-memory stream is read "
                    "from RAM in one shot and must use chunk_size=1 (set it explicitly)."
                )
            self._cfg[name] = cfg
        self._root_batch_size = self._cfg[_PRIMARY].batch_size

        # Each link is matched to the primary on its ``match_on`` columns (⊆ the columns they share).
        primary_cols = self._streams[_PRIMARY].group_by
        for name in self._links:
            shared = set(primary_cols) & set(self._streams[name].group_by)
            if not set(self._streams[name].match_on) <= shared:
                raise ValueError(
                    f"stream {name!r} match_on {self._streams[name].match_on} must be ⊆ the columns it shares "
                    f"with the primary ({sorted(shared)})."
                )

        # One independent sub-generator per stream (spawned off ``default_rng(seed)``); the samplers deepcopy
        # these rather than advance them, so a stream's oracle/target/link samplers all start from the identical
        # state and stay in lockstep, and the whole stream is reproducible from the seed.
        rng = np.random.default_rng(seed)
        self._rngs: dict[str, np.random.Generator] = dict(
            zip(sorted(self._streams), rng.spawn(len(self._streams)), strict=True)
        )

        # Per stream: resolve its Source (materializing an in_memory stream into RAM) and pull that source's
        # tuple-labelled categorical / weight vector / leaf list (obs only — no cell matrices). The Source
        # factorizes each ``(source_key, group_by)`` once, so streams sharing a key reuse it.
        self._resolved: dict[str, Source] = {}
        self._st: dict[str, dict] = {}
        for name, s in self._streams.items():
            base = self._sources[s.source_key]
            src = base.materialize(s) if s.in_memory else base
            f = src.factorize(s.group_by)
            self._resolved[name] = src
            self._st[name] = {"leaves": f.leaves, "w": weight_vector(s.weights, f.leaves), "cats": f.cats}
        self._validate_label_lookups()  # strict: a stream's label_lookup must cover its positive-weight leaves

        # A natural epoch over the primary: its cell count // the primary's batch_size. The primary drives the
        # zip, so every stream draws the same number of batches (its own batch_size ⇒ its own row count).
        n_root_obs = len(self._st[_PRIMARY]["cats"])
        self._n_batches = max(1, n_root_obs // self._root_batch_size)

        # Oracle template: deepcopied into the primary's loader and each link's inner, so their per-batch class
        # draws agree. All of a stream's reps share the stream's rng, so the (identical) samplers select the
        # same rows every batch — a stream's reps are aligned (same cells).
        self._oracle_sampler = self._new_class_sampler(_PRIMARY)
        self._loaders: dict[str, dict[str, AnnbatchLoader]] = {
            _PRIMARY: self._build_per_rep_loaders(_PRIMARY, deepcopy(self._oracle_sampler))
        }
        for name in self._links:
            # match_on set → bound to the primary's per-batch class; match_on=() → an independent
            # (unconditional) draw from the link's own weights.
            sampler = self._new_bound_sampler(name) if self._streams[name].match_on else self._new_class_sampler(name)
            self._loaders[name] = self._build_per_rep_loaders(name, sampler)

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
        self._pos = 0

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def from_paths(
        cls,
        paths: Mapping[str, str | list[str]],
        *,
        primary: Stream,
        links: Mapping[str, Stream] | None = None,
        seed: int = 0,
        to: str | None = None,
        preload_to_gpu: bool = False,
        **sampler_kwargs: Unpack[SamplerKwargs],
    ) -> Loader:
        """Build a :class:`Loader` from zarr ``{source_key: path | [paths]}``, opened backed.

        Each path is opened reading only the **union** of the reps and ``group_by`` columns the streams
        naming that ``source_key`` actually use (via :func:`~scfit.data._io.open_source`) — so obs columns
        nobody groups on are never touched. Everything else matches :meth:`__init__`.
        """
        streams = {_PRIMARY: primary, **dict(links or {})}
        reps: dict[str, set] = {}
        cols: dict[str, set] = {}
        for s in streams.values():
            reps.setdefault(s.source_key, set()).update(s.reps)
            cols.setdefault(s.source_key, set()).update(s.group_by)
        sources = {
            k: open_source(p, keys=sorted(reps.get(k, ())), cols=sorted(cols.get(k, ())))
            for k, p in dict(paths).items()
        }
        return cls(
            sources, primary=primary, links=links, seed=seed, to=to, preload_to_gpu=preload_to_gpu, **sampler_kwargs
        )

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
        # Inner = a copy of the oracle, so the link draws the same per-batch schedule. Match on the shared
        # ``match_on`` columns; the link's leaf weights (0 for excluded leaves) go in as the *secondary*
        # ``classes`` so only positive-weight link leaves are drawn within each matched context.
        cfg, cats = self._cfg[name], self._st[name]["cats"]
        primary, link = self._streams[_PRIMARY], self._streams[name]
        try:
            return BoundClassSampler(
                deepcopy(self._oracle_sampler),
                cfg.chunk_size,
                cfg.preload_nchunks,
                cfg.batch_size,
                classes_to_bind_on=cats,
                # primary tuple position → link tuple position, per shared ``match_on`` column
                on={primary.group_by.index(c): link.group_by.index(c) for c in link.match_on},
                classes=cats,
                class_weights=self._st[name]["w"],
                rng=deepcopy(self._rngs[name]),
            )
        except ValueError as e:
            raise ValueError(f"stream {name!r}: {e}") from e

    def _build_per_rep_loaders(self, name: str, sampler: ClassSampler | BoundClassSampler) -> dict[str, AnnbatchLoader]:
        # one annbatch Loader per rep of the stream, keyed by rep loc; all deepcopy the same sampler.
        src, s, to = self._resolved[name], self._streams[name], self._cfg[name].to
        loaders: dict[str, AnnbatchLoader] = {}
        for loc in s.reps:
            base = AnnbatchLoader(
                batch_sampler=deepcopy(sampler), return_index=False, to=to, preload_to_gpu=self._preload_to_gpu
            )
            backings = src.rep(loc)
            # in-memory arrays → add_adatas (obs-free X wrapper); backed (zarr / CSRDataset) → add_datasets
            loaders[loc] = (
                base.add_adatas([ad.AnnData(X=b) for b in backings])
                if not is_backed_array(backings[0])
                else base.add_datasets(backings)
            )
        return loaders

    def _validate_label_lookups(self) -> None:
        """Strict coverage: a stream's ``label_lookup`` must cover its positive-weight leaves (no silent gaps)."""
        for name, s in self._streams.items():
            if s.label_lookup is None:
                continue
            leaves, weights = self._st[name]["leaves"], self._st[name]["w"]
            missing = [lf for lf, w in zip(leaves, weights, strict=True) if w > 0 and lf not in s.label_lookup]
            if missing:
                raise ValueError(f"stream {name!r}: label_lookup misses positive-weight leaves {missing}.")

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        Every sampler's RNG state is kept, so a reloaded loader resumes the same reproducible stream (the
        next pass) on the next ``__next__``; ``_iters`` is dropped and rebuilt from the restored samplers.
        """
        state = self.__dict__.copy()
        state["_iters"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._iters = None  # force a fresh pass on the next `__next__`, using the restored sampler RNG

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> Loader:
        return self

    def _stream_next(self, name: str) -> tuple[dict, object]:
        """A stream's aligned reps ``{rep loc: rows}`` plus its per-batch ``label`` (category label).

        Identical samplers pick the same aligned rows for every rep, and share one ``label`` per batch.
        """
        reps, label = {}, None
        for loc in self._streams[name].reps:
            batch = next(self._iters[name][loc])
            reps[loc], label = batch["X"], batch["label"]
        return reps, label

    def _leaf_of(self, name: str, label: object) -> tuple:
        """The stream's own leaf: the primary's ``label`` as-is; a link's is ``label`` past ``match_on``."""
        leaf = tuple(label)
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

        # One entry per streamed stream, keyed by name: the primary is the target, each link a source.
        out: dict = {}
        label_of: dict = {}
        for name in (_PRIMARY, *self._links):
            out[name], label_of[name] = self._stream_next(name)
        # per stream with a label_lookup, surface its current label's arrays (strict coverage checked at build).
        labels = {
            name: s.label_lookup[self._leaf_of(name, label_of[name])]
            for name, s in self._streams.items()
            if s.label_lookup is not None
        }
        if labels:
            out["labels"] = labels
        return out
