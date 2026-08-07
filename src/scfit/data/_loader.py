"""``Loader`` — streams matched batches from one primary :class:`Stream` plus named linked streams.

The loader is an infinite stream, rolling a fresh epoch each pass; :meth:`Loader.set_n_iters` sets a pause
point (a StopIteration every ``n_iters`` batches) without perturbing the schedule, so consecutive passes
resume rather than replay. The primary draws a per-batch class schedule ∝ its weights via an annbatch
:class:`~annbatch.samplers.ClassSampler`; every link replays that schedule onto its own cells via a
:class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on its ``match_on`` columns (select via
the link's weights + project via ``match_on``). A batch is ``{stream name: {rep loc: rows}}`` for the
primary and every link, plus ``"leaves"`` — ``{stream name: <group_by tuple>}``, the group each stream drew.
Side arrays keyed by group (a perturbation encoding, a dose vector) are the *consumer's*: index them with
that leaf. Every sampler that must agree within a pass (the schedule oracle, the primary's reps,
each link's inner) is a ``deepcopy`` of one seeded oracle, so they stay in lockstep, a stream's reps read
the same rows, and a pickled loader resumes the same stream.

Streams address their data by ``source_key`` into the ``sources`` mapping; each key resolves to one
``Source`` (an internal wrapper) that owns that dataset's obs factorization, shared by every stream naming
the key. :meth:`Loader.from_paths` opens zarr path(s) backed — reading only the reps + cols the streams use —
and builds that mapping.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import Unpack

import anndata as ad
import numpy as np
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not scfit's `Loader`)

# ClassSampler/BoundClassSampler + the loader's `label`/#256 behaviour are fork-only; the shim vendors them
# over stock annbatch==0.2.1 (and patches AnnbatchLoader.__iter__ on import) so scfit stays PyPI-publishable.
from scfit.data._annbatch_shim import BoundClassSampler, ClassSampler
from scfit.data._io import is_backed_array, open_source
from scfit.data._schema import (
    _PRIMARY,
    _SAMPLER_KEYS,
    SamplerKwargs,
    Stream,
    _check_sampler,
    validate_links,
    weight_vector,
)
from scfit.data._source import Source, build_sources, resolve_source

__all__ = ["Loader"]


class Loader:
    """Yields ``{stream name: {rep loc: rows}}`` (+ ``"leaves"``) — the primary plus its matched links."""

    def __init__(
        self,
        sources: Mapping[str, ad.AnnData | list[ad.AnnData]],
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
        validate_links(primary, links)
        self._streams: dict[str, Stream] = {_PRIMARY: primary, **links}
        self._links: list[str] = list(links)
        self._preload_to_gpu = preload_to_gpu
        self._to = to  # loader-global backend for every stream's per-rep annbatch loaders

        self._sources: dict[str, Source] = build_sources(sources)
        for name, s in self._streams.items():
            for k in s.source_keys:
                if k not in self._sources:
                    raise ValueError(f"stream {name!r}: source_key {k!r} not in sources {sorted(self._sources)}.")

        # Resolve sampler kwargs per stream: the stream's own win; else the loader's; else error.
        self._cfg: dict[str, dict[str, int]] = {}
        for name, s in self._streams.items():
            if not s.reps:
                # ponytail: no metadata-only training stream — the per-rep annbatch loaders ARE what advances
                # the sampler, so a repless stream would yield {} forever (and no label). Wire the sampler up
                # standalone if a label-only training stream ever has a use.
                raise ValueError(
                    f"stream {name!r} has no reps: metadata-only streams are EvalLoader-only (the training "
                    "Loader streams cells). Give it a rep, or iterate it with EvalLoader."
                )
            eff = dict(s.sampler_kwargs or sampler_kwargs)
            if not eff:
                raise ValueError(
                    f"stream {name!r}: sampler kwargs {list(_SAMPLER_KEYS)} set on neither the Stream nor the Loader."
                )
            if s.in_memory and eff["chunk_size"] != 1:
                raise ValueError(
                    f"stream {name!r} is in_memory but chunk_size={eff['chunk_size']}: an in-memory stream is read "
                    "from RAM in one shot and must use chunk_size=1 (set it explicitly)."
                )
            self._cfg[name] = eff
        self._root_batch_size = self._cfg[_PRIMARY]["batch_size"]

        # One RNG per stream, spawned off default_rng(seed). Samplers deepcopy (never advance) these, so a
        # stream's oracle/target/link samplers start identical, stay in lockstep, and are seed-reproducible.
        rng = np.random.default_rng(seed)
        self._rngs: dict[str, np.random.Generator] = dict(
            zip(sorted(self._streams), rng.spawn(len(self._streams)), strict=True)
        )

        # Per stream: resolve its Source (materializing an in_memory stream), then pull its categorical /
        # weights / leaves from obs alone — no cell matrices. Sources sharing a key factorize once.
        self._union_cache: dict[tuple[str, ...], Source] = {}
        self._resolved: dict[str, Source] = {}
        self._st: dict[str, dict] = {}
        for name, s in self._streams.items():
            base = resolve_source(self._sources, s, self._union_cache)
            src = base.materialize(s) if s.in_memory else base
            f = src.factorize(s.group_by)
            self._resolved[name] = src
            self._st[name] = {"leaves": f.leaves, "w": weight_vector(s.weights, f.leaves), "cats": f.cats}

        # Epoch length: primary rows // its batch_size. The underlying stream is *always* infinite — it rolls
        # a fresh epoch of this length each pass — and ``n_iters`` never touches the samplers, so the schedule
        # is independent of any cap.
        n_root_obs = len(self._st[_PRIMARY]["cats"])
        self.n_batches = self._pass_len = max(1, n_root_obs // self._root_batch_size)
        self.n_iters: int | None = None

        # The primary's sampler is the oracle; the primary's reps and each link's inner all deepcopy it.
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
        self._pos = 0  # position within the current epoch (drives the epoch roll)
        self._since_pause = 0  # batches yielded since the last StopIteration, vs ``n_iters``

    def set_n_iters(self, n_iters: int | None) -> Loader:
        """Yield ``n_iters`` batches per pass (``None`` ⇒ never stop). Returns ``self``.

        This is a *pause point*, not a restart: the underlying stream is one infinite schedule, so after a
        pass ends the next ``for`` loop resumes where it left off rather than replaying the same batches.
        """
        if n_iters is not None and n_iters < 1:
            raise ValueError(f"n_iters must be a positive integer or None (got {n_iters!r}).")
        self.n_iters = n_iters
        self._since_pause = 0
        return self

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
            for k in s.source_keys:
                reps.setdefault(k, set()).update(s.reps)
                cols.setdefault(k, set()).update(s.group_by)
        sources = {
            k: open_source(p, keys=sorted(reps.get(k, ())), cols=sorted(cols.get(k, ())))
            for k, p in dict(paths).items()
        }
        return cls(
            sources,
            primary=primary,
            links=links,
            seed=seed,
            to=to,
            preload_to_gpu=preload_to_gpu,
            **sampler_kwargs,
        )

    def _new_class_sampler(self, name: str) -> ClassSampler:
        cfg = self._cfg[name]
        try:  # annbatch enforces its own run-length rule for chunk>1; forward with stream context
            return ClassSampler(
                chunk_size=cfg["chunk_size"],
                preload_nchunks=cfg["preload_nchunks"],
                batch_size=cfg["batch_size"],
                classes=self._st[name]["cats"],
                num_samples=self._pass_len * cfg["batch_size"],
                class_weights=self._st[name]["w"],
                drop_last=True,
                rng=deepcopy(self._rngs[name]),
            )
        except ValueError as e:
            raise ValueError(f"stream {name!r}: {e}") from e

    def _new_bound_sampler(self, name: str) -> BoundClassSampler:
        # Inner = a deepcopy of the oracle (same per-batch schedule). Bind on the shared ``match_on`` columns;
        # the link's own weights (0 = excluded) pick which of its leaves is drawn within each matched context.
        cfg, cats = self._cfg[name], self._st[name]["cats"]
        primary, link = self._streams[_PRIMARY], self._streams[name]
        try:
            return BoundClassSampler(
                deepcopy(self._oracle_sampler),
                cfg["chunk_size"],
                cfg["preload_nchunks"],
                cfg["batch_size"],
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
        # One annbatch Loader per rep; each deepcopies the sampler so the reps stay aligned.
        src, s, to = self._resolved[name], self._streams[name], self._to
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

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        The sampler RNG state is kept, so the next ``__next__`` rebuilds a fresh pass resuming the same stream.
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

    # ponytail: no __len__ — the loader is an infinite stream that ``n_iters`` merely pauses, and a len()
    # that raises half the time is worse than none. Read ``n_iters`` / ``n_batches`` directly if you need a count.

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
        if self.n_iters is not None and self._since_pause >= self.n_iters:
            self._since_pause = 0  # pause here; the next pass resumes the same stream, it does not replay
            raise StopIteration
        if self._iters is None or self._pos >= self._pass_len:
            # (Re)build one iterator per stream/rep from the advancing sampler RNG — a fresh reproducible
            # epoch (first pass, next epoch, or resume after unpickling). Independent of ``n_iters``: a pause
            # mid-epoch leaves ``_pos`` alone, so the next pass picks the epoch up where it stopped.
            self._iters = {
                name: {loc: iter(ld) for loc, ld in loaders.items()} for name, loaders in self._loaders.items()
            }
            self._pos = 0
        self._pos += 1
        self._since_pause += 1

        # One entry per streamed stream, keyed by name: the primary is the target, each link a source.
        out: dict = {}
        leaves: dict = {}
        for name in (_PRIMARY, *self._links):
            out[name], label = self._stream_next(name)
            leaves[name] = self._leaf_of(name, label)
        out["leaves"] = leaves  # the group each stream drew this batch — index your own encodings with it
        return out
