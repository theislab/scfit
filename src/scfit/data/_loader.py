"""``Loader`` — streams matched per-node batches from a :class:`Scheme`, keyed by node name.

Each pass is a fresh epoch. The root (target) node draws a per-batch class schedule ∝ its weights via an
annbatch :class:`~annbatch.samplers.ClassSampler`; every bound child replays that schedule onto its own
cells via an annbatch :class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on the bind's
``common`` columns (select via child weights + project via ``common``). A batch is ``{node name: {rep
loc: rows}}`` for the root and every bound child, plus ``"condition"`` — the consumer reads the target
from ``scheme.root`` and the sources from the bound children. Every sampler that must agree within a pass
(the schedule oracle, the target's reps, and each child's inner) is a ``deepcopy`` of the same seeded
oracle state, so they stay in lockstep, a node's reps read the same rows, and a pickled loader resumes
the exact same stream. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not binded's `Loader`)
from annbatch.samplers import BoundClassSampler, ClassSampler

from scfit.data._backend import _attach, _SchemeReader
from scfit.data._condition import ConditionLookup, _condition_from_lookup
from scfit.data._schema import _resolve_config_map

__all__ = ["Loader"]

if TYPE_CHECKING:
    from scfit.data._schema import Bind, Container, SamplerConfig, Scheme


class Loader(_SchemeReader):
    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
        condition_lookup: ConditionLookup | None = None,
        *,
        preload_to_gpu: bool = False,
    ) -> None:
        super().__init__()  # the shared per-(source, cols) obs-factorization cache (`_factorize`)
        self.scheme = scheme
        self._condition_lookup = condition_lookup
        self._preload_to_gpu = preload_to_gpu
        self._cfg = self._resolve_configs(sampler_config)
        self._root_batch_size = self._cfg[
            self.scheme.root
        ].batch_size  # root/target batch size — drives the pass length

        # root's direct children are the bound sources (parity with the previous loader: depth-1 sources)
        self._child_binds: list[Bind] = [b for b in scheme.binds if b.parent == self.scheme.root]

        # one independent sub-generator per node (spawned off a fresh local `default_rng(scheme.seed)`, so
        # every loader from the same seed gets the same set), so nodes don't correlate. These stay pristine —
        # samplers `deepcopy` them rather than advance them in place, so a node's oracle/target/child samplers
        # all start from the identical state and stay in lockstep; each pass advances the copies, not these.
        rng = np.random.default_rng(scheme.seed)
        self._node_rngs: dict[str, np.random.Generator] = dict(
            zip(sorted(scheme.nodes), rng.spawn(len(scheme.nodes)), strict=True)
        )

        # Resolve each node's source (materializing a `Node.in_memory` node into RAM, see _io) and build its
        # leaf partition + weights + tuple-labelled categorical (obs only — no cell matrices). `_prepare`
        # (on the base) factorizes each scheme source's obs ONCE per (source, cols): the perturbed root and
        # its matched-control child over the same source share that single factorization — the control's
        # in-RAM row selection reuses it — so the (possibly 100M-row) obs is never factorized twice.
        self._nodes: dict[str, Container] = {}
        self._st: dict[str, dict] = {}
        for name, node in scheme.nodes.items():
            src, cats, w, leaves = self._prepare(scheme, node)
            self._nodes[name] = src
            self._st[name] = {"node": node, "leaves": leaves, "w": w, "cats": cats}

        # A natural epoch over the root (target) node: its cell count // the root's batch_size. The root
        # drives the zip, so every node draws the same number of batches; each node's num_samples ==
        # _n_batches * that node's batch_size (source rows need not equal target rows).
        n_root_obs = len(self._st[self.scheme.root]["cats"])
        self._n_batches = max(1, n_root_obs // self._root_batch_size)

        # Per node/key: a ClassSampler (root) or BoundClassSampler (child) + an annbatch Loader. The schedule
        # oracle is built once; the root loader's sampler and each bound child's inner are deepcopies of it,
        # so they start from identical state and their per-batch draws agree. All of a node's keys share the
        # node's rng, so the (identical) samplers select the same rows every batch — a node's reps are aligned.
        self._oracle_sampler = self._new_class_sampler(self.scheme.root)  # per-batch condition schedule oracle
        self._loaders: dict[str, dict[str, AnnbatchLoader]] = {}
        self._add_node_loaders(self.scheme.root, deepcopy(self._oracle_sampler))
        for b in self._child_binds:
            self._add_node_loaders(b.child, self._new_bound_sampler(b))

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
        self._schedule: np.ndarray | None = None  # per-batch root leaf code for the current pass
        self._pos = 0

    def _resolve_configs(self, cfg: SamplerConfig | Mapping[str, SamplerConfig]) -> dict[str, SamplerConfig]:
        """Normalize to one config per node (nodes may use different batch sizes).

        Every node feeds a chunked sampler, so each is validated against the shared
        :meth:`_check_in_memory_chunk` rule (``in_memory`` ⇒ ``chunk_size=1``, else raise).
        """
        resolved = _resolve_config_map(cfg, self.scheme.nodes, kind="node")
        for name, node in self.scheme.nodes.items():
            self._check_in_memory_chunk(name, resolved[name], node)
        return resolved

    # ── build ────────────────────────────────────────────────────────────
    def _new_class_sampler(self, name: str) -> ClassSampler:
        try:  # annbatch enforces its own run-length rule for chunk>1; forward with node context
            return ClassSampler(
                chunk_size=self._cfg[name].chunk_size,
                preload_nchunks=self._cfg[name].preload_nchunks,
                batch_size=self._cfg[name].batch_size,
                classes=self._st[name]["cats"],
                num_samples=self._n_batches * self._cfg[name].batch_size,
                class_weights=self._st[name]["w"],
                drop_last=True,
                rng=deepcopy(self._node_rngs[name]),
            )
        except ValueError as e:
            raise ValueError(f"node {name!r}: {e}") from e

    def _new_bound_sampler(self, b: Bind) -> BoundClassSampler:
        # Inner = a copy of the oracle, so the bound draws the same per-batch schedule. Match on the bind's
        # shared columns; the child's leaf weights (0 for excluded leaves, e.g. perturbed cells in a control
        # node) go in as the *secondary* ``classes`` so only positive-weight child leaves are drawn within
        # each matched context — the exclusion `classes_to_bind_on` alone can't express (it groups all cells
        # sharing the context).
        cfg, cats = self._cfg[b.child], self._st[b.child]["cats"]
        parent, child = self._st[b.parent]["node"], self._st[b.child]["node"]
        try:
            return BoundClassSampler(
                deepcopy(self._oracle_sampler),
                cfg.chunk_size,
                cfg.preload_nchunks,
                cfg.batch_size,
                classes_to_bind_on=cats,
                # inner (parent) tuple position → bound (child) tuple position, per shared ``common`` column
                on={parent.cols.index(c): child.cols.index(c) for c in b.common},
                classes=cats,
                class_weights=self._st[b.child]["w"],
                rng=deepcopy(self._node_rngs[b.child]),
            )
        except ValueError as e:
            raise ValueError(f"node {b.child!r}: {e}") from e

    def _add_node_loaders(self, name: str, sampler: ClassSampler | BoundClassSampler) -> None:
        # add the same loader to every key of the node;
        src, node, to = self._nodes[name], self._st[name]["node"], self._cfg[name].to
        loaders: dict[str, AnnbatchLoader] = {}
        for key in node.keys:
            base = AnnbatchLoader(
                batch_sampler=deepcopy(sampler), return_index=False, to=to, preload_to_gpu=self._preload_to_gpu
            )
            loaders[key] = _attach(base, src, key)
        self._loaders[name] = loaders

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self) -> dict[str, object]:
        """Pickle without the live annbatch iterators (generators aren't picklable).

        Every sampler's RNG state is kept, so a reloaded loader resumes the same reproducible stream
        (the next pass) on the next ``__next__``; ``_iters`` is dropped and rebuilt, and the construction-
        only ``_leaf_cache`` is emptied (its codes are already carried by each node's categorical).
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

    def _nodes_next(self, name: str) -> dict[str, np.ndarray]:
        """One batch per key of a node — identical samplers pick the same rows, so the reps are aligned."""
        node = self._st[name]["node"]
        return {key: next(self._iters[name][key])["X"] for key in node.keys}

    def __next__(self) -> dict[str, dict]:
        if self._iters is None or self._pos >= self._n_batches:
            # Start a fresh pass (first pass, next epoch, or resume after unpickling): draw the schedule and
            # (re)build one iterator per node/key. The oracle's `batch_codes()` and each `iter()` each consume
            # one class draw, so — all samplers being deepcopies of the same oracle state — they stay in
            # lockstep and draw the same per-batch class. RNG advances across passes, so an unpickled loader
            # resumes the next pass rather than replaying.
            self._schedule = self._oracle_sampler.batch_codes()
            self._iters = {
                name: {key: iter(ld) for key, ld in loaders.items()} for name, loaders in self._loaders.items()
            }
            self._pos = 0
        j = self._pos
        self._pos += 1

        # One entry per streamed node, keyed by node name: the root (``scheme.root``) is the target, each
        # bound child is a source. Each value is that node's aligned reps ``{rep loc: rows}`` (same cells).
        out: dict = {self.scheme.root: self._nodes_next(self.scheme.root)}
        for b in self._child_binds:  # bound child sources, replayed via their BoundClassSamplers
            out[b.child] = self._nodes_next(b.child)
        if self._condition_lookup is not None:
            leaf = self._st[self.scheme.root]["leaves"][int(self._schedule[j])]  # per-batch category (root leaf)
            out["condition"] = _condition_from_lookup(self._condition_lookup, leaf)
        return out
