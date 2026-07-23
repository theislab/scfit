"""``Loader`` — streams matched ``{source, target, condition}`` batches from a :class:`Scheme`.

Each pass is a fresh epoch. The root (target) node draws a per-batch class schedule ∝ its weights via an
annbatch :class:`~annbatch.samplers.ClassSampler`; every bound child replays that schedule onto its own
cells via an annbatch :class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on the bind's
``common`` columns (select via child weights + project via ``common``). The loader
never wraps annbatch's RNG: every sampler that must agree within a pass (the schedule oracle, the target
reps, and each child's inner) is **reseeded from one per-pass seed** ``(node seed, pass index)``, so
target/condition/source stay aligned, a node's reps read the same rows, and a pickled loader resumes the
exact same stream. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import numpy as np
import pandas as pd
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not binded's `Loader`)
from annbatch.samplers import BoundClassSampler, ClassSampler

from scfit.data._backend import _bind_on, _build_loaders, _SchemeReader
from scfit.data._condition import ConditionLookup, _condition_from_lookup
from scfit.data._schema import Bind, Container, SamplerConfig, Scheme, _resolve_config_map

__all__ = ["Loader"]


class Loader(_SchemeReader):
    """Yields ``{"source", "target", "condition"}`` batches; every node streams through its own loader."""

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
        condition_lookup: ConditionLookup | None = None,
    ) -> None:
        super().__init__()  # the shared per-(source, cols) obs-factorization cache (`_factorize`)
        self.s = scheme
        self._condition_lookup = condition_lookup
        self._cfg = self._resolve_configs(sampler_config)
        self._B = self._cfg[self.s.root].batch_size  # root/target batch size — drives the pass length

        # root's direct children are the bound sources (parity with the previous loader: depth-1 sources)
        self._child_binds: list[Bind] = [b for b in scheme.binds if b.parent == self.s.root]

        # per-node stable sub-seed from one SeedSequence, so nodes don't correlate; a pass's seed is
        # (sub-seed, pass index) → a pass is fully reproducible from its index.
        self._node_seeds: dict[str, int] = {
            name: int(seq.generate_state(1)[0])
            for name, seq in zip(
                sorted(scheme.nodes), np.random.SeedSequence(scheme.seed).spawn(len(scheme.nodes)), strict=True
            )
        }

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
        n_root_obs = len(self._st[self.s.root]["cats"])
        self._n_batches = max(1, n_root_obs // self._B)

        self._build_samplers_and_loaders()

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
        self._schedule: np.ndarray | None = None  # per-batch root leaf code for the current pass
        self._pos = 0

    def _resolve_configs(self, cfg: SamplerConfig | Mapping[str, SamplerConfig]) -> dict[str, SamplerConfig]:
        """Normalize to one config per node (nodes may use different batch sizes).

        Every node feeds a chunked sampler, so each is validated against the shared
        :meth:`_check_in_memory_chunk` rule (``in_memory`` ⇒ ``chunk_size=1``, else raise).
        """
        resolved = _resolve_config_map(cfg, self.s.nodes, kind="node")
        for name, node in self.s.nodes.items():
            self._check_in_memory_chunk(name, resolved[name], node)
        return resolved

    # ── build ────────────────────────────────────────────────────────────
    def _new_class_sampler(self, name: str) -> ClassSampler:
        cfg = self._cfg[name]
        try:  # annbatch enforces its own run-length rule for chunk>1; forward with node context
            return ClassSampler(
                chunk_size=cfg.chunk_size,
                preload_nchunks=cfg.preload_nchunks,
                batch_size=cfg.batch_size,
                classes=self._st[name]["cats"],
                num_samples=self._n_batches * cfg.batch_size,
                class_weights=self._st[name]["w"],
                drop_last=True,
                rng=np.random.default_rng(self._node_seeds[name]),
            )
        except ValueError as e:
            raise ValueError(f"node {name!r}: {e}") from e

    def _new_bound_sampler(self, b: Bind) -> BoundClassSampler:
        inner = self._new_class_sampler(self.s.root)
        # Match on the bind's shared columns; the child's leaf weights (0 for excluded leaves, e.g.
        # perturbed cells in a control node) go in as the *secondary* class so only positive-weight
        # child leaves are drawn within each matched context — the exclusion `classes_to_bind_on` alone
        # can't express (it groups all cells sharing the context).
        return self._make_bound(
            b,
            inner,
            on=_bind_on(self._st[b.parent]["node"], self._st[b.child]["node"], b.common),
            classes=self._st[b.child]["cats"],
            class_weights=self._st[b.child]["w"],
        )

    def _make_bound(
        self,
        b: Bind,
        inner: ClassSampler,
        *,
        on: dict[int, int] | None,
        classes: pd.Categorical | None = None,
        class_weights: np.ndarray | None = None,
    ) -> BoundClassSampler:
        cfg = self._cfg[b.child]
        try:
            return BoundClassSampler(
                inner,
                cfg.chunk_size,
                cfg.preload_nchunks,
                cfg.batch_size,
                classes_to_bind_on=self._st[b.child]["cats"],
                on=on,
                classes=classes,
                class_weights=class_weights,
                rng=np.random.default_rng(self._node_seeds[b.child]),
            )
        except ValueError as e:
            raise ValueError(f"node {b.child!r}: {e}") from e

    def _build_samplers_and_loaders(self) -> None:
        """Per node/key: a ClassSampler (root) or BoundClassSampler (child) + annbatch Loader.

        A per-child schedule *oracle* and each bound's inner are root-seeded so their class draws agree
        with the target's; all of a node's keys share the node seed, so the (identical) samplers select
        the same rows every batch — every rep of a node is the same cells.
        """
        self._oracle = self._new_class_sampler(self.s.root)  # supplies the per-batch condition schedule

        self._loaders: dict[str, dict[str, AnnbatchLoader]] = {}
        self._add_node_loaders(self.s.root, lambda: self._new_class_sampler(self.s.root))
        for b in self._child_binds:
            self._add_node_loaders(b.child, lambda b=b: self._new_bound_sampler(b))

    def _add_node_loaders(self, name: str, make_sampler: Callable[[], ClassSampler | BoundClassSampler]) -> None:
        node = self._st[name]["node"]
        self._loaders[name] = _build_loaders(self._nodes[name], node, self._cfg[name], make_sampler)

    # ── per-pass scheduling ────────────────────────────────────────────────
    def _start_pass(self) -> None:
        """Draw the schedule and rebuild iterators for a fresh epoch (advancing every sampler's RNG once).

        The oracle's ``batch_codes()`` and each target/child ``iter()`` each consume one class draw, so —
        all root-referencing samplers having started from the root seed — the oracle, the target reps and
        every bound child's inner stay in lockstep and draw the *same* per-batch class each pass. The RNG
        advances across passes (a real epoch stream), so a pickled loader — whose sampler RNG state is
        kept — resumes the next pass rather than replaying.
        """
        self._schedule = self._oracle.batch_codes()
        self._iters = {name: {key: iter(ld) for key, ld in loaders.items()} for name, loaders in self._loaders.items()}
        self._pos = 0

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
        self._iters = None  # force `_start_pass` on the next `__next__`, using the restored sampler RNG

    # ── iteration ──────────────────────────────────────────────────────────
    def __iter__(self) -> Loader:
        return self

    def _nodes_next(self, name: str) -> dict[str, np.ndarray]:
        """One batch per key of a node — identical samplers pick the same rows, so the reps are aligned."""
        node = self._st[name]["node"]
        return {key: next(self._iters[name][key])["X"] for key in node.keys}

    def __next__(self) -> dict[str, np.ndarray]:
        if self._iters is None or self._pos >= self._n_batches:
            self._start_pass()  # first pass, next epoch, or resume after unpickling
        j = self._pos
        self._pos += 1

        st = self._st[self.s.root]
        node = st["node"]
        leaf = st["leaves"][int(self._schedule[j])]  # per-batch category — from the schedule oracle

        out: dict = {}
        self._emit_rep(out, "target", self._nodes_next(self.s.root), node)  # target + aligned target_reps
        if self._condition_lookup is not None:
            out["condition"] = _condition_from_lookup(self._condition_lookup, leaf)
        for b in self._child_binds:  # bound child source, replayed via its BoundClassSampler
            self._emit_rep(out, "source", self._nodes_next(b.child), self._st[b.child]["node"])
        return out
