"""``Loader`` — streams matched per-node batches from a :class:`Scheme`, keyed by node name.

Each pass is a fresh epoch. The root (target) node draws a per-batch class schedule ∝ its weights via an
annbatch :class:`~annbatch.samplers.ClassSampler`; every bound child replays that schedule onto its own
cells via an annbatch :class:`~annbatch.samplers.BoundClassSampler` — matched by *label* on the bind's
``common`` columns (select via child weights + project via ``common``). A batch is ``{node name: {rep
loc: rows}}`` for the root and every bound child, plus ``"annotations"`` — the consumer reads the target
from ``scheme.root`` and the sources from the bound children. Every sampler that must agree within a pass
(the schedule oracle, the target's reps, and each child's inner) is a ``deepcopy`` of the same seeded
oracle state, so they stay in lockstep, a node's reps read the same rows, and a pickled loader resumes
the exact same stream. See ``README.md``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from annbatch import Loader as AnnbatchLoader  # annbatch's low-level per-rep loader (not binded's `Loader`)
from annbatch.samplers import BoundClassSampler, ClassSampler

from scfit.data._io import (
    get_from_container,
    is_backed_array,
    leaf_codes,
    materialize_node,
    obs_columns,
)
from scfit.data._schema import Container, Node, SamplerConfig, Scheme, _resolve_config_map, _weight_vector

__all__ = ["Annotations", "Loader"]

# Per-node annotations: ``{node name: {leaf: {realm: array}}}`` — for a node, a lookup from that node's
# leaf (its ``cols`` combination) to the named arrays for a batch drawn from that leaf (e.g. the
# perturbation encoding). Keyed by the SAME leaf tuples as the node's ``weights``. Each batch surfaces
# ``{node name: the current leaf's arrays}`` under ``"annotations"`` for every node given here — the node's
# per-batch leaf is read from annbatch's ``comb`` (for a bound child, the child leaf is ``comb``'s tail).
type Annotations = Mapping[str, Mapping[tuple, Mapping[str, np.ndarray]]]

if TYPE_CHECKING:
    from scfit.data._schema import Bind, Container, SamplerConfig, Scheme


class Loader:
    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig | Mapping[str, SamplerConfig],
        annotations: Annotations | None = None,
        *,
        preload_to_gpu: bool = False,
    ) -> None:
        # (source name, cols) -> its full-obs (codes, leaves): construction-only scratch behind `_factorize`,
        # emptied on pickle (each node's categorical already carries its own codes).
        self._leaf_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, list[tuple]]] = {}
        self.scheme = scheme
        self._annotations = annotations
        self._preload_to_gpu = preload_to_gpu
        self._cfg = self._resolve_configs(sampler_config)
        self._root_batch_size = self._cfg[
            self.scheme.root
        ].batch_size  # root/target batch size — drives the pass length

        self._child_binds: list[Bind] = [b for b in scheme.binds if b.parent == self.scheme.root]
        # a bound child's `comb` is the joint (common…, child leaf…); its own leaf is the tail past `common`.
        self._child_common: dict[str, tuple[str, ...]] = {b.child: b.common for b in self._child_binds}

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
        self._validate_annotations()  # strict: every named node's positive-weight leaves must be covered

        # A natural epoch over the root (target) node: its cell count // the root's batch_size. The root
        # drives the zip, so every node draws the same number of batches; each node's num_samples ==
        # _n_batches * that node's batch_size (source rows need not equal target rows).
        n_root_obs = len(self._st[self.scheme.root]["cats"])
        self._n_batches = max(1, n_root_obs // self._root_batch_size)

        # Per node/key: a ClassSampler (root) or BoundClassSampler (child) + an annbatch Loader. The schedule
        # oracle is built once; the root loader's sampler and each bound child's inner are deepcopies of it,
        # so they start from identical state and their per-batch draws agree. All of a node's keys share the
        # node's rng, so the (identical) samplers select the same rows every batch — a node's reps are aligned.
        self._oracle_sampler = self._new_class_sampler(self.scheme.root)  # template: deepcopied for root + child inners
        self._loaders: dict[str, dict[str, AnnbatchLoader]] = {}
        self._add_node_loaders(self.scheme.root, deepcopy(self._oracle_sampler))
        for b in self._child_binds:
            self._add_node_loaders(b.child, self._new_bound_sampler(b))

        self._iters: dict[str, dict[str, Iterator[dict]]] | None = None
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

    @staticmethod
    def _check_in_memory_chunk(name: str, cfg: SamplerConfig, node: Node) -> None:
        """Reject a chunked config on an ``in_memory`` node."""
        if node.in_memory and cfg.chunk_size != 1:
            raise ValueError(
                f"node {name!r} sets in_memory=True but chunk_size={cfg.chunk_size}: an in-memory node is "
                "read from RAM in one shot and must use chunk_size=1 (set it explicitly)."
            )

    def _validate_annotations(self) -> None:
        """Strict coverage: every named node's positive-weight leaf must have an annotation (no silent gaps)."""
        if self._annotations is None:
            return
        for name, node_ann in self._annotations.items():
            if name not in self._st:
                raise ValueError(f"annotations reference unknown node {name!r}; nodes are {sorted(self._st)}.")
            leaves, weights = self._st[name]["leaves"], self._st[name]["w"]
            missing = [lf for lf, w in zip(leaves, weights, strict=True) if w > 0 and lf not in node_ann]
            if missing:
                raise ValueError(f"annotations for node {name!r} miss positive-weight leaves {missing}.")

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
        # one annbatch Loader per rep of the node, keyed by the rep's anndata.acc accessor.
        src, node, to = self._nodes[name], self._st[name]["node"], self._cfg[name].to
        loaders: dict = {}
        for key in node.keys:
            base = AnnbatchLoader(
                batch_sampler=deepcopy(sampler), return_index=False, to=to, preload_to_gpu=self._preload_to_gpu
            )
            backings = get_from_container(src, key)
            # in-memory adata → add_adatas (obs-free X wrapper); backed adata / list of backed → add_datasets
            loaders[key] = (
                base.add_adatas([ad.AnnData(X=b) for b in backings])
                if isinstance(src, ad.AnnData) and not is_backed_array(backings[0])
                else base.add_datasets(backings)
            )
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

    def _nodes_next(self, name: str) -> tuple[dict, object]:
        """A node's aligned reps ``{rep loc: rows}`` plus its per-batch ``comb`` (category label).

        Identical samplers pick the same aligned rows for every rep, and share one ``comb`` per batch.
        """
        node = self._st[name]["node"]
        reps, comb = {}, None
        for key in node.keys:
            batch = next(self._iters[name][key])
            reps[key], comb = batch["X"], batch["comb"]
        return reps, comb

    def _node_leaf(self, name: str, comb: object) -> tuple:
        """The node's own leaf for this batch: the root's ``comb`` as-is; a child's is ``comb`` past ``common``."""
        leaf = tuple(comb)
        return leaf if name == self.scheme.root else leaf[len(self._child_common[name]) :]

    def __next__(self) -> dict[str, dict]:
        if self._iters is None or self._pos >= self._n_batches:
            # Start a fresh pass (first pass, next epoch, or resume after unpickling): (re)build one iterator
            # per node/key from the (advancing) sampler RNG, so each pass is a fresh reproducible epoch.
            self._iters = {
                name: {key: iter(ld) for key, ld in loaders.items()} for name, loaders in self._loaders.items()
            }
            self._pos = 0
        self._pos += 1

        # One entry per streamed node, keyed by node name: the root (``scheme.root``) is the target, each
        # bound child is a source. Each value is that node's aligned reps ``{rep loc: rows}`` (same cells).
        out: dict = {}
        combs: dict = {}
        for name in (self.scheme.root, *(b.child for b in self._child_binds)):
            out[name], combs[name] = self._nodes_next(name)
        if self._annotations is not None:
            # per named node, look up that node's current leaf (from its `comb`) — strict coverage was
            # checked at construction, so every drawn leaf is present.
            out["annotations"] = {
                name: node_ann[self._node_leaf(name, combs[name])] for name, node_ann in self._annotations.items()
            }
        return out

    def _factorize(self, src: Container, source: str, cols: tuple[str, ...]) -> tuple[np.ndarray, list[tuple]]:
        """The source's ``(codes, leaves)`` over ``cols`` — read + factorized once, cached by ``(source, cols)``."""
        ck = (source, cols)
        if ck not in self._leaf_cache:
            self._leaf_cache[ck] = leaf_codes(obs_columns(src, cols), cols)
        return self._leaf_cache[ck]

    def _node_stats(self, src: Container, node: Node) -> tuple[pd.Categorical, np.ndarray, list[tuple]]:
        """Tuple-labelled categorical, normalized weight vector, and leaf list for a node read from ``src``.

        ``src`` is the node's scheme source (not a materialized subset); its factorization is shared via
        :meth:`_factorize`. The categorical and weight vector are cheap and always rebuilt — nodes over the
        same ``(source, cols)`` differ only in ``weights``.
        """
        codes, leaves = self._factorize(src, node.source, node.cols)
        return _flat_categorical(codes, leaves), _weight_vector(node.weights, leaves), leaves

    def _prepare(self, scheme: Scheme, node: Node) -> tuple[Container, pd.Categorical, np.ndarray, list[tuple]]:
        """A node's resolved source + its categorical, weight vector and leaf list (obs only).

        An ``in_memory`` node is materialized into RAM here (positive-weight rows only): the row selection
        reuses the *cached* source factorization rather than re-reading the source obs, and the subset's
        own ``(codes, leaves)`` are derived by :func:`materialize_node` from that same factorization — so no
        obs is factorized more than once. A non-materialized node reads straight from the scheme source.
        """
        src = scheme.sources[node.source]
        if node.in_memory:
            codes, leaves = self._factorize(src, node.source, node.cols)
            src, codes, leaves = materialize_node(src, node, (codes, leaves))
            return src, _flat_categorical(codes, leaves), _weight_vector(node.weights, leaves), leaves
        cats, w, leaves = self._node_stats(src, node)
        return src, cats, w, leaves


def _flat_categorical(codes: np.ndarray, leaves: list[tuple]) -> pd.Categorical:
    categories = pd.MultiIndex.from_tuples(leaves).to_flat_index()
    return pd.Categorical.from_codes(codes, categories=categories)
