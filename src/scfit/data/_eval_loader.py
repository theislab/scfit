"""``EvalLoader`` — control-rooted eval reader: a Sequential control *inner* + a bound perturbed target.

Anchors on the source (control) — "there is no perturbed without a source". Two loaders, both driven by
one deterministic :class:`~annbatch.samplers.SequentialClassSampler` over the control populations:

* the **source** loader *is* that inner — it reads each scheduled control population **in full** (all its
  controls), and
* the **target** loader is an :class:`~annbatch.samplers.BoundClassSampler` on the *same* inner, matched on
  the bind's ``common`` (context) columns, that **samples** a perturbed leaf (drug) within each matched
  context.

annbatch does all the class matching; nothing is derived or updated per pass here. The condition of each
batch is the perturbed leaf the bound drew (read from an identically-seeded oracle, so it lines up with
the target loader's own draw). Which control populations are visited — and how many batches — is set via
the schedule (``iter_conditions``). Works over an in-memory ``AnnData`` and a ``DatasetCollection`` alike.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from annbatch.samplers import BoundClassSampler, SequentialClassSampler

from scfit.data._backend import _bind_on, _build_loaders, _SchemeReader
from scfit.data._condition import ConditionLookup, _condition_from_lookup
from scfit.data._schema import SamplerConfig, Scheme

__all__ = ["EvalLoader"]


class EvalLoader(_SchemeReader):
    """Yield ``{"source", "target", "condition", "leaf"}`` per control population (source-anchored).

    Parameters
    ----------
    scheme
        A prepared :class:`~binded.Scheme` (root = perturbed/target node, one bound control child).
        Its bind's ``common`` columns are the matching context (e.g. ``cell_line``).
    sampler_config
        Read parameters for the bound perturbed-target sampler (``batch_size`` = target cells per
        condition; controls are read in full regardless).
    condition_lookup
        Maps a perturbed leaf (over the target node's ``cols``) to its structured raw condition:
        categorical realms are integer indices and feature realms are float vectors, each with a leading
        singleton batch axis. Model-side modules own the encoding.
    seed
        Seed for the target's drug sampling (reproducible across ``iter_conditions`` calls).
    """

    def __init__(
        self,
        scheme: Scheme,
        sampler_config: SamplerConfig,
        condition_lookup: ConditionLookup | None = None,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()  # the shared per-(source, cols) obs-factorization cache (`_factorize`)
        self.s = scheme
        self._condition_lookup = condition_lookup
        self._seed = seed
        binds = [b for b in scheme.binds if b.parent == scheme.root]
        if len(binds) != 1:
            raise ValueError("EvalLoader expects exactly one bound source of the root.")
        b = binds[0]
        self._pert = scheme.nodes[scheme.root]  # perturbed / target
        self._ctrl = scheme.nodes[b.child]  # control / source
        self._context = b.common
        self._cfg = sampler_config
        # Only the target feeds a chunked sampler (BoundClassSampler); the source is read in full by a
        # chunk-agnostic SequentialClassSampler. So the shared in-memory ⇒ chunk_size=1 rule is enforced on
        # the target only — an in-memory control carries no chunk constraint.
        self._check_in_memory_chunk(scheme.root, sampler_config, self._pert)
        # Both nodes honor Node.in_memory uniformly via `_prepare` (materialize the selected cells into RAM
        # once) and share the base's single obs factorization per (source, cols) — control and perturbed
        # over the same source factorize exactly once. The perturbed target is materialized only if its
        # in_memory flag is set; it defaults to streamed since it can be large.
        self._src, self._ctrl_cats, self._ctrl_w, cl = self._prepare(scheme, self._ctrl)
        self._src_p, self._pert_cats, self._pert_w, pl = self._prepare(scheme, self._pert)
        self._ctrl_leaves = cl
        # inner (control) tuple position -> target tuple position, for each shared context column
        self._on = _bind_on(self._ctrl, self._pert, self._context)
        # Only visit control populations that actually have >=1 positive-weight target leaf sharing
        # their context. split_scheme carries a bound child's (control) weights through UNCHANGED --
        # they are not restricted to the split (by design, so match_context columns outside split_by
        # stay fully available). For a held-out scheme whose split_by *is* (a superset of) the bind's
        # context, that means the control side still spans every context the full (pre-split) scheme
        # ever had, so without this filter the schedule below would visit contexts with zero matching
        # target leaves here and the bound sampler would raise on the very first such context.
        ctrl_pos, pert_pos = tuple(self._on.keys()), tuple(self._on.values())
        available_contexts = {tuple(pl[i][p] for p in pert_pos) for i in range(len(pl)) if self._pert_w[i] > 0}
        self._ctrl_codes = np.array(
            [
                i
                for i in range(len(cl))
                if self._ctrl_w[i] > 0 and tuple(cl[i][p] for p in ctrl_pos) in available_contexts
            ],
            dtype=np.int64,
        )
        if self._ctrl_codes.size == 0:
            raise ValueError("no control population (positive-weight control leaf) to evaluate.")

    @property
    def control_populations(self) -> list[tuple]:
        """The control leaves (contexts) this loader iterates over."""
        return [self._ctrl_leaves[i] for i in self._ctrl_codes]

    def _inner(self, schedule: np.ndarray) -> SequentialClassSampler:
        # deterministic: same schedule ⇒ same control-population order across the source loader and every
        # bound inner, so source[j] and target[j] refer to the same matched context.
        return SequentialClassSampler(self._ctrl_cats, schedule=schedule)

    def _bound(self, schedule: np.ndarray) -> BoundClassSampler:
        cfg = self._cfg
        return BoundClassSampler(
            self._inner(schedule),
            cfg.chunk_size,
            cfg.preload_nchunks,
            cfg.batch_size,
            classes_to_bind_on=self._pert_cats,
            on=self._on,
            classes=self._pert_cats,  # secondary = the perturbed leaf; weights pick positive-weight drugs
            class_weights=self._pert_w,
            rng=np.random.default_rng(self._seed),
        )

    def iter_conditions(self, n_conditions: int | None = None) -> Iterator[dict]:
        """Yield one batch per scheduled control population.

        With ``n_conditions`` set, the control populations are cycled to that many batches (each re-reads
        the population's controls and the bound samples a fresh drug); otherwise every control population
        is visited once.
        """
        if n_conditions is None:
            schedule = self._ctrl_codes.copy()
        else:
            reps = int(np.ceil(n_conditions / self._ctrl_codes.size))
            schedule = np.tile(self._ctrl_codes, reps)[:n_conditions].astype(np.int64)

        # condition oracle: identically-seeded bound → its per-batch drawn perturbed leaf lines up with
        # the target loader's own draw (annbatch reproduces the same class sequence from the same seed).
        oracle = self._bound(schedule)
        vocab = oracle.vocab
        ctx = len(self._context)
        cond_leaves = [tuple(vocab[int(c)])[ctx:] for c in oracle.batch_codes()]  # strip the shared context prefix

        src_loaders = _build_loaders(self._src, self._ctrl, self._cfg, lambda: self._inner(schedule))
        tgt_loaders = _build_loaders(self._src_p, self._pert, self._cfg, lambda: self._bound(schedule))
        src_iters = {k: iter(ld) for k, ld in src_loaders.items()}
        tgt_iters = {k: iter(ld) for k, ld in tgt_loaders.items()}
        skeys, tkeys = list(src_iters), list(tgt_iters)

        for j in range(len(schedule)):
            src = {k: next(src_iters[k])["X"] for k in skeys}
            tgt = {k: next(tgt_iters[k])["X"] for k in tkeys}
            leaf = cond_leaves[j]
            out: dict = {"leaf": leaf}
            self._emit_rep(out, "source", src, self._ctrl)  # source + aligned source_reps
            self._emit_rep(out, "target", tgt, self._pert)  # target + aligned target_reps
            if self._condition_lookup is not None:
                out["condition"] = _condition_from_lookup(self._condition_lookup, leaf)
            yield out
