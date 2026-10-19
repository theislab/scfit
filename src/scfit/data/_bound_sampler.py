"""``BoundClassSampler``: a link's sampler, drawing each batch from the class the primary drew for it."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from annbatch.samplers import ClassSampler
from annbatch.utils import split_given_size

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import pandas as pd
    from annbatch.types import LoadRequest

__all__ = ["BoundClassSampler"]


class BoundClassSampler(ClassSampler):
    """Replays ``inner``'s per-batch class schedule onto another obs table.

    For every batch ``inner`` yields, this sampler yields one batch of its own ``batch_size`` observations,
    read as whole chunks of a single class of ``classes``: one whose labels agree with the inner batch's on
    the ``on`` positions (``{inner tuple position: own tuple position}``), drawn ∝ ``class_weights`` among
    those (a non-positive weight excludes a class). Both samplers see the same batches, so iterate a
    ``deepcopy`` of the sampler ``inner`` was copied from to stay in lockstep with it.

    Parameters
    ----------
    inner
        The class sampler whose schedule is replayed.
    chunk_size, preload_nchunks, batch_size
        Sizing for this sampler; ``batch_size`` must be a multiple of ``chunk_size``.
    inner_classes
        ``inner``'s per-observation classes, whose categories are tuples.
    classes
        This sampler's per-observation classes, whose categories are tuples.
    on
        ``{inner tuple position: own tuple position}`` for the label positions that must agree.
    class_weights
        One weight per category of ``classes``; ``None`` is uniform.
    rng
        Random number generator for the within-context draws and the chunk starts.
    """

    def __init__(
        self,
        inner: ClassSampler,
        chunk_size: int,
        preload_nchunks: int,
        batch_size: int,
        *,
        inner_classes: pd.Categorical,
        classes: pd.Categorical,
        on: Mapping[int, int],
        class_weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if batch_size % chunk_size:
            raise ValueError(f"batch_size ({batch_size}) must be a multiple of chunk_size ({chunk_size}).")
        super().__init__(
            chunk_size,
            preload_nchunks,
            batch_size,
            classes=classes,
            num_samples=inner.n_batches(len(inner_classes)) * batch_size,
            class_weights=class_weights,
            drop_last=True,
            rng=rng,
        )
        self._inner = inner
        self._inner_classes = inner_classes
        self._inner_chunk_size = inner._chunk_size

        # Own drawable classes per shared key, with their probabilities renormalized within the key.
        def key(label: tuple, positions: Mapping[int, int] | list[int]) -> tuple:
            return tuple(label[p] for p in positions)

        own_codes, own_probs = self._rle_manager.codes, self._rle_manager.weights
        by_key: dict[tuple, list[int]] = {}
        for i, code in enumerate(own_codes):
            by_key.setdefault(key(classes.categories[code], list(on.values())), []).append(i)
        self._choices = {k: (own_codes[ix], own_probs[ix] / own_probs[ix].sum()) for k, ix in by_key.items()}
        # Per inner class code: the key it binds on. Every class inner can draw must be drawable here.
        self._inner_key = [key(label, list(on)) for label in inner_classes.categories]
        missing = {self._inner_key[c] for c in inner._rle_manager.codes} - self._choices.keys()
        if missing:
            raise ValueError(f"The inner sampler can draw classes {sorted(missing)} with no positive-weight match.")

    def _iter_requests(self) -> Iterator[LoadRequest]:
        inner_codes = self._inner_classes.codes
        batch_codes = []
        for request in self._inner.sample(len(self._inner_classes)):
            starts = [s.start for s in request["requests"]]
            for split in request["splits"]:
                pos = int(split[0])  # every row of an inner batch is of the same class
                row = starts[pos // self._inner_chunk_size] + pos % self._inner_chunk_size
                codes, probs = self._choices[self._inner_key[inner_codes[row]]]
                batch_codes.append(self._rng.choice(codes, p=probs))
        per_batch = self._batch_size // self._chunk_size
        slices = self._rle_manager.slices_from_classes(np.repeat(np.asarray(batch_codes), per_batch), self._rng)
        for window in itertools.batched(slices, self._preload_nchunks):
            splits = split_given_size(np.arange(len(window) * self._chunk_size), self._batch_size)
            for batch in splits:
                self._rng.shuffle(batch)
            yield {"requests": list(window), "splits": splits}
