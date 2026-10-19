"""``BoundClassSampler`` replays the inner sampler's per-batch class onto another obs table."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from annbatch.samplers import ClassSampler

from scfit.data._bound_sampler import BoundClassSampler


def _cats(labels: list[tuple], n: int) -> pd.Categorical:
    return pd.Categorical.from_codes(
        np.repeat(np.arange(len(labels)), n), pd.MultiIndex.from_tuples(labels).to_flat_index()
    )


def _inner(classes: pd.Categorical) -> ClassSampler:
    return ClassSampler(4, 2, 4, classes=classes, num_samples=40, drop_last=True, rng=np.random.default_rng(0))


def test_each_batch_matches_the_inner_batch_context():
    inner_classes = _cats([("A", "x"), ("B", "x")], 8)
    own = _cats([("A", "ctrl"), ("B", "ctrl"), ("B", "other")], 8)
    inner = _inner(inner_classes)
    bound = BoundClassSampler(
        deepcopy(inner),
        4,
        2,
        4,
        inner_classes=inner_classes,
        classes=own,
        on={0: 0},
        class_weights=np.array([1.0, 1.0, 0.0]),
        rng=np.random.default_rng(1),
    )
    inner_lines = [
        inner_classes.categories[inner_classes.codes[r.start]][0] for req in inner.sample(16) for r in req["requests"]
    ]
    own_labels = [own.categories[own.codes[r.start]] for req in bound.sample(24) for r in req["requests"]]
    assert [label[0] for label in own_labels] == inner_lines
    assert ("B", "other") not in own_labels  # zero weight: never drawn


def test_an_inner_class_without_a_match_is_refused():
    inner_classes = _cats([("A", "x"), ("C", "x")], 8)
    with pytest.raises(ValueError, match="no positive-weight match"):
        BoundClassSampler(
            _inner(inner_classes), 4, 2, 4, inner_classes=inner_classes, classes=_cats([("A", "ctrl")], 8), on={0: 0}
        )
