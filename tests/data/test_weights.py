"""The per-leaf weight vector — what carries weight IS the selection.

``weight_vector`` resolves a ``{combo: weight}`` dict against a stream's ordered leaves into normalized
per-leaf probabilities (a leaf absent from the mapping, or with weight 0, is excluded); ``None`` means
uniform over every leaf. No sampler is built here — pure dict/array math.
"""

from __future__ import annotations

import numpy as np
import pytest

from scfit.data._schema import weight_vector

LEAVES = [("A", "d1"), ("A", "d2"), ("B", "d1")]


def test_weight_vector_normalizes_to_a_distribution():
    v = weight_vector({("A", "d1"): 6, ("A", "d2"): 3, ("B", "d1"): 1}, LEAVES)
    np.testing.assert_allclose(v, [0.6, 0.3, 0.1])
    assert v.sum() == pytest.approx(1.0)


def test_weight_vector_treats_absent_leaf_as_zero():
    v = weight_vector({("A", "d1"): 1, ("B", "d1"): 1}, LEAVES)  # (A, d2) absent → excluded
    np.testing.assert_allclose(v, [0.5, 0.0, 0.5])


def test_weight_vector_none_is_uniform():
    v = weight_vector(None, LEAVES)  # None ⇒ every leaf equally likely
    np.testing.assert_allclose(v, [1 / 3, 1 / 3, 1 / 3])
    assert v.sum() == pytest.approx(1.0)


def test_weight_vector_all_zero_over_these_leaves_raises():
    with pytest.raises(ValueError, match="all-zero"):
        weight_vector({("Z", "z"): 1.0}, LEAVES)  # no overlap with the leaves → nothing to sample
