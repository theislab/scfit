"""Weight helpers and the per-leaf weight vector — what carries weight IS the selection.

``uniform`` / ``frequency`` / ``inverse_frequency`` just build ``{combo: weight}`` dicts; ``_weight_vector``
resolves such a dict against a node's ordered leaves into normalized per-leaf probabilities (a leaf absent
from the mapping, or with weight 0, is excluded). No sampler is built here — pure dict/array math.
"""

from __future__ import annotations

import numpy as np
import pytest

from scfit.data._schema import _weight_vector, frequency, inverse_frequency, uniform

LEAVES = [("A", "d1"), ("A", "d2"), ("B", "d1")]


def test_uniform_weights_every_combo_equally():
    assert uniform(LEAVES) == {("A", "d1"): 1.0, ("A", "d2"): 1.0, ("B", "d1"): 1.0}


def test_frequency_is_proportional_to_counts():
    assert frequency({("A", "d1"): 30, ("B", "d1"): 10}) == {("A", "d1"): 30.0, ("B", "d1"): 10.0}


def test_inverse_frequency_balances_rare_vs_abundant():
    assert inverse_frequency({("A", "d1"): 4, ("B", "d1"): 1}) == {("A", "d1"): 0.25, ("B", "d1"): 1.0}


def test_weight_vector_normalizes_to_a_distribution():
    v = _weight_vector({("A", "d1"): 6, ("A", "d2"): 3, ("B", "d1"): 1}, LEAVES)
    np.testing.assert_allclose(v, [0.6, 0.3, 0.1])
    assert v.sum() == pytest.approx(1.0)


def test_weight_vector_treats_absent_leaf_as_zero():
    v = _weight_vector({("A", "d1"): 1, ("B", "d1"): 1}, LEAVES)  # (A, d2) absent → excluded
    np.testing.assert_allclose(v, [0.5, 0.0, 0.5])


def test_weight_vector_all_zero_over_these_leaves_raises():
    with pytest.raises(ValueError, match="all-zero"):
        _weight_vector({("Z", "z"): 1.0}, LEAVES)  # no overlap with the node's leaves → nothing to sample
