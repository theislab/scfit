import numpy as np
import pytest

from scfit.data._condition import _condition_from_lookup


def test_condition_lookup_preserves_arrays_and_dtypes() -> None:
    categorical = np.array([[1, 2]], dtype=np.int32)
    feature = np.array([[[1.0, 2.0]]], dtype=np.float64)

    resolved = _condition_from_lookup(
        lambda leaf: {"category": categorical, "feature": feature},
        ("context", "condition"),
    )

    assert resolved["category"] is categorical
    assert resolved["feature"] is feature
    assert resolved["category"].dtype == np.int32
    assert resolved["feature"].dtype == np.float64


@pytest.mark.parametrize(
    ("lookup", "error", "match"),
    [
        pytest.param(lambda leaf: np.array([[1]]), TypeError, "must return a mapping", id="not-mapping"),
        pytest.param(lambda leaf: {}, ValueError, "empty mapping", id="empty"),
        pytest.param(lambda leaf: {1: np.array([[1]])}, TypeError, "realm names", id="invalid-realm"),
        pytest.param(lambda leaf: {"drug": [[1]]}, TypeError, "must be a numpy array", id="not-array"),
        pytest.param(
            lambda leaf: {"drug": np.array([1])},
            ValueError,
            "leading singleton batch axis",
            id="missing-batch-axis",
        ),
        pytest.param(
            lambda leaf: {"drug": np.array([[1], [2]])},
            ValueError,
            "leading singleton batch axis",
            id="non-singleton-batch-axis",
        ),
        pytest.param(
            lambda leaf: {"drug": np.array([["a"]])},
            TypeError,
            "integer or floating dtype",
            id="non-numeric",
        ),
    ],
)
def test_condition_lookup_rejects_invalid_structure(lookup, error, match) -> None:
    with pytest.raises(error, match=match):
        _condition_from_lookup(lookup, ("context", "condition"))
