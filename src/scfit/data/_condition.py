from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

__all__ = ["ConditionLookup"]

type Leaf = tuple[object, ...]
type ConditionLookup = Callable[[Leaf], Mapping[str, np.ndarray]]


def _condition_from_lookup(lookup: ConditionLookup, leaf: Leaf) -> dict[str, np.ndarray]:
    """Resolve one class-coherent target leaf without changing array dtype or device semantics."""
    condition = lookup(leaf)
    if not isinstance(condition, Mapping):
        raise TypeError(
            f"condition_lookup must return a mapping of realm names to numpy arrays, found {type(condition).__name__}."
        )
    if not condition:
        raise ValueError("condition_lookup returned an empty mapping.")

    resolved: dict[str, np.ndarray] = {}
    for realm, value in condition.items():
        if not isinstance(realm, str) or not realm:
            raise TypeError(f"condition realm names must be non-empty strings, found {realm!r}.")
        if not isinstance(value, np.ndarray):
            raise TypeError(f"condition_lookup realm {realm!r} must be a numpy array, found {type(value).__name__}.")
        if value.ndim < 2 or value.shape[0] != 1:
            raise ValueError(
                f"condition_lookup realm {realm!r} must have a leading singleton batch axis, found shape {value.shape}."
            )
        if not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)):
            raise TypeError(
                f"condition_lookup realm {realm!r} must have an integer or floating dtype, found {value.dtype}."
            )
        # Dtype/device conversion belongs to the model compute boundary, not data preparation.
        resolved[realm] = value
    return resolved
