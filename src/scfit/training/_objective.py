from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

import torch

__all__ = [
    "Objective",
    "register_objective",
    "build_objective",
    "OBJECTIVE_REGISTRY",
]


class Objective(abc.ABC):
    """Turn a ``(model, batch)`` pair into ``(loss, logs)`` — the trainable objective seam."""

    @abc.abstractmethod
    def compute_loss(self, model: torch.nn.Module, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return ``(loss, logs)`` for ``batch`` under ``model``."""
        ...


# Registry so third-party objectives are discoverable by name. An objective builder returns an
# :class:`Objective`. (Top-level architectures are reconstructed from their component spec via
# :mod:`scfit.registry`, not a builder-by-name registry.)
OBJECTIVE_REGISTRY: dict[str, Callable[..., Objective]] = {}


def register_objective(name: str) -> Callable[[Callable[..., Objective]], Callable[..., Objective]]:
    """Decorator: register an objective builder under ``name`` (errors on a duplicate name)."""

    def deco(builder: Callable[..., Objective]) -> Callable[..., Objective]:
        """Register ``builder`` under the captured ``name`` and return it unchanged."""
        if name in OBJECTIVE_REGISTRY:
            raise ValueError(f"Objective {name!r} already registered.")
        OBJECTIVE_REGISTRY[name] = builder
        return builder

    return deco


def build_objective(name: str, *args: Any, **kwargs: Any) -> Objective:
    """Build a registered objective by ``name`` (errors with the available names if unknown)."""
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(f"Objective {name!r} not registered. Available: {sorted(OBJECTIVE_REGISTRY)}.")
    return OBJECTIVE_REGISTRY[name](*args, **kwargs)
