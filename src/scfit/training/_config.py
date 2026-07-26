"""Shared Component base classes for the two portable model roles.

Every model family expresses its architecture and its objective as a :class:`scfit.registry.Component`. These
two abstract family bases carry no ``type_id`` (so they stay unregistered and usable as the ``expected``
family in ``from_spec``); concrete configs — ``GeneEncoderConfig``, ``MLPVelocityConfig``,
``OTFMObjectiveConfig``, ``ContrastiveObjectiveConfig`` — subclass them and register with a ``type_id``.

``build(context)`` turns the portable config into a runtime object: an :class:`torch.nn.Module` for an
architecture, an :class:`scfit.training.Objective` for an objective.
"""

from __future__ import annotations

from typing import Any

from scfit.registry import Component

__all__ = ["ArchitectureConfig", "ObjectiveConfig"]


class ArchitectureConfig(Component):
    """A portable architecture recipe (unregistered family base). ``build(ctx) -> torch.nn.Module``."""

    def build(self, context: Any = None) -> Any:  # -> torch.nn.Module
        """Construct the architecture ``torch.nn.Module`` (implemented by concrete configs)."""
        raise NotImplementedError(f"{type(self).__name__} must implement build(self, context).")


class ObjectiveConfig(Component):
    """A portable objective recipe (unregistered family base). ``build(ctx) -> scfit.training.Objective``."""

    def build(self, context: Any = None) -> Any:  # -> Objective
        """Construct the :class:`scfit.training.Objective` (implemented by concrete configs)."""
        raise NotImplementedError(f"{type(self).__name__} must implement build(self, context).")
