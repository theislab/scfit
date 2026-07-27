"""scfit — the family-neutral core (registry, families, training, data, metrics).

``import scfit`` stays light and **torch-free**: only ``registry`` (cattrs) and ``families`` (stdlib) load
eagerly, so ``scfit.families.available_families()`` never drags in torch. ``data`` / ``metrics`` / ``training``
(which import torch/lightning) load lazily on first attribute access or explicit submodule import.
"""

from __future__ import annotations

from . import families, registry

__all__ = ["registry", "families", "nn", "data", "metrics", "training"]

_LAZY = frozenset({"nn", "data", "metrics", "training"})


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
