"""scfit — the family-neutral core (registry + streaming data).

``import scfit`` stays light and **torch-free**: only ``registry`` (cattrs) loads eagerly. ``data`` (the
annbatch streaming stack, ``scfit[data]``) loads lazily on first attribute access or explicit submodule
import, so a consumer that only wants the component registry never drags in the data deps.
"""

from __future__ import annotations

from . import registry

__all__ = ["registry", "data"]

_LAZY = frozenset({"data"})


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
