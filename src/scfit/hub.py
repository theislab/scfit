"""Portable model bundles — the family-neutral substrate for sharing models (e.g. on the Hugging Face Hub).

A *bundle* is a directory with two files:

- ``model.safetensors`` — the model weights (a torch ``state_dict``), stored without pickled Python.
- ``config.json`` — a JSON envelope naming the ``family`` that can rebuild the model and each
  :class:`scfit.registry.Component` that describes it (as portable ``to_spec()`` dicts), plus any
  ``extra`` JSON artifacts the family needs (e.g. a gene vocabulary).

This module owns only the read/write of that bundle; it imports and builds **no** family. Reconstructing a
live model from a bundle (``from_pretrained``) and Hub push/pull build on top and land next — they dispatch
on ``config["family"]`` through the :mod:`scfit.families` entry-point registry, keeping this layer neutral.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from scfit.registry import Component, parse

__all__ = ["BUNDLE_FORMAT", "load_bundle", "save_pretrained"]

#: Bundle envelope version, bumped only on a breaking change to the on-disk layout.
BUNDLE_FORMAT = 1
_CONFIG_NAME = "config.json"
_WEIGHTS_NAME = "model.safetensors"


def save_pretrained(
    model: torch.nn.Module,
    path: str | Path,
    *,
    family: str,
    components: dict[str, Component],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``model``'s weights and its component specs to a portable bundle directory, and return it.

    ``family`` names the :mod:`scfit.families` builder that can rebuild the model; ``components`` maps each
    model slot (e.g. ``"encoder"``, ``"objective"``) to the :class:`~scfit.registry.Component` describing it,
    serialized via ``to_spec()``; ``extra`` holds any JSON-serializable side artifacts the family needs.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), str(path / _WEIGHTS_NAME))
    config = {
        "format": BUNDLE_FORMAT,
        "family": family,
        "components": {name: component.to_spec() for name, component in components.items()},
        "extra": extra or {},
    }
    (path / _CONFIG_NAME).write_text(json.dumps(config, indent=2))
    return path


def load_bundle(path: str | Path) -> tuple[dict[str, Any], dict[str, Component], dict[str, torch.Tensor]]:
    """Read a bundle written by :func:`save_pretrained`, family-neutrally.

    Returns the raw ``config`` dict, the parsed ``{slot: Component}`` map (each spec run back through
    :func:`scfit.registry.parse`), and the weight ``state_dict``. It builds no family — turning these into a
    live model is ``from_pretrained``'s job.
    """
    path = Path(path)
    config = json.loads((path / _CONFIG_NAME).read_text())
    fmt = config.get("format")
    if fmt != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {fmt!r}; this scfit reads format {BUNDLE_FORMAT}.")
    components = {name: parse(spec) for name, spec in config.get("components", {}).items()}
    state_dict = load_file(str(path / _WEIGHTS_NAME))
    return config, components, state_dict
