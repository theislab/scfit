"""Save and load a spec with its weights, in the Hugging Face Hub layout.

A folder holds ``config.json`` (the spec), one ``<name>/model.safetensors`` per state and a ``README.md``
model card. A folder reads back under the package versions that wrote it.

Share one with ``huggingface_hub.HfApi``: ``create_repo(repo_id, exist_ok=True)``, then
``upload_folder(repo_id=repo_id, folder_path=path)``; :func:`load_pretrained` takes the repo id.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from scfit.registry import Component, parse

__all__ = ["save_pretrained", "load_pretrained"]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


def save_pretrained(
    path: str | Path,
    spec: Component,
    states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Writes ``spec`` and each named state (e.g. ``{"module": module.state_dict()}``) to the folder ``path``.

    Everything is checked before the folder is touched. Tensors sharing memory (tied weights) are refused.

    :raises FileExistsError: If ``path`` already holds a spec and ``overwrite`` is `False`.
    :raises ValueError: If a state name is not a plain folder name, or a state ties tensors.
    :raises scfit.registry.PortabilityError: If the spec holds a live object.
    """
    out = Path(path)
    document = spec.to_spec()
    tensors = {name: _checked(name, state) for name, state in (states or {}).items()}
    if (out / CONFIG_NAME).exists():
        if not overwrite:
            raise FileExistsError(f"{out / CONFIG_NAME} already exists. Use overwrite=True.")
        for stale in out.glob(f"*/{WEIGHTS_NAME}"):  # a state left from the earlier save would load back with this one
            stale.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG_NAME).write_text(json.dumps(document, indent=2))
    for name, state in tensors.items():
        (out / name).mkdir(exist_ok=True)
        save_file(state, out / name / WEIGHTS_NAME)
    (out / "README.md").write_text(_model_card(document))


def load_pretrained(
    path_or_repo_id: str | Path, *, revision: str | None = None
) -> tuple[Component, dict[str, dict[str, torch.Tensor]]]:
    """The spec and the named states saved in a local folder, or in a Hub repo (``"org/name"``)."""
    src = Path(path_or_repo_id)
    if not src.is_dir():
        if isinstance(path_or_repo_id, Path) or src.exists() or str(path_or_repo_id).startswith((".", "/", "~")):
            raise FileNotFoundError(f"No saved folder at {src}.")
        src = Path(snapshot_download(str(path_or_repo_id), revision=revision))
    spec = parse(json.loads((src / CONFIG_NAME).read_text()))
    return spec, {f.parent.name: load_file(f) for f in src.glob(f"*/{WEIGHTS_NAME}")}


def _checked(name: str, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``state`` as contiguous tensors, after checking it can be written under ``name``."""
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"State name {name!r} is not a plain folder name.")
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise TypeError(f"State {name!r} holds values that are not tensors.")
    storages = [v.untyped_storage().data_ptr() for v in state.values() if v.numel()]
    if len(storages) != len(set(storages)):
        raise ValueError(f"State {name!r} ties tensors (shared memory), which safetensors cannot store.")
    return {k: v.contiguous() for k, v in state.items()}


def _model_card(document: Mapping[str, Any]) -> str:
    tags = "\n".join(f"- {json.dumps(tag)}" for tag in ["scfit", *sorted(_types(document))])
    return f"---\nlibrary_name: scfit\ntags:\n{tags}\n---\n\nLoad with `scfit.pretrained.load_pretrained(repo_id)`.\n"


def _types(value: Any) -> set[str]:
    """Every component type a spec names, nested ones included."""
    if isinstance(value, Mapping):
        own = {value["type"]} if isinstance(value.get("type"), str) else set()
        return own.union(*(_types(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_types(v) for v in value))
    return set()
