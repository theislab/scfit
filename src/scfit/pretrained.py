"""Save, load and share a spec with its weights, in the Hugging Face Hub layout.

A folder holds ``config.json`` (the spec), one ``<name>/model.safetensors`` per state and a ``README.md``
model card, so it can be pushed to the Hub as is. A folder reads back under the package versions that wrote it.
Needs ``scfit[hub]``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scfit.registry import Component, parse

if TYPE_CHECKING:
    import torch

__all__ = ["save_pretrained", "load_pretrained", "push_to_hub"]

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

    :raises FileExistsError: If ``path`` already holds a spec and ``overwrite`` is `False`.
    :raises scfit.registry.PortabilityError: If the spec holds a live object, before anything is written.
    """
    from safetensors.torch import save_file

    out = Path(path)
    document = spec.to_spec()
    if not overwrite and (out / CONFIG_NAME).exists():
        raise FileExistsError(f"{out / CONFIG_NAME} already exists. Use overwrite=True.")
    out.mkdir(parents=True, exist_ok=True)
    (out / CONFIG_NAME).write_text(json.dumps(document, indent=2))
    for name, state in (states or {}).items():
        (out / name).mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in state.items()}, out / name / WEIGHTS_NAME)
    (out / "README.md").write_text(_model_card(document))


def load_pretrained(
    path_or_repo_id: str | Path, *, revision: str | None = None
) -> tuple[Component, dict[str, dict[str, torch.Tensor]]]:
    """The spec and the named states saved in a local folder, or in a Hub repo (``"org/name"``)."""
    from safetensors.torch import load_file

    src = Path(path_or_repo_id)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(str(path_or_repo_id), revision=revision))
    spec = parse(json.loads((src / CONFIG_NAME).read_text()))
    return spec, {f.parent.name: load_file(f) for f in src.glob(f"*/{WEIGHTS_NAME}")}


def push_to_hub(path: str | Path, repo_id: str, *, private: bool | None = None, token: str | None = None) -> str:
    """Uploads the folder ``path`` to the Hub repo ``repo_id``, creating it if needed; returns the commit url."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    return str(api.upload_folder(repo_id=repo_id, folder_path=path))


def _model_card(document: Mapping[str, Any]) -> str:
    tags = "\n".join(f"- {tag}" for tag in ["scfit", *sorted(_types(document))])
    return f"---\nlibrary_name: scfit\ntags:\n{tags}\n---\n\nLoad with `scfit.pretrained.load_pretrained(repo_id)`.\n"


def _types(value: Any) -> set[str]:
    """Every component type a spec names, nested ones included."""
    if isinstance(value, Mapping):
        own = {value["type"]} if isinstance(value.get("type"), str) else set()
        return own.union(*(_types(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_types(v) for v in value))
    return set()
