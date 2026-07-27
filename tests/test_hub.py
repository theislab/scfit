"""Portable bundle round-trip: save_pretrained -> load_bundle recovers specs, extra, and weights exactly.

Family-neutral layer only (no family build, no Hugging Face) — that is intentionally deferred to the
``from_pretrained`` / push-pull work that builds on this substrate.
"""

from __future__ import annotations

import dataclasses
import json

import pytest
import torch

from scfit.hub import BUNDLE_FORMAT, load_bundle, save_pretrained
from scfit.registry import Component


@dataclasses.dataclass
class _Enc(Component, type_id="test.hub_enc", version=1):
    dim: int = 4

    def build(self, context=None):
        return torch.nn.Linear(self.dim, self.dim)


def test_bundle_round_trips_specs_extra_and_weights(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    cfg = _Enc(dim=4)

    out = save_pretrained(model, tmp_path, family="test_family", components={"encoder": cfg}, extra={"note": "x"})
    assert (out / "model.safetensors").exists()
    assert (out / "config.json").exists()

    config, components, state_dict = load_bundle(tmp_path)
    assert config["format"] == BUNDLE_FORMAT
    assert config["family"] == "test_family"
    assert config["extra"] == {"note": "x"}
    assert components["encoder"] == cfg  # spec parsed back into an equal Component
    for key, value in model.state_dict().items():
        assert torch.equal(state_dict[key], value)  # weights recovered exactly


def test_load_bundle_rejects_unknown_format(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"format": 999, "family": "x", "components": {}}))
    with pytest.raises(ValueError, match="unsupported bundle format"):
        load_bundle(tmp_path)  # format is checked before any weights are read
