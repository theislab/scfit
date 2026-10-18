"""A spec and its weights round-trip through a Hub-layout folder, locally or via a repo id."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from scfit.pretrained import load_pretrained, save_pretrained  # noqa: E402
from scfit.registry import Component, component  # noqa: E402


@component("test.pretrained.head")
class _Head(Component):
    width: int = 2


@component("test.pretrained.model")
class _Model(Component):
    head: _Head = _Head()
    depth: int = 1


def test_round_trips_spec_and_states(tmp_path):
    spec, weights = _Model(depth=3), {"weight": torch.randn(2, 3)}
    save_pretrained(tmp_path, spec, {"module": weights})

    loaded, states = load_pretrained(tmp_path)
    assert loaded == spec
    assert states.keys() == {"module"}
    assert torch.equal(states["module"]["weight"], weights["weight"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["README.md", "config.json", "module"]


def test_model_card_tags_every_type(tmp_path):
    save_pretrained(tmp_path, _Model())
    card = (tmp_path / "README.md").read_text()
    assert "library_name: scfit" in card
    for tag in ("scfit", "test.pretrained.model", "test.pretrained.head"):
        assert f"- {tag}\n" in card


def test_refuses_to_overwrite(tmp_path):
    save_pretrained(tmp_path, _Model())
    with pytest.raises(FileExistsError):
        save_pretrained(tmp_path, _Model())
    save_pretrained(tmp_path, _Model(depth=2), overwrite=True)
    assert load_pretrained(tmp_path)[0] == _Model(depth=2)


def test_overwrite_drops_states_of_the_earlier_save(tmp_path):
    save_pretrained(tmp_path, _Model(), {"module": {"w": torch.zeros(1)}, "old": {"w": torch.ones(1)}})
    save_pretrained(tmp_path, _Model(), {"module": {"w": torch.ones(1)}}, overwrite=True)
    assert load_pretrained(tmp_path)[1].keys() == {"module"}


def test_a_repo_id_is_downloaded(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    save_pretrained(tmp_path, _Model(depth=4))
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda repo_id, revision=None: calls.append((repo_id, revision)) or tmp_path,
    )
    assert load_pretrained("org/model", revision="v1")[0] == _Model(depth=4)
    assert calls == [("org/model", "v1")]
