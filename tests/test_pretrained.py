"""A spec and its weights round-trip through a Hub-layout folder, locally or via a repo id."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from scfit.pretrained import load_pretrained, push_to_hub, save_pretrained  # noqa: E402
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


def test_a_repo_id_is_downloaded(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    save_pretrained(tmp_path, _Model(depth=4))
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, revision=None: calls.append((repo_id, revision)) or tmp_path
    )
    assert load_pretrained("org/model", revision="v1")[0] == _Model(depth=4)
    assert calls == [("org/model", "v1")]


def test_push_uploads_the_folder(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = []

    class _Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, private=None, exist_ok=False):
            calls.append(("create", repo_id, private, exist_ok))

        def upload_folder(self, repo_id, folder_path):
            calls.append(("upload", repo_id, folder_path))
            return "https://huggingface.co/org/model/commit/abc"

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert push_to_hub(tmp_path, "org/model", private=True).endswith("/commit/abc")
    assert calls == [("create", "org/model", True, True), ("upload", "org/model", tmp_path)]
