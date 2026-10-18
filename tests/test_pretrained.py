"""A spec and its weights round-trip through a Hub-layout folder, locally or via a repo id."""

from __future__ import annotations

import pytest
import torch

from scfit.pretrained import load_pretrained, save_pretrained
from scfit.registry import Component, component


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
        assert f'- "{tag}"\n' in card


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


def test_a_missing_local_folder_is_not_sent_to_the_hub(tmp_path):
    for missing in (tmp_path / "nope", "./nope", "/nope"):
        with pytest.raises(FileNotFoundError, match="No saved folder"):
            load_pretrained(missing)


def test_leaves_a_folder_it_did_not_save_alone(tmp_path):
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "model.safetensors").write_text("not ours")
    save_pretrained(tmp_path, _Model())
    assert (tmp_path / "other" / "model.safetensors").read_text() == "not ours"


@pytest.mark.parametrize(
    ("states", "error"),
    [
        ({"../escape": {"w": torch.zeros(1)}}, ValueError),
        ({"a/b": {"w": torch.zeros(1)}}, ValueError),
        ({"module": {"w": 1}}, TypeError),
        ({"module": {"a": (w := torch.zeros(2)), "b": w}}, ValueError),
    ],
    ids=["escaping_name", "nested_name", "not_a_tensor", "tied"],
)
def test_a_bad_state_leaves_the_earlier_save_intact(tmp_path, states, error):
    save_pretrained(tmp_path, _Model(depth=2), {"module": {"w": torch.ones(1)}})
    with pytest.raises(error):
        save_pretrained(tmp_path, _Model(depth=3), states, overwrite=True)
    spec, loaded = load_pretrained(tmp_path)
    assert spec == _Model(depth=2)
    assert torch.equal(loaded["module"]["w"], torch.ones(1))
    assert not (tmp_path.parent / "escape").exists()
