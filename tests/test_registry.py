"""The registry's public surface: specs, open families, portability, and the config-to-implementation link."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from scfit.registry import Builds, Component, PortabilityError, component, config_of, parse, to_spec


@component("test.widget")
class _Widget(Component):
    width: int = 3


def test_round_trips_through_a_json_spec():
    spec = _Widget(width=5).to_spec()
    assert spec == {"type": "test.widget", "width": 5}
    assert parse(json.loads(json.dumps(spec))) == _Widget(width=5)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"type": "test.widget", "nope": 1}, "Extra inputs"),
        ({"type": "test.nonexistent"}, "Unknown type"),
        ({"type": "test.widget", "width": "wide"}, "valid integer"),
    ],
    ids=["typo_field", "unknown_type", "bad_value"],
)
def test_bad_specs_are_rejected(spec, match):
    with pytest.raises(ValidationError, match=match):
        parse(spec)


def test_components_are_frozen():
    with pytest.raises(ValidationError):
        _Widget().width = 4  # pyright: ignore[reportAttributeAccessIssue]  (the point)


class _Enc(Component):  # an open family: members register anywhere
    pass


@component("test.one_hot")
class _OneHot(_Enc):
    categories: tuple[str, ...] | None = None


@component("test.label")
class _Label(_Enc):
    classes: tuple[str, ...] | None = None


@component("test.encoders")
class _Encoders(Component):
    by_key: dict[str, _Enc] = {}
    ordered: list[_Enc] = []
    one: _Enc | None = None
    either: _OneHot | _Label | None = None


def test_a_family_field_keeps_each_members_class_and_fields():
    config = _Encoders(
        by_key={"a": _OneHot(categories=("x",)), "b": _Label()},
        ordered=[_Label(classes=("y",))],
        one=_OneHot(),
        either=_Label(),
    )
    spec = config.to_spec()
    assert spec["by_key"]["a"] == {"type": "test.one_hot", "categories": ["x"]}  # not cut to the base's fields
    assert parse(json.loads(json.dumps(spec))) == config


def test_a_spec_of_the_wrong_family_is_rejected():
    with pytest.raises(ValidationError, match="is not a _Enc"):
        _Encoders.from_spec({"type": "test.encoders", "one": {"type": "test.widget"}})


@component("test.node")
class _Node(Component):
    children: list[_Node] = []


def test_a_component_may_name_itself():
    tree = _Node(children=[_Node(), _Node(children=[_Node()])])
    assert parse(tree.to_spec()) == tree


@component("test.holder")
class _Holder(Component):
    obj: Any = None


@pytest.mark.parametrize("live", [object(), len], ids=["instance", "callable"])
def test_live_objects_have_no_spec(live):
    with pytest.raises(PortabilityError):
        to_spec(_Holder(obj=live))


def test_non_finite_floats_are_rejected():
    @component("test.finite")
    class _Finite(Component):
        x: float = 0.0

    with pytest.raises(ValidationError):
        _Finite(x=float("nan"))


def test_an_unregistered_subclass_has_no_spec():
    class _Unregistered(_Widget):
        pass

    with pytest.raises(TypeError, match="not registered"):
        _Unregistered().to_spec()


def test_type_ids_are_unique():
    with pytest.raises(ValueError, match="already registered"):

        @component("test.widget")
        class _Again(Component):
            pass


class _Thing:
    pass


class _ThingFamily[T](Component, Builds[T]):
    def build(self) -> T:
        raise NotImplementedError


@component("test.thing")
class _ThingConfig(_ThingFamily[_Thing]):
    size: int = 1

    def build(self) -> _Thing:
        return _Thing()


def test_the_generic_argument_links_config_and_implementation():
    assert _ThingConfig.__builds__ is _Thing
    assert config_of(_Thing) is _ThingConfig
    assert parse(_ThingConfig(size=2).to_spec()) == _ThingConfig(size=2)


def test_builds_must_agree_with_the_generic_argument():
    with pytest.raises(TypeError, match=r"bind Builds\[_Thing\]"):

        @component("test.disagree", builds=_Widget)
        class _Disagree(_ThingFamily[_Thing]):
            pass


def test_one_config_per_implementation():
    with pytest.raises(ValueError, match="already built by"):

        @component("test.second")
        class _Second(_ThingFamily[_Thing]):
            pass
