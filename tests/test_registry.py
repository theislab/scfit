"""The registry's public surface: specs, open families, portability, and the config-to-implementation link."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError, model_validator

from scfit.registry import Component, PortabilityError, component, config_of, parse, to_spec


@component("test.widget")
class _Widget(Component):
    width: int = 3


def test_round_trips_through_a_json_spec():
    spec = _Widget(width=5).to_spec()
    assert spec == {"type": "test.widget", "version": 1, "width": 5}
    assert parse(json.loads(json.dumps(spec))) == _Widget(width=5)


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ({"type": "test.widget", "version": 1, "nope": 1}, "Extra inputs"),
        ({"type": "test.nonexistent"}, "Unknown type"),
        ({"type": "test.widget", "version": 1, "width": "wide"}, "valid integer"),
        ({"type": "test.widget", "version": 2}, "Unsupported"),
        ({"type": "test.widget", "version": True}, "Unsupported"),
        ({"type": ["test.widget"], "version": 1}, "is a string"),
    ],
    ids=["typo_field", "unknown_type", "bad_value", "bad_version", "bool_version", "list_type"],
)
def test_bad_specs_are_rejected(spec, match):
    with pytest.raises(ValidationError, match=match):
        parse(spec)


@component("test.positive")
class _Positive(Component):
    n: int = 1

    @model_validator(mode="after")
    def _check(self) -> _Positive:
        if self.n <= 0:
            raise ValueError("n must be positive")
        return self


def test_copies_and_construction_are_validated():
    with pytest.raises(ValidationError, match="positive"):
        _Positive().model_copy(update={"n": 0})
    with pytest.raises(ValidationError, match="positive"):
        _Positive.model_construct(n=0)
    assert _Positive().model_copy(update={"n": 2}) == _Positive(n=2)


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
    assert spec["by_key"]["a"] == {
        "type": "test.one_hot",
        "version": 1,
        "categories": ["x"],
    }  # not cut to the base's fields
    assert parse(json.loads(json.dumps(spec))) == config


def test_a_family_field_needs_a_type():
    with pytest.raises(ValidationError, match="family base"):
        _Encoders.model_validate({"one": {"categories": ("x",)}})


def test_a_spec_of_the_wrong_family_is_rejected():
    with pytest.raises(ValidationError, match="is not a _Enc"):
        _Encoders.from_spec({"type": "test.encoders", "version": 1, "one": {"type": "test.widget", "version": 1}})


@component("test.node")
class _Node(Component):
    children: list[_Node] = []


def test_a_component_may_name_itself():
    tree = _Node(children=[_Node(), _Node(children=[_Node()])])
    assert parse(tree.to_spec()) == tree


@component("test.holder")
class _Holder(Component):
    obj: Any = None


@pytest.mark.parametrize(
    "live",
    [object(), len, float("nan"), {1, 2}, {1: "a"}, b"ab"],
    ids=["instance", "callable", "nan", "set", "int_key", "bytes"],
)
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


def test_a_nested_unregistered_subclass_has_no_spec():
    class _Unregistered(_OneHot):
        pass

    with pytest.raises(PortabilityError, match="not registered"):
        _Encoders(one=_Unregistered()).to_spec()


def test_a_member_keeps_its_fields_under_any_pydantic_entry_point():
    from pydantic import BaseModel, TypeAdapter

    class _Plain(BaseModel):
        enc: _Enc

    member = _OneHot(categories=("x",))
    assert _Plain.model_validate_json(_Plain(enc=member).model_dump_json()).enc == member
    assert TypeAdapter(list[_Enc]).dump_python([member])[0]["categories"] == ("x",)


def test_type_and_version_are_reserved():
    with pytest.raises(TypeError, match="reserved"):

        @component("test.reserved")
        class _Reserved(Component):
            version: int = 1


def test_type_ids_are_unique():
    with pytest.raises(ValueError, match="already registered"):

        @component("test.widget")
        class _Again(Component):
            pass


class _Thing:
    pass


@component("test.thing", builds=_Thing)
class _ThingConfig(Component):
    size: int = 1

    def build(self) -> _Thing:
        return _Thing()


def test_builds_links_config_and_implementation():
    assert _ThingConfig.__builds__ is _Thing
    assert config_of(_Thing) is _ThingConfig


def test_one_config_per_implementation():
    with pytest.raises(ValueError, match="already built by"):

        @component("test.second", builds=_Thing)
        class _Second(Component):
            def build(self) -> _Thing:
                return _Thing()


def test_older_accepted_version_still_loads():
    @component("test.grown", version=2, versions=(1, 2))
    class _Grown(Component):
        width: int = 3
        height: int = 1

    assert parse({"type": "test.grown", "version": 1, "width": 5}) == _Grown(width=5)
    assert _Grown().to_spec()["version"] == 2
