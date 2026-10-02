"""Registry smoke tests: portable-spec round-trip, unknown-field rejection, live-instance guard.

The registry is exercised end-to-end by the downstream flow/foundation toolboxes; these keep scfit's own
suite honest about the public :mod:`scfit.registry` surface.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict, Unpack

import attrs
import pytest

from scfit.params import Default, ParamsComponent
from scfit.registry import Builds, Component, PortabilityError, component, config_of, parse, to_spec


@component("test.widget")
class _Widget(Component):
    width: int = 3

    def build(self):
        return self.width * 2


def test_round_trips_through_portable_spec():
    spec = _Widget(width=5).to_spec()
    assert spec == {"type": "test.widget", "version": 1, "config": {"width": 5}}
    rebuilt = parse(spec)
    assert isinstance(rebuilt, _Widget)
    assert rebuilt.build() == 10


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="Unknown field"):
        parse({"type": "test.widget", "version": 1, "config": {"nope": 1}})


def test_unknown_type_rejected():
    with pytest.raises(ValueError, match="Unknown type"):
        parse({"type": "test.nonexistent", "version": 1, "config": {}})


@component("test.holder")
class _Holder(Component):
    obj: object = None


@pytest.mark.parametrize("live", [object(), len, {("a",): 1}], ids=["instance", "callable", "tuple_key"])
def test_anything_but_components_and_json_has_no_spec(live):
    with pytest.raises(PortabilityError, match="portable"):
        to_spec(_Holder(obj=live))


def test_json_data_is_portable():
    assert to_spec(_Holder(obj={"a": [1, 2.5, None, "x", True]}))["config"]["obj"] == {"a": [1, 2.5, None, "x", True]}


class _Enc(Component):
    pass


@component("test.one_hot")
class _OneHot(_Enc):
    categories: list[str] | None = None


@component("test.label")
class _Label(_Enc):
    categories: list[str] | None = None


@component("test.encoders")
class _Encoders(Component):
    by_key: dict[str, _Enc] | None = None
    ordered: list[_Enc] = dataclasses.field(default_factory=list)
    pair: tuple[_Enc, _Enc] | None = None
    many: tuple[_Enc, ...] = ()


def test_components_in_containers_keep_their_type():
    config = _Encoders(
        by_key={"a": _OneHot(), "b": _Label(categories=["x"])},
        ordered=[_Label(), _OneHot()],
        pair=(_OneHot(), _Label()),
        many=(_Label(),),
    )
    spec = config.to_spec()
    assert spec["config"]["by_key"]["a"]["type"] == "test.one_hot"
    assert parse(spec) == config


def test_unregistered_subclass_has_no_spec():
    class _Unregistered(_Widget):
        pass

    with pytest.raises(TypeError, match="not registered"):
        _Unregistered().to_spec()


def test_components_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        _Widget().width = 4  # type: ignore[misc]


def test_older_accepted_version_still_loads():
    @component("test.grown", version=2, versions=(1, 2))
    class _Grown(Component):
        width: int = 3
        height: int = 1

    old = {"type": "test.grown", "version": 1, "config": {"width": 5}}
    assert parse(old) == _Grown(width=5)
    assert _Grown().to_spec()["version"] == 2
    with pytest.raises(ValueError, match="Unsupported"):
        parse({**old, "version": 3})


class _WidgetParams(TypedDict, total=False):
    enc: Annotated[_Enc, Default(_OneHot())]
    width: Annotated[int, Default(3)]


@component("test.with_params")
class _WithParams(ParamsComponent[_WidgetParams]):
    pass


def test_params_pin_defaults_and_keep_nested_types():
    config = _WithParams(params={"enc": _Label(categories=["x"])})
    spec = config.to_spec()
    assert spec["config"]["params"]["width"] == 3  # the default is written into the spec
    assert spec["config"]["params"]["enc"]["type"] == "test.label"
    assert _WithParams.from_spec(spec) == config


def test_params_reject_unknown_keys():
    with pytest.raises(ValueError, match="Unknown"):
        _WithParams(params={"widht": 4})  # pyright: ignore[reportArgumentType]  (the typo is the point)
    with pytest.raises(ValueError, match="Unknown"):
        _WithParams.from_spec({"type": "test.with_params", "version": 1, "config": {"params": {"widht": 4}}})


@component("test.node")
class _Node(Component):
    children: list[_Node] = dataclasses.field(default_factory=list)


def test_a_component_may_name_itself():
    tree = _Node(children=[_Node(), _Node(children=[_Node()])])
    assert parse(tree.to_spec()) == tree


@component("test.either")
class _Either(Component):
    enc: _OneHot | _Label | None = None


def test_a_union_of_components_parses_each_member():
    assert parse(_Either(enc=_Label()).to_spec()) == _Either(enc=_Label())


class _Str(str):
    pass


@dataclasses.dataclass
class _Plain:
    x: int = 1


@attrs.define
class _PlainAttrs:
    x: int = 1


class _Color(enum.Enum):
    RED = "red"


@pytest.mark.parametrize(
    "value",
    [_Str("a"), float("nan"), _Plain(), _PlainAttrs(), _Color.RED, {"c": _Color.RED}, {1: "a"}],
    ids=["str_subclass", "nan", "dataclass", "attrs", "enum", "enum_in_dict", "int_key"],
)
def test_only_exact_json_and_components_are_portable(value):
    with pytest.raises(PortabilityError):
        to_spec(_Holder(obj=value))


def test_tuples_are_written_as_lists():
    assert to_spec(_Holder(obj=(1, "a")))["config"]["obj"] == [1, "a"]


class _Impl:
    def __init__(self, **params: Unpack[_WidgetParams]) -> None:
        self.params = params


@component("test.linked", builds=_Impl)
class _Linked(ParamsComponent[_WidgetParams]):
    pass


class _OtherParams(TypedDict, total=False):
    width: Annotated[int, Default(3)]


class _OtherImpl:
    def __init__(self, **params: Unpack[_OtherParams]) -> None: ...


@component("test.mislinked", builds=_OtherImpl)
class _Mislinked(ParamsComponent[_WidgetParams]):
    pass


def test_config_is_found_from_its_implementation():
    assert config_of(_Impl) is _Linked
    assert _Linked.__builds__ is _Impl


def test_params_must_match_what_the_implementation_unpacks():
    with pytest.raises(TypeError, match="unpacks _OtherParams"):
        _Mislinked()


def test_one_config_per_implementation():
    with pytest.raises(ValueError, match="already built by"):

        @component("test.second", builds=_Impl)
        class _Second(ParamsComponent[_WidgetParams]):
            pass


@component()
class _Intermediate[Q: Mapping[str, Any]](ParamsComponent[Q]):
    pass


@component("test.deep")
class _Deep(_Intermediate[_WidgetParams]):
    pass


def test_params_type_resolves_through_generic_bases():
    assert _Deep().params.get("width") == 3
    assert _Deep.from_spec(_Deep(params={"width": 5}).to_spec()) == _Deep(params={"width": 5})


class _Thing:
    pass


class _ThingFamily[T](Builds[T], Component):
    def build(self) -> T:
        raise NotImplementedError


@component("test.thing")
class _ThingConfig(_ThingFamily[_Thing]):
    def build(self) -> _Thing:
        return _Thing()


def test_generic_argument_gives_builds():
    assert _ThingConfig.__builds__ is _Thing
    assert config_of(_Thing) is _ThingConfig


def test_builds_must_agree_with_the_generic_argument():
    with pytest.raises(TypeError, match=r"bind Builds\[_Thing\]"):

        @component("test.disagree", builds=_Impl)
        class _Disagree(_ThingFamily[_Thing]):
            pass


def test_fields_of_an_undecorated_base_are_not_dropped_silently():
    class _Undecorated(Component):
        sigma: float = 0.0

    with pytest.raises(TypeError, match=r"_Undecorated declares \['sigma'\]"):

        @component("test.dropped")
        class _Child(_Undecorated):
            eps: float = 1.0


def test_a_field_less_family_base_needs_no_decorator():
    class _Family(Component):
        pass

    @component("test.kept")
    class _Child(_Family):
        eps: float = 1.0

    assert parse(_Child(eps=2.0).to_spec()) == _Child(eps=2.0)
