"""Registry smoke tests: portable-spec round-trip, unknown-field rejection, live-instance guard.

The registry is exercised end-to-end by the downstream flow/foundation toolboxes; these keep scfit's own
suite honest about the public :mod:`scfit.registry` surface.
"""

from __future__ import annotations

import dataclasses

import pytest

from scfit.registry import Component, PortabilityError, RngComponent, parse, register_live, to_spec


@dataclasses.dataclass
class _Widget(Component, type_id="test.widget", version=1):
    width: int = 3

    def build(self, context=None):
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


def test_live_instance_has_no_portable_spec():
    @register_live
    class _Live:
        pass

    @dataclasses.dataclass
    class _Holder(Component, type_id="test.holder", version=1):
        obj: object = None

        def build(self, context=None):
            return self.obj

    with pytest.raises(PortabilityError):
        to_spec(_Holder(obj=_Live()))


class _Enc(Component):
    pass


@dataclasses.dataclass
class _OneHot(_Enc, type_id="test.one_hot", version=1):
    categories: list[str] | None = None


@dataclasses.dataclass
class _Label(_Enc, type_id="test.label", version=1):
    categories: list[str] | None = None


@dataclasses.dataclass
class _Encoders(Component, type_id="test.encoders", version=1):
    by_key: dict[str, _Enc] | None = None
    ordered: list[_Enc] = dataclasses.field(default_factory=list)
    pair: tuple[_Enc, _Enc] | None = None
    many: tuple[_Enc, ...] = ()


def test_components_in_containers_keep_their_type():
    config = _Encoders(
        by_key={"a": _OneHot(), "b": _Label(["x"])},
        ordered=[_Label(), _OneHot()],
        pair=(_OneHot(), _Label()),
        many=(_Label(),),
    )
    spec = config.to_spec()
    assert spec["config"]["by_key"]["a"]["type"] == "test.one_hot"
    assert parse(spec) == config


def test_rng_component_builds_from_the_given_rng():
    np = pytest.importorskip("numpy")

    @dataclasses.dataclass
    class _Draw(RngComponent, type_id="test.draw", version=1):
        n: int = 3

        def build(self, context=None, *, rng):
            return rng.random(self.n)

    config = parse(_Draw().to_spec())
    assert isinstance(config, RngComponent)
    np.testing.assert_array_equal(config.build(rng=np.random.default_rng(0)), np.random.default_rng(0).random(3))
    with pytest.raises(TypeError):
        config.build()
