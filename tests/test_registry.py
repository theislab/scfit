"""Registry smoke tests: portable-spec round-trip, unknown-field rejection, live-instance guard.

The registry is exercised end-to-end by the downstream flow/foundation toolboxes; these keep scfit's own
suite honest about the public :mod:`scfit.registry` surface.
"""

from __future__ import annotations

import dataclasses

import pytest

from scfit.registry import Component, PortabilityError, parse, register_live, to_spec


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
