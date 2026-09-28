from __future__ import annotations

from typing import Annotated, Required, TypedDict, Unpack

import pytest

from scfit.params import Default, resolve_init_params, resolve_params, validates


class _Params(TypedDict, total=False):
    size: Annotated[int, Default(4)]
    name: Annotated[str, Default("a")]


@validates(_Params)
def _check(p: dict) -> None:
    if p["size"] <= 0:
        raise ValueError("size must be positive")


def test_resolve_merges_over_defaults():
    assert resolve_params({"size": 8}, _Params) == {"size": 8, "name": "a"}
    assert resolve_params(None, _Params) == {"size": 4, "name": "a"}


def test_validator_runs_on_merged_params():
    with pytest.raises(ValueError, match="positive"):
        resolve_params({"size": 0}, _Params)


class _WithRequired(TypedDict, total=False):
    size: Required[int]
    name: Annotated[str, Default("a")]


def test_required_key_has_no_default_and_must_be_given():
    assert resolve_params({"size": 2}, _WithRequired) == {"size": 2, "name": "a"}
    with pytest.raises(ValueError, match="Missing required"):
        resolve_params({}, _WithRequired)


class _Base:
    def __init__(self, **params: Unpack[_Params]) -> None:
        self.params = resolve_init_params(self, params)


class _More(_Params, total=False):
    extra: Annotated[bool, Default(True)]


class _Derived(_Base):
    def __init__(self, **params: Unpack[_More]) -> None:
        super().__init__(**params)


def test_init_params_resolve_against_the_most_derived_spec():
    assert _Base().params == {"size": 4, "name": "a"}
    assert _Derived(size=2).params == {"size": 2, "name": "a", "extra": True}


class _WithList(TypedDict, total=False):
    xs: Annotated[list[int], Default([])]


def test_mutable_defaults_are_not_shared():
    resolve_params(None, _WithList)["xs"].append(1)
    assert resolve_params(None, _WithList)["xs"] == []
