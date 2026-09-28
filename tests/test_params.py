from __future__ import annotations

from typing import Annotated, Required, TypedDict

import pytest

from scfit.params import Default, defaults_of, resolve_params, validates


class _Params(TypedDict, total=False):
    size: Annotated[int, Default(4)]
    name: Annotated[str, Default("a")]


@validates(_Params)
def _check(p: dict) -> None:
    if p["size"] <= 0:
        raise ValueError("size must be positive")


def test_resolve_merges_over_defaults():
    assert resolve_params({"size": 8}, _Params) == {"size": 8, "name": "a"}
    assert resolve_params(None, _Params) == defaults_of(_Params)


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
