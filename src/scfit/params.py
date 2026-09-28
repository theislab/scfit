"""Parameter bags as TypedDicts: each key declares its default with ``Annotated[type, Default(value)]``.

A class takes ``**params: Unpack[XParams]`` and calls :func:`resolve_params` once; its config subclasses
:class:`ParamsComponent`, narrows ``params: XParams`` and forwards it with ``**``. The TypedDict is then the
only place a parameter is declared: its type, its default and its docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, cast, get_type_hints

from scfit.registry import Component, component, required_keys

__all__ = ["Default", "ParamsComponent", "defaults_of", "resolve_params", "validates"]


@dataclass(frozen=True, slots=True)
class Default:
    """A params key's default, declared as ``Annotated[type, Default(value)]``."""

    value: Any


def defaults_of[T: Mapping[str, Any]](spec: type[T]) -> T:
    """The :class:`Default` of every optional key of ``spec``; raises if one has none.

    A ``Required[...]`` key has no default: :func:`resolve_params` demands it instead.
    """
    defaults = {}
    required = required_keys(spec)
    for key, hint in get_type_hints(spec, include_extras=True).items():
        if key in required:
            continue
        marker = next((m for m in getattr(hint, "__metadata__", ()) if isinstance(m, Default)), None)
        if marker is None:
            raise TypeError(f"`{spec.__name__}.{key}` is missing a `Default(...)` in its annotation.")
        defaults[key] = marker.value
    return cast("T", defaults)


_cached_defaults = cache(defaults_of)  # only merged from, never handed out, so sharing it is safe
_VALIDATORS: dict[type, Callable[[dict[str, Any]], None]] = {}


def validates[F: Callable[[dict[str, Any]], None]](spec: type) -> Callable[[F], F]:
    """Register the decorated function as ``spec``'s validator, run by :func:`resolve_params`.

    It may coerce the merged mapping in place, and raises on invalid values.
    """

    def register(validate: F) -> F:
        _VALIDATORS[spec] = validate
        return validate

    return register


def resolve_params[T: Mapping[str, Any]](params: Mapping[str, Any] | None, spec: type[T]) -> T:
    """Merge ``params`` over the defaults of ``spec`` and validate the result. Unknown keys raise."""
    defaults = _cached_defaults(spec)
    if params is not None and not isinstance(params, Mapping):
        raise TypeError(f"params must be a mapping or None; got {type(params).__name__}.")
    required = required_keys(spec)
    unknown = set(params or ()) - set(defaults) - required
    if unknown:
        known = sorted({*defaults, *required})
        raise ValueError(f"Unknown {spec.__name__} key(s): {sorted(unknown)}; expected from {known}.")
    missing = required - set(params or ())
    if missing:
        raise ValueError(f"Missing required {spec.__name__} key(s): {sorted(missing)}.")
    merged = {**defaults, **(params or {})}
    if (validate := _VALIDATORS.get(spec)) is not None:
        validate(merged)
    return cast("T", merged)


@component()
class ParamsComponent(Component):
    """A component whose fields are one params bag. Subclasses narrow ``params`` to their TypedDict.

    The params are resolved on construction, so the spec records every value, defaults included, and a
    later change of a default never alters what an old spec builds.
    """

    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spec = get_type_hints(type(self))["params"]
        object.__setattr__(self, "params", resolve_params(self.params, spec))
