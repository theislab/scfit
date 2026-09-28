"""Parameter bags as TypedDicts: each key declares its default with ``Annotated[type, Default(value)]``.

A class takes ``**params: Unpack[XParams]`` and calls :func:`resolve_params` once; its config subclasses
:class:`ParamsComponent`, narrows ``params: XParams`` and forwards it with ``**``. The TypedDict is then the
only place a parameter is declared: its type, its default and its docstring.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any, cast, get_args, get_type_hints

from scfit.registry import Component, component

__all__ = ["Default", "ParamsComponent", "defaults_of", "resolve_init_params", "resolve_params", "validates"]


@dataclass(frozen=True, slots=True)
class Default:
    """A params key's default, declared as ``Annotated[type, Default(value)]``."""

    value: Any


def defaults_of[T: Mapping[str, Any]](spec: type[T]) -> T:
    """The :class:`Default` of every key of ``spec`` that has one. A key without one is required."""
    defaults = {}
    for key, hint in get_type_hints(spec, include_extras=True).items():
        marker = next((m for m in getattr(hint, "__metadata__", ()) if isinstance(m, Default)), None)
        if marker is not None:
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
    keys = set(get_type_hints(spec))
    unknown = set(params or ()) - keys
    if unknown:
        raise ValueError(f"Unknown {spec.__name__} key(s): {sorted(unknown)}; expected from {sorted(keys)}.")
    missing = keys - set(defaults) - set(params or ())
    if missing:
        raise ValueError(f"Missing required {spec.__name__} key(s): {sorted(missing)}.")
    merged = {**defaults, **(params or {})}
    if (validate := _VALIDATORS.get(spec)) is not None:
        validate(merged)
    return cast("T", merged)


@cache
def _init_spec(cls: type) -> type:
    return get_args(get_type_hints(cls.__init__)["params"])[0]  # `**params: Unpack[XParams]` -> XParams


def resolve_init_params(obj: object, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve ``params`` against the TypedDict that ``type(obj).__init__`` takes as ``**params: Unpack[...]``.

    Called from a base ``__init__``, it resolves against the most derived class's params, so a subclass that
    extends them only annotates its own ``__init__``.
    """
    return resolve_params(params, _init_spec(type(obj)))


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
