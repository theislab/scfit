"""The component registry and portable specs for scfit, on pydantic.

A :class:`Component` is a frozen pydantic model, registered under a stable ``type_id`` with :func:`component`.
A spec is ``{"type": type_id, "version": n, **fields}``, plain JSON. Packages built on scfit subclass it, so a family stays
open: a spec's ``type`` picks the registered class, wherever a field names the family base.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self, get_args, get_origin

from pydantic import BaseModel, ConfigDict, SerializationInfo, model_serializer, model_validator
from pydantic_core import PydanticSerializationError

__all__ = ["Builds", "Component", "PortabilityError", "component", "config_of", "to_spec", "parse"]

_REGISTRY: dict[str, type[Component]] = {}
_CONFIGS: dict[type, type[Component]] = {}  # implementation -> the config that builds it


class PortabilityError(Exception):
    """Raised when a config holds something a spec cannot, e.g. a live object."""


class Builds[T]:
    """Generic marker: a config whose bases bind ``Builds[X]`` builds ``X``.

    A family base declares it once, ``class PathConfig[T: Path](Component, Builds[T])`` with
    ``def build(self) -> T``; a config then names its implementation in its generic argument,
    ``class LinearConfig(PathConfig[LinearPath])``, which types ``build`` and gives :func:`component` the link.
    """


class Component(BaseModel):
    """Base for every portable config. Register a concrete one with :func:`component`.

    A subclass left unregistered is a family base, usable as the expected family in :meth:`from_spec` and as
    a field type: a spec's ``type`` then picks the registered member. How a config turns into a runtime
    object is up to its family, which declares its own typed ``build``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    __type_id__: ClassVar[str | None] = None
    __version__: ClassVar[int] = 1  # written
    __versions__: ClassVar[frozenset[int]] = frozenset({1})  # accepted when read
    __builds__: ClassVar[type | None] = None  # the implementation `build` returns, set by `component`

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch(cls, value: Any, handler: Callable[[Any], Self]) -> Self:
        """A spec's ``type`` picks the registered class, so a field typed as a family base stays open."""
        if not (isinstance(value, Mapping) and "type" in value):
            return handler(value)
        type_id, version = value["type"], value.get("version")
        fields = {k: v for k, v in value.items() if k not in ("type", "version")}
        target = _REGISTRY.get(type_id)
        if target is None:
            raise ValueError(f"Unknown type {type_id!r}; registered: {sorted(_REGISTRY)}.")
        if not issubclass(target, cls):
            raise ValueError(f"type {type_id!r} ({target.__name__}) is not a {cls.__name__}.")
        if version not in target.__versions__:
            raise ValueError(f"Unsupported {type_id!r} version {version!r}; accepted: {sorted(target.__versions__)}.")
        return handler(fields) if target is cls else target.model_validate(fields)  # pyright: ignore[reportReturnType]

    @model_serializer(mode="wrap")
    def _tag(self, handler: Callable[[Self], dict[str, Any]], info: SerializationInfo) -> dict[str, Any]:
        fields = handler(self)
        return fields if self.__type_id__ is None else {"type": self.__type_id__, "version": self.__version__, **fields}

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """As pydantic's, but every component is written as its own class, not the field's declared base."""
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump(**kwargs)

    def model_dump_json(self, **kwargs: Any) -> str:
        """As pydantic's, with every component written as its own class."""
        kwargs.setdefault("serialize_as_any", True)
        return super().model_dump_json(**kwargs)

    def to_spec(self) -> dict[str, Any]:
        """Return the portable JSON spec; raises :class:`PortabilityError` on anything that is not JSON."""
        return to_spec(self)

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Self:
        """Parse a spec into a config, rejecting one whose type is not a ``cls``."""
        if not (isinstance(spec, Mapping) and "type" in spec):
            raise ValueError("A spec is a mapping with a 'type'.")
        return cls.model_validate(spec)


def component[C: Component](
    type_id: str | None = None,
    *,
    builds: type | None = None,
    version: int = 1,
    versions: tuple[int, ...] | None = None,
) -> Callable[[type[C]], type[C]]:
    """Register a `Component` subclass under ``type_id``, the stable name its specs carry.

    Specs are written at ``version``; ``versions`` lists every version one may be read at, e.g. ``(1, 2)``
    after adding a field with a default, so v1 specs still load.

    ``builds`` names the implementation the config's ``build`` returns, so :func:`config_of` finds the config for
    a class; a base binding :class:`Builds` gives it without the keyword. One config per implementation.
    """
    if type_id is not None and (not isinstance(type_id, str) or not type_id):
        raise TypeError("type_id must be a non-empty string.")
    accepted = frozenset(versions) if versions is not None else frozenset({version})
    if version not in accepted or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in accepted):
        raise ValueError(f"{type_id!r}: bad version/versions ({version!r}, {sorted(accepted)}).")

    def register(cls: type[C]) -> type[C]:
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise TypeError(f"@component needs a Component subclass, got {cls!r}.")
        if type_id is None:
            return cls
        bound = _type_arguments(cls).get(Builds.__type_params__[0])
        bound = bound if isinstance(bound, type) else None
        if builds is not None and bound is not None and builds is not bound:
            raise TypeError(f"{cls.__name__}: builds={builds.__name__} but its bases bind Builds[{bound.__name__}].")
        target = builds or bound
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        linked = _CONFIGS.get(target) if target is not None else None
        if linked is not None and linked is not cls:
            raise ValueError(f"{target.__name__} is already built by {linked.__name__}.")  # pyright: ignore[reportOptionalMemberAccess]
        cls.__type_id__, cls.__version__, cls.__versions__, cls.__builds__ = type_id, version, accepted, target
        _REGISTRY[type_id] = cls
        if target is not None:
            _CONFIGS[target] = cls
        return cls

    return register


def config_of(implementation: type) -> type[Component]:
    """The config registered as building ``implementation``."""
    try:
        return _CONFIGS[implementation]
    except KeyError:
        raise KeyError(f"no config builds {implementation.__name__}.") from None


def _type_arguments(cls: type) -> dict[object, object]:
    """``{type parameter: argument}`` bound in ``cls``'s bases, through pydantic's parametrized classes too."""
    bound: dict[object, object] = {}
    for klass in cls.__mro__:  # most derived first, so an argument that is itself a parameter resolves
        meta = klass.__dict__.get("__pydantic_generic_metadata__") or {}
        if meta.get("origin") is not None:  # e.g. `PathConfig[Linear]`, which pydantic makes a class
            for parameter, argument in zip(
                meta["origin"].__pydantic_generic_metadata__["parameters"], meta["args"], strict=False
            ):
                bound[parameter] = bound.get(argument, argument)
        for base in klass.__dict__.get("__orig_bases__", ()):
            for parameter, argument in zip(
                getattr(get_origin(base), "__type_params__", ()), get_args(base), strict=False
            ):
                bound[parameter] = bound.get(argument, argument)
    return bound


def to_spec(config: Component) -> dict[str, Any]:
    """Config -> portable JSON spec. Typed ``Any`` like :func:`json.loads`: a spec is data to write out."""
    if config.__type_id__ is None or _REGISTRY.get(config.__type_id__) is not type(config):
        raise TypeError(f"{type(config).__name__} is not registered; decorate it with @component(type_id).")
    try:
        return config.model_dump(mode="json")
    except PydanticSerializationError as e:
        raise PortabilityError(f"{type(config).__name__} holds something a spec cannot: {e}") from None


def parse(spec: Mapping[str, object]) -> Component:
    """A spec -> the registered :class:`Component` its ``type`` names."""
    return Component.from_spec(spec)
