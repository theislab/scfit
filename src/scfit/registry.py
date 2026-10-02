"""The component registry and portable specs for scfit, on pydantic.

A :class:`Component` is a frozen pydantic model, registered under a stable ``type_id`` with :func:`component`.
A spec is ``{"type": type_id, "version": n, **fields}``, plain JSON. Packages built on scfit subclass it, so a family stays
open: a spec's ``type`` picks the registered class, wherever a field names the family base.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, SerializationInfo, ValidationInfo, model_serializer, model_validator
from pydantic_core import PydanticSerializationError

__all__ = ["Component", "PortabilityError", "component", "config_of", "to_spec", "parse"]

_REGISTRY: dict[str, type[Component]] = {}
_CONFIGS: dict[type, type[Component]] = {}  # implementation -> the config that builds it
_UNLINKED: list[type[Component]] = []  # registered configs whose `build` return names a class not defined yet


class PortabilityError(Exception):
    """Raised when a config holds something a spec cannot, e.g. a live object."""


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
    __builds__: ClassVar[type | None] = None  # the class its own `build` is annotated to return

    @model_validator(mode="wrap")
    @classmethod
    def _dispatch(cls, value: Any, handler: Callable[[Any], Self], info: ValidationInfo) -> Self:
        """A spec's ``type`` picks the registered class, so a field typed as a family base stays open."""
        if not (isinstance(value, Mapping) and "type" in value):
            return handler(value)
        type_id, version = value["type"], value.get("version")
        fields = {k: v for k, v in value.items() if k not in ("type", "version")}
        if not isinstance(type_id, str):
            raise ValueError(f"A spec's type is a string, got {type_id!r}.")
        target = _REGISTRY.get(type_id)
        if target is None:
            raise ValueError(f"Unknown type {type_id!r}; registered: {sorted(_REGISTRY)}.")
        if not issubclass(target, cls):
            raise ValueError(f"type {type_id!r} ({target.__name__}) is not a {cls.__name__}.")
        if type(version) is not int or version not in target.__versions__:
            raise ValueError(f"Unsupported {type_id!r} version {version!r}; accepted: {sorted(target.__versions__)}.")
        return handler(fields) if target is cls else target.model_validate(fields, context=info.context)  # pyright: ignore[reportReturnType]

    @model_serializer(mode="wrap")
    def _tag(self, handler: Callable[[Self], dict[str, Any]], info: SerializationInfo) -> dict[str, Any]:
        if not info.serialize_as_any:  # else pydantic writes a member as its field's declared base, dropping fields
            # ponytail: include/exclude are not forwarded; forward them if a caller needs them on members
            return type(self).__pydantic_serializer__.to_python(
                self,
                mode=info.mode,
                by_alias=info.by_alias,
                exclude_unset=info.exclude_unset,
                exclude_defaults=info.exclude_defaults,
                exclude_none=info.exclude_none,
                round_trip=info.round_trip,
                context=info.context,
                serialize_as_any=True,
            )
        if self.__type_id__ is not None and _REGISTRY.get(self.__type_id__) is not type(self):
            raise TypeError(f"{type(self).__name__} is not registered; decorate it with @component(type_id).")
        fields = handler(self)
        return fields if self.__type_id__ is None else {"type": self.__type_id__, "version": self.__version__, **fields}

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
    type_id: str,
    *,
    version: int = 1,
    versions: tuple[int, ...] | None = None,
) -> Callable[[type[C]], type[C]]:
    """Register a `Component` subclass under ``type_id``, the stable name its specs carry.

    Specs are written at ``version``; ``versions`` lists every version one may be read at, e.g. ``(1, 2)``
    after adding a field with a default, so v1 specs still load.

    A ``build`` defined on the class links it to the class that ``build`` is annotated to return, so
    :func:`config_of` finds the config for an implementation. One config per implementation.
    """
    if not isinstance(type_id, str) or not type_id:
        raise TypeError("type_id must be a non-empty string.")
    accepted = frozenset(versions) if versions is not None else frozenset({version})
    if version not in accepted or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in accepted):
        raise ValueError(f"{type_id!r}: bad version/versions ({version!r}, {sorted(accepted)}).")

    def register(cls: type[C]) -> type[C]:
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise TypeError(f"@component needs a Component subclass, got {cls!r}.")
        if reserved := {"type", "version"} & cls.model_fields.keys():
            raise TypeError(f"{cls.__name__}: {sorted(reserved)} are reserved for the spec.")
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        cls.__type_id__, cls.__version__, cls.__versions__ = type_id, version, accepted
        _REGISTRY[type_id] = cls
        try:
            _link(cls)
        except NameError:  # e.g. the implementation is defined below its config; linked on first `config_of`
            _UNLINKED.append(cls)
        return cls

    return register


def config_of(implementation: type) -> type[Component]:
    """The config registered as building ``implementation``."""
    for cls in list(_UNLINKED):
        try:
            _link(cls)
            _UNLINKED.remove(cls)
        except NameError:
            pass
    try:
        return _CONFIGS[implementation]
    except KeyError:
        raise KeyError(f"no config builds {implementation.__name__}.") from None


def _link(cls: type[Component]) -> None:
    """Link ``cls`` to the class its own ``build`` returns; an inherited ``build`` links nothing."""
    build = cls.__dict__.get("build")
    if build is None:
        return
    target = inspect.get_annotations(build).get("return")
    if isinstance(target, str):  # only the return: other hints may be TYPE_CHECKING-only imports
        target = eval(target, build.__globals__)  # what `typing.get_type_hints` does for each hint
    if not isinstance(target, type):
        return
    linked = _CONFIGS.setdefault(target, cls)
    if linked is not cls:
        raise ValueError(f"{target.__name__} is already built by {linked.__name__}.")
    cls.__builds__ = target


def to_spec(config: Component) -> dict[str, Any]:
    """Config -> portable JSON spec. Typed ``Any`` like :func:`json.loads`: a spec is data to write out."""
    if config.__type_id__ is None or _REGISTRY.get(config.__type_id__) is not type(config):
        raise TypeError(f"{type(config).__name__} is not registered; decorate it with @component(type_id).")
    try:
        spec = config.model_dump(mode="json")
    except PydanticSerializationError as e:
        raise PortabilityError(f"{type(config).__name__} holds something a spec cannot: {e}") from None
    if type(config).model_validate(spec) != config:  # pydantic coerces e.g. NaN, sets, enums in `Any` fields
        raise PortabilityError(f"{type(config).__name__} does not read back from its spec; it holds non-JSON values.")
    return spec


def parse(spec: Mapping[str, object]) -> Component:
    """A spec -> the registered :class:`Component` its ``type`` names."""
    return Component.from_spec(spec)
