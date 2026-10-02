"""The uniform component registry + portable-spec (de)serialization for scfit.

One :class:`Component` base for every registrable/portable thing — sub-components, top-level architectures
and objectives all use the same pattern. This is scfit's shared foundation: the flow toolbox (``sc_flow``),
foundation toolboxes and any other ecosystem plugin subclass it, so specs stay portable across packages.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import math
import types
from collections.abc import Callable, Mapping
from functools import cache
from typing import (
    Any,
    ClassVar,
    Self,
    TypeGuard,
    TypeVar,
    Union,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

# Deliberately cattrs, not pydantic/msgspec: this is a public foundation downstream packages subclass, so
# the dependency surface stays minimal and the on-disk format stable. It also coerces OmegaConf
# ListConfig/DictConfig leaf values natively. See the design decision for the full rationale.
import attrs
import cattrs

__all__ = ["Builds", "Component", "PortabilityError", "component", "config_of", "to_spec", "parse"]

_REGISTRY: dict[str, type[Component]] = {}
_CONFIGS: dict[type, type[Component]] = {}  # implementation -> the config that builds it
# loud on a typo'd leaf field; errors raised as-is rather than grouped, so callers see the ValueError
_converter = cattrs.Converter(forbid_extra_keys=True, detailed_validation=False)


class PortabilityError(Exception):
    """Raised when a config holds something other than components and JSON data, e.g. a live object."""


class Builds[T]:
    """Generic marker: a config whose bases bind ``Builds[X]`` builds ``X``.

    A family base declares it once, e.g. ``class PathConfig[T: Path](Builds[T], Component)`` with
    ``def build(self) -> T``; each config then names its implementation only in its generic argument,
    ``class LinearConfig(PathConfig[LinearPath])``, which types ``build`` and gives :func:`component` its
    ``builds``.
    """


class Component:
    """Base for every portable config.

    Register a concrete config with :func:`component`, the only way to register one. A subclass left
    unregistered is a *family base* (e.g. ``Objective``, ``Combiner``), usable as the ``expected`` family
    in :meth:`from_spec`.

    A component is only the portable description. How it turns into a runtime object is up to its family:
    a family base declares its own typed ``build``, taking whatever runtime inputs that family needs.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]  # set by `component`
    __type_id__: ClassVar[str]
    __version__: ClassVar[int]  # written
    __versions__: ClassVar[frozenset[int]]  # accepted when read
    __builds__: ClassVar[type | None] = None  # the implementation `build` returns, set by `component`

    def to_spec(self) -> dict[str, Any]:
        """Return this config as a portable ``{type, version, config}`` dict; raises on live instances."""
        return to_spec(self)

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Self:
        """Parse a spec into a config, enforcing that it is a ``cls``.

        ``Combiner.from_spec(spec)`` returns a validated ``Combiner``, and rejects a spec whose type is not one.
        """
        return _structure_component(spec, cls)


@dataclass_transform(frozen_default=True, kw_only_default=True)
def component[C: Component](
    type_id: str | None = None,
    *,
    builds: type | None = None,
    version: int = 1,
    versions: tuple[int, ...] | None = None,
) -> Callable[[type[C]], type[C]]:
    """Make a `Component` subclass a frozen, keyword-only dataclass and register it under ``type_id``.

    Without ``type_id`` the class is a family base with fields, left unregistered. Field annotations are
    resolved on first use, so a field may name its own class or one defined later in the module.

    ``builds`` names the implementation the config's ``build`` returns, so :func:`config_of` finds the config
    for a class; a base binding :class:`Builds` gives it without the keyword. One config per implementation.

    Specs are written at ``version``. ``versions`` lists every version one may be read at, e.g. ``(1, 2)``
    after adding a field with a default, so v1 specs still load.
    """
    if type_id is not None and (not isinstance(type_id, str) or not type_id):
        raise TypeError("type_id must be a non-empty string.")
    accepted = frozenset(versions) if versions is not None else frozenset({version})
    if version not in accepted or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in accepted):
        raise ValueError(f"{type_id!r}: bad version/versions ({version!r}, {sorted(accepted)}).")

    def register(cls: type[C]) -> type[C]:
        if not (isinstance(cls, type) and issubclass(cls, Component)):
            raise TypeError(f"@component needs a Component subclass, got {cls!r}.")
        cls = dataclasses.dataclass(frozen=True, kw_only=True)(cls)
        _check_no_dropped_fields(cls)
        if type_id is None:
            return cls
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        cls.__type_id__, cls.__version__, cls.__versions__ = type_id, version, accepted
        _REGISTRY[type_id] = cls
        bound = _type_arguments(cls).get(Builds.__type_params__[0])
        bound = bound if isinstance(bound, type) else None
        if builds is not None and bound is not None and builds is not bound:
            raise TypeError(f"{cls.__name__}: builds={builds.__name__} but its bases bind Builds[{bound.__name__}].")
        target = builds or bound
        if target is not None:
            linked = _CONFIGS.get(target)
            if linked is not None and linked is not cls:
                raise ValueError(f"{target.__name__} is already built by {linked.__name__}.")
            cls.__builds__ = target
            _CONFIGS[target] = cls
        return cls

    return register


def _check_no_dropped_fields(cls: type) -> None:
    """Raise if a Component base declares fields that did not become fields of ``cls``.

    A base without ``@component()`` is not a dataclass, so its annotations are silently left out of every
    subclass: they cannot be passed and never reach the spec. Names starting with ``_`` are metadata.
    """
    fields = {f.name for f in dataclasses.fields(cls)}
    for base in cls.__mro__[1:]:
        if issubclass(base, Component) and base is not Component:
            dropped = [n for n in inspect.get_annotations(base) if not n.startswith("_") and n not in fields]
            if dropped:
                raise TypeError(
                    f"{base.__name__} declares {dropped}, which are not fields of {cls.__name__}; "
                    f"decorate {base.__name__} with @component()."
                )


def config_of(implementation: type) -> type[Component]:
    """The config registered with ``builds=implementation``."""
    try:
        return _CONFIGS[implementation]
    except KeyError:
        raise KeyError(f"no config builds {implementation.__name__}; register one with builds=.") from None


def _type_arguments(cls: type) -> dict[object, object]:
    """``{type parameter: argument}`` bound anywhere in ``cls``'s bases, e.g. ``ParamsComponent[X]`` gives ``P: X``."""
    bound: dict[object, object] = {}
    for klass in cls.__mro__:  # most derived first, so an argument that is itself a parameter resolves
        for base in klass.__dict__.get("__orig_bases__", ()):
            parameters = getattr(get_origin(base), "__type_params__", ())
            for parameter, argument in zip(parameters, get_args(base), strict=False):
                bound[parameter] = bound.get(argument, argument)
    return bound


@cache
def _field_types(cls: type) -> dict[str, object]:
    """``cls``'s field annotations, with the type arguments of generic bases filled in."""
    bound = _type_arguments(cls)
    return {k: bound.get(v, v) if isinstance(v, TypeVar) else v for k, v in get_type_hints(cls).items()}


def _is_component(tp: object) -> TypeGuard[type[Component]]:
    return isinstance(tp, type) and issubclass(tp, Component)


def _to_json(value: object) -> object:
    """Exact JSON types only, so a subclass is never written as its base and read back as something else."""
    if type(value) in (list, tuple):
        return [_to_json(v) for v in value]  # pyright: ignore[reportGeneralTypeIssues]
    if type(value) is dict:
        return {k: _to_json(v) for k, v in _str_keyed(value).items()}
    if value is None or type(value) in (bool, int, str) or (type(value) is float and math.isfinite(value)):
        return value
    raise PortabilityError(
        f"{type(value).__name__} has no portable form: a spec holds only components and JSON data. "
        "Pass a component instead of a live object to make this config serializable."
    )


def _str_keyed[M: Mapping[Any, Any]](mapping: M) -> M:
    if not all(type(k) is str for k in mapping):
        raise PortabilityError(f"dict keys must be strings to be portable; got {sorted(map(repr, mapping))}.")
    return mapping


def _envelope(config: Component) -> dict[str, Any]:
    if _REGISTRY.get(getattr(config, "__type_id__", "")) is not type(config):
        raise TypeError(f"{type(config).__name__} is not registered; decorate it with @component(type_id).")
    inner = {f.name: _converter.unstructure(getattr(config, f.name)) for f in dataclasses.fields(config)}
    return {"type": config.__type_id__, "version": config.__version__, "config": inner}


def to_spec(config: Component) -> dict[str, Any]:
    """Config -> portable ``{type, version, config}`` dict (JSON-ready). Raises on live instances.

    Typed ``Any`` like :func:`json.loads`: a spec is data to write out, not to index into.
    """
    return _to_json(_envelope(config))  # pyright: ignore[reportReturnType]  (cattrs passes unknown objects through as-is; caught here)


def _reject_shape(value: object) -> object:
    raise PortabilityError(f"{type(value).__name__} has fields but is not a registered component; make it one.")


def _reject_enum(value: enum.Enum) -> object:
    # written as its value, it would read back as that value wherever the field is not typed as the enum
    raise PortabilityError(f"{type(value).__name__} is an enum; use a Literal of its values instead.")


def _structure_union(value: object, union: Any) -> object:
    """A union with components: the spec's ``type`` picks the member, so sibling components stay apart."""
    args = get_args(union)
    if value is None and type(None) in args:
        return None
    comps = [a for a in args if _is_component(a)]
    if isinstance(value, Mapping) and "type" in value:
        target = _REGISTRY.get(value["type"])
        expected = next((c for c in comps if target is not None and issubclass(target, c)), comps[0])
        return _structure_component(value, expected)
    others = tuple(a for a in args if a is not type(None) and not _is_component(a))
    if not others:
        raise TypeError(f"Expected a spec mapping for {union}; found {type(value).__name__}.")
    rest: Any = Union[others]  # noqa: UP007  (built from a tuple; Any as typing has no TypeForm, PEP 747)
    return _converter.structure(value, rest)


def _structure_params(value: object, spec: Any) -> dict[str, Any]:
    """A params bag may be partial (its defaults are filled on construction), so it is not cattrs' TypedDict."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{spec.__name__} must be a mapping; found {type(value).__name__}.")
    hints = get_type_hints(spec)
    unknown = set(value) - set(hints)
    if unknown:
        raise ValueError(f"Unknown {spec.__name__} key(s): {sorted(unknown)}; expected from {sorted(hints)}.")
    return {k: _converter.structure(v, hints[k]) for k, v in value.items()}


def _structure_component[C: Component](spec: object, expected: type[C]) -> C:
    if not isinstance(spec, Mapping):
        raise TypeError(f"Expected a spec mapping; found {type(spec).__name__}.")
    unknown_env = set(spec) - {"type", "version", "config"}
    if unknown_env:
        raise ValueError(f"Unknown spec field(s): {sorted(unknown_env)}.")
    missing = {"type", "version", "config"} - set(spec)
    if missing:
        raise ValueError(f"Missing spec field(s): {sorted(missing)}.")
    type_id, version, cfg = spec["type"], spec["version"], spec["config"]
    if not isinstance(type_id, str) or not type_id:
        raise TypeError("spec 'type' must be a non-empty string.")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise TypeError("spec 'version' must be a positive integer.")
    if not isinstance(cfg, Mapping):
        raise TypeError("spec 'config' must be a mapping.")
    try:
        target = _REGISTRY[type_id]
    except KeyError:
        raise ValueError(f"Unknown type {type_id!r}; registered: {sorted(_REGISTRY)}.") from None
    if not issubclass(target, expected):
        raise ValueError(f"type {type_id!r} ({target.__name__}) is not a {expected.__name__}.")
    if version not in target.__versions__:
        raise ValueError(f"Unsupported {type_id!r} config version {version}; accepted: {sorted(target.__versions__)}.")
    field_types: dict[str, Any] = _field_types(target)  # type forms, typed Any until PEP 747
    known = {f.name for f in dataclasses.fields(target)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"Unknown field(s) for {type_id!r}: {sorted(unknown)}; allowed: {sorted(known)}.")
    kwargs = {name: _converter.structure(cfg[name], field_types[name]) for name in cfg}
    return target(**kwargs)  # __post_init__ runs here (validation + canonicalization)


# cattrs walks dicts, lists, tuples and optionals; these hooks add only what is ours. Later hooks win.
# keys are checked before cattrs unstructures them, which would turn a tuple key into an unhashable list
_converter.register_unstructure_hook(dict, lambda d: {k: _converter.unstructure(v) for k, v in _str_keyed(d).items()})
_converter.register_unstructure_hook_func(
    lambda t: (dataclasses.is_dataclass(t) or attrs.has(t)) and not _is_component(t), _reject_shape
)
_converter.register_unstructure_hook_func(lambda t: isinstance(t, type) and issubclass(t, enum.Enum), _reject_enum)
_converter.register_unstructure_hook_func(_is_component, _envelope)  # by the runtime class, so subclasses keep theirs
_converter.register_structure_hook_func(is_typeddict, _structure_params)
_converter.register_structure_hook_func(
    lambda t: get_origin(t) in (Union, types.UnionType) and any(_is_component(a) for a in get_args(t)),
    _structure_union,
)
_converter.register_structure_hook_func(_is_component, _structure_component)


def parse(spec: Mapping[str, object]) -> Component:
    """``{type, version, config}`` dict -> validated :class:`Component` (dispatched by the registry)."""
    return _structure_component(spec, Component)
