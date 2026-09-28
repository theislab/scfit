"""The uniform component registry + portable-spec (de)serialization for scfit.

One :class:`Component` base for every registrable/portable thing — sub-components, top-level architectures
and objectives all use the same pattern. This is scfit's shared foundation: the flow toolbox (``sc_flow``),
foundation toolboxes and any other ecosystem plugin subclass it, so specs stay portable across packages.
"""

from __future__ import annotations

import dataclasses
import math
import types
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    ClassVar,
    Self,
    TypeGuard,
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
import cattrs

__all__ = ["Component", "PortabilityError", "component", "config_of", "to_spec", "parse"]

_REGISTRY: dict[str, type[Component]] = {}
_CONFIGS: dict[type, type[Component]] = {}  # implementation -> the config that builds it
_converter = cattrs.Converter(forbid_extra_keys=True)  # loud on a typo'd LEAF field
_hints_cache: dict[type, dict[str, object]] = {}


class PortabilityError(Exception):
    """Raised when a config holds something other than components and JSON data, e.g. a live object."""


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
    __builds__: ClassVar[type | None] = None  # the implementation `build` returns, set by `component(builds=)`

    def to_spec(self) -> dict[str, Any]:
        """Return this config as a portable ``{type, version, config}`` dict; raises on live instances."""
        return to_spec(self)

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Self:
        """Parse a spec into a config, enforcing that it is a ``cls`` (family-scoped, typed entry).

        Replaces the per-family ``validate_<family>_spec`` free functions: ``Combiner.from_spec(spec)``
        returns a validated ``Combiner``, and rejects a spec whose type is not a ``Combiner``.
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
    for a class. One config per implementation.

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
        if type_id is None:
            return cls
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        cls.__type_id__, cls.__version__, cls.__versions__ = type_id, version, accepted
        _REGISTRY[type_id] = cls
        if builds is not None:
            linked = _CONFIGS.get(builds)
            if linked is not None and linked is not cls:
                raise ValueError(f"{builds.__name__} is already built by {linked.__name__}.")
            cls.__builds__ = builds
            _CONFIGS[builds] = cls
        return cls

    return register


def config_of(implementation: type) -> type[Component]:
    """The config registered with ``builds=implementation``."""
    try:
        return _CONFIGS[implementation]
    except KeyError:
        raise KeyError(f"no config builds {implementation.__name__}; register one with builds=.") from None


def _field_types(cls: type) -> dict[str, object]:
    if cls not in _hints_cache:
        _hints_cache[cls] = get_type_hints(cls)
    return _hints_cache[cls]


def _is_component(tp: object) -> TypeGuard[type[Component]]:
    return isinstance(tp, type) and issubclass(tp, Component)


def _has_component(tp: object) -> bool:
    return _is_component(tp) or any(_has_component(a) for a in get_args(tp))


def _is_json(value: object) -> bool:
    """Exact JSON types only, so a subclass is never written as its base and read back as something else."""
    if type(value) is float:
        return math.isfinite(value)
    if value is None or type(value) in (bool, int, str):
        return True
    if type(value) is list:
        return all(_is_json(v) for v in value)
    return type(value) is dict and all(type(k) is str and _is_json(v) for k, v in value.items())


def _unstructure_field(value: object) -> object:
    if isinstance(value, Component):
        return to_spec(value)  # nested sub-component -> nested envelope
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise PortabilityError(f"dict keys must be strings to be portable; got {sorted(map(repr, value))}.")
        return {k: _unstructure_field(v) for k, v in value.items()}
    if type(value) in (list, tuple):
        return [_unstructure_field(v) for v in value]  # pyright: ignore[reportGeneralTypeIssues]
    if dataclasses.is_dataclass(value):  # a shape of its own must be a component, so its type is recorded
        raise PortabilityError(f"{type(value).__name__} is a dataclass, not a registered component; make it one.")
    leaf = _converter.unstructure(value)
    if not _is_json(leaf):  # portable means components and JSON data; anything else is a live object
        raise PortabilityError(
            f"{type(value).__name__} has no portable form: a spec holds only components and JSON data. "
            "Pass a component instead of a live object to make this config serializable."
        )
    return leaf


def to_spec(config: Component) -> dict[str, Any]:
    """Config -> portable ``{type, version, config}`` dict (JSON-ready). Raises on live instances.

    Typed ``Any`` like :func:`json.loads`: a spec is data to write out, not to index into.
    """
    if _REGISTRY.get(getattr(config, "__type_id__", "")) is not type(config):
        raise TypeError(f"{type(config).__name__} is not registered; decorate it with @component(type_id).")
    inner = {f.name: _unstructure_field(getattr(config, f.name)) for f in dataclasses.fields(config)}
    return {"type": config.__type_id__, "version": config.__version__, "config": inner}


def _structure_field(value: object, ftype: Any) -> object:
    """Structure one spec value as the annotation ``ftype``, e.g. ``dict[str, Enc] | None``.

    ``ftype`` is ``Any`` because typing cannot yet express a type form (PEP 747 ``TypeForm``).
    """
    origin, args = get_origin(ftype), get_args(ftype)
    if origin is Union or origin is types.UnionType:  # incl. Optional and the spec|live escape hatch
        if value is None and type(None) in args:
            return None
        comps = [a for a in args if _is_component(a)]
        if comps and isinstance(value, Mapping) and "type" in value:  # always take the spec branch on load
            target = _REGISTRY.get(value["type"])
            expected = next((c for c in comps if target is not None and issubclass(target, c)), comps[0])
            return _structure_component(value, expected)
        others = [a for a in args if a is not type(None) and not _is_component(a)]
        if len(others) == 1:
            return _structure_field(value, others[0])
        return _converter.structure(value, ftype)
    if _is_component(ftype):
        return _structure_component(value, ftype)
    if is_typeddict(ftype) and isinstance(value, Mapping):  # a params bag: each key keeps its own type
        hints = get_type_hints(ftype)
        unknown = set(value) - set(hints)
        if unknown:
            raise ValueError(f"Unknown {ftype.__name__} key(s): {sorted(unknown)}; expected from {sorted(hints)}.")
        return {k: _structure_field(v, hints[k]) for k, v in value.items()}
    if args and _has_component(ftype):  # a container of components: recurse so each keeps its type
        if isinstance(origin, type) and issubclass(origin, Mapping) and isinstance(value, Mapping):
            key_type, value_type = args
            return {_converter.structure(k, key_type): _structure_field(v, value_type) for k, v in value.items()}
        if origin is tuple and isinstance(value, Sequence) and not (len(args) == 2 and args[1] is Ellipsis):
            return tuple(_structure_field(v, t) for v, t in zip(value, args, strict=True))
        if isinstance(origin, type) and issubclass(origin, Sequence) and isinstance(value, Sequence):
            items = [_structure_field(v, args[0]) for v in value]
            return tuple(items) if origin is tuple else items
    return _converter.structure(value, ftype)  # cattrs does the leaf recursion + OmegaConf coercion


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
    field_types = _field_types(target)
    known = {f.name for f in dataclasses.fields(target)}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"Unknown field(s) for {type_id!r}: {sorted(unknown)}; allowed: {sorted(known)}.")
    kwargs = {name: _structure_field(cfg[name], field_types[name]) for name in cfg}
    return target(**kwargs)  # __post_init__ runs here (validation + canonicalization)


def parse(spec: Mapping[str, object]) -> Component:
    """``{type, version, config}`` dict -> validated :class:`Component` (dispatched by the registry)."""
    return _structure_component(spec, Component)
