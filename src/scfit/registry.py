"""The uniform component registry + portable-spec (de)serialization for scfit.

One :class:`Component` base for every registrable/portable thing — sub-components, top-level architectures
and objectives all use the same pattern. This is scfit's shared foundation: the flow toolbox (``sc_flow``),
foundation toolboxes and any other ecosystem plugin subclass it, so specs stay portable across packages.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self, TypeGuard, Union, get_args, get_origin, get_type_hints

# Deliberately cattrs, not pydantic/msgspec: this is a public foundation downstream packages subclass, so
# the dependency surface stays minimal and the on-disk format stable. It also coerces OmegaConf
# ListConfig/DictConfig leaf values natively. See the design decision for the full rationale.
import cattrs

__all__ = ["Component", "RngComponent", "PortabilityError", "to_spec", "parse", "register_live"]

type JSON = None | bool | int | float | str | list[JSON] | dict[str, JSON]
"""What a spec is made of."""

_REGISTRY: dict[str, type[Component]] = {}
_converter = cattrs.Converter(forbid_extra_keys=True)  # loud on a typo'd LEAF field
_hints_cache: dict[type, dict[str, object]] = {}


class PortabilityError(Exception):
    """Raised when a config holding a live runtime instance is asked for its portable spec."""


class Component:
    """Base for every registrable/portable config.

    A concrete subclass opts in by passing ``type_id=`` (and optionally ``version=`` / ``versions=``) in
    its class header — that single line auto-registers it. A subclass with **no** ``type_id`` is an
    *abstract family base* (e.g. ``Objective``, ``Combiner``) and is intentionally left unregistered so it
    can be used as the ``expected`` family in :func:`parse` / :meth:`from_spec`.

    A component is only the portable description. How it turns into a runtime object is up to its family:
    a family base declares its own typed ``build``, taking whatever runtime inputs that family needs.
    """

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]  # concrete configs are dataclasses
    __type_id__: ClassVar[str]
    __version__: ClassVar[int]  # stamped when WRITING a spec
    __versions__: ClassVar[frozenset[int]]  # accepted set when READING a spec

    def __init_subclass__(
        cls,
        *,
        type_id: str | None = None,
        version: int = 1,
        versions: tuple[int, ...] | None = None,
        **kw: Any,
    ) -> None:
        super().__init_subclass__(**kw)
        if type_id is None:
            return  # abstract family base — not portable on its own
        if not isinstance(type_id, str) or not type_id:
            raise TypeError("type_id must be a non-empty string.")
        accepted = frozenset(versions) if versions is not None else frozenset({version})
        if version not in accepted or any((not isinstance(v, int)) or isinstance(v, bool) or v <= 0 for v in accepted):
            raise ValueError(f"{type_id!r}: bad version/versions ({version}, {sorted(accepted)}).")
        existing = _REGISTRY.get(type_id)
        if existing is not None and existing is not cls:
            raise ValueError(f"type_id {type_id!r} already registered to {existing.__name__}.")
        cls.__type_id__, cls.__version__, cls.__versions__ = type_id, version, accepted
        _REGISTRY[type_id] = cls

    def to_spec(self) -> dict[str, JSON]:
        """Return this config as a portable ``{type, version, config}`` dict; raises on live instances."""
        return to_spec(self)

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Self:
        """Parse a spec into a config, enforcing that it is a ``cls`` (family-scoped, typed entry).

        Replaces the per-family ``validate_<family>_spec`` free functions: ``Combiner.from_spec(spec)``
        returns a validated ``Combiner``, and rejects a spec whose type is not a ``Combiner``.
        """
        return _structure_component(spec, cls)


class RngComponent(Component):
    """Family marker for a config whose ``build`` takes a keyword-only ``rng: numpy.random.Generator``.

    Such a config holds no seed. The caller derives ``rng`` from the run's seed, so every stream traces back
    to one place. A component needing several independent streams splits its ``rng`` with ``rng.spawn(n)``.
    """


def register_live(cls: type) -> type:
    """Mark a live runtime type as non-portable: exporting a config that holds one raises loudly.

    The escape hatch — a config field typed ``Family | LiveFamily`` trains/builds with a live instance, but
    :func:`to_spec` on that config raises :class:`PortabilityError` instead of silently dropping it.
    """

    def _raise(_obj: object) -> JSON:
        raise PortabilityError(
            f"{cls.__name__} is a runtime-only instance and has no portable config; pass a Component spec "
            f"instead of a live object to make this config serializable."
        )

    _converter.register_unstructure_hook(cls, _raise)
    return cls


def _field_types(cls: type) -> dict[str, object]:
    if cls not in _hints_cache:
        _hints_cache[cls] = get_type_hints(cls)
    return _hints_cache[cls]


def _is_component(tp: object) -> TypeGuard[type[Component]]:
    return isinstance(tp, type) and issubclass(tp, Component)


def _has_component(tp: object) -> bool:
    return _is_component(tp) or any(_has_component(a) for a in get_args(tp))


def _unstructure_field(value: object) -> JSON:
    if isinstance(value, Component):
        return to_spec(value)  # nested sub-component -> nested envelope
    if isinstance(value, Mapping):
        return {_converter.unstructure(k): _unstructure_field(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_unstructure_field(v) for v in value]
    return _converter.unstructure(value)  # leaves + live-instance guard (raises)


def to_spec(config: Component) -> dict[str, JSON]:
    """Config -> portable ``{type, version, config}`` dict (JSON-ready). Raises on live instances."""
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
        comp = next((a for a in args if _is_component(a)), None)
        if comp is not None and isinstance(value, Mapping) and "type" in value:
            return _structure_component(value, comp)  # always take the spec branch on load
        others = [a for a in args if a is not type(None) and not _is_component(a)]
        if len(others) == 1:
            return _structure_field(value, others[0])
        return _converter.structure(value, ftype)
    if _is_component(ftype):
        return _structure_component(value, ftype)
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
