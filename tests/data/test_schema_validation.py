"""Structural validation of the scheme spec — Node / Bind / Scheme / config-map guards, no data read.

These are the data-free ``__post_init__`` (and ``_resolve_config_map``) checks: they only assert the
*shape* of a scheme is accepted or rejected, so they build no :class:`~scfit.data.Loader` and touch no cell
matrix. Every scheme is spelled out inline — the graph under test is right there in the case.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import anndata as ad
import numpy as np
import pytest

from scfit.data import Bind, Node, SamplerConfig, Scheme
from scfit.data._schema import _resolve_config_map

SRC = ad.AnnData(np.zeros((2, 2), dtype="float32"))  # placeholder source; Scheme never reads it
COLS = ("cell_line", "drug")
CFG = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)


def _node(source: str = "data", cols=COLS, **kw) -> Node:
    return Node(source, cols, **kw)


# ── Node ──────────────────────────────────────────────────────────────────────────────────────────


def test_node_normalizes_keys_to_tuple():
    assert Node("data", COLS, "X").keys == ("X",)  # a bare loc string becomes a 1-tuple
    assert Node("data", COLS, ("X", "obsm/rep")).keys == ("X", "obsm/rep")


def test_node_weights_default_is_independent_per_instance():
    a, b = Node("data", COLS), Node("data", COLS)
    assert a.weights == {} and b.weights == {} and a.weights is not b.weights


def test_node_is_frozen():
    with pytest.raises(FrozenInstanceError):
        _node().source = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kw", "msg"),
    [
        pytest.param({"cols": ()}, "cols must be non-empty", id="empty_cols"),
        pytest.param({"keys": ""}, "non-empty representation", id="empty_key"),
        pytest.param({"keys": ("X", "")}, "non-empty representation", id="one_empty_key"),
        pytest.param({"weights": {("A",): 1.0}}, "arity", id="weight_arity"),  # arity 1 != 2 cols
        pytest.param({"weights": {("A", "d1"): -1.0}}, "non-negative", id="negative_weight"),
    ],
)
def test_node_rejects(kw: dict, msg: str):
    with pytest.raises(ValueError, match=msg):
        _node(**kw)


# ── Bind ──────────────────────────────────────────────────────────────────────────────────────────


def test_bind_common_defaults_to_empty():
    assert Bind("root", "child").common == ()


# ── Scheme ────────────────────────────────────────────────────────────────────────────────────────


def _scheme(nodes: dict, root: str, binds: tuple = ()) -> Scheme:
    return Scheme(sources={"data": SRC}, nodes=nodes, root=root, seed=0, binds=binds)


def test_scheme_single_node_ok():
    _scheme({"root": _node()}, "root")


def test_scheme_depth1_star_ok():
    _scheme(
        {"root": _node(), "c1": _node(cols=("cell_line",)), "c2": _node(cols=("drug",))},
        "root",
        binds=(Bind("root", "c1", ("cell_line",)), Bind("root", "c2", ("drug",))),
    )


@pytest.mark.parametrize(
    ("nodes", "root", "binds", "msg"),
    [
        pytest.param({"a": _node()}, "root", (), "not in nodes", id="root_absent"),
        pytest.param({"root": _node(source="ghost")}, "root", (), "unknown source", id="unknown_source"),
        pytest.param({"root": _node()}, "root", (Bind("root", "ghost"),), "unknown node", id="bind_unknown_node"),
        pytest.param(  # a deeper tree is rejected: every bind's parent must be the root (depth-1 star)
            {"root": _node(), "c": _node(), "gc": _node()},
            "root",
            (Bind("root", "c"), Bind("c", "gc")),
            "must be the root",
            id="parent_not_root",
        ),
        pytest.param(  # the same child bound twice
            {"root": _node(), "c": _node()},
            "root",
            (Bind("root", "c"), Bind("root", "c")),
            "bound more than once",
            id="child_bound_twice",
        ),
        pytest.param(  # a bind whose child is the root gives the root a parent
            {"root": _node(), "c": _node()},
            "root",
            (Bind("root", "root"),),
            "root must have no parent",
            id="root_has_parent",
        ),
        pytest.param({"root": _node(), "orphan": _node()}, "root", (), "not bound to the root", id="orphan_node"),
        pytest.param(
            {"root": _node(), "c": _node(cols=("drug",))},
            "root",
            (Bind("root", "c", ("cell_line",)),),
            "shared cols",
            id="common_not_shared",
        ),
    ],
)
def test_scheme_rejects(nodes: dict, root: str, binds: tuple, msg: str):
    with pytest.raises(ValueError, match=msg):
        _scheme(nodes, root, binds)


# ── SamplerConfig map ───────────────────────────────────────────────────────────────────────────────


def test_config_single_applies_to_every_node():
    assert _resolve_config_map(CFG, ["a", "b"], kind="node") == {"a": CFG, "b": CFG}


@pytest.mark.parametrize(
    ("config", "msg"),
    [
        pytest.param({"a": CFG}, "missing config", id="missing_node"),  # b absent
        pytest.param({"a": CFG, "b": CFG, "c": CFG}, "unknown node", id="extra_node"),
        pytest.param({"a": CFG, "b": 5}, "must be SamplerConfig", id="bad_value"),
        pytest.param(5, "must be a SamplerConfig", id="wrong_type"),
    ],
)
def test_config_map_rejects(config, msg: str):
    with pytest.raises(ValueError, match=msg):
        _resolve_config_map(config, ["a", "b"], kind="node")
