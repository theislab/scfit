"""Weights are really sampled — empirical leaf frequency tracks the configured weights.

Mirrors annbatch's ``test_class_draw_shares`` at the :class:`~scfit.data.Loader` level (decode the batch,
count leaves over many draws), covering uniform / explicit ratios / zero-or-absent exclusion. Plus the
scfit-only property: a bound child's leaf distribution is weight-controlled *within* the matched context
(``P(child extra cols | common)``), which annbatch's ``ClassSampler`` tests have no counterpart for.

Fixed ``seed`` → the stream is reproducible, so ``atol`` is a safe band, not a flaky guess.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice

import pytest

pytest.importorskip("annbatch")

from scheme_helpers import assert_shares, encoded_adata, leaf_shares, only_leaf, rep

from scfit.data import Bind, Loader, Node, SamplerConfig, Scheme
from scfit.data._schema import uniform

COLS = ("cell_line", "drug")
CFG = SamplerConfig(batch_size=8, chunk_size=1, preload_nchunks=8)


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        pytest.param(
            {("A", "d1"): 1, ("A", "d2"): 1, ("B", "d1"): 1, ("B", "d2"): 1},
            {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25},
            id="uniform",
        ),
        pytest.param(
            {("A", "d1"): 6, ("A", "d2"): 3, ("B", "d1"): 1},  # (B, d2) absent → excluded
            {(0, 0): 0.6, (0, 1): 0.3, (1, 0): 0.1},
            id="explicit_absent_excluded",
        ),
        pytest.param(
            {("A", "d1"): 1, ("A", "d2"): 0, ("B", "d1"): 1, ("B", "d2"): 0},  # zero weight excludes
            {(0, 0): 0.5, (1, 0): 0.5},
            id="zero_excludes",
        ),
    ],
)
def test_root_draw_shares(weights: dict, expected: dict):
    adata = encoded_adata(("A", "B"), ("d1", "d2"), n_per_combo=64)
    scheme = Scheme(sources={"data": adata}, nodes={"root": Node("data", COLS, "X", weights)}, root="root", seed=0)
    assert_shares(leaf_shares(Loader(scheme, CFG), "root", 5000), expected)


def test_source_leaf_follows_child_weights_within_context():
    # child binds on cell_line but partitions on (cell_line, drug): within line A the source drug must be
    # drawn 3:1 — i.e. P(drug | cell_line) is weight-controlled (the BoundClassSampler select + project).
    adata = encoded_adata(("A", "B"), ("d1", "d2"), n_per_combo=64)
    scheme = Scheme(
        sources={"data": adata},
        nodes={
            "root": Node("data", COLS, "X", uniform([("A", "d1"), ("A", "d2"), ("B", "d1"), ("B", "d2")])),
            "src": Node("data", COLS, "X", {("A", "d1"): 3, ("A", "d2"): 1, ("B", "d1"): 1, ("B", "d2"): 1}),
        },
        root="root",
        seed=0,
        binds=(Bind("root", "src", common=("cell_line",)),),
    )
    loader = Loader(scheme, CFG)
    drug_when_a = Counter(
        only_leaf(rep(b, "src"))[1]  # the source drug code...
        for b in islice(loader, 6000)
        if only_leaf(rep(b, "root"))[0] == 0  # ...on batches whose target is line A (code 0)
    )
    total = sum(drug_when_a.values())
    assert total > 0, "no line-A batches drawn"
    assert abs(drug_when_a[0] / total - 0.75) < 0.03  # d1 ≈ 75% within line A
