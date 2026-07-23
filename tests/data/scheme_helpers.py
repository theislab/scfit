"""Shared perturbation-test fixtures: the cellflow-shaped Scheme + the tiny AnnData builders it reads.

All test-only (moved out of ``binded``'s public API; production cellflow assembles the scheme internally
via :func:`cellflow.data._annbatch.build_annbatch_training`). A perturbation dataset is the product of
cell lines × drugs, with ``control`` marking the control drug:

* :func:`encoded_adata` — X *encodes each cell's identity* (``[cell_line, drug, is_control, row_id]``, see
  the :data:`LINE`/:data:`DRUG`/:data:`IS_CONTROL`/:data:`ROW_ID` column indices), so a yielded batch row
  can be decoded and checked against the leaf it must belong to.
* :func:`feature_adata` — a realistic random matrix, for metric / pickle tests that don't decode identity.
* :func:`perturbation_scheme` — the two-node ``perturbed root ← bound control`` Scheme over any such source
  (matched on ``context``); options cover an in-memory control and per-node reps.

Importable bare (``from scheme_helpers import ...``) via the ``pythonpath = ["tests/data"]`` pytest setting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import islice

import anndata as ad
import numpy as np
import pandas as pd

from scfit.data import Bind, Node, Scheme
from scfit.data._io import obs_columns
from scfit.data._schema import Container

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2", "d3")
CONTROL = "control"

#: Column indices of the identity encoding produced by :func:`encoded_adata`.
LINE, DRUG, IS_CONTROL, ROW_ID = 0, 1, 2, 3


def uniform(combos: Sequence[tuple]) -> dict[tuple, float]:
    """Every combination equally likely."""
    return {tuple(c): 1.0 for c in combos}


def frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ its cell count (favor abundant conditions)."""
    return {tuple(k): float(c) for k, c in counts.items()}


def inverse_frequency(counts: Mapping[tuple, int]) -> dict[tuple, float]:
    """Sample each combination ∝ 1 / cell count (balance rare vs abundant conditions)."""
    return {tuple(k): 1.0 / c for k, c in counts.items()}


def codes(values: Sequence[str]) -> dict[str, int]:
    """Stable value → integer code map (enumeration order)."""
    return {v: i for i, v in enumerate(values)}


def perturbation_obs(
    lines: Sequence[str] = LINES,
    drugs: Sequence[str] = DRUGS,
    n_per_combo: int = 8,
    *,
    index_prefix: str = "",
) -> pd.DataFrame:
    """obs for the product of ``lines`` × ``drugs`` (``n_per_combo`` cells each), with a ``control`` flag."""
    rows = [(cl, dr) for cl in lines for dr in drugs for _ in range(n_per_combo)]
    obs = pd.DataFrame(rows, columns=["cell_line", "drug"])
    obs["control"] = obs["drug"] == CONTROL
    obs.index = [f"{index_prefix}{i}" for i in range(len(obs))]
    return obs


def encoded_adata(
    lines: Sequence[str] = LINES,
    drugs: Sequence[str] = DRUGS,
    n_per_combo: int = 8,
    *,
    line: Mapping[str, int] | None = None,
    drug: Mapping[str, int] | None = None,
    obsm_rep: bool = False,
    row_id_start: int = 0,
    index_prefix: str = "",
) -> ad.AnnData:
    """AnnData whose X encodes each cell's identity so a yielded batch row can be decoded and checked.

    X columns are ``[cell_line code, drug code, is_control, global row id]`` (see :data:`LINE` etc.).
    ``line``/``drug`` override the code maps (default: enumeration of ``lines``/``drugs``) — pass explicit
    maps when several stores must share one global vocabulary. ``row_id_start`` offsets the row ids so
    ids stay unique across stores; ``obsm_rep`` adds an ``obsm['rep']`` copy of X (an aligned rep).
    """
    obs = perturbation_obs(lines, drugs, n_per_combo, index_prefix=index_prefix)
    line = codes(lines) if line is None else line
    drug = codes(drugs) if drug is None else drug
    x = np.stack(
        [
            obs["cell_line"].map(line).to_numpy(),
            obs["drug"].map(drug).to_numpy(),
            obs["control"].to_numpy().astype(float),
            row_id_start + np.arange(len(obs)),
        ],
        axis=1,
    ).astype("float32")
    adata = ad.AnnData(x, obs=obs)
    if obsm_rep:
        adata.obsm["rep"] = adata.X.copy()
    return adata


def feature_adata(
    lines: Sequence[str] = LINES,
    drugs: Sequence[str] = DRUGS,
    n_per_combo: int = 8,
    *,
    n_genes: int = 12,
    seed: int = 0,
    obsm_rep: bool = False,
) -> ad.AnnData:
    """AnnData with a realistic random feature matrix (metric / pickle tests — no identity to decode)."""
    obs = perturbation_obs(lines, drugs, n_per_combo)
    x = np.random.default_rng(seed).standard_normal((len(obs), n_genes)).astype("float32")
    adata = ad.AnnData(x, obs=obs)
    if obsm_rep:
        adata.obsm["rep"] = adata.X.copy()
    return adata


def write_zarr(adata: ad.AnnData, path) -> str:
    """Write ``adata`` to a zarr store (silencing the zarr v2/v3 default-format warning); returns the path."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata.write_zarr(str(path))
    return str(path)


def perturbation_scheme(
    source: Container,
    *,
    context: Sequence[str] = ("cell_line",),
    perturbation: Sequence[str] = ("drug",),
    control_values: Mapping[str, object] | None = None,
    key: str | Sequence[str] = "X",
    ctrl_key: str | Sequence[str] | None = None,
    ctrl_in_memory: bool = False,
    seed: int = 0,
) -> Scheme:
    """The two-node perturbation Scheme: root = perturbed combos, child = control combos, bound on context.

    Control vs perturbed is encoded purely by which combinations carry weight (no ``select`` step); the
    control node is bound to the perturbed root on ``context``, so each batch's control cells come from the
    same context (cell line, …) as the perturbed cells — the source↔target matching. Parameters mirror
    cellflow: ``context`` = ``split_covariates``, ``perturbation`` = the perturbation columns,
    ``control_values`` = which value marks control per column (default ``{"drug": "control"}``), ``key`` =
    ``sample_rep``. ``ctrl_key`` overrides the control node's reps (default: same as ``key``);
    ``ctrl_in_memory`` materializes the control cells into RAM (see :func:`~binded._io.materialize_node`).
    """
    context, perturbation = tuple(context), tuple(perturbation)
    control_values = {"drug": CONTROL} if control_values is None else dict(control_values)
    ctrl_key = key if ctrl_key is None else ctrl_key
    cols = (*context, *perturbation)
    combos = [tuple(r) for r in obs_columns(source, cols).drop_duplicates().to_numpy()]

    def is_control(combo: tuple) -> bool:
        return all(combo[cols.index(c)] == v for c, v in control_values.items())

    pert = [c for c in combos if not is_control(c)]
    ctrl = [c for c in combos if is_control(c)]
    return Scheme(
        sources={"data": source},
        nodes={
            "pert": Node("data", cols, key, uniform(pert)),
            "ctrl": Node("data", cols, ctrl_key, uniform(ctrl), in_memory=ctrl_in_memory),
        },
        root="pert",
        binds=(Bind("pert", "ctrl", common=context),),
        seed=seed,
    )


# ── batch-reading helpers (the read side; they hide the {node: {rep loc: rows}} batch schema) ──────────


def rep(batch, node: str, key: str = "X") -> np.ndarray:
    """Rows of one rep of a node's batch — ``batch[node][key]`` as a dense ndarray."""
    return np.asarray(batch[node][key])


def only_leaf(rows: np.ndarray) -> tuple[int, int]:
    """Assert ``rows`` is a single leaf and return its decoded ``(cell_line, drug)`` codes.

    For an :func:`encoded_adata` batch: X column :data:`LINE`/:data:`DRUG` hold the codes, so a pure batch
    has exactly one unique value in each. Raises (via assert) if the batch mixes conditions — which is also
    how the pureness tests check coherence.
    """
    line, drug = np.unique(rows[:, LINE]), np.unique(rows[:, DRUG])
    assert line.size == 1 and drug.size == 1, "batch mixes conditions"
    return int(line[0]), int(drug[0])


def leaf_shares(loader, node: str, n: int) -> dict[tuple[int, int], float]:
    """Empirical fraction of ``n`` batches whose ``node`` batch is each leaf (decoded from X)."""
    seen = Counter(only_leaf(rep(b, node)) for b in islice(loader, n))
    total = sum(seen.values())
    return {leaf: c / total for leaf, c in seen.items()}


def assert_shares(actual: Mapping, expected: Mapping, atol: float = 0.03) -> None:
    """Assert the sampled leaf set equals ``expected`` and each share is within ``atol`` of its target."""
    assert set(actual) == set(expected), f"sampled leaves {sorted(actual)} != {sorted(expected)}"
    for leaf, exp in expected.items():
        assert abs(actual[leaf] - exp) <= atol, f"{leaf}: share {actual[leaf]:.3f} vs expected {exp}"
