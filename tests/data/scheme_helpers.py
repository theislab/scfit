"""Shared perturbation-test fixtures: the ``Stream``/``Loader`` builders + the tiny AnnData makers.

All test-only. A perturbation dataset is the product of cell lines × drugs, with ``control`` marking the
control drug:

* :func:`encoded_adata` — X *encodes each cell's identity* (``[cell_line, drug, is_control, row_id]``, see
  the :data:`LINE`/:data:`DRUG`/:data:`IS_CONTROL`/:data:`ROW_ID` column indices), so a yielded batch row
  can be decoded and checked against the label it must belong to.
* :func:`feature_adata` — a realistic random matrix, for pickle / metric tests that don't decode identity.
* :func:`perturbation_streams` / :func:`perturbation_loader` — the ``primary`` (perturbed) + matched
  ``ctrl`` (control) streams over any such source, the control linked to the primary on ``context``.

Batch schema (new API): ``{stream name: {rep loc: rows}}`` for ``"primary"`` and each link, plus
``"leaves"`` (the ``group_by`` tuple each stream drew this batch).

Importable bare (``from scheme_helpers import ...``) via the ``pythonpath = ["tests/data"]`` pytest setting.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import islice

import anndata as ad
import numpy as np
import pandas as pd

from scfit.data import Loader, Stream
from scfit.data._io import obs_columns

LINES = ("A", "B")
DRUGS = ("control", "d1", "d2", "d3")
CONTROL = "control"

#: The shared ``source_key`` the perturbation streams (primary + matched control) both name — one dataset.
KEY = "pert"

#: Column indices of the identity encoding produced by :func:`encoded_adata`.
LINE, DRUG, IS_CONTROL, ROW_ID = 0, 1, 2, 3


def uniform(combos: Sequence[tuple]) -> dict[tuple, float]:
    """Every combination equally likely (a plain ``{combo: 1.0}`` weight map — test convenience)."""
    return {tuple(c): 1.0 for c in combos}


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
    maps when several stores must share one global vocabulary. ``row_id_start`` offsets the row ids so ids
    stay unique across stores; ``obsm_rep`` adds an ``obsm['rep']`` copy of X (an aligned rep).
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
    """AnnData with a realistic random feature matrix (pickle / metric tests — no identity to decode)."""
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


def perturbation_streams(
    source,
    *,
    context: Sequence[str] = ("cell_line",),
    perturbation: Sequence[str] = ("drug",),
    control_values: Mapping[str, object] | None = None,
    rep: str | Sequence[str] = "X",
    ctrl_rep: str | Sequence[str] | None = None,
    ctrl_in_memory: bool = False,
    ctrl_match_on: Sequence[str] | None = None,
) -> tuple[Stream, dict[str, Stream]]:
    """The two streams — ``primary`` (perturbed) and linked ``ctrl`` (control) — matched on ``context``.

    Control vs perturbed is encoded purely by which combinations carry weight (uniform over each side).
    Returns ``(primary, {"ctrl": ...})`` — splat into :class:`~scfit.data.Loader`. ``ctrl_match_on``
    overrides the control's ``match_on`` (default = ``context``; pass ``()`` for an unconditional control).
    ``ctrl_in_memory`` materializes the control cells.
    """
    context, perturbation = tuple(context), tuple(perturbation)
    control_values = {"drug": CONTROL} if control_values is None else dict(control_values)
    ctrl_rep = rep if ctrl_rep is None else ctrl_rep
    match_on = context if ctrl_match_on is None else tuple(ctrl_match_on)
    cols = (*context, *perturbation)
    combos = [tuple(r) for r in obs_columns(source, cols).drop_duplicates().to_numpy()]

    def is_control(combo: tuple) -> bool:
        return all(combo[cols.index(c)] == v for c, v in control_values.items())

    pert = [c for c in combos if not is_control(c)]
    ctrl = [c for c in combos if is_control(c)]
    primary = Stream(KEY, group_by=cols, reps=rep, weights=uniform(pert))
    control = Stream(
        KEY, group_by=cols, reps=ctrl_rep, weights=uniform(ctrl), match_on=match_on, in_memory=ctrl_in_memory
    )
    return primary, {"ctrl": control}


def perturbation_loader(
    source,
    *,
    seed: int = 0,
    batch_size: int = 8,
    chunk_size: int = 1,
    preload_nchunks: int = 8,
    to: str | None = "torch",
    preload_to_gpu: bool = False,
    **stream_kwargs,
) -> Loader:
    """A :class:`~scfit.data.Loader` over :func:`perturbation_streams` with the given read parameters."""
    primary, links = perturbation_streams(source, **stream_kwargs)
    return Loader(
        {KEY: source},
        primary=primary,
        links=links,
        seed=seed,
        batch_size=batch_size,
        chunk_size=chunk_size,
        preload_nchunks=preload_nchunks,
        to=to,
        preload_to_gpu=preload_to_gpu,
    )


# ── batch-reading helpers (the read side; they hide the {stream: {rep loc: rows}} batch schema) ──────────


def rep(batch, stream: str, key: str = "X") -> np.ndarray:
    """Rows of one rep of a stream's batch — ``batch[stream][key]`` as a dense ndarray."""
    return np.asarray(batch[stream][key])


def only_leaf(rows: np.ndarray) -> tuple[int, int]:
    """Assert ``rows`` is a single label and return its decoded ``(cell_line, drug)`` codes.

    For an :func:`encoded_adata` batch: X column :data:`LINE`/:data:`DRUG` hold the codes, so a pure batch
    has exactly one unique value in each. Raises (via assert) if the batch mixes conditions — which is also
    how the pureness tests check coherence.
    """
    line, drug = np.unique(rows[:, LINE]), np.unique(rows[:, DRUG])
    assert line.size == 1 and drug.size == 1, "batch mixes conditions"
    return int(line[0]), int(drug[0])


def leaf_shares(loader, stream: str, n: int) -> dict[tuple[int, int], float]:
    """Empirical fraction of ``n`` batches whose ``stream`` batch is each label (decoded from X)."""
    seen = Counter(only_leaf(rep(b, stream)) for b in islice(loader, n))
    total = sum(seen.values())
    return {leaf: c / total for leaf, c in seen.items()}


def assert_shares(actual: Mapping, expected: Mapping, atol: float = 0.03) -> None:
    """Assert the sampled label set equals ``expected`` and each share is within ``atol`` of its target."""
    assert set(actual) == set(expected), f"sampled labels {sorted(actual)} != {sorted(expected)}"
    for leaf, exp in expected.items():
        assert abs(actual[leaf] - exp) <= atol, f"{leaf}: share {actual[leaf]:.3f} vs expected {exp}"
