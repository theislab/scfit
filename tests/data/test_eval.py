import anndata as ad
import numpy as np
import pandas as pd

from scfit.data import EvalLoader, Stream

CELL_LINES = ["A", "B"]
DRUGS = ["control", "d1", "d2"]


def _adata(n_per_combo: int = 5) -> ad.AnnData:
    obs = pd.DataFrame(
        [(cl, dr) for cl in CELL_LINES for dr in DRUGS for _ in range(n_per_combo)],
        columns=["cell_line", "drug"],
    ).astype("category")
    obs.index = obs.index.astype(str)
    x = np.arange(len(obs) * 3, dtype="float32").reshape(len(obs), 3)  # identifiable rows
    return ad.AnnData(X=x, obs=obs)


def test_full_pass_is_deterministic_and_matched():
    """Every perturbed group once; each matched to its own cell line's controls."""
    adata = _adata()
    obs = adata.obs.reset_index(drop=True)
    primary = Stream(
        "d", group_by=["cell_line", "drug"], reps=("X",),
        weights={(cl, dr): 1.0 for cl in CELL_LINES for dr in ("d1", "d2")},
    )
    control = Stream(
        "d", group_by=["cell_line", "drug"], reps=("X",),
        weights={(cl, "control"): 1.0 for cl in CELL_LINES}, match_on=["cell_line"],
    )
    loader = EvalLoader({"d": adata}, primary=primary, links={"control": control})

    assert len(loader) == 4  # 2 cell lines x 2 perturbed drugs
    seen = []
    for batch in loader:
        cl, dr = batch["leaf"]
        seen.append((cl, dr))
        pert = np.flatnonzero((obs.cell_line == cl) & (obs.drug == dr))
        ctrl = np.flatnonzero((obs.cell_line == cl) & (obs.drug == "control"))
        assert np.array_equal(batch["primary"]["X"], adata.X[pert])
        assert np.array_equal(batch["control"]["X"], adata.X[ctrl])  # matched, controls-only
    assert set(seen) == {(cl, dr) for cl in CELL_LINES for dr in ("d1", "d2")}


def test_max_per_group_deduplicates_to_unique_combos():
    """max_per_group=1 yields exactly one representative per group (the unique group_by combos)."""
    adata = _adata()
    stream = Stream("d", group_by=["cell_line", "drug"], reps=("X",))  # no weights -> every group
    loader = EvalLoader({"d": adata}, primary=stream, max_per_group=1)

    batches = list(loader)
    assert len(loader) == len(batches) == len(CELL_LINES) * len(DRUGS)
    assert all(b["primary"]["X"].shape[0] == 1 for b in batches)
    assert {b["leaf"] for b in batches} == {(cl, dr) for cl in CELL_LINES for dr in DRUGS}


def test_random_subsample_is_reproducible_and_capped():
    """subsample="random" caps each group and gives the same draw across passes for a fixed seed."""
    adata = _adata(n_per_combo=8)
    stream = Stream("d", group_by=["cell_line", "drug"], reps=("X",))
    runs = [
        list(EvalLoader({"d": adata}, primary=stream, max_per_group=3, subsample="random", seed=0)) for _ in range(2)
    ]
    assert all(b["primary"]["X"].shape[0] == 3 for b in runs[0])  # capped to N
    for x, y in zip(runs[0], runs[1], strict=True):
        assert np.array_equal(x["primary"]["X"], y["primary"]["X"])  # reproducible for a fixed seed
