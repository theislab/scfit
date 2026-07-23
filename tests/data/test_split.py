"""Tests for :func:`binded.split_scheme` / :func:`binded.split_assignment`.

The split is a weights-only transform on a :class:`~binded.Scheme`, so these exercise it on a scheme
built from a tiny in-memory ``AnnData`` via the ``perturbation_scheme`` test helper — no
``DatasetCollection`` and no ``Loader`` are constructed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("annbatch")  # binded imports annbatch; skip the module if it is not installed

from scheme_helpers import feature_adata, perturbation_scheme

from scfit.data import (
    SamplerConfig,
    Scheme,
    resolve_split_configs,
    split_assignment,
    split_scheme,
)

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}  # split_scheme has no default ratios; pass explicitly


def _toy_scheme(n_drugs: int = 5, cell_lines: tuple[str, ...] = ("A", "B"), seed: int = 0) -> Scheme:
    """A perturbation scheme with ``n_drugs`` perturbations across ``cell_lines`` (+ a control each)."""
    drugs = ["control", *[f"d{i}" for i in range(1, n_drugs + 1)]]
    adata = feature_adata(cell_lines, drugs, 2, n_genes=3)  # 2 cells per combo; X unused by the split
    return perturbation_scheme(adata, seed=seed)


def _combos(scheme: Scheme, node: str) -> set[tuple]:
    n = scheme.nodes[node]
    return {combo for combo, w in n.weights.items() if w > 0}


def _drugs(scheme: Scheme) -> set[str]:
    root = scheme.nodes[scheme.root]
    i = root.cols.index("drug")
    return {combo[i] for combo, w in root.weights.items() if w > 0}


class TestSplitScheme:
    def test_returns_named_schemes(self):
        splits = split_scheme(_toy_scheme(), split_by=["drug"], ratios=RATIOS)
        assert set(splits) == {"train", "val", "test"}
        for sch in splits.values():
            assert isinstance(sch, Scheme)
            assert sch.root == "pert"  # root node name unchanged, only its weights are restricted

    def test_partition_covers_and_is_disjoint(self):
        scheme = _toy_scheme(n_drugs=5)
        splits = split_scheme(scheme, split_by=["drug"], ratios=RATIOS, random_state=0)
        drug_sets = [_drugs(s) for s in splits.values()]
        # disjoint
        for a in range(len(drug_sets)):
            for b in range(a + 1, len(drug_sets)):
                assert drug_sets[a].isdisjoint(drug_sets[b])
        # cover exactly the original perturbation drugs
        assert set().union(*drug_sets) == _drugs(scheme)
        # and the combinations partition the original root combinations
        combo_union = set().union(*(_combos(s, "pert") for s in splits.values()))
        assert combo_union == _combos(scheme, "pert")

    def test_ratio_sizes(self):
        # 5 drugs, 60/20/20 -> 3/1/1 drug-groups; each drug spans 2 cell lines -> 6/2/2 combos
        splits = split_scheme(_toy_scheme(n_drugs=5), split_by=["drug"], ratios=RATIOS, random_state=1)
        assert {k: len(_drugs(v)) for k, v in splits.items()} == {"train": 3, "val": 1, "test": 1}
        assert {k: len(_combos(v, "pert")) for k, v in splits.items()} == {"train": 6, "val": 2, "test": 2}

    def test_groups_by_projection_when_split_by_is_subset(self):
        # split_by=["drug"] but cols=(cell_line, drug): both cell lines of a drug share a split
        splits = split_scheme(_toy_scheme(cell_lines=("A", "B")), split_by=["drug"], ratios=RATIOS, random_state=2)
        for sch in splits.values():
            root = sch.nodes[sch.root]
            by_drug: dict[str, set[str]] = {}
            di, ci = root.cols.index("drug"), root.cols.index("cell_line")
            for combo, w in root.weights.items():
                if w > 0:
                    by_drug.setdefault(combo[di], set()).add(combo[ci])
            for drug, lines in by_drug.items():
                assert lines == {"A", "B"}, f"{drug} split across cell lines: {lines}"

    def test_controls_carried_through_unchanged(self):
        scheme = _toy_scheme()
        splits = split_scheme(scheme, split_by=["drug"], ratios=RATIOS)
        for sch in splits.values():
            assert sch.nodes["ctrl"].weights == scheme.nodes["ctrl"].weights
            assert sch.sources is scheme.sources  # same source objects, no copy

    def test_reproducible_for_same_seed(self):
        a = split_scheme(_toy_scheme(), split_by=["drug"], ratios=RATIOS, random_state=7)
        b = split_scheme(_toy_scheme(), split_by=["drug"], ratios=RATIOS, random_state=7)
        assert {k: _drugs(v) for k, v in a.items()} == {k: _drugs(v) for k, v in b.items()}

    def test_force_training_values(self):
        # d1 must land in train regardless of the shuffle seed
        for seed in range(5):
            splits = split_scheme(
                _toy_scheme(), split_by=["drug"], ratios=RATIOS, force_training_values={"drug": "d1"}, random_state=seed
            )
            assert "d1" in _drugs(splits["train"])
            assert "d1" not in _drugs(splits["val"])
            assert "d1" not in _drugs(splits["test"])

    def test_split_by_context_column_with_custom_ratios(self):
        # "or whatever": hold out a whole cell line, two-way split
        scheme = _toy_scheme(cell_lines=("A", "B"))
        splits = split_scheme(scheme, split_by=["cell_line"], ratios={"train": 0.5, "test": 0.5})
        assert set(splits) == {"train", "test"}
        lines = [{c[scheme.nodes["pert"].cols.index("cell_line")] for c in _combos(s, "pert")} for s in splits.values()]
        assert lines[0].isdisjoint(lines[1])
        assert set().union(*lines) == {"A", "B"}
        for sch in splits.values():  # controls for BOTH lines stay available in each split
            assert sch.nodes["ctrl"].weights == scheme.nodes["ctrl"].weights


class TestSplitAssignment:
    def test_assignment_table(self):
        scheme = _toy_scheme(n_drugs=5)
        splits = split_scheme(scheme, split_by=["drug"], ratios=RATIOS, random_state=0)
        df = split_assignment(splits)
        assert list(df.columns) == ["cell_line", "drug", "split"]
        assert len(df) == len(_combos(scheme, "pert"))  # one row per target combination
        assert set(df["split"]) == {"train", "val", "test"}
        # every (cell_line, drug) appears exactly once
        assert not df.duplicated(subset=["cell_line", "drug"]).any()


class TestSplitValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"split_by": [], "ratios": RATIOS}, "split_by must be non-empty"),
            ({"split_by": ["nope"], "ratios": RATIOS}, "not in the root node's cols"),
            ({"split_by": ["drug"], "ratios": {"train": 0.5, "val": 0.2, "test": 0.2}}, "sum to 1.0"),
            ({"split_by": ["drug"], "ratios": {"train": 1.2, "val": -0.2}}, "must all be > 0"),
            (
                {"split_by": ["drug"], "ratios": RATIOS, "force_training_values": {"cell_line": "A"}},
                "subset of split_by",
            ),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            split_scheme(_toy_scheme(), **kwargs)

    def test_too_few_groups_for_ratios_raises(self):
        # 2 cell lines cannot fill 3 non-empty splits
        with pytest.raises(ValueError, match="received 0 of"):
            split_scheme(_toy_scheme(cell_lines=("A", "B")), split_by=["cell_line"], ratios=RATIOS)


class TestResolveSplitConfigs:
    NAMES = ["train", "val", "test"]

    def test_single_config_object_applies_to_all(self):
        cfg = SamplerConfig(batch_size=256, chunk_size=1, preload_nchunks=256)
        out = resolve_split_configs(cfg, self.NAMES)
        assert set(out) == set(self.NAMES)
        assert all(out[n] is cfg for n in self.NAMES)  # same object shared

    def test_per_split_mapping_ok(self):
        train, val, test = (
            SamplerConfig(batch_size=256, chunk_size=1, preload_nchunks=256),
            SamplerConfig(batch_size=64, chunk_size=1, preload_nchunks=64),
            SamplerConfig(batch_size=64, chunk_size=1, preload_nchunks=64),
        )
        out = resolve_split_configs({"train": train, "val": val, "test": test}, self.NAMES)
        assert out == {"train": train, "val": val, "test": test}

    def test_per_split_missing_split_raises(self):
        with pytest.raises(ValueError, match="missing config"):
            resolve_split_configs({"train": SamplerConfig(batch_size=1, chunk_size=1, preload_nchunks=1)}, self.NAMES)

    def test_per_split_unknown_split_raises(self):
        cfg = SamplerConfig(batch_size=1, chunk_size=1, preload_nchunks=1)
        with pytest.raises(ValueError, match="unknown split"):
            resolve_split_configs({"train": cfg, "val": cfg, "test": cfg, "bogus": cfg}, self.NAMES)

    def test_per_split_non_sampler_config_value_raises(self):
        cfg = SamplerConfig(batch_size=1, chunk_size=1, preload_nchunks=1)
        with pytest.raises(ValueError, match="must be SamplerConfig"):
            resolve_split_configs({"train": cfg, "val": cfg, "test": {"batch_size": 1}}, self.NAMES)
