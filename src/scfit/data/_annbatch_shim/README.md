# `_annbatch_shim` — a contained, temporary hack

**This is not nice. Ship it anyway.** It's small, isolated in this one package, tested, and it unblocks
publishing `scfit` to PyPI *now* instead of blocking the release on an upstream `annbatch` change. Delete it
when that upstream change lands (see [Removing this](#removing-this)).

## Why it exists

scfit's streaming loader needs three things that only live on the theislab annbatch fork
(`selmanozleyen/annbatch`), not in any upstream annbatch release:

1. `BoundClassSampler` (class-coherent link sampler, built on a `_RunClassSampler` base extracted from
   `ClassSampler`),
2. the per-batch class surfaced as `batch["label"]`, and
3. the #256 multi-dataset purity fix (correct request→buffer scatter).

Depending on the fork by git URL (`annbatch @ git+https://…`) makes **scfit itself un-publishable to
PyPI** — PyPI rejects direct URL references anywhere in package metadata. So we pin a *released*
`annbatch==0.2.1` and re-add the three bits here.

We can do this safely because `git diff v0.2.1 <fork-base>` is **empty** for `loader.py`, `abc/`,
`samplers/`, and `types.py` — stock 0.2.1 is byte-identical to what the fork's code was written against.

## What's inside

| file | origin |
|------|--------|
| `_base_class_sampler.py`, `_class_sampler.py`, `_bound_class_sampler.py`, `_utils.py` | **vendored verbatim** from the fork (ruff-excluded so they stay diffable for syncs) |
| `_loader_patch.py` | re-execs the fork's `Loader.__iter__` in `annbatch.loader`'s namespace and rebinds it (the `label` + #256 changes); version-guarded to `0.2.1` |
| `__init__.py` | re-exports the samplers and applies the loader patch on import |

## Containment (the only two couplings)

1. `scfit/data/_loader.py` imports `ClassSampler` / `BoundClassSampler` from here (not `annbatch.samplers`).
2. Importing this package rebinds `annbatch.loader.Loader.__iter__` **once**, as a side effect — the only
   global mutation. It's guarded to `annbatch==0.2.1` and raises loudly on any drift.

Nothing else in scfit touches annbatch internals.

## Invariant

`annbatch` **must** stay pinned `==0.2.1` in `pyproject.toml`. The loader patch reproduces that exact
release's `__iter__`; a bump will trip the version guard (loud failure), which is your signal to re-vendor
and re-verify — never a silent wrong result.

## Removing this

When the fork's features are upstreamed into an annbatch release:

1. Bump the dep to the released `annbatch>=X` (that ships `BoundClassSampler`, `batch["label"]`, #256).
2. In `scfit/data/_loader.py`, import the samplers from `annbatch.samplers` again.
3. Delete this package and its `pyproject.toml` ruff `extend-exclude` entry.
4. Run `pytest tests/data` — parity is already covered there.
