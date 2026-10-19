# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `Stream(group_by=())`: every cell is one group, whose leaf is `()`, for unconditional streams (no grouping column needed)

## [0.0.10] - 2026-10-19

### Added

- `data.obs_columns` and `data.open_source` are public: obs columns of one AnnData or several read as one dataset, and zarr stores opened backed. `open_source` (and `load_backed_adata`) take `uns=` to read named `.uns` entries

### Fixed

- Several AnnData in one source with different categories in a grouping column no longer fail to load (the categories are unioned)

### Changed

- annbatch `>=0.3.1` replaces the `==0.2.1` pin and the shim over it (vendored samplers and a patched `Loader.__iter__`): the primary samples with annbatch's `ClassSampler`, links with a new `BoundClassSampler` built on it, and each batch's group is read from the leaf the loaders carry as obs

## [0.0.9] - 2026-10-19

### Added

- `pretrained.save_pretrained` / `load_pretrained`: a spec with its weights as `config.json` plus `<name>/model.safetensors`, in the Hugging Face Hub layout; `load_pretrained` also takes a Hub repo id. torch, safetensors and huggingface-hub are now dependencies

### Removed

- Spec versions: a spec is `{"type": type_id, **fields}`, and `@component` takes no `version`/`versions`. Specs written with a `"version"` key no longer load; a saved spec reads back under the package versions that wrote it

## [0.0.8] - 2026-10-02

### Added

- `registry` on pydantic: a `Component` is a frozen pydantic model, and a spec is `{"type": type_id, "version": n, **fields}` JSON. `@component(type_id, version=, versions=)` registers it; a field typed as a family base parses any registered member, so families stay open to other packages, and each member is written with its own fields
- `@component(type_id, builds=Impl)` links a config to its implementation (`registry.config_of`)
- `data.SamplerParams`: a small model of the sampler settings, passed as `sampler=` (or a plain mapping) to `Loader` and `Loader.from_paths`, with per-stream overrides in `stream_samplers=`

### Removed

- cattrs and attrs
- The `{type, version, config}` envelope
- `registry.register_live`: anything a spec cannot hold (a live object, or a value that does not read back) raises `PortabilityError` when the spec is written
- `registry.build`, `Component.build` and `Component.build_spec`: each family declares its own typed `build`
- `data.SamplerKwargs`, the `batch_size` / `chunk_size` / `preload_nchunks` keywords, and sampler settings on `Stream`: pass `sampler=` / `stream_samplers=` to the loader

### Fixed

- `registry`: components inside `dict`, `list` and `tuple` fields keep their type through `to_spec` / `parse` ([#14](https://github.com/theislab/scfit/issues/14))
