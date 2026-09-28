# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- `registry.component`: the one way to register a config; makes it a frozen, keyword-only dataclass
- `scfit.params`: parameter bags as TypedDicts with `Annotated[type, Default(value)]` keys, `resolve_params`, `validates`, and `ParamsComponent`, a component whose `params` field subclasses narrow to their TypedDict
- `data.SamplerParams`: the sampler settings as a params TypedDict, passed as `sampler=` to `Loader` and `Loader.from_paths`, with per-stream overrides in `stream_samplers=`

### Removed

- `registry.build`, `Component.build` and `Component.build_spec`: each family declares its own typed `build`
- The `type_id=` / `version=` / `versions=` class keywords: use `@component(type_id, version=..., versions=...)`
- `data.SamplerKwargs`, the `batch_size` / `chunk_size` / `preload_nchunks` keywords, and sampler settings on `Stream`: pass `sampler=` / `stream_samplers=` to the loader

### Fixed

- `registry`: TypedDict fields are structured key by key; components inside `dict`, `list` and `tuple` fields keep their type through `to_spec` / `parse` ([#14](https://github.com/theislab/scfit/issues/14))
