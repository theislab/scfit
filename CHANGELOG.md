# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `registry` on pydantic: a `Component` is a frozen pydantic model, and a spec is `{"type": type_id, "version": n, **fields}` JSON. `@component(type_id, version=, versions=)` registers it; a field typed as a family base parses any registered member, so families stay open to other packages, and each member is written with its own fields
- a config's own `build` return annotation links it to its implementation (`registry.config_of`)
- `data.SamplerParams`: a small model of the sampler settings, passed as `sampler=` (or a plain mapping) to `Loader` and `Loader.from_paths`, with per-stream overrides in `stream_samplers=`

### Removed

- cattrs: validation, defaults, frozen models and JSON come from pydantic
- The `{type, version, config}` envelope: `type` and `version` sit next to the fields
- `registry.register_live`: anything a spec cannot hold (a live object, or a value that does not read back) raises `PortabilityError` when the spec is written
- `registry.build`, `Component.build` and `Component.build_spec`: each family declares its own typed `build`
- `data.SamplerKwargs`, the `batch_size` / `chunk_size` / `preload_nchunks` keywords, and sampler settings on `Stream`: pass `sampler=` / `stream_samplers=` to the loader

### Fixed

- `registry`: components inside `dict`, `list` and `tuple` fields keep their type through `to_spec` / `parse` ([#14](https://github.com/theislab/scfit/issues/14))
