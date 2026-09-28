# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- Basic tool, preprocessing and plotting functions
- `registry.RngComponent`, a family marker for configs whose `build` takes a caller-supplied `rng`
- `registry.component`: the one way to register a config; makes it a frozen, keyword-only dataclass

### Removed

- `registry.build`, `Component.build` and `Component.build_spec`: each family declares its own typed `build`
- The `type_id=` / `version=` / `versions=` class keywords: use `@component(type_id, version=...)`

### Fixed

- `registry`: components inside `dict`, `list` and `tuple` fields keep their type through `to_spec` / `parse` ([#14](https://github.com/theislab/scfit/issues/14))
