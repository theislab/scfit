# Contributing

```bash
uv run --group test pytest                     # tests
uvx prek run --all-files                       # lint and format
uv run --group doc sphinx-build -M html docs docs/_build -W  # docs
```

Add a line to `CHANGELOG.md` under `[Unreleased]` for user-facing changes.
To release, date the `[Unreleased]` section and publish a GitHub release tagged `vX.Y.Z`; the release workflow uploads it to PyPI.
