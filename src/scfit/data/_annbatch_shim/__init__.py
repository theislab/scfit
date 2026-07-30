"""A hack, contained. Fork-only annbatch features reproduced over stock ``annbatch==0.2.1``.

Let's be honest: this package is not nice. It vendors part of a fork verbatim and monkeypatches a
third-party class at import time. But there is no reason to wait on it — it is small, fully contained in
this one package, covered by scfit's data test suite, and it unblocks shipping scfit to PyPI *today*
instead of gating the whole release on an upstream annbatch change. Ship it; delete it later.

**Why it exists.** scfit's streaming loader needs three things upstream annbatch does not release — they
live only on the theislab fork (``selmanozleyen/annbatch``): the class-coherent :class:`BoundClassSampler`
(built on a ``_RunClassSampler`` base extracted from :class:`ClassSampler`), the per-batch class surfaced as
``batch["label"]``, and the #256 multi-dataset purity fix. Depending on that fork by git URL would make
scfit *itself* un-publishable to PyPI (direct URL references are rejected in package metadata). So instead
we pin a *released* ``annbatch==0.2.1`` and re-add the missing bits here.

**How it's contained.** Everything hacky is in this package; the rest of scfit stays clean. The only two
couplings, both documented:
  1. :mod:`scfit.data._loader` imports :class:`ClassSampler` / :class:`BoundClassSampler` from here instead
     of ``annbatch.samplers``.
  2. Importing this package rebinds ``annbatch.loader.Loader.__iter__`` as a one-time side effect (see
     :mod:`._loader_patch`) — the only global mutation, guarded to ``annbatch==0.2.1`` so it fails loudly on
     any version drift rather than silently corrupting batches.
The vendored sampler modules (``_base_class_sampler``, ``_class_sampler``, ``_bound_class_sampler``,
``_utils``) are copied *verbatim* from the fork (and excluded from ruff, see pyproject) so they stay
diffable against upstream for future syncs.

**When to delete it.** Upstream the fork's features (BoundClassSampler, the label surfacing, #256), then
depend on a released ``annbatch>=X``, drop this package, and point ``_loader`` back at ``annbatch.samplers``.
See ``README.md`` in this directory for the step-by-step removal.
"""

from __future__ import annotations

from . import _loader_patch  # noqa: F401  # side effect: rebinds annbatch.loader.Loader.__iter__
from ._bound_class_sampler import BoundClassSampler
from ._class_sampler import ClassSampler

__all__ = ["BoundClassSampler", "ClassSampler"]
