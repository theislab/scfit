r"""Declarative, index-free class-mapping data loading over annbatch.

Everything is a :class:`Stream` — one streamed population over a source (its grouping columns, reps,
per-group weights, an optional per-group array lookup, and, for a matched *link*, the columns it shares
with the primary). :class:`Loader` takes one primary Stream plus any number of named linked Streams and
yields matched batches keyed by stream name (``{stream name: {rep loc: rows}}``, plus ``"groups"``): the
primary is the target, each link a source drawn from the same ``match_on`` context. No row indices are
exposed.

See ``README.md`` (next to this file) for the model and the cellflow / sc-flow-tools mapping.
"""

from scfit.data._loader import Loader
from scfit.data._schema import Stream

__all__ = ["Loader", "Stream"]
