r"""Declarative, index-free class-mapping data loading over annbatch.

Everything is a :class:`Stream` — one streamed population over a :class:`Source` (its ``source_key`` into
the loader's ``sources`` mapping, plus the grouping columns, reps, per-group weights, an optional per-group
array lookup, and, for a matched *link*, the columns it shares with the primary). :class:`Loader` takes the
``sources`` mapping, one primary Stream, and any number of named linked Streams, and yields matched batches
keyed by stream name (``{stream name: {rep loc: rows}}``, plus ``"labels"``): the primary is the target,
each link a source drawn from the same ``match_on`` context. No row indices are exposed.

A :class:`Source` (one dataset — a list of ``AnnData`` — addressed by ``source_key``) owns that dataset's
obs factorization, shared by every stream naming the key. Pass in-memory ``AnnData`` directly, or use
:meth:`Loader.from_paths` to open zarr path(s) backed.

See ``README.md`` (next to this file) for the model and the cellflow / sc-flow-tools mapping.
"""

from scfit.data._loader import Loader
from scfit.data._schema import Stream
from scfit.data._source import Source

__all__ = ["Loader", "Source", "Stream"]
