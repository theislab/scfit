r"""Declarative, index-free class-mapping data loading over annbatch.

Everything is a :class:`Stream` — one streamed population over a source. :class:`Loader` takes the
``sources`` mapping, one primary Stream, and any number of named linked Streams, and yields matched batches
keyed by stream name (``{stream name: {rep loc: rows}}``, plus ``"labels"``): the primary is the target,
each link a source drawn from the same ``match_on`` context. No row indices are exposed.

Pass in-memory ``AnnData`` (or a list) directly under each ``source_key``, or use :meth:`Loader.from_paths`
to open zarr path(s) backed. Each key resolves to a ``Source`` owning that dataset's obs factorization,
shared by every stream naming the key.
"""

from scfit.data._eval import EvalLoader
from scfit.data._loader import Loader
from scfit.data._schema import Stream

__all__ = ["EvalLoader", "Loader", "Stream"]
