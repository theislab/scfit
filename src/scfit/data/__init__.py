"""Declarative, index-free class-mapping data loading over annbatch.

Everything is a :class:`Stream` — one streamed population over a source. :class:`Loader` yields stochastic
matched batches for training and :class:`EvalLoader` a deterministic full-coverage pass; both key their
output by stream name and expose no row indices.
"""

from scfit.data._eval import EvalLoader
from scfit.data._loader import Loader
from scfit.data._schema import Stream

__all__ = ["EvalLoader", "Loader", "Stream"]
