r"""Declarative, index-free class-mapping sampler over annbatch.

A :class:`Scheme` is a root :class:`Node` (the target) plus its direct children over named cell
sources — a depth-1 star; each node partitions its source's cells into leaves (unique
column-combinations) with a per-combination weight mapping. :class:`Loader` streams matched batches
keyed by node name (``{node name: {rep loc: rows}}``, plus ``"condition"``) — one condition per
batch — where the root's per-batch category is drawn from its weights (annbatch ``ClassSampler``) and
each bound child replays that schedule via an annbatch ``BoundClassSampler`` (matched on the bind's
shared columns) so the loaders zip batch-for-batch. No row indices are exposed: the scheme is
columns / keys / weights.

See ``README.md`` (next to this file) for the model, the sampling schemes, and the mapping to
cellflow and sc-flow-tools use-cases.
"""

from scfit.data._condition import ConditionLookup
from scfit.data._io import leaf_codes
from scfit.data._loader import Loader
from scfit.data._schema import (
    Bind,
    Container,
    Node,
    SamplerConfig,
    Scheme,
)
from scfit.data._split import resolve_split_configs, split_assignment, split_scheme

__all__ = [
    "Bind",
    "ConditionLookup",
    "Container",
    "Loader",
    "Node",
    "SamplerConfig",
    "Scheme",
    "leaf_codes",
    "resolve_split_configs",
    "split_assignment",
    "split_scheme",
]
