"""Runtime shim: bring stock ``annbatch==0.2.1``'s ``Loader.__iter__`` up to the theislab fork.

scfit pins **stock** ``annbatch==0.2.1`` (so scfit itself is PyPI-publishable), but its loader needs two
fork-only behaviours that live *inside* the ``Loader.__iter__`` generator and cannot be patched surgically:

* the #256 multi-dataset purity fix (scatter, not gather, the request->buffer index map), and
* surfacing the class sampler's per-batch category as ``batch["label"]``.

So we re-exec the fork's ``__iter__`` in annbatch.loader's own module namespace (giving it the exact same
globals: ``convert``, ``LoaderOutput``, ``zsync`` ...) and rebind it. This is a deliberate, fragile hack:
it is valid ONLY against annbatch 0.2.1 (verified: v0.2.1 == the fork's base for loader.py), so we assert
the pinned version and fail loudly on any drift rather than silently corrupting batches. See the vendored
samplers in this package (they emit the ``combs`` this reads) and pyproject's ``annbatch==0.2.1`` pin.
"""

from __future__ import annotations

from importlib.metadata import version

import annbatch.loader as _al

_PINNED = "0.2.1"

_FORK_ITER = '''
def __iter__(
    self,
) -> Iterator[LoaderOutput[OutputInMemoryArray]]:
    check_lt_1(
        [len(self._train_datasets), self.n_obs],
        ["Number of datasets", "Number of observations"],
    )
    is_sparse = issubclass(self.dataset_type, ad.abc.CSRDataset | sp.csr_matrix | sp.csr_array)
    # Create `positions` variable so we don't need to run `np.arange` (O(n)) every time
    positions = np.empty(0, dtype=np.intp)
    for load_request in self._batch_sampler.sample(self.n_obs):
        requests_to_load = load_request.get("requests", None)
        if requests_to_load is None:
            requests_to_load = load_request.get("chunks", None)
            if requests_to_load is not None:
                # this is for backwards compat.
                warn(
                    "The `chunks` key in the load request is deprecated and will be removed in a future version. Please use `requests` instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise KeyError("load_request must contain either 'requests' or 'chunks'.")
        splits = load_request["splits"]

        dataset_index_to_rows, order = self._requests_to_dataset_rows(requests_to_load)

        # The buffer below is filled in dataset order, but ``splits`` are expressed in the
        # sampler's `LoadRequest.request` order. ``inv`` maps a request-order position to its buffer position so
        # the split semantics are independent of how chunks were regrouped across datasets.
        # ``order`` is a permutation of ``range(n)``, so every used slot is overwritten -- the
        # reused buffer never carries stale values from a previous request.
        # NB: this is the INVERSE permutation (scatter), not a gather. ``order`` maps a buffer
        # position to its request-order position; ``inv`` must invert that. ``inv = positions[order]``
        # is only correct when ``order`` is self-inverse (e.g. a single dataset, identity order), and
        # silently pulls rows from the wrong dataset once chunks regroup across several datasets.
        n = order.size
        inv_buffer = np.empty(n, dtype=np.intp)
        if n > positions.size:
            positions = np.arange(n, dtype=np.intp)
        inv = inv_buffer[:n]
        inv[order] = positions[:n]

        raw_out: CSRContainer | np.ndarray = zsync.sync(self._index_datasets(dataset_index_to_rows))

        if is_sparse:
            in_memory_data = self._sp_module.csr_matrix(
                tuple(self._np_module.asarray(e) for e in raw_out.elems),
                shape=raw_out.shape,
                dtype=_cupy_dtype(raw_out.dtype) if self._preload_to_gpu else raw_out.dtype,
            )
        else:
            in_memory_data = self._np_module.asarray(raw_out)

        concatenated_obs: None | pd.DataFrame = self._maybe_accumulate_obs(dataset_index_to_rows)
        in_memory_indices: None | np.ndarray = self._maybe_accumulate_indices(dataset_index_to_rows)
        combs = load_request.get("combs")  # per-split category label (class samplers only)
        for i, split in enumerate(splits):
            sel = inv[split]
            data = in_memory_data[sel]
            out: LoaderOutput = {
                "X": data if self._to is None else convert(data, self._preload_to_gpu, self._to),
                "obs": concatenated_obs.iloc[sel] if concatenated_obs is not None else None,
                "var": self._var,
                "index": in_memory_indices[sel] if in_memory_indices is not None else None,
            }
            if combs is not None:
                out["label"] = combs[i]
            yield out

        # https://github.com/cupy/cupy/issues/9625
        if self._preload_to_gpu and is_sparse:
            self._np_module.get_default_memory_pool().free_all_blocks()
'''


def _apply() -> None:
    have = version("annbatch")
    if have != _PINNED:
        raise RuntimeError(
            f"scfit's annbatch shim targets annbatch=={_PINNED} but found {have}. The fork's "
            "Loader.__iter__ (label surfacing + #256 fix) is reproduced against 0.2.1 internals; refusing "
            "to rebind against a different version. Pin annbatch=={_PINNED} or update the shim."
        )
    ns: dict = {}
    exec(compile(_FORK_ITER, "<scfit annbatch shim: Loader.__iter__>", "exec"), _al.__dict__, ns)
    _al.Loader.__iter__ = ns["__iter__"]


_apply()
