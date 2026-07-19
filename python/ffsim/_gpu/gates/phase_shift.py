# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import cupy  # type: ignore
import numpy as np

from ffsim._gpu import validate_vec
from ffsim._gpu._kernels import launch

# The kernel scales the submatrix selected by the row and column indices in
# place, one thread per selected entry. Advanced indexing instead, as in
# ``vec[cupy.ix_(rows, cols)] *= phase``, gathers the entries into a new array,
# scales it, and scatters it back, which allocates about 1.5 times the size of
# the selection and is several times slower.
#
# The kernel takes the strides of the matrix, so it works on any layout and a
# strided view is modified in place, as with the CPU kernel. The column index
# varies fastest across threads, which keeps accesses to a row coalesced when
# the matrix is C-contiguous.
_MODULE = cupy.RawModule(
    code=r"""
#include <cupy/complex.cuh>

extern "C" __global__ void apply_phase_shift(
    complex<double>* vec,
    complex<double> phase,
    const unsigned long long* row_indices,
    const unsigned long long* col_indices,
    bool has_row_indices,
    bool has_col_indices,
    long long n_rows,
    long long n_cols,
    long long stride_row,
    long long stride_col)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rows * n_cols) return;
    long long r = idx / n_cols;
    long long c = idx % n_cols;
    long long i = has_row_indices ? (long long)row_indices[r] : r;
    long long j = has_col_indices ? (long long)col_indices[c] : c;
    vec[i * stride_row + j * stride_col] *= phase;
}
"""
)


def _as_indices(indices: cupy.ndarray | np.ndarray) -> cupy.ndarray:
    return cupy.ascontiguousarray(cupy.asarray(indices), dtype=np.uint64)


def apply_phase_shift_in_place(
    vec: cupy.ndarray,
    phase: complex,
    row_indices: cupy.ndarray | None = None,
    col_indices: cupy.ndarray | None = None,
) -> None:
    """Apply a phase shift to a submatrix of a matrix in-place.

    Multiplies ``vec[i, j]`` by ``phase`` for every ``i`` in ``row_indices`` and
    every ``j`` in ``col_indices``. Passing ``None`` for either selects all rows or
    all columns. The indices must not contain duplicates.

    Args:
        vec: The matrix, which may have any memory layout.
        phase: The phase to apply.
        row_indices: The indices of the rows to apply the phase to.
        col_indices: The indices of the columns to apply the phase to.
    """
    validate_vec(vec, require_contiguous=False)
    dim_a, dim_b = vec.shape
    # The kernel never reads an index array whose flag is unset, so an empty
    # array stands in for it.
    rows = (
        cupy.empty(0, dtype=np.uint64)
        if row_indices is None
        else _as_indices(row_indices)
    )
    cols = (
        cupy.empty(0, dtype=np.uint64)
        if col_indices is None
        else _as_indices(col_indices)
    )
    n_rows = dim_a if row_indices is None else len(rows)
    n_cols = dim_b if col_indices is None else len(cols)
    stride_row, stride_col = (stride // vec.itemsize for stride in vec.strides)
    launch(
        _MODULE,
        "apply_phase_shift",
        n_rows * n_cols,
        (
            vec,
            np.complex128(phase),
            rows,
            cols,
            np.bool_(row_indices is not None),
            np.bool_(col_indices is not None),
            np.int64(n_rows),
            np.int64(n_cols),
            np.int64(stride_row),
            np.int64(stride_col),
        ),
    )
