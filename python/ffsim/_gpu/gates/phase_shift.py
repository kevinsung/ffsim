# (C) Copyright IBM 2025.
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

# Scaling the selected slices with a kernel avoids the temporary that advanced
# indexing would allocate: ``vec[indices] *= phase`` gathers the slices into a
# new array, scales it, and scatters it back, which for an orbital rotation
# costs about half the size of the state vector on every phase shift.
#
# The two kernels differ only in which axis the slices index, so that the beta
# sector can be scaled in the columns of the vector without transposing.
_MODULE = cupy.RawModule(
    code=r"""
#include <cupy/complex.cuh>

extern "C" __global__ void apply_phase_shift_rows(
    complex<double>* vec,
    complex<double> phase,
    const unsigned long long* indices,
    long long n_indices,
    long long dim_a,
    long long dim_b)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim_b) return;
    vec[indices[idx / dim_b] * dim_b + idx % dim_b] *= phase;
}

extern "C" __global__ void apply_phase_shift_cols(
    complex<double>* vec,
    complex<double> phase,
    const unsigned long long* indices,
    long long n_indices,
    long long dim_a,
    long long dim_b)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_indices * dim_a) return;
    vec[(idx / n_indices) * dim_b + indices[idx % n_indices]] *= phase;
}
"""
)

_KERNEL_NAMES = {
    0: "apply_phase_shift_rows",
    1: "apply_phase_shift_cols",
}


def apply_phase_shift_in_place(
    vec: cupy.ndarray, phase: complex, indices: cupy.ndarray, axis: int = 0
) -> None:
    """Apply a phase shift to slices of a state vector.

    Args:
        vec: The state vector, as a two-dimensional array.
        phase: The phase to apply.
        indices: The indices of the slices to apply the phase to.
        axis: The axis of ``vec`` indexed by ``indices``.
    """
    validate_vec(vec)
    n_indices = len(indices)
    dim_a, dim_b = vec.shape
    launch(
        _MODULE,
        _KERNEL_NAMES[axis],
        n_indices * (dim_b if axis == 0 else dim_a),
        (
            vec,
            np.complex128(phase),
            cupy.ascontiguousarray(indices),
            np.int64(n_indices),
            np.int64(dim_a),
            np.int64(dim_b),
        ),
    )
