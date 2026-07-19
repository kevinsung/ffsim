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

# The phase for a determinant is the product of the phases of its occupied
# orbitals. Accumulating that product in a register keeps the kernel free of
# scratch memory. Gathering the occupations into a dim x nocc array and reducing
# it instead would allocate nocc times the size of the state vector whenever the
# occupations index the long axis, as they do for spinless systems.
#
# The two kernels differ only in which axis the occupations index, so that the
# beta sector can be applied to the columns of the vector without transposing.
_MODULE = cupy.RawModule(
    code=r"""
#include <cupy/complex.cuh>

extern "C" __global__ void apply_num_op_sum_evolution_rows(
    complex<double>* vec,
    const complex<double>* phases,
    const unsigned long long* occupations,
    long long dim_a,
    long long dim_b,
    long long nocc)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim_a * dim_b) return;
    const unsigned long long* orbs = occupations + (idx / dim_b) * nocc;
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < nocc; j++) {
        phase *= phases[orbs[j]];
    }
    vec[idx] *= phase;
}

extern "C" __global__ void apply_num_op_sum_evolution_cols(
    complex<double>* vec,
    const complex<double>* phases,
    const unsigned long long* occupations,
    long long dim_a,
    long long dim_b,
    long long nocc)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim_a * dim_b) return;
    const unsigned long long* orbs = occupations + (idx % dim_b) * nocc;
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < nocc; j++) {
        phase *= phases[orbs[j]];
    }
    vec[idx] *= phase;
}
"""
)

_KERNEL_NAMES = {
    0: "apply_num_op_sum_evolution_rows",
    1: "apply_num_op_sum_evolution_cols",
}


def apply_num_op_sum_evolution_in_place(
    vec: cupy.ndarray,
    phases: cupy.ndarray,
    occupations: cupy.ndarray,
    axis: int = 0,
) -> None:
    """Apply time evolution by a sum of number operators in-place.

    Args:
        vec: The state vector, as a two-dimensional array.
        phases: The phase associated with each orbital.
        occupations: The occupied orbitals of each determinant of the spin sector.
        axis: The axis of ``vec`` indexed by ``occupations``.
    """
    validate_vec(vec)
    dim_a, dim_b = vec.shape
    launch(
        _MODULE,
        _KERNEL_NAMES[axis],
        dim_a * dim_b,
        (
            vec,
            cupy.ascontiguousarray(phases),
            occupations,
            np.int64(dim_a),
            np.int64(dim_b),
            np.int64(occupations.shape[1]),
        ),
    )
