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

from functools import cache

import cupy  # type: ignore
import numpy as np

from ffsim._gpu import validate_vec
from ffsim._gpu._kernels import launch

# The two kernels differ only in which axis the rotated pairs index, so that the
# beta sector can be rotated in the columns of the vector without transposing it
# into a contiguous copy. Each maps consecutive threads onto the axis that is
# contiguous in memory for that case.
_MODULE = cupy.RawModule(
    code=r"""
#include <cupy/complex.cuh>

__device__ __forceinline__ void rotate(
    complex<double>* val_i, complex<double>* val_j, double c, complex<double> s)
{
    complex<double> i_old = *val_i;
    complex<double> j_old = *val_j;
    *val_i = c * i_old + s * j_old;
    *val_j = c * j_old - conj(s) * i_old;
}

extern "C" __global__ void apply_givens_rotation_rows(
    complex<double>* vec,
    double c,
    complex<double> s,
    const unsigned long long* slice1,
    const unsigned long long* slice2,
    long long n_pairs,
    long long dim_a,
    long long dim_b)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs * dim_b) return;
    long long pair = idx / dim_b;
    long long col = idx % dim_b;
    rotate(vec + slice1[pair] * dim_b + col, vec + slice2[pair] * dim_b + col, c, s);
}

extern "C" __global__ void apply_givens_rotation_cols(
    complex<double>* vec,
    double c,
    complex<double> s,
    const unsigned long long* slice1,
    const unsigned long long* slice2,
    long long n_pairs,
    long long dim_a,
    long long dim_b)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs * dim_a) return;
    long long row = idx / n_pairs;
    long long pair = idx % n_pairs;
    complex<double>* base = vec + row * dim_b;
    rotate(base + slice1[pair], base + slice2[pair], c, s);
}
"""
)

_KERNEL_NAMES = {
    0: "apply_givens_rotation_rows",
    1: "apply_givens_rotation_cols",
}


def apply_givens_rotation_in_place(
    vec: cupy.ndarray,
    c: float,
    s: complex,
    slice1: cupy.ndarray,
    slice2: cupy.ndarray,
    axis: int = 0,
) -> None:
    """Apply a Givens rotation between two slices of a state vector.

    Args:
        vec: The state vector, as a two-dimensional array.
        c: The cosine of the rotation angle.
        s: The sine of the rotation angle.
        slice1: The indices of the first slice of each rotated pair.
        slice2: The indices of the second slice of each rotated pair.
        axis: The axis of ``vec`` indexed by the slices.
    """
    validate_vec(vec)
    n_pairs = len(slice1)
    dim_a, dim_b = vec.shape
    launch(
        _MODULE,
        _KERNEL_NAMES[axis],
        n_pairs * (dim_b if axis == 0 else dim_a),
        (
            vec,
            np.float64(c),
            np.complex128(s),
            cupy.ascontiguousarray(slice1),
            cupy.ascontiguousarray(slice2),
            np.int64(n_pairs),
            np.int64(dim_a),
            np.int64(dim_b),
        ),
    )


# The caches below are keyed on the current CUDA device as well as the orbitals,
# because a CuPy array may only be used on the device it was allocated on.
# Without the device in the key, a multi-GPU process would hand device 0's
# pointers to a kernel launched on device 1.


@cache
def _zero_one_subspace_indices(
    device_id: int, norb: int, nocc: int, target_orbs: tuple[int, int]
) -> cupy.ndarray:
    from ffsim.gates.orbital_rotation import _zero_one_subspace_indices as cpu

    return cupy.asarray(cpu(norb, nocc, target_orbs).astype(np.uint64))


def zero_one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, int]
) -> cupy.ndarray:
    """Device copy of the indices where the target orbitals are 01 or 10."""
    return _zero_one_subspace_indices(
        cupy.cuda.runtime.getDevice(), norb, nocc, target_orbs
    )


@cache
def _one_subspace_indices(
    device_id: int, norb: int, nocc: int, target_orbs: tuple[int, ...]
) -> cupy.ndarray:
    from ffsim.gates.orbital_rotation import _one_subspace_indices as cpu

    return cupy.asarray(cpu(norb, nocc, target_orbs).astype(np.uint64))


def one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, ...]
) -> cupy.ndarray:
    """Device copy of the indices where the target orbitals are 1."""
    return _one_subspace_indices(cupy.cuda.runtime.getDevice(), norb, nocc, target_orbs)
