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

_BLOCK_SIZE = 256

_GIVENS_KERNEL = cupy.RawKernel(
    r"""
#include <cupy/complex.cuh>

extern "C" __global__ void apply_givens_rotation(
    complex<double>* vec,
    double c,
    complex<double> s,
    const unsigned long long* slice1,
    const unsigned long long* slice2,
    long long n_pairs,
    long long dim_b)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_pairs * dim_b) return;
    long long pair = idx / dim_b;
    long long col = idx % dim_b;
    complex<double>* val_i = vec + slice1[pair] * dim_b + col;
    complex<double>* val_j = vec + slice2[pair] * dim_b + col;
    complex<double> i_old = *val_i;
    complex<double> j_old = *val_j;
    *val_i = c * i_old + s * j_old;
    *val_j = c * j_old - conj(s) * i_old;
}
""",
    "apply_givens_rotation",
)


def apply_givens_rotation_in_place(
    vec: cupy.ndarray,
    c: float,
    s: complex,
    slice1: cupy.ndarray,
    slice2: cupy.ndarray,
) -> None:
    """Apply a Givens rotation between two slices of a state vector."""
    n_pairs = len(slice1)
    if not n_pairs:
        return
    assert vec.dtype == cupy.complex128
    assert vec.flags.c_contiguous
    dim_b = vec.shape[1]
    size = n_pairs * dim_b
    n_blocks = (size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _GIVENS_KERNEL(
        (n_blocks,),
        (_BLOCK_SIZE,),
        (
            vec,
            np.float64(c),
            np.complex128(s),
            cupy.ascontiguousarray(slice1),
            cupy.ascontiguousarray(slice2),
            np.int64(n_pairs),
            np.int64(dim_b),
        ),
    )


@cache
def zero_one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, int]
) -> cupy.ndarray:
    """Device copy of the indices where the target orbitals are 01 or 10."""
    from ffsim.gates.orbital_rotation import _zero_one_subspace_indices

    return cupy.asarray(
        _zero_one_subspace_indices(norb, nocc, target_orbs).astype(np.uint64)
    )


@cache
def one_subspace_indices(
    norb: int, nocc: int, target_orbs: tuple[int, ...]
) -> cupy.ndarray:
    """Device copy of the indices where the target orbitals are 1."""
    from ffsim.gates.orbital_rotation import _one_subspace_indices

    return cupy.asarray(
        _one_subspace_indices(norb, nocc, target_orbs).astype(np.uint64)
    )
