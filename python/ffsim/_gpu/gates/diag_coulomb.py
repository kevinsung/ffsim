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

_BLOCK_SIZE = 256

_MODULE = cupy.RawModule(
    code=r"""
#include <cupy/complex.cuh>

extern "C" __global__ void compute_beta_phases_num_rep(
    const complex<double>* mat_exp_bb,
    const unsigned long long* occupations_b,
    long long dim_b,
    long long n_beta,
    long long norb,
    complex<double>* beta_phases)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim_b) return;
    const unsigned long long* orbs = occupations_b + i * n_beta;
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < n_beta; j++) {
        unsigned long long orb_1 = orbs[j];
        for (long long k = j; k < n_beta; k++) {
            phase *= mat_exp_bb[orb_1 * norb + orbs[k]];
        }
    }
    beta_phases[i] = phase;
}

extern "C" __global__ void compute_alpha_phases_num_rep(
    const complex<double>* mat_exp_aa,
    const complex<double>* mat_exp_ab,
    const unsigned long long* occupations_a,
    long long dim_a,
    long long n_alpha,
    long long norb,
    complex<double>* alpha_phases,
    complex<double>* phase_map)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim_a) return;
    const unsigned long long* orbs = occupations_a + i * n_alpha;
    complex<double>* row = phase_map + i * norb;
    for (long long k = 0; k < norb; k++) {
        row[k] = complex<double>(1.0, 0.0);
    }
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < n_alpha; j++) {
        unsigned long long orb_1 = orbs[j];
        const complex<double>* mat_row = mat_exp_ab + orb_1 * norb;
        for (long long k = 0; k < norb; k++) {
            row[k] *= mat_row[k];
        }
        for (long long k = j; k < n_alpha; k++) {
            phase *= mat_exp_aa[orb_1 * norb + orbs[k]];
        }
    }
    alpha_phases[i] = phase;
}

extern "C" __global__ void apply_phases_num_rep(
    complex<double>* vec,
    const complex<double>* alpha_phases,
    const complex<double>* beta_phases,
    const complex<double>* phase_map,
    const unsigned long long* occupations_b,
    long long dim_a,
    long long dim_b,
    long long n_beta,
    long long norb)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim_a * dim_b) return;
    long long a = idx / dim_b;
    long long b = idx % dim_b;
    complex<double> phase = alpha_phases[a] * beta_phases[b];
    const unsigned long long* orbs = occupations_b + b * n_beta;
    const complex<double>* row = phase_map + a * norb;
    for (long long j = 0; j < n_beta; j++) {
        phase *= row[orbs[j]];
    }
    vec[idx] *= phase;
}

extern "C" __global__ void compute_beta_phases_z_rep(
    const complex<double>* mat_exp_bb,
    const complex<double>* mat_exp_bb_conj,
    const long long* strings_b,
    long long dim_b,
    long long norb,
    complex<double>* beta_phases)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim_b) return;
    long long str0 = strings_b[i];
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < norb; j++) {
        long long sign_j = (str0 >> j) & 1;
        for (long long k = j + 1; k < norb; k++) {
            long long sign_k = (str0 >> k) & 1;
            const complex<double>* mat =
                (sign_j ^ sign_k) ? mat_exp_bb_conj : mat_exp_bb;
            phase *= mat[j * norb + k];
        }
    }
    beta_phases[i] = phase;
}

extern "C" __global__ void compute_alpha_phases_z_rep(
    const complex<double>* mat_exp_aa,
    const complex<double>* mat_exp_aa_conj,
    const complex<double>* mat_exp_ab,
    const complex<double>* mat_exp_ab_conj,
    const long long* strings_a,
    long long dim_a,
    long long norb,
    complex<double>* alpha_phases,
    complex<double>* phase_map)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim_a) return;
    long long str0 = strings_a[i];
    complex<double>* row = phase_map + i * norb;
    for (long long k = 0; k < norb; k++) {
        row[k] = complex<double>(1.0, 0.0);
    }
    complex<double> phase(1.0, 0.0);
    for (long long j = 0; j < norb; j++) {
        long long sign_j = (str0 >> j) & 1;
        const complex<double>* mat_row =
            (sign_j ? mat_exp_ab_conj : mat_exp_ab) + j * norb;
        for (long long k = 0; k < norb; k++) {
            row[k] *= mat_row[k];
        }
        for (long long k = j + 1; k < norb; k++) {
            long long sign_k = (str0 >> k) & 1;
            const complex<double>* mat =
                (sign_j ^ sign_k) ? mat_exp_aa_conj : mat_exp_aa;
            phase *= mat[j * norb + k];
        }
    }
    alpha_phases[i] = phase;
}

extern "C" __global__ void apply_phases_z_rep(
    complex<double>* vec,
    const complex<double>* alpha_phases,
    const complex<double>* beta_phases,
    const complex<double>* phase_map,
    const long long* strings_b,
    long long dim_a,
    long long dim_b,
    long long norb)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim_a * dim_b) return;
    long long a = idx / dim_b;
    long long b = idx % dim_b;
    long long str0 = strings_b[b];
    complex<double> phase = alpha_phases[a] * beta_phases[b];
    const complex<double>* row = phase_map + a * norb;
    for (long long j = 0; j < norb; j++) {
        complex<double> phase_shift = row[j];
        if ((str0 >> j) & 1) {
            phase_shift = conj(phase_shift);
        }
        phase *= phase_shift;
    }
    vec[idx] *= phase;
}
"""
)


def _launch(kernel_name: str, size: int, args: tuple) -> None:
    n_blocks = (size + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _MODULE.get_function(kernel_name)((n_blocks,), (_BLOCK_SIZE,), args)


def apply_diag_coulomb_evolution_in_place_num_rep(
    vec: cupy.ndarray,
    mat_exp_aa: np.ndarray,
    mat_exp_ab: np.ndarray,
    mat_exp_bb: np.ndarray,
    norb: int,
    occupations_a: cupy.ndarray,
    occupations_b: cupy.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place."""
    assert vec.dtype == cupy.complex128
    assert vec.flags.c_contiguous
    dim_a, dim_b = vec.shape
    n_alpha = occupations_a.shape[1]
    n_beta = occupations_b.shape[1]
    mat_exp_aa = cupy.asarray(mat_exp_aa, dtype=complex)
    mat_exp_ab = cupy.asarray(mat_exp_ab, dtype=complex)
    mat_exp_bb = cupy.asarray(mat_exp_bb, dtype=complex)
    alpha_phases = cupy.empty(dim_a, dtype=complex)
    beta_phases = cupy.empty(dim_b, dtype=complex)
    phase_map = cupy.empty((dim_a, norb), dtype=complex)
    _launch(
        "compute_beta_phases_num_rep",
        dim_b,
        (
            mat_exp_bb,
            occupations_b,
            np.int64(dim_b),
            np.int64(n_beta),
            np.int64(norb),
            beta_phases,
        ),
    )
    _launch(
        "compute_alpha_phases_num_rep",
        dim_a,
        (
            mat_exp_aa,
            mat_exp_ab,
            occupations_a,
            np.int64(dim_a),
            np.int64(n_alpha),
            np.int64(norb),
            alpha_phases,
            phase_map,
        ),
    )
    _launch(
        "apply_phases_num_rep",
        dim_a * dim_b,
        (
            vec,
            alpha_phases,
            beta_phases,
            phase_map,
            occupations_b,
            np.int64(dim_a),
            np.int64(dim_b),
            np.int64(n_beta),
            np.int64(norb),
        ),
    )


def apply_diag_coulomb_evolution_in_place_z_rep(
    vec: cupy.ndarray,
    mat_exp_aa: np.ndarray,
    mat_exp_ab: np.ndarray,
    mat_exp_bb: np.ndarray,
    mat_exp_aa_conj: np.ndarray,
    mat_exp_ab_conj: np.ndarray,
    mat_exp_bb_conj: np.ndarray,
    norb: int,
    strings_a: cupy.ndarray,
    strings_b: cupy.ndarray,
) -> None:
    """Apply time evolution by a diagonal Coulomb operator in-place, Z rep."""
    assert vec.dtype == cupy.complex128
    assert vec.flags.c_contiguous
    dim_a, dim_b = vec.shape
    mat_exp_aa = cupy.asarray(mat_exp_aa, dtype=complex)
    mat_exp_ab = cupy.asarray(mat_exp_ab, dtype=complex)
    mat_exp_bb = cupy.asarray(mat_exp_bb, dtype=complex)
    mat_exp_aa_conj = cupy.asarray(mat_exp_aa_conj, dtype=complex)
    mat_exp_ab_conj = cupy.asarray(mat_exp_ab_conj, dtype=complex)
    mat_exp_bb_conj = cupy.asarray(mat_exp_bb_conj, dtype=complex)
    alpha_phases = cupy.empty(dim_a, dtype=complex)
    beta_phases = cupy.empty(dim_b, dtype=complex)
    phase_map = cupy.empty((dim_a, norb), dtype=complex)
    _launch(
        "compute_beta_phases_z_rep",
        dim_b,
        (
            mat_exp_bb,
            mat_exp_bb_conj,
            strings_b,
            np.int64(dim_b),
            np.int64(norb),
            beta_phases,
        ),
    )
    _launch(
        "compute_alpha_phases_z_rep",
        dim_a,
        (
            mat_exp_aa,
            mat_exp_aa_conj,
            mat_exp_ab,
            mat_exp_ab_conj,
            strings_a,
            np.int64(dim_a),
            np.int64(norb),
            alpha_phases,
            phase_map,
        ),
    )
    _launch(
        "apply_phases_z_rep",
        dim_a * dim_b,
        (
            vec,
            alpha_phases,
            beta_phases,
            phase_map,
            strings_b,
            np.int64(dim_a),
            np.int64(dim_b),
            np.int64(norb),
        ),
    )
