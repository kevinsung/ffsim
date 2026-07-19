# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for simulating circuits and Trotter evolution on the GPU."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim

cupy = pytest.importorskip("cupy")

try:
    _n_devices = cupy.cuda.runtime.getDeviceCount()
except Exception:
    _n_devices = 0
if not _n_devices:
    pytest.skip("No CUDA device available", allow_module_level=True)

RNG = np.random.default_rng(306316061490345155131532323046719724284)


def _assert_matches_cpu(gpu_result, cpu_result):
    assert isinstance(gpu_result, cupy.ndarray)
    np.testing.assert_allclose(cupy.asnumpy(gpu_result), cpu_result, atol=1e-12)


@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_apply_unitary_ucj_spin_balanced(with_final_orbital_rotation: bool):
    """Test applying a spin-balanced UCJ operator to a GPU state vector."""
    norb = 5
    nelec = (3, 2)
    op = ffsim.random.random_ucj_op_spin_balanced(
        norb,
        n_reps=2,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec_gpu = cupy.asarray(vec)
    cpu_result = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
    gpu_result = ffsim.apply_unitary(vec_gpu, op, norb=norb, nelec=nelec)
    _assert_matches_cpu(gpu_result, cpu_result)
    # copy=True should leave the original vector untouched
    np.testing.assert_array_equal(cupy.asnumpy(vec_gpu), vec)


@pytest.mark.parametrize("with_final_orbital_rotation", [False, True])
def test_apply_unitary_ucj_spin_unbalanced(with_final_orbital_rotation: bool):
    """Test applying a spin-unbalanced UCJ operator to a GPU state vector."""
    norb = 5
    nelec = (3, 2)
    op = ffsim.random.random_ucj_op_spin_unbalanced(
        norb,
        n_reps=2,
        with_final_orbital_rotation=with_final_orbital_rotation,
        seed=RNG,
    )
    vec = ffsim.hartree_fock_state(norb, nelec)
    vec_gpu = cupy.asarray(vec)
    cpu_result = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
    gpu_result = ffsim.apply_unitary(vec_gpu, op, norb=norb, nelec=nelec)
    _assert_matches_cpu(gpu_result, cpu_result)


def test_simulate_trotter_diag_coulomb_split_op():
    """Test Trotter simulation of a diagonal Coulomb Hamiltonian on the GPU."""
    norb = 5
    nelec = (3, 2)
    hamiltonian = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    cpu_result = ffsim.simulate_trotter_diag_coulomb_split_op(
        vec, hamiltonian, time=0.7, norb=norb, nelec=nelec, n_steps=3, order=1
    )
    gpu_result = ffsim.simulate_trotter_diag_coulomb_split_op(
        vec_gpu, hamiltonian, time=0.7, norb=norb, nelec=nelec, n_steps=3, order=1
    )
    _assert_matches_cpu(gpu_result, cpu_result)
