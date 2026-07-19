# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for applying gates to state vectors on the GPU."""

from __future__ import annotations

import itertools

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

RNG = np.random.default_rng(207565096174961528831938608099311132632)


def _assert_matches_cpu(gpu_result, cpu_result):
    assert isinstance(gpu_result, cupy.ndarray)
    np.testing.assert_allclose(cupy.asnumpy(gpu_result), cpu_result, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_apply_orbital_rotation_spinful(norb: int, nelec: tuple[int, int]):
    """Test applying an orbital rotation on the GPU, spinful."""
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    mat_a = ffsim.random.random_unitary(norb, seed=RNG)
    mat_b = ffsim.random.random_unitary(norb, seed=RNG)
    mats: list[np.ndarray | tuple[np.ndarray | None, np.ndarray | None]] = [
        mat_a,
        (mat_a, mat_b),
        (mat_a, None),
        (None, mat_b),
        (None, None),
    ]
    for mat in mats:
        cpu_result = ffsim.apply_orbital_rotation(vec, mat, norb, nelec)
        gpu_result = ffsim.apply_orbital_rotation(vec_gpu, mat, norb, nelec)
        _assert_matches_cpu(gpu_result, cpu_result)
        # copy=True should leave the original vector untouched
        np.testing.assert_array_equal(cupy.asnumpy(vec_gpu), vec)


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_apply_orbital_rotation_spinless(norb: int, nocc: int):
    """Test applying an orbital rotation on the GPU, spinless."""
    dim = ffsim.dim(norb, nocc)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    mat = ffsim.random.random_unitary(norb, seed=RNG)
    cpu_result = ffsim.apply_orbital_rotation(vec, mat, norb, nocc)
    gpu_result = ffsim.apply_orbital_rotation(vec_gpu, mat, norb, nocc)
    _assert_matches_cpu(gpu_result, cpu_result)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_apply_num_op_sum_evolution_spinful(norb: int, nelec: tuple[int, int]):
    """Test applying number operator sum evolution on the GPU, spinful."""
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    coeffs_a = RNG.standard_normal(norb)
    coeffs_b = RNG.standard_normal(norb)
    time = RNG.standard_normal()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    all_coeffs: list[np.ndarray | tuple[np.ndarray | None, np.ndarray | None]] = [
        coeffs_a,
        (coeffs_a, coeffs_b),
        (coeffs_a, None),
        (None, coeffs_b),
    ]
    for coeffs in all_coeffs:
        for rotation in [None, orbital_rotation]:
            cpu_result = ffsim.apply_num_op_sum_evolution(
                vec, coeffs, time, norb, nelec, orbital_rotation=rotation
            )
            gpu_result = ffsim.apply_num_op_sum_evolution(
                vec_gpu, coeffs, time, norb, nelec, orbital_rotation=rotation
            )
            _assert_matches_cpu(gpu_result, cpu_result)


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_apply_num_op_sum_evolution_spinless(norb: int, nocc: int):
    """Test applying number operator sum evolution on the GPU, spinless."""
    dim = ffsim.dim(norb, nocc)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    coeffs = RNG.standard_normal(norb)
    time = RNG.standard_normal()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    for rotation in [None, orbital_rotation]:
        cpu_result = ffsim.apply_num_op_sum_evolution(
            vec, coeffs, time, norb, nocc, orbital_rotation=rotation
        )
        gpu_result = ffsim.apply_num_op_sum_evolution(
            vec_gpu, coeffs, time, norb, nocc, orbital_rotation=rotation
        )
        _assert_matches_cpu(gpu_result, cpu_result)


@pytest.mark.parametrize(
    "norb, nelec, z_representation",
    [
        (norb, nelec, z_representation)
        for (norb, nelec), z_representation in itertools.product(
            ffsim.testing.generate_norb_nelec(exhaustive=False), [False, True]
        )
    ],
)
def test_apply_diag_coulomb_evolution_spinful(
    norb: int, nelec: tuple[int, int], z_representation: bool
):
    """Test applying diagonal Coulomb evolution on the GPU, spinful."""
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    mat_aa = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    mat_ab = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    mat_bb = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    time = RNG.standard_normal()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    mats: list[
        np.ndarray | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ] = [
        mat_aa,
        (mat_aa, mat_ab, mat_bb),
        (mat_aa, None, None),
        (None, mat_ab, None),
        (None, None, mat_bb),
    ]
    for mat in mats:
        for rotation in [None, orbital_rotation]:
            cpu_result = ffsim.apply_diag_coulomb_evolution(
                vec,
                mat,
                time,
                norb,
                nelec,
                orbital_rotation=rotation,
                z_representation=z_representation,
            )
            gpu_result = ffsim.apply_diag_coulomb_evolution(
                vec_gpu,
                mat,
                time,
                norb,
                nelec,
                orbital_rotation=rotation,
                z_representation=z_representation,
            )
            _assert_matches_cpu(gpu_result, cpu_result)


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_apply_diag_coulomb_evolution_spinless(norb: int, nocc: int):
    """Test applying diagonal Coulomb evolution on the GPU, spinless."""
    dim = ffsim.dim(norb, nocc)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    mat = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    time = RNG.standard_normal()
    orbital_rotation = ffsim.random.random_unitary(norb, seed=RNG)
    for rotation in [None, orbital_rotation]:
        cpu_result = ffsim.apply_diag_coulomb_evolution(
            vec, mat, time, norb, nocc, orbital_rotation=rotation
        )
        gpu_result = ffsim.apply_diag_coulomb_evolution(
            vec_gpu, mat, time, norb, nocc, orbital_rotation=rotation
        )
        _assert_matches_cpu(gpu_result, cpu_result)


def test_basic_gates():
    """Test applying basic gates on the GPU."""
    norb = 5
    nelec = (3, 2)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    vec_gpu = cupy.asarray(vec)
    theta = RNG.standard_normal()
    phi = RNG.standard_normal()

    for target_orbs in itertools.combinations(range(norb), 2):
        for spin in ffsim.Spin:
            cpu_result = ffsim.apply_givens_rotation(
                vec, theta, target_orbs, norb, nelec, spin, phi=phi
            )
            gpu_result = ffsim.apply_givens_rotation(
                vec_gpu, theta, target_orbs, norb, nelec, spin, phi=phi
            )
            _assert_matches_cpu(gpu_result, cpu_result)

            cpu_result = ffsim.apply_tunneling_interaction(
                vec, theta, target_orbs, norb, nelec, spin
            )
            gpu_result = ffsim.apply_tunneling_interaction(
                vec_gpu, theta, target_orbs, norb, nelec, spin
            )
            _assert_matches_cpu(gpu_result, cpu_result)

            cpu_result = ffsim.apply_num_num_interaction(
                vec, theta, target_orbs, norb, nelec, spin
            )
            gpu_result = ffsim.apply_num_num_interaction(
                vec_gpu, theta, target_orbs, norb, nelec, spin
            )
            _assert_matches_cpu(gpu_result, cpu_result)

            cpu_result = ffsim.apply_hop_gate(
                vec, theta, target_orbs, norb, nelec, spin
            )
            gpu_result = ffsim.apply_hop_gate(
                vec_gpu, theta, target_orbs, norb, nelec, spin
            )
            _assert_matches_cpu(gpu_result, cpu_result)

            cpu_result = ffsim.apply_fsim_gate(
                vec, theta, phi, target_orbs, norb, nelec, spin
            )
            gpu_result = ffsim.apply_fsim_gate(
                vec_gpu, theta, phi, target_orbs, norb, nelec, spin
            )
            _assert_matches_cpu(gpu_result, cpu_result)

    for target_orb in range(norb):
        for spin in ffsim.Spin:
            cpu_result = ffsim.apply_num_interaction(
                vec, theta, target_orb, norb, nelec, spin
            )
            gpu_result = ffsim.apply_num_interaction(
                vec_gpu, theta, target_orb, norb, nelec, spin
            )
            _assert_matches_cpu(gpu_result, cpu_result)

        cpu_result = ffsim.apply_on_site_interaction(
            vec, theta, target_orb, norb, nelec
        )
        gpu_result = ffsim.apply_on_site_interaction(
            vec_gpu, theta, target_orb, norb, nelec
        )
        _assert_matches_cpu(gpu_result, cpu_result)
