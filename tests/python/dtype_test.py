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

"""Tests for state vector data type handling."""

from __future__ import annotations

from typing import Any

import numpy as np

import ffsim

RNG = np.random.default_rng(223104777685777786317612746867353875493)

# (input dtype, expected output dtype)
STATE_DTYPES = [
    (np.complex128, np.complex128),
    (np.complex64, np.complex64),
    (np.float64, np.complex128),
    (np.float32, np.complex64),
]


def assert_preserves_dtype(func, dim: int):
    """Check how func(vec, copy) handles the data type of vec.

    Complex dtypes should be preserved, real dtypes promoted to the complex dtype of
    the same precision, and the result should match double precision.
    """
    for in_dtype, out_dtype in STATE_DTYPES:
        atol = 1e-5 if out_dtype == np.complex64 else 1e-12
        vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=in_dtype)
        original = vec.copy()
        expected = func(vec.astype(np.complex128), True)

        result = func(vec, True)
        assert result.dtype == out_dtype
        assert not np.shares_memory(result, vec)
        np.testing.assert_array_equal(vec, original)
        np.testing.assert_allclose(result, expected, atol=atol)

        result = func(vec, False)
        assert result.dtype == out_dtype
        np.testing.assert_allclose(result, expected, atol=atol)


def test_gates_spinful_dtype():
    """Spinful gates preserve complex dtypes and promote real dtypes."""
    norb, nelec = 5, (3, 2)
    dim = ffsim.dim(norb, nelec)
    mat = ffsim.random.random_unitary(norb, seed=RNG)
    mat_b = ffsim.random.random_unitary(norb, seed=RNG)
    dc_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    coeffs = RNG.standard_normal(norb)
    quad_mat = ffsim.random.random_hermitian(norb, seed=RNG)
    gates = [
        lambda v, copy: ffsim.apply_orbital_rotation(v, mat, norb, nelec, copy=copy),
        lambda v, copy: ffsim.apply_orbital_rotation(
            v, (mat, mat_b), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_orbital_rotation(
            v, mat.astype(np.complex64), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_num_op_sum_evolution(
            v, coeffs, 0.7, norb, nelec, orbital_rotation=mat, copy=copy
        ),
        lambda v, copy: ffsim.apply_num_op_sum_evolution(
            v, coeffs.astype(np.float32), 0.7, norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_diag_coulomb_evolution(
            v, dc_mat, 0.7, norb, nelec, orbital_rotation=mat, copy=copy
        ),
        lambda v, copy: ffsim.apply_diag_coulomb_evolution(
            v, dc_mat, 0.7, norb, nelec, z_representation=True, copy=copy
        ),
        lambda v, copy: ffsim.apply_diag_coulomb_evolution(
            v, dc_mat.astype(np.float32), 0.7, norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_diag_coulomb_evolution(
            v, (dc_mat, None, dc_mat), 0.7, norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_quad_ham_evolution(
            v, quad_mat, 0.7, norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_givens_rotation(
            v, 0.7, (1, 2), norb, nelec, phi=0.3, copy=copy
        ),
        lambda v, copy: ffsim.apply_tunneling_interaction(
            v, 0.7, (1, 2), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_num_interaction(v, 0.7, 1, norb, nelec, copy=copy),
        lambda v, copy: ffsim.apply_num_num_interaction(
            v, 0.7, (1, 2), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_on_site_interaction(
            v, 0.7, 1, norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_num_op_prod_interaction(
            v, 0.7, ([1, 2], [0]), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_hop_gate(v, 0.7, (1, 2), norb, nelec, copy=copy),
        lambda v, copy: ffsim.apply_fsim_gate(
            v, 0.7, 0.3, (1, 2), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_fswap_gate(v, (1, 2), norb, nelec, copy=copy),
    ]
    for gate in gates:
        assert_preserves_dtype(gate, dim)


def test_gates_spinless_dtype():
    """Spinless gates preserve complex dtypes and promote real dtypes."""
    norb, nelec = 5, 2
    dim = ffsim.dim(norb, nelec)
    mat = ffsim.random.random_unitary(norb, seed=RNG)
    dc_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=RNG)
    coeffs = RNG.standard_normal(norb)
    gates = [
        lambda v, copy: ffsim.apply_orbital_rotation(v, mat, norb, nelec, copy=copy),
        lambda v, copy: ffsim.apply_num_op_sum_evolution(
            v, coeffs, 0.7, norb, nelec, orbital_rotation=mat, copy=copy
        ),
        lambda v, copy: ffsim.apply_diag_coulomb_evolution(
            v, dc_mat, 0.7, norb, nelec, orbital_rotation=mat, copy=copy
        ),
        lambda v, copy: ffsim.apply_givens_rotation(
            v, 0.7, (1, 2), norb, nelec, copy=copy
        ),
        lambda v, copy: ffsim.apply_num_num_interaction(
            v, 0.7, (1, 2), norb, nelec, copy=copy
        ),
    ]
    for gate in gates:
        assert_preserves_dtype(gate, dim)


def test_apply_unitary_dtype():
    """apply_unitary preserves complex dtypes and promotes real dtypes."""
    norb = 5
    cases: list[tuple[Any, int | tuple[int, int]]] = [
        (
            ffsim.random.random_ucj_op_spin_balanced(
                norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
            ),
            (3, 2),
        ),
        (
            ffsim.random.random_ucj_op_spin_unbalanced(
                norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
            ),
            (3, 2),
        ),
        (
            ffsim.random.random_ucj_op_spinless(
                norb, n_reps=2, with_final_orbital_rotation=True, seed=RNG
            ),
            2,
        ),
        (ffsim.random.random_givens_ansatz_op(norb, seed=RNG), (3, 2)),
        (ffsim.random.random_givens_ansatz_op(norb, seed=RNG), 2),
        (ffsim.random.random_num_num_ansatz_op_spin_balanced(norb, seed=RNG), (3, 2)),
        (
            ffsim.random.random_uccsd_op_restricted(
                norb, 2, with_final_orbital_rotation=True, seed=RNG
            ),
            (2, 2),
        ),
        (
            ffsim.random.random_uccsd_op_restricted_real(
                norb, 2, with_final_orbital_rotation=True, seed=RNG
            ),
            (2, 2),
        ),
        (
            ffsim.random.random_uccsd_op_unrestricted(
                norb, (3, 2), with_final_orbital_rotation=True, seed=RNG
            ),
            (3, 2),
        ),
        (
            ffsim.random.random_uccsd_op_unrestricted_real(
                norb, (3, 2), with_final_orbital_rotation=True, seed=RNG
            ),
            (3, 2),
        ),
        (
            ffsim.random.random_uccsd_op_spinless(
                norb, 2, with_final_orbital_rotation=True, seed=RNG
            ),
            2,
        ),
        (
            ffsim.random.random_uccsd_op_spinless_real(
                norb, 2, with_final_orbital_rotation=True, seed=RNG
            ),
            2,
        ),
    ]
    for op, nelec in cases:
        assert_preserves_dtype(
            lambda v, copy: ffsim.apply_unitary(v, op, norb, nelec, copy=copy),
            ffsim.dim(norb, nelec),
        )


def test_trotter_dtype():
    """Trotter simulation preserves complex dtypes and promotes real dtypes."""
    norb, nelec = 5, (3, 2)
    dim = ffsim.dim(norb, nelec)
    df_ham = ffsim.random.random_double_factorized_hamiltonian(norb, seed=RNG)
    dc_ham = ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG)
    assert_preserves_dtype(
        lambda v, copy: ffsim.simulate_trotter_double_factorized(
            v, df_ham, 0.5, norb=norb, nelec=nelec, n_steps=2, order=1, copy=copy
        ),
        dim,
    )
    assert_preserves_dtype(
        lambda v, copy: ffsim.simulate_trotter_diag_coulomb_split_op(
            v, dc_ham, 0.5, norb=norb, nelec=nelec, n_steps=2, order=1, copy=copy
        ),
        dim,
    )
    for in_dtype, out_dtype in STATE_DTYPES:
        vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=in_dtype)
        result, expected = (
            ffsim.simulate_qdrift_double_factorized(
                v, df_ham, 0.5, norb=norb, nelec=nelec, n_steps=3, n_samples=2, seed=1
            )
            for v in (vec, vec.astype(np.complex128))
        )
        assert result.dtype == out_dtype
        np.testing.assert_allclose(result, expected, atol=1e-5)
        # No evolution returns the initial state, still promoted to complex
        for time, n_steps, n_samples in [(0.0, 3, 1), (0.5, 0, 1), (0.0, 3, 2)]:
            result = ffsim.simulate_qdrift_double_factorized(
                vec,
                df_ham,
                time,
                norb=norb,
                nelec=nelec,
                n_steps=n_steps,
                n_samples=n_samples,
            )
            assert result.dtype == out_dtype
            assert not np.shares_memory(result, vec)
            np.testing.assert_array_equal(result, np.broadcast_to(vec, result.shape))


def test_linear_operator_single_precision():
    """Linear operators compute in double precision for single-precision input."""
    norb, nelec = 5, (3, 2)
    dim = ffsim.dim(norb, nelec)
    mat = ffsim.random.random_unitary(norb, seed=RNG)
    linops = [
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian(norb, seed=RNG), norb, nelec
        ),
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_molecular_hamiltonian_unrestricted(
                norb, seed=RNG, dtype=float
            ),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_diagonal_coulomb_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_double_factorized_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.random.random_single_factorized_hamiltonian(norb, seed=RNG),
            norb,
            nelec,
        ),
        ffsim.linear_operator(
            ffsim.fermion_operator(
                ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
            ),
            norb,
            nelec,
        ),
        ffsim.contract.one_body_linop(
            ffsim.random.random_hermitian(norb, seed=RNG), norb, nelec
        ),
        ffsim.contract.diag_coulomb_linop(
            ffsim.random.random_real_symmetric_matrix(norb, seed=RNG),
            norb,
            nelec,
            orbital_rotation=mat,
        ),
        ffsim.contract.num_op_sum_linop(
            RNG.standard_normal(norb), norb, nelec, orbital_rotation=mat
        ),
    ]
    for linop in linops:
        for dtype, double_dtype in [
            (np.complex64, np.complex128),
            (np.float32, np.float64),
        ]:
            vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=dtype)
            double_vec = vec.astype(double_dtype)
            original = vec.copy()
            double_original = double_vec.copy()
            result = linop @ vec
            expected = linop @ double_vec
            assert result.dtype == expected.dtype
            np.testing.assert_allclose(result, expected)
            np.testing.assert_array_equal(vec, original)
            np.testing.assert_array_equal(double_vec, double_original)
