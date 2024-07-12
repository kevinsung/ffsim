# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for gates."""

import itertools

import numpy as np
import pytest
import quimb
import quimb.tensor
from qiskit.quantum_info import SparsePauliOp, Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", [(4, (2, 2)), (5, (3, 2))])
def test_orbital_rotation(norb: int, nelec: tuple[int, int]):
    """Test orbital rotation."""
    rng = np.random.default_rng(3315)
    # TODO test on random state
    vec = ffsim.hartree_fock_state(norb, nelec)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    rotated_vec = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    statevector = Statevector(
        ffsim.qiskit.ffsim_vec_to_qiskit_vec(rotated_vec, norb, nelec)
    )

    x_ops = [
        SparsePauliOp.from_sparse_list([("X", (i,), 1.0)], num_qubits=2 * norb)
        for i in range(2 * norb)
    ]
    y_ops = [
        SparsePauliOp.from_sparse_list([("Y", (i,), 1.0)], num_qubits=2 * norb)
        for i in range(2 * norb)
    ]
    z_ops = [
        SparsePauliOp.from_sparse_list([("Z", (i,), 1.0)], num_qubits=2 * norb)
        for i in range(2 * norb)
    ]
    x_expectations_expected = [statevector.expectation_value(op).real for op in x_ops]
    y_expectations_expected = [statevector.expectation_value(op).real for op in y_ops]
    z_expectations_expected = [statevector.expectation_value(op).real for op in z_ops]

    circuit = quimb.tensor.Circuit.from_gates(
        itertools.chain(
            ffsim.quimb.prepare_hartree_fock_gates(norb, nelec),
            ffsim.quimb.orbital_rotation_gates(orbital_rotation),
        )
    )
    x_expectations = [
        circuit.local_expectation(quimb.pauli("X"), (i,)) for i in range(2 * norb)
    ]
    y_expectations = [
        circuit.local_expectation(quimb.pauli("Y"), (i,)) for i in range(2 * norb)
    ]
    z_expectations = [
        circuit.local_expectation(quimb.pauli("Z"), (i,)) for i in range(2 * norb)
    ]

    np.testing.assert_allclose(x_expectations, x_expectations_expected, atol=1e-12)
    np.testing.assert_allclose(y_expectations, y_expectations_expected, atol=1e-12)
    np.testing.assert_allclose(z_expectations, z_expectations_expected, atol=1e-12)
