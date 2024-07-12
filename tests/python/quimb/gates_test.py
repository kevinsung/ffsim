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

import numpy as np
import pytest
import quimb
import quimb.tensor
from qiskit.quantum_info import SparsePauliOp, Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", [(3, (2, 1)), (4, (2, 2))])
def test_orbital_rotation(norb: int, nelec: tuple[int, int]):
    """Test orbital rotation."""
    rng = np.random.default_rng(3315)

    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=rng)
    qiskit_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec, norb, nelec)
    statevector = Statevector(qiskit_vec)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    rotated_vec = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    rotated_vec = ffsim.apply_orbital_rotation(
        rotated_vec, orbital_rotation, norb=norb, nelec=nelec
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

    tensor = quimb.tensor.Tensor(
        qiskit_vec.reshape([2] * 2 * norb),
        inds=[f"k{i}" for i in range(2 * norb - 1, -1, -1)],
    )
    tn = quimb.tensor.TensorNetwork([tensor])
    psi0 = quimb.tensor.tensor_1d.TensorNetwork1DVector.from_TN(
        tn, L=2 * norb, site_tag_id="I{}", site_ind_id="k{}"
    )
    circuit = quimb.tensor.Circuit(psi0=psi0)
    gates = list(ffsim.quimb.orbital_rotation_gates(orbital_rotation))
    circuit.apply_gates(gates)
    circuit.apply_gates(gates)

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
