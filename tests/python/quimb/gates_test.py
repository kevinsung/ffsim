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
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CPhaseGate, PhaseGate, XGate, XXPlusYYGate
from qiskit.quantum_info import Statevector

import ffsim


@pytest.mark.parametrize("norb, nelec", [(3, (2, 1)), (4, (2, 2))])
def test_orbital_rotation(norb: int, nelec: tuple[int, int]):
    """Test orbital rotation."""
    rng = np.random.default_rng(3315)

    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=rng)
    qiskit_vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec, norb, nelec)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    rotated_vec = ffsim.apply_orbital_rotation(
        vec, orbital_rotation, norb=norb, nelec=nelec
    )
    rotated_vec = ffsim.apply_orbital_rotation(
        rotated_vec, orbital_rotation, norb=norb, nelec=nelec
    )
    final_state_qiskit = ffsim.qiskit.ffsim_vec_to_qiskit_vec(rotated_vec, norb, nelec)

    tensor = quimb.tensor.Tensor(
        qiskit_vec.reshape([2] * 2 * norb).transpose(),
        inds=[f"k{i}" for i in range(2 * norb)],
    )
    tn = quimb.tensor.TensorNetwork([tensor])
    psi0 = quimb.tensor.tensor_1d.TensorNetwork1DVector.from_TN(
        tn, L=2 * norb, site_tag_id="I{}", site_ind_id="k{}"
    )
    circuit = quimb.tensor.Circuit(psi0=psi0)
    gates = list(ffsim.quimb.orbital_rotation_gates(orbital_rotation))
    circuit.apply_gates(gates)
    circuit.apply_gates(gates)

    final_state = circuit.to_dense(reverse=True).reshape(-1)
    ffsim.testing.assert_allclose_up_to_global_phase(final_state, final_state_qiskit)


def test_quimb_circuit():
    qubits = QuantumRegister(3)
    circuit = QuantumCircuit(qubits)
    a, b, c = qubits
    circuit.append(XGate(), [a])
    circuit.append(XXPlusYYGate(0.1, 0.2), [a, b])
    circuit.append(PhaseGate(0.3), [b])
    circuit.append(CPhaseGate(0.4), [b, c])

    quimb_circuit = ffsim.quimb.quimb_circuit(circuit)

    qiskit_vec = np.array(Statevector(circuit))
    quimb_vec = quimb_circuit.to_dense(reverse=True).reshape(-1)

    ffsim.testing.assert_allclose_up_to_global_phase(quimb_vec, qiskit_vec)
