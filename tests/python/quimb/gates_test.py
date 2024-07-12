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


def test_quimb_circuit_basic():
    """Test quimb circuit with basic gate."""
    rng = np.random.default_rng(3377)
    qubits = QuantumRegister(8)
    circuit = QuantumCircuit(qubits)
    a, b, c, d, e, f, g, h = qubits

    circuit.append(XGate(), [a])
    circuit.append(XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)), [a, b])
    circuit.append(PhaseGate(rng.uniform(-10, 10)), [b])
    circuit.append(CPhaseGate(rng.uniform(-10, 10)), [b, c])
    circuit.append(XGate(), [d])
    circuit.append(XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)), [e, f])
    circuit.append(PhaseGate(rng.uniform(-10, 10)), [g])
    circuit.append(CPhaseGate(rng.uniform(-10, 10)), [g, h])
    circuit.append(XGate(), [f])
    circuit.append(XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)), [g, h])

    quimb_circuit = ffsim.quimb.quimb_circuit(circuit)
    qiskit_vec = np.array(Statevector(circuit))
    quimb_vec = quimb_circuit.to_dense(reverse=True).reshape(-1)
    ffsim.testing.assert_allclose_up_to_global_phase(quimb_vec, qiskit_vec)


def test_quimb_circuit_fermionic():
    """Test quimb circuit with fermionic gates."""
    rng = np.random.default_rng(7564)
    norb = 4
    nelec = (2, 2)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)

    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.append(
        ffsim.qiskit.DiagCoulombEvolutionJW(norb, diag_coulomb_mat, time=1.0), qubits
    )

    quimb_circuit = ffsim.quimb.quimb_circuit(
        circuit.decompose("hartree_fock_jw").decompose()
    )
    qiskit_vec = np.array(Statevector(circuit))
    quimb_vec = quimb_circuit.to_dense(reverse=True).reshape(-1)
    ffsim.testing.assert_allclose_up_to_global_phase(quimb_vec, qiskit_vec)


def test_quimb_circuit_ucj():
    """Test quimb circuit with UCJ circuit."""
    rng = np.random.default_rng(7564)
    norb = 4
    nelec = (2, 2)
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)

    ucj_op = ffsim.random.random_ucj_op_spin_balanced(norb=norb, n_reps=1, seed=rng)

    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    quimb_circuit = ffsim.quimb.quimb_circuit(circuit.decompose(reps=2))
    qiskit_vec = np.array(Statevector(circuit))
    quimb_vec = quimb_circuit.to_dense(reverse=True).reshape(-1)
    ffsim.testing.assert_allclose_up_to_global_phase(quimb_vec, qiskit_vec)
