# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FfsimSampler."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CPhaseGate,
    PhaseGate,
    RZGate,
    RZZGate,
    XGate,
    XXPlusYYGate,
)
from qiskit.primitives import StatevectorSampler

import ffsim


def _fidelity(probs1: dict, probs2: dict) -> float:
    result = 0.0
    for bitstring in probs1.keys() | probs2.keys():
        prob1 = probs1.get(bitstring, 0)
        prob2 = probs2.get(bitstring, 0)
        result += math.sqrt(prob1 * prob2)
    return result**2


def _brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 4)))
def test_random_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test sampler with random gates."""
    rng = np.random.default_rng(12285)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    diag_coulomb_mat = ffsim.random.random_real_symmetric_matrix(norb, seed=rng)
    ucj_op_balanced = ffsim.random.random_ucj_op_spin_balanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )
    ucj_op_unbalanced = ffsim.random.random_ucj_op_spin_unbalanced(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )
    df_hamiltonian_num_rep = ffsim.random.random_double_factorized_hamiltonian(
        norb, rank=3, z_representation=False, seed=rng
    )
    df_hamiltonian_z_rep = ffsim.random.random_double_factorized_hamiltonian(
        norb, rank=3, z_representation=True, seed=rng
    )
    interaction_pairs = list(_brickwork(norb, norb))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = rng.uniform(-np.pi, np.pi, size=norb)
    givens_ansatz_op = ffsim.GivensAnsatzOp(
        norb, interaction_pairs, thetas, phis=phis, phase_angles=phase_angles
    )

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.append(
        ffsim.qiskit.DiagCoulombEvolutionJW(norb, diag_coulomb_mat, time=1.0), qubits
    )
    circuit.append(ffsim.qiskit.GivensAnsatzOpJW(givens_ansatz_op), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op_balanced), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op_unbalanced), qubits)
    circuit.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(
            df_hamiltonian_num_rep, time=1.0
        ),
        qubits,
    )
    circuit.append(
        ffsim.qiskit.SimulateTrotterDoubleFactorizedJW(df_hamiltonian_z_rep, time=1.0),
        qubits,
    )
    circuit.measure_all()

    # Sample using ffsim Sampler
    shots = 5000
    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    # Compute empirical distribution
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nelec)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots

    # Compute exact probability distribution
    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False)
    )
    exact_probs = np.abs(vec) ** 2

    # Check fidelity
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(1, 4)))
def test_random_gates_spinless(norb: int, nocc: int):
    """Test sampler with random spinless gates."""
    rng = np.random.default_rng(52622)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    interaction_pairs = list(_brickwork(norb, norb))
    thetas = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phis = rng.uniform(-np.pi, np.pi, size=len(interaction_pairs))
    phase_angles = rng.uniform(-np.pi, np.pi, size=norb)
    givens_ansatz_op = ffsim.GivensAnsatzOp(
        norb, interaction_pairs, thetas, phis=phis, phase_angles=phase_angles
    )
    ucj_op = ffsim.random.random_ucj_op_spinless(
        norb, n_reps=2, with_final_orbital_rotation=True, seed=rng
    )

    # Construct circuit
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockSpinlessJW(norb, nocc), qubits)
    circuit.append(
        ffsim.qiskit.OrbitalRotationSpinlessJW(norb, orbital_rotation), qubits
    )
    circuit.append(ffsim.qiskit.GivensAnsatzOpSpinlessJW(givens_ansatz_op), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinlessJW(ucj_op), qubits)
    circuit.measure_all()

    # Sample using ffsim Sampler
    shots = 1000
    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    # Compute empirical distribution
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nocc)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nocc), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots

    # Compute exact probability distribution
    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False)
    )
    exact_probs = np.abs(vec) ** 2

    # Check fidelity
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999


@pytest.mark.parametrize("norb, nelec", ffsim.testing.generate_norb_nelec(range(1, 4)))
def test_measure_subset_spinful(norb: int, nelec: tuple[int, int]):
    """Test measuring a subset of qubits."""
    rng = np.random.default_rng(5332)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)

    # Construct circuit
    qubits = QuantumRegister(2 * norb, name="q")
    clbits = ClassicalRegister(norb, name="meas")
    measured_qubits = list(rng.choice(qubits, size=len(clbits), replace=False))
    measured_clbits = list(rng.choice(clbits, size=len(clbits), replace=False))
    circuit = QuantumCircuit(qubits, clbits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure(measured_qubits, measured_clbits)

    # Sample using ffsim Sampler
    shots = 3000
    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    ffsim_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(ffsim_probs.values()), 1)

    # Sample using Qiskit Sampler
    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    qiskit_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(qiskit_probs.values()), 1)

    # Check fidelity
    assert _fidelity(ffsim_probs, qiskit_probs) > 0.99


@pytest.mark.parametrize("norb, nocc", ffsim.testing.generate_norb_nocc(range(2, 4)))
def test_measure_subset_spinless(norb: int, nocc: int):
    """Test measuring a subset of qubits, spinless."""
    rng = np.random.default_rng(5332)

    # Initialize test objects
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(
        norb, (nocc, 0), seed=rng
    )[0]

    # Construct circuit
    qubits = QuantumRegister(norb, name="q")
    clbits = ClassicalRegister(norb - 1, name="meas")
    measured_qubits = list(rng.choice(qubits, size=len(clbits), replace=False))
    measured_clbits = list(rng.choice(clbits, size=len(clbits), replace=False))
    circuit = QuantumCircuit(qubits, clbits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantSpinlessJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure(measured_qubits, measured_clbits)

    # Sample using ffsim Sampler
    shots = 3000
    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    ffsim_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(ffsim_probs.values()), 1)

    # Sample using Qiskit Sampler
    sampler = StatevectorSampler(default_shots=shots, seed=rng)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    qiskit_probs = {bitstring: count / shots for bitstring, count in counts.items()}
    np.testing.assert_allclose(sum(qiskit_probs.values()), 1)

    # Check fidelity
    assert _fidelity(ffsim_probs, qiskit_probs) > 0.99


@pytest.mark.parametrize(
    "norb, nelec, global_depolarizing",
    [
        (5, (3, 2), 0.0),
        (5, (3, 2), 0.1),
        (5, (3, 2), 1.0),
    ],
)
def test_global_depolarizing(
    norb: int, nelec: tuple[int, int], global_depolarizing: float
):
    """Test sampler with global depolarizing noise."""
    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.OrbitalRotationJW(norb, orbital_rotation), qubits)
    circuit.measure_all()

    # Sample using ffsim Sampler
    shots = 10_000
    sampler = ffsim.qiskit.FfsimSampler(
        default_shots=shots, global_depolarizing=global_depolarizing, seed=rng
    )
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    # Compute empirical distribution
    strings, counts = zip(*samples.items())
    assert sum(counts) == shots
    addresses = [int(s, 2) for s in strings]
    dim = 1 << 2 * norb
    empirical_probs = np.zeros(dim, dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots

    # Compute exact probability distribution
    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False)
    ).vec
    vec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec, norb, nelec)
    exact_probs = np.abs(vec) ** 2
    expected_probs = (1 - global_depolarizing) * exact_probs + global_depolarizing / dim

    # Check fidelity
    fidelity = np.sum(np.sqrt(exact_probs * empirical_probs))
    expected_fidelity = np.sum(np.sqrt(exact_probs * expected_probs))
    assert np.allclose(fidelity, expected_fidelity, rtol=1e-2, atol=1e-3)


def test_reproducible_with_seed():
    """Test sampler with random gates."""
    rng = np.random.default_rng(14062)

    norb = 4
    nelec = (2, 2)

    qubits = QuantumRegister(2 * norb, name="q")

    orbital_rotation = ffsim.random.random_unitary(norb, seed=rng)
    occupied_orbitals = ffsim.testing.random_occupied_orbitals(norb, nelec, seed=rng)

    circuit = QuantumCircuit(qubits)
    circuit.append(
        ffsim.qiskit.PrepareSlaterDeterminantJW(
            norb, occupied_orbitals, orbital_rotation=orbital_rotation
        ),
        qubits,
    )
    circuit.measure_all()

    shots = 3000

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=12345)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts_1 = pub_result.data.meas.get_counts()

    sampler = ffsim.qiskit.FfsimSampler(default_shots=shots, seed=12345)
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    counts_2 = pub_result.data.meas.get_counts()

    assert counts_1 == counts_2


def test_edge_cases():
    """Test edge cases."""
    with pytest.raises(
        ValueError, match="Circuit must contain at least one instruction."
    ):
        qubits = QuantumRegister(1, name="q")
        circuit = QuantumCircuit(qubits)
        circuit.measure_all()
        sampler = ffsim.qiskit.FfsimSampler(default_shots=1)
        pub = (circuit,)
        job = sampler.run([pub])
        _ = job.result()


@pytest.mark.parametrize(
    "norb, nelec",
    [
        (norb, nelec)
        for norb, nelec in ffsim.testing.generate_norb_nelec(range(1, 4))
        if nelec != (0, 0)
    ],
)
def test_qiskit_gates_spinful(norb: int, nelec: tuple[int, int]):
    """Test sampler with Qiskit gates, spinful."""
    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    n_alpha, n_beta = nelec
    for i in range(n_alpha):
        circuit.append(XGate(), [qubits[i]])
    for i in range(n_beta):
        circuit.append(XGate(), [qubits[norb + i]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[norb + j], qubits[norb + i]],
        )
    for q in qubits:
        circuit.append(PhaseGate(rng.uniform(-10, 10)), [q])
    for i, j in _brickwork(2 * norb, norb):
        circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    for q in qubits:
        circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    for i, j in _brickwork(2 * norb, norb):
        circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[norb + i], qubits[norb + j]],
        )
    circuit.measure_all()

    # Sample using ffsim Sampler
    shots = 5000
    sampler = ffsim.qiskit.FfsimSampler(
        default_shots=shots, norb=norb, nelec=nelec, seed=rng
    )
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    # Compute empirical distribution
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nelec)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nelec), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots

    # Compute exact probability distribution
    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False), norb=norb, nelec=nelec
    )
    exact_probs = np.abs(vec) ** 2

    # Check fidelity
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999


@pytest.mark.parametrize(
    "norb, nocc",
    [
        (norb, nocc)
        for norb, nocc in ffsim.testing.generate_norb_nocc(range(1, 4))
        if nocc
    ],
)
def test_qiskit_gates_spinless(norb: int, nocc: int):
    """Test sampler with Qiskit gates, spinless."""
    rng = np.random.default_rng(12285)

    # Construct circuit
    qubits = QuantumRegister(norb)
    circuit = QuantumCircuit(qubits)
    for i in range(nocc):
        circuit.append(XGate(), [qubits[i]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    for q in qubits:
        circuit.append(PhaseGate(rng.uniform(-10, 10)), [q])
    for i, j in _brickwork(norb, norb):
        circuit.append(CPhaseGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    for q in qubits:
        circuit.append(RZGate(rng.uniform(-10, 10)), [q])
    for i, j in _brickwork(norb, norb):
        circuit.append(RZZGate(rng.uniform(-10, 10)), [qubits[i], qubits[j]])
    for i, j in _brickwork(norb, norb):
        circuit.append(
            XXPlusYYGate(rng.uniform(-10, 10), rng.uniform(-10, 10)),
            [qubits[i], qubits[j]],
        )
    circuit.measure_all()

    # Sample using ffsim Sampler
    shots = 5000
    sampler = ffsim.qiskit.FfsimSampler(
        default_shots=shots, norb=norb, nelec=nocc, seed=rng
    )
    pub = (circuit,)
    job = sampler.run([pub])
    result = job.result()
    pub_result = result[0]
    samples = pub_result.data.meas.get_counts()

    # Compute empirical distribution
    strings, counts = zip(*samples.items())
    addresses = ffsim.strings_to_addresses(strings, norb, nocc)
    assert sum(counts) == shots
    empirical_probs = np.zeros(ffsim.dim(norb, nocc), dtype=float)
    empirical_probs[addresses] = np.array(counts) / shots

    # Compute exact probability distribution
    vec = ffsim.qiskit.final_state_vector(
        circuit.remove_final_measurements(inplace=False), norb=norb, nelec=nocc
    )
    exact_probs = np.abs(vec) ** 2

    # Check fidelity
    assert np.sum(np.sqrt(exact_probs * empirical_probs)) > 0.999
