import cmath
import math
from collections.abc import Iterator

import numpy as np
import quimb.tensor
from qiskit.circuit import Instruction, QuantumCircuit

from ffsim import linalg


def prepare_hartree_fock_gates(norb: int, nelec: int) -> Iterator[quimb.tensor.Gate]:
    n_alpha, n_beta = nelec
    for orb in range(n_alpha):
        yield quimb.tensor.Gate("X", params=[], qubits=[orb])
    for orb in range(n_beta):
        yield quimb.tensor.Gate("X", params=[], qubits=[orb + norb])


def orbital_rotation_gates(orbital_rotation: np.ndarray) -> Iterator[quimb.tensor.Gate]:
    # TODO support different orbital rotations for each spin
    norb, _ = orbital_rotation.shape
    givens_rotations, phase_shifts = linalg.givens_decomposition(orbital_rotation)
    for sigma in range(2):
        for c, s, i, j in givens_rotations:
            yield quimb.tensor.Gate(
                "RZ", params=[cmath.phase(s)], qubits=[i + sigma * norb]
            )
            yield quimb.tensor.Gate(
                "GIVENS",
                params=[math.acos(c)],
                qubits=[i + sigma * norb, j + sigma * norb],
            )
            yield quimb.tensor.Gate(
                "RZ", params=[-cmath.phase(s)], qubits=[i + sigma * norb]
            )
        for i, phase_shift in enumerate(phase_shifts):
            yield quimb.tensor.Gate(
                "RZ", params=[cmath.phase(phase_shift)], qubits=[i + sigma * norb]
            )


def quimb_circuit(circuit: QuantumCircuit) -> quimb.tensor.Circuit:
    quimb_circuit = quimb.tensor.Circuit(circuit.num_qubits)
    for instruction in circuit.data:
        op = instruction.operation
        qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        quimb_circuit.apply_gates(list(quimb_gates(op, qubits)))
    return quimb_circuit


def quimb_gates(op: Instruction, qubits: list[int]) -> Iterator[quimb.tensor.Gate]:
    if op.name == "x":
        yield quimb.tensor.Gate("X", params=[], qubits=qubits)
    if op.name == "p":
        yield quimb.tensor.Gate("RZ", params=op.params, qubits=qubits)
    if op.name == "cp":
        (theta,) = op.params
        a, b = qubits
        yield quimb.tensor.Gate("RZZ", params=[-theta], qubits=[a, b])
        yield quimb.tensor.Gate("RZ", params=[theta], qubits=[a])
        yield quimb.tensor.Gate("RZ", params=[theta], qubits=[b])
    if op.name == "xx_plus_yy":
        theta, beta = op.params
        phi = beta + 0.5 * math.pi
        a, b = qubits
        yield quimb.tensor.Gate("RZ", params=[phi], qubits=[a])
        yield quimb.tensor.Gate("GIVENS", params=[0.5 * theta], qubits=[a, b])
        yield quimb.tensor.Gate("RZ", params=[-phi], qubits=[a])
