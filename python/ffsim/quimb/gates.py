import cmath
import math

import numpy as np
import quimb.tensor

from ffsim import linalg


def prepare_hartree_fock_gates(norb: int, nelec: int):
    n_alpha, n_beta = nelec
    for orb in range(n_alpha):
        yield quimb.tensor.Gate("X", params=[], qubits=[orb])
    for orb in range(n_beta):
        yield quimb.tensor.Gate("X", params=[], qubits=[orb + norb])


def orbital_rotation_gates(orbital_rotation: np.ndarray):
    norb, _ = orbital_rotation.shape
    givens_rotations, phase_shifts = linalg.givens_decomposition(orbital_rotation)
    for sigma in range(2):
        for c, s, i, j in givens_rotations:
            # TODO test whether first or second RZ should have minus sign
            yield quimb.tensor.Gate(
                "RZ", params=[-cmath.phase(s)], qubits=[i + sigma * norb]
            )
            yield quimb.tensor.Gate(
                "GIVENS",
                params=[math.acos(c)],
                qubits=[i + sigma * norb, j + sigma * norb],
            )
            yield quimb.tensor.Gate(
                "RZ", params=[cmath.phase(s)], qubits=[i + sigma * norb]
            )
        for i, phase_shift in enumerate(phase_shifts):
            # TODO test whether phase should have minus sign
            yield quimb.tensor.Gate(
                "RZ", params=[-cmath.phase(phase_shift)], qubits=[i + sigma * norb]
            )
