# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for transformations between operator representations."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim

RNG = np.random.default_rng(107154886220866265173154836244421466745)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_jordan_wigner_random(norb: int, nelec: tuple[int, int]):
    """Test the Jordan-Wigner transformation on random operators."""
    op = ffsim.random.random_fermion_hamiltonian(norb, seed=RNG)
    qubit_op = ffsim.jordan_wigner(op, norb=norb)
    expected = ffsim.qiskit.qubit_operator_to_sparse_pauli_op(qubit_op, 2 * norb)
    np.testing.assert_allclose(
        expected.to_matrix(),
        ffsim.qiskit.jordan_wigner(op, norb=norb).to_matrix(),
        atol=1e-12,
    )


def test_jordan_wigner_inferred_norb():
    """Test that norb is inferred from the operator when not specified."""
    op = ffsim.FermionOperator({(ffsim.cre_a(2), ffsim.des_b(1)): 1.0})
    # norb is inferred as 3, so beta orbital 1 maps to qubit 4.
    assert ffsim.jordan_wigner(op) == ffsim.jordan_wigner(op, norb=3)
    assert ffsim.jordan_wigner(op) != ffsim.jordan_wigner(op, norb=4)


def test_jordan_wigner_bad_norb():
    """Test passing a bad number of spatial orbitals raises errors."""
    op = ffsim.FermionOperator({(ffsim.cre_a(3),): 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        _ = ffsim.jordan_wigner(op, norb=-1)
    with pytest.raises(ValueError, match="fewer"):
        _ = ffsim.jordan_wigner(op, norb=3)


def test_jordan_wigner_negative_orbital():
    """Test that a negative orbital index raises an error."""
    op = ffsim.FermionOperator({(ffsim.cre_a(-1),): 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        _ = ffsim.jordan_wigner(op)


def test_jordan_wigner_empty():
    """Test transforming an empty operator."""
    assert ffsim.jordan_wigner(ffsim.FermionOperator({})) == ffsim.QubitOperator({})
    assert ffsim.jordan_wigner(
        ffsim.FermionOperator({}), norb=3
    ) == ffsim.QubitOperator({})


def test_jordan_wigner_applies_tolerance_after_accumulating_terms():
    """Test tolerance is applied after contributions from all terms are summed."""
    op = ffsim.FermionOperator(
        {
            (ffsim.cre_a(0), ffsim.des_a(0)): 1.0,
            (ffsim.cre_a(1), ffsim.des_a(1)): 1.0,
        }
    )
    assert ffsim.jordan_wigner(op, norb=2, tol=0.75) == ffsim.QubitOperator(
        {frozenset(): 1.0}
    )
