# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for QubitOperator."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim
from ffsim import QubitOperator


def test_add():
    """Test adding QubitOperators."""
    op1 = QubitOperator({frozenset({ffsim.x(0), ffsim.z(2)}): 1.5})
    op2 = QubitOperator(
        {
            frozenset({ffsim.y(1), ffsim.z(3)}): 1,
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.5 + 1j,
        }
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.z(2)}): 1.5,
            frozenset({ffsim.y(1), ffsim.z(3)}): 1,
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.5 + 1j,
        }
    )
    assert op1 + op2 == expected

    op1 += op2
    assert op1 == expected


def test_add_like_terms():
    """Test that adding like terms combines their coefficients."""
    op1 = QubitOperator({frozenset({ffsim.x(0), ffsim.z(2)}): 1.5})
    op2 = QubitOperator({frozenset({ffsim.x(0), ffsim.z(2)}): 0.5})
    expected = QubitOperator({frozenset({ffsim.x(0), ffsim.z(2)}): 2.0})
    assert op1 + op2 == expected


def test_subtract():
    """Test subtracting QubitOperators."""
    op1 = QubitOperator({frozenset({ffsim.x(0), ffsim.z(2)}): 1.5})
    op2 = QubitOperator(
        {
            frozenset({ffsim.y(1), ffsim.z(3)}): 1,
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.5 + 1j,
        }
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.z(2)}): 1.5,
            frozenset({ffsim.y(1), ffsim.z(3)}): -1,
            frozenset({ffsim.x(0), ffsim.y(1)}): -1.5 - 1j,
        }
    )
    assert op1 - op2 == expected

    op1 -= op2
    assert op1 == expected


def test_neg():
    """Test negating QubitOperators."""
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.5,
            frozenset({ffsim.z(2)}): 1 + 1.5j,
        }
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): -1.5,
            frozenset({ffsim.z(2)}): -1 - 1.5j,
        }
    )
    assert -op == expected


def test_mul():
    """Test multiplying QubitOperators.

    Uses Pauli multiplication rules:
      X*X = I,  Y*Y = I,  Z*Z = I
      X*Y = iZ, Y*X = -iZ
      Y*Z = iX, Z*Y = -iX
      Z*X = iY, X*Z = -iY
    """
    # Single qubit: X * X = I
    op1 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    expected = QubitOperator({frozenset(): 1.0})
    assert op1 * op2 == expected

    # Single qubit: X * Y = iZ
    op1 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.y(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.z(0)}): 1j})
    assert op1 * op2 == expected

    # Single qubit: Y * X = -iZ
    op1 = QubitOperator({frozenset({ffsim.y(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.z(0)}): -1j})
    assert op1 * op2 == expected

    # Single qubit: Y * Z = iX
    op1 = QubitOperator({frozenset({ffsim.y(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.z(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.x(0)}): 1j})
    assert op1 * op2 == expected

    # Single qubit: Z * Y = -iX
    op1 = QubitOperator({frozenset({ffsim.z(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.y(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.x(0)}): -1j})
    assert op1 * op2 == expected

    # Single qubit: Z * X = iY
    op1 = QubitOperator({frozenset({ffsim.z(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.y(0)}): 1j})
    assert op1 * op2 == expected

    # Single qubit: X * Z = -iY
    op1 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.z(0)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.y(0)}): -1j})
    assert op1 * op2 == expected

    # Different qubits: Paulis simply tensor together
    op1 = QubitOperator({frozenset({ffsim.x(0)}): 1.0})
    op2 = QubitOperator({frozenset({ffsim.y(1)}): 1.0})
    expected = QubitOperator({frozenset({ffsim.x(0), ffsim.y(1)}): 1.0})
    assert op1 * op2 == expected

    # Multi-term multiplication: (X0 + Y1) * (Z0 + Z2)
    op1 = QubitOperator(
        {
            frozenset({ffsim.x(0)}): 0.5,
            frozenset({ffsim.y(1)}): 0.75,
        }
    )
    op2 = QubitOperator(
        {
            frozenset({ffsim.z(0)}): 1.0,
            frozenset({ffsim.z(2)}): 1.0,
        }
    )
    # X0 * Z0 = -iY0, X0 * Z2 = X0 Z2, Y1 * Z0 = Y1 Z0, Y1 * Z2 = -iX1 (wait, Y*Z=iX)
    # Y1 * Z2 = Y1 Z2 (different qubits)
    expected = QubitOperator(
        {
            frozenset({ffsim.y(0)}): 0.5 * (-1j),  # X0 * Z0 = -iY0
            frozenset({ffsim.x(0), ffsim.z(2)}): 0.5,  # X0 * Z2 (different qubits)
            frozenset({ffsim.y(1), ffsim.z(0)}): 0.75,  # Y1 * Z0 (different qubits)
            frozenset({ffsim.y(1), ffsim.z(2)}): 0.75,  # Y1 * Z2 (different qubits)
        }
    )
    assert op1 * op2 == expected


def test_mul_scalar():
    """Test multiplying by a scalar."""
    op = 1j * QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5j,
            frozenset({ffsim.z(2)}): 0.75j,
        }
    )
    assert op == expected

    op = 2 * QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1,
            frozenset({ffsim.z(2)}): 1.5,
        }
    )
    assert op == expected

    op *= 2
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 2,
            frozenset({ffsim.z(2)}): 3,
        }
    )
    assert op == expected


def test_div():
    """Test division."""
    op = (
        QubitOperator(
            {
                frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
                frozenset({ffsim.z(2)}): 0.75,
            }
        )
        / 2
    )
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.25,
            frozenset({ffsim.z(2)}): 0.375,
        }
    )
    assert op == expected

    op /= 2
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.125,
            frozenset({ffsim.z(2)}): 0.1875,
        }
    )
    assert op == expected


def test_pow():
    """Test exponentiation by an integer."""
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    assert op**0 == QubitOperator({frozenset(): 1})
    assert op**1 == op
    assert op**2 == op * op
    assert op**3 == op * op * op
    assert pow(op, 2) == op * op
    with pytest.raises(ValueError, match="mod argument"):
        _ = pow(op, 2, 2)  # type: ignore


def test_frozenset_keys_order_independence():
    """Test that frozenset keys treat different orderings as the same term."""
    op1 = QubitOperator({frozenset({ffsim.x(0), ffsim.y(1), ffsim.z(2)}): 1.5})
    op2 = QubitOperator({frozenset({ffsim.z(2), ffsim.x(0), ffsim.y(1)}): 1.5})
    assert op1 == op2


def test_get_set():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    assert op[frozenset({ffsim.x(0), ffsim.y(1)})] == 0.5

    op[frozenset({ffsim.x(0), ffsim.y(1)})] = 0.25
    assert op[frozenset({ffsim.x(0), ffsim.y(1)})] == 0.25


def test_del():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    assert op[frozenset({ffsim.x(0), ffsim.y(1)})] == 0.5

    del op[frozenset({ffsim.x(0), ffsim.y(1)})]
    assert frozenset({ffsim.x(0), ffsim.y(1)}) not in op


def test_len():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    assert len(op) == 2


def test_iter():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
            frozenset({ffsim.x(0), ffsim.z(2)}): 1.0,
        }
    )
    assert set(op) == {
        frozenset({ffsim.x(0), ffsim.y(1)}),
        frozenset({ffsim.z(2)}),
        frozenset({ffsim.x(0), ffsim.z(2)}),
    }


def test_many_body_order():
    op = QubitOperator({})
    assert op.many_body_order() == 0

    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
        }
    )
    assert op.many_body_order() == 2

    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1), ffsim.z(2), ffsim.x(3)}): 0.5,
            frozenset({ffsim.z(2)}): 0.75,
            frozenset({ffsim.x(0), ffsim.y(1), ffsim.z(2)}): 0.5,
        }
    )
    assert op.many_body_order() == 4


def test_adjoint():
    """Test adjoint method.

    Since Paulis are Hermitian (X†=X, Y†=Y, Z†=Z), the adjoint of a QubitOperator
    is obtained by taking the complex conjugate of all coefficients. The frozenset
    keys are unchanged.
    """
    op = QubitOperator(
        {
            frozenset(): 1 + 2j,
            frozenset({ffsim.x(0), ffsim.y(1)}): 1 + 1j,
            frozenset({ffsim.z(2)}): 2 - 3j,
        }
    )
    expected = QubitOperator(
        {
            frozenset(): 1 - 2j,
            frozenset({ffsim.x(0), ffsim.y(1)}): 1 - 1j,
            frozenset({ffsim.z(2)}): 2 + 3j,
        }
    )
    assert op.adjoint() == expected

    # Adjoint of adjoint gives back original
    assert op.adjoint().adjoint() == op

    # Empty operator
    op = QubitOperator({})
    assert op.adjoint() == op


def test_simplify():
    """Test simplify."""
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.0,
            frozenset({ffsim.z(2)}): 1e-9,
            frozenset({ffsim.x(3)}): -1e-7,
            frozenset({ffsim.y(4), ffsim.z(5)}): 0.5,
            frozenset({ffsim.x(0)}): 1e-10 + 2e-10j,
        }
    )

    # Test with default tolerance
    op_copy = op.copy()
    op_copy.simplify()
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.0,
            frozenset({ffsim.x(3)}): -1e-7,
            frozenset({ffsim.y(4), ffsim.z(5)}): 0.5,
        }
    )
    assert op_copy == expected

    # Test with custom tolerance
    op_copy = op.copy()
    op_copy.simplify(1e-6)
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.0,
            frozenset({ffsim.y(4), ffsim.z(5)}): 0.5,
        }
    )
    assert op_copy == expected

    # Test with small tolerance
    op_copy = op.copy()
    op_copy.simplify(tol=1e-12)
    expected = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.0,
            frozenset({ffsim.z(2)}): 1e-9,
            frozenset({ffsim.x(3)}): -1e-7,
            frozenset({ffsim.y(4), ffsim.z(5)}): 0.5,
            frozenset({ffsim.x(0)}): 1e-10 + 2e-10j,
        }
    )
    assert op_copy == expected

    # Test that original operator is unchanged
    original = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1.0,
            frozenset({ffsim.z(2)}): 1e-9,
            frozenset({ffsim.x(3)}): -1e-7,
            frozenset({ffsim.y(4), ffsim.z(5)}): 0.5,
            frozenset({ffsim.x(0)}): 1e-10 + 2e-10j,
        }
    )
    assert op == original

    # Test with empty operator
    empty_op = QubitOperator({})
    empty_op.simplify()
    assert empty_op == QubitOperator({})

    # Test with all small terms
    small_op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1e-9,
            frozenset({ffsim.z(2)}): 1e-10,
        }
    )
    small_op.simplify()
    assert small_op == QubitOperator({})

    # Check that it returns None
    assert op.simplify() is None  # type: ignore


def test_approx_eq():
    """Test approximate equality."""
    op1 = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5,
            frozenset({ffsim.z(2)}): -0.5,
        }
    )
    op2 = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 0.5 + 1e-7,
            frozenset({ffsim.z(2)}): -0.5 - 1e-7,
        }
    )
    assert ffsim.approx_eq(op1, op2)
    assert ffsim.approx_eq(op1, op2, rtol=0, atol=1e-7)
    assert not ffsim.approx_eq(op1, op2, rtol=0)


def test_repr_equivalent():
    """Test that repr evaluates to an equivalent object."""
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1,
            frozenset({ffsim.z(2)}): 0.5,
            frozenset({ffsim.x(0), ffsim.z(2)}): -0.5j,
            frozenset(): 1 - 0.5j,
        }
    )
    assert eval(repr(op)) == op


def test_copy():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1,
            frozenset({ffsim.z(2)}): 0.5,
        }
    )
    copy = op.copy()
    assert copy == op

    copy *= 2
    assert copy != op


def test_trace():
    """Test the _trace_ method.

    Qubit layout: qubits 0..norb-1 are alpha spin-orbitals, norb..2*norb-1 are beta.

    For a Pauli term:
    - Any X or Y factor gives zero trace.
    - An all-Z term contributes
        f(norb, n_alpha, s_alpha) * f(norb, n_beta, s_beta)
      where s_alpha / s_beta are the numbers of Z operators in each spin sector and
        f(N, n, s) = sum_{k=0}^{min(s,n)} (-1)^k * C(s,k) * C(N-s, n-k).
    """
    norb = 3
    nelec = (1, 2)
    dim = ffsim.dim(norb, nelec)  # C(3,1) * C(3,2) = 3 * 3 = 9

    # Identity: trace = dim(norb, nelec)
    op = QubitOperator({frozenset(): 1.0})
    assert ffsim.trace(op, norb=norb, nelec=nelec) == dim

    # X or Y term: trace = 0
    assert (
        ffsim.trace(
            QubitOperator({frozenset({ffsim.x(0)}): 1.0}), norb=norb, nelec=nelec
        )
        == 0
    )
    assert (
        ffsim.trace(
            QubitOperator({frozenset({ffsim.y(1)}): 1.0}), norb=norb, nelec=nelec
        )
        == 0
    )
    # X mixed with Z: trace = 0
    assert (
        ffsim.trace(
            QubitOperator({frozenset({ffsim.x(0), ffsim.z(1)}): 1.0}),
            norb=norb,
            nelec=nelec,
        )
        == 0
    )

    # Z_0 (alpha orbital 0, norb=3, nelec=(1,2)):
    #   f(3, 1, 1) = C(1,0)*C(2,1) - C(1,1)*C(2,0) = 2 - 1 = 1
    #   f(3, 2, 0) = C(3,2) = 3
    #   trace = 1 * 3 = 3
    np.testing.assert_allclose(
        ffsim.trace(
            QubitOperator({frozenset({ffsim.z(0)}): 1.0}), norb=norb, nelec=nelec
        ),
        3,
    )

    # Z_3 = Z on beta orbital 0 (qubit norb=3), nelec=(1,2):
    #   f(3, 1, 0) = C(3,1) = 3
    #   f(3, 2, 1) = C(1,0)*C(2,2) - C(1,1)*C(2,1) = 1 - 2 = -1
    #   trace = 3 * (-1) = -3
    np.testing.assert_allclose(
        ffsim.trace(
            QubitOperator({frozenset({ffsim.z(norb)}): 1.0}), norb=norb, nelec=nelec
        ),
        -3,
    )

    # Z_0 * Z_3 (one alpha Z, one beta Z):
    #   f(3, 1, 1) = 1,  f(3, 2, 1) = -1
    #   trace = 1 * (-1) = -1
    np.testing.assert_allclose(
        ffsim.trace(
            QubitOperator({frozenset({ffsim.z(0), ffsim.z(norb)}): 1.0}),
            norb=norb,
            nelec=nelec,
        ),
        -1,
    )

    # Qubit index out of the 2*norb system: contributes 0
    np.testing.assert_allclose(
        ffsim.trace(
            QubitOperator({frozenset({ffsim.z(2 * norb)}): 1.0}), norb=norb, nelec=nelec
        ),
        0,
    )

    # Linearity: trace of a sum equals sum of traces
    op = QubitOperator(
        {
            frozenset(): 0.5,
            frozenset({ffsim.z(0)}): 1.5,
            frozenset({ffsim.x(1)}): 2.0,
        }
    )
    expected = 0.5 * dim + 1.5 * 3 + 2.0 * 0
    np.testing.assert_allclose(ffsim.trace(op, norb=norb, nelec=nelec), expected)

    # Complex coefficients
    op = QubitOperator({frozenset(): 1 + 2j, frozenset({ffsim.z(0)}): 1j})
    expected = (1 + 2j) * dim + 1j * 3
    np.testing.assert_allclose(ffsim.trace(op, norb=norb, nelec=nelec), expected)

    # Verify against direct calculation: sum diagonal matrix elements in the Fock basis
    rng = np.random.default_rng(2024)
    norb2 = 2
    nelec2 = (1, 1)
    # Build a random operator with only Z terms (so direct sum is easy)
    z_coeffs = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    op = QubitOperator(
        {
            frozenset(): z_coeffs[0],
            frozenset({ffsim.z(0)}): z_coeffs[1],
            frozenset({ffsim.z(norb2)}): z_coeffs[2],
            frozenset({ffsim.z(0), ffsim.z(norb2)}): z_coeffs[3],
        }
    )
    # Fock states for norb=2, nelec=(1,1): alpha occupancies and beta occupancies
    # alpha: {0} or {1}; beta: {0} or {1}
    # qubit layout: [alpha0, alpha1, beta0, beta1]
    # states: (1,0,1,0), (1,0,0,1), (0,1,1,0), (0,1,0,1)
    fock_states = [(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)]
    direct_trace = sum(
        z_coeffs[0]  # identity
        + z_coeffs[1] * (-1) ** s[0]  # Z_0
        + z_coeffs[2] * (-1) ** s[2]  # Z_2 = beta orbital 0
        + z_coeffs[3] * (-1) ** s[0] * (-1) ** s[2]  # Z_0 * Z_2
        for s in fock_states
    )
    np.testing.assert_allclose(ffsim.trace(op, norb=norb2, nelec=nelec2), direct_trace)


def test_mapping_methods():
    op = QubitOperator(
        {
            frozenset({ffsim.x(0), ffsim.y(1)}): 1,
            frozenset({ffsim.z(2)}): 0.5,
            frozenset({ffsim.x(0), ffsim.z(2)}): -0.5j,
            frozenset(): 1 - 0.5j,
        }
    )
    assert op.keys() == {
        frozenset({ffsim.x(0), ffsim.y(1)}),
        frozenset({ffsim.z(2)}),
        frozenset({ffsim.x(0), ffsim.z(2)}),
        frozenset(),
    }
    assert set(op.values()) == {1, 0.5, -0.5j, 1 - 0.5j}
    assert op.items() == {
        (frozenset({ffsim.x(0), ffsim.y(1)}), 1),
        (frozenset({ffsim.z(2)}), 0.5),
        (frozenset({ffsim.x(0), ffsim.z(2)}), -0.5j),
        (frozenset(), 1 - 0.5j),
    }
