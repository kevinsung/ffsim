# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QubitOperator class."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass

import numpy as np

from ffsim.operators.qubit_action import QubitAction

# Pauli multiplication table: (p1, p2) -> (result_pauli_or_None, phase)
# None means the result is the identity (qubit removed from the term).
_PAULI_MUL: dict[tuple[str, str], tuple[str | None, complex]] = {
    ("X", "X"): (None, 1),
    ("Y", "Y"): (None, 1),
    ("Z", "Z"): (None, 1),
    ("X", "Y"): ("Z", 1j),
    ("Y", "X"): ("Z", -1j),
    ("Y", "Z"): ("X", 1j),
    ("Z", "Y"): ("X", -1j),
    ("Z", "X"): ("Y", 1j),
    ("X", "Z"): ("Y", -1j),
}


def _mul_pauli_terms(
    term1: frozenset[QubitAction], term2: frozenset[QubitAction]
) -> tuple[frozenset[QubitAction], complex]:
    """Multiply two Pauli terms, returning the result term and accumulated phase."""
    qubit_to_pauli: dict[int, str] = {action.qubit: action.pauli for action in term1}
    phase: complex = 1
    for action in term2:
        qubit = action.qubit
        if qubit in qubit_to_pauli:
            result_pauli, p = _PAULI_MUL[(qubit_to_pauli[qubit], action.pauli)]
            phase *= p
            if result_pauli is None:
                del qubit_to_pauli[qubit]
            else:
                qubit_to_pauli[qubit] = result_pauli
        else:
            qubit_to_pauli[qubit] = action.pauli
    return frozenset(
        QubitAction(qubit=q, pauli=p) for q, p in qubit_to_pauli.items()
    ), phase


@dataclass
class QubitOperator(MutableMapping):
    """A qubit operator.

    A linear combination of tensor products of Pauli operators. Because Pauli
    operators on different qubits commute, the keys are frozensets of QubitAction
    values, each specifying a qubit index and a Pauli label ('X', 'Y', or 'Z').
    The identity term is represented by an empty frozenset.
    """

    coeffs: dict[frozenset[QubitAction], complex]

    def copy(self) -> QubitOperator:
        return QubitOperator(self.coeffs.copy())

    def __getitem__(self, key: frozenset[QubitAction]) -> complex:
        return self.coeffs[key]

    def __setitem__(self, key: frozenset[QubitAction], val: complex) -> None:
        self.coeffs[key] = val

    def __delitem__(self, key: frozenset[QubitAction]) -> None:
        del self.coeffs[key]

    def __iter__(self) -> Iterator[frozenset[QubitAction]]:
        return iter(self.coeffs)

    def __len__(self) -> int:
        return len(self.coeffs)

    def __iadd__(self, other) -> QubitOperator:
        if isinstance(other, QubitOperator):
            for term, coeff in other.coeffs.items():
                if term in self.coeffs:
                    self.coeffs[term] += coeff
                else:
                    self.coeffs[term] = coeff
            return self
        return NotImplemented

    def __add__(self, other) -> QubitOperator:
        result = self.copy()
        result += other
        return result

    def __isub__(self, other) -> QubitOperator:
        if isinstance(other, QubitOperator):
            for term, coeff in other.coeffs.items():
                if term in self.coeffs:
                    self.coeffs[term] -= coeff
                else:
                    self.coeffs[term] = -coeff
            return self
        return NotImplemented

    def __sub__(self, other) -> QubitOperator:
        result = self.copy()
        result -= other
        return result

    def __neg__(self) -> QubitOperator:
        result = self.copy()
        result *= -1
        return result

    def __imul__(self, other) -> QubitOperator:
        if isinstance(other, (int, float, complex)):
            for key in self.coeffs:
                self.coeffs[key] *= other
            return self
        return NotImplemented

    def __rmul__(self, other) -> QubitOperator:
        if isinstance(other, (int, float, complex)):
            result = self.copy()
            result *= other
            return result
        return NotImplemented

    def __mul__(self, other) -> QubitOperator:
        if isinstance(other, QubitOperator):
            new_coeffs: dict[frozenset[QubitAction], complex] = {}
            for term1, coeff1 in self.coeffs.items():
                for term2, coeff2 in other.coeffs.items():
                    new_term, phase = _mul_pauli_terms(term1, term2)
                    new_coeff = phase * coeff1 * coeff2
                    if new_term in new_coeffs:
                        new_coeffs[new_term] += new_coeff
                    else:
                        new_coeffs[new_term] = new_coeff
            return QubitOperator(new_coeffs)
        return NotImplemented

    def __itruediv__(self, other) -> QubitOperator:
        if isinstance(other, (int, float, complex)):
            for key in self.coeffs:
                self.coeffs[key] /= other
            return self
        return NotImplemented

    def __truediv__(self, other) -> QubitOperator:
        result = self.copy()
        result /= other
        return result

    def __pow__(self, exponent, modulo=None) -> QubitOperator:
        if isinstance(exponent, int):
            if modulo is not None:
                raise ValueError("mod argument not supported")
            result = QubitOperator({frozenset(): 1})
            for _ in range(exponent):
                result = result * self
            return result
        return NotImplemented

    def many_body_order(self) -> int:
        """Return the maximum number of Pauli factors in any term."""
        return max((len(term) for term in self.coeffs), default=0)

    def adjoint(self) -> QubitOperator:
        """Return the adjoint (Hermitian conjugate) of the operator."""
        return QubitOperator(
            {term: coeff.conjugate() for term, coeff in self.coeffs.items()}
        )

    def simplify(self, tol: float = 1e-8) -> None:
        """Remove terms whose coefficient magnitude is at or below tol."""
        keys_to_delete = [key for key, val in self.coeffs.items() if abs(val) <= tol]
        for key in keys_to_delete:
            del self.coeffs[key]

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, QubitOperator):
            for key in self.keys() | other.keys():
                if not np.isclose(
                    self.get(key, 0), other.get(key, 0), rtol=rtol, atol=atol
                ):
                    return False
            return True
        return NotImplemented

    def __repr__(self) -> str:
        def _term_repr(term: frozenset[QubitAction]) -> str:
            # Repr as plain tuples so eval() works without QubitAction in scope.
            # QubitAction is a NamedTuple, so QubitAction(q, p) == (q, p) and they
            # share the same hash, meaning the reconstructed operator compares equal.
            sorted_actions = sorted(term)
            if not sorted_actions:
                return "frozenset()"
            actions_str = ", ".join(repr(tuple(a)) for a in sorted_actions)
            return f"frozenset({{{actions_str}}})"

        items_str = ", ".join(
            f"{_term_repr(term)}: {coeff!r}" for term, coeff in self.coeffs.items()
        )
        return f"QubitOperator({{{items_str}}})"
