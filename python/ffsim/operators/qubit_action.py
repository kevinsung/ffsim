# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The QubitAction NamedTuple and construction functions."""

from typing import NamedTuple


class QubitAction(NamedTuple):
    """A Pauli action on a qubit."""

    pauli: str  # one of 'X', 'Y', 'Z'
    qubit: int  # index of the qubit to act on


def x(qubit: int) -> QubitAction:
    """Return a Pauli X action on the given qubit.

    Args:
        qubit: The index of the qubit to act on.
    """
    return QubitAction(pauli="X", qubit=qubit)


def y(qubit: int) -> QubitAction:
    """Return a Pauli Y action on the given qubit.

    Args:
        qubit: The index of the qubit to act on.
    """
    return QubitAction(pauli="Y", qubit=qubit)


def z(qubit: int) -> QubitAction:
    """Return a Pauli Z action on the given qubit.

    Args:
        qubit: The index of the qubit to act on.
    """
    return QubitAction(pauli="Z", qubit=qubit)
