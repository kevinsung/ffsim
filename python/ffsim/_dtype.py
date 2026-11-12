# This code is a Qiskit project.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Data type handling for state vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import DTypeLike

_SUPPORTED_STATE_DTYPES = (np.dtype(np.complex64), np.dtype(np.complex128))


def complex_state_dtype(dtype: DTypeLike) -> np.dtype:
    """Return the complex dtype that gates use for a vector of the given dtype.

    Single-precision inputs (float16, float32, complex64) map to complex64.
    Double-precision floats, integers, and booleans map to complex128.
    Extended-precision and non-numeric dtypes are not supported.

    Raises:
        TypeError: The dtype is not supported.
    """
    dtype = np.dtype(dtype)
    if dtype.kind in "biu":
        return np.dtype(np.complex128)
    if dtype.kind in "fc":
        target = np.result_type(dtype, np.complex64)
        if target in _SUPPORTED_STATE_DTYPES:
            return target
    raise TypeError(
        f"Unsupported state vector dtype: {dtype}. "
        "Supported dtypes are complex64 and complex128. "
        "Real inputs are promoted to the complex dtype of the same precision."
    )


def as_complex_state_vector(vec: np.ndarray, *, copy: bool) -> np.ndarray:
    """Promote a state vector to a supported complex dtype.

    If ``copy`` is True, the returned array never shares memory with ``vec``.
    If ``copy`` is False, ``vec`` itself is returned when it already has a
    supported dtype.
    """
    vec = np.asarray(vec)
    return vec.astype(complex_state_dtype(vec.dtype), copy=copy)


def as_double_precision(vec: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """Cast a vector to double precision, keeping it real or complex.

    Used before calling PySCF FCI kernels, which only handle float64 data.

    If ``copy`` is True, the returned array never shares memory with ``vec``.
    If ``copy`` is False, ``vec`` itself is returned when it is already in double
    precision.
    """
    vec = np.asarray(vec)
    target = np.complex128 if np.iscomplexobj(vec) else np.float64
    return vec.astype(target, copy=copy)
