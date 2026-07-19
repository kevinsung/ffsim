# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""GPU (CUDA) implementations of state vector simulation kernels.

This subpackage provides CuPy-based implementations of the gate application
kernels from ``ffsim._lib``. CuPy is an optional dependency; nothing in this
module's top level imports it, so importing ffsim does not require CuPy.
"""

from __future__ import annotations

from typing import Any


def is_gpu_array(vec: Any) -> bool:
    """Return whether an array is a CuPy array, without importing CuPy."""
    return type(vec).__module__.partition(".")[0] == "cupy"
