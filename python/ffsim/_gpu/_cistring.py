# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cached device copies of CI string data."""

from __future__ import annotations

from functools import cache

import cupy  # type: ignore
import numpy as np

# The caches below are keyed on the current CUDA device as well as the shape of
# the data, because a CuPy array may only be used on the device it was allocated
# on. Without the device in the key, a multi-GPU process would hand device 0's
# pointers to a kernel launched on device 1.


@cache
def _occslst(device_id: int, norb: int, nocc: int) -> cupy.ndarray:
    from ffsim._cistring import gen_occslst

    return cupy.asarray(gen_occslst(range(norb), nocc).astype(np.uint64))


def occslst(norb: int, nocc: int) -> cupy.ndarray:
    """Device copy of the occupations list for a spin sector."""
    return _occslst(cupy.cuda.runtime.getDevice(), norb, nocc)


@cache
def _strings(device_id: int, norb: int, nocc: int) -> cupy.ndarray:
    from ffsim._cistring import make_strings

    return cupy.asarray(make_strings(range(norb), nocc).astype(np.int64))


def strings(norb: int, nocc: int) -> cupy.ndarray:
    """Device copy of the CI strings for a spin sector."""
    return _strings(cupy.cuda.runtime.getDevice(), norb, nocc)
