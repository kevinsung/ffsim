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


@cache
def occslst(norb: int, nocc: int) -> cupy.ndarray:
    """Device copy of the occupations list for a spin sector."""
    from ffsim._cistring import gen_occslst

    return cupy.asarray(gen_occslst(range(norb), nocc).astype(np.uint64))


@cache
def strings(norb: int, nocc: int) -> cupy.ndarray:
    """Device copy of the CI strings for a spin sector."""
    from ffsim._cistring import make_strings

    return cupy.asarray(make_strings(range(norb), nocc).astype(np.int64))
