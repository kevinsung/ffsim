# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import cupy  # type: ignore


def apply_num_op_sum_evolution_in_place(
    vec: cupy.ndarray, phases: cupy.ndarray, occupations: cupy.ndarray
) -> None:
    """Apply time evolution by a sum of number operators in-place."""
    row_phases = cupy.prod(phases[occupations], axis=1)
    vec *= row_phases[:, None]
