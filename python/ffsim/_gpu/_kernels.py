# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helpers for launching raw CUDA kernels."""

from __future__ import annotations

from functools import cache
from typing import Any

import cupy  # type: ignore

BLOCK_SIZE = 256


@cache
def _get_function(device_id: int, module: Any, name: str) -> Any:
    """Look up a kernel, once per device.

    A ``RawModule`` compiles separately for each device, so the looked up
    function may only be launched on the device that was current when it was
    retrieved.
    """
    return module.get_function(name)


def launch(module: Any, name: str, size: int, args: tuple) -> None:
    """Launch a kernel with one thread per element.

    Args:
        module: The module containing the kernel.
        name: The name of the kernel.
        size: The number of elements to process.
        args: The kernel arguments.
    """
    if not size:
        return
    kernel = _get_function(cupy.cuda.runtime.getDevice(), module, name)
    n_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    kernel((n_blocks,), (BLOCK_SIZE,), args)
