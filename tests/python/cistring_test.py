# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for cistring functions implemented in Rust."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf.fci import cistring as pyscf_cistring

from ffsim._lib import (
    gen_linkstr_index,
    gen_linkstr_index_trilidx,
    gen_occslst,
    make_strings,
)

CASES = [(0, 0), (1, 0), (1, 1), (4, 2), (6, 3), (8, 4), (10, 5)]


@pytest.mark.parametrize("norb, nocc", CASES)
def test_make_strings_matches_pyscf(norb: int, nocc: int) -> None:
    result = make_strings(norb, nocc)
    expected = pyscf_cistring.make_strings(range(norb), nocc)
    assert result.dtype == np.dtype("int64"), f"dtype mismatch: {result.dtype}"
    assert np.array_equal(result, expected), (
        f"make_strings({norb}, {nocc}) mismatch\n"
        f"got:      {result}\nexpected: {expected}"
    )


@pytest.mark.parametrize("norb, nocc", CASES)
def test_gen_occslst_matches_pyscf(norb: int, nocc: int) -> None:
    result = gen_occslst(norb, nocc)
    expected = pyscf_cistring.gen_occslst(range(norb), nocc).astype(np.uint)
    assert result.dtype == np.dtype(np.uintp), f"dtype mismatch: {result.dtype}"
    assert np.array_equal(result, expected), (
        f"gen_occslst({norb}, {nocc}) mismatch\ngot:\n{result}\nexpected:\n{expected}"
    )


@pytest.mark.parametrize("norb, nocc", CASES)
def test_gen_linkstr_index_matches_pyscf(norb: int, nocc: int) -> None:
    result = gen_linkstr_index(norb, nocc)
    expected = pyscf_cistring.gen_linkstr_index(range(norb), nocc)
    assert result.dtype == np.dtype("int32"), f"dtype mismatch: {result.dtype}"
    assert np.array_equal(result, expected), (
        f"gen_linkstr_index({norb}, {nocc}) mismatch\n"
        f"got:\n{result}\nexpected:\n{expected}"
    )


@pytest.mark.parametrize("norb, nocc", CASES)
def test_gen_linkstr_index_trilidx_matches_pyscf(norb: int, nocc: int) -> None:
    result = gen_linkstr_index_trilidx(norb, nocc)
    # pyscf leaves garbage in column 1 for trilidx; zero it before comparing.
    expected = pyscf_cistring.gen_linkstr_index_trilidx(range(norb), nocc).copy()
    if expected.ndim == 3 and expected.shape[1] > 0:
        expected[:, :, 1] = 0
    assert result.dtype == np.dtype("int32"), f"dtype mismatch: {result.dtype}"
    assert np.array_equal(result, expected), (
        f"gen_linkstr_index_trilidx({norb}, {nocc}) mismatch\n"
        f"got:\n{result}\nexpected:\n{expected}"
    )


@pytest.mark.parametrize(
    "norb, nocc",
    [(0, 0), (4, 2), (6, 3)],
)
def test_caching_returns_same_buffer(norb: int, nocc: int) -> None:
    """Verify that repeated calls return the same underlying numpy buffer."""
    a = make_strings(norb, nocc)
    b = make_strings(norb, nocc)
    assert a.ctypes.data == b.ctypes.data, "make_strings: cache miss on second call"

    c = gen_occslst(norb, nocc)
    d = gen_occslst(norb, nocc)
    assert c.ctypes.data == d.ctypes.data, "gen_occslst: cache miss on second call"

    e = gen_linkstr_index(norb, nocc)
    f = gen_linkstr_index(norb, nocc)
    assert e.ctypes.data == f.ctypes.data, (
        "gen_linkstr_index: cache miss on second call"
    )

    g = gen_linkstr_index_trilidx(norb, nocc)
    h = gen_linkstr_index_trilidx(norb, nocc)
    assert g.ctypes.data == h.ctypes.data, (
        "gen_linkstr_index_trilidx: cache miss on second call"
    )


def test_arrays_are_read_only() -> None:
    """Verify that returned arrays are marked read-only."""
    arr = make_strings(4, 2)
    assert not arr.flags.writeable, "make_strings array should be read-only"

    arr = gen_occslst(4, 2)
    assert not arr.flags.writeable, "gen_occslst array should be read-only"

    arr = gen_linkstr_index(4, 2)
    assert not arr.flags.writeable, "gen_linkstr_index array should be read-only"

    arr = gen_linkstr_index_trilidx(4, 2)
    assert not arr.flags.writeable, (
        "gen_linkstr_index_trilidx array should be read-only"
    )
