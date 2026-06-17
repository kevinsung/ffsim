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

import math

import numpy as np
import pytest
from pyscf.fci import cistring as pyscf_cistring

from ffsim._lib import (
    addr_from_occupied,
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
    nwords = max(1, math.ceil(norb / 64))
    assert result.dtype == np.dtype("uint64"), f"dtype mismatch: {result.dtype}"
    assert result.ndim == 2, f"expected 2-D array, got shape {result.shape}"
    assert result.shape[1] == nwords, f"expected {nwords} words, got {result.shape[1]}"
    # For norb <= 64, word 0 of each row must equal pyscf's i64 value reinterpreted as
    # u64.
    assert np.array_equal(result[:, 0], expected.astype(np.uint64)), (
        f"make_strings({norb}, {nocc}) word-0 mismatch\n"
        f"got:      {result[:, 0]}\nexpected: {expected.astype(np.uint64)}"
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


# ---------------------------------------------------------------------------
# norb > 64 smoke tests
# ---------------------------------------------------------------------------

LARGE_NORB_CASES = [(65, 2), (80, 3), (128, 2)]


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_make_strings_large_norb_shape(norb: int, nocc: int) -> None:
    nwords = math.ceil(norb / 64)
    result = make_strings(norb, nocc)
    assert result.dtype == np.dtype("uint64")
    assert result.ndim == 2
    assert result.shape == (math.comb(norb, nocc), nwords)


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_gen_occslst_large_norb(norb: int, nocc: int) -> None:
    result = gen_occslst(norb, nocc)
    assert result.shape == (math.comb(norb, nocc), nocc)
    # All orbital indices must be in [0, norb-1].
    if result.size > 0:
        assert int(result.min()) >= 0
        assert int(result.max()) < norb


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_gen_linkstr_index_large_norb(norb: int, nocc: int) -> None:
    n = math.comb(norb, nocc)
    nvir = norb - nocc
    nlink = nocc + nocc * nvir
    result = gen_linkstr_index(norb, nocc)
    assert result.shape == (n, nlink, 4)
    # All addresses must be in [0, n-1].
    addrs = result[:, :, 2]
    assert int(addrs.min()) >= 0
    assert int(addrs.max()) < n


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_addr_from_occupied_large_norb(norb: int, nocc: int) -> None:
    strings = make_strings(norb, nocc)
    occslst = gen_occslst(norb, nocc)
    n = math.comb(norb, nocc)
    for i in range(min(n, 20)):  # spot-check first 20 strings
        occ = occslst[i].tolist()
        addr = addr_from_occupied(norb, nocc, occ)
        assert addr == i, f"addr_from_occupied mismatch at index {i}: got {addr}"
    _ = strings  # ensure make_strings is exercised


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_make_strings_consistency_with_occslst(norb: int, nocc: int) -> None:
    """Check that word 0 of each string encodes the correct low-64 occupations."""
    strings = make_strings(norb, nocc)
    occslst = gen_occslst(norb, nocc)
    nwords = strings.shape[1]
    for i in range(min(math.comb(norb, nocc), 20)):
        # Reconstruct the expected words from the occupation list.
        expected_words = [0] * nwords
        for o in occslst[i]:
            expected_words[o >> 6] |= 1 << (o & 63)
        for w in range(nwords):
            assert strings[i, w] == expected_words[w], (
                f"string {i} word {w}: got {strings[i, w]:#x}, "
                f"expected {expected_words[w]:#x}"
            )


@pytest.mark.parametrize("norb, nocc", LARGE_NORB_CASES)
def test_addr_from_occupied_roundtrip(norb: int, nocc: int) -> None:
    """addr_from_occupied must round-trip through occslst for all checked strings."""
    occslst = gen_occslst(norb, nocc)
    n = math.comb(norb, nocc)
    # Check last 10 strings as well (catches boundary issues).
    indices = list(range(min(n, 10))) + list(range(max(0, n - 10), n))
    for i in indices:
        occ = occslst[i].tolist()
        addr = addr_from_occupied(norb, nocc, occ)
        assert addr == i, (
            f"addr_from_occupied roundtrip failed at index {i}: got {addr}"
        )
