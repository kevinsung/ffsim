// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::{Array2, Array3};
use num_integer::binomial;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyArrayMethods};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Module-level caches — keyed on (norb, nocc).
// On a cache hit the same numpy buffer is returned (refcount bump, no copy).
// All returned arrays are marked read-only.
// ---------------------------------------------------------------------------

static MAKE_STRINGS_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray2<u64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_OCCSLST_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray2<usize>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_LINKSTR_INDEX_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray3<i32>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_LINKSTR_INDEX_TRILIDX_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray3<i32>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// BitString / BitStringMut — zero-copy views over a &[u64] word slice.
//
// Each string of `norb` orbitals is stored as `nwords(norb)` u64 words.
// Bit `o` lives in word `o / 64` at bit position `o % 64` (LSB = 0).
// ---------------------------------------------------------------------------

/// Number of u64 words needed to store `norb` orbital bits.
#[inline]
pub fn nwords(norb: usize) -> usize {
    norb.div_ceil(64)
}

/// Immutable view over a single CI string stored as packed u64 words.
#[derive(Clone, Copy)]
pub struct BitString<'a> {
    pub words: &'a [u64],
}

impl<'a> BitString<'a> {
    #[inline]
    pub fn contains(&self, o: usize) -> bool {
        (self.words[o >> 6] >> (o & 63)) & 1 == 1
    }

    /// Number of set bits strictly between positions `lo` and `hi` (exclusive
    /// on both ends).  Used to compute the fermionic sign of a single excitation.
    pub fn popcount_between(&self, lo: usize, hi: usize) -> u32 {
        if hi <= lo + 1 {
            return 0;
        }
        let first = lo + 1; // inclusive
        let last = hi - 1; // inclusive
        let w_first = first >> 6;
        let b_first = first & 63;
        let w_last = last >> 6;
        let b_last = last & 63;
        if w_first == w_last {
            let mask = if b_last == 63 {
                !0u64 << b_first
            } else {
                ((2u64 << b_last) - 1) & (!0u64 << b_first)
            };
            (self.words[w_first] & mask).count_ones()
        } else {
            // First partial word.
            let mut count = (self.words[w_first] >> b_first).count_ones();
            // Full middle words.
            for w in (w_first + 1)..w_last {
                count += self.words[w].count_ones();
            }
            // Last partial word.
            let mask_last = if b_last == 63 {
                !0u64
            } else {
                (2u64 << b_last) - 1
            };
            count += (self.words[w_last] & mask_last).count_ones();
            count
        }
    }
}

/// Mutable view over a single CI string stored as packed u64 words.
pub struct BitStringMut<'a> {
    pub words: &'a mut [u64],
}

impl<'a> BitStringMut<'a> {
    #[inline]
    #[allow(dead_code)]
    pub fn set(&mut self, o: usize) {
        self.words[o >> 6] |= 1u64 << (o & 63);
    }

    #[inline]
    pub fn toggle(&mut self, o: usize) {
        self.words[o >> 6] ^= 1u64 << (o & 63);
    }

    /// Apply single excitation a† i |self⟩ in place (toggle bits i and a).
    #[inline]
    pub fn excite(&mut self, i: usize, a: usize) {
        self.toggle(i);
        self.toggle(a);
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust core algorithms (no GIL required)
// ---------------------------------------------------------------------------

/// Return the combinadic rank of `string` within C(norb, nocc).
///
/// Matches pyscf's `_str2addr`: scans set bits MSB-to-LSB; for each set bit
/// at position `o`, adds C(o, k) then decrements k.
fn combinadic_addr(norb: usize, nocc: usize, string: BitString<'_>) -> i32 {
    let mut addr: i64 = 0;
    let mut k = nocc as i64;
    for o in (0..norb).rev() {
        if string.contains(o) {
            addr += binomial(o as i64, k);
            k -= 1;
        }
    }
    addr as i32
}

/// Recursive scratch-based CI-string generator in pyscf row order.
///
/// `out` accumulates complete strings; `scratch` holds bits from higher
/// orbitals accumulated so far; `orbs` is the number of remaining orbitals
/// (0..orbs-1); `nelec` is the number of electrons still to place.
///
/// Row order: strings without orbital `orbs-1` first, then strings with it.
fn gen_strings_scratch(out: &mut Vec<u64>, scratch: &mut [u64], orbs: usize, nelec: usize) {
    if nelec == 0 {
        out.extend_from_slice(scratch);
        return;
    }
    if nelec == orbs {
        for o in 0..orbs {
            scratch[o >> 6] |= 1u64 << (o & 63);
        }
        out.extend_from_slice(scratch);
        for o in 0..orbs {
            scratch[o >> 6] &= !(1u64 << (o & 63));
        }
        return;
    }
    // Without orbital `orbs-1`.
    gen_strings_scratch(out, scratch, orbs - 1, nelec);
    // With orbital `orbs-1`.
    scratch[(orbs - 1) >> 6] |= 1u64 << ((orbs - 1) & 63);
    gen_strings_scratch(out, scratch, orbs - 1, nelec - 1);
    scratch[(orbs - 1) >> 6] &= !(1u64 << ((orbs - 1) & 63));
}

/// Build the list of FCI occupation strings for `nocc` electrons in `norb`
/// orbitals.
///
/// Returns a flat Vec<u64> of length `n_strings * nwords(norb)`.
/// Row order matches pyscf's `cistring.make_strings(range(norb), nocc)`.
fn make_strings_core(norb: usize, nocc: usize) -> Vec<u64> {
    if nocc > norb {
        return vec![];
    }
    let nw = nwords(norb).max(1); // at least 1 word even for norb==0
    let n = binomial(norb, nocc) as usize;
    let mut result = Vec::with_capacity(n * nw);
    let mut scratch = vec![0u64; nw];
    gen_strings_scratch(&mut result, &mut scratch, norb, nocc);
    result
}

/// Build the occupation-list table: each row lists the sorted orbital indices
/// occupied in the corresponding FCI string.  Row order matches
/// `make_strings_core`.
///
/// Output shape: `(C(norb, nocc), nocc)`, dtype `usize`.
fn gen_occslst_core(norb: usize, nocc: usize) -> Array2<usize> {
    if nocc > norb {
        return Array2::zeros((0, nocc));
    }
    let n = binomial(norb, nocc) as usize;
    let nw = nwords(norb).max(1);
    let mut result = Array2::<usize>::zeros((n, nocc));
    let strings = make_strings_core(norb, nocc);
    for (row_idx, chunk) in strings.chunks_exact(nw).enumerate() {
        let string = BitString { words: chunk };
        let mut col = 0;
        for o in 0..norb {
            if string.contains(o) {
                result[[row_idx, col]] = o;
                col += 1;
            }
        }
    }
    result
}

/// Build the link-string index table for single excitations.
///
/// For each FCI string `str0`, the table records all single excitations
/// `a^+ i |str0>`:
/// - First `nocc` rows per string: diagonal entries `(i, i, addr(str0), +1)`
///   for each occupied orbital `i`.
/// - Next `nocc * nvir` rows: `(a, i, addr(str1), sign)` for every occupied
///   `i` (outer loop) and virtual `a` (inner loop).
///
/// If `trilidx` is true, column 0 contains the triangular index
/// `max(a,i)*(max(a,i)+1)/2 + min(a,i)` instead of `a`, and column 1 is zero.
///
/// Output shape: `(C(norb,nocc), nocc + nocc*nvir, 4)`, dtype `i32`.
fn gen_linkstr_index_core(norb: usize, nocc: usize, trilidx: bool) -> Array3<i32> {
    let nvir = norb - nocc.min(norb);
    let nlink = nocc + nocc * nvir;
    let nw = nwords(norb).max(1);
    let strings = make_strings_core(norb, nocc);
    let na = strings.len() / nw;
    let mut result = Array3::<i32>::zeros((na, nlink, 4));

    for (str_idx, chunk) in strings.chunks_exact(nw).enumerate() {
        let str0 = BitString { words: chunk };
        let occupied: Vec<usize> = (0..norb).filter(|&o| str0.contains(o)).collect();
        let virtual_orbs: Vec<usize> = (0..norb).filter(|&o| !str0.contains(o)).collect();

        let addr0 = combinadic_addr(norb, nocc, str0);
        for (k, &i) in occupied.iter().enumerate() {
            if trilidx {
                result[[str_idx, k, 0]] = (i * (i + 1) / 2 + i) as i32;
                result[[str_idx, k, 1]] = 0;
            } else {
                result[[str_idx, k, 0]] = i as i32;
                result[[str_idx, k, 1]] = i as i32;
            }
            result[[str_idx, k, 2]] = addr0;
            result[[str_idx, k, 3]] = 1;
        }

        let mut link_idx = nocc;
        for &i in &occupied {
            for &a in &virtual_orbs {
                let mut str1_words = chunk.to_vec();
                BitStringMut {
                    words: &mut str1_words,
                }
                .excite(i, a);
                let addr1 =
                    combinadic_addr(norb, nocc, BitString { words: &str1_words });

                let lo = i.min(a);
                let hi = i.max(a);
                let sign = if str0.popcount_between(lo, hi) % 2 == 0 {
                    1i32
                } else {
                    -1
                };

                if trilidx {
                    result[[str_idx, link_idx, 0]] = (hi * (hi + 1) / 2 + lo) as i32;
                    result[[str_idx, link_idx, 1]] = 0;
                } else {
                    result[[str_idx, link_idx, 0]] = a as i32;
                    result[[str_idx, link_idx, 1]] = i as i32;
                }
                result[[str_idx, link_idx, 2]] = addr1;
                result[[str_idx, link_idx, 3]] = sign;
                link_idx += 1;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Python-exposed functions with internal caches
// ---------------------------------------------------------------------------

/// Return FCI occupation strings for `nocc` electrons in `norb` orbitals.
///
/// Returns a read-only uint64 array of shape ``(C(norb, nocc), nwords)``
/// where ``nwords = ceil(norb / 64)`` (minimum 1).
/// Subsequent calls with the same arguments return the same buffer.
#[pyfunction]
pub fn make_strings(py: Python<'_>, norb: usize, nocc: usize) -> Py<PyArray2<u64>> {
    {
        let cache = MAKE_STRINGS_CACHE.lock().unwrap();
        if let Some(arr) = cache.get(&(norb, nocc)) {
            return arr.clone_ref(py);
        }
    }
    let nw = nwords(norb).max(1);
    let raw = py.detach(|| make_strings_core(norb, nocc));
    let n = if raw.is_empty() { 0 } else { raw.len() / nw };
    let data = ndarray::Array2::from_shape_vec((n, nw), raw)
        .expect("shape mismatch in make_strings");
    let arr = data.into_pyarray(py);
    arr.readwrite().make_nonwriteable();
    let py_arr = arr.unbind();
    let mut cache = MAKE_STRINGS_CACHE.lock().unwrap();
    cache
        .entry((norb, nocc))
        .or_insert_with(|| py_arr.clone_ref(py));
    cache[&(norb, nocc)].clone_ref(py)
}

/// Return the occupation-list table for `nocc` electrons in `norb` orbitals.
///
/// Returns a read-only usize array of shape ``(C(norb, nocc), nocc)``.
/// Subsequent calls with the same arguments return the same buffer.
#[pyfunction]
pub fn gen_occslst(py: Python<'_>, norb: usize, nocc: usize) -> Py<PyArray2<usize>> {
    {
        let cache = GEN_OCCSLST_CACHE.lock().unwrap();
        if let Some(arr) = cache.get(&(norb, nocc)) {
            return arr.clone_ref(py);
        }
    }
    let data = py.detach(|| gen_occslst_core(norb, nocc));
    let arr = data.into_pyarray(py);
    arr.readwrite().make_nonwriteable();
    let py_arr = arr.unbind();
    let mut cache = GEN_OCCSLST_CACHE.lock().unwrap();
    cache
        .entry((norb, nocc))
        .or_insert_with(|| py_arr.clone_ref(py));
    cache[&(norb, nocc)].clone_ref(py)
}

/// Return the link-string index table for single excitations.
///
/// Returns a read-only i32 array of shape
/// ``(C(norb, nocc), nocc + nocc*(norb-nocc), 4)``.
/// Subsequent calls with the same arguments return the same buffer.
#[pyfunction]
pub fn gen_linkstr_index(py: Python<'_>, norb: usize, nocc: usize) -> Py<PyArray3<i32>> {
    {
        let cache = GEN_LINKSTR_INDEX_CACHE.lock().unwrap();
        if let Some(arr) = cache.get(&(norb, nocc)) {
            return arr.clone_ref(py);
        }
    }
    let data = py.detach(|| gen_linkstr_index_core(norb, nocc, false));
    let arr = data.into_pyarray(py);
    arr.readwrite().make_nonwriteable();
    let py_arr = arr.unbind();
    let mut cache = GEN_LINKSTR_INDEX_CACHE.lock().unwrap();
    cache
        .entry((norb, nocc))
        .or_insert_with(|| py_arr.clone_ref(py));
    cache[&(norb, nocc)].clone_ref(py)
}

/// Return the triangular-index variant of the link-string index table.
///
/// Column 0 contains ``max(a,i)*(max(a,i)+1)/2 + min(a,i)``; column 1 is
/// zero.  All other fields match :func:`gen_linkstr_index`.
/// Subsequent calls with the same arguments return the same buffer.
#[pyfunction]
pub fn gen_linkstr_index_trilidx(py: Python<'_>, norb: usize, nocc: usize) -> Py<PyArray3<i32>> {
    {
        let cache = GEN_LINKSTR_INDEX_TRILIDX_CACHE.lock().unwrap();
        if let Some(arr) = cache.get(&(norb, nocc)) {
            return arr.clone_ref(py);
        }
    }
    let data = py.detach(|| gen_linkstr_index_core(norb, nocc, true));
    let arr = data.into_pyarray(py);
    arr.readwrite().make_nonwriteable();
    let py_arr = arr.unbind();
    let mut cache = GEN_LINKSTR_INDEX_TRILIDX_CACHE.lock().unwrap();
    cache
        .entry((norb, nocc))
        .or_insert_with(|| py_arr.clone_ref(py));
    cache[&(norb, nocc)].clone_ref(py)
}

/// Return the combinadic rank of an occupation list within C(norb, nocc).
///
/// `occupied` must be a sorted list of orbital indices (ascending).
/// Equivalent to the internal `combinadic_addr` but takes an occupation list
/// directly, avoiding the need to build a bitmask.
#[pyfunction]
pub fn addr_from_occupied(norb: usize, nocc: usize, occupied: Vec<usize>) -> i32 {
    let nw = nwords(norb).max(1);
    let mut words = vec![0u64; nw];
    for &o in &occupied {
        words[o >> 6] |= 1u64 << (o & 63);
    }
    combinadic_addr(norb, nocc, BitString { words: &words })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_to_scalar(chunk: &[u64]) -> u64 {
        chunk[0]
    }

    #[test]
    fn test_make_strings_edge_cases() {
        let nw = nwords(0).max(1);
        let s = make_strings_core(0, 0);
        assert_eq!(s.len(), nw); // 1 string of all-zero words
        assert!(s.iter().all(|&w| w == 0));

        let nw4 = nwords(4).max(1);
        let s = make_strings_core(4, 0);
        assert_eq!(s.len(), nw4);
        assert!(s.iter().all(|&w| w == 0));

        let s = make_strings_core(4, 4);
        assert_eq!(flat_to_scalar(&s[..nw4]), 15u64);

        let s = make_strings_core(4, 5);
        assert!(s.is_empty());
    }

    #[test]
    fn test_make_strings_norb4_nocc2() {
        // pyscf: array([3, 5, 6, 9, 10, 12])
        let nw = nwords(4).max(1);
        let raw = make_strings_core(4, 2);
        let scalars: Vec<u64> = raw.chunks_exact(nw).map(|c| c[0]).collect();
        assert_eq!(scalars, vec![3u64, 5, 6, 9, 10, 12]);
    }

    #[test]
    fn test_gen_occslst_norb4_nocc2() {
        let tbl = gen_occslst_core(4, 2);
        assert_eq!(tbl.shape(), &[6, 2]);
        let expected: &[[usize; 2]] = &[[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]];
        for (r, exp) in expected.iter().enumerate() {
            assert_eq!(tbl[[r, 0]], exp[0]);
            assert_eq!(tbl[[r, 1]], exp[1]);
        }
    }

    #[test]
    fn test_gen_occslst_edge_cases() {
        let tbl = gen_occslst_core(4, 0);
        assert_eq!(tbl.shape(), &[1, 0]);
        let tbl = gen_occslst_core(4, 5);
        assert_eq!(tbl.shape(), &[0, 5]);
    }

    #[test]
    fn test_gen_linkstr_index_norb4_nocc2() {
        let tbl = gen_linkstr_index_core(4, 2, false);
        assert_eq!(tbl.shape(), &[6, 6, 4]);
        // pyscf row0: [[0,0,0,1],[1,1,0,1],[2,0,2,-1],[3,0,4,-1],[2,1,1,1],[3,1,3,1]]
        let row0_expected: &[[i32; 4]] = &[
            [0, 0, 0, 1],
            [1, 1, 0, 1],
            [2, 0, 2, -1],
            [3, 0, 4, -1],
            [2, 1, 1, 1],
            [3, 1, 3, 1],
        ];
        for (k, exp) in row0_expected.iter().enumerate() {
            assert_eq!(
                [
                    tbl[[0, k, 0]],
                    tbl[[0, k, 1]],
                    tbl[[0, k, 2]],
                    tbl[[0, k, 3]]
                ],
                *exp,
                "row0 link {k} mismatch"
            );
        }
    }

    #[test]
    fn test_gen_linkstr_index_trilidx_norb4_nocc2() {
        let tbl = gen_linkstr_index_core(4, 2, true);
        assert_eq!(tbl.shape(), &[6, 6, 4]);
        let row0_expected: &[[i32; 4]] = &[
            [0, 0, 0, 1],
            [2, 0, 0, 1],
            [3, 0, 2, -1],
            [6, 0, 4, -1],
            [4, 0, 1, 1],
            [7, 0, 3, 1],
        ];
        for (k, exp) in row0_expected.iter().enumerate() {
            assert_eq!(
                [
                    tbl[[0, k, 0]],
                    tbl[[0, k, 1]],
                    tbl[[0, k, 2]],
                    tbl[[0, k, 3]]
                ],
                *exp,
                "trilidx row0 link {k} mismatch"
            );
        }
    }

    #[test]
    fn test_gen_linkstr_index_nocc0() {
        let tbl = gen_linkstr_index_core(4, 0, false);
        assert_eq!(tbl.shape(), &[1, 0, 4]);
    }

    #[test]
    fn test_gen_linkstr_index_nocc_eq_norb() {
        let tbl = gen_linkstr_index_core(4, 4, false);
        assert_eq!(tbl.shape(), &[1, 4, 4]);
    }

    #[test]
    fn test_combinadic_addr_norb4_nocc2() {
        let nw = nwords(4).max(1);
        let raw = make_strings_core(4, 2);
        for (expected_addr, chunk) in raw.chunks_exact(nw).enumerate() {
            let s = BitString { words: chunk };
            assert_eq!(
                combinadic_addr(4, 2, s),
                expected_addr as i32,
                "addr mismatch for string {:#b}",
                chunk[0]
            );
        }
    }

    #[test]
    fn test_addr_from_occupied() {
        let occ_lists: &[&[usize]] = &[
            &[0, 1],
            &[0, 2],
            &[1, 2],
            &[0, 3],
            &[1, 3],
            &[2, 3],
        ];
        for (expected, &occ) in occ_lists.iter().enumerate() {
            assert_eq!(
                addr_from_occupied(4, 2, occ.to_vec()),
                expected as i32,
                "addr_from_occupied mismatch for {:?}",
                occ
            );
        }
    }

    #[test]
    fn test_popcount_between_single_word() {
        // string = 0b1011010 (bits 1,3,4,6 set)
        let words = [0b1011010u64];
        let s = BitString { words: &words };
        // strictly between 0 and 6 → bits 1,3,4 → 3
        assert_eq!(s.popcount_between(0, 6), 3);
        // strictly between 1 and 4 → bit 3 → 1
        assert_eq!(s.popcount_between(1, 4), 1);
        // strictly between 4 and 6 → no bits → 0
        assert_eq!(s.popcount_between(4, 6), 0);
        // adjacent → 0
        assert_eq!(s.popcount_between(1, 2), 0);
    }

    #[test]
    fn test_popcount_between_multi_word() {
        // word0 = all ones, word1 = 0
        let words = [!0u64, 0u64];
        let s = BitString { words: &words };
        // strictly between 0 and 65 → bits 1..64 in word0 → 63
        assert_eq!(s.popcount_between(0, 65), 63);
        // strictly between 62 and 66 → bit 63 → 1
        assert_eq!(s.popcount_between(62, 66), 1);
        // strictly between 63 and 65 → no bits → 0
        assert_eq!(s.popcount_between(63, 65), 0);
    }

    #[test]
    fn test_make_strings_norb65_nocc2() {
        let norb = 65;
        let nocc = 2;
        let nw = nwords(norb).max(1);
        assert_eq!(nw, 2);
        let raw = make_strings_core(norb, nocc);
        let expected_n = norb * (norb - 1) / 2;
        assert_eq!(raw.len(), expected_n * nw);
        // First string: orbitals 0 and 1 → word0 = 0b11, word1 = 0
        assert_eq!(raw[0], 0b11u64);
        assert_eq!(raw[1], 0u64);
        // Last string: orbitals 63 and 64 → word0 has bit 63, word1 has bit 0
        let last = (expected_n - 1) * nw;
        assert_eq!(raw[last], 1u64 << 63);
        assert_eq!(raw[last + 1], 1u64);
    }

    #[test]
    fn test_gen_occslst_norb65_nocc2() {
        let norb = 65;
        let nocc = 2;
        let n = norb * (norb - 1) / 2;
        let tbl = gen_occslst_core(norb, nocc);
        assert_eq!(tbl.shape(), &[n, 2]);
        assert_eq!(tbl[[0, 0]], 0);
        assert_eq!(tbl[[0, 1]], 1);
        assert_eq!(tbl[[n - 1, 0]], 63);
        assert_eq!(tbl[[n - 1, 1]], 64);
    }

    #[test]
    fn test_gen_linkstr_index_norb65_nocc2() {
        let norb = 65;
        let nocc = 2;
        let n = norb * (norb - 1) / 2;
        let nvir = norb - nocc;
        let nlink = nocc + nocc * nvir;
        let tbl = gen_linkstr_index_core(norb, nocc, false);
        assert_eq!(tbl.shape(), &[n, nlink, 4]);
        // All addresses in [0, n-1].
        for i in 0..n {
            for k in 0..nlink {
                let addr = tbl[[i, k, 2]];
                assert!(
                    addr >= 0 && (addr as usize) < n,
                    "addr {addr} out of bounds at [{i},{k}]"
                );
            }
        }
    }

    #[test]
    fn test_combinadic_addr_norb65_nocc2() {
        let norb = 65;
        let nocc = 2;
        let nw = nwords(norb).max(1);
        let raw = make_strings_core(norb, nocc);
        for (expected_addr, chunk) in raw.chunks_exact(nw).enumerate() {
            let s = BitString { words: chunk };
            assert_eq!(
                combinadic_addr(norb, nocc, s),
                expected_addr as i32,
                "addr mismatch at index {expected_addr}"
            );
        }
    }

    #[test]
    fn test_addr_from_occupied_norb65() {
        // Row order: strings without orbital 64 first (C(64,2)=2016 strings),
        // then strings with orbital 64 (C(64,1)=64 strings).
        // [0, 64] is the first string in the "with orbital 64" block → index 2016.
        let addr = addr_from_occupied(65, 2, vec![0, 64]);
        assert_eq!(addr, 2016);
        // [63, 64] is the last string → index 2016+63 = 2079.
        let addr = addr_from_occupied(65, 2, vec![63, 64]);
        assert_eq!(addr, 2079);
    }
}
