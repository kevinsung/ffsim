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
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods};
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Module-level caches — keyed on (norb, nocc).
// On a cache hit the same numpy buffer is returned (refcount bump, no copy).
// All returned arrays are marked read-only.
// ---------------------------------------------------------------------------

static MAKE_STRINGS_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray1<i64>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_OCCSLST_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray2<usize>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_LINKSTR_INDEX_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray3<i32>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static GEN_LINKSTR_INDEX_TRILIDX_CACHE: Lazy<Mutex<HashMap<(usize, usize), Py<PyArray3<i32>>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// Pure-Rust core algorithms (no GIL required)
// ---------------------------------------------------------------------------

/// Return the combinadic rank of `string` within the set C(norb, nocc).
///
/// Matches pyscf's `_str2addr`: scans set bits MSB-to-LSB; for each set bit
/// at position `o`, adds C(o, nelec) then decrements nelec.
fn combinadic_addr(norb: usize, nocc: usize, string: i64) -> i32 {
    let mut addr: i64 = 0;
    let mut k = nocc as i64; // remaining electrons (decrements after each set bit)
    for o in (0..norb).rev() {
        if (string >> o) & 1 == 1 {
            addr += binomial(o as i64, k);
            k -= 1;
        }
    }
    addr as i32
}

/// Build the list of FCI occupation strings for `nocc` electrons in `norb`
/// orbitals, as i64 bitmasks.
///
/// Row order matches pyscf's `cistring.make_strings(range(norb), nocc)`:
/// strings not occupying orbital `norb-1` come first (recursion on orb[:-1]),
/// followed by strings occupying orbital `norb-1`.
///
/// Edge cases:
/// - `nocc == 0`: returns `[0]`.
/// - `nocc == norb`: returns `[(1 << norb) - 1]`.
/// - `nocc > norb`: returns `[]`.
fn make_strings_core(norb: usize, nocc: usize) -> Vec<i64> {
    if nocc > norb {
        return vec![];
    }
    if nocc == 0 {
        return vec![0];
    }
    // Use an iterative approach: build level by level.
    // Level 0: one string with the lowest `nocc` orbitals set.
    // At each level we extend the previous level's strings by one orbital.
    let n = binomial(norb, nocc) as usize;
    let mut result = Vec::with_capacity(n);

    // Recursive helper via explicit stack to avoid recursion-depth issues.
    gen_strings_iter(&mut result, 0, norb, nocc);
    result
}

/// Recursive helper that fills `out` in pyscf's row order.
/// `base` is the bitmask accumulated so far from higher orbitals,
/// `orbs` is the number of orbitals remaining (we consider orbs 0..orbs-1),
/// `nelec` is the number of electrons still to place.
fn gen_strings_iter(out: &mut Vec<i64>, base: i64, orbs: usize, nelec: usize) {
    if nelec == 0 {
        out.push(base);
        return;
    }
    if nelec == orbs {
        // All remaining orbitals must be occupied.
        let mask = (1i64 << orbs) - 1;
        out.push(base | mask);
        return;
    }
    // Strings without orbital `orbs-1` first.
    gen_strings_iter(out, base, orbs - 1, nelec);
    // Strings with orbital `orbs-1`.
    gen_strings_iter(out, base | (1i64 << (orbs - 1)), orbs - 1, nelec - 1);
}

/// Build the occupation-list table: each row lists the sorted orbital indices
/// occupied in the corresponding FCI string.  Row order matches
/// `make_strings_core`.
///
/// Output shape: `(C(norb, nocc), nocc)`, dtype `usize`.
/// Edge cases: `nocc == 0` → shape `(1, 0)`; `nocc > norb` → shape `(0, nocc)`.
fn gen_occslst_core(norb: usize, nocc: usize) -> Array2<usize> {
    if nocc > norb {
        return Array2::zeros((0, nocc));
    }
    let n = binomial(norb, nocc) as usize;
    let mut result = Array2::<usize>::zeros((n, nocc));
    let strings = make_strings_core(norb, nocc);
    for (row_idx, &string) in strings.iter().enumerate() {
        let mut col = 0;
        for o in 0..norb {
            if (string >> o) & 1 == 1 {
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
///   for each occupied orbital `i` (no actual excitation).
/// - Next `nocc * nvir` rows: `(a, i, addr(str1), sign)` for every occupied
///   `i` (outer loop) and virtual `a` (inner loop).
///
/// If `trilidx` is true, column 0 contains the triangular index
/// `max(a,i)*(max(a,i)+1)/2 + min(a,i)` instead of `a`, and column 1 is set
/// to zero (rather than `i`).
///
/// Output shape: `(C(norb,nocc), nocc + nocc*nvir, 4)`, dtype `i32`.
fn gen_linkstr_index_core(norb: usize, nocc: usize, trilidx: bool) -> Array3<i32> {
    let nvir = norb - nocc.min(norb);
    let nlink = nocc + nocc * nvir;
    let strings = make_strings_core(norb, nocc);
    let na = strings.len();
    let mut result = Array3::<i32>::zeros((na, nlink, 4));

    // Pre-build occupied and virtual lists for each string.
    for (str_idx, &str0) in strings.iter().enumerate() {
        let occupied: Vec<usize> = (0..norb).filter(|&o| (str0 >> o) & 1 == 1).collect();
        let virtual_orbs: Vec<usize> = (0..norb).filter(|&o| (str0 >> o) & 1 == 0).collect();

        // Diagonal entries.
        for (k, &i) in occupied.iter().enumerate() {
            let addr = combinadic_addr(norb, nocc, str0);
            if trilidx {
                let tri = (i * (i + 1) / 2 + i) as i32;
                result[[str_idx, k, 0]] = tri;
                result[[str_idx, k, 1]] = 0;
            } else {
                result[[str_idx, k, 0]] = i as i32;
                result[[str_idx, k, 1]] = i as i32;
            }
            result[[str_idx, k, 2]] = addr;
            result[[str_idx, k, 3]] = 1;
        }

        // Off-diagonal entries: outer loop over occupied i, inner over virtual a.
        let mut link_idx = nocc;
        for &i in &occupied {
            for &a in &virtual_orbs {
                let str1 = (str0 ^ (1i64 << i)) | (1i64 << a);
                let addr1 = combinadic_addr(norb, nocc, str1);
                // Sign: (-1)^(number of occupied orbitals strictly between i and a).
                let lo = i.min(a);
                let hi = i.max(a);
                // mask covers orbitals strictly between lo and hi.
                let mask = if hi > lo + 1 {
                    ((1i64 << (hi - lo - 1)) - 1) << (lo + 1)
                } else {
                    0
                };
                let sign = if (str0 & mask).count_ones() % 2 == 0 {
                    1i32
                } else {
                    -1
                };
                if trilidx {
                    let tri = (hi * (hi + 1) / 2 + lo) as i32;
                    result[[str_idx, link_idx, 0]] = tri;
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
/// Returns a read-only i64 array of shape ``(C(norb, nocc),)``.
/// Subsequent calls with the same arguments return the same buffer.
#[pyfunction]
pub fn make_strings(py: Python<'_>, norb: usize, nocc: usize) -> Py<PyArray1<i64>> {
    {
        let cache = MAKE_STRINGS_CACHE.lock().unwrap();
        if let Some(arr) = cache.get(&(norb, nocc)) {
            return arr.clone_ref(py);
        }
    }
    let data = py.detach(|| make_strings_core(norb, nocc));
    let arr = data.into_pyarray(py);
    arr.readwrite().make_nonwriteable();
    let py_arr = arr.unbind();
    let mut cache = MAKE_STRINGS_CACHE.lock().unwrap();
    cache.entry((norb, nocc)).or_insert_with(|| py_arr.clone_ref(py));
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
    cache.entry((norb, nocc)).or_insert_with(|| py_arr.clone_ref(py));
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
    cache.entry((norb, nocc)).or_insert_with(|| py_arr.clone_ref(py));
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
    cache.entry((norb, nocc)).or_insert_with(|| py_arr.clone_ref(py));
    cache[&(norb, nocc)].clone_ref(py)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_strings_edge_cases() {
        assert_eq!(make_strings_core(0, 0), vec![0]);
        assert_eq!(make_strings_core(4, 0), vec![0]);
        assert_eq!(make_strings_core(4, 4), vec![15]);
        assert_eq!(make_strings_core(4, 5), vec![]);
    }

    #[test]
    fn test_make_strings_norb4_nocc2() {
        // pyscf: array([3, 5, 6, 9, 10, 12])
        assert_eq!(
            make_strings_core(4, 2),
            vec![
                0b0011, // 3
                0b0101, // 5
                0b0110, // 6
                0b1001, // 9
                0b1010, // 10
                0b1100, // 12
            ]
        );
    }

    #[test]
    fn test_gen_occslst_norb4_nocc2() {
        let tbl = gen_occslst_core(4, 2);
        assert_eq!(tbl.shape(), &[6, 2]);
        // pyscf: [[0,1],[0,2],[1,2],[0,3],[1,3],[2,3]]
        let expected: &[[usize; 2]] = &[
            [0, 1],
            [0, 2],
            [1, 2],
            [0, 3],
            [1, 3],
            [2, 3],
        ];
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
        // shape: (6, 6, 4)
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
                [tbl[[0, k, 0]], tbl[[0, k, 1]], tbl[[0, k, 2]], tbl[[0, k, 3]]],
                *exp,
                "row0 link {k} mismatch"
            );
        }
    }

    #[test]
    fn test_gen_linkstr_index_trilidx_norb4_nocc2() {
        let tbl = gen_linkstr_index_core(4, 2, true);
        assert_eq!(tbl.shape(), &[6, 6, 4]);
        // pyscf trilidx row0 (col1 garbage zeroed):
        // [[0,0,0,1],[2,0,0,1],[3,0,2,-1],[6,0,4,-1],[4,0,1,1],[7,0,3,1]]
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
                [tbl[[0, k, 0]], tbl[[0, k, 1]], tbl[[0, k, 2]], tbl[[0, k, 3]]],
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
        // C(4,4) = 1, nlink = 4 + 4*0 = 4
        assert_eq!(tbl.shape(), &[1, 4, 4]);
    }

    #[test]
    fn test_combinadic_addr_norb4_nocc2() {
        // strings: [3,5,6,9,10,12] -> addresses [0,1,2,3,4,5]
        let strings = make_strings_core(4, 2);
        for (expected_addr, &s) in strings.iter().enumerate() {
            assert_eq!(
                combinadic_addr(4, 2, s),
                expected_addr as i32,
                "addr mismatch for string {s:#b}"
            );
        }
    }
}
