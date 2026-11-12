// This code is a Qiskit project.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ndarray::ArrayViewMut2;
use num_complex::Complex;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::dtype::{cast_complex, ComplexVecMut, Real};

/// Apply a Givens rotation between two slices of a vector
#[pyfunction]
pub fn apply_givens_rotation_in_place(
    vec: ComplexVecMut,
    c: f64,
    s: Complex64,
    slice1: PyReadonlyArray1<usize>,
    slice2: PyReadonlyArray1<usize>,
) -> PyResult<()> {
    let slice1 = slice1.as_slice()?;
    let slice2 = slice2.as_slice()?;
    match vec {
        ComplexVecMut::Complex128(mut vec) => {
            apply_givens_rotation(vec.as_array_mut(), c, s, slice1, slice2)
        }
        ComplexVecMut::Complex64(mut vec) => {
            apply_givens_rotation(vec.as_array_mut(), c, s, slice1, slice2)
        }
    }
}

fn apply_givens_rotation<T: Real>(
    mut vec: ArrayViewMut2<Complex<T>>,
    c: f64,
    s: Complex64,
    slice1: &[usize],
    slice2: &[usize],
) -> PyResult<()> {
    if !vec.is_standard_layout() {
        return Err(PyValueError::new_err("vec must be C-contiguous."));
    }
    if slice1.len() != slice2.len() {
        return Err(PyValueError::new_err(
            "slice1 and slice2 must have the same length.",
        ));
    }
    let dim_a = vec.shape()[0];
    if slice1.iter().chain(slice2).any(|&i| i >= dim_a) {
        return Err(PyValueError::new_err("Row index out of bounds."));
    }
    if slice1.is_empty() {
        return Ok(());
    }

    let c = T::from_f64(c);
    let s = cast_complex::<T>(s);
    let dim_b = vec.shape()[1];

    let n_pairs = slice1.len();
    let n_threads = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .min(n_pairs);

    // Sequential execution for single thread or small slice
    if n_threads == 1 || n_pairs < 128 {
        let vec_ptr = vec.as_mut_ptr();
        for (&i, &j) in slice1.iter().zip(slice2) {
            unsafe {
                _apply_givens_rotation_to_pair(vec_ptr, i, j, dim_b, c, s);
            }
        }
        return Ok(());
    }

    // Parallel execution
    let chunk_size = n_pairs.div_ceil(n_threads);
    let vec_ptr = vec.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        for k in 0..n_threads {
            let start = k * chunk_size;
            if start >= n_pairs {
                break;
            }
            let end = (start + chunk_size).min(n_pairs);
            let slice1_chunk = &slice1[start..end];
            let slice2_chunk = &slice2[start..end];
            scope.spawn(move || {
                let vec_ptr = vec_ptr as *mut Complex<T>;
                for (&i, &j) in slice1_chunk.iter().zip(slice2_chunk) {
                    unsafe {
                        _apply_givens_rotation_to_pair(vec_ptr, i, j, dim_b, c, s);
                    }
                }
            });
        }
    });
    Ok(())
}

/// Apply Givens rotation to a pair of rows
///
/// # Safety
///
/// `vec_ptr` must point to a C-contiguous array with at least `max(i, j) + 1`
/// rows of length `dim_b`, and rows `i` and `j` must be distinct and not be
/// accessed concurrently by another thread.
unsafe fn _apply_givens_rotation_to_pair<T: Real>(
    vec_ptr: *mut Complex<T>,
    i: usize,
    j: usize,
    dim_b: usize,
    c: T,
    s: Complex<T>,
) {
    let row_i = std::slice::from_raw_parts_mut(vec_ptr.add(i * dim_b), dim_b);
    let row_j = std::slice::from_raw_parts_mut(vec_ptr.add(j * dim_b), dim_b);
    for k in 0..dim_b {
        // This is BLAS zrot but it's not available
        // See https://github.com/qiskit-community/ffsim/issues/28
        let i_old = row_i[k];
        let j_old = row_j[k];
        row_i[k] = i_old * c + j_old * s;
        row_j[k] = j_old * c - i_old * s.conj();
    }
}
