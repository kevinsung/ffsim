// This code is a Qiskit project.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Support for state vectors in single and double precision.
//!
//! Gate kernels are generic over the real type `T` of the vector entries
//! `Complex<T>`. Python entry points accept a [`ComplexVecMut`], which holds
//! either a complex64 or a complex128 array, and dispatch to the generic kernel.
//! Gate parameters are always passed in double precision and rounded to `T`
//! once inside the kernel.

use num_complex::Complex;
use num_traits::{Float, NumAssign};
use numpy::{Complex32, Complex64, PyReadwriteArray2};
use pyo3::prelude::*;

/// Real type of the entries of a state vector.
pub trait Real: Float + NumAssign + Send + Sync + 'static {
    /// Round a double-precision number to this type.
    fn from_f64(x: f64) -> Self;
}

impl Real for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}

impl Real for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// A mutable two-dimensional complex64 or complex128 array.
#[derive(FromPyObject)]
pub enum ComplexVecMut<'py> {
    Complex128(PyReadwriteArray2<'py, Complex64>),
    Complex64(PyReadwriteArray2<'py, Complex32>),
}

impl ComplexVecMut<'_> {
    /// The shape of the array.
    pub fn dim(&self) -> (usize, usize) {
        match self {
            ComplexVecMut::Complex128(vec) => vec.as_array().dim(),
            ComplexVecMut::Complex64(vec) => vec.as_array().dim(),
        }
    }
}

/// Round a double-precision complex number to precision `T`.
#[inline]
pub fn cast_complex<T: Real>(z: Complex64) -> Complex<T> {
    Complex::new(T::from_f64(z.re), T::from_f64(z.im))
}
