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

use ndarray::{ArrayView1, ArrayView2, ArrayViewMut2, Zip};
use num_complex::Complex;
use numpy::Complex64;
use numpy::PyReadonlyArray1;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use crate::dtype::{cast_complex, ComplexVecMut, Real};

/// Apply time evolution by a sum of number operators in-place.
#[pyfunction]
pub fn apply_num_op_sum_evolution_in_place(
    vec: ComplexVecMut,
    phases: PyReadonlyArray1<Complex64>,
    occupations: PyReadonlyArray2<usize>,
) {
    let phases = phases.as_array();
    let occupations = occupations.as_array();
    match vec {
        ComplexVecMut::Complex128(mut vec) => {
            apply_num_op_sum_evolution(vec.as_array_mut(), phases, occupations)
        }
        ComplexVecMut::Complex64(mut vec) => {
            apply_num_op_sum_evolution(vec.as_array_mut(), phases, occupations)
        }
    }
}

fn apply_num_op_sum_evolution<T: Real>(
    mut vec: ArrayViewMut2<Complex<T>>,
    phases: ArrayView1<Complex64>,
    occupations: ArrayView2<usize>,
) {
    Zip::from(vec.rows_mut())
        .and(occupations.rows())
        .par_for_each(|mut row, orbs| {
            // The phase is accumulated in double precision and rounded once.
            let mut phase = Complex64::new(1.0, 0.0);
            orbs.for_each(|&orb| phase *= phases[orb]);
            let phase = cast_complex::<T>(phase);
            row.map_inplace(|val| *val *= phase);
        });
}
