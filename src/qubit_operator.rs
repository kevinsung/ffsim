// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use num_integer::binomial;
use numpy::Complex64;
use pyo3::exceptions::PyKeyError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyFrozenSet;
use std::collections::HashMap;

// Internal key type: sorted Vec of (pauli_byte, qubit) where pauli_byte is b'X', b'Y', or b'Z'.
// Sorting ensures canonical form so that equivalent frozensets map to the same key.
type PauliTerm = Vec<(u8, i32)>;

#[pyclass]
struct KeysIterator {
    keys: std::vec::IntoIter<PauliTerm>,
}

#[pymethods]
impl KeysIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Bound<'_, PyFrozenSet>>> {
        match slf.keys.next() {
            None => Ok(None),
            Some(key) => {
                let py = slf.py();
                Ok(Some(key_to_frozenset(py, &key)?))
            }
        }
    }
}

/// Convert a Python frozenset of QubitAction (or equivalent tuples) to a canonical sorted PauliTerm.
///
/// Each element of the frozenset is expected to be an indexable object (namedtuple or plain tuple)
/// with element 0 being the Pauli label ('X', 'Y', or 'Z') and element 1 being the qubit index.
fn extract_term(obj: &Bound<'_, PyAny>) -> PyResult<PauliTerm> {
    let mut actions: PauliTerm = Vec::new();
    for item in obj.try_iter()? {
        let item = item?;
        let pauli: String = item.get_item(0)?.extract()?;
        let qubit: i32 = item.get_item(1)?.extract()?;
        if pauli.len() != 1 {
            return Err(PyValueError::new_err(
                "Pauli label must be a single character.",
            ));
        }
        let pauli_byte = pauli.as_bytes()[0];
        if !matches!(pauli_byte, b'X' | b'Y' | b'Z') {
            return Err(PyValueError::new_err("Pauli label must be 'X', 'Y', or 'Z'."));
        }
        actions.push((pauli_byte, qubit));
    }
    actions.sort_unstable();
    Ok(actions)
}

/// Convert a canonical PauliTerm back to a Python frozenset of (str, int) tuples.
fn key_to_frozenset<'py>(py: Python<'py>, key: &PauliTerm) -> PyResult<Bound<'py, PyFrozenSet>> {
    let tuples: Vec<(String, i32)> = key
        .iter()
        .map(|(p, q)| ((*p as char).to_string(), *q))
        .collect();
    PyFrozenSet::new(py, tuples)
}

fn format_complex(val: Complex64) -> String {
    if val.im < 0.0 {
        format!("{}{}j", val.re, val.im)
    } else {
        format!("{}+{}j", val.re, val.im)
    }
}

/// Multiply two Pauli characters on the same qubit.
///
/// Returns `(result_pauli_or_0, re, im)` where `result_pauli_or_0 == 0` indicates the identity
/// (the qubit should be removed from the term) and `(re, im)` is the accumulated phase factor.
fn pauli_mul(p1: u8, p2: u8) -> (u8, f64, f64) {
    if p1 == p2 {
        // XX = YY = ZZ = I (phase 1)
        return (0, 1.0, 0.0);
    }
    match (p1, p2) {
        (b'X', b'Y') => (b'Z', 0.0, 1.0),   // XY = iZ
        (b'Y', b'X') => (b'Z', 0.0, -1.0),  // YX = -iZ
        (b'Y', b'Z') => (b'X', 0.0, 1.0),   // YZ = iX
        (b'Z', b'Y') => (b'X', 0.0, -1.0),  // ZY = -iX
        (b'Z', b'X') => (b'Y', 0.0, 1.0),   // ZX = iY
        (b'X', b'Z') => (b'Y', 0.0, -1.0),  // XZ = -iY
        _ => unreachable!(),
    }
}

/// Multiply two Pauli terms (as canonical sorted PauliTerms), returning the result term and phase.
fn mul_pauli_terms(term1: &PauliTerm, term2: &PauliTerm) -> (PauliTerm, Complex64) {
    let mut qubit_to_pauli: HashMap<i32, u8> = term1.iter().map(|&(p, q)| (q, p)).collect();
    let mut phase = Complex64::new(1.0, 0.0);
    for &(p2, qubit) in term2 {
        if let Some(&p1) = qubit_to_pauli.get(&qubit) {
            let (result_p, re, im) = pauli_mul(p1, p2);
            phase *= Complex64::new(re, im);
            if result_p == 0 {
                // Result is identity: remove this qubit from the term
                qubit_to_pauli.remove(&qubit);
            } else {
                *qubit_to_pauli.get_mut(&qubit).unwrap() = result_p;
            }
        } else {
            // Paulis on different qubits tensor together
            qubit_to_pauli.insert(qubit, p2);
        }
    }
    let mut result: PauliTerm = qubit_to_pauli.into_iter().map(|(q, p)| (p, q)).collect();
    result.sort_unstable();
    (result, phase)
}

/// Compute the trace contribution of Z operators on a single spin sector.
///
/// Returns Σ_{k=0}^{min(num_z, nelec)} (-1)^k * C(num_z, k) * C(norb - num_z, nelec - k).
/// This counts, over all `nelec`-electron configurations of `norb` orbitals, the product
/// of Z eigenvalues on the `num_z` marked orbitals (Z gives +1 for unoccupied, -1 for occupied).
fn z_term_trace(norb: usize, nelec: usize, num_z: usize) -> i64 {
    if num_z > norb {
        return 0;
    }
    let remaining = norb - num_z;
    let mut sum: i64 = 0;
    for k in 0..=num_z.min(nelec) {
        let n_remaining = nelec - k;
        if n_remaining > remaining {
            continue; // C(remaining, n_remaining) = 0
        }
        let c1 = binomial(num_z as u64, k as u64) as i64;
        let c2 = binomial(remaining as u64, n_remaining as u64) as i64;
        if k % 2 == 0 {
            sum += c1 * c2;
        } else {
            sum -= c1 * c2;
        }
    }
    sum
}

/// A qubit operator.
///
/// A QubitOperator represents a linear combination of tensor products of Pauli operators.
/// Because Pauli operators on different qubits commute, the keys are frozensets of
/// ``(pauli, qubit)`` pairs, each specifying a Pauli label (``'X'``, ``'Y'``, or ``'Z'``)
/// and a qubit index. The identity term is represented by an empty frozenset.
/// Initialize a QubitOperator by passing a dictionary mapping the terms in the linear
/// combination to their associated coefficients.
///
/// Example:
///
/// .. code-block:: python
///
///     import ffsim
///
///     op = ffsim.QubitOperator(
///         {
///             frozenset({ffsim.x(0), ffsim.z(2)}): 0.5,
///             frozenset({ffsim.y(1)}): 1 + 1j,
///         }
///     )
///     print(2 * op)
///
/// Args:
///     coeffs (dict[frozenset, complex]): The coefficients of the operator. Keys are
///         frozensets of ``(pauli, qubit)`` pairs (e.g., :class:`QubitAction` namedtuples).
#[pyclass(module = "ffsim", mapping)]
#[derive(Clone)]
pub struct QubitOperator {
    coeffs: HashMap<PauliTerm, Complex64>,
}

#[pymethods]
impl QubitOperator {
    #[new]
    fn new(coeffs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut result: HashMap<PauliTerm, Complex64> = HashMap::new();
        for item in coeffs.call_method0("items")?.try_iter()? {
            let item = item?;
            let key = item.get_item(0)?;
            let coeff: Complex64 = item.get_item(1)?.extract()?;
            let term = extract_term(&key)?;
            result.insert(term, coeff);
        }
        Ok(Self { coeffs: result })
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        let mut items_str: Vec<String> = Vec::new();
        for (key, &val) in &self.coeffs {
            let key_str = if key.is_empty() {
                "frozenset()".to_string()
            } else {
                let actions: Vec<String> = key
                    .iter()
                    .map(|(p, q)| format!("('{}', {})", *p as char, q))
                    .collect();
                format!("frozenset({{{}}})", actions.join(", "))
            };
            let val_str = format_complex(val);
            items_str.push(format!("{key_str}: {val_str}"));
        }
        format!("QubitOperator({{{}}})", items_str.join(", "))
    }

    fn _repr_pretty_str(&self) -> String {
        let mut items_str: Vec<String> = Vec::new();
        for (key, &val) in &self.coeffs {
            let key_str = if key.is_empty() {
                "frozenset()".to_string()
            } else {
                let actions: Vec<String> = key
                    .iter()
                    .map(|(p, q)| {
                        let label = match *p {
                            b'X' => "x",
                            b'Y' => "y",
                            b'Z' => "z",
                            _ => unreachable!(),
                        };
                        format!("{label}({q})")
                    })
                    .collect();
                format!("{{{}}}", actions.join(", "))
            };
            let val_str = if val.im == 0.0 {
                format!("{}", val.re)
            } else if val.im < 0.0 {
                format!("{}{}j", val.re, val.im)
            } else {
                format!("{}+{}j", val.re, val.im)
            };
            items_str.push(format!("    {key_str}: {val_str}"));
        }
        format!("QubitOperator({{\n{}\n}})", items_str.join(",\n"))
    }

    fn _repr_pretty_(&self, p: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        if cycle {
            p.call_method1("text", ("QubitOperator(...)",))?;
        } else {
            p.call_method1("text", (self._repr_pretty_str(),))?;
        }
        Ok(())
    }

    fn __str__(&self) -> String {
        self._repr_pretty_str()
    }

    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Complex64> {
        let term = extract_term(key)?;
        self.coeffs
            .get(&term)
            .ok_or_else(|| PyKeyError::new_err("Term not present in operator."))
            .copied()
    }

    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: Complex64) -> PyResult<()> {
        let term = extract_term(key)?;
        self.coeffs.insert(term, value);
        Ok(())
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let term = extract_term(key)?;
        self.coeffs.remove(&term);
        Ok(())
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let term = extract_term(key)?;
        Ok(self.coeffs.contains_key(&term))
    }

    fn __len__(&self) -> usize {
        self.coeffs.len()
    }

    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<KeysIterator>> {
        let keys = slf.coeffs.keys().cloned().collect::<Vec<_>>().into_iter();
        Py::new(slf.py(), KeysIterator { keys })
    }

    fn __iadd__(&mut self, other: &Self) {
        for (term, coeff) in &other.coeffs {
            let val = self.coeffs.entry(term.clone()).or_default();
            *val += coeff;
        }
    }

    fn __add__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.__iadd__(other);
        result
    }

    fn __isub__(&mut self, other: &Self) {
        for (term, coeff) in &other.coeffs {
            let val = self.coeffs.entry(term.clone()).or_default();
            *val -= coeff;
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.__isub__(other);
        result
    }

    fn __neg__(&self) -> Self {
        let mut result = self.clone();
        result.__imul__(Complex64::new(-1.0, 0.0));
        result
    }

    fn __itruediv__(&mut self, other: Complex64) {
        for coeff in self.coeffs.values_mut() {
            *coeff /= other;
        }
    }

    fn __truediv__(&self, other: Complex64) -> Self {
        let mut coeffs = HashMap::with_capacity(self.coeffs.len());
        for (term, coeff) in &self.coeffs {
            coeffs.insert(term.clone(), coeff / other);
        }
        Self { coeffs }
    }

    fn __imul__(&mut self, other: Complex64) {
        for coeff in self.coeffs.values_mut() {
            *coeff *= other;
        }
    }

    fn __rmul__(&self, other: Complex64) -> Self {
        let mut coeffs = HashMap::with_capacity(self.coeffs.len());
        for (term, coeff) in &self.coeffs {
            coeffs.insert(term.clone(), other * coeff);
        }
        Self { coeffs }
    }

    fn __mul__(&self, other: &Self) -> Self {
        let mut coeffs: HashMap<PauliTerm, Complex64> = HashMap::new();
        for (term1, coeff1) in &self.coeffs {
            for (term2, coeff2) in &other.coeffs {
                let (new_term, phase) = mul_pauli_terms(term1, term2);
                let new_coeff = phase * coeff1 * coeff2;
                let val = coeffs.entry(new_term).or_insert(Complex64::default());
                *val += new_coeff;
            }
        }
        Self { coeffs }
    }

    fn __pow__(&self, exponent: u32, modulo: Option<u32>) -> PyResult<Self> {
        match modulo {
            Some(_) => Err(PyValueError::new_err("mod argument not supported")),
            None => {
                let mut coeffs = HashMap::new();
                coeffs.insert(vec![], Complex64::new(1.0, 0.0));
                let mut result = Self { coeffs };
                for _ in 0..exponent {
                    result = result.__mul__(self);
                }
                Ok(result)
            }
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }

    /// Return the adjoint (Hermitian conjugate) of the operator.
    ///
    /// Since all Pauli operators are Hermitian (X†=X, Y†=Y, Z†=Z), the adjoint of a
    /// QubitOperator is obtained by taking the complex conjugate of all coefficients.
    ///
    /// Returns:
    ///     QubitOperator: The adjoint of the qubit operator.
    fn adjoint(&self) -> Self {
        let mut coeffs = HashMap::with_capacity(self.coeffs.len());
        for (term, coeff) in &self.coeffs {
            coeffs.insert(term.clone(), coeff.conj());
        }
        Self { coeffs }
    }

    /// Return the many-body order of the operator.
    ///
    /// The many-body order is defined as the maximum number of Pauli factors in any term.
    ///
    /// Returns:
    ///     int: The many-body order of the operator.
    fn many_body_order(&self) -> usize {
        self.coeffs.keys().map(|term| term.len()).max().unwrap_or(0)
    }

    /// Remove terms with small coefficients in place.
    ///
    /// Removes terms with coefficients whose absolute value is below the specified tolerance.
    /// Modifies the operator in place.
    ///
    /// Args:
    ///     tol (float): The tolerance threshold. Terms with coefficients whose
    ///         absolute value is less than or equal to this value will be removed.
    #[pyo3(signature = (tol=1e-12))]
    fn simplify(&mut self, tol: f64) -> PyResult<()> {
        self.coeffs.retain(|_, coeff| coeff.norm() > tol);
        Ok(())
    }

    /// Return the trace of the operator in the Fock space sector.
    ///
    /// Computes the trace over all basis states with ``n_alpha`` alpha electrons and
    /// ``n_beta`` beta electrons in ``norb`` spatial orbitals. Qubits 0..norb-1 are
    /// treated as alpha spin-orbitals and qubits norb..2*norb-1 as beta spin-orbitals,
    /// following the Jordan-Wigner convention used by ffsim.
    ///
    /// Only terms consisting entirely of Z operators (and the implicit identity) contribute;
    /// any term containing an X or Y operator has zero trace.
    ///
    /// Args:
    ///     norb (int): The number of spatial orbitals.
    ///     nelec (tuple[int, int]): The number of alpha and beta electrons.
    ///
    /// Returns:
    ///     complex: The trace of the operator.
    fn _trace_(&self, norb: usize, nelec: (usize, usize)) -> Complex64 {
        let (n_alpha, n_beta) = nelec;
        let mut result = Complex64::default();
        for (term, &coeff) in &self.coeffs {
            // Any X or Y flips the qubit, so ⟨s|P|s⟩ = 0; skip this term.
            if term.iter().any(|&(p, _)| p == b'X' || p == b'Y') {
                continue;
            }
            // Count Z operators in the alpha sector (qubit < norb) and beta sector (qubit >= norb).
            let mut n_alpha_z: usize = 0;
            let mut n_beta_z: usize = 0;
            let mut skip = false;
            for &(_, q) in term {
                if q < 0 {
                    skip = true;
                    break;
                }
                let q = q as usize;
                if q >= 2 * norb {
                    // Qubit outside the system; Tr(Z) = 0 in the full qubit space.
                    skip = true;
                    break;
                } else if q < norb {
                    n_alpha_z += 1;
                } else {
                    n_beta_z += 1;
                }
            }
            if skip {
                continue;
            }
            let alpha_trace = z_term_trace(norb, n_alpha, n_alpha_z);
            let beta_trace = z_term_trace(norb, n_beta, n_beta_z);
            result += coeff * (alpha_trace as f64) * (beta_trace as f64);
        }
        result
    }

    fn _approx_eq_(&self, other: &Self, rtol: f64, atol: f64) -> bool {
        for key in self.coeffs.keys().chain(other.coeffs.keys()) {
            let val_self = *self.coeffs.get(key).unwrap_or(&Complex64::default());
            let val_other = *other.coeffs.get(key).unwrap_or(&Complex64::default());
            if (val_self - val_other).norm() > atol + rtol * val_other.norm() {
                return false;
            }
        }
        true
    }
}

impl QubitOperator {
    /// Construct a QubitOperator directly from a coefficient map (crate-internal use).
    pub(crate) fn from_coeffs(coeffs: HashMap<PauliTerm, Complex64>) -> Self {
        Self { coeffs }
    }
}
