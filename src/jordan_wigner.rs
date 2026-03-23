use crate::fermion_operator::FermionOperator;
use crate::qubit_operator::QubitOperator;
use numpy::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// Multiply two single-qubit Pauli bytes (b'I', b'X', b'Y', b'Z').
///
/// Returns `(result_byte, phase)` where `phase` is the scalar phase factor.
#[inline]
fn mul_pauli(a: u8, b: u8) -> (u8, Complex64) {
    match (a, b) {
        (b'I', p) | (p, b'I') => (p, Complex64::new(1.0, 0.0)),
        (b'X', b'X') | (b'Y', b'Y') | (b'Z', b'Z') => (b'I', Complex64::new(1.0, 0.0)),
        (b'X', b'Y') => (b'Z', Complex64::new(0.0, 1.0)),
        (b'X', b'Z') => (b'Y', Complex64::new(0.0, -1.0)),
        (b'Y', b'X') => (b'Z', Complex64::new(0.0, -1.0)),
        (b'Y', b'Z') => (b'X', Complex64::new(0.0, 1.0)),
        (b'Z', b'X') => (b'Y', Complex64::new(0.0, 1.0)),
        (b'Z', b'Y') => (b'X', Complex64::new(0.0, -1.0)),
        _ => unreachable!(),
    }
}

/// Apply Z operators on qubits `0..q`, then apply `main` on qubit `q`.
///
/// Mutates the dense Pauli string `s` in place and returns the accumulated phase.
#[inline]
fn apply_z_string_and_main(s: &mut [u8], q: usize, main: u8) -> Complex64 {
    let mut phase = Complex64::new(1.0, 0.0);
    for i in 0..q {
        let (c, p) = mul_pauli(s[i], b'Z');
        s[i] = c;
        phase *= p;
    }
    let (c, p) = mul_pauli(s[q], main);
    s[q] = c;
    phase *= p;
    phase
}

/// Jordan-Wigner transform of a FermionOperator to a QubitOperator.
///
/// Maps a fermionic operator to a qubit operator using the Jordan-Wigner transformation.
/// Alpha spin-orbitals with orbital index ``orb`` are mapped to qubit ``orb``, and beta
/// spin-orbitals with orbital index ``orb`` are mapped to qubit ``norb + orb``.
///
/// Args:
///     op (FermionOperator): The fermionic operator to transform.
///     norb (int | None): The number of spatial orbitals. When ``None``, the value is
///         inferred as one more than the largest orbital index that appears in the operator,
///         or 0 if the operator is empty.
///     tol (float): Tolerance below which coefficients are dropped. Defaults to 1e-12.
///
/// Returns:
///     QubitOperator: The Jordan-Wigner image of the operator.
#[pyfunction]
#[pyo3(signature = (op, norb=None, tol=1e-12))]
pub fn jordan_wigner(op: &FermionOperator, norb: Option<usize>, tol: f64) -> QubitOperator {
    let norb: usize = norb.unwrap_or_else(|| {
        op.coeffs()
            .keys()
            .flat_map(|ops| ops.iter().map(|&(_, _, orb)| orb as usize + 1))
            .max()
            .unwrap_or(0)
    });
    let n_qubits: usize = 2 * norb;
    let identity = vec![b'I'; n_qubits];

    // Parallel accumulation over fermionic terms: each term independently expands
    // into Pauli products, so the outer loop is embarrassingly parallel.
    let acc: HashMap<Vec<u8>, Complex64> = op
        .coeffs()
        .par_iter()
        .fold(
            HashMap::<Vec<u8>, Complex64>::new,
            |mut acc_local, (ops, &term_coeff)| {
                // Per-term expansion starting from the identity
                let mut current: HashMap<Vec<u8>, Complex64> = HashMap::new();
                current.insert(identity.clone(), Complex64::new(1.0, 0.0));

                // ops: Vec<(action, spin, orb)>
                // action: true=creation, false=annihilation
                // spin:   false=alpha, true=beta
                for &(action, spin, orb_i32) in ops {
                    let orb = orb_i32 as usize;
                    let q = orb + if spin { norb } else { 0 };

                    // a^dag = (X - iY)/2, a = (X + iY)/2
                    let coeff_x = Complex64::new(0.5, 0.0);
                    let coeff_y = if action {
                        Complex64::new(0.0, -0.5)
                    } else {
                        Complex64::new(0.0, 0.5)
                    };

                    let mut next: HashMap<Vec<u8>, Complex64> = HashMap::new();
                    for (ps, c) in current.into_iter() {
                        // X branch
                        let mut s_x = ps.clone();
                        let phase_x = apply_z_string_and_main(&mut s_x, q, b'X');
                        *next.entry(s_x).or_insert(Complex64::new(0.0, 0.0)) +=
                            c * coeff_x * phase_x;

                        // Y branch
                        let mut s_y = ps;
                        let phase_y = apply_z_string_and_main(&mut s_y, q, b'Y');
                        *next.entry(s_y).or_insert(Complex64::new(0.0, 0.0)) +=
                            c * coeff_y * phase_y;
                    }
                    current = next;
                }

                // Accumulate term contribution into the thread-local map
                for (ps, c) in current.into_iter() {
                    let w = c * term_coeff;
                    if w.re.abs() > tol || w.im.abs() > tol {
                        *acc_local.entry(ps).or_insert(Complex64::new(0.0, 0.0)) += w;
                    }
                }
                acc_local
            },
        )
        .reduce(HashMap::<Vec<u8>, Complex64>::new, |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(Complex64::new(0.0, 0.0)) += v;
            }
            a
        });

    // Convert dense Pauli strings to sparse (pauli_byte, qubit) pairs for QubitOperator.
    // Must sort by (pauli_byte, qubit) to match the canonical PauliTerm form used by
    // QubitOperator internally (same ordering as extract_term and mul_pauli_terms).
    let mut coeffs: HashMap<Vec<(u8, i32)>, Complex64> = HashMap::with_capacity(acc.len());
    for (ps_bytes, w) in acc.into_iter() {
        if w.re.abs() <= tol && w.im.abs() <= tol {
            continue;
        }
        let mut sparse: Vec<(u8, i32)> = ps_bytes
            .iter()
            .enumerate()
            .filter(|(_, &ch)| ch != b'I')
            .map(|(q, &ch)| (ch, q as i32))
            .collect();
        sparse.sort_unstable();
        *coeffs.entry(sparse).or_insert(Complex64::new(0.0, 0.0)) += w;
    }

    QubitOperator::from_coeffs(coeffs)
}
