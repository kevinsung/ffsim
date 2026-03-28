# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for UCCSD Jastrow ansatz."""

import dataclasses

import numpy as np
import pytest

import ffsim
from ffsim.variational.uccsd_jastrow import UCCSDJastrowOpRestrictedReal

RNG = np.random.default_rng(129464651738638437173826382716826378261)


def test_real_norb():
    """Test norb property."""
    norb = 5
    nocc = 3
    operator = ffsim.random.random_uccsd_jastrow_op_restricted_real(
        norb, nocc, seed=RNG
    )
    assert operator.norb == norb


def test_real_n_params():
    """Test n_params matches actual parameter count."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_jastrow_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        actual = UCCSDJastrowOpRestrictedReal.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        expected = len(operator.to_parameters())
        assert actual == expected


def test_real_parameters_roundtrip():
    """Test to_parameters / from_parameters roundtrip."""
    norb = 5
    nocc = 3
    for n_steps, order, with_final_orbital_rotation in [
        (1, 0, False),
        (1, 0, True),
        (2, 1, False),
        (4, 2, True),
    ]:
        operator = ffsim.random.random_uccsd_jastrow_op_restricted_real(
            norb,
            nocc,
            n_steps=n_steps,
            order=order,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = UCCSDJastrowOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            n_steps=n_steps,
            order=order,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(roundtripped, operator)


def test_real_from_parameters_wrong_length():
    """Test that from_parameters raises on wrong parameter count."""
    norb = 4
    nocc = 2
    n_params = UCCSDJastrowOpRestrictedReal.n_params(norb, nocc)
    with pytest.raises(ValueError, match="number of parameters"):
        UCCSDJastrowOpRestrictedReal.from_parameters(
            np.zeros(n_params + 1), norb=norb, nocc=nocc
        )


def test_real_approx_eq():
    """Test approximate equality."""
    norb = 5
    nocc = 3
    for with_final_orbital_rotation in [False, True]:
        operator = ffsim.random.random_uccsd_jastrow_op_restricted_real(
            norb,
            nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        roundtripped = UCCSDJastrowOpRestrictedReal.from_parameters(
            operator.to_parameters(),
            norb=norb,
            nocc=nocc,
            with_final_orbital_rotation=with_final_orbital_rotation,
        )
        assert ffsim.approx_eq(operator, roundtripped)
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t1=2 * operator.t1)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, t2=2 * operator.t2)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, n_steps=operator.n_steps + 1)
        )
        assert not ffsim.approx_eq(
            operator, dataclasses.replace(operator, order=operator.order + 2)
        )


def test_real_apply_unitary_norm_preserved():
    """Test that apply_unitary preserves the norm."""
    norb = 5
    nocc = 3
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, (nocc, nocc)), seed=RNG)
    for n_steps, order, with_final_orbital_rotation in [
        (1, 0, False),
        (1, 0, True),
        (2, 1, False),
        (4, 2, True),
    ]:
        operator = ffsim.random.random_uccsd_jastrow_op_restricted_real(
            norb,
            nocc,
            n_steps=n_steps,
            order=order,
            with_final_orbital_rotation=with_final_orbital_rotation,
            seed=RNG,
        )
        result = ffsim.apply_unitary(vec, operator, norb=norb, nelec=(nocc, nocc))
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)


def test_real_apply_unitary_converges_to_exact_uccsd():
    """Test that increasing n_steps and order converges to exact UCCSD."""
    norb = 4
    nocc = 2
    nvrt = norb - nocc
    rng = np.random.default_rng(RNG)
    t1 = rng.standard_normal((nocc, nvrt)) * 0.05
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=rng, dtype=float) * 0.05
    nelec = (nocc, nocc)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)

    exact = ffsim.apply_unitary(
        vec,
        ffsim.variational.UCCSDOpRestrictedReal(t1=t1, t2=t2),
        norb=norb,
        nelec=nelec,
    )

    # Fidelity should strictly improve as n_steps increases (order=0)
    prev_fidelity = 0.0
    for n_steps in [1, 4, 16]:
        op = UCCSDJastrowOpRestrictedReal(t1=t1, t2=t2, n_steps=n_steps)
        result = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
        fidelity = abs(np.vdot(exact, result)) ** 2
        assert fidelity > prev_fidelity
        prev_fidelity = fidelity

    # Higher-order Trotter should outperform order=0 at the same n_steps
    fidelities = {}
    for order in [0, 1, 2]:
        op = UCCSDJastrowOpRestrictedReal(t1=t1, t2=t2, n_steps=4, order=order)
        result = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
        fidelities[order] = abs(np.vdot(exact, result)) ** 2
    assert fidelities[1] > fidelities[0]
    assert fidelities[2] > fidelities[1]

    # order=2 with 4 steps should be essentially exact for small amplitudes
    np.testing.assert_allclose(fidelities[2], 1.0, atol=1e-8)


def test_real_apply_unitary_optimize():
    """Test that optimize=True produces a valid unitary and improves factorization."""
    norb = 4
    nocc = 2
    nvrt = norb - nocc
    rng = np.random.default_rng(RNG)
    t1 = rng.standard_normal((nocc, nvrt)) * 0.05
    t2 = ffsim.random.random_t2_amplitudes(norb, nocc, seed=rng, dtype=float) * 0.05
    nelec = (nocc, nocc)
    vec = ffsim.random.random_state_vector(ffsim.dim(norb, nelec), seed=rng)

    op_default = UCCSDJastrowOpRestrictedReal(t1=t1, t2=t2)
    op_optimized = UCCSDJastrowOpRestrictedReal(t1=t1, t2=t2, optimize=True)

    # Both should preserve the norm
    for op in [op_default, op_optimized]:
        result = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)
        np.testing.assert_allclose(np.linalg.norm(result), 1.0)

    # The optimized factorization should be at least as close to exact UCCSD
    exact = ffsim.apply_unitary(
        vec,
        ffsim.variational.UCCSDOpRestrictedReal(t1=t1, t2=t2),
        norb=norb,
        nelec=nelec,
    )
    fidelity_default = (
        abs(
            np.vdot(exact, ffsim.apply_unitary(vec, op_default, norb=norb, nelec=nelec))
        )
        ** 2
    )
    fidelity_optimized = (
        abs(
            np.vdot(
                exact, ffsim.apply_unitary(vec, op_optimized, norb=norb, nelec=nelec)
            )
        )
        ** 2
    )
    assert fidelity_optimized >= fidelity_default - 1e-8


def test_real_validate_complex_t1_raises():
    """Test that complex t1 raises TypeError."""
    nocc, nvrt = 2, 2
    with pytest.raises(TypeError, match="real-valued t1"):
        UCCSDJastrowOpRestrictedReal(
            t1=np.zeros((nocc, nvrt), dtype=complex),
            t2=np.zeros((nocc, nocc, nvrt, nvrt)),
        )


def test_real_validate_complex_t2_raises():
    """Test that complex t2 raises TypeError."""
    nocc, nvrt = 2, 2
    with pytest.raises(TypeError, match="real-valued t2"):
        UCCSDJastrowOpRestrictedReal(
            t1=np.zeros((nocc, nvrt)),
            t2=np.zeros((nocc, nocc, nvrt, nvrt), dtype=complex),
        )


def test_real_validate_t2_shape_mismatch_raises():
    """Test that mismatched t1/t2 shapes raise ValueError."""
    nocc, nvrt = 2, 2
    with pytest.raises(ValueError, match="t2 shape not consistent"):
        UCCSDJastrowOpRestrictedReal(
            t1=np.zeros((nocc, nvrt)),
            t2=np.zeros((nocc + 1, nocc + 1, nvrt, nvrt)),
        )


def test_real_validate_final_orbital_rotation_shape_raises():
    """Test that wrong final_orbital_rotation shape raises ValueError."""
    norb, nocc, nvrt = 4, 2, 2
    with pytest.raises(ValueError, match="Final orbital rotation shape"):
        UCCSDJastrowOpRestrictedReal(
            t1=np.zeros((nocc, nvrt)),
            t2=np.zeros((nocc, nocc, nvrt, nvrt)),
            final_orbital_rotation=np.eye(norb + 1),
        )
