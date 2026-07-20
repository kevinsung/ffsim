# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the GPU two-body linear operator contraction."""

from __future__ import annotations

import numpy as np
import pytest

import ffsim
from ffsim.contract import two_body

cupy = pytest.importorskip("cupy")

try:
    _n_devices = cupy.cuda.runtime.getDeviceCount()
except Exception:
    _n_devices = 0
if not _n_devices:
    pytest.skip("No CUDA device available", allow_module_level=True)

RNG = np.random.default_rng(59209586123415316441558334635341785206)


def _cpu_matvec(linop_factory, vec, monkeypatch: pytest.MonkeyPatch):
    """Apply a linear operator built with the GPU path disabled."""
    with monkeypatch.context() as patch:
        patch.setattr(two_body, "gpu_available", lambda: False)
        return linop_factory() @ vec


def test_two_body_contraction_matches_pyscf():
    """Test the GPU contraction against pyscf's contract_2e directly."""
    from pyscf.fci.direct_spin1 import absorb_h1e, contract_2e

    from ffsim._cistring import gen_linkstr_index_trilidx
    from ffsim._gpu.contract.two_body import TwoBodyContraction

    assert two_body.gpu_available()

    norb = 5
    nelec = (3, 2)
    n_alpha, n_beta = nelec
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    link_index = (
        gen_linkstr_index_trilidx(range(norb), n_alpha),
        gen_linkstr_index_trilidx(range(norb), n_beta),
    )
    absorbed = absorb_h1e(
        hamiltonian.one_body_tensor, hamiltonian.two_body_tensor, norb, nelec, 0.5
    )
    contraction = TwoBodyContraction(absorbed, norb, link_index)
    dim = ffsim.dim(norb, nelec)
    for dtype in [float, complex]:
        vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=dtype)
        expected = contract_2e(absorbed, vec, norb, nelec, link_index)
        np.testing.assert_allclose(contraction(vec), expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nelec", ffsim.testing.generate_norb_nelec(exhaustive=False)
)
def test_molecular_hamiltonian_spinful(
    norb: int, nelec: tuple[int, int], monkeypatch: pytest.MonkeyPatch
):
    """Test matvec matches the CPU result, spinful."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)

    def linop_factory():
        return ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)

    result = linop_factory() @ vec
    expected = _cpu_matvec(linop_factory, vec, monkeypatch)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize(
    "norb, nocc", ffsim.testing.generate_norb_nocc(exhaustive=False)
)
def test_molecular_hamiltonian_spinless(
    norb: int, nocc: int, monkeypatch: pytest.MonkeyPatch
):
    """Test matvec matches the CPU result, spinless."""
    hamiltonian = ffsim.random.random_molecular_hamiltonian_spinless(
        norb, seed=RNG, dtype=float
    )
    dim = ffsim.dim(norb, nocc)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)

    def linop_factory():
        return ffsim.linear_operator(hamiltonian, norb=norb, nelec=nocc)

    result = linop_factory() @ vec
    expected = _cpu_matvec(linop_factory, vec, monkeypatch)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_real_vector(monkeypatch: pytest.MonkeyPatch):
    """Test contracting with a real state vector."""
    norb = 5
    nelec = (3, 2)
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG, dtype=float)

    def linop_factory():
        return ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)

    result = linop_factory() @ vec
    expected = _cpu_matvec(linop_factory, vec, monkeypatch)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_expm_multiply_taylor(monkeypatch: pytest.MonkeyPatch):
    """Test time evolution via expm_multiply_taylor matches the CPU result."""
    norb = 4
    nelec = (2, 2)
    time = 0.01
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG, dtype=float)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)

    def evolve():
        linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
        return ffsim.linalg.expm_multiply_taylor(-1j * time * linop, vec)

    result = evolve()
    with monkeypatch.context() as patch:
        patch.setattr(two_body, "gpu_available", lambda: False)
        expected = evolve()
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_complex_tensor_falls_back_to_cpu():
    """Test that complex-valued Hamiltonians still work (CPU path)."""
    norb = 4
    nelec = (2, 2)
    hamiltonian = ffsim.random.random_molecular_hamiltonian(norb, seed=RNG)
    assert np.iscomplexobj(hamiltonian.two_body_tensor)
    dim = ffsim.dim(norb, nelec)
    vec = ffsim.random.random_state_vector(dim, seed=RNG)
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    result = linop @ vec
    assert result.shape == (dim,)
    assert np.linalg.norm(result)
