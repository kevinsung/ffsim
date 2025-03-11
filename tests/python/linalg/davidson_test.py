# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Davidson diagonalization method."""

import numpy as np
import pyscf
import pyscf.mcscf

import ffsim


def test_n2():
    # Build N2 molecule
    mol = pyscf.gto.Mole()
    mol.build(
        atom=[["N", (0, 0, 0)], ["N", (1.0, 0, 0)]],
        basis="6-31g",
        # symmetry="Dooh",
    )

    # Define active space
    n_frozen = 6
    active_space = range(n_frozen, mol.nao_nr())

    # Get molecular data and Hamiltonian
    scf = pyscf.scf.RHF(mol).run()
    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(scf, norb, (n_alpha, n_beta))
    mo = cas.sort_mo(active_space, base=0)
    one_body_tensor, core_energy = cas.get_h1cas(mo)
    two_body_integrals = cas.get_h2cas(mo)
    two_body_tensor = pyscf.ao2mo.restore(1, two_body_integrals, norb)

    # Run FCI using PySCF
    cas = pyscf.mcscf.CASCI(scf, ncas=norb, nelecas=nelec)
    fci_energy, _, _, _, _ = cas.kernel()

    # Run Davidson
    eigs, vecs = ffsim.linalg.davidson1(
        one_body_tensor, two_body_tensor, norb=norb, nelec=nelec, nroots=2
    )

    # Check
    np.testing.assert_allclose(eigs[0] + core_energy, fci_energy)
    linop = ffsim.linear_operator(
        ffsim.MolecularHamiltonian(
            one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
        ),
        norb=norb,
        nelec=nelec,
    )
    for eig, vec in zip(eigs, vecs.T):
        np.testing.assert_allclose(linop @ vec, eig * vec, atol=1e-8, rtol=1e-5)
