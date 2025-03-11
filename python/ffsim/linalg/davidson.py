# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Wrapper for PySCF's Davidson diagonalization method."""

import numpy as np
from pyscf.fci.direct_nosym import absorb_h1e, contract_2e, make_hdiag
from pyscf.fci.direct_spin1 import get_init_guess
from pyscf.lib import davidson1 as pyscf_davidson1
from pyscf.lib import make_diag_precond

from ffsim.cistring import gen_linkstr_index


def davidson1(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    nroots: int = 1,
    level_shift: float = 1e-3,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper for PySCF's Davidson diagonalization method.

    Use ``kwargs`` to specify additional keyword arguments to PySCF's ``davidson1``
    method. See
    https://pyscf.org/pyscf_api_docs/pyscf.lib.html#pyscf.lib.linalg_helper.davidson1

    Args:
        one_body_tensor: The one-body tensor.
        two_body_tensor: The two-body tensor.
        norb: The number of spatial orbitals.
        nelec: The numbers of alpha and beta electrons.
        nroots: The number of eigenvalue-eigenvector pairs to compute.
        level_shift: The level shift.
        kwargs:
    """
    n_alpha, n_beta = nelec
    linkstr_index_a = gen_linkstr_index(range(norb), n_alpha)
    linkstr_index_b = gen_linkstr_index(range(norb), n_beta)
    link_index = (linkstr_index_a, linkstr_index_b)
    two_body = absorb_h1e(one_body_tensor, two_body_tensor, norb, nelec, 0.5)
    hdiag = make_hdiag(one_body_tensor, two_body_tensor, norb=norb, nelec=nelec)
    x0 = get_init_guess(norb, nelec, nroots=nroots, hdiag=hdiag)[0]
    precond = make_diag_precond(hdiag, level_shift=level_shift)
    converged, eigs, vecs = pyscf_davidson1(
        lambda xs: [
            contract_2e(two_body, x, norb, nelec, link_index=link_index) for x in xs
        ],
        x0=x0,
        precond=precond,
        nroots=nroots,
        **kwargs,
    )
    if not all(converged):
        raise RuntimeError("Davidson method failed to converge.")
    return eigs, np.stack(vecs, axis=1)
