# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The MolecularData class."""

from __future__ import annotations

import bz2
import dataclasses
import gzip
import lzma
import os
from collections.abc import Iterable
from typing import Callable

import numpy as np
import orjson
import pyscf
import pyscf.cc
import pyscf.mcscf
import pyscf.mp
import pyscf.symm
import pyscf.tools
from typing_extensions import deprecated

from ffsim.hamiltonians import MolecularHamiltonian


@dataclasses.dataclass
class MolecularData:
    """Class for storing molecular data.

    Attributes:
        core_energy (float): The core energy.
        one_body_integrals (np.ndarray): The one-body integrals.
        two_body_integrals (np.ndarray): The two-body integrals in compressed format.
        norb (int): The number of spatial orbitals.
        nelec (tuple[int, int]): The number of alpha and beta electrons.
        atom (list[tuple[str, tuple[float, float, float]]] | None): The coordinates of
            the atoms in the molecule.
        basis (str | None): The basis set, e.g. "sto-6g".
        spin (int | None): The spin of the molecule.
        symmetry (str | None): The symmetry of the molecule.
        mo_coeff (np.ndarray | None): Molecular orbital coefficients in the AO basis.
        mo_occ (np.ndarray | None): Molecular orbital occupancies.
        active_space (list[int] | None): The molecular orbitals included in the active
            space.
        hf_energy (float | None): The Hartree-Fock energy.
        hf_mo_coeff (np.ndarray | None): Hartree-Fock canonical orbital coefficients in
            the AO basis.
        hf_mo_occ (np.ndarray | None): Hartree-Fock canonical orbital occupancies.
        mp2_energy (float | None): The MP2 energy.
        mp2_t2 (np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None): The
            MP2 t2 amplitudes.
        ccsd_energy (float | None): The CCSD energy.
        ccsd_t1 (np.ndarray | tuple[np.ndarray, np.ndarray] | None): The CCSD t1
            amplitudes.
        ccsd_t2 (np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None): The
            CCSD t2 amplitudes.
        fci_energy (float | None): The FCI energy.
        fci_vec (np.ndarray | None): The FCI state vector.
        dipole_integrals (np.ndarray | None): The dipole integrals.
        orbital_symmetries (list[str] | None): The orbital symmetries.
    """

    # Molecular integrals
    core_energy: float
    one_body_integrals: np.ndarray
    two_body_integrals: np.ndarray
    # Number of orbitals and numbers of alpha and beta electrons
    norb: int
    nelec: tuple[int, int]
    # Molecule information corresponding to attributes of pyscf.gto.Mole
    atom: list[tuple[str, tuple[float, float, float]]] | None = None
    basis: str | None = None
    spin: int | None = None
    symmetry: str | None = None
    # active space information
    mo_coeff: np.ndarray | None = None
    mo_occ: np.ndarray | None = None
    active_space: list[int] | None = None
    # Hartree-Fock data
    hf_energy: float | None = None
    hf_mo_coeff: np.ndarray | None = None
    hf_mo_occ: np.ndarray | None = None
    # MP2 data
    mp2_energy: float | None = None
    mp2_t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # CCSD data
    ccsd_energy: float | None = None
    ccsd_t1: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
    ccsd_t2: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    # FCI data
    fci_energy: float | None = None
    fci_vec: np.ndarray | None = None
    # other information
    dipole_integrals: np.ndarray | None = None
    orbital_symmetries: list[str] | None = None

    @property
    def hamiltonian(self) -> MolecularHamiltonian:
        """The Hamiltonian defined by the molecular data."""
        return MolecularHamiltonian(
            one_body_tensor=self.one_body_integrals,
            two_body_tensor=pyscf.ao2mo.restore(1, self.two_body_integrals, self.norb),
            constant=self.core_energy,
        )

    @property
    def mole(self) -> pyscf.gto.Mole:
        """The PySCF Mole class for this molecular data."""
        mol = pyscf.gto.Mole()
        return mol.build(
            atom=self.atom,
            basis=self.basis,
            spin=self.spin,
            symmetry=self.symmetry,
        )

    @staticmethod
    def from_fcidump(file: str | bytes | os.PathLike) -> MolecularData:
        """Initialize a MolecularData from an FCIDUMP file.

        Args:
            file: The FCIDUMP file path.
        """
        data = pyscf.tools.fcidump.read(file, verbose=False)
        n_electrons = data["NELEC"]
        spin = data["MS2"]
        n_alpha = (n_electrons + spin) // 2
        n_beta = (n_electrons - spin) // 2
        return MolecularData(
            core_energy=data["ECORE"],
            one_body_integrals=data["H1"],
            two_body_integrals=data["H2"],
            norb=data["NORB"],
            nelec=(n_alpha, n_beta),
            spin=spin,
        )

    @staticmethod
    def from_scf(
        hartree_fock: pyscf.scf.hf.SCF, active_space: Iterable[int] | None = None
    ) -> "MolecularData":
        """Initialize a MolecularData object from a Hartree-Fock calculation.

        Args:
            hartree_fock: The Hartree-Fock object.
            active_space: An optional list of orbitals to use for the active space.
        """
        if not hartree_fock.e_tot:
            hartree_fock = hartree_fock.run()
        hf_energy = hartree_fock.e_tot

        mol: pyscf.gto.Mole = hartree_fock.mol

        # Get core energy and one- and two-body integrals.
        if active_space is None:
            norb = mol.nao_nr()
            active_space = range(norb)
        active_space = list(active_space)
        norb = len(active_space)
        n_electrons = int(sum(hartree_fock.mo_occ[active_space]))
        n_alpha = (n_electrons + mol.spin) // 2
        n_beta = (n_electrons - mol.spin) // 2
        cas = pyscf.mcscf.CASCI(hartree_fock, norb, (n_alpha, n_beta))
        mo = cas.sort_mo(active_space, base=0)
        one_body_tensor, core_energy = cas.get_h1cas(mo)
        two_body_integrals = cas.get_h2cas(mo)

        # Get dipole integrals.
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        nuc_charge_center = np.einsum("z,zx->x", charges, coords) / charges.sum()
        with mol.with_common_origin(nuc_charge_center):
            dipole_integrals = mol.intor("cint1e_r_sph", comp=3)
            mo_coeffs = hartree_fock.mo_coeff[:, active_space]
            dipole_integrals = np.einsum(
                "xij,ip,jq->xpq", dipole_integrals, mo_coeffs, mo_coeffs
            )

        # Get orbital symmetries.
        orbsym = None
        if mol.symmetry:
            orbsym = list(
                pyscf.symm.label_orb_symm(
                    mol,
                    mol.irrep_name,
                    mol.symm_orb,
                    hartree_fock.mo_coeff[:, active_space],
                )
            )

        return MolecularData(
            core_energy=core_energy,
            one_body_integrals=one_body_tensor,
            two_body_integrals=two_body_integrals,
            norb=norb,
            nelec=(n_alpha, n_beta),
            atom=mol.atom,
            basis=mol.basis,
            spin=mol.spin,
            symmetry=mol.symmetry or None,
            mo_coeff=hartree_fock.mo_coeff,
            mo_occ=hartree_fock.mo_occ,
            active_space=active_space,
            hf_energy=hf_energy,
            dipole_integrals=dipole_integrals,
            orbital_symmetries=orbsym,
        )

    @staticmethod
    @deprecated(
        "The from_mole method is deprecated. Instead, pass an SCF object directly to "
        "from_scf."
    )
    def from_mole(
        molecule: pyscf.gto.Mole,
        active_space: Iterable[int] | None = None,
        scf_func=pyscf.scf.RHF,
    ) -> "MolecularData":
        """Initialize a MolecularData object from a PySCF molecule.

        Args:
            molecule: The molecule.
            active_space: An optional list of orbitals to use for the active space.
            scf_func: The PySCF SCF function to use for the Hartree-Fock calculation.
        """
        hartree_fock = scf_func(molecule)
        hartree_fock.run()
        return MolecularData.from_scf(hartree_fock, active_space=active_space)

    def run_fci(self, *, store_fci_vec: bool = False) -> None:
        """Run FCI and store results."""
        if self.mo_coeff is None or self.mo_occ is None or self.active_space is None:
            raise RuntimeError(
                "The run_fci method requires the mo_coeff, mo_occ, and active_space "
                "attributes to be set to non-None values."
            )
        n_alpha, n_beta = self.nelec
        scf_func = pyscf.scf.RHF if n_alpha == n_beta else pyscf.scf.ROHF
        scf = scf_func(self.mole)
        scf.mo_coeff = self.mo_coeff
        scf.mo_occ = self.mo_occ
        cas = pyscf.mcscf.CASCI(scf, ncas=self.norb, nelecas=self.nelec)
        mo = cas.sort_mo(self.active_space, mo_coeff=self.mo_coeff, base=0)
        _, _, fci_vec, _, _ = cas.kernel(mo_coeff=mo)
        self.fci_energy = cas.e_tot
        if store_fci_vec:
            self.fci_vec = fci_vec

    def run_mp2(self, *, store_t2: bool = False):
        """Run MP2 and store results."""
        if self.mo_coeff is None or self.mo_occ is None or self.active_space is None:
            raise RuntimeError(
                "The run_mp2 method requires the mo_coeff, mo_occ, and active_space "
                "attributes to be set to non-None values."
            )
        n_alpha, n_beta = self.nelec
        scf_func = pyscf.scf.RHF if n_alpha == n_beta else pyscf.scf.ROHF
        scf = scf_func(self.mole)
        cas = pyscf.mcscf.CASCI(scf, ncas=self.norb, nelecas=self.nelec)
        mo = cas.sort_mo(self.active_space, mo_coeff=self.mo_coeff, base=0)
        frozen = [i for i in range(self.norb) if i not in self.active_space]
        mo_coeff: np.ndarray | tuple[np.ndarray, np.ndarray]
        mo_occ: np.ndarray | tuple[np.ndarray, np.ndarray]
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        if n_alpha != n_beta:
            mo_coeff = (mo_coeff, mo_coeff)
            if n_alpha > n_beta:
                mo_occ_a = mo_occ > 0
                mo_occ_b = mo_occ == 2
            else:
                mo_occ_a = mo_occ == 2
                mo_occ_b = mo_occ > 0
            mo_occ = (mo_occ_a, mo_occ_b)
            mo = (mo, mo)
        mp2_solver = pyscf.mp.MP2(
            scf, frozen=frozen or None, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        _, mp2_t2 = mp2_solver.kernel(mo_coeff=mo)
        self.mp2_energy = mp2_solver.e_tot
        if store_t2:
            self.mp2_t2 = mp2_t2

    def run_ccsd(
        self,
        t1: np.ndarray | None = None,
        t2: np.ndarray | None = None,
        *,
        store_t1: bool = False,
        store_t2: bool = False,
    ) -> None:
        """Run CCSD and store results."""
        if self.mo_coeff is None or self.mo_occ is None or self.active_space is None:
            raise RuntimeError(
                "The run_ccsd method requires the mo_coeff, mo_occ, and active_space "
                "attributes to be set to non-None values."
            )
        n_alpha, n_beta = self.nelec
        scf_func = pyscf.scf.RHF if n_alpha == n_beta else pyscf.scf.ROHF
        scf = scf_func(self.mole)
        frozen = [i for i in range(self.norb) if i not in self.active_space]
        mo_coeff: np.ndarray | tuple[np.ndarray, np.ndarray]
        mo_occ: np.ndarray | tuple[np.ndarray, np.ndarray]
        mo_coeff = self.mo_coeff
        mo_occ = self.mo_occ
        if n_alpha != n_beta:
            mo_coeff = (mo_coeff, mo_coeff)
            if n_alpha > n_beta:
                mo_occ_a = mo_occ > 0
                mo_occ_b = mo_occ == 2
            else:
                mo_occ_a = mo_occ == 2
                mo_occ_b = mo_occ > 0
            mo_occ = (mo_occ_a, mo_occ_b)
        ccsd_solver = pyscf.cc.CCSD(
            scf, frozen=frozen or None, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        _, ccsd_t1, ccsd_t2 = ccsd_solver.kernel(t1=t1, t2=t2)
        self.ccsd_energy = ccsd_solver.e_tot
        if store_t1:
            self.ccsd_t1 = ccsd_t1
        if store_t2:
            self.ccsd_t2 = ccsd_t2

    def to_json(
        self, file: str | bytes | os.PathLike, compression: str | None = None
    ) -> None:
        """Serialize to JSON format, optionally compressed, and save to disk.

        Args:
            file: The file path to save to.
            compression: The optional compression algorithm to use.
                Options: ``"gzip"``, ``"bz2"``, ``"lzma"``.
        """

        def default(obj):
            if isinstance(obj, np.ndarray):
                return np.ascontiguousarray(obj)
            raise TypeError

        open_func: dict[str | None, Callable] = {
            None: open,
            "gzip": gzip.open,
            "bz2": bz2.open,
            "lzma": lzma.open,
        }
        with open_func[compression](file, "wb") as f:
            f.write(
                orjson.dumps(self, option=orjson.OPT_SERIALIZE_NUMPY, default=default)
            )

    @staticmethod
    def from_json(
        file: str | bytes | os.PathLike, compression: str | None = None
    ) -> MolecularData:
        """Load a MolecularData from a (possibly compressed) JSON file.

        Args:
            file: The file path to read from.
            compression: The compression algorithm, if any, which was used to compress
                the file.
                Options: ``"gzip"``, ``"bz2"``, ``"lzma"``.

        Returns: The MolecularData object.
        """
        open_func: dict[str | None, Callable] = {
            None: open,
            "gzip": gzip.open,
            "bz2": bz2.open,
            "lzma": lzma.open,
        }
        with open_func[compression](file, "rb") as f:
            data = orjson.loads(f.read())

        def as_array_or_none(val):
            if val is None:
                return None
            return np.asarray(val)

        def as_array_tuple_or_none(val):
            if val is None:
                return None
            return tuple(np.asarray(arr) for arr in val)

        nelec = tuple(data["nelec"])
        n_alpha, n_beta = nelec
        arrays_func = as_array_or_none if n_alpha == n_beta else as_array_tuple_or_none
        atom = data.get("atom")
        if atom is not None:
            atom = [(element, tuple(coordinates)) for element, coordinates in atom]

        return MolecularData(
            core_energy=data["core_energy"],
            one_body_integrals=np.asarray(data["one_body_integrals"]),
            two_body_integrals=np.asarray(data["two_body_integrals"]),
            norb=data["norb"],
            nelec=nelec,
            atom=atom,
            basis=data.get("basis"),
            spin=data.get("spin"),
            symmetry=data.get("symmetry"),
            mo_coeff=as_array_or_none(data.get("mo_coeff")),
            mo_occ=as_array_or_none(data.get("mo_occ")),
            active_space=data.get("active_space"),
            hf_energy=data.get("hf_energy"),
            hf_mo_coeff=as_array_or_none(data.get("hf_mo_coeff")),
            hf_mo_occ=as_array_or_none(data.get("hf_mo_occ")),
            mp2_energy=data.get("mp2_energy"),
            mp2_t2=arrays_func(data.get("mp2_t2")),
            ccsd_energy=data.get("ccsd_energy"),
            ccsd_t1=arrays_func(data.get("ccsd_t1")),
            ccsd_t2=arrays_func(data.get("ccsd_t2")),
            fci_energy=data.get("fci_energy"),
            fci_vec=as_array_or_none(data.get("fci_vec")),
            dipole_integrals=as_array_or_none(data.get("dipole_integrals")),
            orbital_symmetries=data.get("orbital_symmetries"),
        )
