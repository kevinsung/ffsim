# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary coupled cluster, singles and doubles ansatz, Trotterized in Jastrow form."""

from __future__ import annotations

import itertools
from dataclasses import InitVar, dataclass, field
from typing import Any, cast

import numpy as np
import scipy.linalg

from ffsim import gates, linalg, protocols
from ffsim.linalg.double_factorized_decomposition import reconstruct_t2
from ffsim.linalg.util import unitary_from_parameters, unitary_to_parameters
from ffsim.trotter._util import simulate_trotter_step_iterator


@dataclass(frozen=True)
class UCCSDJastrowOpRestrictedReal(
    protocols.SupportsApplyUnitary, protocols.SupportsApproximateEquality
):
    r"""Real-valued restricted UCCSD ansatz, Trotterized in Jastrow form.

    This class implements a Trotterized approximation to the restricted unitary coupled
    cluster singles and doubles (UCCSD) ansatz using the Jastrow decomposition
    structure.

    The restricted UCCSD operator is :math:`e^{\mathcal{T} - \mathcal{T}^\dagger}` where
    the cluster operator :math:`\mathcal{T}` contains single and double excitations:

    .. math::

        \mathcal{T} = \sum_{ia} t_{ia} a^\dagger_i a_a
        + \sum_{ijab} t_{ijab} a^\dagger_i a^\dagger_j a_b a_a.

    The "restricted" convention means that alpha and beta electrons share the same
    spatial orbitals and the same t-amplitudes. "Real-valued" means that the
    t-amplitudes are real. Note that the ``final_orbital_rotation``, if included, is
    allowed to be complex-valued.

    This operator approximates the UCCSD operator by Trotterizing the UCCSD generator
    and expressing each Trotter step in Jastrow form — as a sequence of orbital
    rotations interleaved with diagonal Coulomb evolutions, mirroring the structure
    of the :class:`UCJOpSpinBalanced` ansatz. The quality of the approximation is
    controlled by ``n_steps`` and ``order``: increasing either improves accuracy at the
    cost of circuit depth. The T2 amplitudes are decomposed via double factorization,
    and the T1 amplitudes contribute a one-body orbital rotation that is interleaved
    with the T2 Trotter terms. This operator is related to (but generally not
    identical to) the ansatz produced by :meth:`UCJOpSpinBalanced.from_t_amplitudes
    <ffsim.UCJOpSpinBalanced.from_t_amplitudes>`, which corresponds to a zeroth-order
    single Trotter step treating only the T2 terms.

    The double factorization of the t2 amplitudes is computed once during
    initialization and cached for use in subsequent calls to ``_apply_unitary_``.

    To support variational optimization of the orbital basis, an optional final
    orbital rotation can be included in the operator, to be performed at the end.

    Attributes:
        t1 (np.ndarray): The t1 amplitudes, as a real-valued Numpy array of shape
            ``(nocc, nvrt)``.
        t2 (np.ndarray): The t2 amplitudes, as a real-valued Numpy array of shape
            ``(nocc, nocc, nvrt, nvrt)``.
        n_steps (int): The number of Trotter steps. A larger number of steps
            yields a more accurate approximation of the UCCSD operator but also
            increases circuit depth.
        order (int): The order of the Trotter-Suzuki decomposition formula.
            Order 0 corresponds to the asymmetric (Lie-Trotter) formula;
            higher even orders correspond to higher-order Suzuki formulas with
            improved accuracy.
        final_orbital_rotation (np.ndarray | None): The optional final orbital
            rotation, as a Numpy array of shape ``(norb, norb)``.
        tol (float): Tolerance for truncating small eigenvalues in the
            double-factorized decomposition of the t2 amplitudes.

    Args:
        max_terms: An optional upper bound on the number of terms in the
            double-factorized decomposition of the t2 amplitudes. See
            :func:`ffsim.linalg.double_factorized_t2` for details.
        optimize: Whether to optimize the double-factorized decomposition of the t2
            amplitudes to minimize the decomposition error. See
            :func:`ffsim.linalg.double_factorized_t2` for details.
        method: The optimization method. See the documentation of
            `scipy.optimize.minimize`_ for possible values.
            This argument is ignored if ``optimize`` is set to ``False``.
        callback: Callback function for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if ``optimize`` is set to ``False``.
        options: Options for the optimization. See the documentation of
            `scipy.optimize.minimize`_ for usage.
            This argument is ignored if ``optimize`` is set to ``False``.
        diag_coulomb_indices: Indices of diagonal Coulomb matrix entries that are
            allowed to be nonzero. If ``None``, all entries are allowed. Each index
            pair must be upper triangular, i.e., of the form :math:`(i, j)` where
            :math:`i \leq j`. See :func:`ffsim.linalg.double_factorized_t2` for
            details. This argument is ignored if ``optimize`` is set to ``False``.
        regularization: See :func:`ffsim.linalg.double_factorized_t2` for a
            description of this argument.
            This argument is ignored if ``optimize`` is set to ``False``.
        multi_stage_start: See :func:`ffsim.linalg.double_factorized_t2` for a
            description of this argument.
            This argument is ignored if ``optimize`` is set to ``False``.
        multi_stage_step: See :func:`ffsim.linalg.double_factorized_t2` for a
            description of this argument.
            This argument is ignored if ``optimize`` is set to ``False``.
        validate: Whether to validate the operator attributes. Setting this to False
            skips validation, which is useful if you need to create many instances
            of this class and are confident that the attributes are valid.
        rtol: Relative numerical tolerance for validation checks.
        atol: Absolute numerical tolerance for validation checks.

    .. _scipy.optimize.minimize: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    t1: np.ndarray  # shape: (nocc, nvrt)
    t2: np.ndarray  # shape: (nocc, nocc, nvrt, nvrt)
    n_steps: int = 1
    order: int = 0
    final_orbital_rotation: np.ndarray | None = None  # shape: (norb, norb)
    tol: float = 1e-8
    max_terms: InitVar[int | None] = None
    optimize: InitVar[bool] = False
    method: InitVar[str] = "L-BFGS-B"
    callback: InitVar[Any] = None
    options: InitVar[dict | None] = None
    diag_coulomb_indices: InitVar[list[tuple[int, int]] | None] = None
    regularization: InitVar[float] = 0
    multi_stage_start: InitVar[int | None] = None
    multi_stage_step: InitVar[int | None] = None
    validate: InitVar[bool] = True
    rtol: InitVar[float] = 1e-5
    atol: InitVar[float] = 1e-8
    diag_coulomb_mats: np.ndarray = field(init=False, repr=False, compare=False)
    orbital_rotations: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(
        self,
        max_terms: int | None,
        optimize: bool,
        method: str,
        callback: Any,
        options: dict | None,
        diag_coulomb_indices: list[tuple[int, int]] | None,
        regularization: float,
        multi_stage_start: int | None,
        multi_stage_step: int | None,
        validate: bool,
        rtol: float,
        atol: float,
    ):
        if np.iscomplexobj(self.t1):
            raise TypeError(
                "UCCSDJastrowOpRestrictedReal only accepts real-valued t1 amplitudes. "
                "Please pass a t1 amplitudes tensor with a real-valued data type."
            )
        if np.iscomplexobj(self.t2):
            raise TypeError(
                "UCCSDJastrowOpRestrictedReal only accepts real-valued t2 amplitudes. "
                "Please pass a t2 amplitudes tensor with a real-valued data type."
            )

        nocc, nvrt = self.t1.shape
        if validate:
            norb = nocc + nvrt
            if self.t2.shape != (nocc, nocc, nvrt, nvrt):
                raise ValueError(
                    "t2 shape not consistent with t1 shape. "
                    f"Expected {(nocc, nocc, nvrt, nvrt)} but got {self.t2.shape}"
                )
            if self.final_orbital_rotation is not None:
                if self.final_orbital_rotation.shape != (norb, norb):
                    raise ValueError(
                        "Final orbital rotation shape not consistent with t1 shape. "
                        f"Expected {(norb, norb)} but got "
                        f"{self.final_orbital_rotation.shape}"
                    )
                if not linalg.is_unitary(
                    self.final_orbital_rotation, rtol=rtol, atol=atol
                ):
                    raise ValueError("Final orbital rotation was not unitary.")

        diag_coulomb_mats, orbital_rotations = linalg.double_factorized_t2(
            self.t2,
            tol=self.tol,
            max_terms=max_terms,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            diag_coulomb_indices=diag_coulomb_indices,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )
        reconstructed = reconstruct_t2(diag_coulomb_mats, orbital_rotations, nocc=nocc)
        residual = reconstructed - self.t2
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "diag_coulomb_mats", diag_coulomb_mats)
        object.__setattr__(self, "orbital_rotations", orbital_rotations)

    @property
    def norb(self) -> int:
        """The number of spatial orbitals."""
        nocc, nvrt = self.t1.shape
        return nocc + nvrt

    @staticmethod
    def n_params(
        norb: int, nocc: int, *, with_final_orbital_rotation: bool = False
    ) -> int:
        """Return the number of parameters of an ansatz with given settings.

        Args:
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.

        Returns:
            The number of parameters of the ansatz.
        """
        nvrt = norb - nocc
        n_pairs = nocc * nvrt
        # t1 has n_pairs parameters
        # t2 has n_pairs * (n_pairs + 1) // 2 parameters (symmetric under exchange)
        # Final orbital rotation has norb**2 parameters
        return (
            n_pairs
            + n_pairs * (n_pairs + 1) // 2
            + with_final_orbital_rotation * norb**2
        )

    @staticmethod
    def from_parameters(
        params: np.ndarray,
        *,
        norb: int,
        nocc: int,
        n_steps: int = 1,
        order: int = 0,
        tol: float = 1e-8,
        with_final_orbital_rotation: bool = False,
        max_terms: int | None = None,
        diag_coulomb_indices: list[tuple[int, int]] | None = None,
        optimize: bool = False,
        method: str = "L-BFGS-B",
        callback: Any = None,
        options: dict | None = None,
        regularization: float = 0,
        multi_stage_start: int | None = None,
        multi_stage_step: int | None = None,
    ) -> UCCSDJastrowOpRestrictedReal:
        """Initialize the operator from a real-valued parameter vector.

        Args:
            params: The real-valued parameter vector.
            norb: The number of spatial orbitals.
            nocc: The number of spatial orbitals that are occupied by electrons.
            n_steps: The number of Trotter steps.
            order: The order of the Trotter-Suzuki decomposition formula.
            tol: Tolerance for truncating small eigenvalues in the double-factorized
                decomposition of the t2 amplitudes.
            with_final_orbital_rotation: Whether to include a final orbital rotation
                in the operator.
            max_terms: An optional upper bound on the number of terms in the
                double-factorized decomposition of the t2 amplitudes.
            diag_coulomb_indices: Indices of diagonal Coulomb matrix entries that are
                allowed to be nonzero. Ignored if ``optimize`` is ``False``.
            optimize: Whether to optimize the double-factorized decomposition of the
                t2 amplitudes. See :func:`ffsim.linalg.double_factorized_t2` for
                details.
            method: The optimization method. Ignored if ``optimize`` is ``False``.
            callback: Callback function for the optimization. Ignored if ``optimize``
                is ``False``.
            options: Options for the optimization. Ignored if ``optimize`` is
                ``False``.
            regularization: See :func:`ffsim.linalg.double_factorized_t2`.
                Ignored if ``optimize`` is ``False``.
            multi_stage_start: See :func:`ffsim.linalg.double_factorized_t2`.
                Ignored if ``optimize`` is ``False``.
            multi_stage_step: See :func:`ffsim.linalg.double_factorized_t2`.
                Ignored if ``optimize`` is ``False``.

        Returns:
            The operator constructed from the given parameters.

        Raises:
            ValueError: The number of parameters passed did not match the number
                expected based on the function inputs.
        """
        expected = UCCSDJastrowOpRestrictedReal.n_params(
            norb, nocc, with_final_orbital_rotation=with_final_orbital_rotation
        )
        if len(params) != expected:
            raise ValueError(
                "The number of parameters passed did not match the number expected "
                "based on the function inputs. "
                f"Expected {expected} but got {len(params)}."
            )
        nvrt = norb - nocc
        t1 = np.zeros((nocc, nvrt))
        t2 = np.zeros((nocc, nocc, nvrt, nvrt))
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            t1[i, a - nocc] = params[index]
            index += 1
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            t2[i, j, a - nocc, b - nocc] = params[index]
            t2[j, i, b - nocc, a - nocc] = params[index]
            index += 1
        # Final orbital rotation
        final_orbital_rotation = None
        if with_final_orbital_rotation:
            final_orbital_rotation = unitary_from_parameters(params[index:], norb)
        return UCCSDJastrowOpRestrictedReal(
            t1=t1,
            t2=t2,
            n_steps=n_steps,
            order=order,
            final_orbital_rotation=final_orbital_rotation,
            tol=tol,
            max_terms=max_terms,
            diag_coulomb_indices=diag_coulomb_indices,
            optimize=optimize,
            method=method,
            callback=callback,
            options=options,
            regularization=regularization,
            multi_stage_start=multi_stage_start,
            multi_stage_step=multi_stage_step,
        )

    def to_parameters(self) -> np.ndarray:
        """Convert the operator to a real-valued parameter vector.

        Returns:
            The real-valued parameter vector.
        """
        nocc, nvrt = self.t1.shape
        norb = nocc + nvrt
        n_params = UCCSDJastrowOpRestrictedReal.n_params(
            norb,
            nocc,
            with_final_orbital_rotation=self.final_orbital_rotation is not None,
        )
        params = np.zeros(n_params)
        occ_vrt_pairs = list(itertools.product(range(nocc), range(nocc, norb)))
        index = 0
        # t1
        for i, a in occ_vrt_pairs:
            params[index] = self.t1[i, a - nocc]
            index += 1
        # t2
        for (i, a), (j, b) in itertools.combinations_with_replacement(occ_vrt_pairs, 2):
            params[index] = self.t2[i, j, a - nocc, b - nocc]
            index += 1
        # Final orbital rotation
        if self.final_orbital_rotation is not None:
            params[index:] = unitary_to_parameters(self.final_orbital_rotation)
        return params

    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        nocc, _nvrt = self.t1.shape

        # One-body generator from T1 amplitudes: anti-Hermitian matrix whose
        # exponential is the T1 orbital rotation.
        one_body_tensor = np.zeros((norb, norb))
        one_body_tensor[:nocc, nocc:] = -self.t1
        one_body_tensor[nocc:, :nocc] = self.t1.T

        diag_coulomb_mats = self.diag_coulomb_mats
        orbital_rotations = self.orbital_rotations
        n_terms = len(diag_coulomb_mats)
        step_time = 1.0 / self.n_steps

        current_basis = np.eye(norb, dtype=complex)
        for _ in range(self.n_steps):
            for term_index, time in simulate_trotter_step_iterator(
                1 + n_terms, step_time, self.order
            ):
                if term_index == 0:
                    # One-body (T1) term: update orbital basis accumulator.
                    current_basis = (
                        scipy.linalg.expm(time * one_body_tensor) @ current_basis
                    )
                else:
                    orbital_rotation = orbital_rotations[term_index - 1]
                    diag_coulomb_mat = diag_coulomb_mats[term_index - 1]
                    vec = gates.apply_orbital_rotation(
                        vec,
                        orbital_rotation.T.conj() @ current_basis,
                        norb=norb,
                        nelec=nelec,
                        copy=False,
                    )
                    vec = gates.apply_diag_coulomb_evolution(
                        vec,
                        (diag_coulomb_mat, diag_coulomb_mat, diag_coulomb_mat),
                        time=-time,
                        norb=norb,
                        nelec=nelec,
                        copy=False,
                    )
                    current_basis = orbital_rotation

        if self.final_orbital_rotation is None:
            vec = gates.apply_orbital_rotation(
                vec, current_basis, norb=norb, nelec=nelec, copy=False
            )
        else:
            vec = gates.apply_orbital_rotation(
                vec,
                self.final_orbital_rotation @ current_basis,
                norb=norb,
                nelec=nelec,
                copy=False,
            )
        return vec

    def _approx_eq_(self, other, rtol: float, atol: float) -> bool:
        if isinstance(other, UCCSDJastrowOpRestrictedReal):
            if not np.allclose(self.t1, other.t1, rtol=rtol, atol=atol):
                return False
            if not np.allclose(self.t2, other.t2, rtol=rtol, atol=atol):
                return False
            if self.n_steps != other.n_steps:
                return False
            if self.order != other.order:
                return False
            if (self.final_orbital_rotation is None) != (
                other.final_orbital_rotation is None
            ):
                return False
            if self.final_orbital_rotation is not None:
                return np.allclose(
                    cast(np.ndarray, self.final_orbital_rotation),
                    cast(np.ndarray, other.final_orbital_rotation),
                    rtol=rtol,
                    atol=atol,
                )
            return True
        return NotImplemented
