# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# This file contains code derived from gpu4pyscf
# (https://github.com/pyscf/gpu4pyscf), file gpu4pyscf/fci/direct_spin1.py,
# which carries the following notice:
#
# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU contraction of a two-body operator with a state vector.

This is a CuPy port of ``pyscf.fci.direct_spin1.contract_2e``, adapted from
gpu4pyscf. Unlike the gpu4pyscf version, the device-resident data (the
two-body tensor and the CI string address tables) is uploaded once at
construction so that repeated matvecs only transfer the state vector, complex
state vectors are supported by contracting the real and imaginary parts
separately, and the address tables are uint32 with 64-bit index arithmetic
(gpu4pyscf uses uint16 tables and 32-bit arithmetic, limiting each spin sector
to 65,535 CI strings).
"""

from __future__ import annotations

import cupy  # type: ignore
import numpy as np

_TILE = 32

_CODE = r"""
#define TILE 32
extern "C" {
__global__
void _build_t1(double *ci0, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned int *addra, unsigned int *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    long long stra0 = (long long)blockIdx.y * blockDim.y;
    long long strb = strb0 + tx;
    long long stra = stra0 + ty;

    long long nab = na * TILE;
    long long ab_id = stra * TILE + tx;
    __shared__ unsigned int _addra[TILE*TILE];
    __shared__ unsigned int _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, j0, j;
    long long str1;
    int dj = TILE;
    double val;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                val = 0;
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val = sign * ci0[str1*nb+strb];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    val += sign * ci0[stra*nb+str1];
                }
                t1[(j0+j)*nab + ab_id] = val;
            }
        }
        __syncthreads();
    }
}

__global__
void _gather(double *out, double *t1,
    long long strb0, long long na, long long nb, long long nnorb,
    unsigned int *addra, unsigned int *addrb, char *signa, char *signb)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    long long stra0 = (long long)blockIdx.y * blockDim.y;
    long long strb = strb0 + tx;
    long long stra = stra0 + ty;
    long long nab = na * TILE;
    long long ab_id = stra * TILE + tx;
    __shared__ unsigned int _addra[TILE*TILE];
    __shared__ unsigned int _addrb[TILE*TILE];
    __shared__ char _signa[TILE*TILE];
    __shared__ char _signb[TILE*TILE];
    int sign, j0, j;
    long long str1;
    int dj = TILE;
    double val = 0.;

    for (j0 = 0; j0 < nnorb; j0+=TILE) {
        _addra[ty*TILE+tx] = addra[(j0+ty)*na+stra0+tx];
        _addrb[ty*TILE+tx] = addrb[(j0+ty)*nb+strb0+tx];
        _signa[ty*TILE+tx] = signa[(j0+ty)*na+stra0+tx];
        _signb[ty*TILE+tx] = signb[(j0+ty)*nb+strb0+tx];
        if (j0 + TILE > nnorb) {
            dj = nnorb - j0;
        }
        __syncthreads();
        if (stra < na && strb < nb) {
            for (j = 0; j < dj; j++) {
                sign = _signa[j*TILE+ty];
                str1 = _addra[j*TILE+ty];
                if (sign != 0) {
                    val += sign * t1[(j0+j)*nab + (str1*TILE+tx)];
                }

                sign = _signb[j*TILE+tx];
                str1 = _addrb[j*TILE+tx];
                if (sign != 0) {
                    out[stra*nb+str1] += sign * t1[(j0+j)*nab + ab_id];
                }
            }
        }
        __syncthreads();
    }
    // Guard added relative to gpu4pyscf: without it, threads with
    // stra >= na or strb >= nb perform an out-of-bounds read-modify-write
    // that races with in-bounds accumulations and corrupts the result.
    if (stra < na && strb < nb) {
        out[stra*nb+strb] += val;
    }
}
}"""

_MODULE = cupy.RawModule(code=_CODE)
_BUILD_T1 = _MODULE.get_function("_build_t1")
_GATHER = _MODULE.get_function("_gather")


def fits_in_memory(norb: int, na: int, nb: int) -> bool:
    """Return whether the contraction working set fits in free GPU memory.

    Estimates the dominant allocations (the real state vector and output, plus
    the t1 and gt1 work buffers) and compares against the memory available on
    the device, including memory cached in the CuPy pool.
    """
    nnorb = norb * (norb + 1) // 2
    required = 2 * na * nb * 8 + 2 * nnorb * na * _TILE * 8
    free, _ = cupy.cuda.runtime.memGetInfo()
    return required <= free + cupy.get_default_memory_pool().free_bytes()


def _link_index_to_addrs(
    link_index: np.ndarray, nnorb: int
) -> tuple[cupy.ndarray, cupy.ndarray]:
    """Device copies of the CI string address and sign tables for a spin sector."""
    na = link_index.shape[0]
    ia = link_index[:, :, 0].T
    addr = np.zeros((nnorb, na), dtype=np.uint32)
    sign = np.zeros((nnorb, na), dtype=np.int8)
    idx = np.arange(na)
    addr[ia, idx] = link_index[:, :, 2].T
    sign[ia, idx] = link_index[:, :, 3].T
    # Add paddings to avoid illegal address in kernel
    _addr = cupy.empty((nnorb + _TILE, na), dtype=np.uint32)[:nnorb]
    _sign = cupy.empty((nnorb + _TILE, na), dtype=np.int8)[:nnorb]
    _addr.set(addr)
    _sign.set(sign)
    return _addr, _sign


class TwoBodyContraction:
    """Contract a two-body tensor with state vectors on the GPU.

    The two-body tensor must be given in the "absorbed" form returned by
    ``pyscf.fci.direct_spin1.absorb_h1e``, with shape ``(nnorb, nnorb)`` where
    ``nnorb = norb * (norb + 1) // 2``, and the link index in the compressed
    form returned by ``gen_linkstr_index_trilidx``.
    """

    def __init__(
        self,
        two_body_tensor: np.ndarray,
        norb: int,
        link_index: tuple[np.ndarray, np.ndarray],
    ) -> None:
        nnorb = norb * (norb + 1) // 2
        assert two_body_tensor.shape == (nnorb, nnorb)
        link_index_a, link_index_b = link_index
        self._na: int = link_index_a.shape[0]
        self._nb: int = link_index_b.shape[0]
        self._nnorb = nnorb
        self._eri = cupy.asarray(two_body_tensor, dtype=cupy.float64)
        self._addra, self._signa = _link_index_to_addrs(link_index_a, nnorb)
        if link_index_b is link_index_a:
            self._addrb, self._signb = self._addra, self._signa
        else:
            self._addrb, self._signb = _link_index_to_addrs(link_index_b, nnorb)

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(vec):
            result = self._contract_real(vec.real).astype(complex)
            result.imag = self._contract_real(vec.imag)
            return result
        return self._contract_real(vec)

    def _contract_real(self, vec: np.ndarray) -> np.ndarray:
        na, nb, nnorb = self._na, self._nb, self._nnorb
        ci0 = cupy.asarray(vec, dtype=cupy.float64).reshape(na, nb)
        out = cupy.zeros_like(ci0)
        threads = (_TILE, _TILE)
        blocks = (1, (na + _TILE - 1) // _TILE)
        rest_args = (
            np.int64(na),
            np.int64(nb),
            np.int64(nnorb),
            self._addra,
            self._addrb,
            self._signa,
            self._signb,
        )
        t1 = cupy.empty((nnorb, na * _TILE))
        gt1 = cupy.empty((nnorb, na * _TILE))
        for strb0 in range(0, nb, _TILE):
            _BUILD_T1(blocks, threads, (ci0, t1, np.int64(strb0), *rest_args))
            self._eri.dot(t1, out=gt1)
            _GATHER(blocks, threads, (out, gt1, np.int64(strb0), *rest_args))
        return out.get().reshape(vec.shape)
