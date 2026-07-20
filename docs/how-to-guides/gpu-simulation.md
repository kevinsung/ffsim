# How to simulate on a GPU with CUDA

ffsim can run state vector simulations on NVIDIA GPUs using
[CuPy](https://cupy.dev/). GPU support covers the circuit simulation path:
gate application functions (`apply_orbital_rotation`, `apply_num_op_sum_evolution`,
`apply_diag_coulomb_evolution`, and the basic gates), `apply_unitary`
(including unitary cluster Jastrow operators), and Trotter time evolution.

## Installation

GPU support requires CuPy, which is an optional dependency. Install ffsim with
the extra matching your CUDA version:

```bash
pip install "ffsim[cuda12]"  # CUDA 12.x
pip install "ffsim[cuda13]"  # CUDA 13.x
```

or install the corresponding CuPy package (`cupy-cuda12x` or `cupy-cuda13x`)
directly. See the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html)
for details. Note that your GPU must be supported by the CUDA version you
choose; for example, CUDA 13 dropped support for Maxwell, Pascal, and Volta
GPUs, so those require the `cuda12` extra.

## Usage

To run a simulation on the GPU, convert the state vector to a CuPy array with
`cupy.asarray` and pass it to ffsim functions as usual. Functions dispatch on
the array type: given a CuPy array, they execute CUDA kernels on the GPU and
return a CuPy array. Gate parameters, such as orbital rotation matrices, remain
Numpy arrays.

```python
import cupy
import numpy as np

import ffsim

norb = 16
nelec = (8, 8)

rng = np.random.default_rng(1234)
op = ffsim.random.random_ucj_op_spin_balanced(norb, n_reps=4, seed=rng)

# Create the initial state on the CPU and transfer it to the GPU
vec = cupy.asarray(ffsim.hartree_fock_state(norb, nelec))

# The simulation runs on the GPU and returns a CuPy array
vec = ffsim.apply_unitary(vec, op, norb=norb, nelec=nelec)

# Transfer the final state back to the CPU if needed
vec_cpu = cupy.asnumpy(vec)
```

State vectors must have dtype `complex128` (the same requirement as the CPU
implementation).

## Supported operations

The following operations accept CuPy state vectors:

- `ffsim.apply_orbital_rotation`
- `ffsim.apply_num_op_sum_evolution`
- `ffsim.apply_diag_coulomb_evolution` (both the standard and Z representations)
- The basic gates: `ffsim.apply_givens_rotation`, `ffsim.apply_tunneling_interaction`,
  `ffsim.apply_num_interaction`, `ffsim.apply_num_num_interaction`,
  `ffsim.apply_on_site_interaction`, `ffsim.apply_hop_gate`, `ffsim.apply_fsim_gate`,
  and `ffsim.apply_quad_ham_evolution`
- `ffsim.apply_unitary` with any operator composed of the above, including
  the unitary cluster Jastrow (UCJ) operators
- Trotter simulation: `ffsim.simulate_trotter_diag_coulomb_split_op` and
  `ffsim.simulate_trotter_double_factorized`

In addition, the linear operator returned by `ffsim.linear_operator` for a
`MolecularHamiltonian` (or `MolecularHamiltonianSpinless`) with real-valued
tensors automatically performs its matrix-vector products on the GPU when CuPy
and a CUDA device are available. This linear operator takes and returns Numpy
arrays, so code that uses it (for example, time evolution via
`scipy.sparse.linalg.expm_multiply`) speeds up without any changes. The GPU is
not used for Hamiltonians with complex-valued tensors, or when the computation
does not fit in GPU memory; in those cases the computation falls back to the
CPU.

Other operations, such as other operator contractions, expectation values,
sampling, and the Qiskit integration, currently run only on the CPU. Transfer
the state vector back with `cupy.asnumpy` to use them.
