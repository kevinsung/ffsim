# Installation

ffsim is supported directly on Linux and macOS.

ffsim is not supported directly on Windows. Windows users have two main options:

- Use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/). WSL provides a Linux environment where ffsim can be pip installed from PyPI or from source.
- Use ffsim within Docker. See [Use within Docker](#use-within-docker).

## Pip install

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). You can install it by running

```bash
pip install ffsim
```

For improved performance on [x86](https://en.wikipedia.org/wiki/X86) systems, considering [installing from source](#install-from-source).

### GPU acceleration

If you have an NVIDIA GPU, you can install the extra matching your CUDA version:

```bash
pip install "ffsim[cuda12]"  # for CUDA 12
pip install "ffsim[cuda13]"  # for CUDA 13
```

The GPU wheels are only published for Linux. Your GPU must also be supported by the CUDA version you choose; for example, CUDA 13 dropped support for Maxwell, Pascal, and Volta GPUs, so those require the `cuda12` extra.

These extras enable GPU acceleration in two independent parts of ffsim.

**State vector simulation.** The extras install [CuPy](https://cupy.dev/), which ffsim uses to run circuit simulation on the GPU: the gate application functions, `apply_unitary`, and Trotter time evolution. To use it, transfer your state vector to the GPU with `cupy.asarray`; ffsim dispatches on the array type, so no other changes to your code are required. The speedup grows with the size of the state vector, and small systems may run faster on the CPU, where the cost is dominated by kernel launch overhead. To force the CPU path for comparison, keep your state vector as a Numpy array. See [How to simulate on a GPU with CUDA](how-to-guides/gpu-simulation.md) for details and for the list of supported operations.

**Linear algebra.** Some functions in ffsim, such as orbital optimization and the compressed double factorization, are implemented with [JAX](https://docs.jax.dev/), which uses the CPU by default. The extras install a CUDA-enabled JAX, and no changes to your code are required, because JAX selects the GPU automatically once the plugin is present. Here too the speedup grows with the number of orbitals, and is negligible below roughly 16 orbitals, where the cost is dominated by kernel launch overhead rather than by the linear algebra. To force the CPU path for comparison, set the environment variable `JAX_PLATFORMS=cpu`.

By default, JAX preallocates a large fraction of the GPU's memory when it initializes its CUDA backend, which leaves less available for state vectors. If you use both parts together, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` so that JAX allocates only what it needs.

## Install from source

You can use pip to install ffsim from source. For example:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
pip install .
```

Installing from source may improve performance on x86 systems because the Rust extensions in the PyPI wheels are compiled with `-C target-cpu=x86-64`, which targets the baseline x86-64 instruction set for broad compatibility. When you build from source, ffsim is configured to compile its Rust extensions with `-C target-cpu=native`, so the Rust compiler can emit optimized instructions (e.g., AVX2, AVX-512) for your specific CPU.

Similarly, you can install [PySCF](https://pyscf.org/) from source with `-DBUILD_MARCH_NATIVE=ON` to enable CPU-specific optimizations in PySCF's C extensions. See [PySCF's installation instructions](https://pyscf.org/user/install.html#build-from-source) for details.

## Use within Docker

We provide a [Dockerfile](https://github.com/qiskit-community/ffsim/blob/main/Dockerfile) and a [compose.yaml](https://github.com/qiskit-community/ffsim/blob/main/compose.yaml) file, which you can use to build a [Docker](https://www.docker.com/) image with just a few simple commands:

```bash
git clone https://github.com/qiskit-community/ffsim.git
cd ffsim
docker compose build
docker compose up
```

Depending on your system configuration, you may need to type `sudo` before each `docker` command.

Once the container is running, navigate to <http://localhost:58888> in a web browser to access the Jupyter Notebook interface.

The home directory includes a subdirectory named `persistent-volume`. All work you’d like to save should be placed in this directory, as it is the only one that will be saved across different container runs.
