# SBD Python Bindings

Python bindings for the Selected Basis Diagonalization (SBD) library with dual CPU/GPU backend support.

## Overview

SBD (Selected Basis Diagonalization) is a high-performance library for quantum chemistry calculations. The Python bindings provide access to SBD's **Two-Particle Basis (TPB)** diagonalization method with support for both CPU and GPU backends.

**Key Features:**
- **TPB diagonalization** for quantum chemistry Hamiltonians
- Dual backend: CPU (OpenMP) and GPU (CUDA), switchable at runtime
- MPI parallelization
- Integration with [qiskit-addon-sqd](https://github.com/Qiskit/qiskit-addon-sqd) for SQD workflows

**Note:** The Python bindings expose the TPB method only. Other SBD methods (CAOP, GDB) are available through C++ CLI apps in `/apps`.

## Installation

### Prerequisites

**Required:** Python 3.11+, MPI (OpenMPI/MPICH), BLAS (OpenBLAS/MKL), pybind11, mpi4py, numpy.

**Optional (GPU):** NVIDIA HPC SDK (nvc++), CUDA-capable GPU, CUDA-aware MPI.

### Get the SBD C++ source

The SBD C++ headers and apps come from upstream
[r-ccs-cms/sbd](https://github.com/r-ccs-cms/sbd) via a git submodule at
`vendor/sbd-upstream/`, **pinned at a specific commit** (recorded in
`.gitmodules`; run `git submodule status` to see the current SHA).

```bash
git clone --recurse-submodules https://github.com/hfwen0502/sbd.git
# or, if you cloned without --recurse-submodules:
git submodule update --init --recursive
```

If you need a newer upstream revision (for a recently-landed GPU fix
etc.), advance the local submodule and rebuild:

```bash
git submodule update --remote vendor/sbd-upstream
pip install -e . --no-build-isolation --force-reinstall --no-deps
```

### Environment Variables

```bash
export MPI_HOME=/path/to/mpi
export BLAS_LIB_PATH=/path/to/blas/lib
export BLAS_LIBS=openblas  # or mkl_rt

# macOS: use system clang to match Python's libc++
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# NVIDIA Thrust GPU backend (optional, requires NVHPC nvc++)
export NVHPC_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers
export CC=nvc
export CXX=nvc++
# Clear these — inherited flags (e.g. from conda) may be gcc-specific
# and break nvc++ compilation
export CFLAGS=''
export CXXFLAGS=''

# OpenMP-offload GPU backend (optional, requires LLVM/clang trunk
# with the offload runtime built — see .github/SETUP_LLVM_OFFLOAD.txt).
# Auto-detected from LLVM_HOME if set; only built when invoked
# explicitly with SBD_BUILD_BACKEND=gpu_omp_nvidia.
export LLVM_HOME=/path/to/llvm-offload
```

### Build

Three backends are available; one `SBD_BUILD_BACKEND` mode per `pip
install` invocation. Their `_core_*.so` files coexist, so users who
want multiple backends just run `pip install` again with a different
mode.

```bash
# Auto-detect: builds CPU always, plus NVHPC Thrust GPU if NVHPC found
pip install -e . --no-build-isolation

# Explicit modes
SBD_BUILD_BACKEND=cpu                pip install -e . --no-build-isolation
SBD_BUILD_BACKEND=gpu                pip install -e . --no-build-isolation  # NVHPC Thrust
SBD_BUILD_BACKEND=both               pip install -e . --no-build-isolation  # CPU + Thrust
SBD_BUILD_BACKEND=gpu_omp_nvidia     pip install -e . --no-build-isolation  # OpenMP-offload, NVIDIA
```

| Backend | Module | Compiler | Macros | Device strings |
|---|---|---|---|---|
| CPU OpenMP host | `_core_cpu` | gcc/clang | `-fopenmp` (host) | `'cpu'` |
| NVIDIA Thrust | `_core_gpu` | NVHPC nvc++ | `-DSBD_THRUST -cuda -gpu=sm_XX` | `'gpu'`, `'gpu-nvidia'`, `'cuda'` |
| NVIDIA OpenMP-offload | `_core_gpu_omp_nvidia` | LLVM clang++ | `-DUSE_GPU -DUSE_OMP_OFFLOAD -fopenmp-targets=nvptx64-nvidia-cuda` | `'gpu-omp'`, `'gpu-nvidia-omp'` |

**`SBD_BUILD_BACKEND=gpu_omp_nvidia` must be invoked alone** (not
combined with `cpu` / `gpu` / `both`) because clang, gcc, and nvc++
can't share a single distutils `CC`/`CXX` setting. The setup runs
ignoring `auto`'s preferences when this mode is set, builds only
`_core_gpu_omp_nvidia.so`, and leaves any pre-existing `_core_cpu.so`
or `_core_gpu.so` untouched. Requires `LLVM_HOME` pointing at an LLVM
trunk install with the offload runtime built — see
[SETUP_LLVM_OFFLOAD.txt](SETUP_LLVM_OFFLOAD.txt).

**Other GPU backends not exposed via `setup.py`:** the bindings
themselves are GPU-path-agnostic — `bindings.cpp` just wraps the
templated SBD API, and the underlying library picks between Thrust
and OpenMP-offload kernels via `-D` macros at compile time. Adding an
AMD OMP-offload backend (`_core_gpu_omp_amd`) would mean cloning the
`build_gpu_omp_nvidia` block in `setup.py`, renaming the `Extension`
and `-DSBD_MODULE_NAME`, and swapping `-fopenmp-targets` /
`--offload-arch` to `amdgcn-amd-amdhsa` / `gfxXXX` per upstream's
`vendor/sbd-upstream/apps/.../Configuration` AMD example. We don't
ship that today; happy to revisit when AMD hardware is in our test
loop.

### Verify

```bash
python -c "import sbd; print(sbd.available_backends())"
# CPU only:                                 ['cpu']
# CPU + NVHPC Thrust:                       ['cpu', 'gpu']
# CPU + NVHPC Thrust + OMP-offload NVIDIA:  ['cpu', 'gpu', 'gpu-nvidia-omp']
```

## Usage

### Quick Start

```python
import sbd

# No explicit init() needed — auto-initializes on first use
config = sbd.TPB_SBD()
config.adet_comm_size = 2
config.bdet_comm_size = 2
config.max_it = 100
config.eps = 1e-4

results = sbd.tpb_diag_from_files(
    fcidumpfile='data/h2o/fcidump.txt',
    adetfile='data/h2o/h2o-1em4-alpha.txt',
    sbd_data=config,
)

print(f"Energy: {results['energy']:.10f} Hartree")
sbd.finalize()
```

### Runtime backend switching

Whichever backends were built coexist as separate `_core_*.so` modules
and load at `import sbd` into independent pybind11 namespaces. Pick
one per call with the `device` parameter:

```python
import sbd

# All compiled backends are auto-loaded
sbd.available_backends()
# e.g. ['cpu', 'gpu', 'gpu-nvidia-omp']

# Per-call override — auto-initializes on first use
result_cpu     = sbd.tpb_diag(..., device='cpu')
result_thrust  = sbd.tpb_diag(..., device='gpu')         # alias 'gpu-nvidia', 'cuda'
result_omp     = sbd.tpb_diag(..., device='gpu-omp')     # alias 'gpu-nvidia-omp'

# Or set a default device explicitly (optional)
sbd.init(device='gpu')      # default = NVHPC Thrust
result = sbd.tpb_diag(...)

# Or get the backend module directly
backend = sbd.get_backend('gpu-omp')
fcidump = backend.LoadFCIDump('fcidump.txt')
```

In `auto` mode (the default), `_resolve_device('auto')` picks the first
compiled GPU backend in the order Thrust → OMP-offload → CPU.

### Resource Cleanup

```python
results = sbd.tpb_diag_from_files(...)
sbd.finalize()  # optional — syncs GPU and resets state
```

`finalize()` calls `cudaDeviceSynchronize()` on GPU backends and resets Python state. It does **not** call `cudaDeviceReset()` (avoids CUDA-aware MPI conflicts) or `MPI_Finalize()` (handled by mpi4py).

## Integration with qiskit-addon-sqd

SBD can serve as the eigensolver backend for qiskit-addon-sqd's SQD workflow.

**Note:** Requires the `patch-ferminon-sbd` branch of [hfwen0502/qiskit-addon-sqd](https://github.com/hfwen0502/qiskit-addon-sqd) for MPI-aware solver support.

```python
from functools import partial
from sbd.sbd_solver import solve_sci_batch
from sbd.device_config import DeviceConfig
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

# No sbd.init() and no explicit mpi_comm needed — solve_sci_batch
# auto-initializes the SBD backend on first call and falls back to
# MPI.COMM_WORLD when mpi_comm is not provided.
sbd_solver = partial(
    solve_sci_batch,
    sbd_config={"method": 0, "eps": 1e-8, "max_it": 100},
    device_config=DeviceConfig.gpu(),  # or .cpu(), .gpu_omp()
)

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    sci_solver=sbd_solver,
    norb=norb, nelec=nelec,
    samples_per_batch=300, num_batches=3, max_iterations=5,
    symmetrize_spin=True,
)
```

See `python/examples/run_sqd_sbd.py` for a complete example.

### Comparison with qiskit-addon-dice-solver

| Feature | dice-solver | SBD |
|---------|------------|-----|
| Solver | DICE (subprocess) | SBD (in-process) |
| GPU | No | Yes (CUDA) |
| MPI | Spawns processes | Direct integration |
| I/O | Temp files | In-memory |

## Examples

Located in `python/examples/`:

- **`run_sbd_diag.py`** — Standalone TPB diagonalization (no Qiskit dependency)
- **`run_sqd_sbd.py`** — SQD loop with SBD solver (random or hardware bitstrings)

See [python/examples/README.md](../python/examples/README.md) for usage details.

## API Reference

### Initialization

| Function | Description |
|----------|-------------|
| `sbd.init(device, comm_backend)` | **Optional.** Initialize MPI, set default device (`'cpu'`, `'gpu'`, `'auto'`). Auto-called on first use with defaults. |
| `sbd.finalize()` | Sync GPU, reset state. Does not call `MPI_Finalize` |
| `sbd.is_initialized()` | Check init status |

### Backend Access

| Function | Description |
|----------|-------------|
| `sbd.get_backend(device=None)` | Get backend module (`_core_cpu`, `_core_gpu`, or `_core_gpu_omp_nvidia`). `None` = default |
| `sbd.available_backends()` | List of compiled backends, e.g. `['cpu']`, `['cpu', 'gpu']`, `['cpu', 'gpu', 'gpu-nvidia-omp']` |

### Query

| Function | Description |
|----------|-------------|
| `sbd.get_device()` | Default device name |
| `sbd.get_rank()` | MPI rank |
| `sbd.get_world_size()` | MPI world size |
| `sbd.get_comm()` | MPI communicator |
| `sbd.barrier()` | MPI barrier |

### Configuration

```python
config = sbd.TPB_SBD()
```

| Attribute | Default | Description |
|-----------|---------|-------------|
| `method` | 0 | 0=Davidson, 1=Davidson+Ham, 2=Lanczos, 3=Lanczos+Ham |
| `max_it` | 1 | Max iterations |
| `eps` | 1e-4 | Convergence tolerance |
| `max_nb` | 10 | Max basis vectors |
| `do_rdm` | 0 | 0=density only, 1=full RDM |
| `bit_length` | 20 | Bit length for determinants |
| `adet_comm_size` | 1 | Alpha determinant communicator size |
| `bdet_comm_size` | 1 | Beta determinant communicator size |
| `task_comm_size` | 1 | Task communicator size |

Total MPI ranks = `task_comm_size × adet_comm_size × bdet_comm_size`.

### Diagonalization

```python
# From files
results = sbd.tpb_diag_from_files(fcidumpfile, adetfile, sbd_data,
                                   loadname="", savename="", device=None)

# From data structures
results = sbd.tpb_diag(fcidump, adet, bdet, sbd_data,
                        loadname="", savename="", device=None)
```

**Returns:** `dict` with keys `energy`, `density`, `carryover_adet`, `carryover_bdet`, `one_p_rdm`, `two_p_rdm`.

The optional `device` parameter overrides the default set by `init()`.

### Utilities

```python
fcidump = sbd.LoadFCIDump("fcidump.txt", device=None)
dets = sbd.LoadAlphaDets("alphadets.txt", bit_length, total_bit_length, device=None)
string = sbd.makestring(det, bit_length, total_bit_length, device=None)
det = sbd.from_string(s, bit_length, total_bit_length, device=None)
sbd.print_info()
```

## Backend Architecture

- Each backend is a separate pybind11 module — `_core_cpu.so`, `_core_gpu.so` (NVHPC Thrust), `_core_gpu_omp_nvidia.so` (LLVM OpenMP-offload). All compiled from the same `python/bindings.cpp` source with different `-D` macros (`SBD_THRUST`, `USE_GPU + USE_OMP_OFFLOAD`, or neither for CPU) and different compilers (gcc/clang, nvc++, clang++ respectively). Distinct C++ namespaces — no symbol collision when multiple coexist.
- All compiled backends are loaded eagerly at `import sbd` into `sbd._backends`. Aliases (`'gpu-omp'` → `'gpu-nvidia-omp'`, `'cuda'`/`'gpu-nvidia'` → `'gpu'`) live in `sbd._device_aliases`.
- `get_backend(device)` resolves aliases and returns the appropriate module; all wrapper functions accept an optional `device` parameter.
- GPU device assignment: `gpu_id = mpi_rank % num_gpus` (set per `tpb_diag()` call in `bindings.cpp`); same logic for both Thrust and OMP-offload paths.
- Backends differ in which phases run on the GPU vs the host. Davidson and the matvec (`mult`) live on the GPU under both Thrust and OMP-offload. The diagonal-Hamiltonian preconditioner (`makeQChamDiagTerms`) is GPU-resident under Thrust but runs on the host under OMP-offload (no `#pragma omp target` port in `tpb/qcham.h`) — see `.github/SETUP_LLVM_OFFLOAD.txt` for details.

## Troubleshooting

**`ImportError` on macOS (symbol not found):** Python's libc++ and Homebrew clang's libc++ may differ. Use system clang: `CC=/usr/bin/clang CXX=/usr/bin/clang++`.

**`ImportError: _core_cpu`:** Backend not built. Rebuild: `pip install -e . --no-build-isolation -v`

**GPU not building:** Check `which nvc++` and set `NVHPC_HOME`.

**MPI errors:** Verify `MPI_HOME`, check `python -c "from mpi4py import MPI; print(MPI.Get_version())"`.

## Performance Tips

**CPU:** `OMP_NUM_THREADS` = cores per MPI rank.
**GPU (Thrust):** 1 rank per GPU, `OMP_NUM_THREADS=1`, use method 0 (matrix-free Davidson).
**GPU (OMP-offload):** 1 rank per GPU, `OMP_NUM_THREADS` ≈ socket-local cores per rank, **pin each rank to one socket** (e.g. `mpirun --map-by ppr:N:socket --bind-to socket …` or wrap with `numactl --cpunodebind=… --membind=…`). Without pinning, the host-side `makeQChamDiagTerms` loop and the host-side orchestration inside Davidson degrade ~7× and 2–3× respectively because OMP threads thrash across NUMA nodes. Long unpinned runs have also produced spurious `cuMemAlloc[Host|Managed]` failures from libomptarget that don't reproduce when pinned.

---

**Repository:** https://github.com/hfwen0502/sbd
