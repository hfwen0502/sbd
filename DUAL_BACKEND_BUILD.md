# Dual Backend Build System

## Overview

The SBD Python bindings now support building **two separate backend libraries** in a single installation:

- **`sbd._core_cpu`**: CPU backend (always built)
- **`sbd._core_gpu`**: GPU backend (built if CUDA or HIP detected)

The Python package automatically selects the best available backend at import time, with the ability to switch backends at runtime.

## Architecture

### Build System

```
setup.py
├── Detect CUDA (nvcc)
├── Detect HIP (hipcc)
├── Build _core_cpu (always)
│   ├── Compile with g++/gcc
│   ├── Link with MPI, BLAS, LAPACK
│   └── No GPU flags
└── Build _core_gpu (if GPU detected)
    ├── Compile with g++/gcc
    ├── Link with MPI, BLAS, LAPACK, CUDA/HIP
    └── Add -DSBD_THRUST flag
```

### Python Package Structure

```
sbd/
├── __init__.py          # Auto-loads best backend
├── _core_cpu.so         # CPU backend (always present)
├── _core_gpu.so         # GPU backend (if built)
└── device_config.py     # Legacy helper (deprecated)
```

### Runtime Backend Selection

```python
import sbd

# Automatic selection (prefers GPU if available)
print(sbd.get_backend())  # 'gpu' or 'cpu'

# Check what's available
print(sbd.available_backends())  # ['cpu', 'gpu'] or ['cpu']

# Switch backends
sbd.use_backend('cpu')   # Force CPU
sbd.use_backend('gpu')   # Force GPU (if available)
```

## Building

### Prerequisites

**Required:**
- Python 3.7+
- pybind11
- mpi4py
- MPI implementation (OpenMPI, MPICH, etc.)
- BLAS/LAPACK
- C++17 compiler (g++/gcc)

**Optional (for GPU backend):**
- CUDA Toolkit + nvcc (for NVIDIA GPUs)
- ROCm + hipcc (for AMD GPUs)

### Installation

```bash
cd /path/to/sbd

# Standard installation (builds both backends if GPU available)
pip install -e .

# Force CPU-only build
SBD_ENABLE_CUDA=0 SBD_ENABLE_HIP=0 pip install -e .

# Force CUDA build (fails if CUDA not available)
SBD_ENABLE_CUDA=1 pip install -e .
```

### Build Output

During installation, you'll see:

```
======================================================================
Building CPU Backend
======================================================================
✓ CPU backend will be built

======================================================================
Checking for GPU Support
======================================================================
✓ CUDA detected - will build GPU backend
Building GPU backend with CUDA
✓ GPU backend will be built

======================================================================
Total extensions to build: 2
  - sbd._core_cpu
  - sbd._core_gpu
======================================================================
```

## Usage

### Basic Usage (Auto Backend)

```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD

# Automatically uses best available backend
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6

results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

print(f"Used backend: {sbd.get_backend()}")
print(f"Energy: {results['energy']}")
```

### Explicit Backend Selection

```python
import sbd

# Check available backends
print("Available:", sbd.available_backends())

# Use CPU backend
sbd.use_backend('cpu')
results_cpu = sbd.tpb_diag_from_files(...)

# Use GPU backend (if available)
try:
    sbd.use_backend('gpu')
    results_gpu = sbd.tpb_diag_from_files(...)
except ImportError:
    print("GPU backend not available")
```

### Backend Comparison

```python
import sbd
import time

for backend in sbd.available_backends():
    sbd.use_backend(backend)
    
    start = time.time()
    results = sbd.tpb_diag_from_files(...)
    elapsed = time.time() - start
    
    print(f"{backend.upper()}: {results['energy']:.10f} ({elapsed:.3f}s)")
```

### Command-Line Backend Selection

```python
# h2o_backend_selection.py
import sys
import sbd

backend = sys.argv[1] if len(sys.argv) > 1 else 'auto'

if backend != 'auto':
    sbd.use_backend(backend)

# Run calculation...
```

```bash
# Use CPU
mpirun -np 4 python h2o_backend_selection.py cpu

# Use GPU
mpirun -np 4 python h2o_backend_selection.py gpu

# Auto-select
mpirun -np 4 python h2o_backend_selection.py
```

## API Reference

### Backend Management Functions

#### `sbd.get_backend()`
Returns the name of the currently active backend.

```python
backend = sbd.get_backend()  # 'cpu' or 'gpu'
```

#### `sbd.available_backends()`
Returns list of available backends.

```python
backends = sbd.available_backends()  # ['cpu', 'gpu'] or ['cpu']
```

#### `sbd.use_backend(backend_name)`
Switch to a specific backend.

```python
sbd.use_backend('cpu')   # Switch to CPU
sbd.use_backend('gpu')   # Switch to GPU (raises ImportError if not available)
```

**Parameters:**
- `backend_name` (str): Backend to use ('cpu' or 'gpu')

**Raises:**
- `ImportError`: If requested backend is not available
- `ValueError`: If backend_name is invalid

#### `sbd.print_backend_info()`
Print information about backends.

```python
sbd.print_backend_info()
```

Output:
```
======================================================================
SBD Python Bindings
======================================================================
Version: 1.2.0
Current backend: gpu
Available backends: cpu, gpu
======================================================================
```

### Environment Variables

#### `SBD_PRINT_INFO`
Print backend info on import.

```bash
export SBD_PRINT_INFO=1
python -c "import sbd"
```

#### `SBD_ENABLE_CUDA` (build-time)
Control CUDA backend build.

```bash
SBD_ENABLE_CUDA=0 pip install -e .  # Disable CUDA
SBD_ENABLE_CUDA=1 pip install -e .  # Force CUDA
```

#### `SBD_ENABLE_HIP` (build-time)
Control HIP backend build.

```bash
SBD_ENABLE_HIP=0 pip install -e .  # Disable HIP
SBD_ENABLE_HIP=1 pip install -e .  # Force HIP
```

## Examples

### Example 1: Simple Backend Selection

```python
#!/usr/bin/env python3
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD

# Print available backends
if comm.Get_rank() == 0:
    print(f"Available backends: {sbd.available_backends()}")
    print(f"Current backend: {sbd.get_backend()}")

# Run calculation with current backend
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6

results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

if comm.Get_rank() == 0:
    print(f"Energy: {results['energy']}")
```

### Example 2: Backend Comparison

See `python/examples/h2o_backend_selection.py` for a complete example that:
- Allows command-line backend selection
- Compares CPU vs GPU performance
- Handles missing backends gracefully

```bash
# Run with CPU
mpirun -np 4 python examples/h2o_backend_selection.py cpu

# Run with GPU
mpirun -np 4 python examples/h2o_backend_selection.py gpu

# Compare both backends
mpirun -np 4 python examples/h2o_backend_selection.py --compare
```

## Troubleshooting

### GPU Backend Not Built

**Symptom:**
```python
>>> sbd.available_backends()
['cpu']
```

**Solutions:**

1. **Check CUDA/HIP installation:**
```bash
nvcc --version  # For CUDA
hipcc --version # For HIP
```

2. **Reinstall with verbose output:**
```bash
pip uninstall sbd
pip install -e . -v
```

3. **Check build log for GPU detection:**
```
✓ CUDA detected - will build GPU backend
```
or
```
✗ CUDA not detected - GPU backend will not be built
```

### Cannot Switch to GPU Backend

**Symptom:**
```python
>>> sbd.use_backend('gpu')
ImportError: GPU backend not available
```

**Cause:** GPU backend was not built during installation.

**Solution:** Reinstall with GPU support (see above).

### Both Backends Give Different Results

**Symptom:** CPU and GPU backends produce slightly different energies.

**Cause:** Floating-point arithmetic differences between CPU and GPU.

**Expected:** Differences should be < 1e-10 for well-conditioned problems.

**Check:**
```python
import sbd

sbd.use_backend('cpu')
results_cpu = sbd.tpb_diag_from_files(...)

sbd.use_backend('gpu')
results_gpu = sbd.tpb_diag_from_files(...)

diff = abs(results_cpu['energy'] - results_gpu['energy'])
print(f"Energy difference: {diff:.2e}")  # Should be very small
```

## Technical Details

### Compilation Flags

**CPU Backend:**
```
-std=c++17 -fopenmp -O2 -fPIC
```

**GPU Backend (CUDA):**
```
-std=c++17 -fopenmp -O2 -fPIC -DSBD_THRUST -D__CUDACC__
-I/usr/local/cuda/include
-L/usr/local/cuda/lib64 -lcudart
```

**GPU Backend (HIP):**
```
-std=c++17 -fopenmp -O2 -fPIC -DSBD_THRUST
-I/opt/rocm/include
-L/opt/rocm/lib -lamdhip64
```

### Backend Detection Logic

```python
# In __init__.py
try:
    from . import _core_gpu as _backend
    _backend_name = "gpu"
except ImportError:
    from . import _core_cpu as _backend
    _backend_name = "cpu"
```

Preference order:
1. GPU backend (if available)
2. CPU backend (fallback)

### Memory Considerations

- **CPU backend**: Uses system RAM
- **GPU backend**: Uses GPU memory + system RAM
- GPU backend may require more memory for determinant storage

## Migration from Old API

### Old API (device_config.py)

```python
import sbd

device_config = sbd.DeviceConfig.auto()
device_config.apply(config)
```

### New API (dual backend)

```python
import sbd

# Automatic backend selection (no code needed)
# Or explicit:
sbd.use_backend('gpu')  # or 'cpu'
```

The old `DeviceConfig` class is deprecated but still available for backward compatibility.

## Summary

The dual backend build system provides:

✅ **Single Installation**: One `pip install` for both backends  
✅ **Automatic Selection**: Best backend chosen at import  
✅ **Runtime Switching**: Change backends without reinstalling  
✅ **Graceful Degradation**: Falls back to CPU if GPU unavailable  
✅ **Easy Testing**: Compare CPU vs GPU results easily  
✅ **Clean API**: Simple Python interface  

This design follows PyTorch's approach of building multiple backends in a single package while keeping the user API simple and intuitive.