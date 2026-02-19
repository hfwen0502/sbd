# Quick Start: Dual Backend System

## TL;DR

```bash
# Install (builds both CPU and GPU backends if CUDA/HIP available)
pip install -e .

# Use in Python (automatically selects best backend)
python
>>> import sbd
>>> sbd.get_backend()  # 'gpu' or 'cpu'
>>> sbd.available_backends()  # ['cpu', 'gpu'] or ['cpu']
```

## Installation

### Standard Install (Recommended)

```bash
cd /path/to/sbd
pip install -e .
```

This will:
- ✅ Always build CPU backend
- ✅ Build GPU backend if CUDA or HIP detected
- ✅ Auto-detect MPI, BLAS, LAPACK

### CPU-Only Install

```bash
SBD_ENABLE_CUDA=0 SBD_ENABLE_HIP=0 pip install -e .
```

### Force GPU Install

```bash
SBD_ENABLE_CUDA=1 pip install -e .  # Fails if CUDA not found
```

## Basic Usage

### Auto Backend Selection

```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD

# Automatically uses best available backend (GPU preferred)
config = sbd.TPB_SBD()
config.max_it = 100
config.eps = 1e-6

results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

print(f"Backend: {sbd.get_backend()}")
print(f"Energy: {results['energy']}")
```

### Explicit Backend Selection

```python
import sbd

# Force CPU
sbd.use_backend('cpu')
results_cpu = sbd.tpb_diag_from_files(...)

# Force GPU (if available)
try:
    sbd.use_backend('gpu')
    results_gpu = sbd.tpb_diag_from_files(...)
except ImportError:
    print("GPU backend not available")
```

### Command-Line Backend Selection

```bash
# Create script with backend argument
cat > run.py << 'EOF'
import sys
import sbd
from mpi4py import MPI

backend = sys.argv[1] if len(sys.argv) > 1 else 'auto'
if backend != 'auto':
    sbd.use_backend(backend)

# Run calculation...
config = sbd.TPB_SBD()
results = sbd.tpb_diag_from_files(
    comm=MPI.COMM_WORLD,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)
print(f"Backend: {sbd.get_backend()}, Energy: {results['energy']}")
EOF

# Run with different backends
mpirun -np 4 python run.py cpu
mpirun -np 4 python run.py gpu
mpirun -np 4 python run.py  # auto
```

## Check Installation

```python
import sbd

# Print backend info
sbd.print_backend_info()

# Output:
# ======================================================================
# SBD Python Bindings
# ======================================================================
# Version: 1.2.0
# Current backend: gpu
# Available backends: cpu, gpu
# ======================================================================
```

## Examples

### Example 1: Simple Calculation

```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD
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

```python
import sbd
import time

for backend in sbd.available_backends():
    sbd.use_backend(backend)
    
    start = time.time()
    results = sbd.tpb_diag_from_files(...)
    elapsed = time.time() - start
    
    print(f"{backend.upper()}: {elapsed:.3f}s")
```

### Example 3: Robust Backend Selection

```python
import sbd

# Try GPU, fall back to CPU
preferred_backend = 'gpu'
try:
    sbd.use_backend(preferred_backend)
    print(f"Using {preferred_backend.upper()} backend")
except ImportError:
    print(f"{preferred_backend.upper()} not available, using {sbd.get_backend().upper()}")

# Run calculation...
```

## API Quick Reference

| Function | Description |
|----------|-------------|
| `sbd.get_backend()` | Get current backend name |
| `sbd.available_backends()` | List available backends |
| `sbd.use_backend(name)` | Switch to specific backend |
| `sbd.print_backend_info()` | Print backend information |

## Troubleshooting

### GPU Backend Not Available

```python
>>> sbd.available_backends()
['cpu']  # Only CPU available
```

**Solution:** Reinstall with CUDA/HIP:
```bash
# Check CUDA
nvcc --version

# Reinstall
pip uninstall sbd
pip install -e . -v  # Verbose output shows GPU detection
```

### Import Error

```python
>>> import sbd
ImportError: Could not import SBD backend
```

**Solution:** Rebuild the package:
```bash
pip install -e . --force-reinstall
```

### Different Results on CPU vs GPU

Small differences (< 1e-10) are normal due to floating-point arithmetic.

```python
sbd.use_backend('cpu')
e_cpu = sbd.tpb_diag_from_files(...)['energy']

sbd.use_backend('gpu')
e_gpu = sbd.tpb_diag_from_files(...)['energy']

print(f"Difference: {abs(e_cpu - e_gpu):.2e}")  # Should be very small
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `SBD_PRINT_INFO` | Print info on import | `export SBD_PRINT_INFO=1` |
| `SBD_ENABLE_CUDA` | Control CUDA build | `SBD_ENABLE_CUDA=0 pip install -e .` |
| `SBD_ENABLE_HIP` | Control HIP build | `SBD_ENABLE_HIP=0 pip install -e .` |

## Complete Example Script

```python
#!/usr/bin/env python3
"""
Complete example: H2O calculation with backend selection
"""

from mpi4py import MPI
import sbd
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Parse backend from command line
    backend = sys.argv[1] if len(sys.argv) > 1 else 'auto'
    
    # Switch backend if requested
    if backend != 'auto':
        try:
            sbd.use_backend(backend)
        except ImportError as e:
            if rank == 0:
                print(f"Warning: {e}")
                print(f"Using {sbd.get_backend()} backend instead")
    
    # Print info
    if rank == 0:
        print(f"Backend: {sbd.get_backend().upper()}")
        print(f"Available: {sbd.available_backends()}")
    
    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = 100
    config.eps = 1e-6
    config.method = 0  # Davidson
    
    # Run calculation
    results = sbd.tpb_diag_from_files(
        comm=comm,
        sbd_data=config,
        fcidumpfile="fcidump_h2o.txt",
        adetfile="alphadets_h2o.txt"
    )
    
    # Print results
    if rank == 0:
        print(f"Energy: {results['energy']:.10f}")
        print(f"Carryover dets: {len(results['co_adet'])}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
# Auto-select backend
mpirun -np 4 python script.py

# Force CPU
mpirun -np 4 python script.py cpu

# Force GPU
mpirun -np 4 python script.py gpu
```

## Next Steps

- See `DUAL_BACKEND_BUILD.md` for detailed documentation
- See `python/examples/h2o_backend_selection.py` for complete example
- See `BACKEND_ARCHITECTURE.md` for PyTorch-style factory pattern design

## Summary

The dual backend system provides:

✅ **Automatic**: Best backend selected by default  
✅ **Flexible**: Switch backends at runtime  
✅ **Simple**: Clean Python API  
✅ **Robust**: Graceful fallback to CPU  
✅ **Fast**: GPU acceleration when available  

Just `pip install` and use - the system handles the rest!