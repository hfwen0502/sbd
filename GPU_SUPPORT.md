# GPU Support in SBD Python Bindings

## Important Notice

**The Python bindings currently support the CPU version of the SBD library.** The SBD library itself supports GPU execution when compiled with NVIDIA HPC SDK (nvc++) and THRUST, but the Python bindings are designed to work with the CPU-compiled version.

## Current Architecture

```
┌─────────────────────────────────────────┐
│  Python Bindings (this implementation)  │
│  - Built with g++/gcc                   │
│  - Links against CPU version of SBD     │
│  - Provides DeviceConfig for future use │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  SBD C++ Library                        │
│  - CPU version: Built with g++          │
│  - GPU version: Built with nvc++ (separate) │
└─────────────────────────────────────────┘
```

## DeviceConfig Class - Future-Ready Design

The `DeviceConfig` class is included in the Python bindings as a **future-ready** interface. While the current Python bindings link against the CPU version, the API is designed to support GPU execution when available.

### Current Behavior

```python
import sbd

# Auto-detect (currently always uses CPU)
device_config = sbd.DeviceConfig.auto()
device_config.apply(config)
# Output: "No GPU detected, using CPU"

# The GPU parameters are available in the binding
# but have no effect with CPU-compiled library
config = sbd.TPB_SBD()
try:
    config.use_precalculated_dets = True  # Available if compiled with THRUST
    config.max_memory_gb_for_determinants = 16
except AttributeError:
    print("GPU parameters not available (CPU version)")
```

## Using GPU: Two Approaches

### Approach 1: Use C++ GPU Executable (Recommended for GPU)

For GPU calculations, use the native C++ executable:

```bash
# 1. Build SBD with GPU support
cd /path/to/sbd/apps/chemistry_tpb_selected_basis_diagonalization

# Edit Configuration to enable CUDA flags:
# CCFLAGS= -mp -cuda -fast -Minfo=accel \
#          --diag_suppress declared_but_not_referenced,set_but_not_used \
#          -fmax-errors=0 \
#          -I/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/include/cccl \
#          -I/usr/local/cuda/include \
#          -DSBD_THRUST

# 2. Build with nvc++ (requires NVIDIA HPC SDK)
module load nvhpc  # On HPC systems
make CXX=nvc++

# 3. Run GPU calculation
mpirun -np 16 ./diag \
    --fcidump fcidump.txt \
    --adetfile alphadets.txt \
    --method 0 \
    --iteration 100 \
    --tolerance 1.0e-6 \
    --rdm 0
```

### Approach 2: Use Python Bindings (CPU Only, Currently)

For CPU calculations with Python convenience:

```bash
# 1. Build Python bindings (CPU version)
cd /path/to/sbd
pip install -e .

# 2. Run Python calculation
mpirun -np 4 python script.py
```

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
```

## Why This Design?

### Challenges with GPU Python Bindings

1. **Compiler Incompatibility**: 
   - GPU version requires nvc++ (NVIDIA HPC SDK)
   - Python bindings use g++/gcc with pybind11
   - Mixing compilers is complex and error-prone

2. **Build Complexity**:
   - CUDA flags: `-mp -cuda -fast -Minfo=accel -DSBD_THRUST`
   - Special include paths for CUDA/CCCL
   - Different linking requirements

3. **Deployment**:
   - GPU version requires NVIDIA HPC SDK on user systems
   - CPU version works with standard g++
   - Separate builds would require different Python packages

### Current Solution

- **Python bindings**: CPU version (easy to build and deploy)
- **GPU calculations**: Use C++ executable directly
- **DeviceConfig**: API ready for future GPU support

## Future GPU Support in Python Bindings

To enable GPU support in Python bindings in the future, the following would be needed:

### Option 1: Dual-Build System

```python
# In setup.py (future enhancement)
if os.environ.get('SBD_GPU') == '1':
    # Build with nvc++ and CUDA flags
    ext_modules = [
        Extension(
            'sbd._core_gpu',
            ['python/bindings.cpp'],
            extra_compile_args=['-mp', '-cuda', '-DSBD_THRUST', ...],
        )
    ]
else:
    # Build with g++ (current)
    ext_modules = [
        Extension(
            'sbd._core',
            ['python/bindings.cpp'],
            extra_compile_args=['-std=c++17', '-fopenmp', ...],
        )
    ]
```

### Option 2: Runtime Library Loading

```python
# Load appropriate library at runtime
try:
    from ._core_gpu import *
    GPU_AVAILABLE = True
except ImportError:
    from ._core import *
    GPU_AVAILABLE = False
```

### Option 3: Separate GPU Package

```bash
# Two separate packages
pip install sbd          # CPU version
pip install sbd-gpu      # GPU version (requires NVIDIA HPC SDK)
```

## DeviceConfig API (Future-Ready)

The `DeviceConfig` class is included now for API consistency:

```python
import sbd

# This API will work when GPU support is added
device_config = sbd.DeviceConfig.auto()  # Will detect GPU when available
device_config.apply(config)

# Check capabilities
info = sbd.get_device_info()
print(info)  # Currently: {'gpu_available': False, ...}

# Print device info
sbd.print_device_info()
# Currently outputs:
# ============================================================
# SBD Device Information
# ============================================================
# ✗ No GPU detected
# ✓ CPU Available: Always
# ============================================================
```

## Recommendations

### For CPU Calculations
✅ **Use Python bindings** - Easy to use, good for:
- Development and testing
- Small to medium systems
- Integration with Python workflows
- Interactive calculations

### For GPU Calculations
✅ **Use C++ executable** - Best performance, required for:
- Large-scale production runs
- Maximum GPU performance
- Systems requiring GPU acceleration
- HPC environments with GPU nodes

### Hybrid Workflow

```bash
# 1. Develop and test with Python (CPU)
python develop_workflow.py

# 2. Run production calculations with C++ (GPU)
mpirun -np 16 ./diag --fcidump ... --method 0

# 3. Analyze results with Python
python analyze_results.py
```

## Building SBD with GPU Support

### Requirements
- NVIDIA HPC SDK (provides nvc++)
- CUDA Toolkit
- MPI implementation
- BLAS/LAPACK

### Build Steps

```bash
# 1. Load NVIDIA HPC SDK
module load nvhpc  # On HPC systems
# or
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/2025/compilers/bin:$PATH

# 2. Navigate to application directory
cd /path/to/sbd/apps/chemistry_tpb_selected_basis_diagonalization

# 3. Edit Configuration file
# Uncomment the CUDA flags section:
# CCFLAGS= -mp -cuda -fast -Minfo=accel \
#          --diag_suppress declared_but_not_referenced,set_but_not_used \
#          -fmax-errors=0 \
#          -I/opt/nvidia/hpc_sdk/Linux_x86_64/2025/cuda/include/cccl \
#          -I/usr/local/cuda/include \
#          -DSBD_THRUST

# 4. Build
make CXX=nvc++

# 5. Verify
./diag --help
```

### Running GPU Calculations

```bash
# Set GPU device (if multiple GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run with MPI
mpirun -np 16 -x OMP_NUM_THREADS=1 ./diag \
    --fcidump fcidump_Fe4S4.txt \
    --adetfile AlphaDets.txt \
    --method 0 \
    --block 10 \
    --iteration 4 \
    --tolerance 1.0e-4 \
    --adet_comm_size 2 \
    --bdet_comm_size 2 \
    --task_comm_size 2 \
    --rdm 0
```

## Summary

| Feature | Python Bindings | C++ Executable |
|---------|----------------|----------------|
| **Device** | CPU only | CPU or GPU |
| **Compiler** | g++/gcc | g++ (CPU) or nvc++ (GPU) |
| **Ease of Use** | ✅ Easy | Moderate |
| **Performance** | Good (CPU) | Best (GPU) |
| **Integration** | ✅ Python ecosystem | Command-line |
| **GPU Support** | ❌ Not yet | ✅ Yes |
| **Recommended For** | Development, testing | Production, GPU |

## Future Work

To add GPU support to Python bindings:
1. Create separate build configuration for GPU
2. Handle nvc++ compilation in setup.py
3. Manage CUDA dependencies
4. Test with various NVIDIA HPC SDK versions
5. Create separate PyPI package (sbd-gpu)

For now, use the C++ executable for GPU calculations and Python bindings for CPU calculations and development.