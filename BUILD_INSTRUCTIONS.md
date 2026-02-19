# Quick Build Instructions for SBD Python Bindings

## The setup.py has been updated to automatically detect MPI library paths!

### Step 1: Install Prerequisites

```bash
# Install Python dependencies
pip3 install pybind11 mpi4py numpy

# Make sure MPI is installed and mpicc is in PATH
which mpicc  # Should show path to mpicc
```

### Step 2: Build and Install

```bash
cd /path/to/sbd
pip3 install -e .
```

The updated `setup.py` will automatically:
- Detect MPI include and library paths using `mpicc --showme`
- Find the correct MPI libraries
- Configure the build appropriately

### If Build Still Fails

#### Option 1: Specify Library Paths Manually

If automatic detection fails, you can specify paths manually:

```bash
# Find your library paths
find /usr -name "libmpi.so" 2>/dev/null
find /usr -name "liblapack.so" 2>/dev/null
find /usr -name "libblas.so" 2>/dev/null

# Set environment variables
export LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Then build
pip3 install -e .
```

#### Option 2: Use OpenBLAS (Recommended)

OpenBLAS provides both BLAS and LAPACK:

```bash
# Install OpenBLAS
sudo yum install openblas-devel  # CentOS/RHEL
# or
sudo apt-get install libopenblas-dev  # Ubuntu/Debian

# The libraries are usually in:
# /usr/lib64/libopenblas.so (CentOS/RHEL)
# /usr/lib/x86_64-linux-gnu/libopenblas.so (Ubuntu/Debian)

# Build
pip3 install -e .
```

#### Option 3: Modify setup.py Directly

Edit `setup.py` and change the libraries line:

```python
# Instead of:
libraries = mpi_libs + ['lapack', 'blas']

# Try:
libraries = mpi_libs + ['openblas']  # If using OpenBLAS
# or
libraries = mpi_libs  # If MPI libs include everything needed
```

### Verification

After successful build:

```bash
# Test import
python3 -c "import sbd; print(sbd.__version__)"

# Should print: 1.2.0
```

### Common Issues

**Issue: "cannot find -lmpi"**
```bash
# Find MPI library
find /usr -name "libmpi*.so" 2>/dev/null

# If found in /usr/lib64/openmpi/lib/, add to LIBRARY_PATH:
export LIBRARY_PATH=/usr/lib64/openmpi/lib:$LIBRARY_PATH
pip3 install -e . --force-reinstall
```

**Issue: "cannot find -llapack" or "cannot find -lblas"**
```bash
# Option 1: Install OpenBLAS
sudo yum install openblas-devel

# Option 2: Find existing libraries
ldconfig -p | grep -E "lapack|blas"

# Option 3: Use system BLAS/LAPACK
sudo yum install lapack-devel blas-devel
```

**Issue: MPI module not loaded**
```bash
# On HPC systems, load MPI module first
module load mpi/openmpi-x86_64
# or
module load openmpi

# Then build
pip3 install -e .
```

### Debug Build

To see exactly what's happening:

```bash
python3 setup.py build_ext --inplace -v 2>&1 | tee build.log
```

Review `build.log` for the exact compiler and linker commands used.

### Success!

Once built successfully, you can use the bindings:

```python
from mpi4py import MPI
import sbd

comm = MPI.COMM_WORLD
config = sbd.TPB_SBD()
config.max_it = 100

results = sbd.tpb_diag_from_files(
    comm=comm,
    sbd_data=config,
    fcidumpfile="fcidump.txt",
    adetfile="alphadets.txt"
)

print(f"Energy: {results['energy']}")
```

## Need More Help?

See the detailed guides:
- `INSTALL_PYTHON.md` - Complete installation guide
- `README_PYTHON.md` - API documentation
- `TROUBLESHOOTING_BUILD.md` - Detailed troubleshooting