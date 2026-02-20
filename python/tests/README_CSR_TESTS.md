# CSR Export Test Suite

## Overview

The `test_csr_export.py` script provides comprehensive testing for the CSR (Compressed Sparse Row) Hamiltonian export functionality.

## Test Cases

### Test 1: Basic CSR Export
- Loads H2O test data (small determinant set)
- Exports Hamiltonian to CSR format
- Validates CSR format structure:
  - NumPy array types
  - Correct array dimensions
  - Sparsity statistics

### Test 2: Hermitian Property
- Verifies that H = H† (matrix is Hermitian)
- Checks maximum deviation from Hermiticity
- Essential for quantum chemistry Hamiltonians

### Test 3: Energy Comparison
- Compares SciPy eigsh energy vs SBD Davidson energy
- Should match within 1e-6 Ha tolerance
- Validates correctness of CSR export

### Test 4: Matrix Properties
- Checks diagonal elements (should be real)
- Verifies no NaN or Inf values
- Reports matrix statistics

### Test 5: GPU Backend (Optional)
- Tests CSR export with GPU backend
- Skipped if GPU support not compiled
- Verifies both backends produce valid output

## Running the Tests

### Prerequisites
```bash
# Install Python dependencies
pip install numpy scipy mpi4py

# Build SBD Python bindings
python setup.py build_ext --inplace
```

### Run Tests
```bash
# Single process
mpirun -np 1 python python/tests/test_csr_export.py

# Multiple processes (MPI)
mpirun -np 4 python python/tests/test_csr_export.py
```

## Expected Output

```
======================================================================
CSR HAMILTONIAN EXPORT TEST SUITE
======================================================================

======================================================================
TEST 1: Basic CSR Export
======================================================================
Loaded 100 alpha determinants
Loaded 100 beta determinants
Hilbert space dimension: 10000

Exporting Hamiltonian to CSR format...

CSR Format Verification:
  Matrix shape: (10000, 10000)
  Number of non-zeros: 450000
  Truncated: False
  Sparsity: 4.50%

✓ CSR format validation passed

======================================================================
TEST 2: Hermitian Property
======================================================================
Max |H - H†|: 1.23e-15
✓ Matrix is Hermitian

======================================================================
TEST 3: Energy Comparison (CSR vs SBD)
======================================================================

Solving with SciPy eigsh...
SciPy energy: -76.2417531234 Ha

Solving with SBD Davidson...
SBD energy:   -76.2417531234 Ha

Energy difference: 3.45e-12 Ha
✓ Energies match within tolerance

======================================================================
TEST 4: Matrix Properties
======================================================================

Diagonal statistics:
  Min: -15.234567
  Max: 12.345678
  Mean: -2.345678
  All real: True

Data quality:
  Contains NaN: False
  Contains Inf: False
✓ Matrix properties are valid

======================================================================
TEST 5: GPU Backend (Optional)
======================================================================
GPU backend available
✓ GPU backend CSR export successful
  Matrix shape: (10000, 10000)
  Non-zeros: 450000

======================================================================
TEST SUMMARY
======================================================================
basic          : ✓ PASS
hermitian      : ✓ PASS
energy         : ✓ PASS
properties     : ✓ PASS
gpu            : ✓ PASS

✓ All required tests passed!
```

## Test Data

The tests use H2O data from `data/h2o/`:
- `fcidump.txt` - Molecular integrals
- `h2o-1em6-alpha.txt` - Alpha determinants (10^-6 threshold)

This provides a small but realistic test case (~100 determinants).

## Troubleshooting

### Import Errors
If you see import errors for numpy/scipy/mpi4py, install them:
```bash
pip install numpy scipy mpi4py
```

### Module Not Found: sbd
Build the Python bindings first:
```bash
python setup.py build_ext --inplace
```

### GPU Backend Fails
This is expected if GPU support wasn't compiled. The test will skip GPU tests automatically.

### Energy Mismatch
Small differences (<1e-6 Ha) are acceptable due to:
- Different convergence tolerances
- Numerical precision differences
- Iterative solver variations

Larger differences indicate a problem with the CSR export.

## Performance Notes

- **Small problems** (<1000 dets): Both SBD and SciPy are fast
- **Medium problems** (1000-10000 dets): SBD Davidson is faster
- **Large problems** (>10000 dets): Use SBD's built-in solvers, not CSR export

The CSR export is designed for **interoperability** with Python scientific computing, not for maximum performance on large problems.

## Integration with CuPy

For GPU acceleration with exported CSR matrices:

```python
import cupy as cp
from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu

# Export from SBD
csr = sbd.export_hamiltonian_csr(fcidump, adet, bdet, bit_length=20)

# Move to GPU
H_gpu = cp.sparse.csr_matrix((
    cp.array(csr['data']), 
    cp.array(csr['indices']), 
    cp.array(csr['indptr'])
))

# Solve on GPU
energy, wfn = eigsh_gpu(H_gpu, k=1, which='SA')
```

This allows using GPU eigensolvers from Python ecosystem while leveraging SBD's Hamiltonian construction.