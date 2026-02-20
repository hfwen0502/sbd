# CSR Hamiltonian Export Design

## Overview

Export SBD's internal Hamiltonian representation to standard CSR (Compressed Sparse Row) format for use with external eigensolvers (SciPy, CuPy, etc.).

## Constraints

- **Maximum non-zeros**: 10^8 elements (~800 MB for double precision)
- **Target use case**: Small to medium problems suitable for direct eigensolvers
- **Memory efficient**: Build CSR incrementally, don't store full dense matrix

## API Design

### Python API

```python
import sbd
import scipy.sparse.linalg as spla

# Initialize and build Hamiltonian
sbd.init(device='cpu', comm_backend='mpi')
fcidump = sbd.LoadFCIDump('fcidump.txt')
adet = sbd.LoadAlphaDets('alpha.txt', bit_length=20, total_bit_length=20)
bdet = adet  # For closed shell

# Export to CSR format
csr_data = sbd.export_hamiltonian_csr(
    fcidump=fcidump,
    adet=adet,
    bdet=bdet,
    bit_length=20,
    max_nnz=int(1e8)  # Maximum non-zeros
)

# Returns dict with:
# {
#   'data': np.array of values,
#   'indices': np.array of column indices,
#   'indptr': np.array of row pointers,
#   'shape': (n, n),
#   'nnz': number of non-zeros,
#   'truncated': bool (True if hit max_nnz limit)
# }

# Use with SciPy
from scipy.sparse import csr_matrix
H = csr_matrix((csr_data['data'], csr_data['indices'], csr_data['indptr']), 
               shape=csr_data['shape'])

# Diagonalize
eigenvalues, eigenvectors = spla.eigsh(H, k=1, which='SA')
print(f"Ground state energy: {eigenvalues[0]}")
```

### C++ Implementation Strategy

```cpp
// In python/bindings.cpp
m.def("export_hamiltonian_csr",
    [](const sbd::FCIDump& fcidump,
       const std::vector<std::vector<size_t>>& adet,
       const std::vector<std::vector<size_t>>& bdet,
       size_t bit_length,
       size_t max_nnz) {
        
        // 1. Build Hamiltonian elements (reuse makeQCham logic)
        // 2. Collect all (i, j, value) triplets
        // 3. Sort by row, then column
        // 4. Convert to CSR format
        // 5. Return as Python dict
        
        py::dict result;
        result["data"] = data_array;
        result["indices"] = indices_array;
        result["indptr"] = indptr_array;
        result["shape"] = py::make_tuple(n, n);
        result["nnz"] = nnz;
        result["truncated"] = (nnz >= max_nnz);
        
        return result;
    },
    py::arg("fcidump"),
    py::arg("adet"),
    py::arg("bdet"),
    py::arg("bit_length"),
    py::arg("max_nnz") = 100000000);
```

## Implementation Steps

### Step 1: Collect Hamiltonian Elements
- Reuse `makeQCham` or `makeQChamDiagTerms` logic
- Store as vector of (row, col, value) triplets
- Stop if exceeds max_nnz

### Step 2: Sort Triplets
- Sort by row index (primary), column index (secondary)
- Required for CSR format

### Step 3: Build CSR Arrays
```cpp
// Pseudocode
std::vector<double> data;
std::vector<int> indices;
std::vector<int> indptr(n+1, 0);

int current_row = 0;
indptr[0] = 0;

for (auto& triplet : sorted_triplets) {
    while (current_row < triplet.row) {
        current_row++;
        indptr[current_row] = data.size();
    }
    data.push_back(triplet.value);
    indices.push_back(triplet.col);
}

// Fill remaining indptr entries
while (current_row < n) {
    current_row++;
    indptr[current_row] = data.size();
}
```

### Step 4: Return to Python
- Convert C++ vectors to NumPy arrays
- Return as dictionary

## Memory Estimates

For max_nnz = 10^8:
- `data`: 8 bytes × 10^8 = 800 MB
- `indices`: 4 bytes × 10^8 = 400 MB  
- `indptr`: 4 bytes × n (typically << 10^8)
- **Total**: ~1.2 GB

## Test Cases

### Test 1: Small H2O (10 determinants)
- Matrix size: ~100 × 100
- Non-zeros: ~1000
- Compare SBD vs SciPy eigsh

### Test 2: Medium Problem (1000 determinants)
- Matrix size: ~10^6 × 10^6
- Non-zeros: ~10^7
- Verify CSR format correctness

### Test 3: Truncation Test
- Set max_nnz = 1000
- Verify truncation flag is set
- Check that first 1000 elements are correct

## Limitations

1. **Single rank only**: CSR export runs on rank 0 (no MPI distribution)
2. **Memory bound**: Limited by max_nnz parameter
3. **CPU only initially**: GPU (CuPy) support in future
4. **No symmetry exploitation**: Stores full matrix (could optimize for Hermitian)

## Future Enhancements

1. **Distributed CSR**: Keep matrix distributed across MPI ranks
2. **GPU support**: Return CuPy sparse matrix
3. **Symmetry**: Store only lower/upper triangle
4. **Iterative build**: Stream to disk if too large for memory