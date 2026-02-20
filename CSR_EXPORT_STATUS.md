# CSR Hamiltonian Export - Implementation Status

## Current Status: API Defined, Full Implementation Pending

### ✅ Completed (Phase 1)

1. **Design Document** (`CSR_EXPORT_DESIGN.md`)
   - Complete API specification
   - Memory constraints (10^8 element limit)
   - Implementation strategy
   - Test plan

2. **C++ Placeholder** (`python/bindings.cpp`)
   - Function signature defined
   - Parameter validation
   - Clear error message for users
   - Documentation

3. **Python API** (`python/__init__.py`)
   - `sbd.export_hamiltonian_csr()` function
   - Complete docstring with examples
   - Error handling

4. **Demo Script** (`python/examples/csr_export_demo.py`)
   - Shows intended usage
   - Demonstrates SciPy integration
   - Includes CuPy example

5. **Git Branch** (`feature/csr-hamiltonian-export`)
   - Based on `simplified-api`
   - Ready for full implementation

### 🔄 Pending (Phase 2)

The full implementation requires:

#### 1. Hamiltonian Construction (~150 lines C++)
```cpp
// Need to implement in python/bindings.cpp
struct HamiltonianTriplet {
    size_t row;
    size_t col;
    double value;
};

std::vector<HamiltonianTriplet> build_hamiltonian_triplets(
    const FCIDump& fcidump,
    const std::vector<std::vector<size_t>>& adet,
    const std::vector<std::vector<size_t>>& bdet,
    size_t bit_length,
    size_t max_nnz
);
```

**Challenges:**
- Reuse existing `makeQCham` logic
- Handle Slater-Condon rules efficiently
- Manage memory for large problems
- Stop gracefully at max_nnz limit

#### 2. CSR Conversion (~50 lines C++)
```cpp
void triplets_to_csr(
    const std::vector<HamiltonianTriplet>& triplets,
    size_t n,
    std::vector<double>& data,
    std::vector<int>& indices,
    std::vector<int>& indptr
);
```

**Steps:**
- Sort triplets by (row, col)
- Build CSR arrays
- Handle symmetric matrix (optional optimization)

#### 3. Python Integration (~20 lines C++)
```cpp
// Return as NumPy arrays
py::dict result;
result["data"] = py::array_t<double>(data.size(), data.data());
result["indices"] = py::array_t<int>(indices.size(), indices.data());
result["indptr"] = py::array_t<int>(indptr.size(), indptr.data());
result["shape"] = py::make_tuple(n, n);
result["nnz"] = data.size();
result["truncated"] = (data.size() >= max_nnz);
```

#### 4. Testing
- Small H2O test (10 determinants)
- Medium test (100 determinants)
- Truncation test (max_nnz limit)
- Compare SBD vs SciPy energies

#### 5. Documentation
- Update README with CSR export examples
- Add to API documentation
- Performance guidelines

## Estimated Effort

- **Hamiltonian construction**: 4-6 hours
  - Study existing code
  - Implement triplet collection
  - Handle edge cases
  
- **CSR conversion**: 1-2 hours
  - Straightforward algorithm
  - Testing

- **Integration & testing**: 2-3 hours
  - Python wrapper
  - Test cases
  - Bug fixes

**Total**: 7-11 hours for full implementation

## Why Not Implemented Yet?

The CSR export requires deep understanding of SBD's internal Hamiltonian construction logic, which is complex and distributed across multiple files:

- `include/sbd/chemistry/basic/qcham.h` - Hamiltonian building
- `include/sbd/chemistry/basic/excitation.h` - Slater-Condon rules
- `include/sbd/chemistry/basic/helpers.h` - Helper structures
- `include/sbd/chemistry/tpb/mult.h` - Matrix-vector multiplication

Rather than rush an incomplete implementation, we've:
1. Defined the complete API
2. Created placeholder with clear error message
3. Documented the design thoroughly
4. Set up infrastructure for future implementation

## Next Steps

When ready to implement:

1. **Study Phase** (2 hours)
   - Deep dive into `makeQCham` function
   - Understand helper structures
   - Map out data flow

2. **Prototype** (3 hours)
   - Implement basic triplet collection
   - Test on tiny system (2-3 determinants)
   - Verify correctness

3. **Optimize** (2 hours)
   - Add memory limits
   - Handle large systems
   - Performance tuning

4. **Test & Document** (2 hours)
   - Comprehensive test suite
   - Update documentation
   - Create examples

## Current Workaround

For users who need external eigensolvers now:
1. Use SBD's built-in Davidson/Lanczos (recommended)
2. Export wavefunction and reconstruct Hamiltonian in Python (slow)
3. Wait for full CSR implementation

## Branch Status

```
feature/csr-hamiltonian-export (current)
├── CSR_EXPORT_DESIGN.md (complete design)
├── CSR_EXPORT_STATUS.md (this file)
├── python/bindings.cpp (placeholder function)
├── python/__init__.py (Python API)
└── python/examples/csr_export_demo.py (usage demo)
```

Ready to merge placeholder to `simplified-api` or continue with full implementation.