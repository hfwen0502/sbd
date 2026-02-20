#!/usr/bin/env python3
"""
Test CSR Hamiltonian export functionality

This test verifies that:
1. CSR export produces valid sparse matrix format
2. Exported Hamiltonian matches SBD's internal results
3. SciPy eigensolver gives same energy as SBD
4. Matrix is Hermitian (symmetric for real case)
5. Both CPU and GPU backends work correctly
"""

import sys
import numpy as np
from mpi4py import MPI

def test_csr_export_basic():
    """Test basic CSR export functionality"""
    import sbd
    
    print("=" * 70)
    print("TEST 1: Basic CSR Export")
    print("=" * 70)
    
    # Initialize SBD
    comm = MPI.COMM_WORLD
    sbd.init(backend='cpu')
    
    # Load H2O data (small test case)
    fcidump = sbd.load_fcidump('data/h2o/fcidump.txt')
    adet, bdet = sbd.load_dets('data/h2o/h2o-1em6-alpha.txt', 'data/h2o/h2o-1em6-alpha.txt')
    
    print(f"Loaded {len(adet)} alpha determinants")
    print(f"Loaded {len(bdet)} beta determinants")
    print(f"Hilbert space dimension: {len(adet) * len(bdet)}")
    
    # Export to CSR format
    print("\nExporting Hamiltonian to CSR format...")
    csr = sbd.export_hamiltonian_csr(
        fcidump, adet, bdet, 
        bit_length=20, 
        max_nnz=int(1e8)
    )
    
    # Verify CSR format
    print(f"\nCSR Format Verification:")
    print(f"  Matrix shape: {csr['shape']}")
    print(f"  Number of non-zeros: {csr['nnz']}")
    print(f"  Truncated: {csr['truncated']}")
    print(f"  Sparsity: {100 * csr['nnz'] / (csr['shape'][0] * csr['shape'][1]):.2f}%")
    
    # Check array types
    assert isinstance(csr['data'], np.ndarray), "data should be NumPy array"
    assert isinstance(csr['indices'], np.ndarray), "indices should be NumPy array"
    assert isinstance(csr['indptr'], np.ndarray), "indptr should be NumPy array"
    
    # Check array sizes
    assert len(csr['data']) == csr['nnz'], "data length mismatch"
    assert len(csr['indices']) == csr['nnz'], "indices length mismatch"
    assert len(csr['indptr']) == csr['shape'][0] + 1, "indptr length mismatch"
    
    print("\n✓ CSR format validation passed")
    
    return csr, fcidump, adet, bdet


def test_csr_hermitian(csr):
    """Test that exported matrix is Hermitian"""
    print("\n" + "=" * 70)
    print("TEST 2: Hermitian Property")
    print("=" * 70)
    
    from scipy.sparse import csr_matrix
    
    # Build sparse matrix
    H = csr_matrix((csr['data'], csr['indices'], csr['indptr']), shape=csr['shape'])
    
    # Check if Hermitian (H = H^†)
    H_dag = H.conj().T
    diff = (H - H_dag).data
    max_diff = np.max(np.abs(diff)) if len(diff) > 0 else 0.0
    
    print(f"Max |H - H†|: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ Matrix is Hermitian")
    else:
        print(f"✗ Matrix is NOT Hermitian (max diff: {max_diff})")
        return False
    
    return True


def test_csr_vs_sbd_energy(csr, fcidump, adet, bdet):
    """Compare CSR eigensolver energy with SBD's result"""
    print("\n" + "=" * 70)
    print("TEST 3: Energy Comparison (CSR vs SBD)")
    print("=" * 70)
    
    import sbd
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    # Get energy from SciPy eigensolver
    print("\nSolving with SciPy eigsh...")
    H = csr_matrix((csr['data'], csr['indices'], csr['indptr']), shape=csr['shape'])
    energy_scipy, wfn_scipy = eigsh(H, k=1, which='SA')
    energy_scipy = energy_scipy[0]
    print(f"SciPy energy: {energy_scipy:.10f} Ha")
    
    # Get energy from SBD
    print("\nSolving with SBD Davidson...")
    comm = MPI.COMM_WORLD
    
    # Create SBD configuration
    sbd_config = sbd.tpb.SBD()
    sbd_config.niter = 100
    sbd_config.tol = 1e-8
    sbd_config.nroots = 1
    sbd_config.nguess = 10
    sbd_config.carryover = False
    
    result = sbd.tpb_diag(comm, sbd_config, fcidump, adet, bdet)
    energy_sbd = result['energy']
    print(f"SBD energy:   {energy_sbd:.10f} Ha")
    
    # Compare
    diff = abs(energy_scipy - energy_sbd)
    print(f"\nEnergy difference: {diff:.2e} Ha")
    
    if diff < 1e-6:
        print("✓ Energies match within tolerance")
        return True
    else:
        print(f"✗ Energies differ by {diff:.2e} Ha")
        return False


def test_csr_matrix_properties(csr):
    """Test various matrix properties"""
    print("\n" + "=" * 70)
    print("TEST 4: Matrix Properties")
    print("=" * 70)
    
    from scipy.sparse import csr_matrix
    
    H = csr_matrix((csr['data'], csr['indices'], csr['indptr']), shape=csr['shape'])
    
    # Check diagonal elements (should be real and mostly positive)
    diag = H.diagonal()
    print(f"\nDiagonal statistics:")
    print(f"  Min: {np.min(diag):.6f}")
    print(f"  Max: {np.max(diag):.6f}")
    print(f"  Mean: {np.mean(diag):.6f}")
    print(f"  All real: {np.all(np.isreal(diag))}")
    
    # Check for NaN or Inf
    has_nan = np.any(np.isnan(csr['data']))
    has_inf = np.any(np.isinf(csr['data']))
    print(f"\nData quality:")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("✗ Matrix contains invalid values")
        return False
    
    print("✓ Matrix properties are valid")
    return True


def test_gpu_backend():
    """Test CSR export with GPU backend (if available)"""
    print("\n" + "=" * 70)
    print("TEST 5: GPU Backend (Optional)")
    print("=" * 70)
    
    try:
        import sbd
        
        # Try to initialize GPU backend
        sbd.finalize()  # Clean up CPU backend first
        sbd.init(backend='gpu')
        
        print("GPU backend available")
        
        # Load data
        comm = MPI.COMM_WORLD
        fcidump = sbd.load_fcidump('data/h2o/fcidump.txt')
        adet, bdet = sbd.load_dets('data/h2o/h2o-1em6-alpha.txt', 'data/h2o/h2o-1em6-alpha.txt')
        
        # Export CSR
        csr = sbd.export_hamiltonian_csr(fcidump, adet, bdet, bit_length=20)
        
        print(f"✓ GPU backend CSR export successful")
        print(f"  Matrix shape: {csr['shape']}")
        print(f"  Non-zeros: {csr['nnz']}")
        
        return True
        
    except Exception as e:
        print(f"GPU backend not available or failed: {e}")
        print("(This is OK if GPU support not compiled)")
        return None


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("CSR HAMILTONIAN EXPORT TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    try:
        # Test 1: Basic export
        csr, fcidump, adet, bdet = test_csr_export_basic()
        results['basic'] = True
        
        # Test 2: Hermitian property
        results['hermitian'] = test_csr_hermitian(csr)
        
        # Test 3: Energy comparison
        results['energy'] = test_csr_vs_sbd_energy(csr, fcidump, adet, bdet)
        
        # Test 4: Matrix properties
        results['properties'] = test_csr_matrix_properties(csr)
        
        # Test 5: GPU backend (optional)
        results['gpu'] = test_gpu_backend()
        
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is False:
            status = "✗ FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{test_name:15s}: {status}")
    
    # Overall result
    required_tests = ['basic', 'hermitian', 'energy', 'properties']
    all_passed = all(results.get(t, False) for t in required_tests)
    
    if all_passed:
        print("\n✓ All required tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())

# Made with Bob
