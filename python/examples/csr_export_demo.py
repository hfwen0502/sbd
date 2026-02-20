#!/usr/bin/env python3
"""
CSR Hamiltonian Export Demo

This demonstrates the intended usage of the CSR export feature for
integrating SBD with external eigensolvers (SciPy, CuPy, etc.).

STATUS: Feature not yet fully implemented. This is a design demo.

Usage:
    mpirun -np 1 python csr_export_demo.py
"""

import sys

def main():
    try:
        import sbd
        import numpy as np
        
        # Initialize SBD
        sbd.init(device='cpu', comm_backend='mpi')
        
        rank = sbd.get_rank()
        
        if rank == 0:
            print("="*70)
            print("CSR Hamiltonian Export Demo")
            print("="*70)
            print("\nThis demonstrates the intended API for CSR export.")
            print("Full implementation coming soon!\n")
        
        # Load molecular data
        fcidump_file = '../../data/h2o/fcidump.txt'
        adet_file = '../../data/h2o/h2o-1em3-alpha.txt'
        
        if rank == 0:
            print(f"Loading FCIDUMP: {fcidump_file}")
            print(f"Loading determinants: {adet_file}")
        
        fcidump = sbd.LoadFCIDump(fcidump_file)
        # For H2O: 13 orbitals, so we need total_bit_length >= 13 bits for alpha determinants
        # Using bit_length=64 (default size_t) and total_bit_length=64 for simplicity
        adet = sbd.LoadAlphaDets(adet_file, bit_length=64, total_bit_length=64)
        bdet = adet  # Closed shell
        
        if rank == 0:
            print(f"Number of determinants: {len(adet)}")
            print(f"Hilbert space dimension: {len(adet)} × {len(bdet)} = {len(adet) * len(bdet)}")
        
        # Try to export to CSR format
        if rank == 0:
            print("\nAttempting CSR export...")
        
        try:
            csr_data = sbd.export_hamiltonian_csr(
                fcidump=fcidump,
                adet=adet,
                bdet=bdet,
                bit_length=64,  # Must match the bit_length used in LoadAlphaDets
                max_nnz=int(1e8)
            )
            
            # If implemented, would use like this:
            if rank == 0:
                print("\n✓ CSR export successful!")
                print(f"  Matrix shape: {csr_data['shape']}")
                print(f"  Non-zeros: {csr_data['nnz']}")
                print(f"  Truncated: {csr_data['truncated']}")
                
                # Convert to scipy sparse matrix
                from scipy.sparse import csr_matrix
                import scipy.sparse.linalg as spla
                
                H = csr_matrix(
                    (csr_data['data'], csr_data['indices'], csr_data['indptr']),
                    shape=csr_data['shape']
                )
                
                print("\nDiagonalizing with SciPy eigsh...")
                eigenvalues, eigenvectors = spla.eigsh(H, k=1, which='SA')
                
                print(f"Ground state energy: {eigenvalues[0]:.6f} Hartree")
                
        except RuntimeError as e:
            if rank == 0:
                print(f"\n⚠️  Expected error (feature not yet implemented):")
                print(f"    {str(e)}")
                print("\n" + "="*70)
                print("Intended Usage (when implemented):")
                print("="*70)
                print("""
# Export Hamiltonian to CSR format
csr_data = sbd.export_hamiltonian_csr(
    fcidump=fcidump,
    adet=adet,
    bdet=bdet,
    bit_length=20,
    max_nnz=int(1e8)  # Limit to 10^8 non-zeros
)

# Use with SciPy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

H = csr_matrix((csr_data['data'], csr_data['indices'], csr_data['indptr']),
               shape=csr_data['shape'])

# Diagonalize
eigenvalues, eigenvectors = eigsh(H, k=1, which='SA')
print(f"Ground state energy: {eigenvalues[0]}")

# Or use with CuPy for GPU acceleration
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from cupyx.scipy.sparse.linalg import eigsh as eigsh_gpu

# Move data to GPU
data_gpu = cp.array(csr_data['data'])
indices_gpu = cp.array(csr_data['indices'])
indptr_gpu = cp.array(csr_data['indptr'])

H_gpu = csr_matrix_gpu((data_gpu, indices_gpu, indptr_gpu),
                        shape=csr_data['shape'])
eigenvalues_gpu, _ = eigsh_gpu(H_gpu, k=1, which='SA')
print(f"Ground state energy (GPU): {eigenvalues_gpu[0]}")
""")
                print("="*70)
        
        sbd.finalize()
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

# Made with Bob
