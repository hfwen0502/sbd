#!/usr/bin/env python3
"""
Minimal CSR export test with tiny system
"""

import sys
import sbd
import numpy as np

def main():
    # Initialize
    sbd.init(device='cpu', comm_backend='mpi')
    
    rank = sbd.get_rank()
    
    if rank == 0:
        print("="*70)
        print("Minimal CSR Export Test")
        print("="*70)
    
    # Load H2O data but use smallest determinant file
    fcidump_file = '../../data/h2o/fcidump.txt'
    adet_file = '../../data/h2o/h2o-1em8-alpha.txt'  # Smallest file
    
    if rank == 0:
        print(f"\nLoading FCIDUMP: {fcidump_file}")
        print(f"Loading determinants: {adet_file}")
    
    fcidump = sbd.LoadFCIDump(fcidump_file)
    adet = sbd.LoadAlphaDets(adet_file, bit_length=64, total_bit_length=64)
    bdet = adet
    
    if rank == 0:
        print(f"Number of determinants: {len(adet)}")
        print(f"Hilbert space: {len(adet)} × {len(bdet)} = {len(adet) * len(bdet)}")
    
    # Try CSR export with very small max_nnz to limit computation
    if rank == 0:
        print("\nAttempting CSR export with max_nnz=1000...")
    
    try:
        csr_data = sbd.export_hamiltonian_csr(
            fcidump=fcidump,
            adet=adet,
            bdet=bdet,
            bit_length=64,
            max_nnz=1000  # Very small limit
        )
        
        if rank == 0:
            print("\n✓ CSR export successful!")
            print(f"  Shape: {csr_data['shape']}")
            print(f"  Non-zeros: {csr_data['nnz']}")
            print(f"  Truncated: {csr_data['truncated']}")
            
    except Exception as e:
        if rank == 0:
            print(f"\n✗ CSR export failed: {e}")
            import traceback
            traceback.print_exc()
        return 1
    
    sbd.finalize()
    return 0

if __name__ == '__main__':
    sys.exit(main())

# Made with Bob
