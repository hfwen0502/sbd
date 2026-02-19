#!/usr/bin/env python3
"""
Example: H2O calculation with backend selection

This example demonstrates backend selection using environment variables.

IMPORTANT: Due to pybind11 limitations, only ONE backend can be loaded per
Python process. The backend must be selected BEFORE importing sbd.

Usage:
    # Use CPU backend
    SBD_BACKEND=cpu mpirun -np 4 python h2o_backend_selection.py
    
    # Use GPU backend
    SBD_BACKEND=gpu mpirun -np 4 python h2o_backend_selection.py
    
    # Auto-select (prefers GPU)
    mpirun -np 4 python h2o_backend_selection.py
"""

import os
import sys
from mpi4py import MPI

# Parse command line to set environment variable
if len(sys.argv) > 1:
    backend = sys.argv[1].lower()
    if backend in ['cpu', 'gpu', 'auto']:
        os.environ['SBD_BACKEND'] = backend
        os.environ['SBD_VERBOSE'] = '1'  # Show backend selection
    else:
        print(f"Invalid backend: {backend}")
        print("Usage: python h2o_backend_selection.py [cpu|gpu|auto]")
        sys.exit(1)

# NOW import sbd (backend is selected during import)
import sbd

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Print backend information
    if rank == 0:
        print("\n" + "="*70)
        print("SBD Backend Selection Example - H2O Calculation")
        print("="*70)
        sbd.print_backend_info()
        print()
    
    # Configure calculation
    config = sbd.TPB_SBD()
    config.max_it = 100
    config.eps = 1e-6
    config.method = 0  # Davidson
    config.do_rdm = 0  # Density only
    
    # File paths (adjust as needed)
    fcidump_file = "fcidump_h2o.txt"
    adet_file = "alphadets_h2o.txt"
    
    if rank == 0:
        print("Configuration:")
        print(f"  Backend: {sbd.get_backend().upper()}")
        print(f"  Method: Davidson")
        print(f"  Max iterations: {config.max_it}")
        print(f"  Tolerance: {config.eps}")
        print(f"  FCIDUMP file: {fcidump_file}")
        print(f"  Alpha dets file: {adet_file}")
        print()
    
    # Run calculation
    try:
        if rank == 0:
            print("Running TPB diagonalization...")
        
        results = sbd.tpb_diag_from_files(
            comm=comm,
            sbd_data=config,
            fcidumpfile=fcidump_file,
            adetfile=adet_file
        )
        
        if rank == 0:
            print("\n" + "="*70)
            print("Results")
            print("="*70)
            print(f"Backend used: {sbd.get_backend().upper()}")
            print(f"Ground state energy: {results['energy']:.10f}")
            print(f"Density: {results['density'][:10]}...")  # First 10 elements
            print(f"Number of carryover determinants: {len(results['co_adet'])}")
            print("="*70 + "\n")
    
    except FileNotFoundError as e:
        if rank == 0:
            print(f"\n✗ Error: {e}")
            print("\nThis example requires H2O input files:")
            print(f"  - {fcidump_file}")
            print(f"  - {adet_file}")
            print("\nPlease provide these files or modify the file paths in the script.")
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Error during calculation: {e}")
            import traceback
            traceback.print_exc()

def compare_backends():
    """
    Compare CPU and GPU backends by running in separate processes
    """
    import subprocess
    import time
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank != 0:
        return
    
    print("\n" + "="*70)
    print("Backend Comparison (Running in Separate Processes)")
    print("="*70)
    
    available = sbd.available_backends()
    print(f"Compiled backends: {available}\n")
    
    if len(available) < 2:
        print(f"Only {len(available)} backend(s) available.")
        print("Cannot compare backends.")
        return
    
    results = {}
    
    for backend_name in available:
        print(f"Testing {backend_name.upper()} backend...")
        
        # Run in subprocess with specific backend
        env = os.environ.copy()
        env['SBD_BACKEND'] = backend_name
        
        start = time.time()
        try:
            # This would need to be a separate script
            # For now, just show the concept
            print(f"  Would run: SBD_BACKEND={backend_name} mpirun -np 4 python script.py")
            print(f"  (Comparison requires separate script runs)")
        except Exception as e:
            print(f"  Error: {e}")
        
        elapsed = time.time() - start
    
    print("="*70 + "\n")
    print("Note: To properly compare backends, run the script twice:")
    print("  SBD_BACKEND=cpu mpirun -np 4 python h2o_backend_selection.py")
    print("  SBD_BACKEND=gpu mpirun -np 4 python h2o_backend_selection.py")
    print()

if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare_backends()
    else:
        main()

# Made with Bob
