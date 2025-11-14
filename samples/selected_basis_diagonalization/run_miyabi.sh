#!/bin/sh
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=8:mpiprocs=8
#PBS -l walltime=2:00:00
#PBS -W group_list=gz09
#PBS -j oe
#------- Program execution -------#
unset OMPI_MCA_mca_base_env_list
export MIYABI=G
cd /work/gz09/z30542/qcsc/sbd/samples/selected_basis_diagonalization

export OMP_NUM_THREADS=1
# mpirun -x PATH -x LD_LIBRARY_PATH -x MIYABI -n 1 ./diag --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.txt --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 --adet_comm_size 1 --bdet_comm_size 1 --task_comm_size 1 --init 0 --shuffle 0 --carryover_ratio 0.5 --savename wf --dump_matrix_form_wf matrixformwf.txt --carryoverfile carryover.txt --rdm 0

mpirun -x PATH -x LD_LIBRARY_PATH -x MIYABI -n 8 ./diag --fcidump fcidump_Fe4S4.txt --adetfile AlphaDets.bin --method 0 --block 10 --iteration 1 --tolerance 1.0e-4 --eps 1.0e-4 --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2 --init 0 --shuffle 0 --carryover_ratio 0.5 --savename wf --carryoverfile carryover.txt --rdm 0 --max_time 24000

