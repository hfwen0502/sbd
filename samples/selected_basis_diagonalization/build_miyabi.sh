#!/bin/sh
#------ qsub option --------#
#PBS -q regular-g
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=1:00:00
#PBS -W group_list=gz09
#PBS -j oe
#------- Program execution -------#
unset OMPI_MCA_mca_base_env_list
export MIYABI=G
cd /work/gz09/z30542/qcsc/sbd/samples/selected_basis_diagonalization

module unload nvidia/24.9
module load nvidia/25.9

make

