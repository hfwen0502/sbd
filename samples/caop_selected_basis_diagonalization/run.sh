mpirun -np 8 ./diag --hamfile ham_chain_12.txt \
       --basisfiles basisr0.txt,basisr1.txt --sort_basis 0 --redist_basis 1 \
       --t_comm_size 2 --b_comm_size 2 \
       --method 1 --iteration 10 --block 10 --tolerance 1.0e-4 --fermionsign 0 \
       --system_size 12
