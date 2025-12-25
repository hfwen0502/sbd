/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for general determinant basis
*/
#ifndef SBD_CHEMISTRY_GDB_MULT_H
#define SBD_CHEMISTRY_GDB_MULT_H

namespace sbd {

  namespace gdb {

    template <typename ElemT>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<std::vector<size_t*>> & ih,
	      const std::vector<std::vector<size_t*>> & jh,
	      const std::vector<std::vector<ElemT*>> & hij,
	      const std::vector<std::vector<size_t>> & len,
	      const std::vector<int> & slide,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      
    }
    
    template <typename ElemT>
    void mult(const std::vector<ElemT> & hii,
	      const std::vector<ElemT> & wk,
	      std::vector<ElemT> & wb,
	      size_t bit_length,
	      size_t norb,
	      const std::vector<std::vector<size_t>> & det,
	      const std::vector<std::vector<size_t>> & adet,
	      const std::vector<std::vector<size_t>> & bdet,
	      const DetIndexMap & idxmap,
	      const std::vector<ExcitationLookup> & exidx,
	      const ElemT & I0,
	      const oneInt<ElemT> & I1,
	      const twoInt<ElemT> & I2,
	      MPI_Comm h_comm,
	      MPI_Comm b_comm,
	      MPI_Comm t_comm) {
      
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      
    }

  } // end namespace gdb
  
} // end namespace sbd

#endif
