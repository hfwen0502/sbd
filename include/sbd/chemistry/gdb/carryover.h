/**
@file sbd/chemistry/gdb/carryover.h
@brief functions for carryover bitstrings for gdb
*/
#ifndef SBD_CHEMISTRY_GDB_CARRYOVER_H
#define SBD_CHEMISTRY_GDB_CARRYOVER_H

namespace sbd {
  namespace gdb {

    // evaluate the diagonal part of the reduced density matrix
    template <typename ElemT>
    void GetDetWeightOrder(const std::vector<ElemT> & w,
			   size_t bit_length,
			   size_t norb,
			   MPI_Comm b_comm,
			   std::vector<size_t> & idets) {
      
    }
    
  }
}

#endif
