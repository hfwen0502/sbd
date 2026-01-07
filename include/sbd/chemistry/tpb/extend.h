/**
@file sbd/chemistry/tpb/extend.h
@brief function to extend the determinant basis
*/
#ifndef SBD_CHEMISTRY_TPB_EXTEND_H
#define SBD_CHEMISTRY_TPB_EXTEND_H

namespace sbd {

  template <typename ElemT, typename RealT>
  void extendhalfdetsingles(const std::vector<ElemT> & w,
			    const std::vector<std::vector<size_t>> & adet,
			    const std::vector<std::vector<size_t>> & bdet,
			    size_t bit_length,
			    size_t norb,
			    const size_t adet_comm_size,
			    const size_t bdet_comm_size,
			    MPI_Comm b_comm,
			    RealT cutoff) {

    int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
    int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);

    size_t adet_begin = 0;
    size_t adet_end   = adet.size();
    size_t bdet_begin = 0;
    size_t bdet_end   = bdet.size();
    int a_comm_size = static_cast<int>(adet_comm_size);
    int b_comm_size = static_cast<int>(bdet_comm_size);
    assert( mpi_size_b == a_comm_size * b_comm_size );
    int a_comm_rank = mpi_rank_b / b_comm_size;
    int b_comm_rank = mpi_rank_b % b_comm_size;

    get_mpi_range(a_comm_size,a_comm_rank,adet_begin,adet_end);
    get_mpi_range(b_comm_size,b_comm_rank,bdet_begin,bdet_end);

    size_t adet_size = adet_end - adet_begin;
    size_t bdet_size = bdet_end - bdet_begin;
    size_t det_size = adet_size * bdet_size;
    size_t det_lengh = ( 2*norb + bit_length - 1 ) / bit_length;
    std::vector<std::vector<size_t>> det(det_size);
    for(size_t ia=adet_begin; ia < adet_end; ia++) {
      for(size_t ib=bdet_begin; ib < bdet_end; ib++) {
	size_t idx = (ia - adet_begin) * bdet_size + ib - bdet_begin;
	det[idx].resize(det_length);
	DetFromAlphaBeta(adet[ia],bdet[ib],bit_length,norb,det[idx]);
      }
    }
    // now writing
  }
  
}

#endif
