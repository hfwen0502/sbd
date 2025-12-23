/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_H
#define SBD_CHEMISTRY_GDB_HELPER_H

namespace sbd {

  /**
     DetBasisMapper is an array to find relation between dets.
   */
  struct DetIndexMap {
    std::vector<size_t> AdetToDetLen;
    std::vector<size_t> BdetToDetLen;
    std::vector<size_t*> AdetToBdetSM;
    std::vector<size_t*> AdetToDetSM;
    std::vector<size_t*> BdetToAdetSM;
    std::vector<size_t*> BdetToDetSM;
  };

  /**
     Labels connected 
   */
  struct DetExcitationHelper {
    std::vector<size_t> SelfFromAdetLen;
    std::vector<size_t> SelfFromBdetLen;
    std::vector<size_t> SinglesFromAdetLen;
    std::vector<size_t> SinglesFromBdetLen;
    std::vector<size_t*> SelfFromAdetSM;
    std::vector<size_t*> SelfFromBdetSM;
    std::vector<size_t*> SinglesFromAdetSM;
    std::vector<size_t*> SinglesFromBdetSM;
  };

  void getHalfDets(const std::vector<std::vector<size_t>> & det,
		   size_t bit_length,
		   size_t norb,
		   std::vector<std::vector<size_t>> & adet,
		   std::vector<std::vector<size_t>> & bdet,
		   std::vector<size_t> & adet_count,
		   std::vector<size_t> & bdet_count) {
    adet.resize(det.size());
    bdet.resize(det.size());
    for(size_t i=0; i < det.size(); i++) {
      getAdet(det[i],bit_length,norb,adet[i]);
      getBdet(det[i],bit_length,norb,bdet[i]);
    }
    std::sort(adet.begin(),adet.end(),
	      [](const std::vector<size_t> & x,
		 const std::vector<size_t> & y) {
		return x < y;
	      });
    std::sort(bdet.begin(),bdet.end(),
	      [](const std::vector<size_t> & x,
		 const std::vector<size_t> & y) {
		return x < y;
	      });
    auto adet_sorted = adet;
    auto bdet_sorted = bdet;
    adet.erase(std::unique(adet.begin(),adet.end()),adet.end());
    bdet.erase(std::unique(bdet.begin(),bdet.end()),bdet.end());
    adet_count.resize(adet.size(),0);
    bdet_count.resize(bdet.size(),0);
    size_t u=0;
    size_t count=0;
    for(size_t k=0; k < adet_sorted.size(); k++) {
      if( adet_sorted[k] != adet[u] ) {
	adet_count[u] = count;
	u++;
	count = 0;
      } else {
	count++;
      }
    }
    adet_count[adet.size()-1] = count;
    u=0;
    count=0;
    for(size_t k=0; k < bdet_sorted.size(); k++) {
      if( bdet_sorted[k] != bdet[u] ) {
	bdet_count[u] = count;
	u++;
	count=0;
      } else {
	count++;
      }
    }
    bdet_count[bdet.size()-1] = count;
  }

  void makeDetIndexMap(const std::vector<std::vector<size_t>> & det,
		       const std::vector<std::vector<size_t>> & adet,
		       const std::vector<std::vector<size_t>> & bdet,
		       const std::vector<size_t> & adet_count,
		       const std::vector<size_t> & bdet_count,
		       size_t bit_length,
		       size_t norb,
		       std::vector<std::vector<size_t>> & adet_to_bdet,
		       std::vector<std::vector<size_t>> & adet_to_det,
		       std::vector<std::vector<size_t>> & bdet_to_adet,
		       std::vector<std::vector<size_t>> & bdet_to_det) {
    adet_to_bdet.reisze(adet.size());
    adet_to_det.resize(adet.size());
    bdet_to_adet.resize(bdet.size());
    bdet_to_det.resize(bdet.size());
    std::vector<size_t> adet_count(adet.size());
    std::vector<size_t> bdet_count(bdet.size());
    for(size_t k=0; k < adet.size(); k++) {
      adet_to_bdet[k].reserve(adet_count[k]);
      adet_to_det[k].reserve(adet_count[k]);
    }
    for(size_t k=0; k < bdet.size(); k++) {
      bdet_to_adet[k].reserve(bdet_count[k]);
      bdet_to_det[k].reserve(bdet_count[k]);
    }
    size_t hdet_size = (norb + bit_length - 1) / bit_length;
    std::vector<size_t> adet_temp(hdet_size);
    std::vector<size_t> bdet_temp(hdet_size);
    for(size_t i=0; i < det.size(); i++) {
      getAdet(det[i],bit_length,norb,adet_temp);
      getBdet(det[i],bit_length,norb,bdet_temp);
      auto itia = std::lower_bound(adet.begin(),adet.end(),adet_temp);
      auto itib = std::lower_bound(bdet.begin(),bdet.end(),bdet_temp);
      size_t ia = std::distance(adet.begin(),itia);
      size_t ib = std::distance(bdet.begin(),itib);
      adet_to_bdet[ia].push_back(ib);
      bdet_to_adet[ib].push_back(ia);
      adet_to_det[ia].push_back(i);
      bdet_to_det[ib].push_back(i);
    }
  }

  void makeHdetSingles(const std::vector<std::vector<size_t>> & hdet_bra,
		       const std::vector<std::vector<size_t>> & hdet_ket,
		       size_t bit_length,
		       size_t norb,
		       std::vector<std::vector<size_t>> & samedet,
		       std::vector<std::vector<size_t>> & singles) {
    samedet.resize(hdet_bra.size());
    singles.resize(hdet_bra.size());
#pragma omp parallel for
    for(size_t i=0; i < hdet_bra.size(); i++) {
      size_t zcount = 0;
      size_t scount = 0;
      for(size_t j=0; j < hdet_ket.size(); j++) {
	int d = difference(hdet_bra[i],hdet_ket[j],bit_length,norb);
	if( d == 0 ) zcount++;
	if( d == 2 ) scount++;
      }
      samedet.reserve(zcount);
      singles.reserve(scount);
      for(size_t j=0; j < hdet_ket.size(); j++) {
	int d = difference(hdet_bra[i],hdet_ket[j],bit_length,norb);
	if( d == 0 ) {
	  samedet[i].push_back(j);
	} else if( d == 2 ) {
	  singles[i].push_back(j);
	}
      }
    }
  }

  void MpiSlide(const DetIndexMap & send_map,
		const std::vector<size_t> & send_storage,
		DetIndexMap & recv_map,
		std::vector<size_t> & recv_storage,
		int slide,
		MPI_Comm comm) {
    MpiSlide(send_map.AdetToDetLen,recv_map.AdetToDetLen,slide,comm);
    MpiSlide(send_map.BdetToDetLen,recv_map.BdetToDetLen,slide,comm);
    MpiSlide(send_storage,recv_storage,slide,comm);
    size_t recv_size_a = 0;
    size_t recv_size_b = 0;
    for(size_t i=0; i < recv_map.AdetToDetLen.size(); i++) {
      recv_size_a += recv_map.AdetToDetLen[i];
    }
    for(size_t i=0; i < map_send.AdetToDetLen.size(); i++) {
      recv_size_b += recv_map.BdetToDetLen[i];
    }
    recv_map.AdetToDetSM(recv_map.AdetToDetLen.size());
    recv_map.AdetToBdetSM(recv_map.AdetToBdetLen.size());
    recv_map.BdetToDetSM(recv_map.BdetToDetLen.size());
    recv_map.BdetToAdetSM(recv_map.BdetToAdetLen.size());
    size_t * begin = recv_memory.data();
    size_t counter = 0;
    for(size_t i=0; i < recv_size_a; i++) {
      recv_map.AdetToDetSM[i] = begin + counter;
      counter += helper.AdetToDetLen[i];
    }
    for(size_t i=0; i < recv_size_a; i++) {
      recv_map.AdetToBdetSM[i] = begin + counter;
      counter += helper.AdetToDetLen[i];
    }
    for(size_t i=0; i < recv_size_b; i++) {
      recv_map.BdetToDetSM[i] = begin + counter;
      counter += helper.BdetToDetLen[i];
    }
    for(size_t i=0; i < recv_size_b; i++) {
      recv_map.BdetToAdetSM[i] = begin + counter;
      counter += helper.BdetToDetLen[i];
    }
  }

  void DetBasisCommunicator(MPI_Comm comm,
			    int h_comm_size,
			    int b_comm_size,
			    int t_comm_size,
			    MPI_Comm & h_comm,
			    MPI_Comm & b_comm,
			    MPI_Comm & t_comm) {
    int mpi_size; MPI_Comm_size(comm,&mpi_size);
    int mpi_rank; MPI_Comm_rank(comm,&mpi_rank);
    int a_comm_size = b_comm_size * t_comm_size;
    MPI_Comm a_comm;
    int a_comm_color = mpi_rank / a_comm_size;
    int h_comm_color = mpi_rank % a_comm_size;
    MPI_Comm_split(comm,a_comm_color,mpi_rank,&a_comm);
    MPI_Comm_split(comm,h_comm_color,mpi_rank,&h_comm);

    int mpi_size_a; MPI_Comm_size(a_comm,&mpi_size_a);
    int mpi_rank_a; MPI_Comm_rank(a_comm,&mpi_rank_a);

    int t_comm_color = mpi_rank_a % b_comm_size;
    int b_comm_color = mpi_rank_a / b_comm_size;
    MPI_Comm_split(a_comm,t_comm_color,mpi_rank,&t_comm);
    MPI_Comm_split(a_comm,b_comm_color,mpi_rank,&b_comm);
  }


  

  
  
}

#endif
