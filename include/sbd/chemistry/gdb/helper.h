/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_H
#define SBD_CHEMISTRY_GDB_HELPER_H

namespace sbd {

  namespace gdb {

    /**
       DetBasisMapper is an array to find relation between dets.
    */
    struct DetIndexMap {
      std::vector<size_t> storage;
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
    struct ExcitationLookup {
      std::vector<size_t> storage;
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
    
    void makeDetIndexMap(const std::vector<std::vector<size_t>> & det,
			 const std::vector<std::vector<size_t>> & adet,
			 const std::vector<std::vector<size_t>> & bdet,
			 const std::vector<size_t> & adet_count,
			 const std::vector<size_t> & bdet_count,
			 size_t bit_length,
			 size_t norb,
			 DetIndexMap & idxmap) {
      std::vector<std::vector<size_t>> adet_to_bdet;
      std::vector<std::vector<size_t>> adet_to_det;
      std::vector<std::vector<size_t>> bdet_to_adet;
      std::vector<std::vector<size_t>> bdet_to_det;
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,bit_length,norb,
		      adet_to_bdet,adet_to_det,
		      bdet_to_adet,bdet_to_det);
      idxmap.AdetToBdetSM.resize(adet_to_bdet.size());
      idxmap.AdetToDetSM.resize(adet_to_det.size());
      idxmap.BdetToAdetSM.resize(bdet_to_adet.size());
      idxmap.BdetToDetSM.resize(bdet_to_det.size());
      idxmap.AdetToDetLen.resize(adet_to_det.size());
      idxmap.BdetToDetLen.resize(bdet_to_det.size());
      size_t aidx_size = 0;
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToDetLen[k] = adet_to_det[k].size();
	aidx_size += adet_to_det[k].size();
      }
      size_t bidx_size = 0;
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToDetLen[k] = bdet_to_det[k].size();
	bidx_size += bdet_to_det[k].size();
      }
      size_t total_size = 2*aidx_size + 2*bidx_size;
      idxmap.storage.resize(total_size);
      size_t * begin = idxmap.storage.begin();
      size_t counter = 0;
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToDetSM[k] = begin + counter;
	counter += adet_to_det[k].size();
      }
      for(size_t k=0; k < adet_to_det.size(); k++) {
	idxmap.AdetToBdetSM[k] = begin + counter;
	counter += adet_to_det[k].size();
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToDetSM[k] = begin + counter;
	counter += bdet_to_det[k].size();
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	idxmap.BdetToAdetSM[k] = begin + counter;
	counter += bdet_to_det[k].size();
      }
      
      for(size_t k=0; k < adet_to_det.size(); k++) {
	std::memcpy(idxmap.AdetToDetSM[k],
		    adet_to_det[k].data(),
		    idxmap.AdetToDetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < adet_to_det.size(); k++) {
	std::memcpy(idxmap.AdetToBdetSM[k],
		    adet_to_bdet[k].data(),
		    idxmap.AdetToBdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	std::memcpy(idxmap.BdetToDetSM[k],
		    bdet_to_det[k].data(),
		    idxmap.BdetToDetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < bdet_to_det.size(); k++) {
	std::memcpy(idxmap.BdetToAdetSM[k],
		    bdet_to_adet[k].data(),
		    idxmap.BdetToDetLen[k]*sizeof(size_t));
      }
    }
    
    
    void makeExcitationLookup(const std::vector<std::vector<size_t>> & hdet_bra,
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
    
    void makeExcitationLookup(const std::vector<std::vector<size_t>> & adet_bra,
			      const std::vector<std::vector<size_t>> & bdet_bra,
			      const std::vector<std::vector<size_t>> & adet_ket,
			      const std::vector<std::vector<size_t>> & bdet_ket,
			      size_t bit_length,
			      size_t norb,
			      ExcitationLookup & exidx) {
      std::vector<std::vector<size_t>> selfdet_a;
      std::vector<std::vector<size_t>> singles_a;
      std::vector<std::vector<size_t>> selfdet_b;
      std::vector<std::vector<size_t>> singles_b;
      makeExcitationLookup(adet_bra,adet_ket,bit_length,norb,selfdet_a,singles_a);
      makeExcitationLookup(bdet_bra,bdet_ket,bit_length,norb,selfdet_b,singles_b);
      exidx.SelfFromAdetLen.resize(selfdet_a.size());
      exidx.SinglesFromAdetLen.resize(singles_a.size());
      exidx.SelfFromBdetLen.resize(selfdet_b.size());
      exidx.SinglesFromBdetLen.resize(singles_b.size());
      size_t total_size = 0;
      for(size_t k=0; k < selfdet_a.size(); k++) {
	exidx.SelfFromAdetLen[k] = selfdet_a[k].size();
	total_size += selfdet_a[k].size();
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	exidx.SinglesFromAdetLen[k] = singles_a[k].size();
	total_size += singles_a[k].size();
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	exidx.SelfFromBdetLen[k] = selfdet_b[k].size();
	total_size += selfdet_b[k].size();
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	exidx.SinglesFromBdetLen[k] = singles_b[k].size();
	total_size += singles_b[k].size();
      }
      exidx.storage.resize(total_size);
      size_t * begin = exidx.storage.begin();
      size_t counter = 0;
      for(size_t k=0; k < selfdet_a.size(); k++) {
	exidx.SelfFromAdetSM[k] = begin + counter;
	counter += exidx.SelfFromAdetLen[k];
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	exidx.SinglesFromAdetSM[k] = begin + counter;
	counter += exidx.SinglesFromAdetLen[k];
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	exidx.SelfFromBdetSM[k] = begin + counter;
	counter += exidx.SelfFromBdetLen[k];
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	exidx.SinglesFromBdetSM[k] = begin + counter;
	counter += exidx.SinglesFromBdetLen[k];
      }
      
      for(size_t k=0; k < selfdet_a.size(); k++) {
	std::memcpy(exidx.SelfFromAdetSM[k],
		    selfdet_a[k].data(),
		    exidx.SelfFromAdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < singles_a.size(); k++) {
	std::memcpy(exidx.SinglesFromAdetSM[k],
		    singles_a[k].data(),
		    exidx.SinglesFromAdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < selfdet_b.size(); k++) {
	std::memcpy(exidx.SelfFromBdetSM[k],
		    selfdet_b[k].data(),
		    exidx.SelfFromBdetLen[k]*sizeof(size_t));
      }
      for(size_t k=0; k < singles_b.size(); k++) {
	std::memcpy(exidx.SinglesFromBdetSM[k],
		    singles_b[k].data(),
		    exidx.SinglesFromBdetLen[k]*sizeof(size_t));
      }
    }
    
    void DetIdxMapCopy(const DetIdxMap & idxmap,
		       DetIdxMap & new_idxmap) {
      new_idxmap.storage = idxmap.storage; // hard copy
      new_idxmap.AdetToDetLen = idxmap.AdetToDetLen;
      new_idxmap.BdetToDetLen = idxmap.BdetToDetLen;
      new_idxmap.AdetToDetSM.resize(new_idxmap.AdetToDetLen.size());
      new_idxmap.AdetToBdetSM.resize(new_idxmap.AdetToDetLen.size());
      new_idxmap.BdetToDetSM.resize(new_idxmap.BdetToDetLen.size());
      new_idxmap.BdetToAdetSM.resize(new_idxmap.BdetToDetLen.size());
      size_t * begin = new_idxmap.storage.begin();
      size_t counter = 0;
      for(size_t i=0; i < new_idxmap.AdetToDetLen.size(); i++) {
	new_idxmap.AdetToDetSM[i] = begin + counter;
	counter += new_idxmap.AdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.AdetToDetLen.size(); i++) {
	new_idxmap.AdetToBdetSM[i] = begin + counter;
	counter += new_idxmap.AdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.BdetToDetLen.size(); i++) {
	new_idxmap.BdetToDetSM[i] = begin + counter;
	counter += new_idxmap.BdetToDetLen[i];
      }
      for(size_t i=0; i < new_idxmap.BdetToDetLen.size(); i++) {
	new_idxmap.BdetToAdetSM[i] = begin + counter;
	counter += new_idxmap.BdetToDetLen[i];
      }
    }
    
    void MpiSlide(const DetIndexMap & send_map,
		  DetIndexMap & recv_map,
		  int slide,
		  MPI_Comm comm) {
      MpiSlide(send_map.AdetToDetLen,recv_map.AdetToDetLen,slide,comm);
      MpiSlide(send_map.BdetToDetLen,recv_map.BdetToDetLen,slide,comm);
      MpiSlide(send.storage,recv.storage,slide,comm);
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
      size_t * begin = recv.storage.data();
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
    
    void MakeHelpers(const std::vector<std::vector<size_t>> & det,
		     size_t bit_length,
		     size_t norb,
		     DetIndexMap & idxmap,
		     std::vector<ExcitationLookup> & exidx,
		     MPI_Comm h_comm,
		     MPI_Comm b_comm,
		     MPI_Comm t_comm) {
      
      int mpi_size_h; MPI_Comm_size(h_comm,&mpi_size_h);
      int mpi_rank_h; MPI_Comm_rank(h_comm,&mpi_rank_h);
      int mpi_size_b; MPI_Comm_size(b_comm,&mpi_size_b);
      int mpi_rank_b; MPI_Comm_rank(b_comm,&mpi_rank_b);
      int mpi_size_t; MPI_Comm_size(t_comm,&mpi_size_t);
      int mpi_rank_t; MPI_Comm_rank(t_comm,&mpi_rank_t);
      
      size_t task_begin = 0;
      size_t task_end   = static_cast<size_t>(mpi_size_t);
      get_mpi_range(mpi_size_t,mpi_rank_t,task_begin,task_end);
      size_t task_size = task_end-task_begin;
      exidx.resize(task_size);
      
      std::vector<std::vector<size_t>> adet;
      std::vector<std::vector<size_t>> bdet;
      std::vector<size_t> adet_count;
      std::vector<size_t> bdet_count;
      getHalfDets(det,bit_length,norb,adet,bdet,adet_count,bdet_count);
      makeDetIndexMap(det,adet,bdet,adet_count,bdet_count,
		      bit_length,norb,idxmap);
      
      std::vector<std::vector<size_t>> ket_det;
      std::vector<std::vector<size_t>> ket_adet;
      std::vector<std::vector<size_t>> ket_bdet;
      if ( task_begin != static_cast<size_t>(0) ) {
	int slide = - static_cast<int>(task_begin);
	MpiSlide(det,ket_det,slide,t_comm);
	MpiSlide(adet,ket_adet,slide,t_comm);
	MpiSlide(bdet,ket_bdet,slide,t_comm);
      } else {
	ket_det = det;
	ket_adet = ket_adet;
	ket_bdet = ket_bdet;
      }
      
      for(size_t task=task_begin; task < task_end; task++) {
	makeExcitationLookup(adet,bdet,adet_ket,bdet_ket,
			     bit_length,norb,
			     exidx[task]);
	if( task != task_end-1 ) {
	  std::vector<std::vector<size_t>> send_det;
	  std::vector<std::vector<size_t>> send_adet;
	  std::vector<std::vector<size_t>> send_bdet;
	  DetIndexMap send_idxmap;
	  std::swap(ket_det,send_det);
	  std::swap(ket_adet,send_adet);
	  std::swap(ket_bdet,send_bdet);
	  int slide = -1;
	  MpiSlide(send_det,ket_det,slide,t_comm);
	  MpiSlide(send_adet,ket_adet,slide,t_comm);
	  MpiSlide(send_bdet,ket_bdet,slide,t_comm);
	}
      }
    }

    
  } // end namespace gdb
  
} // end namespace sbd

#endif
