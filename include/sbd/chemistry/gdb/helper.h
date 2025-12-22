/**
@file sbd/chemistry/gdb/helpers.h
@brief Helper array to construct Hamiltonian for general determinant basis
 */
#ifndef SBD_CHEMISTRY_GDB_HELPER_H
#define SBD_CHEMISTRY_GDB_HELPER_H

namespace sbd {

  struct DetBaseHelpers {
    std::vector<std::vector<size_t>> AdetToBdet;
    std::vector<std::vector<size_t>> AdetToDet;
    std::vector<std::vector<size_t>> BdetToAdet;
    std::vector<std::vector<size_t>> BdetToDet;
    std::vector<std::vector<size_t>> SinglesFromAdet;
    std::vector<std::vector<size_t>> SinglesFromBdet;

    size_t * AdetToBdetLen;
    size_t * SinglesFromAdetLen;
    size_t * BdetToAdetLen;
    size_t * SinglesFromBdetLen;

    std::vector<size_t*> AdetToBdetSM;
    std::vector<size_t*> AdetToDetSM;
    std::vector<size_t*> SinglesFromAdetSM;
    std::vector<size_t*> BdetToAdetSM;
    std::vector<size_t*> BdetToDetSM;
    std::vector<size_t*> SinglesFromBdetSM;
  };

  void getAdetBdet(const std::vector<std::vector<size_t>> & det,
		   size_t bit_length,
		   size_t norb,
		   std::vector<std::vector<size_t>> & adet,
		   std::vector<std::vector<size_t>> & bdet) {
    adet.resize(det.size());
    bdet.resize(det.size());
    for(size_t i=0; i < det.size(); i++) {
      getAdet(det[i],bit_length,norb,adet[i]);
      getBdet(det[i],bit_length,norb,bdet[i]);
    }
    sort_bitarray(adet);
    sort_bitarray(bdet);
  }

  void getHdetSingles(const std::vector<std::vector<size_t>> & hdet_bra,
		      const std::vector<std::vector<size_t>> & hdet_ket,
		      size_t bit_length,
		      size_t norb,
		      std::vector<std::vector<size_t>> & samedet,
		      std::vector<std::vector<size_t>> & singles) {
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

  void getHdetToHdet(const std::vector<std::vector<size_t> & det,
		     const std::vector<std::vector<size_t>> & adet,
		     const std::vector<std::vector<size_t>> & bdet,
		     const std::vector<size_t> & adet_count,
		     const std::vector<size_t> & bdet_count,
		     size_t bit_length,
		     size_t norb,
		     std::vector<std::vector<size_t>> & adet_to_bdet,
		     std::vector<std::vector<size_t>> & adet_to_det,
		     std::vector<std::vector<size_t>> & bdet_to_adet,
		     std::vector<std::vector<size_t>> & bdet_to_det,
		     std::map<std::vector<size_t>,size_t> & adet_n,
		     std::map<std::vector<size_t>,size_t> & bdet_n) {
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
  
}

#endif
