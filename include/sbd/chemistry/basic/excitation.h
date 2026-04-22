/**
@file sbd/chemistry/basic/excitation.h
@brief function to find excitation from a determinant
 */
#ifndef SBD_CHEMISTRY_BASIC_EXCITATION_H
#define SBD_CHEMISTRY_BASIC_EXCITATION_H
namespace sbd {

  void single_from_hdet(const std::vector<size_t> & hdet_base,
			size_t bit_length,
			size_t norb,
			size_t num_closed,
			const std::vector<int> & open_base,
			const std::vector<int> & closed_base,
			std::vector<std::vector<size_t>> & hdet_ex) {
    // supporsed that open and closed are obtained priory by using getOpenClosed function for hdet_base.
    size_t num_ex = num_closed * (norb - num_closed);
    hdet_ex.resize(num_ex);
    size_t ex_count = 0;
    for(size_t j=0; j < num_closed; j++) {
      for(size_t k=0; k < norb-num_closed; k++) {
	hdet_ex[ex_count] = hdet_base;
	setocc(hdet_ex[ex_count],bit_length,closed_base[j],false);
	setocc(hdet_ex[ex_count],bit_length,open_base[k],true);
	ex_count++;
      }
    }
  }
  
  void single_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			std::vector<std::vector<size_t>> & edet) {
    std::vector<int> open_base(norb);
    std::vector<int> closed_base(norb);
    int nc = getOpenClosed(hdet,bit_length,norb,open_base,closed_base);
    size_t numc = static_cast<size_t>(nc);
    single_from_hdet(hdet,bit_length,norb,numc,open_base,closed_base,edet);
  }

  /**
     Generate all same-spin double excitations from a half-determinant.
     For each pair of occupied orbitals (i,j) with j<i and each pair of
     virtual orbitals (a,b) with b<a, produce the excitation i->a, j->b.
     Count: C(num_closed,2) * C(num_vir,2).
   */
  void double_from_hdet(const std::vector<size_t> & hdet_base,
			size_t bit_length,
			size_t norb,
			size_t num_closed,
			const std::vector<int> & open_base,
			const std::vector<int> & closed_base,
			std::vector<std::vector<size_t>> & hdet_ex) {
    size_t num_vir = norb - num_closed;
    if( num_closed < 2 || num_vir < 2 ) {
      hdet_ex.resize(0);
      return;
    }
    size_t num_ex = (num_closed * (num_closed - 1) / 2)
                  * (num_vir * (num_vir - 1) / 2);
    hdet_ex.resize(num_ex);
    size_t ex_count = 0;
    for(size_t i=1; i < num_closed; i++) {
      for(size_t j=0; j < i; j++) {
	for(size_t a=1; a < num_vir; a++) {
	  for(size_t b=0; b < a; b++) {
	    hdet_ex[ex_count] = hdet_base;
	    setocc(hdet_ex[ex_count],bit_length,closed_base[i],false);
	    setocc(hdet_ex[ex_count],bit_length,closed_base[j],false);
	    setocc(hdet_ex[ex_count],bit_length,open_base[a],true);
	    setocc(hdet_ex[ex_count],bit_length,open_base[b],true);
	    ex_count++;
	  }
	}
      }
    }
  }

  void double_from_hdet(const std::vector<size_t> & hdet,
			size_t bit_length,
			size_t norb,
			std::vector<std::vector<size_t>> & edet) {
    std::vector<int> open_base(norb);
    std::vector<int> closed_base(norb);
    int nc = getOpenClosed(hdet,bit_length,norb,open_base,closed_base);
    size_t numc = static_cast<size_t>(nc);
    double_from_hdet(hdet,bit_length,norb,numc,open_base,closed_base,edet);
  }

  /**
     Generate ERI-screened single excitations from a half-determinant.
     For single excitation j->k, the effective coupling is:
       h_eff = |h_{jk}| + sum_m |v_{jk,mm} - v_{jm,mk}|
     where m runs over all other occupied orbitals.
     Only excitations with h_eff > eri_threshold are generated.
     Orbital indices in closed_base/open_base are spatial (0-based).
   */
  template <typename ElemT>
  void single_from_hdet_screened(const std::vector<size_t> & hdet_base,
				 size_t bit_length,
				 size_t norb,
				 size_t num_closed,
				 const std::vector<int> & open_base,
				 const std::vector<int> & closed_base,
				 const sbd::oneInt<ElemT> & I1,
				 const sbd::twoInt<ElemT> & I2,
				 ElemT eri_threshold,
				 std::vector<std::vector<size_t>> & hdet_ex) {
    hdet_ex.clear();
    for(size_t j=0; j < num_closed; j++) {
      int iorb = closed_base[j];
      for(size_t k=0; k < norb-num_closed; k++) {
	int aorb = open_base[k];
	// Compute effective 1e coupling: Fock-like matrix element
	// h_{ia} + sum_m [v_{ia,mm} - v_{im,ma}]
	ElemT h_eff = std::abs(I1.Value(2*iorb, 2*aorb));
	for(size_t m=0; m < num_closed; m++) {
	  int morb = closed_base[m];
	  // Same-spin Coulomb - Exchange
	  h_eff += std::abs(I2.Value(2*iorb, 2*aorb, 2*morb, 2*morb)
			    - I2.Value(2*iorb, 2*morb, 2*morb, 2*aorb));
	  // Opposite-spin Coulomb (beta occupied contribute too)
	  h_eff += std::abs(I2.Value(2*iorb, 2*aorb, 2*morb+1, 2*morb+1));
	}
	if( h_eff > eri_threshold ) {
	  std::vector<size_t> det = hdet_base;
	  setocc(det,bit_length,iorb,false);
	  setocc(det,bit_length,aorb,true);
	  hdet_ex.push_back(det);
	}
      }
    }
  }

  /**
     Generate ERI-screened same-spin double excitations from a half-determinant.
     For double excitation (i,j)->(a,b), the relevant integral is
       |v_{ia,jb} - v_{ib,ja}| (antisymmetrized two-electron integral).
     Only excitations where this exceeds eri_threshold are generated.
   */
  template <typename ElemT>
  void double_from_hdet_screened(const std::vector<size_t> & hdet_base,
				 size_t bit_length,
				 size_t norb,
				 size_t num_closed,
				 const std::vector<int> & open_base,
				 const std::vector<int> & closed_base,
				 const sbd::twoInt<ElemT> & I2,
				 ElemT eri_threshold,
				 std::vector<std::vector<size_t>> & hdet_ex) {
    hdet_ex.clear();
    size_t num_vir = norb - num_closed;
    if( num_closed < 2 || num_vir < 2 ) return;
    for(size_t i=1; i < num_closed; i++) {
      int iorb = closed_base[i];
      for(size_t j=0; j < i; j++) {
	int jorb = closed_base[j];
	for(size_t a=1; a < num_vir; a++) {
	  int aorb = open_base[a];
	  for(size_t b=0; b < a; b++) {
	    int borb = open_base[b];
	    // Antisymmetrized two-electron integral: <ij||ab> = (ia|jb) - (ib|ja)
	    ElemT v_direct  = I2.Value(2*iorb, 2*aorb, 2*jorb, 2*borb);
	    ElemT v_exchange = I2.Value(2*iorb, 2*borb, 2*jorb, 2*aorb);
	    if( std::abs(v_direct - v_exchange) > eri_threshold ) {
	      std::vector<size_t> det = hdet_base;
	      setocc(det,bit_length,iorb,false);
	      setocc(det,bit_length,jorb,false);
	      setocc(det,bit_length,aorb,true);
	      setocc(det,bit_length,borb,true);
	      hdet_ex.push_back(det);
	    }
	  }
	}
      }
    }
  }

}

#endif
