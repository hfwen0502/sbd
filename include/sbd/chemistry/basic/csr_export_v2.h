/**
 * @file sbd/chemistry/basic/csr_export_v2.h
 * @brief Export Hamiltonian matrix in CSR format using makeQCham approach
 */
#ifndef SBD_CHEMISTRY_BASIC_CSR_EXPORT_V2_H
#define SBD_CHEMISTRY_BASIC_CSR_EXPORT_V2_H

#include <vector>
#include <algorithm>
#include <map>
#include "sbd/chemistry/basic/determinants.h"
#include "sbd/chemistry/basic/integrals.h"

namespace sbd {

/**
 * Triplet format for sparse matrix: (row, col, value)
 */
template <typename ElemT>
struct MatrixTriplet {
    size_t row;
    size_t col;
    ElemT value;
    
    bool operator<(const MatrixTriplet& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/**
 * Build Hamiltonian using the proven makeQCham approach
 * This mimics what method=1 does but simplified for CSR export
 */
template <typename ElemT>
bool buildHamiltonianTripletsV2(
    const std::vector<std::vector<size_t>>& adet,
    const std::vector<std::vector<size_t>>& bdet,
    size_t bit_length,
    size_t norb,
    const ElemT& I0,
    const oneInt<ElemT>& I1,
    const twoInt<ElemT>& I2,
    size_t max_nnz,
    std::vector<MatrixTriplet<ElemT>>& triplets)
{
    size_t n_adet = adet.size();
    size_t n_bdet = bdet.size();
    size_t n = n_adet * n_bdet;
    
    std::cerr << "[CSR V2] Using makeQCham-style approach" << std::endl;
    std::cerr << "  n_adet=" << n_adet << ", n_bdet=" << n_bdet << std::endl;
    
    triplets.clear();
    triplets.reserve(std::min(max_nnz, n * 100));
    
    // Step 1: Compute diagonal elements (like makeQCham does)
    std::cerr << "[CSR V2] Computing diagonal elements..." << std::endl;
    std::vector<ElemT> hii(n);
    
    for (size_t ia = 0; ia < n_adet; ++ia) {
        for (size_t ib = 0; ib < n_bdet; ++ib) {
            size_t k = ia * n_bdet + ib;
            std::vector<size_t> det = DetFromAlphaBeta(adet[ia], bdet[ib], bit_length, norb);
            hii[k] = ZeroExcite(det, bit_length, norb, I0, I1, I2);
            
            // Add to triplets
            if (std::abs(hii[k]) > 1e-12) {
                MatrixTriplet<ElemT> t;
                t.row = k;
                t.col = k;
                t.value = hii[k];
                triplets.push_back(t);
            }
        }
    }
    
    std::cerr << "[CSR V2] Diagonal elements computed: " << triplets.size() << std::endl;
    
    // Step 2: Compute off-diagonal elements
    // Use a map to store (i,j) -> value to avoid duplicates
    std::cerr << "[CSR V2] Computing off-diagonal elements..." << std::endl;
    std::map<std::pair<size_t, size_t>, ElemT> offdiag;
    
    size_t count = 0;
    bool truncated = false;
    
    // For each determinant pair
    for (size_t ia = 0; ia < n_adet && !truncated; ++ia) {
        for (size_t ib = 0; ib < n_bdet && !truncated; ++ib) {
            size_t i = ia * n_bdet + ib;
            std::vector<size_t> det_i = DetFromAlphaBeta(adet[ia], bdet[ib], bit_length, norb);
            
            // Check connections to other determinants
            for (size_t ja = 0; ja < n_adet && !truncated; ++ja) {
                for (size_t jb = 0; jb < n_bdet && !truncated; ++jb) {
                    size_t j = ja * n_bdet + jb;
                    
                    if (i == j) continue;  // Skip diagonal
                    if (j > i) continue;   // Only lower triangle
                    
                    std::vector<size_t> det_j = DetFromAlphaBeta(adet[ja], bdet[jb], bit_length, norb);
                    
                    size_t orbDiff;
                    ElemT h_ij = Hij(det_i, det_j, bit_length, norb, I0, I1, I2, orbDiff);
                    
                    if (std::abs(h_ij) > 1e-12) {
                        // Store both (i,j) and (j,i)
                        offdiag[{i, j}] = h_ij;
                        offdiag[{j, i}] = h_ij;
                        count += 2;
                        
                        if (count + triplets.size() >= max_nnz) {
                            truncated = true;
                            break;
                        }
                    }
                }
            }
            
            if (ia % 10 == 0 && ib == 0) {
                std::cerr << "  Progress: " << ia << "/" << n_adet << " (" << count << " off-diag)" << std::endl;
            }
        }
    }
    
    // Add off-diagonal elements to triplets
    for (const auto& kv : offdiag) {
        MatrixTriplet<ElemT> t;
        t.row = kv.first.first;
        t.col = kv.first.second;
        t.value = kv.second;
        triplets.push_back(t);
    }
    
    std::cerr << "[CSR V2] Total triplets: " << triplets.size() << std::endl;
    std::cerr << "[CSR V2] Truncated: " << (truncated ? "yes" : "no") << std::endl;
    
    return !truncated;
}

/**
 * Convert triplet format to CSR format
 */
template <typename ElemT>
void tripletsToCSR(
    std::vector<MatrixTriplet<ElemT>>& triplets,
    size_t n,
    std::vector<ElemT>& data,
    std::vector<int>& indices,
    std::vector<int>& indptr)
{
    // Sort triplets by (row, col)
    std::sort(triplets.begin(), triplets.end());
    
    // Allocate CSR arrays
    size_t nnz = triplets.size();
    data.resize(nnz);
    indices.resize(nnz);
    indptr.resize(n + 1);
    
    // Build CSR format
    size_t current_row = 0;
    indptr[0] = 0;
    
    for (size_t k = 0; k < nnz; ++k) {
        // Fill indptr for any empty rows
        while (current_row < triplets[k].row) {
            current_row++;
            indptr[current_row] = k;
        }
        
        data[k] = triplets[k].value;
        indices[k] = static_cast<int>(triplets[k].col);
    }
    
    // Fill remaining indptr entries
    while (current_row < n) {
        current_row++;
        indptr[current_row] = nnz;
    }
}

} // namespace sbd

#endif // SBD_CHEMISTRY_BASIC_CSR_EXPORT_V2_H

// Made with Bob
