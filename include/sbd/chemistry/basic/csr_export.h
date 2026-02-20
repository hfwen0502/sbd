/**
 * @file sbd/chemistry/basic/csr_export.h
 * @brief Export Hamiltonian matrix in CSR (Compressed Sparse Row) format
 */
#ifndef SBD_CHEMISTRY_BASIC_CSR_EXPORT_H
#define SBD_CHEMISTRY_BASIC_CSR_EXPORT_H

#include <vector>
#include <algorithm>
#include <tuple>
#include <complex>
#include <cmath>
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
    
    // For sorting: first by row, then by column
    bool operator<(const MatrixTriplet& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

/**
 * Build Hamiltonian matrix in triplet format for TPB (two-particle basis)
 *
 * @param adet Alpha determinants (each is vector<size_t>)
 * @param bdet Beta determinants (each is vector<size_t>)
 * @param bit_length Bit length for determinant representation
 * @param norb Number of orbitals
 * @param I0 Nuclear repulsion energy
 * @param I1 One-electron integrals
 * @param I2 Two-electron integrals
 * @param max_nnz Maximum number of non-zeros to collect
 * @param triplets Output vector of (row, col, value) triplets
 * @return true if completed, false if truncated at max_nnz
 */
template <typename ElemT>
bool buildHamiltonianTriplets(
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
    size_t n = n_adet * n_bdet;  // Total Hilbert space dimension
    
    // Debug output
    std::cerr << "[CSR Export Debug] Starting buildHamiltonianTriplets" << std::endl;
    std::cerr << "  n_adet=" << n_adet << ", n_bdet=" << n_bdet << ", n=" << n << std::endl;
    std::cerr << "  bit_length=" << bit_length << ", norb=" << norb << std::endl;
    std::cerr << "  max_nnz=" << max_nnz << std::endl;
    std::cerr << "  I0=" << I0 << std::endl;
    std::cerr << "  I1.norbs=" << I1.norbs << ", I1.store.size()=" << I1.store.size() << std::endl;
    std::cerr << "  I2.norbs=" << I2.norbs << ", I2.store.size()=" << I2.store.size() << std::endl;
    std::cerr << "  I2.DirectMat.size()=" << I2.DirectMat.size() << std::endl;
    std::cerr << "  I2.ExchangeMat.size()=" << I2.ExchangeMat.size() << std::endl;
    
    triplets.clear();
    triplets.reserve(std::min(max_nnz, n * 100));  // Estimate: ~100 non-zeros per row
    
    bool truncated = false;
    size_t orbDiff;
    
    // Validate determinant sizes
    size_t expected_det_size = (norb + bit_length - 1) / bit_length;
    std::cerr << "  expected_det_size (alpha/beta)=" << expected_det_size << std::endl;
    for (size_t i = 0; i < n_adet; ++i) {
        if (adet[i].size() != expected_det_size) {
            throw std::runtime_error(
                "Alpha determinant size mismatch: expected " +
                std::to_string(expected_det_size) + " but got " +
                std::to_string(adet[i].size())
            );
        }
    }
    for (size_t i = 0; i < n_bdet; ++i) {
        if (bdet[i].size() != expected_det_size) {
            throw std::runtime_error(
                "Beta determinant size mismatch: expected " +
                std::to_string(expected_det_size) + " but got " +
                std::to_string(bdet[i].size())
            );
        }
    }
    
    // Loop over all determinant pairs
    for (size_t ia = 0; ia < n_adet && !truncated; ++ia) {
        for (size_t ib = 0; ib < n_bdet && !truncated; ++ib) {
            size_t row = ia * n_bdet + ib;  // Row index in full matrix
            
            // Create full determinant from alpha and beta parts
            // In TPB format, alpha and beta spins are interleaved
            std::vector<size_t> det_i = DetFromAlphaBeta(adet[ia], bdet[ib], bit_length, norb);
            
            // Debug: Check determinant size
            if (det_i.empty()) {
                throw std::runtime_error(
                    "DetFromAlphaBeta returned empty determinant for ia=" +
                    std::to_string(ia) + ", ib=" + std::to_string(ib)
                );
            }
            
            // Debug: Print first iteration info
            if (ia == 0 && ib == 0) {
                size_t expected_det_size_full = (2*norb + bit_length - 1) / bit_length;
                std::cerr << "[CSR Debug] First determinant (ia=0, ib=0):" << std::endl;
                std::cerr << "  det_i.size()=" << det_i.size() << std::endl;
                std::cerr << "  expected_det_size_full=" << expected_det_size_full << std::endl;
                std::cerr << "  adet[0].size()=" << adet[0].size() << std::endl;
                std::cerr << "  bdet[0].size()=" << bdet[0].size() << std::endl;
                
                if (det_i.size() != expected_det_size_full) {
                    throw std::runtime_error(
                        "Determinant size mismatch after DetFromAlphaBeta: " +
                        std::to_string(det_i.size()) + " vs expected " +
                        std::to_string(expected_det_size_full) +
                        " (norb=" + std::to_string(norb) +
                        ", bit_length=" + std::to_string(bit_length) + ")"
                    );
                }
                
                std::cerr << "  About to call ZeroExcite..." << std::endl;
            }
            
            // Diagonal element
            ElemT h_diag = ZeroExcite(det_i, bit_length, norb, I0, I1, I2);
            
            if (ia == 0 && ib == 0) {
                std::cerr << "  ZeroExcite returned: " << h_diag << std::endl;
            }
            if (std::abs(h_diag) > 1e-12) {
                MatrixTriplet<ElemT> triplet_diag;
                triplet_diag.row = row;
                triplet_diag.col = row;
                triplet_diag.value = h_diag;
                triplets.push_back(triplet_diag);
                
                if (triplets.size() >= max_nnz) {
                    truncated = true;
                    break;
                }
            }
            
            // Off-diagonal elements: loop over all other determinants
            for (size_t ja = 0; ja < n_adet && !truncated; ++ja) {
                for (size_t jb = 0; jb < n_bdet && !truncated; ++jb) {
                    size_t col = ja * n_bdet + jb;
                    
                    // Skip if same determinant (already did diagonal)
                    if (row == col) continue;
                    
                    // Only compute lower triangle (Hamiltonian is Hermitian)
                    // We'll add both (i,j) and (j,i) for full matrix
                    if (col > row) continue;
                    
                    // Create full determinant from alpha and beta parts
                    std::vector<size_t> det_j = DetFromAlphaBeta(adet[ja], bdet[jb], bit_length, norb);
                    
                    // Compute matrix element using Slater-Condon rules
                    ElemT h_ij = Hij(det_i, det_j, bit_length, norb, I0, I1, I2, orbDiff);
                    
                    if (std::abs(h_ij) > 1e-12) {
                        // Add both (i,j) and (j,i) for symmetric matrix
                        MatrixTriplet<ElemT> triplet_ij;
                        triplet_ij.row = row;
                        triplet_ij.col = col;
                        triplet_ij.value = h_ij;
                        triplets.push_back(triplet_ij);
                        
                        if (row != col) {
                            MatrixTriplet<ElemT> triplet_ji;
                            triplet_ji.row = col;
                            triplet_ji.col = row;
                            triplet_ji.value = h_ij;  // Hamiltonian is Hermitian, for real h_ij this equals conj(h_ij)
                            triplets.push_back(triplet_ji);
                        }
                        
                        if (triplets.size() >= max_nnz) {
                            truncated = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    return !truncated;
}

/**
 * Convert triplet format to CSR format
 * 
 * @param triplets Input triplets (will be sorted in-place)
 * @param n Matrix dimension
 * @param data Output: non-zero values
 * @param indices Output: column indices
 * @param indptr Output: row pointers
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

#endif // SBD_CHEMISTRY_BASIC_CSR_EXPORT_H

// Made with Bob
