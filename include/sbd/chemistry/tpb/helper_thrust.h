/**
@file sbd/chemistry/tpb/helpers.h
@brief Helper array to construct Hamiltonian for parallel taskers for distributed basis
 */
#ifndef SBD_CHEMISTRY_TPB_HELPER_THRUST_H
#define SBD_CHEMISTRY_TPB_HELPER_THRUST_H

namespace sbd {

#include "sbd/chemistry/tpb/helper.h"

//#include <thrust/device_vector.h>
#include <thrust/host_vector.h>





template <typename ElemT>
class TaskHelpersThrust {
public:
    size_t braAlphaStart;
    size_t braAlphaEnd;
    size_t ketAlphaStart;
    size_t ketAlphaEnd;
    size_t braBetaStart;
    size_t braBetaEnd;
    size_t ketBetaStart;
    size_t ketBetaEnd;
    size_t taskType;
    size_t adetShift;
    size_t bdetShift;

    size_t* base_memory;
    size_t* SinglesFromAlphaBraIndex;
    size_t* SinglesFromAlphaKetIndex;
    size_t* DoublesFromAlphaBraIndex;
    size_t* DoublesFromAlphaKetIndex;
    size_t* SinglesFromBetaBraIndex;
    size_t* SinglesFromBetaKetIndex;
    size_t* DoublesFromBetaBraIndex;
    size_t* DoublesFromBetaKetIndex;

    size_t* SinglesFromAlphaLen;
    size_t* SinglesFromBetaLen;
    size_t* DoublesFromAlphaLen;
    size_t* DoublesFromBetaLen;

    size_t** SinglesFromAlphaSM;
    size_t** SinglesFromBetaSM;
    size_t** DoublesFromAlphaSM;
    size_t** DoublesFromBetaSM;

    // pre-calculated E(i,j)
    ElemT* Eij;
    size_t* Eij_len;
    size_t* Eij_offset;

    size_t size_single_alpha = 0;
    size_t size_double_alpha = 0;
    size_t size_single_beta = 0;
    size_t size_double_beta = 0;

    TaskHelpersThrust() {}

    TaskHelpersThrust(const TaskHelpersThrust<ElemT>& other)
    {
        braAlphaStart = other.braAlphaStart;
        braAlphaEnd = other.braAlphaEnd;
        ketAlphaStart = other.ketAlphaStart;
        ketAlphaEnd = other.ketAlphaEnd;
        braBetaStart = other.braBetaStart;
        braBetaEnd = other.braBetaEnd;
        ketBetaStart = other.ketBetaStart;
        ketBetaEnd = other.ketBetaEnd;
        taskType = other.taskType;
        adetShift = other.adetShift;
        bdetShift = other.bdetShift;


        /*SinglesFromAlphaLen = other.SinglesFromAlphaLen;
        SinglesFromBetaLen = other.SinglesFromBetaLen;
        DoublesFromAlphaLen = other.DoublesFromAlphaLen;
        DoublesFromBetaLen = other.DoublesFromBetaLen;
        SinglesFromAlphaSM = other.SinglesFromAlphaSM;
        SinglesFromBetaSM = other.SinglesFromBetaSM;
        DoublesFromAlphaSM = other.DoublesFromAlphaSM;
        DoublesFromBetaSM = other.DoublesFromBetaSM;*/

        base_memory = other.base_memory;
        SinglesFromAlphaBraIndex = other.SinglesFromAlphaBraIndex;
        SinglesFromAlphaKetIndex = other.SinglesFromAlphaKetIndex;
        DoublesFromAlphaBraIndex = other.DoublesFromAlphaBraIndex;
        DoublesFromAlphaKetIndex = other.DoublesFromAlphaKetIndex;
        SinglesFromBetaBraIndex = other.SinglesFromBetaBraIndex;
        SinglesFromBetaKetIndex = other.SinglesFromBetaKetIndex;
        DoublesFromBetaBraIndex = other.DoublesFromBetaBraIndex;
        DoublesFromBetaKetIndex = other.DoublesFromBetaKetIndex;

        Eij = other.Eij;
        Eij_len = other.Eij_len;
        Eij_offset = other.Eij_offset;

        size_single_alpha = other.size_single_alpha;
        size_double_alpha = other.size_double_alpha;
        size_single_beta = other.size_single_beta;
        size_double_beta = other.size_double_beta;
    }

    TaskHelpersThrust(thrust::device_vector<size_t>& storage, thrust::device_vector<ElemT>& veij, const TaskHelpers& helper, bool precalculate = false)
    {
        braAlphaStart = helper.braAlphaStart;
        braAlphaEnd = helper.braAlphaEnd;
        ketAlphaStart = helper.ketAlphaStart;
        ketAlphaEnd = helper.ketAlphaEnd;
        braBetaStart = helper.braBetaStart;
        braBetaEnd = helper.braBetaEnd;
        ketBetaStart = helper.ketBetaStart;
        ketBetaEnd = helper.ketBetaEnd;
        taskType = helper.taskType;
        adetShift = helper.adetShift;
        bdetShift = helper.bdetShift;

        size_t braAlphaSize = helper.braAlphaEnd - helper.braAlphaStart;
        size_t braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        thrust::host_vector<size_t> offset_ij;

        size_t size = 0;

        size_single_alpha = 0;
        size_double_alpha = 0;
        size_single_beta = 0;
        size_double_beta = 0;
        for(size_t i=0; i < braAlphaSize; i++) {
            size_single_alpha += helper.SinglesFromAlphaLen[i];
            size_double_alpha += helper.DoublesFromAlphaLen[i];
        }
        size += size_single_alpha + size_double_alpha;
        for(size_t i=0; i < braBetaSize; i++) {
            size_single_beta += helper.SinglesFromBetaLen[i];
            size_double_beta += helper.DoublesFromBetaLen[i];
        }
        size += size_single_beta + size_double_beta;

        size_t offset = 0;
        storage.resize(size*2);

        base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        std::vector<size_t> indices;
        size_t count = 0;
        size_t pos = 0;

        SinglesFromAlphaBraIndex = base_memory + count;
        SinglesFromAlphaKetIndex = base_memory + count + size_single_alpha;
        indices.resize(size_single_alpha * 2);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        for(size_t i=0; i < braAlphaSize; i++) {
            for (size_t j = 0; j < helper.SinglesFromAlphaLen[i]; j++) {
                indices[pos] = i + helper.braAlphaStart;
                indices[size_single_alpha + pos] = helper.SinglesFromAlphaSM[i][j];
                pos++;
            }
        }
        thrust::copy_n(indices.begin(), size_single_alpha * 2, storage.begin() + count);
        count += size_single_alpha * 2;

        pos = 0;
        DoublesFromAlphaBraIndex = base_memory + count;
        DoublesFromAlphaKetIndex = base_memory + count + size_double_alpha;
        indices.resize(size_double_alpha * 2);
        for(size_t i=0; i < braAlphaSize; i++) {
            for (size_t j = 0; j < helper.DoublesFromAlphaLen[i]; j++) {
                indices[pos] = i + helper.braAlphaStart;
                indices[size_double_alpha + pos] = helper.DoublesFromAlphaSM[i][j];
                pos++;
            }
        }
        thrust::copy_n(indices.begin(), size_double_alpha * 2, storage.begin() + count);
        count += size_double_alpha * 2;

        pos = 0;
        SinglesFromBetaBraIndex = base_memory + count;
        SinglesFromBetaKetIndex = base_memory + count + size_single_beta;
        indices.resize(size_single_beta * 2);
        for(size_t i=0; i < braBetaSize; i++) {
            for (size_t j = 0; j < helper.SinglesFromBetaLen[i]; j++) {
                indices[pos] = i + helper.braBetaStart;
                indices[size_single_beta + pos] = helper.SinglesFromBetaSM[i][j];
                pos++;
            }
        }
        thrust::copy_n(indices.begin(), size_single_beta * 2, storage.begin() + count);
        count += size_single_beta * 2;

        pos = 0;
        DoublesFromBetaBraIndex = base_memory + count;
        DoublesFromBetaKetIndex = base_memory + count + size_double_beta;
        indices.resize(size_double_beta * 2);
        for(size_t i=0; i < braBetaSize; i++) {
            for (size_t j = 0; j < helper.DoublesFromBetaLen[i]; j++) {
                indices[pos] = i + helper.braBetaStart;
                indices[size_double_beta + pos] = helper.DoublesFromBetaSM[i][j];
                pos++;
            }
        }
        thrust::copy_n(indices.begin(), size_double_beta * 2, storage.begin() + count);
        count += size_double_beta * 2;

        /*SinglesFromAlphaLen = base_memory + count;
        thrust::copy_n(helper.SinglesFromAlphaLen, nAlpha, storage.begin() + count);
        count += nAlpha;

        DoublesFromAlphaLen = base_memory + count;
        thrust::copy_n(helper.DoublesFromAlphaLen, nAlpha, storage.begin() + count);
        count += nAlpha;

        SinglesFromBetaLen = base_memory + count;
        thrust::copy_n(helper.SinglesFromBetaLen, nBeta, storage.begin() + count);
        count += nBeta;

        DoublesFromBetaLen = base_memory + count;
        thrust::copy_n(helper.DoublesFromBetaLen, nBeta, storage.begin() + count);
        count += nBeta;

        // copy index
        thrust::host_vector<size_t> offsets(std::max(nAlpha, nBeta));
        size_t* begin = helper.SinglesFromAlphaSM[0];

        for(size_t i=0; i < nAlpha; i++) {
            offsets[i] = (size_t)(helper.SinglesFromAlphaSM[i] - begin + base_memory + data_start);
        }
        SinglesFromAlphaSM = (size_t**)base_memory + count;
        thrust::copy_n(offsets.begin(), nAlpha, storage.begin() + count);
        count += nAlpha;

        for(size_t i=0; i < nAlpha; i++) {
            offsets[i] = (size_t)(helper.DoublesFromAlphaSM[i] - begin + base_memory + data_start);
        }
        DoublesFromAlphaSM = (size_t**)base_memory + count;
        thrust::copy_n(offsets.begin(), nAlpha, storage.begin() + count);
        count += nAlpha;

        for(size_t i=0; i < nBeta; i++) {
            offsets[i] = (size_t)(helper.SinglesFromBetaSM[i] - begin + base_memory + data_start);
        }
        SinglesFromBetaSM = (size_t**)base_memory + count;
        thrust::copy_n(offsets.begin(), nBeta, storage.begin() + count);
        count += nBeta;

        for(size_t i=0; i < nBeta; i++) {
            offsets[i] = (size_t)(helper.DoublesFromBetaSM[i] - begin + base_memory + data_start);
        }
        DoublesFromBetaSM = (size_t**)base_memory + count;
        thrust::copy_n(offsets.begin(), nBeta, storage.begin() + count);
        count += nBeta;

        // copy SM data
        thrust::copy_n(begin, size - data_start, storage.begin() + count);
        count += (size - data_start);*/

        /*if (precalculate) {
            Eij = (ElemT*)thrust::raw_pointer_cast(veij.data());

            Eij_offset = base_memory + count;
            thrust::copy_n(offset_ij.begin(), braAlphaSize * braBetaSize, storage.begin() + count);
        }*/
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
