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
    size_t* SinglesFromAlphaLen;
    size_t* SinglesFromBetaLen;
    size_t* DoublesFromAlphaLen;
    size_t* DoublesFromBetaLen;

    size_t* base_memory;
    size_t** SinglesFromAlphaSM;
    size_t** SinglesFromBetaSM;
    size_t** DoublesFromAlphaSM;
    size_t** DoublesFromBetaSM;

    // pre-calculated E(i,j)
    ElemT* Eij;
    size_t* Eij_len;
    size_t* Eij_offset;

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

        SinglesFromAlphaLen = other.SinglesFromAlphaLen;
        SinglesFromBetaLen = other.SinglesFromBetaLen;
        DoublesFromAlphaLen = other.DoublesFromAlphaLen;
        DoublesFromBetaLen = other.DoublesFromBetaLen;
        base_memory = other.base_memory;
        SinglesFromAlphaSM = other.SinglesFromAlphaSM;
        SinglesFromBetaSM = other.SinglesFromBetaSM;
        DoublesFromAlphaSM = other.DoublesFromAlphaSM;
        DoublesFromBetaSM = other.DoublesFromBetaSM;

        Eij = other.Eij;
        Eij_len = other.Eij_len;
        Eij_offset = other.Eij_offset;
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
        thrust::host_vector<size_t> offset_ij(braAlphaSize * braBetaSize);

        size_t nAlpha = helper.SinglesFromAlphaSM.size();
        size_t nBeta = helper.SinglesFromBetaSM.size();
        size_t data_start = nAlpha*4 + nBeta*4;
        size_t size = data_start;

        for(size_t i=0; i < nAlpha; i++) {
            size += helper.SinglesFromAlphaLen[i];
            size += helper.DoublesFromAlphaLen[i];
        }
        for(size_t i=0; i < nBeta; i++) {
            size += helper.SinglesFromBetaLen[i];
            size += helper.DoublesFromBetaLen[i];
        }

        size_t offset = 0;
        if (precalculate) {
            if (taskType == 0) {
                for(size_t ia = 0; ia < braAlphaSize; ia++) {
                    for(size_t ib = 0; ib < braBetaSize; ib++) {
                        size_t idx = ia * braBetaSize + ib;
                        offset_ij[idx] = offset;
                        offset += helper.SinglesFromAlphaLen[ia] * helper.SinglesFromBetaLen[ib];
                    }
                }
                veij.resize(offset, 0.0);
            } else if (taskType == 1) {
                for(size_t ia = 0; ia < braAlphaSize; ia++) {
                    for(size_t ib = 0; ib < braBetaSize; ib++) {
                        size_t idx = ia * braBetaSize + ib;
                        offset_ij[idx] = offset;
                        offset += helper.SinglesFromBetaLen[ia] + helper.DoublesFromBetaLen[ib];
                    }
                }
            } else {    // type = 2
                for(size_t ia = 0; ia < braAlphaSize; ia++) {
                    for(size_t ib = 0; ib < braBetaSize; ib++) {
                        size_t idx = ia * braBetaSize + ib;
                        offset_ij[idx] = offset;
                        offset += helper.SinglesFromAlphaLen[ia] + helper.DoublesFromAlphaLen[ib];
                    }
                }
            }

            veij.resize(offset, 0.0);
            storage.resize(size + braAlphaSize * braBetaSize);
        } else {
            storage.resize(size);
        }
        base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;
        SinglesFromAlphaLen = base_memory + count;
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
        count += (size - data_start);

        if (precalculate) {
            Eij = (ElemT*)thrust::raw_pointer_cast(veij.data());

            size_t mem_size = storage.size() * sizeof(size_t) + offset * sizeof(ElemT);
            std::cout << " Memory size for Hamiltonian = " << (double)mem_size / 1073741824.0 << " GiB " << std::endl;

            Eij_offset = base_memory + count;
            thrust::copy_n(offset_ij.begin(), braAlphaSize * braBetaSize, storage.begin() + count);
        }
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
