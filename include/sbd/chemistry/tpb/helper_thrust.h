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

        base_memory = other.base_memory;
        SinglesFromAlphaBraIndex = other.SinglesFromAlphaBraIndex;
        SinglesFromAlphaKetIndex = other.SinglesFromAlphaKetIndex;
        DoublesFromAlphaBraIndex = other.DoublesFromAlphaBraIndex;
        DoublesFromAlphaKetIndex = other.DoublesFromAlphaKetIndex;
        SinglesFromBetaBraIndex = other.SinglesFromBetaBraIndex;
        SinglesFromBetaKetIndex = other.SinglesFromBetaKetIndex;
        DoublesFromBetaBraIndex = other.DoublesFromBetaBraIndex;
        DoublesFromBetaKetIndex = other.DoublesFromBetaKetIndex;

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
        std::vector<size_t> offset_single_alpha(braAlphaSize);
        std::vector<size_t> offset_double_alpha(braAlphaSize);
        for(size_t i=0; i < braAlphaSize; i++) {
            offset_single_alpha[i] = size_single_alpha;
            offset_double_alpha[i] = size_double_alpha;
            size_single_alpha += helper.SinglesFromAlphaLen[i];
            size_double_alpha += helper.DoublesFromAlphaLen[i];
        }
        size += size_single_alpha + size_double_alpha;

        std::vector<size_t> offset_single_beta(braBetaSize);
        std::vector<size_t> offset_double_beta(braBetaSize);
        for(size_t i=0; i < braBetaSize; i++) {
            offset_single_beta[i] = size_single_beta;
            offset_double_beta[i] = size_double_beta;
            size_single_beta += helper.SinglesFromBetaLen[i];
            size_double_beta += helper.DoublesFromBetaLen[i];
        }
        size += size_single_beta + size_double_beta;

        storage.resize(size*2);
        base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;

        SinglesFromAlphaBraIndex = base_memory + count;
        SinglesFromAlphaKetIndex = base_memory + count + size_single_alpha;
        for(size_t i=0; i < braAlphaSize; i++) {
            thrust::fill_n(storage.begin() + count + offset_single_alpha[i], helper.SinglesFromAlphaLen[i], i + helper.braAlphaStart);
            thrust::copy_n(helper.SinglesFromAlphaSM[i], helper.SinglesFromAlphaLen[i], storage.begin() + count + size_single_alpha + offset_single_alpha[i]);
        }
        count += size_single_alpha * 2;

        DoublesFromAlphaBraIndex = base_memory + count;
        DoublesFromAlphaKetIndex = base_memory + count + size_double_alpha;
        for(size_t i=0; i < braAlphaSize; i++) {
            thrust::fill_n(storage.begin() + count + offset_double_alpha[i], helper.DoublesFromAlphaLen[i], i + helper.braAlphaStart);
            thrust::copy_n(helper.DoublesFromAlphaSM[i], helper.DoublesFromAlphaLen[i], storage.begin() + count + size_double_alpha + offset_double_alpha[i]);
        }
        count += size_double_alpha * 2;

        SinglesFromBetaBraIndex = base_memory + count;
        SinglesFromBetaKetIndex = base_memory + count + size_single_beta;
        for(size_t i=0; i < braBetaSize; i++) {
            thrust::fill_n(storage.begin() + count + offset_single_beta[i], helper.SinglesFromBetaLen[i], i + helper.braBetaStart);
            thrust::copy_n(helper.SinglesFromBetaSM[i], helper.SinglesFromBetaLen[i], storage.begin() + count + size_single_beta + offset_single_beta[i]);
        }
        count += size_single_beta * 2;

        DoublesFromBetaBraIndex = base_memory + count;
        DoublesFromBetaKetIndex = base_memory + count + size_double_beta;
        for(size_t i=0; i < braBetaSize; i++) {
            thrust::fill_n(storage.begin() + count + offset_double_beta[i], helper.DoublesFromBetaLen[i], i + helper.braBetaStart);
            thrust::copy_n(helper.DoublesFromBetaSM[i], helper.DoublesFromBetaLen[i], storage.begin() + count + size_double_beta + offset_double_beta[i]);
        }
        count += size_double_beta * 2;
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
