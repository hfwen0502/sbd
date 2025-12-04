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

    // offset for non-load balanced mode
    size_t* SinglesFromAlphaOffset;
    size_t* SinglesFromBetaOffset;
    size_t* DoublesFromAlphaOffset;
    size_t* DoublesFromBetaOffset;

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

        SinglesFromAlphaOffset = other.SinglesFromAlphaOffset;
        SinglesFromBetaOffset = other.SinglesFromBetaOffset;
        DoublesFromAlphaOffset = other.DoublesFromAlphaOffset;
        DoublesFromBetaOffset = other.DoublesFromBetaOffset;

        size_single_alpha = other.size_single_alpha;
        size_double_alpha = other.size_double_alpha;
        size_single_beta = other.size_single_beta;
        size_double_beta = other.size_double_beta;
    }

    TaskHelpersThrust(thrust::device_vector<size_t>& storage, const TaskHelpers& helper, bool store_offset = false)
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
        // storage for offsets
        if (store_offset) {
            size += ((braAlphaSize + 1) + (braBetaSize + 1)) * 2;
        }

        size_single_alpha = 0;
        size_double_alpha = 0;
        size_single_beta = 0;
        size_double_beta = 0;
        std::vector<size_t> offset_single_alpha(braAlphaSize + 1);
        std::vector<size_t> offset_double_alpha(braAlphaSize + 1);
        for(size_t i=0; i < braAlphaSize; i++) {
            offset_single_alpha[i] = size_single_alpha;
            offset_double_alpha[i] = size_double_alpha;
            size_single_alpha += helper.SinglesFromAlphaLen[i];
            size_double_alpha += helper.DoublesFromAlphaLen[i];
        }
        offset_single_alpha[braAlphaSize] = size_single_alpha;
        offset_double_alpha[braAlphaSize] = size_double_alpha;
        size += size_single_alpha + size_double_alpha;

        std::vector<size_t> offset_single_beta(braBetaSize + 1);
        std::vector<size_t> offset_double_beta(braBetaSize + 1);
        for(size_t i=0; i < braBetaSize; i++) {
            offset_single_beta[i] = size_single_beta;
            offset_double_beta[i] = size_double_beta;
            size_single_beta += helper.SinglesFromBetaLen[i];
            size_double_beta += helper.DoublesFromBetaLen[i];
        }
        offset_single_beta[braBetaSize] = size_single_beta;
        offset_double_beta[braBetaSize] = size_double_beta;
        size += size_single_beta + size_double_beta;

        if (!store_offset)
            size *= 2;
        storage.resize(size);
        base_memory = (size_t*)thrust::raw_pointer_cast(storage.data());

        size_t count = 0;

        if (store_offset) {
            // store offsets for non-balanced
            SinglesFromAlphaOffset = base_memory + count;
            thrust::copy_n(offset_single_alpha.begin(), braAlphaSize + 1, storage.begin() + count);
            count += braAlphaSize + 1;

            DoublesFromAlphaOffset = base_memory + count;
            thrust::copy_n(offset_double_alpha.begin(), braAlphaSize + 1, storage.begin() + count);
            count += braAlphaSize + 1;

            SinglesFromBetaOffset = base_memory + count;
            thrust::copy_n(offset_single_beta.begin(), braBetaSize + 1, storage.begin() + count);
            count += braBetaSize + 1;

            DoublesFromBetaOffset = base_memory + count;
            thrust::copy_n(offset_double_beta.begin(), braBetaSize + 1, storage.begin() + count);
            count += braBetaSize + 1;
        }

        if (store_offset) {
            SinglesFromAlphaKetIndex = base_memory + count;
        } else {
            SinglesFromAlphaBraIndex = base_memory + count + size_single_alpha;
            SinglesFromAlphaKetIndex = base_memory + count;
        }
        for(size_t i=0; i < braAlphaSize; i++) {
            thrust::copy_n(helper.SinglesFromAlphaSM[i], helper.SinglesFromAlphaLen[i], storage.begin() + count + offset_single_alpha[i]);
            if (!store_offset)
                thrust::fill_n(storage.begin() + size_single_alpha + count + offset_single_alpha[i], helper.SinglesFromAlphaLen[i], i + helper.braAlphaStart);
        }
        if (!store_offset)
            count += size_single_alpha;
        count += size_single_alpha;

        if (store_offset) {
            DoublesFromAlphaKetIndex = base_memory + count;
        } else {
            DoublesFromAlphaBraIndex = base_memory + count + size_double_alpha;
            DoublesFromAlphaKetIndex = base_memory + count;
        }
        for(size_t i=0; i < braAlphaSize; i++) {
            thrust::copy_n(helper.DoublesFromAlphaSM[i], helper.DoublesFromAlphaLen[i], storage.begin() + count + offset_double_alpha[i]);
            if (!store_offset)
                thrust::fill_n(storage.begin() + count + size_double_alpha + offset_double_alpha[i], helper.DoublesFromAlphaLen[i], i + helper.braAlphaStart);
        }
        if (!store_offset)
            count += size_double_alpha;
        count += size_double_alpha;

        if (store_offset) {
            SinglesFromBetaKetIndex = base_memory + count;
        } else {
            SinglesFromBetaBraIndex = base_memory + count + size_single_beta;
            SinglesFromBetaKetIndex = base_memory + count;
        }
        for(size_t i=0; i < braBetaSize; i++) {
            thrust::copy_n(helper.SinglesFromBetaSM[i], helper.SinglesFromBetaLen[i], storage.begin() + count + offset_single_beta[i]);
            if (!store_offset)
                thrust::fill_n(storage.begin() + count + size_single_beta + offset_single_beta[i], helper.SinglesFromBetaLen[i], i + helper.braBetaStart);
        }
        if (!store_offset)
            count += size_single_beta;
        count += size_single_beta;

        if (store_offset) {
            DoublesFromBetaKetIndex = base_memory + count;
        } else {
            DoublesFromBetaBraIndex = base_memory + count + size_double_beta;
            DoublesFromBetaKetIndex = base_memory + count;
        }
        for(size_t i=0; i < braBetaSize; i++) {
            thrust::copy_n(helper.DoublesFromBetaSM[i], helper.DoublesFromBetaLen[i], storage.begin() + count + offset_double_beta[i]);
            if (!store_offset)
                thrust::fill_n(storage.begin() + count + size_double_beta + offset_double_beta[i], helper.DoublesFromBetaLen[i], i + helper.braBetaStart);
        }
        if (!store_offset)
            count += size_double_beta;
        count += size_double_beta;
    }
};

}

#endif  //SBD_CHEMISTRY_TPB_HELPER_THRUST_H
