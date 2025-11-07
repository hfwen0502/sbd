/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_THRUST_H
#define SBD_CHEMISTRY_TPB_MULT_THRUST_H

#include <chrono>
#include <cstdio>

#include "sbd/chemistry/tpb/helper_thrust.h"


namespace sbd
{

template <typename ElemT>
class MultDataThrust {
public:
    thrust::device_vector<size_t> adets;
    thrust::device_vector<size_t> bdets;
    ElemT I0;
    oneInt_Thrust<ElemT> I1;
    thrust::device_vector<ElemT> I1_store;
    twoInt_Thrust<ElemT> I2;
    thrust::device_vector<ElemT> I2_store;
    thrust::device_vector<ElemT> I2_dm;
    thrust::device_vector<ElemT> I2_em;
    std::vector<thrust::device_vector<size_t>> helper_storage;
	std::vector<thrust::device_vector<ElemT>> eij_storage;
    size_t bit_length;
    size_t norbs;
    size_t size_D;
    std::vector<TaskHelpersThrust<ElemT>> helper;

    MultDataThrust() {}

    MultDataThrust( const std::vector<std::vector<size_t>> &adets_in,
    const std::vector<std::vector<size_t>> &bdets_in,
    const size_t bit_length_in,
    const size_t norbs_in,
    const std::vector<TaskHelpers> &helper_in,
    const ElemT &I0_in,
    const oneInt<ElemT> &I1_in,
    const twoInt<ElemT> &I2_in,
    int method);
};


template <typename ElemT>
class MultKernelBase {
protected:
    ElemT *Wb;
    ElemT* T;
    size_t *adets;
    size_t *bdets;
    ElemT I0;
    oneInt_Thrust<ElemT> I1;
    twoInt_Thrust<ElemT> I2;
    size_t bit_length;
    size_t norbs;
    size_t size_D;
    size_t mpi_rank_h;
    size_t mpi_size_h;
    size_t *det_local;
public:
    MultKernelBase() {}

    MultKernelBase( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data,
                thrust::device_vector<size_t>& v_det_local
                ) : I0(data.I0), I1(data.I1), I2(data.I2), bit_length(data.bit_length), norbs(data.norbs), size_D(data.size_D)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        T = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
        det_local = (size_t*)thrust::raw_pointer_cast(v_det_local.data());
    }

    __device__ __host__ void DetFromAlphaBeta(size_t *D, const size_t *A, const size_t *B)
    {
        size_t i;
        for (i = 0; i < size_D; i++) {
            D[i] = 0;
        }
        for (i = 0; i < norbs; i++) {
            size_t block = i / bit_length;
            size_t bit_pos = i % bit_length;
            size_t new_block_A = (2 * i) / bit_length;
            size_t new_bit_pos_A = (2 * i) % bit_length;
            size_t new_block_B = (2 * i + 1) / bit_length;
            size_t new_bit_pos_B = (2 * i + 1) % bit_length;

            if (A[block] & (size_t(1) << bit_pos))
            {
                D[new_block_A] |= size_t(1) << new_bit_pos_A;
            }
            if (B[block] & (size_t(1) << bit_pos))
            {
                D[new_block_B] |= size_t(1) << new_bit_pos_B;
            }
        }
    }

    __device__ __host__ ElemT Hij(const size_t *DetA, const size_t *DetB)
    {
        int c[4] = {0, 0, 0, 0};
        int d[4] = {0, 0, 0, 0};
        size_t nc = 0;
        size_t nd = 0;
        size_t L = norbs;

        size_t full_words = (2 * L) / bit_length;
        size_t remaining_bits = (2 * L) % bit_length;

        for (size_t i = 0; i < full_words; ++i) {
            size_t diff_c = DetA[i] & ~DetB[i];
            size_t diff_d = DetB[i] & ~DetA[i];
            for (size_t bit_pos = 0; bit_pos < bit_length; ++bit_pos) {
                if (diff_c & (static_cast<size_t>(1) << bit_pos)) {
                    c[nc] = i * bit_length + bit_pos;
                    nc++;
                }
                if (diff_d & (static_cast<size_t>(1) << bit_pos)) {
                    d[nd] = i * bit_length + bit_pos;
                    nd++;
                }
            }
        }

        if (remaining_bits > 0) {
            size_t mask = (static_cast<size_t>(1) << remaining_bits) - 1;
            size_t diff_c = (DetA[full_words] & ~DetB[full_words]) & mask;
            size_t diff_d = (DetB[full_words] & ~DetA[full_words]) & mask;
            for (size_t bit_pos = 0; bit_pos < remaining_bits; ++bit_pos) {
                if (diff_c & (static_cast<size_t>(1) << bit_pos)) {
                    c[nc] = bit_length * full_words + bit_pos;
                    nc++;
                }
                if (diff_d & (static_cast<size_t>(1) << bit_pos)) {
                    d[nd] = bit_length * full_words + bit_pos;
                    nd++;
                }
            }
        }

        if (nc == 0) {
            return ZeroExcite(DetB, L);
        }
        else if (nc == 1) {
            return OneExcite(DetB, d[0], c[0]);
        }
        else if (nc == 2) {
            return TwoExcite(DetB, d[0], d[1], c[0], c[1]);
        }
        return ElemT(0.0);
    }

    inline __host__ __device__ size_t pop_count_kernel(size_t val)
    {
        size_t count = val;
        count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
        count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
        count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
        count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
        count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
        count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
        return count;
    }

    inline __device__ __host__ void parity(const size_t* dets, const int start, const int end, double& sgn)
    {
        size_t blockStart = start / bit_length;
        size_t bitStart = start % bit_length;

        size_t blockEnd = end / bit_length;
        size_t bitEnd = end % bit_length;

        size_t nonZeroBits = 0; // counter for nonzero bits

        // 1. Count bits in the start block
        if (blockStart == blockEnd) {
            // the case where start and end is same block
            size_t mask = ((size_t(1) << bitEnd) - 1) ^ ((size_t(1) << bitStart) - 1);
            nonZeroBits += pop_count_kernel(dets[blockStart] & mask);
        }
        else {
            // 2. Handle the partial bits in the start block
            if (bitStart != 0) {
                size_t mask = ~((size_t(1) << bitStart) - 1); // count after bitStart
                nonZeroBits += pop_count_kernel(dets[blockStart] & mask);
                blockStart++;
            }

            // 3. Handle full blocks in between
            for (size_t i = blockStart; i < blockEnd; i++) {
                nonZeroBits += pop_count_kernel(dets[i]);
            }

            // 4. Handle the partial bits in the end block
            if (bitEnd != 0) {
                size_t mask = (size_t(1) << bitEnd) - 1; // count before bitEnd
                nonZeroBits += pop_count_kernel(dets[blockEnd] & mask);
            }
        }

        // parity estimation
        sgn *= (-2. * (nonZeroBits % 2) + 1);

        // flip sign if start == 1
        if ((dets[start / bit_length] >> (start % bit_length)) & 1) {
            sgn *= -1.;
        }
    }

  inline __device__ __host__ bool getocc(const size_t* det, int x)
    {
        size_t index = x / bit_length;
        size_t bit_pos = x % bit_length;
        return (det[index] >> bit_pos) & 1;
    }

    inline __device__ __host__ ElemT ZeroExcite(const size_t* det, const size_t L)
    {
        ElemT energy(0.0);

        for (int i = 0; i < 2 * L; i++) {
            if (getocc(det, i)) {
                energy += I1.Value(i, i);
                for (int j = i + 1; j < 2 * L; j++) {
                    if (getocc(det, j)) {
                        energy += I2.DirectValue(i / 2, j / 2);
                        if ((i % 2) == (j % 2)) {
                            energy -= I2.ExchangeValue(i / 2, j / 2);
                        }
                    }
                }
            }
        }
        return energy + I0;
    }

    inline __device__ __host__ ElemT OneExcite(const size_t* det, int i, int a)
    {
        double sgn = 1.0;
        parity(det, std::min(i, a), std::max(i, a), sgn);
        ElemT energy = I1.Value(a, i);
        for (int x = 0; x < size_D; x++) {
            size_t bits = det[x];
            for (int pos = 0; pos < bit_length; pos++) {
                if (bits & 1ULL) {
                    int j = x * bit_length + pos;
                    energy += (I2.Value(a, i, j, j) - I2.Value(a, j, j, i));
                }
                bits >>= 1;
            }
        }
        energy *= ElemT(sgn);
        return energy;
    }

    inline __device__ __host__ ElemT TwoExcite(const size_t* det, int i, int j, int a, int b)
    {
        double sgn = 1.0;
        int I = std::min(i, j);
        int J = std::max(i, j);
        int A = std::min(a, b);
        int B = std::max(a, b);
        parity(det, std::min(I, A), std::max(I, A), sgn);
        parity(det, std::min(J, B), std::max(J, B), sgn);
        if (A > J || B < I)
            sgn *= -1.0;
        return ElemT(sgn) * (I2.Value(A, I, B, J) - I2.Value(A, J, B, I));
    }

    void set_mpi_size(size_t h_rank, size_t h_size)
    {
        mpi_rank_h = h_rank;
        mpi_size_h = h_size;
    }
};

template <typename ElemT>
class MultKernel : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    int task_id;
public:
    MultKernel( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data,
                thrust::device_vector<size_t>& v_det_local
                ) : MultKernelBase<ElemT>(v_wb, v_t, data, v_det_local) {}


    void set_helper(const TaskHelpersThrust<ElemT>& h, int id)
    {
        helper = h;
        task_id = id;
    }

    // taks == 0
    __device__ __host__ void Task0(size_t ia, size_t ib, size_t* DetI, size_t* DetJ)
    {
        size_t braIdx = (ia-helper.braAlphaStart)*(helper.braBetaEnd - helper.braBetaStart)
								+ib-helper.braBetaStart;
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);

            // two-particle excitation composed of single alpha and single beta
            for(size_t j=0; j < helper.SinglesFromAlphaLen[ia-helper.braAlphaStart]; j++) {
                size_t ja = helper.SinglesFromAlphaSM[ia-helper.braAlphaStart][j];
                for(size_t k=0; k < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; k++) {
                    size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][k];
                    size_t ketIdx = (ja-helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                                    +jb-helper.ketBetaStart;
                    this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + jb * this->size_D);

                    ElemT eij = this->Hij(DetI, DetJ);
                    this->Wb[braIdx] += eij * this->T[ketIdx];
                }
            }
        }
    }

    // taks == 1
    __device__ __host__ void Task1(size_t ia, size_t ib, size_t* DetI, size_t* DetJ)
    {
        size_t braIdx = (ia-helper.braAlphaStart)*(helper.braBetaEnd - helper.braBetaStart)
                    +ib-helper.braBetaStart;
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);

            // single beta excitation
            for(size_t j=0; j < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; j++) {
                size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][j];
                size_t ketIdx = (ia-helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb-helper.ketBetaStart;
                this->DetFromAlphaBeta(DetJ, this->adets + ia * this->size_D, this->bdets + jb * this->size_D);
                ElemT eij = this->Hij(DetI ,DetJ);
                this->Wb[braIdx] += eij * this->T[ketIdx];
            }
            // double beta excitation
            for(size_t j=0; j < helper.DoublesFromBetaLen[ib-helper.braBetaStart]; j++) {
                size_t jb = helper.DoublesFromBetaSM[ib-helper.braBetaStart][j];
                size_t ketIdx = (ia-helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb-helper.ketBetaStart;
                this->DetFromAlphaBeta(DetJ, this->adets + ia * this->size_D, this->bdets + jb * this->size_D);
                ElemT eij = this->Hij(DetI, DetJ);
                this->Wb[braIdx] += eij * this->T[ketIdx];
            }
        }
    }

    // taks == 2
    __device__ __host__ void Task2(size_t ia, size_t ib, size_t* DetI, size_t* DetJ)
    {
        size_t braIdx = (ia-helper.braAlphaStart)*(helper.braBetaEnd - helper.braBetaStart)
                    +ib-helper.braBetaStart;
        if( (braIdx % this->mpi_size_h) == this->mpi_rank_h ) {
            this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);

            // single alpha excitation
            for(size_t j=0; j < helper.SinglesFromAlphaLen[ia-helper.braAlphaStart]; j++) {
                size_t ja = helper.SinglesFromAlphaSM[ia-helper.braAlphaStart][j];
                size_t ketIdx = (ja-helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                                +ib-helper.ketBetaStart;
                this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + ib * this->size_D);
                ElemT eij = this->Hij(DetI, DetJ);
                this->Wb[braIdx] += eij * this->T[ketIdx];
            }
            // double alpha excitation
            for(size_t j=0; j < helper.DoublesFromAlphaLen[ia-helper.braAlphaStart]; j++) {
                size_t ja = helper.DoublesFromAlphaSM[ia-helper.braAlphaStart][j];
                size_t ketIdx = (ja-helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                            + ib-helper.ketBetaStart;
                this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + ib * this->size_D);
                ElemT eij = this->Hij(DetI, DetJ);
                this->Wb[braIdx] += eij * this->T[ketIdx];
            }
        }
    }

    // kernel entry point
    __device__ __host__ void operator()(int i)
    {
        size_t ia, ib, braBetaSize;
        size_t* DetI = this->det_local + i * this->size_D * 2;
        size_t* DetJ = DetI + this->size_D;

        braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        ia = i / braBetaSize;
        ib = i - ia * braBetaSize;

        if (helper.taskType == 0) {
            Task0(ia + helper.braAlphaStart, ib + helper.braBetaStart, DetI, DetJ);
        } else if (helper.taskType == 1) {
            Task1(ia + helper.braAlphaStart, ib + helper.braBetaStart, DetI, DetJ);
        } else {
            Task2(ia + helper.braAlphaStart, ib + helper.braBetaStart, DetI, DetJ);
        }
    }
};




template <typename ElemT>
class MakeHamiltonianKernel : public MultKernelBase<ElemT> {
protected:
    TaskHelpersThrust<ElemT> helper;
public:
    MakeHamiltonianKernel(const MultDataThrust<ElemT>& data)
    {
        this->I0 = data.I0;
        this->I1 = data.I1;
        this->I2 = data.I2;
        this->bit_length = data.bit_length;
        this->norbs = data.norbs;
        this->size_D = data.size_D;

        this->adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        this->bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
    }

    void set_helper(const TaskHelpersThrust<ElemT>& h)
    {
        helper = h;
    }

    // pre-calculate Eij
    __device__ __host__ void precalculate_eij(size_t ia, size_t ib, size_t* DetI, size_t* DetJ)
    {
        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart)
								+ ib - helper.braBetaStart;

        this->DetFromAlphaBeta(DetI, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
        size_t offset = this->helper.Eij_offset[braIdx];

        if( this->helper.taskType == 2 ) { // beta range are same
            // single alpha excitation
            for(size_t j=0; j < this->helper.SinglesFromAlphaLen[ia - this->helper.braAlphaStart]; j++) {
                size_t ja = this->helper.SinglesFromAlphaSM[ia-this->helper.braAlphaStart][j];
                this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + ib * this->size_D);
                this->helper.Eij[offset++] = this->Hij(DetI, DetJ);
            }
            // double alpha excitation
            for(size_t j=0; j < this->helper.DoublesFromAlphaLen[ia-this->helper.braAlphaStart]; j++) {
                size_t ja = this->helper.DoublesFromAlphaSM[ia-this->helper.braAlphaStart][j];
                this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + ib * this->size_D);
                this->helper.Eij[offset++] = this->Hij(DetI, DetJ);
            }
        } else if ( this->helper.taskType == 1 ) { // alpha range are same
            // single beta excitation
            for(size_t j=0; j < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; j++) {
                size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][j];
                this->DetFromAlphaBeta(DetJ, this->adets + ia * this->size_D, this->bdets + jb * this->size_D);
                this->helper.Eij[offset++] = this->Hij(DetI ,DetJ);
            }
            // double beta excitation
            for(size_t j=0; j < helper.DoublesFromBetaLen[ib-helper.braBetaStart]; j++) {
                size_t jb = helper.DoublesFromBetaSM[ib-helper.braBetaStart][j];
                this->DetFromAlphaBeta(DetJ, this->adets + ia * this->size_D, this->bdets + jb * this->size_D);
                this->helper.Eij[offset++] = this->Hij(DetI, DetJ);
            }
        } else {
            // two-particle excitation composed of single alpha and single beta
            for(size_t j=0; j < helper.SinglesFromAlphaLen[ia-helper.braAlphaStart]; j++) {
                size_t ja = helper.SinglesFromAlphaSM[ia-helper.braAlphaStart][j];
                for(size_t k=0; k < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; k++) {
                    size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][k];
                    this->DetFromAlphaBeta(DetJ, this->adets + ja * this->size_D, this->bdets + jb * this->size_D);

                    this->helper.Eij[offset++] = this->Hij(DetI, DetJ);
                }
            }
        } // if ( this->helper.taskType == ? )
    }

    // kernel entry point
    __device__ __host__ void operator()(int i)
    {
        size_t ia, ib, braBetaSize;
        braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        ia = i / braBetaSize;
        ib = i - ia * braBetaSize;
        size_t* DetI = new size_t[this->size_D * 2];
        size_t* DetJ = new size_t[this->size_D * 2];

        precalculate_eij(ia + helper.braAlphaStart, ib + helper.braBetaStart, DetI, DetJ);

        delete[] DetI;
        delete[] DetJ;
    }
};

template <typename ElemT>
class MultEijKernel {
protected:
    TaskHelpersThrust<ElemT> helper;
    ElemT *Wb;
    ElemT* T;
    size_t mpi_rank_h;
    size_t mpi_size_h;
public:
    MultEijKernel(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const size_t rank, size_t msize)
    {
        helper = h;
        Wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        T = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        mpi_rank_h = rank;
        mpi_size_h = msize;
    }

    // kernel entry point
    __device__ __host__ void operator()(int idx)
    {
        size_t braIdx = idx;
        size_t ia, ib, braBetaSize;
        braBetaSize = helper.braBetaEnd - helper.braBetaStart;
        ia = idx / braBetaSize;
        ib = idx - ia * braBetaSize;

        ia += helper.braAlphaStart;
        ib += helper.braBetaStart;

        if( (braIdx % mpi_size_h) == mpi_rank_h ) {
            size_t offset = helper.Eij_offset[braIdx];

            if( this->helper.taskType == 2 ) { // beta range are same
                // single alpha excitation
                for(size_t j=0; j < this->helper.SinglesFromAlphaLen[ia - this->helper.braAlphaStart]; j++) {
                    size_t ja = this->helper.SinglesFromAlphaSM[ia-this->helper.braAlphaStart][j];
                    size_t ketIdx = (ja-this->helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                                    +ib-this->helper.ketBetaStart;
                    Wb[braIdx] += helper.Eij[offset++] * T[ketIdx];
                }
                // double alpha excitation
                for(size_t j=0; j < this->helper.DoublesFromAlphaLen[ia-this->helper.braAlphaStart]; j++) {
                    size_t ja = this->helper.DoublesFromAlphaSM[ia-this->helper.braAlphaStart][j];
                    size_t ketIdx = (ja-this->helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                                + ib-this->helper.ketBetaStart;
                    Wb[braIdx] += helper.Eij[offset++] * T[ketIdx];
                }
            } else if ( this->helper.taskType == 1 ) { // alpha range are same
                // single beta excitation
                for(size_t j=0; j < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; j++) {
                    size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][j];
                    size_t ketIdx = (ia-helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb-helper.ketBetaStart;
                    Wb[braIdx] += helper.Eij[offset++] * T[ketIdx];
                }
                // double beta excitation
                for(size_t j=0; j < helper.DoublesFromBetaLen[ib-helper.braBetaStart]; j++) {
                    size_t jb = helper.DoublesFromBetaSM[ib-helper.braBetaStart][j];
                    size_t ketIdx = (ia-helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                                + jb-helper.ketBetaStart;
                    Wb[braIdx] += helper.Eij[offset++] * T[ketIdx];
                }
            } else {
                // two-particle excitation composed of single alpha and single beta
                for(size_t j=0; j < helper.SinglesFromAlphaLen[ia-helper.braAlphaStart]; j++) {
                    size_t ja = helper.SinglesFromAlphaSM[ia-helper.braAlphaStart][j];
                    for(size_t k=0; k < helper.SinglesFromBetaLen[ib-helper.braBetaStart]; k++) {
                        size_t jb = helper.SinglesFromBetaSM[ib-helper.braBetaStart][k];
                        size_t ketIdx = (ja-helper.ketAlphaStart)*(helper.ketBetaEnd - helper.ketBetaStart)
                                        +jb-helper.ketBetaStart;
                        Wb[braIdx] += helper.Eij[offset++] * T[ketIdx];
                    }
                }
            } // if ( this->helper.taskType == ? )
        }
    }
};

// kernel for Wb initialization
template <typename ElemT>
struct Wb_init_kernel {
    __host__ __device__ ElemT operator()(const thrust::tuple<ElemT, ElemT, ElemT>& t) const
    {
        return thrust::get<0>(t) + thrust::get<1>(t) * thrust::get<2>(t);
    }
};

template <typename ElemT>
void mult(const std::vector<ElemT> &hii,
            const std::vector<ElemT> &Wk,
            std::vector<ElemT> &Wb,
            const MultDataThrust<ElemT>& data,
            const size_t adet_comm_size,
            const size_t bdet_comm_size,
            MPI_Comm h_comm,
            MPI_Comm b_comm,
            MPI_Comm t_comm,
            int method)
{
    // this is wrapper mult function with data copy

    // copyin hii
    thrust::device_vector<ElemT> hii_dev(hii.size());
    thrust::copy_n(hii.begin(), hii.size(), hii_dev.begin());

    // copyin Wk
    thrust::device_vector<ElemT> Wk_dev(Wk.size());
    thrust::copy_n(Wk.begin(), Wk.size(), Wk_dev.begin());

    thrust::device_vector<ElemT> Wb_dev(Wk.size(), 0.0);

    mult(hii_dev, Wk_dev, Wb_dev, data,
		  adet_comm_size, bdet_comm_size,
		  h_comm, b_comm, t_comm, method);

    // copyout Wb
    thrust::copy_n(Wb_dev.begin(), Wb_dev.size(), Wb.begin());
}


template <typename ElemT>
void mult(const thrust::device_vector<ElemT> &hii,
            const thrust::device_vector<ElemT> &Wk,
            thrust::device_vector<ElemT> &Wb,
            const MultDataThrust<ElemT>& data,
            const size_t adet_comm_size,
            const size_t bdet_comm_size,
            MPI_Comm h_comm,
            MPI_Comm b_comm,
            MPI_Comm t_comm,
            int method)
{

#ifdef SBD_DEBUG_TUNING
    std::cout << " multiplication by Robert is called " << std::endl;
#endif

    int mpi_rank_h = 0;
    int mpi_size_h = 1;
    MPI_Comm_rank(h_comm, &mpi_rank_h);
    MPI_Comm_size(h_comm, &mpi_size_h);

    int mpi_size_b;
    MPI_Comm_size(b_comm, &mpi_size_b);
    int mpi_rank_b;
    MPI_Comm_rank(b_comm, &mpi_rank_b);
    int mpi_size_t;
    MPI_Comm_size(t_comm, &mpi_size_t);
    int mpi_rank_t;
    MPI_Comm_rank(t_comm, &mpi_rank_t);
    size_t braAlphaSize = 0;
    size_t braBetaSize = 0;
    if (data.helper.size() != 0) {
        braAlphaSize = data.helper[0].braAlphaEnd - data.helper[0].braAlphaStart;
        braBetaSize = data.helper[0].braBetaEnd - data.helper[0].braBetaStart;
    }

    size_t adet_min = 0;
    size_t adet_max = data.adets.size() / data.size_D;
    size_t bdet_min = 0;
    size_t bdet_max = data.bdets.size() / data.size_D;
    get_mpi_range(adet_comm_size, 0, adet_min, adet_max);
    get_mpi_range(bdet_comm_size, 0, bdet_min, bdet_max);
    size_t max_det_size = (adet_max - adet_min) * (bdet_max - bdet_min);

    int num_threads = 1;

    thrust::device_vector<ElemT> T(max_det_size);
    thrust::device_vector<ElemT> R(max_det_size);

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    if (data.helper.size() != 0) {
        Mpi2dSlide_Thrust(Wk, T, adet_comm_size, bdet_comm_size,
                    -data.helper[0].adetShift, -data.helper[0].bdetShift, b_comm);
    }
    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<size_t> det_local_dev;
    if (method == 0)
        det_local_dev.resize(data.size_D * 2 * braAlphaSize * braBetaSize);

    if (mpi_rank_t == 0){
        auto Wb_init_iter = thrust::make_zip_iterator(thrust::make_tuple(Wb.begin(), hii.begin(), T.begin()));
        thrust::transform(thrust::device, Wb_init_iter, Wb_init_iter + T.size(), Wb.begin(), Wb_init_kernel<ElemT>());
    }

    double time_slid = 0.0;
    for (size_t task = 0; task < data.helper.size(); task++) {
#ifdef SBD_DEBUG_MULT
        std::cout << " Start multiplication for task " << task << " at (h,b,t) = ("
                    << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t << "): task type = "
                    << data.helper[task].taskType << ", bra-adet range = ["
                    << data.helper[task].braAlphaStart << "," << data.helper[task].braAlphaEnd << "), bra-bdet range = ["
                    << data.helper[task].braBetaStart << "," << data.helper[task].braBetaEnd << "), ket-adet range = ["
                    << data.helper[task].ketAlphaStart << "," << data.helper[task].ketAlphaEnd << "), ket-bdet range = ["
                    << data.helper[task].ketBetaStart << "," << data.helper[task].ketBetaEnd << "), ket wf =";
        for (size_t i = 0; i < std::min(static_cast<size_t>(4), T.size()); i++)
        {
            std::cout << " " << T[i];
        }
        std::cout << std::endl;
#endif

        braAlphaSize = data.helper[task].braAlphaEnd - data.helper[task].braAlphaStart;
        braBetaSize = data.helper[task].braBetaEnd - data.helper[task].braBetaStart;
        auto ci = thrust::counting_iterator<size_t>(0);
        if (method == 0) {
            MultKernel kernel( Wb, T, data, det_local_dev);
            kernel.set_mpi_size(mpi_rank_h, mpi_size_h);
            kernel.set_helper(data.helper[task], task);

            // run kernel for this task
            thrust::for_each_n(thrust::device, ci, braAlphaSize * braBetaSize, kernel);
        } else {
            MultEijKernel<ElemT> kernel(data.helper[task], Wb, T, mpi_rank_h, mpi_size_h);

            thrust::for_each_n(thrust::device, ci, braAlphaSize * braBetaSize, kernel);
        }


        if (data.helper[task].taskType == 0 && task != data.helper.size() - 1)
        {
#ifdef SBD_DEBUG_MULT
            size_t adet_rank = mpi_rank_b / bdet_comm_size;
            size_t bdet_rank = mpi_rank_b % bdet_comm_size;
            size_t adet_rank_task = (adet_rank + data.helper[task].adetShift) % adet_comm_size;
            size_t bdet_rank_task = (bdet_rank + data.helper[task].bdetShift) % bdet_comm_size;
            size_t adet_rank_next = (adet_rank + data.helper[task + 1].adetShift) % adet_comm_size;
            size_t bdet_rank_next = (bdet_rank + data.helper[task + 1].bdetShift) % bdet_comm_size;
            std::cout << " mult: task " << task << " at mpi process (h,b,t) = ("
                        << mpi_rank_h << "," << mpi_rank_b << "," << mpi_rank_t
                        << "): two-dimensional slide communication from ("
                        << adet_rank_task << "," << bdet_rank_task << ") to ("
                        << adet_rank_next << "," << bdet_rank_next << ")"
                        << std::endl;

#endif
            int adetslide = data.helper[task].adetShift - data.helper[task + 1].adetShift;
            int bdetslide = data.helper[task].bdetShift - data.helper[task + 1].bdetShift;
            R = T;
            auto time_slid_start = std::chrono::high_resolution_clock::now();
            Mpi2dSlide_Thrust(R, T, adet_comm_size, bdet_comm_size, adetslide, bdetslide, b_comm);
            auto time_slid_end = std::chrono::high_resolution_clock::now();
            auto time_slid_count = std::chrono::duration_cast<std::chrono::microseconds>(time_slid_end - time_slid_start).count();
            time_slid += 1.0e-6 * time_slid_count;
        }

    } // end for(size_t task=0; task < data.helper.size(); task++)

    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
    MpiAllreduce_Thrust(Wb, MPI_SUM, t_comm);
    MpiAllreduce_Thrust(Wb, MPI_SUM, h_comm);
    auto time_comm_end = std::chrono::high_resolution_clock::now();

#ifdef SBD_DEBUG_MULT
    auto time_copy_count = std::chrono::duration_cast<std::chrono::microseconds>(time_copy_end - time_copy_start).count();
    auto time_mult_count = std::chrono::duration_cast<std::chrono::microseconds>(time_mult_end - time_mult_start).count();
    auto time_comm_count = std::chrono::duration_cast<std::chrono::microseconds>(time_comm_end - time_comm_start).count();

    double time_copy = 1.0e-6 * time_copy_count;
    double time_mult = 1.0e-6 * time_mult_count;
    double time_comm = 1.0e-6 * time_comm_count;
    std::cout << " mult: time for first copy     = " << time_copy << std::endl;
    std::cout << " mult: time for multiplication = " << time_mult << std::endl;
    std::cout << " mult: time for 2d slide comm  = " << time_slid << std::endl;
    std::cout << " mult: time for allreduce comm = " << time_comm << std::endl;
#endif

} // end function



// contructor for Mult data
template <typename ElemT>
MultDataThrust<ElemT>::MultDataThrust( const std::vector<std::vector<size_t>> &adets_in,
    const std::vector<std::vector<size_t>> &bdets_in,
    const size_t bit_length_in,
    const size_t norbs_in,
    const std::vector<TaskHelpers> &helper_in,
    const ElemT &I0_in,
    const oneInt<ElemT> &I1_in,
    const twoInt<ElemT> &I2_in,
    int method) : I0(I0_in)
{
    bit_length = bit_length_in;
    norbs = norbs_in;
    size_D = (2 * norbs + bit_length - 1) / bit_length;

    // copyin I1
    I1_store.resize(I1_in.store.size());
    thrust::copy_n(I1_in.store.begin(), I1_in.store.size(), I1_store.begin());
    I1 = oneInt_Thrust<ElemT>(I1_store, I1_in.norbs);

    // copyin I2
    I2_store.resize(I2_in.store.size());
    thrust::copy_n(I2_in.store.begin(), I2_in.store.size(), I2_store.begin());
    I2_dm.resize(I2_in.DirectMat.size());
    thrust::copy_n(I2_in.DirectMat.begin(), I2_in.DirectMat.size(), I2_dm.begin());
    I2_em.resize(I2_in.ExchangeMat.size());
    thrust::copy_n(I2_in.ExchangeMat.begin(), I2_in.ExchangeMat.size(), I2_em.begin());
    I2 = twoInt_Thrust<ElemT>(I2_store, I2_in.norbs, I2_dm, I2_em, I2_in.zero, I2_in.maxEntry);

    // copyin adets, bdets
    adets.resize(size_D * adets_in.size());
    bdets.resize(size_D * bdets_in.size());
    for (int i = 0; i < adets_in.size(); i++) {
        thrust::copy(adets_in[i].begin(), adets_in[i].end(), adets.begin() + i * size_D);
    }
    for (int i = 0; i < bdets_in.size(); i++) {
        thrust::copy(bdets_in[i].begin(), bdets_in[i].end(), bdets.begin() + i * size_D);
    }

    // copyin helpers
    helper_storage.resize(helper_in.size());
    eij_storage.resize(helper_in.size());

    for (size_t task = 0; task < helper_in.size(); task++) {
        helper.push_back(TaskHelpersThrust<ElemT>(helper_storage[task], eij_storage[task], helper_in[task], method == 1));

        // make Hamiltonian here
        if (method == 1) {
            MakeHamiltonianKernel<ElemT> kernel(*this);
            kernel.set_helper(helper[task]);

            size_t braAlphaSize = helper[task].braAlphaEnd - helper[task].braAlphaStart;
            size_t braBetaSize = helper[task].braBetaEnd - helper[task].braBetaStart;

            auto ci = thrust::counting_iterator<size_t>(0);
            thrust::for_each_n(thrust::device, ci, braAlphaSize * braBetaSize, kernel);
        }
    }
    if (method == 1) {
        // free unused GPU memory space for method 1
        adets.clear();
        adets.shrink_to_fit();
        bdets.clear();
        bdets.shrink_to_fit();
        I1_store.clear();
        I1_store.shrink_to_fit();
        I2_store.clear();
        I2_store.shrink_to_fit();
        I2_dm.clear();
        I2_dm.shrink_to_fit();
        I2_em.clear();
        I2_em.shrink_to_fit();
    }
}


}

#endif
