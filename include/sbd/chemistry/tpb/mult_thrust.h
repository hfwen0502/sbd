/**
@file sbd/chemistry/tpb/mult.h
@brief Function to perform Hamiltonian operation for twist-basis parallelization scheme
*/
#ifndef SBD_CHEMISTRY_TPB_MULT_THRUST_H
#define SBD_CHEMISTRY_TPB_MULT_THRUST_H

#include <chrono>
#include <cstdio>

#include "sbd/chemistry/tpb/helper_thrust.h"


// per thread DetI, DetJ storage size (1GB max)
#define MAX_DET_SIZE 134217728

namespace sbd
{

template <typename ElemT>
class MultDataThrust {
public:
    thrust::device_vector<size_t> adets;
    thrust::device_vector<size_t> bdets;
    thrust::device_vector<size_t> dets;
    size_t adets_size;
    size_t bdets_size;
    size_t bra_adets_begin;
    size_t bra_adets_end;
    size_t bra_bdets_begin;
    size_t bra_bdets_end;
    size_t ket_adets_begin;
    size_t ket_adets_end;
    size_t ket_bdets_begin;
    size_t ket_bdets_end;
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
    size_t num_max_threads;
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
    size_t adets_size;
    size_t bdets_size;
    ElemT I0;
    oneInt_Thrust<ElemT> I1;
    twoInt_Thrust<ElemT> I2;
    size_t bit_length;
    size_t norbs;
    size_t size_D;
    size_t mpi_rank_h;
    size_t mpi_size_h;
    size_t* adets;
    size_t* bdets;
    size_t* det_I;
    size_t* det_J;
public:
    MultKernelBase() {}

    MultKernelBase( const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data
                ) : I0(data.I0), I1(data.I1), I2(data.I2), bit_length(data.bit_length), norbs(data.norbs), size_D(data.size_D)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(v_wb.data());
        T = (ElemT*)thrust::raw_pointer_cast(v_t.data());
        adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
        det_I = (size_t*)thrust::raw_pointer_cast(data.dets.data());

        adets_size = data.adets_size;
        bdets_size = data.bdets_size;

        if (data.bra_adets_begin != data.ket_adets_begin || data.bra_bdets_begin != data.ket_bdets_begin)
            det_J = det_I + adets_size * bdets_size * size_D;
        else
            det_J = det_I;
    }

    MultKernelBase(const MultDataThrust<ElemT>& data)
                 : I0(data.I0), I1(data.I1), I2(data.I2), bit_length(data.bit_length), norbs(data.norbs), size_D(data.size_D)
    {
        adets = (size_t*)thrust::raw_pointer_cast(data.adets.data());
        bdets = (size_t*)thrust::raw_pointer_cast(data.bdets.data());
        det_I = (size_t*)thrust::raw_pointer_cast(data.dets.data());

        adets_size = data.adets_size;
        bdets_size = data.bdets_size;

        if (data.bra_adets_begin != data.ket_adets_begin || data.bra_bdets_begin != data.ket_bdets_begin)
            det_J = det_I + adets_size * bdets_size * size_D;
        else
            det_J = det_I;
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

            if (A[block] & (size_t(1) << bit_pos)) {
                D[new_block_A] |= size_t(1) << new_bit_pos_A;
            }
            if (B[block] & (size_t(1) << bit_pos)) {
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

    inline __device__ __host__ void AddEnergy(TaskHelpersThrust<ElemT>& helper, size_t i, size_t ia, size_t ib, size_t ja, size_t jb)
    {
        size_t braIdx = (ia - helper.braAlphaStart) * (helper.braBetaEnd - helper.braBetaStart) + ib - helper.braBetaStart;
        if( (braIdx % mpi_size_h) == mpi_rank_h ) {
            size_t ketIdx = (ja - helper.ketAlphaStart) * (helper.ketBetaEnd - helper.ketBetaStart)
                            + jb - helper.ketBetaStart;

            size_t* DetI = this->det_I + ((ia - helper.braAlphaStart) * bdets_size + ib - helper.braBetaStart) * this->size_D;
            size_t* DetJ = this->det_J + ((ja - helper.ketAlphaStart) * bdets_size + jb - helper.ketBetaStart) * this->size_D;

            /*size_t* DetI = this->det_I + i * size_D * 2;
            size_t* DetJ = DetI + size_D;
            DetFromAlphaBeta(DetI, adets + ia * size_D, bdets + ib * size_D);
            DetFromAlphaBeta(DetJ, adets + ja * size_D, bdets + jb * size_D);*/

            ElemT eij = Hij(DetI, DetJ);
            atomicAdd(Wb + braIdx, eij * T[ketIdx]);
        }
    }

    void set_mpi_size(size_t h_rank, size_t h_size)
    {
        mpi_rank_h = h_rank;
        mpi_size_h = h_size;
    }
};

template <typename ElemT>
class DetFromAlphaBetaKernel : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    bool update_I;
    bool update_J;
public:
    DetFromAlphaBetaKernel(const TaskHelpersThrust<ElemT>& h, const MultDataThrust<ElemT>& data, bool i, bool j)
                        : MultKernelBase<ElemT>(data)
    {
        helper = h;
        update_I = i;
        update_J = j;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t a = i / this->bdets_size;
        size_t b = i - a * this->bdets_size;
        size_t* Det;
        if (update_I) {
            Det = this->det_I + i * this->size_D;
            size_t ia = a + helper.braAlphaStart;
            size_t ib = b + helper.braBetaStart;

            this->DetFromAlphaBeta(Det, this->adets + ia * this->size_D, this->bdets + ib * this->size_D);
        }
        if (update_J && this->det_I != this->det_J) {
            Det = this->det_J + i * this->size_D;
            size_t ja = a + helper.ketAlphaStart;
            size_t jb = b + helper.ketBetaStart;
            this->DetFromAlphaBeta(Det, this->adets + ja * this->size_D, this->bdets + jb * this->size_D);
        }
    }
};


template <typename ElemT>
class MultAlphaBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    MultAlphaBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data, size_t o
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = (i + offset) / helper.size_single_beta;
        size_t k = (i + offset) - j * helper.size_single_beta;

        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];
        this->AddEnergy(helper, i, ia, ib, ja, jb);
    }
};


template <typename ElemT>
class MultSingleAlpha : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    MultSingleAlpha(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data, size_t o
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t k = (i + offset) / helper.size_single_alpha;
        size_t j = (i + offset) - k * helper.size_single_alpha;

        size_t ia = helper.SinglesFromAlphaBraIndex[j];
        size_t ja = helper.SinglesFromAlphaKetIndex[j];
        size_t ib = k + helper.braBetaStart;
        this->AddEnergy(helper, i, ia, ib, ja, ib);
    }
};

template <typename ElemT>
class MultDoubleAlpha : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    MultDoubleAlpha(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data, size_t o
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t k = (i + offset) / helper.size_double_alpha;
        size_t j = (i + offset) - k * helper.size_double_alpha;

        size_t ia = helper.DoublesFromAlphaBraIndex[j];
        size_t ja = helper.DoublesFromAlphaKetIndex[j];
        size_t ib = k + helper.braBetaStart;
        this->AddEnergy(helper, i, ia, ib, ja, ib);
    }
};

template <typename ElemT>
class MultSingleBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    MultSingleBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data, size_t o
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = (i + offset) / helper.size_single_beta;
        size_t k = (i + offset) - j * helper.size_single_beta;

        size_t ia = j + helper.braAlphaStart;
        size_t ib = helper.SinglesFromBetaBraIndex[k];
        size_t jb = helper.SinglesFromBetaKetIndex[k];
        this->AddEnergy(helper, i, ia, ib, ia, jb);
    }
};

template <typename ElemT>
class MultDoubleBeta : public MultKernelBase<ElemT>
{
protected:
    TaskHelpersThrust<ElemT> helper;
    size_t offset;
public:
    MultDoubleBeta(const TaskHelpersThrust<ElemT>& h,
                const thrust::device_vector<ElemT>& v_wb,
                const thrust::device_vector<ElemT>& v_t,
                const MultDataThrust<ElemT>& data, size_t o
                ) : MultKernelBase<ElemT>(v_wb, v_t, data)
    {
        helper = h;
        offset = o;
    }

    // kernel entry point
    __device__ __host__ void operator()(size_t i)
    {
        size_t j = (i + offset) / helper.size_double_beta;
        size_t k = (i + offset) - j * helper.size_double_beta;

        size_t ia = j + helper.braAlphaStart;
        size_t ib = helper.DoublesFromBetaBraIndex[k];
        size_t jb = helper.DoublesFromBetaKetIndex[k];
        this->AddEnergy(helper, i, ia, ib, ia, jb);
    }
};

// kernel for Wb initialization
template <typename ElemT>
class Wb_init_kernel {
protected:
    ElemT* Wb;
    ElemT* hii;
    ElemT* T;
public:
    Wb_init_kernel(thrust::device_vector<ElemT>& Wb_in, const thrust::device_vector<ElemT>& hii_in, const thrust::device_vector<ElemT>& T_in)
    {
        Wb = (ElemT*)thrust::raw_pointer_cast(Wb_in.data());
        hii = (ElemT*)thrust::raw_pointer_cast(hii_in.data());
        T = (ElemT*)thrust::raw_pointer_cast(T_in.data());
    }
    __host__ __device__ void operator()(size_t i)
    {
        Wb[i] += hii[i] * T[i];
    }
};

template <typename ElemT>
void mult(const std::vector<ElemT> &hii,
            const std::vector<ElemT> &Wk,
            std::vector<ElemT> &Wb,
            MultDataThrust<ElemT>& data,
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
            MultDataThrust<ElemT>& data,
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

    size_t adet_min = 0;
    size_t adet_max = data.adets.size();
    size_t bdet_min = 0;
    size_t bdet_max = data.bdets.size();
    get_mpi_range(adet_comm_size,0,adet_min,adet_max);
    get_mpi_range(bdet_comm_size,0,bdet_min,bdet_max);
    size_t max_det_size = (adet_max-adet_min)*(bdet_max-bdet_min);

    thrust::device_vector<ElemT> T(max_det_size);
    thrust::device_vector<ElemT> R;

    auto time_copy_start = std::chrono::high_resolution_clock::now();
    if (data.helper.size() != 0) {
        Mpi2dSlide(Wk, T, adet_comm_size, bdet_comm_size,
                    -data.helper[0].adetShift, -data.helper[0].bdetShift, b_comm);
    }
    auto time_copy_end = std::chrono::high_resolution_clock::now();

    auto time_mult_start = std::chrono::high_resolution_clock::now();

    if (mpi_rank_t == 0) {
        auto ci = thrust::counting_iterator<size_t>(0);
        thrust::for_each_n(thrust::device, ci, T.size(), Wb_init_kernel(Wb, hii, T));
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

        // precalculate DetI and DetJ (if update needed)
        bool update_I =  data.bra_adets_begin != data.helper[task].braAlphaStart || data.bra_bdets_begin != data.helper[task].braBetaStart ||
                        data.bra_adets_end != data.helper[task].braAlphaEnd || data.bra_bdets_end != data.helper[task].braBetaEnd;
        bool update_J =  data.ket_adets_begin != data.helper[task].ketAlphaStart || data.ket_bdets_begin != data.helper[task].ketBetaStart ||
                        data.ket_adets_end != data.helper[task].ketAlphaEnd || data.ket_bdets_end != data.helper[task].ketBetaEnd;
        if (update_I || update_J) {
            data.bra_adets_begin = data.helper[task].braAlphaStart;
            data.bra_bdets_begin = data.helper[task].braBetaStart;
            data.bra_adets_end = data.helper[task].braAlphaEnd;
            data.bra_bdets_end = data.helper[task].braBetaEnd;
            data.ket_adets_begin = data.helper[task].ketAlphaStart;
            data.ket_bdets_begin = data.helper[task].ketBetaStart;
            data.ket_adets_end = data.helper[task].ketAlphaEnd;
            data.ket_bdets_end = data.helper[task].ketBetaEnd;

            DetFromAlphaBetaKernel det_kernel(data.helper[task], data, update_I, update_J);
            auto det_ci = thrust::counting_iterator<size_t>(0);
            thrust::for_each_n(thrust::device, det_ci, data.adets_size * data.bdets_size, det_kernel);
        }

        size_t offset;
        size_t size;
        if (data.helper[task].taskType == 2) {
            offset = 0;
            size = data.helper[task].size_single_alpha * braBetaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                MultSingleAlpha single_kernel(data.helper[task], Wb, T, data, offset);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cis = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cis, num_threads, single_kernel);
                offset += num_threads;
            }

            offset = 0;
            size = data.helper[task].size_double_alpha * braBetaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                MultDoubleAlpha double_kernel(data.helper[task], Wb, T, data, offset);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cid = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cid, num_threads, double_kernel);
                offset += num_threads;
            }
        } else if(data.helper[task].taskType == 1) {
            offset = 0;
            size = data.helper[task].size_single_beta * braAlphaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                MultSingleBeta single_kernel(data.helper[task], Wb, T, data, offset);
                single_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cis = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cis, num_threads, single_kernel);
                offset += num_threads;
            }

            offset = 0;
            size = data.helper[task].size_double_beta * braAlphaSize;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                MultDoubleBeta double_kernel(data.helper[task], Wb, T, data, offset);
                double_kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto cid = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, cid, num_threads, double_kernel);
                offset += num_threads;
            }
        } else {
            offset = 0;
            size = data.helper[task].size_single_alpha * data.helper[task].size_single_beta;
            while (offset < size) {
                size_t num_threads = data.num_max_threads;
                if (offset + num_threads > size) {
                    num_threads = size - offset;
                }

                MultAlphaBeta kernel(data.helper[task], Wb, T, data, offset);
                kernel.set_mpi_size(mpi_rank_h, mpi_size_h);

                auto ci = thrust::counting_iterator<size_t>(0);
                thrust::for_each_n(thrust::device, ci, num_threads, kernel);
                offset += num_threads;
            }
        }

        if (data.helper[task].taskType == 0 && task != data.helper.size() - 1) {
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
            R.resize(T.size());
            R = T;
            auto time_slid_start = std::chrono::high_resolution_clock::now();
            Mpi2dSlide(R, T, adet_comm_size, bdet_comm_size, adetslide, bdetslide, b_comm);
            auto time_slid_end = std::chrono::high_resolution_clock::now();
            auto time_slid_count = std::chrono::duration_cast<std::chrono::microseconds>(time_slid_end - time_slid_start).count();
            time_slid += 1.0e-6 * time_slid_count;
            R.clear();
            R.shrink_to_fit();
        }

    } // end for(size_t task=0; task < data.helper.size(); task++)

    auto time_mult_end = std::chrono::high_resolution_clock::now();

    auto time_comm_start = std::chrono::high_resolution_clock::now();
    if (mpi_size_t > 1)
        MpiAllreduce(Wb, MPI_SUM, t_comm);
    if (mpi_size_h > 1)
        MpiAllreduce(Wb, MPI_SUM, h_comm);
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

    adets_size = 0;
    bdets_size = 0;
    bra_adets_begin = 0;
    bra_adets_end = 0;
    bra_bdets_begin = 0;
    bra_bdets_end = 0;
    ket_adets_begin = 0;
    ket_adets_end = 0;
    ket_bdets_begin = 0;
    ket_bdets_end = 0;
    bool bra_ket_same = true;

    // copyin helpers
    helper.clear();
    helper_storage.resize(helper_in.size());
    eij_storage.resize(helper_in.size());
    for (size_t task = 0; task < helper_in.size(); task++) {
        helper.push_back(TaskHelpersThrust<ElemT>(helper_storage[task], eij_storage[task], helper_in[task], method == 1));

        adets_size = std::max(adets_size, std::max(helper[task].braAlphaEnd - helper[task].braAlphaStart, helper[task].ketAlphaEnd - helper[task].ketAlphaStart));
        bdets_size = std::max(bdets_size, std::max(helper[task].braBetaEnd - helper[task].braBetaStart, helper[task].ketBetaEnd - helper[task].ketBetaStart));

        bra_ket_same &= helper[task].braAlphaStart == helper[task].ketAlphaStart;
        bra_ket_same &= helper[task].braAlphaEnd == helper[task].ketAlphaEnd;
        bra_ket_same &= helper[task].braBetaStart == helper[task].ketBetaStart;
        bra_ket_same &= helper[task].braBetaEnd == helper[task].ketBetaEnd;
    }

    // copyin adets, bdets
    adets.resize(size_D * adets_in.size());
    bdets.resize(size_D * bdets_in.size());
    for (int i = 0; i < adets_in.size(); i++) {
        thrust::copy(adets_in[i].begin(), adets_in[i].end(), adets.begin() + i * size_D);
    }
    for (int i = 0; i < bdets_in.size(); i++) {
        thrust::copy(bdets_in[i].begin(), bdets_in[i].end(), bdets.begin() + i * size_D);
    }

    size_t size_det = 0;
    for (size_t task = 0; task < helper.size(); task++) {
        size_t braAlphaSize = helper[task].braAlphaEnd - helper[task].braAlphaStart;
        size_t braBetaSize = helper[task].braBetaEnd - helper[task].braBetaStart;
        size_det = std::max(size_det, std::max(helper[task].size_single_alpha, helper[task].size_double_alpha) * braBetaSize);
        size_det = std::max(size_det, std::max(helper[task].size_single_beta, helper[task].size_double_beta) * braAlphaSize);
        size_det = std::max(size_det, helper[task].size_single_alpha * helper[task].size_single_beta);
    }
    num_max_threads = size_det;

    /*
    // number of max threads, this is enabled when per thread DetI and DetJ storage is used (non-pre calculate)
    if (size_det * size_D * 2 > MAX_DET_SIZE) {
        size_det = MAX_DET_SIZE / (size_D * 2);
        num_max_threads = size_det & (~1023ULL);
    }
    std::cout << " num_max_threads = " << num_max_threads << std::endl;
    */

    // allocate pre-calculated DetI, DetJ
    if (bra_ket_same)
        dets.resize(size_D * adets_size * bdets_size);
    else
        dets.resize(size_D * 2 * adets_size * bdets_size);

    // allocate per thread storage for DetI, DetJ
    // dets.resize(size_D * 2 * size_det);
}


}

#endif
