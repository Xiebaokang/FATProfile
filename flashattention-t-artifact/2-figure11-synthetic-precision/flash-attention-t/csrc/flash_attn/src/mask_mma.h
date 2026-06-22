/******************************************************************************
* Copyright (c) 2024, Tri Dao.
******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>
#include "custom_meta.cuh"
#include "custom_numerical_limits.h"
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

//// CUSTOM NOTE: here used to be a hell lot of unused junk code
//// CUSTOM NOTE: I just simply deleted all of them for clarity

struct MmaMaskChecker{

    /*
        static bypass:
        if seqlen_q is multiple of kBlockM and seqlen_k is multiple of kBlockN,
        then we can bypass the mask
        this condition is directly indicated by the Is_even_MN template parameter
        NOTE: the prefix KU means it's a kernel uniform function
    */
    template <bool Is_even_MN, bool Is_even_M>
    static CUTE_HOST_DEVICE constexpr bool KU_is_static_bypass()
    {
        return Is_even_MN || Is_even_M;
    }

    /*
        dynamic bypass:
        check if the current thread block handling the "m_block"
        NOTE: the prefix BU means it's a block uniform function

        TODO: might better inline the m_block into the function to better let
        the compiler know this can be optimized into the uniform datapath
    */
    template <int CTA_M>
    static CUTE_DEVICE bool BU_is_dynamic_bypass(
        int actual_seqlen_q // this will be binfo.actual_seqlen_q, same as the below "max_seqlen_q" parameter
    ){
        // thread at m_block would handle [k_BlockM * m_block, kBlockM * (m_block + 1)) range of the actual_seqlen_q
        // if the range is fully covered by the max_seqlen_q, then we can bypass the mask
        // otherwise, we need to apply the mask

        int m_block = blockIdx.x;

        int cur_m_block_seqlen_q_max = CTA_M * (m_block + 1);
        bool is_dynamic_bypass = cur_m_block_seqlen_q_max <= actual_seqlen_q;
        #if ENABLE_CUSTOM_DUMP_AND_PRINTF
        if (threadIdx.x == 0 ){
            printf("m_block = %d, cur_m_block_seqlen_q_max = %d, actual_seqlen_q = %d, is_dynamic_bypass = %d\n", m_block, cur_m_block_seqlen_q_max, actual_seqlen_q, is_dynamic_bypass);
        }
        __syncthreads();
        #endif
        return is_dynamic_bypass;
    }

    /*
        thread predicate for -INF filling
    */
    // CUTE_DEVICE constexpr bool TL_is_filling_needed(
    //     int apply_mask_row_idx
    // ){
    //     auto additional_row_idx_layout = make_layout(
    //         make_shape(_2{}, Int<MMA_M>{}),
    //         make_stride(_8{}, Int<16 * kNWarps>{})
    //     );
    // }
};

template <bool Is_even_MN, bool Is_even_M, class AccSTraits, class AccSEngine, class AccSLayout>
CUTE_DEVICE void apply_mma_mask(
    Tensor<AccSEngine, AccSLayout>& acc_s_tensor,
    int actual_seqlen_q,
    int apply_mask_row_idx
){
    constexpr bool is_static_bypass = MmaMaskChecker::KU_is_static_bypass<Is_even_MN, Is_even_M>();
    if constexpr (is_static_bypass) {
        return;
    }
    bool is_block_dynamic_bypass = MmaMaskChecker::BU_is_dynamic_bypass<AccSTraits::CTA_M>(
        actual_seqlen_q
    );
    if (is_block_dynamic_bypass) {
        return;
    }
    // acc_s_relayout has layout ((MMA_2M, MMA_M), (MMA_2N, MMA_N))
    Tensor acc_s_relayout = make_tensor(
        acc_s_tensor.data(),
        convert_layout_acc_rowcol(acc_s_tensor.layout())
    );

    auto additional_row_idx_layout = make_layout(
        make_shape(_2{}, Int<AccSTraits::Acc_S_MMA_M>{}),
        make_stride(_8{}, Int<16 * AccSTraits::NWarps>{})
    );

    constexpr int MMA_2M = decltype(cute::size<0,0>(acc_s_relayout))::value;
    static_assert(MMA_2M == AccSTraits::Acc_S_MMA_2M, "Unexpected acc_s shape");
    constexpr int MMA_M = decltype(cute::size<0,1>(acc_s_relayout))::value;
    static_assert(MMA_M == AccSTraits::Acc_S_MMA_M, "Unexpected acc_s shape");

    constexpr int MMA_2N = decltype(cute::size<1,0>(acc_s_relayout))::value;
    static_assert(MMA_2N == AccSTraits::Acc_S_MMA_2N, "Unexpected acc_s shape");
    constexpr int MMA_N = decltype(cute::size<1,1>(acc_s_relayout))::value;
    static_assert(MMA_N == AccSTraits::Acc_S_MMA_N, "Unexpected acc_s shape");

    static_assert(AccSTraits::Acc_S_MMA_N * 8 == AccSTraits::CTA_N);
    
    #pragma unroll
    for (int mma_2m_idx = 0; mma_2m_idx < MMA_2M; ++mma_2m_idx) {
        #pragma unroll
        for (int mma_m_idx = 0; mma_m_idx < MMA_M; ++mma_m_idx) {
            int additional_row_idx_offset = additional_row_idx_layout(
                mma_2m_idx, mma_m_idx
            );
            int row_idx = apply_mask_row_idx + additional_row_idx_offset;
            bool requires_filling = row_idx >= actual_seqlen_q;
            #pragma unroll
            for (int mma_2n_idx = 0; mma_2n_idx < MMA_2N; ++mma_2n_idx) {
                #pragma unroll
                for (int mma_n_idx = 0; mma_n_idx < MMA_N; ++mma_n_idx) {
                    acc_s_relayout(make_coord(mma_2m_idx, mma_m_idx), make_coord(mma_2n_idx, mma_n_idx)) =
                    requires_filling ? MASK_VALUE : acc_s_relayout(make_coord(mma_2m_idx, mma_m_idx), make_coord(mma_2n_idx, mma_n_idx));
                }
            }
        }
    }
    
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                              const int col_idx_offset_,
                                              const int row_idx_offset,
                                              const int warp_row_stride) {
        static_assert(!(Causal_mask && Is_local), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || Is_local || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Is_local && !Causal_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = MASK_VALUE; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Has_alibi) {
                                    if constexpr (Is_causal) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                    } else {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);

                                    }
                                }
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = MASK_VALUE;
                                    }
                                }
                                if constexpr (Is_local) {
                                    if (col_idx >= col_idx_limit_right || col_idx < col_idx_limit_left) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = MASK_VALUE;
                                    }
                                }
                                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                                    // Causal and Local already handles MN masking
                                    if (col_idx >= max_seqlen_k) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = MASK_VALUE;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

};

} // namespace FLASH_NAMESPACE
