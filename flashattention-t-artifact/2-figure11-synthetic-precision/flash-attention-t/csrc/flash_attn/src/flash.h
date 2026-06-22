/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};


__host__ inline void print_flash_fwd_params(Flash_fwd_params const& params)
{
    printf("----------- Printing Flash_fwd_params Info BEGIN -----------\n");
    printf("-------------- Subclass Qkv_params Info BEGIN --------------\n");
    printf("q_ptr: %p\n", params.q_ptr);
    printf("k_ptr: %p\n", params.k_ptr);
    printf("v_ptr: %p\n", params.v_ptr);
    printf("q_batch_stride: %ld\n", params.q_batch_stride);
    printf("k_batch_stride: %ld\n", params.k_batch_stride);
    printf("v_batch_stride: %ld\n", params.v_batch_stride);
    printf("q_row_stride: %ld\n", params.q_row_stride);
    printf("k_row_stride: %ld\n", params.k_row_stride);
    printf("v_row_stride: %ld\n", params.v_row_stride);
    printf("q_head_stride: %ld\n", params.q_head_stride);
    printf("k_head_stride: %ld\n", params.k_head_stride);
    printf("v_head_stride: %ld\n", params.v_head_stride);
    printf("h (number of heads): %d\n", params.h);
    printf("h_k (number of key heads): %d\n", params.h_k);
    printf("h_h_k_ratio (precompute h / h_k): %d\n", params.h_h_k_ratio);
    printf("-------------- Subclass Qkv_params Info END --------------\n");
    printf("o_ptr: %p\n", params.o_ptr);
    printf("oaccum_ptr: %p\n", params.oaccum_ptr);
    printf("o_batch_stride: %ld\n", params.o_batch_stride);
    printf("o_row_stride: %ld\n", params.o_row_stride);
    printf("o_head_stride: %ld\n", params.o_head_stride);
    printf("p_ptr: %p\n", params.p_ptr);
    printf("softmax_lse_ptr: %p\n", params.softmax_lse_ptr);
    printf("softmax_lseaccum_ptr: %p\n", params.softmax_lseaccum_ptr);
    printf("b: %d\n", params.b);
    printf("seqlen_q: %d\n", params.seqlen_q);
    printf("seqlen_k: %d\n", params.seqlen_k);
    printf("seqlen_knew: %d\n", params.seqlen_knew);
    printf("d: %d\n", params.d);
    printf("seqlen_q_rounded: %d\n", params.seqlen_q_rounded);
    printf("seqlen_k_rounded: %d\n", params.seqlen_k_rounded);
    printf("d_rounded: %d\n", params.d_rounded);
    printf("rotary_dim: %d\n", params.rotary_dim);
    printf("total_q: %d\n", params.total_q);
    printf("scale_softmax: %f\n", params.scale_softmax);
    printf("scale_softmax_log2: %f\n", params.scale_softmax_log2);
    printf("cu_seqlens_q: %p\n", params.cu_seqlens_q);
    printf("cu_seqlens_k: %p\n", params.cu_seqlens_k);
    printf("leftpad_k: %p\n", params.leftpad_k);
    printf("seqused_k: %p\n", params.seqused_k);
    printf("blockmask: %p\n", params.blockmask);
    printf("knew_ptr: %p\n", params.knew_ptr);
    printf("vnew_ptr: %p\n", params.vnew_ptr);
    printf("knew_batch_stride: %ld\n", params.knew_batch_stride);
    printf("vnew_batch_stride: %ld\n", params.vnew_batch_stride);
    printf("knew_row_stride: %ld\n", params.knew_row_stride);
    printf("vnew_row_stride: %ld\n", params.vnew_row_stride);
    printf("knew_head_stride: %ld\n", params.knew_head_stride);
    printf("vnew_head_stride: %ld\n", params.vnew_head_stride);
    printf("rotary_cos_ptr: %p\n", params.rotary_cos_ptr);
    printf("rotary_sin_ptr: %p\n", params.rotary_sin_ptr);
    printf("cache_batch_idx: %p\n", params.cache_batch_idx);
    printf("block_table: %p\n", params.block_table);
    printf("block_table_batch_stride: %ld\n", params.block_table_batch_stride);
    printf("page_block_size: %d\n", params.page_block_size);
    printf("p_dropout: %f\n", params.p_dropout);
    printf("rp_dropout: %f\n", params.rp_dropout);
    printf("scale_softmax_rp_dropout: %f\n", params.scale_softmax_rp_dropout);
    printf("window_size_left: %d\n", params.window_size_left);
    printf("window_size_right: %d\n", params.window_size_right);
    printf("softcap: %f\n", params.softcap);
    printf("philox_args: %p\n", &params.philox_args);
    printf("rng_state: %p\n", params.rng_state);
    printf("is_bf16: %d\n", params.is_bf16);
    printf("is_causal: %d\n", params.is_causal);
    printf("is_seqlens_k_cumulative: %d\n", params.is_seqlens_k_cumulative);
    printf("is_rotary_interleaved: %d\n", params.is_rotary_interleaved);
    printf("num_splits: %d\n", params.num_splits);
    printf("alibi_slopes_ptr: %p\n", params.alibi_slopes_ptr);
    printf("alibi_slopes_batch_stride: %ld\n", params.alibi_slopes_batch_stride);
    printf("unpadded_lse: %d\n", params.unpadded_lse);
    printf("seqlenq_ngroups_swapped: %d\n", params.seqlenq_ngroups_swapped);
    printf("----------- Printing Flash_fwd_params Info END -----------\n");

}


////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
