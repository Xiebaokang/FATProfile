/******************************************************************************
* Copyright (c) 2024, Tri Dao.
******************************************************************************/

#pragma once

#include <cmath>

#include <cute/tensor.hpp>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "philox.cuh"
#include "utils.h"

#include "custom_meta.cuh"
#include "custom_numerical_limits.h"
#include "softmax_mma_acc_s_rescale.h"
#include "softmax_max.cuh"

namespace FLASH_NAMESPACE {

#define USE_STATIC_FOR_EACH 1

#if USE_STATIC_FOR_EACH
#define FOR_START(i, r) for_each(make_int_sequence<r>{}, [&](auto i){
#define FOR_END() });
#else
#define FOR_START(i, r) for(int i = 0; i < r; ++i){
#define FOR_END() }
#endif


  
using namespace cute;

//////////////// MMA CUSTOM
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
    #pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
  #pragma unroll
  for (int i = 0; i < size(dst); i++){
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  thread_reduce_<zero_init>(tensor, summary, op);
  quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
  MaxOp<float> max_op;
  reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
  SumOp<float> sum_op;
  thread_reduce_<zero_init>(tensor, sum, sum_op);
}

CUTE_DEVICE float warp_max_reduce_twopass_offset(
  float const& val,
  float const& offset
){
  constexpr uint32_t FULLMASK = 0xFFFFFFFF;
  // max(a+m, b+m) === max(a, b) + m
  // so we first add the offset to the val
  // this can prevent the compiler from demoting the retval_float from uniform reg to simt reg
  float val_offset = val + offset;
  int val_int = __float_as_int(val_offset);
  int max_val_int = __reduce_max_sync(FULLMASK, val_int);
  int min_val_int = __reduce_min_sync(FULLMASK, val_int);
  int retval = (max_val_int < 0 && min_val_int < 0) ? min_val_int : max_val_int;
  float retval_float = __int_as_float(retval);
  return retval_float;
}

// custom HMMA fragment B value generation algorithm for scale/axpby operantions
// IMPORTANT: PRE: frag_B_tensor should be already zero-initialized
template <class FrgBTensor>
__device__ __forceinline__ void gen_HMMA_scale_frag_B(
    FrgBTensor & frag_B_tensor, // (2,) tensor
    float const& value,
    int const& lane_id
){
    static_assert(decltype(cute::size(frag_B_tensor))::value == 2, "frag_B_tensor should have size of 2");
    
    int rem = lane_id % 9;
    int rem_augmented = (lane_id - 4) % 9;

    if (rem == 0) {
        frag_B_tensor(0) = value;
    }
    if (rem_augmented == 0) {
        frag_B_tensor(1) = value;
    }
}

// custom HMMA fragment B value generation algorithm for horizontal reduce operations
// IMPORTANT: PRE: frag_B_tensor should be already zero-initialized
template <class FrgBTensor>
CUTE_DEVICE void gen_HMMA_horizontal_reduce_frag_B(
    FrgBTensor & frag_B_tensor, // (2,) tensor
    int const& lane_id
){
    static_assert(decltype(cute::size(frag_B_tensor))::value == 2, "frag_B_tensor should have size of 2");
    bool is_3rd_bit_set = (lane_id & 4) != 0;

    if (is_3rd_bit_set) {
        frag_B_tensor(1) = 1.0f;
    } else {
        frag_B_tensor(0) = 1.0f;
    }
}


// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    // If we don't have float around M_LOG2E the multiplication is done in fp64.
    const float max_scaled = max(mi) == MASK_VALUE ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni)  {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)) This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      // The following macro will disable the use of fma.
      // See: https://github.com/pytorch/pytorch/issues/121558 for more details
      // This macro is set in PyTorch and not FlashAttention
      #ifdef UNFUSE_FMA
      tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
      #else
      /*
      if (softcap > 0.0) {
          params.softcap = softmax_scale / softcap;
          params.scale_softmax = softcap;
          params.scale_softmax_log2 = softcap * M_LOG2E;
      } else {
          // Remove potential NaN
          params.softcap = 0.0;
          params.scale_softmax = softmax_scale;
          params.scale_softmax_log2 = softmax_scale * M_LOG2E;
      }
      when softcap is not set, params.scale_softmax = scale will always be < 1  
      */
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
      #endif
    }
  }
}


template <class AccSTraits, class AccOTraits, bool Is_even_MN, bool Is_even_M>
struct MmaSoftmax {
  
  using THIS_CLASS = MmaSoftmax<AccSTraits, AccOTraits, Is_even_MN, Is_even_M>;

  static constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
  static constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
  static constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
  static constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
  static constexpr int warp_size = 32;

  static constexpr int kNRows = 2 * MMA_M; // length of the row max / sum tensor, also the number of rows in the softmax tensor the thread handles

  static constexpr int kBlockM = AccSTraits::CTA_M;
  static constexpr int kBlockN = AccSTraits::CTA_N;
  static constexpr int kNWarps = AccSTraits::NWarps;

  using RowTensorIdxLayout = decltype(make_layout(
    make_shape(Int<MMA_2M>{}, Int<MMA_M>{}),
    GenColMajor{}
  ));
  using RowTensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  RowTensorT row_max, row_sum;

  using MmaMaxTensorT = decltype(make_tensor<float>(Shape<Int<MMA_M>>{}));
  MmaMaxTensorT mma_max_uniform;

  const float warpmax_offset;
  const int actual_seqlen_q;
  const int lane_id;

  using HMMA1688FragBT = decltype(make_tensor<float>(Shape<Int<2>>{}));
  HMMA1688FragBT softmax_scale_log2_frag_B, softmax_reduce_add_frag_B;

  // partial row sum: (MMA_n = 2, MMA_m = 2, MMA_M)
  using MmaPartialSumT = decltype(make_tensor<float>(Shape<Int<2>, Int<2>, Int<MMA_M>>{}));
  MmaPartialSumT mma_partial_row_sum;
  
  CUTE_DEVICE MmaSoftmax(int actual_seqlen_q, float scale_softmax_log2, int tidx, float warpmax_offset = 0.f)
  : actual_seqlen_q(actual_seqlen_q)
  , warpmax_offset(warpmax_offset)
  , lane_id(tidx % THIS_CLASS::warp_size)
  {
    cute::fill(softmax_scale_log2_frag_B, 0.f);
    cute::fill(softmax_reduce_add_frag_B, 0.f);
    gen_HMMA_scale_frag_B(
      softmax_scale_log2_frag_B,
      scale_softmax_log2,
      lane_id
    );
    gen_HMMA_horizontal_reduce_frag_B(
      softmax_reduce_add_frag_B,
      lane_id
    );
  }

  template <
    bool zero_init=true,
    typename EngineTensor, typename LayoutTensor,
    typename EngineRowMax, typename LayoutRowMax,
    typename EngineMmaMax, typename LayoutMmaMax
    >
  CUTE_DEVICE void reduce_mma_max_offset_pred(
    Tensor<EngineTensor, LayoutTensor> const &tensor,
    Tensor<EngineRowMax, LayoutRowMax> &row_max_tensor,
    Tensor<EngineMmaMax, LayoutMmaMax> &mma_max_tensor,
    float const& offset,
    int const& apply_mask_row_idx_offset
  ){
    MaxOp<float> max_op;
    #if USE_DEFAULT_MAX && !USE_BINARY_TREE_MAX
    thread_reduce_<zero_init>(tensor, row_max_tensor, max_op);
    #elif !USE_DEFAULT_MAX && USE_BINARY_TREE_MAX 
    constexpr int reduce_max_fmaxf_ratio = 8;
    reduce_max_binary_max<zero_init, reduce_max_fmaxf_ratio>(tensor, row_max_tensor);
    #else
    #error "Please set exactly one of USE_DEFAULT_MAX or USE_BINARY_TREE_MAX to true"
    #endif
    auto max_tensor_idx_layout = RowTensorIdxLayout{};
    auto additional_row_idx_layout = make_layout(
        make_shape(Int<MMA_2M>{}, Int<MMA_M>{}),
        make_stride(_8{}, Int<16 * kNWarps>{})
    );

    FOR_START(mma_m_idx, MMA_M)
      if constexpr (Is_even_MN || Is_even_M) {
        // seqlen_q is multiple of kBlockM
        // no need to apply mask
        float local_max = max_op(
          row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx)),
          row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx))
        );
        float warp_max = warp_max_reduce_twopass_offset(
          local_max,
          offset
        );
        mma_max_tensor(mma_m_idx) = warp_max;
        row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) = warp_max;
        row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) = warp_max;
      } else {
        // generic case where seqlen_q is not multiple of kBlockM
        // need to apply mask
        int additional_row_idx_offset_0 = additional_row_idx_layout(
          Int<0>{}, mma_m_idx
        );
        int row_idx_0 = apply_mask_row_idx_offset + additional_row_idx_offset_0;
        bool row_idx_0_requires_masking = row_idx_0 >= actual_seqlen_q;
        float row_idx_0_val = row_idx_0_requires_masking ?
          MASK_VALUE : row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx));
        int additional_row_idx_offset_1 = additional_row_idx_layout(
          Int<1>{}, mma_m_idx
        );
        int row_idx_1 = apply_mask_row_idx_offset + additional_row_idx_offset_1;
        bool row_idx_1_requires_masking = row_idx_1 >= actual_seqlen_q;
        float row_idx_1_val = row_idx_1_requires_masking ?
          MASK_VALUE : row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx));
        float local_max = max_op(row_idx_0_val, row_idx_1_val);
        float warp_max = warp_max_reduce_twopass_offset(
          local_max,
          offset
        );
        mma_max_tensor(mma_m_idx) = warp_max;
        row_max_tensor(max_tensor_idx_layout(Int<0>{}, mma_m_idx)) = warp_max;
        row_max_tensor(max_tensor_idx_layout(Int<1>{}, mma_m_idx)) = warp_max;
      }
    FOR_END()

    }
    
  
  template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_o(
    Tensor0 &acc_s,
    Tensor1 &acc_o,
    float softmax_scale_log2,
    int apply_mask_row_idx_offset
  ){
    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    
    #define DEBUG_ASSERT_MMA4_LAYOUT 0
    #if DEBUG_ASSERT_MMA4_LAYOUT
    auto mma_4_layout = AccSTraits::get_mma4_layout();
    for (int i = 0; i<MMA_2M; i++){
      for (int j = 0; j<MMA_2N; j++){
        int mma_4_idx = mma_4_layout(i,j);
        float scores_val = scores(make_coord(i,0), make_coord(j,0));
        float acc_s_val = acc_s(mma_4_idx, 0, 0);
        assert(scores_val == acc_s_val);
        
      }
    }
    #endif
    
    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) {
      reduce_mma_max_offset_pred</*zero_init=*/true>(
        scores, row_max, mma_max_uniform, warpmax_offset, apply_mask_row_idx_offset
      );
      // TODO(rbxu): test the below three methods performance
      // TODO(rbxu): also note that the horizontal ilp ratio needs a scan
      #if LOOP1_USE_ACCS_SCALE_MMA_ONLY && !LOOP1_USE_ACCS_SCALE_SIMT_ONLY && !LOOP1_USE_ACCS_SCALE_ILP_VERT && !LOOP1_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ONLY;
      #elif LOOP1_USE_ACCS_SCALE_SIMT_ONLY && !LOOP1_USE_ACCS_SCALE_MMA_ONLY && !LOOP1_USE_ACCS_SCALE_ILP_VERT && !LOOP1_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::SIMT_ONLY;
      #elif LOOP1_USE_ACCS_SCALE_ILP_VERT && !LOOP1_USE_ACCS_SCALE_MMA_ONLY && !LOOP1_USE_ACCS_SCALE_SIMT_ONLY && !LOOP1_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ILP_VERT;
      #elif LOOP1_USE_ACCS_SCALE_ILP_HORI && !LOOP1_USE_ACCS_SCALE_MMA_ONLY && !LOOP1_USE_ACCS_SCALE_SIMT_ONLY && !LOOP1_USE_ACCS_SCALE_ILP_VERT
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ILP_HORI;
      #else
      #error "Please set exactly one of LOOP1_USE_ACCS_SCALE_MMA_ONLY, LOOP1_USE_ACCS_SCALE_SIMT_ONLY, LOOP1_USE_ACCS_SCALE_ILP_VERT, LOOP1_USE_ACCS_SCALE_ILP_HORI to true"
      #endif
      constexpr bool simt_first = false;
      #if !defined(LOOP1_ACCS_ILP_HORI_RATIO)
      #error "LOOP1_ACCS_ILP_HORI_RATIO is not defined"
      #endif
      constexpr int accSScale_tensor_ratio = LOOP1_ACCS_ILP_HORI_RATIO;
      FLASH_NAMESPACE::AccSScaleExp2<accSScaleExp2Mode>::scale_apply_exp2<accSScale_tensor_ratio, simt_first, AccSTraits, RowTensorIdxLayout>(
        acc_s, mma_max_uniform, softmax_scale_log2, softmax_scale_log2_frag_B
      );
      // FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      FLASH_NAMESPACE::reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      reduce_mma_max_offset_pred</*zero_init=*/false>(
        scores, row_max, mma_max_uniform, warpmax_offset, apply_mask_row_idx_offset
      );
      // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
      Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
      static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
      #pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = !Check_inf
        ? row_max(mi)
        : (row_max(mi) == MASK_VALUE ? 0.0f : row_max(mi));
        float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
      }
      // #endif
      // TODO(rbxu): similar to the above, test the below three methods performance
      // TODO(rbxu): also note that the horizontal ilp ratio needs a scan
      // TODO(rbxu): NOTE the configuration here CAN be different than the above
      #if LOOP2_USE_ACCS_SCALE_MMA_ONLY && !LOOP2_USE_ACCS_SCALE_SIMT_ONLY && !LOOP2_USE_ACCS_SCALE_ILP_VERT && !LOOP2_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ONLY;
      #elif LOOP2_USE_ACCS_SCALE_SIMT_ONLY && !LOOP2_USE_ACCS_SCALE_MMA_ONLY && !LOOP2_USE_ACCS_SCALE_ILP_VERT && !LOOP2_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::SIMT_ONLY;
      #elif LOOP2_USE_ACCS_SCALE_ILP_VERT && !LOOP2_USE_ACCS_SCALE_MMA_ONLY && !LOOP2_USE_ACCS_SCALE_SIMT_ONLY && !LOOP2_USE_ACCS_SCALE_ILP_HORI
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ILP_VERT;
      #elif LOOP2_USE_ACCS_SCALE_ILP_HORI && !LOOP2_USE_ACCS_SCALE_MMA_ONLY && !LOOP2_USE_ACCS_SCALE_SIMT_ONLY && !LOOP2_USE_ACCS_SCALE_ILP_VERT
      constexpr AccSScaleExp2Mode accSScaleExp2Mode = AccSScaleExp2Mode::MMA_ILP_HORI;
      #else
      #error "Please set exactly one of LOOP2_USE_ACCS_SCALE_MMA_ONLY, LOOP2_USE_ACCS_SCALE_SIMT_ONLY, LOOP2_USE_ACCS_SCALE_ILP_VERT, LOOP2_USE_ACCS_SCALE_ILP_HORI to true"
      #endif

      constexpr bool simt_first = false;
      #if !defined(LOOP2_ACCS_ILP_HORI_RATIO)
      #error "LOOP2_ACCS_ILP_HORI_RATIO is not defined"
      #endif
      constexpr int accSScale_tensor_ratio = LOOP2_ACCS_ILP_HORI_RATIO;
      FLASH_NAMESPACE::AccSScaleExp2<accSScaleExp2Mode>::scale_apply_exp2<accSScale_tensor_ratio, simt_first, AccSTraits, RowTensorIdxLayout>(
        acc_s, mma_max_uniform, softmax_scale_log2, softmax_scale_log2_frag_B
      );
      // FLASH_NAMESPACE::scale_apply_exp2(scores, row_max, softmax_scale_log2);
      // We don't do the reduce across threads here since we don't need to use the row_sum.
      // We do that reduce at the end when we need to normalize the softmax.
      FLASH_NAMESPACE::reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
  }
  
  template<bool Is_dropout=false, bool Split=false, typename Tensor0>
  __forceinline__ __device__ RowTensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
    SumOp<float> sum_op;
    orig_quad_allreduce_(row_sum, row_sum, sum_op);
    RowTensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
      lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
      float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
      #pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }
    return lse;
  };

  
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); mi++) {
    summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
    #pragma unroll
    for (int ni = 1; ni < size<1>(tensor); ni++) {
      summary(mi) = op(summary(mi), tensor(mi, ni));
    }
  }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
  CUTE_STATIC_ASSERT_V(size(dst) == size(src));
  #pragma unroll
  for (int i = 0; i < size(dst); i++){
    dst(i) = Allreduce<4>::run(src(i), op);
  }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void orig_reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
  orig_thread_reduce_<zero_init>(tensor, summary, op);
  orig_quad_allreduce_(summary, summary, op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void orig_reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
  MaxOp<float> max_op;
  orig_reduce_<zero_init>(tensor, max, max_op);
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ __forceinline__ void orig_reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
  SumOp<float> sum_op;
  orig_thread_reduce_<zero_init>(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void orig_scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 1, "Only support 1D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    // If max is -inf, then all elements must have been -inf (possibly due to masking).
    // We don't want (-inf - (-inf)) since that would give NaN.
    // If we don't have float around M_LOG2E the multiplication is done in fp64.
    const float max_scaled = max(mi) == MASK_VALUE ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ++ni)  {
      // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
      // max * log_2(e)) This allows the compiler to use the ffma
      // instruction instead of fadd and fmul separately.
      // The following macro will disable the use of fma.
      // See: https://github.com/pytorch/pytorch/issues/121558 for more details
      // This macro is set in PyTorch and not FlashAttention
      #ifdef UNFUSE_FMA
      tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
      #else
      tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
      #endif
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNRows>
struct Softmax {
  
  using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
  TensorT row_max, row_sum;
  
  __forceinline__ __device__ Softmax() {};
  
  template<bool Is_first, bool Check_inf=false, typename Tensor0, typename Tensor1>
  __forceinline__ __device__ void softmax_rescale_o(Tensor0 &acc_s, Tensor1 &acc_o, float softmax_scale_log2) {
    // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
    static_assert(decltype(size<0>(scores))::value == kNRows);
    if (Is_first) {
      FLASH_NAMESPACE::template orig_reduce_max</*zero_init=*/true>(scores, row_max);
      FLASH_NAMESPACE::orig_scale_apply_exp2(scores, row_max, softmax_scale_log2);
      FLASH_NAMESPACE::orig_reduce_sum</*zero_init=*/true>(scores, row_sum);
    } else {
      Tensor scores_max_prev = make_fragment_like(row_max);
      cute::copy(row_max, scores_max_prev);
      FLASH_NAMESPACE::template orig_reduce_max</*zero_init=*/false>(scores, row_max);
      // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
      Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
      static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
      #pragma unroll
      for (int mi = 0; mi < size(row_max); ++mi) {
        float scores_max_cur = !Check_inf
        ? row_max(mi)
        : (row_max(mi) == MASK_VALUE ? 0.0f : row_max(mi));
        float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
        row_sum(mi) *= scores_scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; }
      }
      FLASH_NAMESPACE::orig_scale_apply_exp2(scores, row_max, softmax_scale_log2);
      // We don't do the reduce across threads here since we don't need to use the row_sum.
      // We do that reduce at the end when we need to normalize the softmax.
      FLASH_NAMESPACE::orig_reduce_sum</*zero_init=*/false>(scores, row_sum);
    }
  };
  
  template<bool Is_dropout=false, bool Split=false, typename Tensor0>
  __forceinline__ __device__ TensorT normalize_softmax_lse(Tensor0 &acc_o, float softmax_scale, float rp_dropout=1.0) {
    SumOp<float> sum_op;
    orig_quad_allreduce_(row_sum, row_sum, sum_op);
    TensorT lse = make_fragment_like(row_sum);
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_o.layout()));
    static_assert(decltype(size<0>(acc_o_rowcol))::value == kNRows);
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
      float sum = row_sum(mi);
      float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
      lse(mi) = (sum == 0.f || sum != sum) ? (Split ? -INFINITY : INFINITY) : row_max(mi) * softmax_scale + __logf(sum);
      float scale = !Is_dropout ? inv_sum : inv_sum * rp_dropout;
      #pragma unroll
      for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scale; }
    }
    return lse;
  };
};

#undef USE_STATIC_FOR_EACH
#undef FOR_START
#undef FOR_END
  
}  // namespace FLASH_NAMESPACE
