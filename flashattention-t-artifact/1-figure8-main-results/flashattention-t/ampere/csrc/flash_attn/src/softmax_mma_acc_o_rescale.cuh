#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "utils.h"

#include "custom_meta.cuh"
#include "custom_numerical_limits.h"
#include "softmax_exp.cuh"

namespace FLASH_NAMESPACE {


template <class EngineRowSumOrPartialMmaSum, class LayoutRowSumOrPartialMmaSum>
CUTE_DEVICE constexpr bool is_partial_row_sum(Tensor<EngineRowSumOrPartialMmaSum, LayoutRowSumOrPartialMmaSum> const &tensor) {
  constexpr int layout_rank = LayoutRowSumOrPartialMmaSum::rank;
  return layout_rank == 3;
}

enum class AccOScaleMode {
  MMA_ILP_HORI,
  MMA_ILP_VERT,
  MMA_ONLY,
  SIMT_ONLY
};

template <AccOScaleMode mode>
struct AccOScale;

template <>
struct AccOScale<AccOScaleMode::MMA_ONLY> {
  template <bool Check_inf, int tensor_ratio, class HMMA1688FragBT, class RowTensorIdxLayout, class AccSTraits, class AccOTraits,
    class EngineMmaMax, class LayoutMmaMax,
    class EngineMmaMaxPrev, class LayoutMmaMaxPrev,
    class EngineRowSumOrPartialMmaSum, class LayoutRowSumOrPartialMmaSum,
    class EngineAccO, class LayoutAccO,
    class HMMA1688FragBMaskT
  >
  static CUTE_DEVICE void rescale_acc_o(
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max,
    Tensor<EngineMmaMaxPrev, LayoutMmaMaxPrev> const& mma_max_prev,
    Tensor<EngineRowSumOrPartialMmaSum, LayoutRowSumOrPartialMmaSum> & row_sum,
    Tensor<EngineAccO, LayoutAccO> & acc_o, // expect acc_o in (4, MMA_M, MMA_K) format
    HMMA1688FragBMaskT const& frag_b_mask,
    float const& softmax_scale_log2
  ){
    static_assert(size<0>(LayoutAccO{}) == 4, "Unexpected Acc_O tensor size");
    static_assert(size<1>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_M, "Unexpected Acc_O tensor size");
    static_assert(size<2>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_K, "Unexpected Acc_O tensor size");

    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    auto row_sum_tensor_idx_layout = RowTensorIdxLayout{};
    // partial row sum layout : (MMA_n = 2, MMA_m = 2, MMA_M)
    constexpr bool is_partial = (LayoutRowSumOrPartialMmaSum::rank == 3);

    auto mma_frag_RZ = cute::make_tensor<float>(
      cute::make_layout(
        cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
        cute::GenRowMajor{}
      )
    );
    cute::fill(mma_frag_RZ, 0.f);

    auto mma_frag_B = HMMA1688FragBT{};

    CUTE_UNROLL
    for (int mma_m_idx = 0; mma_m_idx < AccOTraits::Acc_O_MMA_M; ++mma_m_idx){
      float scores_max_cur = !Check_inf
      ? mma_max(mma_m_idx)
      : (mma_max(mma_m_idx) == MASK_VALUE ? 0.0f : mma_max(mma_m_idx));
      float scores_scale = exp2f((mma_max_prev(mma_m_idx) - scores_max_cur) * softmax_scale_log2);

      if constexpr (is_partial) {
        row_sum(Int<0>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<0>{}, Int<1>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<1>{}, mma_m_idx) *= scores_scale;
      } else {
        row_sum(row_sum_tensor_idx_layout(Int<0>{}, mma_m_idx)) *= scores_scale;
        row_sum(row_sum_tensor_idx_layout(Int<1>{}, mma_m_idx)) *= scores_scale;
      }

      mma_frag_B(Int<0>{}) = scores_scale * frag_b_mask(Int<0>{});
      mma_frag_B(Int<1>{}) = scores_scale * frag_b_mask(Int<1>{});

      CUTE_UNROLL
      for (int mma_k_idx = 0; mma_k_idx < AccOTraits::Acc_O_MMA_K; ++mma_k_idx){
        mma_unpack(
          MmaTraits{},
          acc_o(_, mma_m_idx, mma_k_idx),
          acc_o(_, mma_m_idx, mma_k_idx),
          mma_frag_B,
          mma_frag_RZ
        );
        float temp = acc_o(Int<1>{}, mma_m_idx, mma_k_idx);
        acc_o(Int<1>{}, mma_m_idx, mma_k_idx) = __fadd_rz(acc_o(Int<2>{}, mma_m_idx, mma_k_idx), 0.f);
        acc_o(Int<2>{}, mma_m_idx, mma_k_idx) = __fadd_rz(temp, 0.f);
      }
    }
    
  }
};


template <>
struct AccOScale<AccOScaleMode::MMA_ILP_HORI> {
  template <bool Check_inf, int tensor_ratio, class HMMA1688FragBT, class RowTensorIdxLayout, class AccSTraits, class AccOTraits,
    class EngineMmaMax, class LayoutMmaMax,
    class EngineMmaMaxPrev, class LayoutMmaMaxPrev,
    class EngineRowSumOrPartialMmaSum, class LayoutRowSumOrPartialMmaSum,
    class EngineAccO, class LayoutAccO,
    class HMMA1688FragBMaskT
  >
  static CUTE_DEVICE void rescale_acc_o(
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max,
    Tensor<EngineMmaMaxPrev, LayoutMmaMaxPrev> const& mma_max_prev,
    Tensor<EngineRowSumOrPartialMmaSum, LayoutRowSumOrPartialMmaSum> & row_sum,
    Tensor<EngineAccO, LayoutAccO> & acc_o, // expect acc_o in (4, MMA_M, MMA_K) format
    HMMA1688FragBMaskT const& frag_b_mask,
    float const& softmax_scale_log2
  ){
    static_assert(size<0>(LayoutAccO{}) == 4, "Unexpected Acc_O tensor size");
    static_assert(size<1>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_M, "Unexpected Acc_O tensor size");
    static_assert(size<2>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_K, "Unexpected Acc_O tensor size");

    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    auto row_sum_tensor_idx_layout = RowTensorIdxLayout{};
    // partial row sum layout : (MMA_n = 2, MMA_m = 2, MMA_M)
    constexpr bool is_partial = (LayoutRowSumOrPartialMmaSum::rank == 3);

    auto mma_frag_RZ = cute::make_tensor<float>(
      cute::make_layout(
        cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
        cute::GenRowMajor{}
      )
    );
    cute::fill(mma_frag_RZ, 0.f);

    auto mma_frag_B = HMMA1688FragBT{};

    // CUTE_UNROLL
    // for (int mma_m_idx = 0; mma_m_idx < AccOTraits::Acc_O_MMA_M; ++mma_m_idx){
    for_each(make_int_sequence<AccOTraits::Acc_O_MMA_M>{}, [&](auto mma_m_idx){
      float scores_max_cur = !Check_inf
      ? mma_max(mma_m_idx)
      : (mma_max(mma_m_idx) == MASK_VALUE ? 0.0f : mma_max(mma_m_idx));
      float scores_scale = exp2f((mma_max_prev(mma_m_idx) - scores_max_cur) * softmax_scale_log2);
      
      if constexpr (is_partial) {
        row_sum(Int<0>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<0>{}, Int<1>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<1>{}, mma_m_idx) *= scores_scale;
      } else {
        row_sum(row_sum_tensor_idx_layout(Int<0>{}, mma_m_idx)) *= scores_scale;
        row_sum(row_sum_tensor_idx_layout(Int<1>{}, mma_m_idx)) *= scores_scale;
      }

      mma_frag_B(Int<0>{}) = scores_scale * frag_b_mask(Int<0>{});
      mma_frag_B(Int<1>{}) = scores_scale * frag_b_mask(Int<1>{});

      // CUTE_UNROLL
      // for (int mma_k_idx = 0; mma_k_idx < AccOTraits::Acc_O_MMA_K; ++mma_k_idx){
      for_each(make_int_sequence<AccOTraits::Acc_O_MMA_K>{}, [&](auto mma_k_idx){
        if constexpr (mma_k_idx < tensor_ratio) {
          // tensor part
          mma_unpack(
            MmaTraits{},
            acc_o(_, mma_m_idx, mma_k_idx),
            acc_o(_, mma_m_idx, mma_k_idx),
            mma_frag_B,
            mma_frag_RZ
          );
          float temp = acc_o(Int<1>{}, mma_m_idx, mma_k_idx);
          acc_o(Int<1>{}, mma_m_idx, mma_k_idx) = __fadd_rz(acc_o(Int<2>{}, mma_m_idx, mma_k_idx), 0.f);
          acc_o(Int<2>{}, mma_m_idx, mma_k_idx) = __fadd_rz(temp, 0.f);
        } else {
          // simt part
          acc_o(Int<0>{}, mma_m_idx, mma_k_idx) *= scores_scale;
          acc_o(Int<1>{}, mma_m_idx, mma_k_idx) *= scores_scale;
          acc_o(Int<2>{}, mma_m_idx, mma_k_idx) *= scores_scale;
          acc_o(Int<3>{}, mma_m_idx, mma_k_idx) *= scores_scale;
        }
        
      });
    });
    
  }
};

template <>
struct AccOScale<AccOScaleMode::MMA_ILP_VERT> {
  template <bool Check_inf, int tensor_ratio, class HMMA1688FragBT, class RowTensorIdxLayout, class AccSTraits, class AccOTraits,
    class EngineMmaMax, class LayoutMmaMax,
    class EngineMmaMaxPrev, class LayoutMmaMaxPrev,
    class EngineRowSumOrPartialMmaSum, class LayoutRowSumOrPartialMmaSum,
    class EngineAccO, class LayoutAccO,
    class HMMA1688FragBMaskT
  >
  static CUTE_DEVICE void rescale_acc_o(
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max,
    Tensor<EngineMmaMaxPrev, LayoutMmaMaxPrev> const& mma_max_prev,
    Tensor<EngineRowSumOrPartialMmaSum, LayoutRowSumOrPartialMmaSum> & row_sum,
    Tensor<EngineAccO, LayoutAccO> & acc_o, // expect acc_o in (4, MMA_M, MMA_K) format
    HMMA1688FragBMaskT const& frag_b_mask,
    float const& softmax_scale_log2
  ){
    static_assert(size<0>(LayoutAccO{}) == 4, "Unexpected Acc_O tensor size");
    static_assert(size<1>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_M, "Unexpected Acc_O tensor size");
    static_assert(size<2>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_K, "Unexpected Acc_O tensor size");
    static_assert(AccOTraits::Acc_O_MMA_M != 1, "Vertical ILP not possible with MMA_M = 1 (CTA_M = 64)");

    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    auto row_sum_tensor_idx_layout = RowTensorIdxLayout{};
    // partial row sum layout : (MMA_n = 2, MMA_m = 2, MMA_M)
    constexpr bool is_partial = (LayoutRowSumOrPartialMmaSum::rank == 3);

    auto mma_frag_RZ = cute::make_tensor<float>(
      cute::make_layout(
        cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
        cute::GenRowMajor{}
      )
    );
    cute::fill(mma_frag_RZ, 0.f);

    auto mma_frag_B = HMMA1688FragBT{};

    auto mma_m_idx_tensor = Int<0>{};
    auto mma_m_idx_simt = Int<1>{};

    float scores_max_cur_simt = !Check_inf
      ? mma_max(mma_m_idx_simt)
      : (mma_max(mma_m_idx_simt) == MASK_VALUE ? 0.0f : mma_max(mma_m_idx_simt));
    float scores_scale_simt = exp2f((mma_max_prev(mma_m_idx_simt) - scores_max_cur_simt) * softmax_scale_log2);
    float scores_max_cur_tensor = !Check_inf
      ? mma_max(mma_m_idx_tensor)
      : (mma_max(mma_m_idx_tensor) == MASK_VALUE ? 0.0f : mma_max(mma_m_idx_tensor));
    float scores_scale_tensor = exp2f((mma_max_prev(mma_m_idx_tensor) - scores_max_cur_tensor) * softmax_scale_log2);
    
    if constexpr (is_partial) {
      row_sum(Int<0>{}, Int<0>{}, mma_m_idx_tensor) *= scores_scale_tensor;
      row_sum(Int<1>{}, Int<0>{}, mma_m_idx_tensor) *= scores_scale_tensor;
      row_sum(Int<0>{}, Int<1>{}, mma_m_idx_tensor) *= scores_scale_tensor;
      row_sum(Int<1>{}, Int<1>{}, mma_m_idx_tensor) *= scores_scale_tensor;

      row_sum(Int<0>{}, Int<0>{}, mma_m_idx_simt) *= scores_scale_simt;
      row_sum(Int<1>{}, Int<0>{}, mma_m_idx_simt) *= scores_scale_simt;
      row_sum(Int<0>{}, Int<1>{}, mma_m_idx_simt) *= scores_scale_simt;
      row_sum(Int<1>{}, Int<1>{}, mma_m_idx_simt) *= scores_scale_simt;
    } else {
      row_sum(row_sum_tensor_idx_layout(Int<0>{}, mma_m_idx_tensor)) *= scores_scale_tensor;
      row_sum(row_sum_tensor_idx_layout(Int<1>{}, mma_m_idx_tensor)) *= scores_scale_tensor;
      row_sum(row_sum_tensor_idx_layout(Int<0>{}, mma_m_idx_simt)) *= scores_scale_simt;
      row_sum(row_sum_tensor_idx_layout(Int<1>{}, mma_m_idx_simt)) *= scores_scale_simt;
    }

    mma_frag_B(Int<0>{}) = scores_scale_tensor * frag_b_mask(Int<0>{});
    mma_frag_B(Int<1>{}) = scores_scale_tensor * frag_b_mask(Int<1>{});

    // tensor part
    for_each(make_int_sequence<AccOTraits::Acc_O_MMA_K>{}, [&](auto mma_k_idx){
      mma_unpack(
        MmaTraits{},
        acc_o(_, mma_m_idx_tensor, mma_k_idx),
        acc_o(_, mma_m_idx_tensor, mma_k_idx),
        mma_frag_B,
        mma_frag_RZ
      );
      float temp = acc_o(Int<1>{}, mma_m_idx_tensor, mma_k_idx);
      acc_o(Int<1>{}, mma_m_idx_tensor, mma_k_idx) = __fadd_rz(acc_o(Int<2>{}, mma_m_idx_tensor, mma_k_idx), 0.f);
      acc_o(Int<2>{}, mma_m_idx_tensor, mma_k_idx) = __fadd_rz(temp, 0.f);
    });

    // simt part
    for_each(make_int_sequence<AccOTraits::Acc_O_MMA_K>{}, [&](auto mma_k_idx){
      acc_o(Int<0>{}, mma_m_idx_simt, mma_k_idx) *= scores_scale_simt;
      acc_o(Int<1>{}, mma_m_idx_simt, mma_k_idx) *= scores_scale_simt;
      acc_o(Int<2>{}, mma_m_idx_simt, mma_k_idx) *= scores_scale_simt;
      acc_o(Int<3>{}, mma_m_idx_simt, mma_k_idx) *= scores_scale_simt;
    });
    
  }
};



template <>
struct AccOScale<AccOScaleMode::SIMT_ONLY> {
  template <bool Check_inf, int tensor_ratio, class HMMA1688FragBT, class RowTensorIdxLayout, class AccSTraits, class AccOTraits,
    class EngineMmaMax, class LayoutMmaMax,
    class EngineMmaMaxPrev, class LayoutMmaMaxPrev,
    class EngineRowSumOrPartialMmaSum, class LayoutRowSumOrPartialMmaSum,
    class EngineAccO, class LayoutAccO,
    class HMMA1688FragBMaskT
  >
  static CUTE_DEVICE void rescale_acc_o(
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max,
    Tensor<EngineMmaMaxPrev, LayoutMmaMaxPrev> const& mma_max_prev,
    Tensor<EngineRowSumOrPartialMmaSum, LayoutRowSumOrPartialMmaSum> & row_sum,
    Tensor<EngineAccO, LayoutAccO> & acc_o, // expect acc_o in (4, MMA_M, MMA_K) format
    HMMA1688FragBMaskT const& frag_b_mask,
    float const& softmax_scale_log2
  ){
    static_assert(size<0>(LayoutAccO{}) == 4, "Unexpected Acc_O tensor size");
    static_assert(size<1>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_M, "Unexpected Acc_O tensor size");
    static_assert(size<2>(LayoutAccO{}) == AccOTraits::Acc_O_MMA_K, "Unexpected Acc_O tensor size");

    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    auto row_sum_tensor_idx_layout = RowTensorIdxLayout{};
    // partial row sum layout : (MMA_n = 2, MMA_m = 2, MMA_M)
    constexpr bool is_partial = (LayoutRowSumOrPartialMmaSum::rank == 3);

    CUTE_UNROLL
    for (int mma_m_idx = 0; mma_m_idx < AccOTraits::Acc_O_MMA_M; ++mma_m_idx){
      float scores_max_cur = !Check_inf
      ? mma_max(mma_m_idx)
      : (mma_max(mma_m_idx) == MASK_VALUE ? 0.0f : mma_max(mma_m_idx));
      float scores_scale = exp2f((mma_max_prev(mma_m_idx) - scores_max_cur) * softmax_scale_log2);
      if constexpr (is_partial) {
        row_sum(Int<0>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<0>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<0>{}, Int<1>{}, mma_m_idx) *= scores_scale;
        row_sum(Int<1>{}, Int<1>{}, mma_m_idx) *= scores_scale;
      } else {
        row_sum(row_sum_tensor_idx_layout(Int<0>{}, mma_m_idx)) *= scores_scale;
        row_sum(row_sum_tensor_idx_layout(Int<1>{}, mma_m_idx)) *= scores_scale;
      }

      CUTE_UNROLL
      for (int mma_k_idx = 0; mma_k_idx < AccOTraits::Acc_O_MMA_K; ++mma_k_idx){
        // simt only
        acc_o(Int<0>{}, mma_m_idx, mma_k_idx) *= scores_scale;
        acc_o(Int<1>{}, mma_m_idx, mma_k_idx) *= scores_scale;
        acc_o(Int<2>{}, mma_m_idx, mma_k_idx) *= scores_scale;
        acc_o(Int<3>{}, mma_m_idx, mma_k_idx) *= scores_scale; 
      }
    }
    
  }
};

} // end namespace FLASH_NAMESPACE