#pragma once

#include <cmath>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "utils.h"

#include "custom_meta.cuh"
#include "custom_numerical_limits.h"

namespace FLASH_NAMESPACE {

#define USE_STATIC_FOR_EACH 1

#if USE_STATIC_FOR_EACH
#define FOR_START(i, r) for_each(make_int_sequence<r>{}, [&](auto i){
#define FOR_END() });
#else
#define FOR_START(i, r) for(int i = 0; i < r; ++i){
#define FOR_END() }
#endif

enum class AccSScaleExp2Mode {
  MMA_ILP_HORI,
  MMA_ILP_VERT,
  MMA_ONLY,
  SIMT_ONLY
};

template <AccSScaleExp2Mode mode>
struct AccSScaleExp2;

template <>
struct AccSScaleExp2<AccSScaleExp2Mode::MMA_ILP_HORI> {
  template <int tensor_ratio, bool simt_first, class AccSTraits,
            class RowTensorIdxLayout, bool Scale_max = true, 
            class EngineAccS, class LayoutAccS,
            class EngineMmaMax, class LayoutMmaMax,
            class EngineScaleFrag, class LayoutScaleFrag>
  static CUTE_DEVICE void scale_apply_exp2(
    Tensor<EngineAccS, LayoutAccS> & acc_s_tensor, // expect acc_s in the original (4, MMA_M, MMA_N) layout
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max_tensor, // (MMA_2M, MMA_M)
    float const& scale,
    Tensor<EngineScaleFrag, LayoutScaleFrag> const& scale_frag_B // (2,) scale fragment
  ){
    [[maybe_unused]] constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
    constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    [[maybe_unused]] constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
    constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
    
    static_assert(cute::size<0>(LayoutAccS{}) == 4, "Unexpected Acc_S tensor size");
    static_assert(cute::size<1>(LayoutAccS{}) == MMA_M, "Unexpected Acc_S tensor size");
    static_assert(cute::size<2>(LayoutAccS{}) == MMA_N, "Unexpected Acc_S tensor size");
    static_assert(cute::size(LayoutMmaMax{}) == MMA_M, "Unexpected MMA_MAX tensor size");
    
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;

    FOR_START(mma_m_idx, MMA_M)
      const float max_16row = mma_max_tensor(mma_m_idx);
      const float max_16row_scaled = (max_16row == MASK_VALUE) ? 0.f : max_16row * (Scale_max ? scale : float(M_LOG2E));
      
      auto max_frag_C = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
          cute::GenColMajor{}
        )
      );
      cute::fill(max_frag_C, -max_16row_scaled);

      constexpr bool split_exp = false;

      if constexpr (split_exp) {
        
        FOR_START(mma_n_idx, MMA_N)
        if constexpr (simt_first) {
          if constexpr (mma_n_idx < MMA_N - tensor_ratio) {
            // simt part
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
          } else {
            // tensor part
            mma_unpack(
              MmaTraits{},
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              scale_frag_B,
              max_frag_C
            );
      
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx);
          }
        } else {
          if constexpr (mma_n_idx < tensor_ratio) {
            // tensor part
            mma_unpack(
              MmaTraits{},
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              scale_frag_B,
              max_frag_C
            );
      
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx);
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx);
          } else {
            // simt part
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled;
          }
        }
        FOR_END()

        FOR_START(mma_n_idx, MMA_N)
          if constexpr (mma_n_idx < tensor_ratio)
          {
            // tensor part
            float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = temp;
          } else {
            // simt part
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx));
          }
        FOR_END()
      } else { // no split_exp
        FOR_START(mma_n_idx, MMA_N)
        if constexpr (simt_first) {
          if constexpr (mma_n_idx < MMA_N - tensor_ratio) {
            // simt part
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
          } else {
            // tensor part
            // FIX ATTEMPT: using a separate frag D seems to degrade performance significantly?
            mma_unpack(
              MmaTraits{},
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              scale_frag_B,
              max_frag_C
            );
      
            float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = temp;
          }
        } else {
          if constexpr (mma_n_idx < tensor_ratio) {
            // tensor part
            // FIX ATTEMPT: using a separate frag D seems to degrade performance significantly?
            mma_unpack(
              MmaTraits{},
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              acc_s_tensor(_, mma_m_idx, mma_n_idx),
              scale_frag_B,
              max_frag_C
            );
      
            float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx));
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = temp;
          } else {
            // simt part
            acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
            acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
          }
        }
        FOR_END()
      }
    FOR_END()
  }
};

template <>
struct AccSScaleExp2<AccSScaleExp2Mode::MMA_ILP_VERT> {
  template <int tensor_ratio, bool simt_first, class AccSTraits,
            class RowTensorIdxLayout, bool Scale_max = true, 
            class EngineAccS, class LayoutAccS,
            class EngineMmaMax, class LayoutMmaMax,
            class EngineScaleFrag, class LayoutScaleFrag>
  static CUTE_DEVICE void scale_apply_exp2(
    Tensor<EngineAccS, LayoutAccS> & acc_s_tensor, // expect acc_s in the original (4, MMA_M, MMA_N) layout
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max_tensor, // (MMA_2M, MMA_M)
    float const& scale,
    Tensor<EngineScaleFrag, LayoutScaleFrag> const& scale_frag_B // (2,) scale fragment
  ){
    [[maybe_unused]] constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
    constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    [[maybe_unused]] constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
    constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
    
    static_assert(cute::size<0>(LayoutAccS{}) == 4, "Unexpected Acc_S tensor size");
    static_assert(cute::size<1>(LayoutAccS{}) == MMA_M, "Unexpected Acc_S tensor size");
    static_assert(cute::size<2>(LayoutAccS{}) == MMA_N, "Unexpected Acc_S tensor size");
    static_assert(cute::size(LayoutMmaMax{}) == MMA_M, "Unexpected MMA_MAX tensor size");
  
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;
    
    // TENSOR FIRST
    auto mma_m_idx_simt = Int<1>{};
    auto mma_m_idx_tensor = Int<0>{};
  
    const float max_16row_tensor = mma_max_tensor(mma_m_idx_tensor);
    const float max_16row_tensor_scaled = (max_16row_tensor == MASK_VALUE) ? 0.f : max_16row_tensor * (Scale_max ? scale : float(M_LOG2E));
    
    const float max_16row_simt = mma_max_tensor(mma_m_idx_simt);
    const float max_16row_simt_scaled = (max_16row_simt == MASK_VALUE) ? 0.f : max_16row_simt * (Scale_max ? scale : float(M_LOG2E));
  
    auto max_frag_C = cute::make_tensor<float>(
      cute::make_layout(
        cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
        cute::GenColMajor{}
      )
    );
    cute::fill(max_frag_C, -max_16row_tensor_scaled);
  
    /////// DEBUG CONSTRAINT
    static_assert(MMA_M != 1, "Vertical ILP not possible with MMA_M = 1 (CTA_M = 64)");
  
    FOR_START(mma_n_idx, MMA_N)
    if constexpr (simt_first) {

      float simt_val_0 = acc_s_tensor(Int<0>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_1 = acc_s_tensor(Int<1>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_2 = acc_s_tensor(Int<2>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_3 = acc_s_tensor(Int<3>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      
      // FIX ATTEMPT: using a separate frag D seems to degrade performance significantly?
      mma_unpack(
        MmaTraits{},
        acc_s_tensor(_, mma_m_idx_tensor, mma_n_idx),
        acc_s_tensor(_, mma_m_idx_tensor, mma_n_idx),
        scale_frag_B,
        max_frag_C
      );
  
      acc_s_tensor(Int<0>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_0);
      acc_s_tensor(Int<1>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_1);
      acc_s_tensor(Int<2>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_2);
      acc_s_tensor(Int<3>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_3);
  
      float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<0>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<3>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<2>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<1>{}, mma_m_idx_tensor, mma_n_idx) = temp;
    } else {
      // FIX ATTEMPT: using a separate frag D seems to degrade performance significantly?
      mma_unpack(
        MmaTraits{},
        acc_s_tensor(_, mma_m_idx_tensor, mma_n_idx),
        acc_s_tensor(_, mma_m_idx_tensor, mma_n_idx),
        scale_frag_B,
        max_frag_C
      );
  
      float simt_val_0 = acc_s_tensor(Int<0>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_1 = acc_s_tensor(Int<1>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_2 = acc_s_tensor(Int<2>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
      float simt_val_3 = acc_s_tensor(Int<3>{}, mma_m_idx_simt, mma_n_idx) * scale - max_16row_simt_scaled;
  
      acc_s_tensor(Int<0>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_0);
      acc_s_tensor(Int<1>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_1);
      acc_s_tensor(Int<2>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_2);
      acc_s_tensor(Int<3>{}, mma_m_idx_simt, mma_n_idx) = exp2f(simt_val_3);
  
      float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<0>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<3>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<2>{}, mma_m_idx_tensor, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx_tensor, mma_n_idx));
      acc_s_tensor(Int<1>{}, mma_m_idx_tensor, mma_n_idx) = temp;
    }
    FOR_END()
  }
};

template <>
struct AccSScaleExp2<AccSScaleExp2Mode::MMA_ONLY> {
  template <int tensor_ratio, bool simt_first, class AccSTraits,
            class RowTensorIdxLayout, bool Scale_max = true, 
            class EngineAccS, class LayoutAccS,
            class EngineMmaMax, class LayoutMmaMax,
            class EngineScaleFrag, class LayoutScaleFrag>
  static CUTE_DEVICE void scale_apply_exp2(
    Tensor<EngineAccS, LayoutAccS> & acc_s_tensor, // expect acc_s in the original (4, MMA_M, MMA_N) layout
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max_tensor, // (MMA_2M, MMA_M)
    float const& scale,
    Tensor<EngineScaleFrag, LayoutScaleFrag> const& scale_frag_B // (2,) scale fragment
  ){
    [[maybe_unused]] constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
    constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    [[maybe_unused]] constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
    constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
    
    static_assert(cute::size<0>(LayoutAccS{}) == 4, "Unexpected Acc_S tensor size");
    static_assert(cute::size<1>(LayoutAccS{}) == MMA_M, "Unexpected Acc_S tensor size");
    static_assert(cute::size<2>(LayoutAccS{}) == MMA_N, "Unexpected Acc_S tensor size");
    static_assert(cute::size(LayoutMmaMax{}) == MMA_M, "Unexpected MMA_MAX tensor size");
  
    using MmaArchAtom = cute::SM80_16x8x8_F32TF32TF32F32_TN;
    using MmaTraits = cute::MMA_Traits<MmaArchAtom>;
  
    FOR_START(mma_m_idx, MMA_M)
      const float max_16row = mma_max_tensor(mma_m_idx);
      const float max_16row_scaled = (max_16row == MASK_VALUE) ? 0.f : max_16row * (Scale_max ? scale : float(M_LOG2E));
      
      auto max_frag_C = cute::make_tensor<float>(
        cute::make_layout(
          cute::make_shape(Int<2>{}, Int<2>{}), // (MMA_2M, MMA_2N), row major fragment
          cute::GenColMajor{}
        )
      );
      cute::fill(max_frag_C, -max_16row_scaled);
  
      FOR_START(mma_n_idx, MMA_N)
        // FIX ATTEMPT: using a separate frag D seems to degrade performance significantly?
        mma_unpack(
          MmaTraits{},
          acc_s_tensor(_, mma_m_idx, mma_n_idx),
          acc_s_tensor(_, mma_m_idx, mma_n_idx),
          scale_frag_B,
          max_frag_C
        );
  
        float temp = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx));
        acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx));
        acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx));
        acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx));
        acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = temp;
      FOR_END()
    FOR_END()
  }
};

template <>
struct AccSScaleExp2<AccSScaleExp2Mode::SIMT_ONLY> {
  template <int tensor_ratio, bool simt_first, class AccSTraits,
            class RowTensorIdxLayout, bool Scale_max = true, 
            class EngineAccS, class LayoutAccS,
            class EngineMmaMax, class LayoutMmaMax,
            class EngineScaleFrag, class LayoutScaleFrag>
  static CUTE_DEVICE void scale_apply_exp2(
    Tensor<EngineAccS, LayoutAccS> & acc_s_tensor, // expect acc_s in the original (4, MMA_M, MMA_N) layout
    Tensor<EngineMmaMax, LayoutMmaMax> const& mma_max_tensor, // (MMA_2M, MMA_M)
    float const& scale,
    Tensor<EngineScaleFrag, LayoutScaleFrag> const& scale_frag_B // (2,) scale fragment
  ){
    [[maybe_unused]] constexpr int MMA_2M = AccSTraits::Acc_S_MMA_2M;
    constexpr int MMA_M = AccSTraits::Acc_S_MMA_M;
    [[maybe_unused]] constexpr int MMA_2N = AccSTraits::Acc_S_MMA_2N;
    constexpr int MMA_N = AccSTraits::Acc_S_MMA_N;
    
    static_assert(cute::size<0>(LayoutAccS{}) == 4, "Unexpected Acc_S tensor size");
    static_assert(cute::size<1>(LayoutAccS{}) == MMA_M, "Unexpected Acc_S tensor size");
    static_assert(cute::size<2>(LayoutAccS{}) == MMA_N, "Unexpected Acc_S tensor size");
    static_assert(cute::size(LayoutMmaMax{}) == MMA_M, "Unexpected MMA_MAX tensor size");

    FOR_START(mma_m_idx, MMA_M)
      const float max_16row = mma_max_tensor(mma_m_idx);
      const float max_16row_scaled = (max_16row == MASK_VALUE) ? 0.f : max_16row * (Scale_max ? scale : float(M_LOG2E));
      FOR_START(mma_n_idx, MMA_N)
        #ifdef UNFUSE_FMA
        acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(__fmul_rn(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx), scale) - max_16row_scaled);
        acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = exp2f(__fmul_rn(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx), scale) - max_16row_scaled);
        acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(__fmul_rn(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx), scale) - max_16row_scaled);
        acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(__fmul_rn(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx), scale) - max_16row_scaled);
        #else
        acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<0>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
        acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<1>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
        acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<2>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
        acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) = exp2f(acc_s_tensor(Int<3>{}, mma_m_idx, mma_n_idx) * scale - max_16row_scaled);
        #endif
      FOR_END()

    FOR_END()
  }
};

#undef USE_STATIC_FOR_EACH
#undef FOR_START
#undef FOR_END

} // end namespace FLASH_NAMESPACE