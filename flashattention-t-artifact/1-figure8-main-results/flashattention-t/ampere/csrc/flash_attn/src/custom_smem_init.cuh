#pragma once
#include <cute/tensor.hpp>
#include "namespace_config.h"

namespace FLASH_NAMESPACE {

using namespace cute;

template <class EngineSmemTensor, class LayoutSmemTensor>
CUTE_DEVICE auto tile_smem_Q_tensor(
  Tensor<EngineSmemTensor, LayoutSmemTensor> smem_q_tensor
){
  // smem shape M, should be the same as kBlockM (CTA_M)
  constexpr int smem_q_shape_M = size<0>(smem_q_tensor);
  // smem shape N, should be the same as kHeadDim
  constexpr int smem_q_shape_N = size<1>(smem_q_tensor);

  constexpr int tiler_M = 16;
  static_assert(smem_q_shape_M % tiler_M == 0, "smem_q_shape_M must be divisible by tiler_M");

  constexpr auto tiler = make_tile(Int<tiler_M>{}, Int<smem_q_shape_N>{});
  // tiled smem q tensor, with layout ((16, kheadDim), (restM, 1))
  auto tiled_smem_q_tensor = zipped_divide(
    smem_q_tensor,
    tiler
  );
  static_assert(size<1,1>(tiled_smem_q_tensor) == 1, "Tiled smem q tensor should have a size of 1 in the second dimension");
  return tiled_smem_q_tensor;
}

template <int NWarps, typename ElemT>
struct SmemFillTiledCpTraits {
  static_assert(NWarps == 4 || NWarps == 8, "Only 4 or 8 warps are supported for SmemFillTiledCpTraits");
  using CopyAtom = Copy_Atom<UniversalCopy<uint32_t, uint32_t>, ElemT>;
  using ThrLayout128T = Layout<
    Shape<Int<8>, Shape<Int<4>, Int<4>>>,
    Stride<Int<4>, Stride<Int<1>, Int<32>>>
  >;
  using ThrLayout256T = Layout<
    Shape<Shape<Int<2>, Int<8>>, Shape<Int<4>, Int<4>>>,
    Stride<Stride<Int<128>, Int<4>>, Stride<Int<1>, Int<32>>>
  >;
  using ThrLayout = std::conditional_t<
    NWarps == 4,
    ThrLayout128T,
    ThrLayout256T
  >;
  using ValLayout = Layout<
    Shape<Int<1>, Int<2>>,
    Stride<Int<0>, Int<1>>
  >;
  using TiledCp = decltype(
    make_tiled_copy(
      CopyAtom{},
      ThrLayout{},
      ValLayout{}
    )
  );
}; // end struct SmemFillTiledCpTraits

struct SmemFiller {

  template <bool Is_even_MN, bool Is_even_M>
  static CUTE_DEVICE constexpr bool KU_is_static_bypass() {
    return Is_even_MN || Is_even_M;
  }

  template <int CTA_M>
  static CUTE_DEVICE bool BU_is_dynamic_bypass(
    int actual_seqlen_q
  ) {
    int m_block = blockIdx.x;

    int cur_m_block_seqlen_q_max = CTA_M * (m_block + 1);
    bool is_dynamic_bypass = cur_m_block_seqlen_q_max <= actual_seqlen_q;
    return is_dynamic_bypass;
  }

  template <bool VecFill, bool Is_even_MN, bool Is_even_M, int CTA_M, int NWaprs, class ElemT, class EngineSmemTensor, class LayoutSmemTensor>
  static CUTE_DEVICE bool init_smem_q(
    Tensor<EngineSmemTensor, LayoutSmemTensor> smem_q_tensor,
    int actual_seqlen_q,
    int tidx
  ) {
    constexpr bool is_static_bypass = KU_is_static_bypass<Is_even_MN, Is_even_M>();
    if constexpr (is_static_bypass) {
      return false;
    }
    bool is_block_dynamic_bypass = BU_is_dynamic_bypass<CTA_M>(actual_seqlen_q);
    if (is_block_dynamic_bypass) {
      return false;
    }

    constexpr int tiler_M = 16;
    int target_smem_tile_idx = (actual_seqlen_q % CTA_M) / tiler_M;

    // ((16, kheadDim), (restM, 1))
    auto tiled_smem_q_tensor = tile_smem_Q_tensor(smem_q_tensor);
    auto smem_q_tile = tiled_smem_q_tensor(make_coord(_, _), make_coord(target_smem_tile_idx, _0{}));
    auto tiled_cp = typename SmemFillTiledCpTraits<NWaprs, ElemT>::TiledCp{};
    
    if constexpr (VecFill) {
      auto tidfrg_D = tiled_cp.tidfrg_D(smem_q_tile);
      // ((ThrV,ThrX),FrgV=1,(RestM,RestN,...))
      auto tidfrg_D_uint32 = recast<uint32_t>(tidfrg_D);
      static_assert(size<1>(tidfrg_D_uint32) == 1, "Tiled partition should have size 1 in the second dimension");
      constexpr uint32_t fill_value = 0;
      CUTE_UNROLL
      for (int rest_idx = 0; rest_idx < size<2>(tidfrg_D_uint32); ++rest_idx) {
        tidfrg_D_uint32(tidx, _0{}, rest_idx) = fill_value;
      }
    } else {
      auto tidfrg_D = tiled_cp.tidfrg_D(smem_q_tile);
      static_assert(size<1>(tidfrg_D) == 2, "Tiled partition should have size 2 in the second dimension");
      constexpr float fill_value = 0.f;
      CUTE_UNROLL
      for (int rest_idx = 0; rest_idx < size<2>(tidfrg_D); ++rest_idx) {
        tidfrg_D(tidx, _0{}, rest_idx) = ElemT(fill_value);
        tidfrg_D(tidx, _1{}, rest_idx) = ElemT(fill_value);
      }
    }

    return true;
  }
}; // end struct SmemFiller


} // namespace FLASH_NAMESPACE