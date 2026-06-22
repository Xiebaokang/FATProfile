#pragma once
#include <cute/tensor.hpp>
#include "namespace_config.h"

namespace FLASH_NAMESPACE {


struct FlashFwdAccSTensorTraitsUnTemplated
{
  // The full acc_s tensor layout
  // (num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded)
  CUTE_HOST_DEVICE static constexpr auto get_full_acc_s_layout(
    int num_batch, int num_heads, int seqlen_q_rounded, int seqlen_k_rounded
  ){
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    auto ret = make_layout(
      make_shape(num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded),
      cute::GenRowMajor{}
    );
    return ret;
  }
}; // end of struct FlashFwdAccSTensorTraitsUnTemplated

template <class FlashFwdKernelTraits>
struct FlashFwdAccSTensorTraits : FlashFwdAccSTensorTraitsUnTemplated
{
  
  static constexpr int MMAInst_ShapeM = cute::tuple_element_t<0, typename FlashFwdKernelTraits::MMA_Atom_Arch::Shape_MNK>::value;
  static constexpr int MMAInst_ShapeN = cute::tuple_element_t<1, typename FlashFwdKernelTraits::MMA_Atom_Arch::Shape_MNK>::value;
  
  static_assert(MMAInst_ShapeM == 16, "Unexpected MMAInst_ShapeM");
  static_assert(MMAInst_ShapeN == 8, "Unexpected MMAInst_ShapeN");

  static constexpr int CTA_M = FlashFwdKernelTraits::kBlockM;
  static constexpr int CTA_N = FlashFwdKernelTraits::kBlockN;
  static constexpr int NWarps = FlashFwdKernelTraits::kNWarps;

  static constexpr int Acc_S_MMA_2M = 2;
  static constexpr int Acc_S_MMA_2N = 2;
  
  static constexpr int Acc_S_MMA_M = CTA_M / NWarps / MMAInst_ShapeM;
  static constexpr int Acc_S_MMA_N = CTA_N / MMAInst_ShapeN;


  static constexpr int warpsize = 32;


  CUTE_HOST_DEVICE static constexpr auto report(){
    if (cute::thread0()) {
      printf("Custom Report Acc_S traits | Acc_S_MMA_M: %d, Acc_S_MMA_N: %d, CTA_M: %d, CTA_N: %d, NWarps: %d\n",
        Acc_S_MMA_M, Acc_S_MMA_N, CTA_M, CTA_N, NWarps
      );
    }
    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif
  }

  // get (MMA_2M, MMA_2N) -> MMA4 layout
  CUTE_HOST_DEVICE static constexpr auto get_mma4_layout() {
    auto ret = cute::make_layout(
      cute::make_shape(cute::Int<Acc_S_MMA_2M>{}, cute::Int<Acc_S_MMA_2N>{}),
      cute::GenRowMajor{}
    );
    return ret;
  }

  // get (MblockNum, (WarpSize, kNWarps), MMA_2M, MMA_M) -> seqlen_q_rounded layout
  CUTE_HOST_DEVICE static constexpr auto get_seqlen_q_rounded_layout(int m_block_num) {

    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;

    auto warp_lane48_shape = make_shape(Int<4>{}, Int<8>{});
    auto warp_lane48_stride = make_stride(Int<0>{}, Int<1>{});

    auto ret = cute::make_layout(
      make_shape(m_block_num, make_shape(warp_lane48_shape, Int<NWarps>{}), Int<Acc_S_MMA_2M>{}, Int<Acc_S_MMA_M>{}),
      make_stride(Int<CTA_M>{}, make_stride(warp_lane48_stride, Int<16>{}), Int<8>{}, Int<16 * NWarps>{})
    );
    return ret;
  }

  // get (NblockNum, (WarpSize, kNWarps), MMA_2N, MMA_N) -> seqlen_k_rounded layout
  CUTE_HOST_DEVICE static constexpr auto get_seqlen_k_rounded_layout(int n_block_num) {
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;

    auto warp_lane48_shape = make_shape(Int<4>{}, Int<8>{});
    auto warp_lane48_stride = make_stride(Int<2>{}, Int<0>{});

    auto ret = cute::make_layout(
      make_shape(n_block_num, make_shape(warp_lane48_shape, Int<NWarps>{}), Int<Acc_S_MMA_2N>{}, Int<Acc_S_MMA_N>{}),
      make_stride(Int<Acc_S_MMA_N * 8>{}, make_stride(warp_lane48_stride, Int<0>{}), Int<1>{}, Int<8>{})
    );
    return ret;
  }

}; // end of struct FlashFwdAccSTensorTraits


struct FlashFwdRowMaxTensorTraitsUnTemplated
{
  /*
    The full rowmax dump layout
    (num_batch, num_heads, n_steps, seqlen_q_rounded)
  */
  CUTE_HOST_DEVICE static constexpr auto get_full_rowmax_layout(
    int num_batch, int num_heads, int seqlen_q_rounded, int n_steps
  ){
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    auto ret = make_layout(
      make_shape(num_batch, num_heads, seqlen_q_rounded, n_steps),
      cute::GenRowMajor{}
    );
    return ret;
  }
}; // end of struct FlashFwdRowMaxTensorTraitsUnTemplated

template <class FlashFwdKernelTraits>
struct FlashFwdRowMaxTensorTraits : FlashFwdRowMaxTensorTraitsUnTemplated{
  
  static constexpr int MMAInst_ShapeM = cute::tuple_element_t<0, typename FlashFwdKernelTraits::MMA_Atom_Arch::Shape_MNK>::value;
  static_assert(MMAInst_ShapeM == 16, "Unexpected MMAInst_ShapeM");

  static constexpr int CTA_M = FlashFwdKernelTraits::kBlockM;
  static constexpr int CTA_N = FlashFwdKernelTraits::kBlockN;
  static constexpr int NWarps = FlashFwdKernelTraits::kNWarps;
  
  static constexpr int RowMax_2M = 2;

  static constexpr int RowMax_M = CTA_M / NWarps / MMAInst_ShapeM;

  static constexpr int warpsize = 32;

  CUTE_HOST_DEVICE static constexpr auto report(){
    if (cute::thread0()) {
      printf("Custom Report RowMax traits | RowMax_M: %d, CTA_M: %d, NWarps: %d\n",
        RowMax_M, CTA_M, NWarps
      );
    }
    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif
  }

  // get (RowMax_2M, RowMax_M) -> rowmax_idx layout
  CUTE_HOST_DEVICE static constexpr auto get_rowmax_local_layout() {
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    auto ret = make_layout(
      make_shape(Int<RowMax_2M>{}, Int<RowMax_M>{}),
      cute::GenColMajor{}
    );
    return ret;
  }

  // get (MblockNum, (WarpSize, kNWarps), RowMax_2M, RowMax_M) -> seqlen_q_rounded layout
  CUTE_HOST_DEVICE static constexpr auto get_seqlen_q_rounded_layout(
    int m_block_num
  ){
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    
    auto warp_lane48_shape = make_shape(Int<4>{}, Int<8>{});
    auto warp_lane48_stride = make_stride(Int<0>{}, Int<1>{});

    auto ret = make_layout(
      make_shape(m_block_num, make_shape(warp_lane48_shape, Int<NWarps>{}), Int<RowMax_2M>{}, Int<RowMax_M>{}),
      make_stride(Int<CTA_M>{}, make_stride(warp_lane48_stride, Int<16>{}), Int<8>{}, Int<16 * NWarps>{})
    );
    return ret;
  }
  
  

}; // end of struct FlashFwdRowMaxTensorTraits


struct FlashFwdAccOTensorTratisUnTemplated
{
  // acc_o in each block: QK^T = (seqlen_q, seqlen_k), V = (seqlen_k, head_dim), O = (seqlen_q, head_dim)
  CUTE_HOST_DEVICE static constexpr auto get_full_acc_o_layout(
    int num_batch, int num_heads, int seqlen_q_rounded, int head_dim
  ){
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    auto ret = make_layout(
      make_shape(num_batch, num_heads, seqlen_q_rounded, head_dim),
      cute::GenRowMajor{}
    );
    return ret;
  }
}; // end of struct FlashFwdAccOTensorTratisUnTemplated

template <class FlashFwdKernelTraits>
struct FlashFwdAccOTensorTraits : FlashFwdAccOTensorTratisUnTemplated
{
  
  static constexpr int MMAInst_ShapeM = cute::tuple_element_t<0, typename FlashFwdKernelTraits::MMA_Atom_Arch::Shape_MNK>::value;
  static constexpr int MMAInst_ShapeN = cute::tuple_element_t<1, typename FlashFwdKernelTraits::MMA_Atom_Arch::Shape_MNK>::value;

  static_assert(MMAInst_ShapeM == 16, "Unexpected MMAInst_ShapeM");
  static_assert(MMAInst_ShapeN == 8, "Unexpected MMAInst_ShapeN");

  /* @@@@@@@@@@@@@ IMPORTANT @@@@@@@@@@@@@@@
    NOTE: in flash attention, acc_o partitions (kBlockM, kHeadDim):
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    So basically there is no horizontal blocking at all on acc_o !!
    @@@@@@@@@@@@@@ IMPORTANT @@@@@@@@@@@@@@@
  */ 
  static constexpr int CTA_M = FlashFwdKernelTraits::kBlockM;
  static constexpr int CTA_N = FlashFwdKernelTraits::kBlockN;
  static constexpr int CTA_K = FlashFwdKernelTraits::kHeadDim;
  static_assert(CTA_K > 0 && (CTA_K & (CTA_K - 1)) == 0, "CTA_K must be a power of 2");

  static constexpr int NWarps = FlashFwdKernelTraits::kNWarps;

  static constexpr int Acc_O_MMA_2M = 2;
  static constexpr int Acc_O_MMA_2K = 2;

  static constexpr int Acc_O_MMA_M = CTA_M / NWarps / MMAInst_ShapeM;
  static constexpr int Acc_O_MMA_K = CTA_K / MMAInst_ShapeN;

  static constexpr int warpsize = 32;

  CUTE_HOST_DEVICE static constexpr auto report(){
    if (cute::thread0()) {
      printf("Custom Report Acc_O traits | Acc_O_MMA_M: %d, Acc_O_MMA_K: %d, CTA_M: %d, CTA_N: %d, CTA_K: %d, NWarps: %d\n",
        Acc_O_MMA_M, Acc_O_MMA_K, CTA_M, CTA_N, CTA_K, NWarps
      );
    }
    #if defined(__CUDA_ARCH__)
    __syncthreads();
    #endif
  }

  // get (MMA_2M, MMA_2K) -> MMA4 layout
  CUTE_HOST_DEVICE static constexpr auto get_mma4_layout() {
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;
    auto ret = make_layout(
      make_shape(Int<Acc_O_MMA_2M>{}, Int<Acc_O_MMA_2K>{}),
      cute::GenRowMajor{}
    );
    return ret;
  }

  // get (MblockNum, (WarpSize, kNWarps), MMA_2M, MMA_M) -> seqlen_q_rounded layout
  CUTE_HOST_DEVICE static constexpr auto get_seqlen_q_rounded_layout(int m_block_num) {
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;

    auto warp_lane48_shape = make_shape(Int<4>{}, Int<8>{});
    auto warp_lane48_stride = make_stride(Int<0>{}, Int<1>{});

    auto ret = cute::make_layout(
      make_shape(m_block_num, make_shape(warp_lane48_shape, Int<NWarps>{}), Int<Acc_O_MMA_2M>{}, Int<Acc_O_MMA_M>{}),
      make_stride(Int<CTA_M>{}, make_stride(warp_lane48_stride, Int<16>{}), Int<8>{}, Int<16 * NWarps>{})
    );
    return ret;
  }

  // get ((WarpSize, kNWarps), MMA_2K, MMA_K) -> headdim layout
  CUTE_HOST_DEVICE static constexpr auto get_head_dim_layout() {
    using cute::Int;
    using cute::make_layout;
    using cute::make_shape;
    using cute::make_stride;

    auto warp_lane48_shape = make_shape(Int<4>{}, Int<8>{});
    auto warp_lane48_stride = make_stride(Int<2>{}, Int<0>{});

    auto ret = cute::make_layout(
      make_shape(make_shape(warp_lane48_shape, Int<NWarps>{}), Int<Acc_O_MMA_2K>{}, Int<Acc_O_MMA_K>{}),
      make_stride(make_stride(warp_lane48_stride, Int<0>{}), Int<1>{}, Int<8>{})
    );
    return ret;
  }
  
}; // end of struct FlashFwdAccOTensorTraits

} // namespace FLASH_NAMESPACE
