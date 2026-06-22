#pragma once

#include <cuda.h>
#include <stdexcept>
#include <cute/tensor.hpp>

#include "custom_meta.cuh"

namespace FLASH_NAMESPACE {

/////////////////// printf lock /////////////////////

#if ENABLE_CUSTOM_DUMP_AND_PRINTF

__device__ __managed__ int printf_lock = 0;

CUTE_DEVICE void lock_printf() {
    while (atomicCAS(&printf_lock, 0, 1) != 0) {}
}

CUTE_DEVICE void unlock_printf() {
    atomicExch(&printf_lock, 0);
}

#else

CUTE_DEVICE void lock_printf() {}
CUTE_DEVICE void unlock_printf() {}

#endif


/////////////////// acc_s dump //////////////////////

#if ENABLE_CUSTOM_DUMP_AND_PRINTF

__device__ __managed__ float* acc_s_dump_ptr = nullptr;
__device__ __managed__ int acc_s_dump_length = 0;

CUTE_HOST void init_acc_s_dump(
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded
) {
  auto full_layout = FlashFwdAccSTensorTraitsUnTemplated::get_full_acc_s_layout(
    num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded
  );
  int length = cute::cosize(full_layout);
  cudaError_t err = cudaMallocManaged(&acc_s_dump_ptr, length * sizeof(float));
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate acc_s dump");
  }
  memset(acc_s_dump_ptr, 0, length * sizeof(float));
  acc_s_dump_length = length;
  cudaDeviceSynchronize();
}

CUTE_HOST void free_acc_s_dump() {
  if (acc_s_dump_ptr == nullptr) {
    return;
  }
  cudaFree(acc_s_dump_ptr);
}

template <class AccSTraits, class AccSTensor>
CUTE_DEVICE void dump_acc_s(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  AccSTensor & acc_s
){

  if (acc_s_dump_ptr == nullptr) {
    return;
  }

  using cute::Int;
  using cute::make_layout;
  using cute::make_shape;
  using cute::make_stride;
  using cute::make_tensor;
  using cute::make_gmem_ptr;

  auto acc_s_dump_tensor = make_tensor(
    make_gmem_ptr(acc_s_dump_ptr),
    AccSTraits::get_full_acc_s_layout(
      num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded
    )
  );
  
  auto seqlen_q_layout = AccSTraits::get_seqlen_q_rounded_layout(m_block_num);
  auto seqlen_k_layout = AccSTraits::get_seqlen_k_rounded_layout(n_block_num);

  for (int mma_2m_idx = 0; mma_2m_idx < AccSTraits::Acc_S_MMA_2M; mma_2m_idx++)
  {
    for (int mma_2n_idx = 0; mma_2n_idx < AccSTraits::Acc_S_MMA_2N; mma_2n_idx++)
    {
      for (int mma_m_idx = 0; mma_m_idx < AccSTraits::Acc_S_MMA_M; mma_m_idx++)
      {
        for (int mma_n_idx = 0; mma_n_idx < AccSTraits::Acc_S_MMA_N; mma_n_idx++)
        {
          int seqlen_q_index = seqlen_q_layout(
              m_block_idx, tidx, mma_2m_idx, mma_m_idx
          );
          int seqlen_k_index = seqlen_k_layout(
              n_block_idx, tidx, mma_2n_idx, mma_n_idx
          );
          auto mma4_layout = AccSTraits::get_mma4_layout();
          int mma4_idx = mma4_layout(mma_2m_idx, mma_2n_idx);
          float acc_s_val = acc_s(mma4_idx, mma_m_idx, mma_n_idx);
          acc_s_dump_tensor(batch_idx, head_idx, seqlen_q_index, seqlen_k_index) = acc_s_val;
        }
      }
    }
  }
  __syncthreads();
}

__device__ __managed__ float* acc_s_after_softmax_dump_ptr = nullptr;
__device__ __managed__ int acc_s_after_softmax_dump_length = 0;

CUTE_HOST void init_acc_s_after_softmax_dump(
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded
) {
  auto full_layout = FlashFwdAccSTensorTraitsUnTemplated::get_full_acc_s_layout(
    num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded
  );
  auto length = cute::cosize(full_layout);
  cudaError_t err = cudaMallocManaged(&acc_s_after_softmax_dump_ptr, length * sizeof(float));
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate acc_s after softmax dump");
  }
  memset(acc_s_after_softmax_dump_ptr, 0, length * sizeof(float));
  acc_s_after_softmax_dump_length = length;
  cudaDeviceSynchronize();
}

CUTE_HOST void free_acc_s_after_softmax_dump() {
  if (acc_s_after_softmax_dump_ptr == nullptr) {
    return;
  }
  cudaFree(acc_s_after_softmax_dump_ptr);
}


template <class AccSTraits, class AccSTensor>
CUTE_DEVICE void dump_acc_s_after_softmax(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  AccSTensor & acc_s
){

  if (acc_s_after_softmax_dump_ptr == nullptr) {
    return;
  }

  using cute::Int;
  using cute::make_layout;
  using cute::make_shape;
  using cute::make_stride;
  using cute::make_tensor;
  using cute::make_gmem_ptr;

  auto acc_s_dump_tensor = make_tensor(
    make_gmem_ptr(acc_s_after_softmax_dump_ptr),
    AccSTraits::get_full_acc_s_layout(
      num_batch, num_heads, seqlen_q_rounded, seqlen_k_rounded
    )
  );
  
  auto seqlen_q_layout = AccSTraits::get_seqlen_q_rounded_layout(m_block_num);
  auto seqlen_k_layout = AccSTraits::get_seqlen_k_rounded_layout(n_block_num);

  for (int mma_2m_idx = 0; mma_2m_idx < AccSTraits::Acc_S_MMA_2M; mma_2m_idx++)
  {
    for (int mma_2n_idx = 0; mma_2n_idx < AccSTraits::Acc_S_MMA_2N; mma_2n_idx++)
    {
      for (int mma_m_idx = 0; mma_m_idx < AccSTraits::Acc_S_MMA_M; mma_m_idx++)
      {
        for (int mma_n_idx = 0; mma_n_idx < AccSTraits::Acc_S_MMA_N; mma_n_idx++)
        {
          int seqlen_q_index = seqlen_q_layout(
              m_block_idx, tidx, mma_2m_idx, mma_m_idx
          );
          int seqlen_k_index = seqlen_k_layout(
              n_block_idx, tidx, mma_2n_idx, mma_n_idx
          );
          auto mma4_layout = AccSTraits::get_mma4_layout();
          int mma4_idx = mma4_layout(mma_2m_idx, mma_2n_idx);
          float acc_s_val = acc_s(mma4_idx, mma_m_idx, mma_n_idx);
          acc_s_dump_tensor(batch_idx, head_idx, seqlen_q_index, seqlen_k_index) = acc_s_val;
        }
      }
    }
  }
  __syncthreads();
}

__device__ __managed__ float* row_max_dump_ptr = nullptr;
__device__ __managed__ int row_max_dump_length = 0;

CUTE_HOST void init_row_max_dump(
  int num_batch,
  int num_heads,
  int n_steps,
  int seqlen_q_rounded
) {
  auto full_layout = FlashFwdRowMaxTensorTraitsUnTemplated::get_full_rowmax_layout(
    num_batch, num_heads, n_steps, seqlen_q_rounded
  );
  auto length = cute::cosize(full_layout);
  cudaError_t err = cudaMallocManaged(&row_max_dump_ptr, length * sizeof(float));
  if (err != cudaSuccess) {
      throw std::runtime_error("Failed to allocate row max dump");
  }
  memset(row_max_dump_ptr, 0, length * sizeof(float));
  row_max_dump_length = length;
  cudaDeviceSynchronize();
}

CUTE_HOST void free_row_max_dump() {
  if (row_max_dump_ptr == nullptr) {
    return;
  }
  cudaFree(row_max_dump_ptr);
}

template <class RowMaxTraits, class RowMaxTensor>
CUTE_DEVICE void dump_row_max(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  RowMaxTensor & row_max
){
  if (row_max_dump_ptr == nullptr) {
    return;
  }
  
  using cute::Int;
  using cute::make_layout;
  using cute::make_shape;
  using cute::make_stride;
  using cute::make_tensor;
  using cute::make_gmem_ptr;

  // (num_batch, num_heads, seqlen_q_rounded, n_steps)
  auto row_max_dump_tensor = make_tensor(
    make_gmem_ptr(row_max_dump_ptr),
    RowMaxTraits::get_full_rowmax_layout(
      num_batch, num_heads, seqlen_q_rounded, n_block_num
    )
  );
  // (MblockNum, (WarpSize, kNWarps), RowMax_2M, RowMax_M) -> seqlen_q_rounded layout
  auto seqlen_q_layout = RowMaxTraits::get_seqlen_q_rounded_layout(m_block_num);

  static_assert(cute::is_static_v<decltype(cute::size(row_max))>, "RowMax size must be static");
  constexpr int row_max_size = decltype(cute::size(row_max))::value;
  
  
  if constexpr (row_max_size == RowMaxTraits::RowMax_2M * RowMaxTraits::RowMax_M) {
    // dumping the (RowMax_2M, RowMax_M) tensor
    for (int rowmax_2m_idx = 0; rowmax_2m_idx < RowMaxTraits::RowMax_2M; rowmax_2m_idx++)
    {
      for (int rowmax_m_idx = 0; rowmax_m_idx < RowMaxTraits::RowMax_M; rowmax_m_idx++)
      {
        int seqlen_q_index = seqlen_q_layout(
            m_block_idx, tidx, rowmax_2m_idx, rowmax_m_idx
        );
        // (RowMax_2M, RowMax_M) -> rowmax_idx layout
        auto rowmax_layout = RowMaxTraits::get_rowmax_local_layout();
        int rowmax_idx = rowmax_layout(rowmax_2m_idx, rowmax_m_idx);
        row_max_dump_tensor(batch_idx, head_idx, seqlen_q_index, n_block_idx) = row_max(rowmax_idx);
      }
    }
  } else if constexpr (row_max_size == RowMaxTraits::RowMax_M) {
    // dumping the (RowMax_M) tensor
    for (int rowmax_2m_idx = 0; rowmax_2m_idx < RowMaxTraits::RowMax_2M; rowmax_2m_idx++)
    {
      for (int rowmax_m_idx = 0; rowmax_m_idx < RowMaxTraits::RowMax_M; rowmax_m_idx++)
      {
        int seqlen_q_index = seqlen_q_layout(
            m_block_idx, tidx, rowmax_2m_idx, rowmax_m_idx
        );
        row_max_dump_tensor(batch_idx, head_idx, seqlen_q_index, n_block_idx) = row_max(rowmax_m_idx);
      }
    }
  } else {
    static_assert(cute::dependent_false<RowMaxTraits>, "RowMax size wrong");
  }

  
  
  __syncthreads();
  
}

#else

CUTE_HOST void init_acc_s_dump(
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded
) {}

CUTE_HOST void free_acc_s_dump() {}

template <class AccSTraits, class AccSTensor>
CUTE_DEVICE void dump_acc_s(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  AccSTensor & acc_s
) {}

CUTE_HOST void init_acc_s_after_softmax_dump(
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded
) {}

CUTE_HOST void free_acc_s_after_softmax_dump() {}

template <class AccSTraits, class AccSTensor>
CUTE_DEVICE void dump_acc_s_after_softmax(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int seqlen_k_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  AccSTensor & acc_s
) {}

CUTE_HOST void init_row_max_dump(
  int num_batch,
  int num_heads,
  int n_steps,
  int seqlen_q_rounded
) {}

CUTE_HOST void free_row_max_dump() {}

template <class RowMaxTraits, class RowMaxTensor>
CUTE_DEVICE void dump_row_max(
  int m_block_num,
  int n_block_num,
  int num_batch,
  int num_heads,
  int seqlen_q_rounded,
  int m_block_idx,
  int n_block_idx,
  int batch_idx,
  int head_idx,
  int tidx,
  RowMaxTensor & row_max
) {}

#endif


} // namespace FLASH_NAMESPACE