#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cute/tensor.hpp>

// Main entry point for the template "custom_mha_fwd"
// In this header "flash.h" is included, which provides Flash_fwd_params struct
#include <flash_fwd_launch_template.h>

#include <flash_api_custom.cuh>
#include <custom_meta.cuh>

using cutlass::round_up;
using namespace cute;


template <int HeadDim, bool IsCausal, int cta_m, int cta_n>
auto bench_fwd_fp16(
  int batch_size,
  int num_heads,
  int seqlen,
  int iter,
  int warmup
){
  auto q_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  auto k_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  auto v_tensor = at::rand({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  auto out = at::zeros({batch_size, seqlen, num_heads, HeadDim}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
  c10::optional<at::Tensor> out_opt = out;
  c10::optional<at::Tensor> alibi_null_opt = c10::nullopt;

  int seqlen_q_rounded = round_up(seqlen, 128);
  int seqlen_k_rounded = round_up(seqlen, 128);

  auto f_run_causal = [&](){
    custom_mha_fwd_causal<cute::half_t, HeadDim, cta_m, cta_n, 4>(
      q_tensor,
      k_tensor,
      v_tensor,
      out_opt,
      alibi_null_opt,
      0.0,
      1.0,
      -1,
      -1,
      0.0,
      false,
      c10::nullopt,
      0
    );
  };

  auto f_run_noncausal = [&](){
    custom_mha_fwd_noncausal<cute::half_t, HeadDim, cta_m, cta_n, 4>(
      q_tensor,
      k_tensor,
      v_tensor,
      out_opt,
      alibi_null_opt,
      0.0,
      1.0,
      -1,
      -1,
      0.0,
      false,
      c10::nullopt,
      0
    );
  };

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < warmup; ++i) {
    if constexpr (IsCausal) {
      f_run_causal();
    } else {
      f_run_noncausal();
    }
  }
  
  cudaEventRecord(start);
  for (int iter_idx = 0; iter_idx < iter; ++iter_idx) {
    if constexpr (IsCausal) {
      f_run_causal();
    } else {
      f_run_noncausal();
    }
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_elapsed_time_ms;
  cudaEventElapsedTime(&total_elapsed_time_ms, start, stop);
  float avg_time_ms = total_elapsed_time_ms / iter;

  return avg_time_ms;
  

}
