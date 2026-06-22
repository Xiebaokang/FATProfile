#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cute/tensor.hpp>

#include "utils.cuh"

// simply disable everything that is irrelevant to our focus
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_DROPOUT
#define ENABLE_PRINT_CUSTOM_API_REPORT              0
#define ENABLE_PRINT_FLASH_FWD_KERNEL_TEMPLATE_ARGS 0
#define ENABLE_PRINT_DEVICE_DETAILS                 0
#define ENABLE_PRINT_FLASH_FWD_PARAMS               0
#define ENABLE_CUSTOM_DUMP_AND_PRINTF               0

// testing the mma softmax version
#define USE_DEFAULT_NEGINF_MASK                     1
#define USE_ACC_S_LEVEL_INF_MASKING                 0
#define USE_MMA_SOFTMAX                             0

// fmax reduce control
#define USE_BINARY_TREE_MAX                         0
#define USE_DEFAULT_MAX                             1

// loop1 accs scale method control
#define LOOP1_USE_ACCS_SCALE_MMA_ONLY               0
#define LOOP1_USE_ACCS_SCALE_SIMT_ONLY              1
#define LOOP1_USE_ACCS_SCALE_ILP_VERT               0
#define LOOP1_USE_ACCS_SCALE_ILP_HORI               0

#define LOOP1_ACCS_ILP_HORI_RATIO                   0

// loop2 accs scale method control
#define LOOP2_USE_ACCS_SCALE_MMA_ONLY               0
#define LOOP2_USE_ACCS_SCALE_SIMT_ONLY              1
#define LOOP2_USE_ACCS_SCALE_ILP_VERT               0
#define LOOP2_USE_ACCS_SCALE_ILP_HORI               0

#define LOOP2_ACCS_ILP_HORI_RATIO                   0

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


void loop_bench_fwd_fp16(std::string const& csv_filename = ""){
  constexpr int total_tokens = 16384;
  constexpr auto seqlens = cute::make_tuple(Int<128>{}, Int<256>{}, Int<512>{}, Int<1024>{}, Int<2048>{}, Int<4096>{}, Int<8192>{}, Int<16384>{});
  constexpr auto batch_sizes = cute::transform(seqlens, [&](auto seqlen){
    constexpr int seqlen_v = CUTE_STATIC_V(seqlen);
    return Int<total_tokens / seqlen_v>{};
  });
  constexpr int hid_dim = 2048;

  constexpr auto head_dims = cute::make_tuple(Int<64>{}, Int<128>{});
  constexpr auto num_heads = cute::transform(head_dims, [&](auto head_dim){
    constexpr int head_dim_v = CUTE_STATIC_V(head_dim);
    return Int<hid_dim / head_dim_v>{};
  });

  constexpr auto causal_list = cute::make_tuple(cute::false_type{}, cute::true_type{});

  constexpr auto causal_list_len = cute::rank(causal_list);
  constexpr int seqlens_len = cute::rank(seqlens);
  constexpr int head_dims_len = cute::rank(head_dims);

  std::vector<std::string> csv_header = {
    "DataType",
    "Comment",
    "batchsize",
    "nheads",
    "seqlen",
    "headdim",
    "is_causal",
    "time_ms"
  };
  std::vector<std::vector<std::string>> csv_data;
  
  for_each(make_int_sequence<causal_list_len>{}, [&](auto causal_idx){
    auto is_causal = cute::get<causal_idx>(causal_list);
    constexpr bool is_causal_v = CUTE_STATIC_V(is_causal);
    for_each(make_int_sequence<head_dims_len>{}, [&](auto num_head_head_dim_idx){
      auto head_dim = cute::get<num_head_head_dim_idx>(head_dims);
      auto num_head = cute::get<num_head_head_dim_idx>(num_heads);
      constexpr int head_dim_v = CUTE_STATIC_V(head_dim);
      constexpr int num_head_v = CUTE_STATIC_V(num_head);
      for_each(make_int_sequence<seqlens_len>{}, [&](auto batchsize_seqlen_idx){
        auto seqlen = cute::get<batchsize_seqlen_idx>(seqlens);
        auto batch_size = cute::get<batchsize_seqlen_idx>(batch_sizes);
        constexpr int seqlen_v = CUTE_STATIC_V(seqlen);
        constexpr int batch_size_v = CUTE_STATIC_V(batch_size);
        constexpr int iter = 1000;
        constexpr int warmup = 200;
        float avg_time_ms;
        if constexpr (head_dim_v == 64) {
          avg_time_ms = bench_fwd_fp16<head_dim_v, is_causal_v, 128, 128>(
            batch_size_v,
            num_head_v,
            seqlen_v,
            iter,
            warmup
          );
        } else {
          avg_time_ms = bench_fwd_fp16<head_dim_v, is_causal_v, 128, 64>(
            batch_size_v,
            num_head_v,
            seqlen_v,
            iter,
            warmup
          );
        }
        
        csv_data.push_back({
          std::string("FP16-FP32"),
          std::string("fa-t"),
          std::to_string(batch_size_v),
          std::to_string(num_head_v),
          std::to_string(seqlen_v),
          std::to_string(head_dim_v),
          std::to_string(is_causal_v),
          std::to_string(avg_time_ms)
        });
        printf("batch_size: %-10d, seqlen: %-10d, num_heads: %-10d, head_dim: %-10d, causal: %-10d, avg_time_ms: %.6f\n",
          batch_size_v, seqlen_v, num_head_v, head_dim_v, is_causal_v, avg_time_ms);
      });
    });
  });

  if (!csv_filename.empty()) {
    write_result_to_csv(csv_filename, csv_header, csv_data);
  }
  
}

int main(int argc, char* argv[]){
  std::string csv_filename = parse_filename_arg(argc, argv);
  loop_bench_fwd_fp16(csv_filename);
}