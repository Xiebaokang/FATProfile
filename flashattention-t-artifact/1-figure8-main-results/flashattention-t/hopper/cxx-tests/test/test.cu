#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <optional>

#include <cute/tensor.hpp>

// Keep this example focused on the ordinary FA3 forward path.
#define FLASHATTENTION_DISABLE_SPLIT
#define FLASHATTENTION_DISABLE_PAGEDKV
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_APPENDKV
#define FLASHATTENTION_DISABLE_PACKGQA
#define FLASHATTENTION_DISABLE_SM8x
#define FLASHATTENTION_DISABLE_LOCAL

#define ENABLE_CUSTOM_FWD_LAUNCH_TEMPLATE_REPORT 0
#define USE_MMA_SOFTMAX 0
#define USE_MIX_WGMMA 1

#include "custom_api.cuh"

namespace {

struct TConfig {
  int kBlockM = 128;
  int kBlockN = 128;
  int kStage = 2;
  int producer_reg_dealloc = 24;
  int consumer_reg_alloc = 240;
  int p_smem_k_tiles = 0;
  int q_reg_k_tiles = 0;
  int num_consumer = 2;
};

constexpr int kBatch = 1;
constexpr int kSeqlen = 4096;
constexpr int kNumHeads = 16;
constexpr int kHeadDim = 128;

constexpr TConfig cfg{};

template <typename Element, at::ScalarType InputType, bool IsCausal>
void benchmark(const char* dtype_name) {
  const auto fp32 = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
  const auto shape = std::initializer_list<int64_t>{kBatch, kSeqlen, kNumHeads, kHeadDim};

  // FA3 expects [batch, sequence, heads, head_dim]. FP8 output is BF16.
  auto q_fp32 = at::rand(shape, fp32);
  auto k_fp32 = at::rand(shape, fp32);
  auto v_fp32 = at::rand(shape, fp32);
  auto q = q_fp32.to(InputType);
  auto k = k_fp32.to(InputType);
  auto v = v_fp32.to(InputType);
  const auto output_type = InputType == at::kFloat8_e4m3fn ? at::kBFloat16 : at::kHalf;
  auto out = at::empty(shape, fp32.dtype(output_type));
  std::optional<at::Tensor> out_opt = out;
  const float softmax_scale = 1.0f / std::sqrt(float(kHeadDim));

  auto run = [&] {
    if constexpr (IsCausal) {
      custom_mha_fwd_causal<cfg, Element, kHeadDim, kHeadDim>(
          q, k, v, out_opt, softmax_scale);
    } else {
      custom_mha_fwd_noncausal<cfg, Element, kHeadDim, kHeadDim>(
          q, k, v, out_opt, softmax_scale);
    }
  };

  for (int i = 0; i < 50; ++i) run();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, at::cuda::getCurrentCUDAStream());
  for (int i = 0; i < 250; ++i) run();
  cudaEventRecord(stop, at::cuda::getCurrentCUDAStream());
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  const double time_ms = total_ms / 250;
  // QK^T and P*V each cost 2*B*H*S*S*D FLOPs. Causal computes half.
  double flops = 4.0 * kBatch * kNumHeads * kSeqlen * kSeqlen * kHeadDim;
  if constexpr (IsCausal) flops *= 0.5;
  const double tflops = flops / (time_ms * 1.0e9);

  std::printf("%-4s  %-10s  time = %8.3f ms,  throughput = %8.2f TFLOPS\n",
              dtype_name, IsCausal ? "causal" : "non-causal", time_ms,
              tflops);
}

}  // namespace

int main() {
  std::printf("FA3 forward: B=%d, S=%d, H=%d, D=%d\n",
              kBatch, kSeqlen, kNumHeads, kHeadDim);

  benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, false>("FP8");
  benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, true>("FP8");
  benchmark<cutlass::half_t, at::kHalf, false>("FP16");
  benchmark<cutlass::half_t, at::kHalf, true>("FP16");
  return 0;
}
