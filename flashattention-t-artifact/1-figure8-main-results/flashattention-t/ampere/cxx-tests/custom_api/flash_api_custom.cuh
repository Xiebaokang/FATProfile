#include <cuda.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <ATen/ATen.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <flash_fwd_launch_template.h>

#include "utility.hpp"

// using namespace flash;

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


template <typename T>
inline void check_var_value(T const& value, T const& assumed_value, std::string_view const& var_name)
{
  if constexpr (std::is_convertible_v<T, double>){
    double value_double = static_cast<double>(value);
    double assumed_value_double = static_cast<double>(assumed_value);
    if (value_double != assumed_value_double)
    {
      std::fprintf(stderr, "Custom Error: %s == %f, but ASSUMED to be %f\n", var_name.data(), value_double, assumed_value_double);
      throw std::runtime_error("Var value check failed, see above error message");
    }
  } else if constexpr (std::is_convertible_v<T, float>){
    float value_float = static_cast<float>(value);
    float assumed_value_float = static_cast<float>(assumed_value);
    if (value_float != assumed_value_float)
    {
      std::fprintf(stderr, "Custom Error: %s == %f, but ASSUMED to be %f\n", var_name.data(), value_float, assumed_value_float);
      throw std::runtime_error("Var value check failed, see above error message");
    }
  } else if constexpr (std::is_convertible_v<T, int>){
    int value_int = static_cast<int>(value);
    int assumed_value_int = static_cast<int>(assumed_value);
    if (value_int != assumed_value_int)
    {
      std::fprintf(stderr, "Custom Error: %s == %d, but ASSUMED to be %d\n", var_name.data(), value_int, assumed_value_int);
      throw std::runtime_error("Var value check failed, see above error message");
    }
  } else {
    throw std::runtime_error("Unsupported type for var value check");
  }
}

template <typename T>
inline void report_value(T const& value, std::string_view const& value_name){
  #if ENABLE_PRINT_CUSTOM_API_REPORT
  std::printf("Custom Report: %s = %d\n", value_name.data(), value);
  #endif
}

inline void custom_report(const char* format, ...) {
  #if ENABLE_PRINT_CUSTOM_API_REPORT
  va_list args;
  
  // First pass: determine required size
  va_start(args, format);
  int size = vsnprintf(nullptr, 0, format, args) + 1; // +1 for null terminator
  va_end(args);
  
  if (size <= 0) {
      printf("Custom Report: [Formatting error]\n");
      return;
  }
  
  // Allocate buffer on heap
  char* buffer = (char*)malloc(size);
  if (!buffer) {
      printf("Custom Report: [Memory allocation failed]\n");
      return;
  }
  
  // Second pass: actually format the string
  va_start(args, format);
  vsnprintf(buffer, size, format, args);
  va_end(args);
  
  // Print with prefix
  printf("Custom Report: %s", buffer);
  
  // Clean up
  free(buffer);
  #endif
}




static void set_params_fprop(flash::Flash_fwd_params &params,
  // sizes
  const size_t b,
  const size_t seqlen_q,
  const size_t seqlen_k,
  const size_t seqlen_q_rounded,
  const size_t seqlen_k_rounded,
  const size_t h,
  const size_t h_k,
  const size_t d,
  const size_t d_rounded,
  // device pointers
  const at::Tensor q,
  const at::Tensor k,
  const at::Tensor v,
  at::Tensor out,
  void *cu_seqlens_q_d,
  void *cu_seqlens_k_d,
  void *seqused_k,
  void *p_d,
  void *softmax_lse_d,
  float p_dropout,
  float softmax_scale,
  int window_size_left,
  int window_size_right,
  const float softcap,
  bool seqlenq_ngroups_swapped=false,
  const bool unpadded_lse=false) {

  // Reset the parameters
  params = {};

  params.is_bf16 = q.dtype() == torch::kBFloat16;

  // Set the pointers and strides.
  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  // All stride are in elements, not bytes.
  params.q_row_stride = q.stride(-3);
  params.k_row_stride = k.stride(-3);
  params.v_row_stride = v.stride(-3);
  params.q_head_stride = q.stride(-2);
  params.k_head_stride = k.stride(-2);
  params.v_head_stride = v.stride(-2);
  params.o_ptr = out.data_ptr();
  params.o_row_stride = out.stride(-3);
  params.o_head_stride = out.stride(-2);

  if (cu_seqlens_q_d == nullptr) {
    params.q_batch_stride = q.stride(0);
    params.k_batch_stride = k.stride(0);
    params.v_batch_stride = v.stride(0);
    params.o_batch_stride = out.stride(0);
    if (seqlenq_ngroups_swapped) {
      params.q_batch_stride *= seqlen_q;
      params.o_batch_stride *= seqlen_q;
    }
  }

  params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
  params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
  params.seqused_k = static_cast<int *>(seqused_k);

  // P = softmax(QK^T)
  params.p_ptr = p_d;

  // Softmax sum
  params.softmax_lse_ptr = softmax_lse_d;

  // Set the dimensions.
  params.b = b;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = d;
  params.d_rounded = d_rounded;

  // Set the different scale values.
  #ifdef FLASHATTENTION_DISABLE_SOFTCAP
  TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
  #endif
  if (softcap > 0.0) {
    params.softcap = softmax_scale / softcap;
    params.scale_softmax = softcap;
    params.scale_softmax_log2 = softcap * M_LOG2E;
  } else{
    // Remove potential NaN
    params.softcap = 0.0;
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  }

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f - p_dropout;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;
  params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
  TORCH_CHECK(p_dropout < 1.f);
  #ifdef FLASHATTENTION_DISABLE_DROPOUT
  TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
  #endif

  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;

  if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
  if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  #ifdef FLASHATTENTION_DISABLE_LOCAL
  TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
  "This flash attention build does not support local attention.");
  #endif

  params.is_seqlens_k_cumulative = true;

  #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
  TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
  #endif

  params.unpadded_lse = unpadded_lse;
  params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}



// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
static inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
  // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
  // (i.e. it's 11 splits anyway).
  // So we check if the number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
      return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) { max_efficiency = eff; }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) { continue; }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

static inline void report_num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
  report_value(batch_nheads_mblocks, "batch_nheads_mblocks (batchsize x nheads x number of mblocks)");
  report_value(num_SMs, "num_SMs (number of SMs)");
  report_value(num_n_blocks, "num_n_blocks (number of n_blocks)");
  report_value(max_splits, "max_splits (maximum number of splits)");
  

  bool is_bypassed_by_08times_SMs = batch_nheads_mblocks >= 0.8f * num_SMs;
  if (is_bypassed_by_08times_SMs) {
    custom_report("%s\n", green_text("Custom Report: num_splits_heuristic is bypassed by 0.8 times SMs (result should be 1)").c_str());
  } else {
    custom_report("%s\n", red_text("Custom Report: num_splits_heuristic NOT bypassed!!").c_str());
  }
  auto num_splits_result = num_splits_heuristic(batch_nheads_mblocks, num_SMs, num_n_blocks, max_splits);
  report_value(num_splits_result, "num_splits_result (result of calling num_splits_heuristic)");
}

static std::tuple<at::Tensor, at::Tensor> set_params_splitkv(flash::Flash_fwd_params &params, const int batch_size,
  const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
  const int head_size_rounded, const float p_dropout,
  const int num_splits, cudaDeviceProp *dprops, struct c10::TensorOptions opts) {

  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
  const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
  // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
  // In any case we don't expect seqlen_q to be larger than 64 for inference.
  const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
  params.num_splits = num_splits;
  at::Tensor softmax_lse_accum;
  at::Tensor out_accum;

  report_value(block_n, "block_n (Inside set_params_splitkv: CTA_N)");
  report_value(num_n_blocks, "num_n_blocks (Inside set_params_splitkv: number of N blocks)");
  report_value(num_m_blocks, "num_m_blocks (Inside set_params_splitkv: number of M blocks)");

  custom_report("Custom Report: set_params_splitkv received num_splits as %d\n", num_splits);
  if (num_splits == 1) {
    custom_report("Custom Report: Inside set_params_splitkv: num_splits received was 1, the branch logic is bypassed\n");
  }

  if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
    if (num_splits < 1) {
      custom_report("Custom Report: Inside set_params_splitkv: Entering the num_splits < 1 branch, will report num_splits_heuristic\n");
      // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
      params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
      report_num_splits_heuristic(batch_size * num_heads * num_m_blocks, dprops->multiProcessorCount * 2, num_n_blocks, 128);
    }
    if (params.num_splits > 1) {
      custom_report("Custom Report: Inside set_params_splitkv: Entering the num_splits > 1 branch\n");
      softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
      out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
      params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
      params.oaccum_ptr = out_accum.data_ptr();
    }
    TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
  }

  report_value(params.num_splits, "params.num_splits (Inside set_params_splitkv: eventual result of num_splits set in the params)");

  return std::make_tuple(softmax_lse_accum, out_accum);
}


template <typename Dtype, int HeadDim, int CtaM, int CtaN, int NWarps, bool IsCausal>
std::vector<at::Tensor>
custom_mha_fwd_template_core(
  at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &alibi_slopes_, // leave empty in this custom implementation
  const float p_dropout, // must be 0.0 in this custom implementation
  const float softmax_scale,
  int window_size_left, // must be -1 in this custom implementation
  int window_size_right, // must be -1 in this custom implementation
  const float softcap, // must be 0.0 in this custom implementation
  const bool return_softmax,
  c10::optional<at::Generator> gen_,
  int num_splits
) {

  static_assert(std::is_same_v<Dtype,cute::bfloat16_t> || std::is_same_v<Dtype,cute::half_t>, "Unexpected Dtype");

  constexpr bool is_bfloat16 = std::is_same_v<Dtype, cute::bfloat16_t>;
  constexpr bool is_float16 = std::is_same_v<Dtype, cute::half_t>;

  auto dprops = at::cuda::getCurrentDeviceProperties();

  const auto sizes = q.sizes();
  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);

  check_var_value(p_dropout, 0.0f, "p_dropout");
  check_var_value(window_size_left, -1, "window_size_left");
  check_var_value(window_size_right, -1, "window_size_right");
  check_var_value(softcap, 0.0f, "softcap");
  check_var_value(alibi_slopes_.has_value(), false, "alibi_slopes_");

  constexpr bool is_causal = IsCausal;

  //// Copied from mha_fwd

  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value()) { TORCH_CHECK(is_causal == false); }
  if (is_causal) { window_size_right = 0; }

  // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
  // H/t Daniel Haziza
  const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && p_dropout == 0.f && head_size % 8 == 0 && !alibi_slopes_.has_value();
  const int ngroups = num_heads / num_heads_k;
  if (seqlenq_ngroups_swapped) {
      q = q.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
      seqlen_q = ngroups;
      num_heads = num_heads_k;
  }

  ////////////////////////

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out;
  if (out_.has_value()) {
      out = out_.value();
      auto out_dtype = out.dtype();

      if constexpr (is_float16) {
        TORCH_CHECK(out_dtype.toScalarType() == at::ScalarType::Half, "Output tensor must be of dtype torch.float16");
      }
      if constexpr (is_bfloat16) {
        TORCH_CHECK(out_dtype.toScalarType() == at::ScalarType::BFloat16, "Output tensor must be of dtype torch.bfloat16");
      }

      CHECK_DEVICE(out);
      TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
      CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
      if (seqlenq_ngroups_swapped) {
          out = out.reshape({batch_size, num_heads_k, ngroups, head_size}).transpose(1, 2);
      }
  } else {
      out = torch::empty_like(q);
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  // head dim 小于等于 192 时，向上取整到 32 的倍数，否则向上取整到 256
  const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
  // seqlen对其到128
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  // Otherwise the kernel will be launched from cuda:0 device
  // Cast to char to avoid compiler warning about narrowing
  at::cuda::CUDAGuard device_guard{(char)q.get_device()};

  auto opts = q.options();

  auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor p;
  // Only return softmax if there's dropout to reduce compilation time
  // TODO: 下面是原始逻辑，需要修改
  if (return_softmax) {
    TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
    p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
  }
  else {
    p = torch::empty({ 0 }, opts);
  }

  flash::Flash_fwd_params params;
  set_params_fprop(params,
                  batch_size,
                  seqlen_q, seqlen_k,
                  seqlen_q_rounded, seqlen_k_rounded,
                  num_heads, num_heads_k,
                  head_size, head_size_rounded,
                  q, k, v, out,
                  /*cu_seqlens_q_d=*/nullptr,
                  /*cu_seqlens_k_d=*/nullptr,
                  /*seqused_k=*/nullptr,
                  return_softmax ? p.data_ptr() : nullptr,
                  softmax_lse.data_ptr(),
                  p_dropout,
                  softmax_scale,
                  window_size_left,
                  window_size_right,
                  softcap
                  );

  at::Tensor softmax_lse_accum, out_accum;
  std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
      params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
      head_size_rounded, p_dropout, /*num_splits*/ num_splits, dprops, opts);

  // number of times random will be generated per thread, to offset philox counter in thc random
  // state
  // We use a custom RNG that increases the offset by batch_size * nheads * 32.
  int64_t counter_offset = params.b * params.h * 32;
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
  // Forward kernel will populate memory with the seed and offset.
  params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

  if (p_dropout > 0.0)  {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    params.philox_args = gen->philox_cuda_state(counter_offset);
  }

  // NOTE: 我们不关心这个alibi slope，这里直接强制alibi_slopes_ptr为nullptr
  // 关于alibi的具体说明见flash_api_summary.md文档
  // set_params_alibi(params, alibi_slopes_, batch_size, num_heads);
  params.alibi_slopes_ptr = nullptr;

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (seqlen_k == 0){
    // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
    out.zero_();
    softmax_lse.fill_(std::numeric_limits<float>::infinity());
    goto seqlen_k_is_zero;
  }


  flash::run_custom_mha_fwd<
    /*T=*/ Dtype,
    /*Is_causal=*/ is_causal,
    /*HeadDim=*/ HeadDim,
    /*CtaM=*/ CtaM,
    /*CtaN=*/ CtaN,
    /*NWarps=*/ NWarps
  >(params, stream);



seqlen_k_is_zero:

  if (seqlenq_ngroups_swapped) {
    out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
    q = q.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size});
    softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
  }
  return {out, softmax_lse, p, rng_state};

}

template <typename Dtype, int HeadDim, int CtaM, int CtaN, int NWarps>
std::vector<at::Tensor>
custom_mha_fwd_noncausal(
  at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &alibi_slopes_, // leave empty in this custom implementation
  const float p_dropout, // must be 0.0 in this custom implementation
  const float softmax_scale,
  int window_size_left, // must be -1 in this custom implementation
  int window_size_right, // must be -1 in this custom implementation
  const float softcap, // must be 0.0 in this custom implementation
  const bool return_softmax,
  c10::optional<at::Generator> gen_,
  int num_splits
) {
  return custom_mha_fwd_template_core<Dtype, HeadDim, CtaM, CtaN, NWarps, false>(
    q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, window_size_left, window_size_right, softcap, return_softmax, gen_, num_splits);
}

template <typename Dtype, int HeadDim, int CtaM, int CtaN, int NWarps>
std::vector<at::Tensor>
custom_mha_fwd_causal(
  at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &alibi_slopes_, // leave empty in this custom implementation
  const float p_dropout, // must be 0.0 in this custom implementation
  const float softmax_scale,
  int window_size_left, // must be -1 in this custom implementation
  int window_size_right, // must be -1 in this custom implementation
  const float softcap, // must be 0.0 in this custom implementation
  const bool return_softmax,
  c10::optional<at::Generator> gen_,
  int num_splits
) {
  return custom_mha_fwd_template_core<Dtype, HeadDim, CtaM, CtaN, NWarps, true>(
    q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, window_size_left, window_size_right, softcap, return_softmax, gen_, num_splits);
}

template <typename Dtype, int HeadDim, int CtaM, int CtaN, int NWarps>
std::vector<at::Tensor>
custom_mha_fwd(
  at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
  c10::optional<at::Tensor> &alibi_slopes_, // leave empty in this custom implementation
  const float p_dropout, // must be 0.0 in this custom implementation
  const float softmax_scale,
  bool is_causal,
  int window_size_left, // must be -1 in this custom implementation
  int window_size_right, // must be -1 in this custom implementation
  const float softcap, // must be 0.0 in this custom implementation
  const bool return_softmax,
  c10::optional<at::Generator> gen_,
  int num_splits
) {

  if (is_causal) {
    return custom_mha_fwd_causal<Dtype, HeadDim, CtaM, CtaN, NWarps>(
      q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, window_size_left, window_size_right, softcap, return_softmax, gen_, num_splits);
  } else {
    return custom_mha_fwd_noncausal<Dtype, HeadDim, CtaM, CtaN, NWarps>(
      q, k, v, out_, alibi_slopes_, p_dropout, softmax_scale, window_size_left, window_size_right, softcap, return_softmax, gen_, num_splits);
  }

}
