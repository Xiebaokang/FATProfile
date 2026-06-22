import argparse

import torch
import flashinfer


DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
}


def parse_bool(value):
    value = value.strip().lower()
    if value in ("1", "true", "t", "yes", "y"):
        return True
    if value in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one FlashInfer prefill attention shape and mark one call for ncu."
    )
    parser.add_argument("--B", "--batch-size", dest="B", type=int, default=1)
    parser.add_argument("--H", "--num-heads", dest="H", type=int, default=8)
    parser.add_argument("--S", "--seq-len", dest="S", type=int, default=1024)
    parser.add_argument("--D", "--head-dim", dest="D", type=int, default=256)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float16")
    parser.add_argument("--causal", type=parse_bool, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--workspace-mb", type=int, default=128)
    return parser


def rand_tensor(shape, device, dtype):
    return torch.randn(shape, device=device, dtype=dtype)


def profile_once(name, fn):
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(name)
    try:
        return fn()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


def main():
    args = build_parser().parse_args()

    dtype = DTYPES[args.dtype]
    device = "cuda"

    B, H, S, D = args.B, args.H, args.S, args.D

    # FlashInfer ragged prefill 常用布局:
    # q/k/v: [total_tokens, num_heads, head_dim]
    # 这里 total_tokens = B * S
    q = rand_tensor((B * S, H, D), device=device, dtype=dtype)
    k = rand_tensor((B * S, H, D), device=device, dtype=dtype)
    v = rand_tensor((B * S, H, D), device=device, dtype=dtype)

    # 每个 batch 的 query 和 kv 长度都为 S
    # qo_indptr / kv_indptr 长度为 B + 1
    indptr_cpu = torch.arange(
        0,
        (B + 1) * S,
        S,
        dtype=torch.int32,
        device=device,
    )

    qo_indptr = indptr_cpu
    kv_indptr = indptr_cpu

    workspace_buffer = torch.empty(
        args.workspace_mb * 1024 * 1024,
        device=device,
        dtype=torch.uint8,
    )

    # "NHD" 表示 KV cache layout: [token, head, dim]
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        "NHD",
    )

    # plan 阶段一般不放到 profiling 区间里；
    # 如果你只想看 attention kernel，本脚本只标记 run。
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        H,      # num_qo_heads
        H,      # num_kv_heads
        D,      # head_dim
        causal=args.causal,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=torch.float16,
    )

    for _ in range(args.warmup):
        out = wrapper.run(q, k, v)

    torch.cuda.synchronize()

    out = profile_once("flashinfer_profile", lambda: wrapper.run(q, k, v))
    out = out.float()
    torch.cuda.synchronize()

    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
