import argparse

import torch
from torch.nn.attention.flex_attention import flex_attention


DTYPES = {
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
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
        description="Run one FlexAttention shape and optionally mark one call for ncu."
    )
    parser.add_argument("--B", "--batch-size", dest="B", type=int, default=1)
    parser.add_argument("--H", "--num-heads", dest="H", type=int, default=8)
    parser.add_argument("--S", "--seq-len", dest="S", type=int, default=1024)
    parser.add_argument("--D", "--head-dim", dest="D", type=int, default=128)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--causal", type=parse_bool, default=False)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compile", type=parse_bool, default=True)
    return parser


def causal_score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(q_idx >= kv_idx, score, -float("inf"))


def rand_tensor(shape, device, dtype):
    if dtype == torch.float8_e4m3fn:
        return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
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

    if dtype == torch.float8_e4m3fn and args.D % 16 != 0:
        raise ValueError("FP8 FlexAttention requires --D/--head-dim to be a multiple of 16")

    # FlexAttention 常用布局: [B, H, S, D]
    q = rand_tensor((args.B, args.H, args.S, args.D), device=device, dtype=dtype)
    k = rand_tensor((args.B, args.H, args.S, args.D), device=device, dtype=dtype)
    v = rand_tensor((args.B, args.H, args.S, args.D), device=device, dtype=dtype)

    score_mod = causal_score_mod if args.causal else None

    if args.compile:
        flex_attn_func = torch.compile(flex_attention, dynamic=False)
    else:
        flex_attn_func = flex_attention

    # warmup：第一次会触发 torch.compile / triton 编译，建议 ncu 前多预热
    for _ in range(args.warmup):
        out = flex_attn_func(q, k, v, score_mod=score_mod)

    torch.cuda.synchronize()

    out = profile_once(
        "flex_attention_profile",
        lambda: flex_attn_func(q, k, v, score_mod=score_mod),
    )
    # out = out.float()
    torch.cuda.synchronize()

    # 防止极端情况下被认为 out 未使用
    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
