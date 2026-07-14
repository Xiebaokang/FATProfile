import argparse

import torch
import flash_attn_interface


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
        description="Run one FlashAttention-3 shape and optionally mark one call for ncu."
    )
    parser.add_argument("--B", "--batch-size", dest="B", type=int, default=1)
    parser.add_argument("--H", "--num-heads", dest="H", type=int, default=16)
    parser.add_argument("--S", "--seq-len", dest="S", type=int, default=114*160*2)
    parser.add_argument("--D", "--head-dim", dest="D", type=int, default=64)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--causal", type=parse_bool, default=False)
    parser.add_argument("--warmup", type=int, default=10)
    return parser


def rand_tensor(shape, device, dtype):
    if dtype == torch.float8_e4m3fn:
        return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def profile_once(fn):
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("fa3_profile")
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
        raise ValueError("FP8 FA3 requires --D/--head-dim to be a multiple of 16")

    shape = (args.B, args.S, args.H, args.D)
    q = rand_tensor(shape, device=device, dtype=dtype)
    k = rand_tensor(shape, device=device, dtype=dtype)
    v = rand_tensor(shape, device=device, dtype=dtype)

    for _ in range(args.warmup):
        flash_attn_interface.flash_attn_func(q, k, v, causal=args.causal)
    torch.cuda.synchronize()

    out = profile_once(lambda: flash_attn_interface.flash_attn_func(q, k, v, causal=args.causal))
    # out = out.float()
    torch.cuda.synchronize()

    print(out.shape, out.dtype)


if __name__ == "__main__":
    main()
