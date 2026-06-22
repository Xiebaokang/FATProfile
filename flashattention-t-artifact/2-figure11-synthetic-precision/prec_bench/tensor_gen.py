#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import sys
import argparse
import pathlib

_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().absolute()
sys.path.append(str(_THIS_SCRIPT_DIR))

import tensor_gen_namescheme as namescheme


def str_dtype_to_torch_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "fp64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Use 'fp16' or 'fp64'.")
    
def torch_dtype_to_str_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    elif dtype == torch.float64:
        return "fp64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}. Use torch.float16 or torch.float64.")


OUTLIER_BERNOULLI_PROB = 0.001 # a thousandth of the values are outliers

# NOTE: this function expects that the random seed has been set before calling it
# NOTE: torch sdpa expects tensor in (batchsize, nheads, seqlen, headdim) shape
# NOTE: so we generate tensor in such shape and also store them in such shape
def gen_tensor(
    tensor_type: namescheme.TensorType, 
    dtype: torch.dtype,
    batchsize: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    base_variance: float,
    outlier_variance: float,
):
    base_std = base_variance ** 0.5
    outlier_std = outlier_variance ** 0.5
    shape = (batchsize, nheads, seqlen, headdim)
    base_tensor = torch.randn(shape, dtype=dtype) * base_std
    outliers = torch.randn(shape, dtype=dtype) * outlier_std
    mask = torch.bernoulli(torch.full(shape, OUTLIER_BERNOULLI_PROB)).to(torch.bool)
    final_tensor = base_tensor + outliers * mask
    return final_tensor
    
    
    

def gen_qkv_tensors(
    dtype: torch.dtype,
    batchsize: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    base_variance: float,
    outlier_variance: float,
    tensor_save_dir: pathlib.Path,
    seed: int
):
    torch.manual_seed(seed)
    q_tensor = gen_tensor(namescheme.TensorType.Q, dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance)
    k_tensor = gen_tensor(namescheme.TensorType.K, dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance)
    v_tensor = gen_tensor(namescheme.TensorType.V, dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance)
    str_dtype = torch_dtype_to_str_dtype(dtype)
    q_tensor_name = namescheme.get_tensor_name(namescheme.TensorType.Q, str_dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    k_tensor_name = namescheme.get_tensor_name(namescheme.TensorType.K, str_dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    v_tensor_name = namescheme.get_tensor_name(namescheme.TensorType.V, str_dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    # save tensors to .pt files
    if not tensor_save_dir.exists():
        tensor_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(q_tensor, tensor_save_dir / f"{q_tensor_name}.pt")
    torch.save(k_tensor, tensor_save_dir / f"{k_tensor_name}.pt")
    torch.save(v_tensor, tensor_save_dir / f"{v_tensor_name}.pt")


def main():
    argparser = argparse.ArgumentParser(description="tensorgen")
    argparser.add_argument(
        "dtype",
        choices=["fp16", "fp64"]
    )
    argparser.add_argument(
        "batchsize",
        type=int,
        default=4,
        nargs="?"
    )
    argparser.add_argument(
        "seqlen",
        type=int,
        default=4096,
        nargs="?"
    )
    argparser.add_argument(
        "nheads",
        type=int,
        default=16,
        nargs="?"
    )
    argparser.add_argument(
        "headdim",
        type=int,
        default=128,
        nargs="?"
    )
    argparser.add_argument(
        "base_variance",
        type=float,
        default=1,
        nargs="?"
    )
    argparser.add_argument(
        "outlier_variance",
        type=float,
        default=100,
        nargs="?"
    )
    argparser.add_argument(
        "tensor_save_dir",
        nargs="?",
        type=pathlib.Path,
        default=_THIS_SCRIPT_DIR / "tensors.d",
        help="Directory to save generated tensors."
    )
    argparser.add_argument(
        "seed",
        nargs="?",
        type=int,
        default=42
    )

    args = argparser.parse_args()
    dtype = str_dtype_to_torch_dtype(args.dtype)
    batchsize = args.batchsize
    seqlen = args.seqlen
    nheads = args.nheads
    headdim = args.headdim
    base_variance = args.base_variance
    outlier_variance = args.outlier_variance
    tensor_save_dir = args.tensor_save_dir.resolve().expanduser().absolute()
    seed = args.seed
    gen_qkv_tensors(dtype, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, tensor_save_dir, seed)

if __name__ == "__main__":
    main()