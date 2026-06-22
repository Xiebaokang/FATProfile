#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
import sys
import pathlib
from torch.nn.attention import sdpa_kernel, SDPBackend

def run_fp64_math_sdpa(q_tensor, k_tensor, v_tensor):
    q_tensor_fp64 = q_tensor.to(torch.float64)
    k_tensor_fp64 = k_tensor.to(torch.float64)
    v_tensor_fp64 = v_tensor.to(torch.float64)

    (batchsize, nheads, seqlen, headdim) = q_tensor_fp64.shape

    with sdpa_kernel(SDPBackend.MATH):
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            q_tensor_fp64,
            k_tensor_fp64,
            v_tensor_fp64,
            dropout_p=0.0,  # No dropout for precision testing
            is_causal=False,  # Not causal for this benchmark
            scale=None # Let torch handle scaling
        )
    
    return attention_output

def main():
    argparser = argparse.ArgumentParser(description="Run SDPA with FP64 math backend")
    argparser.add_argument(
        "q_tensor_file",
        type=str,
        help="Path to the Q tensor file (.pt)."
    )
    argparser.add_argument(
        "k_tensor_file",
        type=str,
        help="Path to the K tensor file (.pt)."
    )
    argparser.add_argument(
        "v_tensor_file",
        type=str,
        help="Path to the V tensor file (.pt)."
    )
    argparser.add_argument(
        "output_tensor_file",
        type=str,
        help="Path to save the output tensor file (.pt)."
    )
    args = argparser.parse_args()
    q_tensor_path = pathlib.Path(args.q_tensor_file).resolve().expanduser().absolute()
    k_tensor_path = pathlib.Path(args.k_tensor_file).resolve().expanduser().absolute()
    v_tensor_path = pathlib.Path(args.v_tensor_file).resolve().expanduser().absolute()
    output_tensor_path = pathlib.Path(args.output_tensor_file).resolve().expanduser().absolute()

    # Load tensors
    q_tensor = torch.load(q_tensor_path)
    k_tensor = torch.load(k_tensor_path)
    v_tensor = torch.load(v_tensor_path)

    assert q_tensor.shape == k_tensor.shape == v_tensor.shape, "All tensors must have the same shape"
    assert len(q_tensor.shape) == 4, "Tensors must be 4D (batchsize, nheads, seqlen, headdim)"
    assert len(k_tensor.shape) == 4, "Tensors must be 4D (batchsize, nheads, seqlen, headdim)"
    assert len(v_tensor.shape) == 4, "Tensors must be 4D (batchsize, nheads, seqlen, headdim)"

    output_tensor = run_fp64_math_sdpa(q_tensor, k_tensor, v_tensor)
    torch.save(output_tensor, output_tensor_path)
    print(f"Output tensor saved to {output_tensor_path}")

if __name__ == "__main__":
    main()