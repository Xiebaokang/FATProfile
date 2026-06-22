#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import argparse
import os
import sys
import pathlib

def calc_mse(ref_tensor, target_tensor):
    ref_tensor_fp64 = ref_tensor.to(torch.float64)
    target_tensor_fp64 = target_tensor.to(torch.float64)
    mse = torch.mean((ref_tensor_fp64 - target_tensor_fp64) ** 2)
    return mse.item()

def calc_rmse(ref_tensor, target_tensor):
    mse = calc_mse(ref_tensor, target_tensor)
    rmse = mse ** 0.5
    return rmse

def main():
    argparser = argparse.ArgumentParser(description="calculate error metrics for tensor files")
    argparser.add_argument(
        "ref_tensor",
        type=str,
        help="Reference tensor file (.pt) to compare against."
    )
    argparser.add_argument(
        "target_tensor",
        type=str,
        help="Target tensor file (.pt) to compare with the reference tensor."
    )
    args = argparser.parse_args()
    
    ref_tensor_path = pathlib.Path(args.ref_tensor).resolve().expanduser().absolute()
    target_tensor_path = pathlib.Path(args.target_tensor).resolve().expanduser().absolute()

    ref_tensor = torch.load(ref_tensor_path)
    target_tensor = torch.load(target_tensor_path)
    
    assert ref_tensor.shape == target_tensor.shape, "Tensors must have the same shape"

    mse = calc_mse(ref_tensor, target_tensor)
    rmse = calc_rmse(ref_tensor, target_tensor)

    print(f"Mean Squared Error (MSE): {mse:.10f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.10f}")
    
if __name__ == "__main__":
    main()