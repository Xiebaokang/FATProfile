#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
import parse

TENSOR_NAME_TEMPLATE = "tensor{}:{}:bs{}:s{}:nh{}:hd{}:bv{}:ov{}:seed{}"

class TensorType(Enum):
    Q = "Q"
    K = "K"
    V = "V"
    O = "O"

def get_tensor_name(
        tensor_type: TensorType,
        dtype: str,
        batchsize: int,
        seqlen: int,
        nheads: int,
        headdim: int,
        base_variance: float,
        outlier_variance: float,
        seed: int
    ):
    base_variance = int(base_variance * 100)
    outlier_variance = int(outlier_variance * 100)
    assert dtype == "fp16" or dtype == "fp64", "dtype must be either 'fp16' or 'fp64'"
    tensor_name = TENSOR_NAME_TEMPLATE.format(
        tensor_type.value,
        dtype,
        batchsize,
        seqlen,
        nheads,
        headdim,
        base_variance,
        outlier_variance,
        seed
    )
    return tensor_name
        
def parse_tensor_name(tensor_name: str):
    r = parse.parse(TENSOR_NAME_TEMPLATE, tensor_name)

    return {
        "type": TensorType(r[0]),
        "dtype": r[1],
        "batchsize": int(r[2]),
        "seqlen": int(r[3]),
        "nheads": int(r[4]),
        "headdim": int(r[5]),
        "base_variance": float(r[6]) / 100.0,
        "outlier_variance": float(r[7]) / 100.0,
        "seed": int(r[8])
    }

if __name__ == "__main__":
    # test round-trip conversion
    tensor_name = get_tensor_name(
        TensorType.Q,
        "fp16",
        4,
        4096,
        16,
        128,
        10.0,
        100.0,
        42
    )
    print(f"Generated tensor name: {tensor_name}")
    parsed = parse_tensor_name(tensor_name)
    print(f"Parsed tensor name: {parsed}")
    assert parsed["type"] == TensorType.Q
    assert parsed["dtype"] == "fp16"
    assert parsed["batchsize"] == 4
    assert parsed["seqlen"] == 4096
    assert parsed["nheads"] == 16
    assert parsed["headdim"] == 128
    assert parsed["base_variance"] == 10.0
    assert parsed["outlier_variance"] == 100.0
    assert parsed["seed"] == 42
    print("Round-trip conversion successful!")