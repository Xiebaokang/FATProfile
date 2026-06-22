#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import sys
import os
import pathlib
import subprocess
import time
from enum import Enum
import torch
import json

_THIS_SCRIPT_DIR = pathlib.Path(__file__).parent.resolve().expanduser().absolute()
sys.path.append(str(_THIS_SCRIPT_DIR))

import tensor_gen_namescheme as namescheme
import tensor_compare as tcompare
from runner_json import SDPAVariant, SDPACompareResultEntry, PrecisionTestResult

# the python venv for running tensor_gen.py
TENSOR_GEN_VENV_DIR = pathlib.Path("/home/david/workspace/PPoPP26-FlashAttentionTensor/extra_benchmarks/precision/.venv")
# the python venv for running fp64 math sdpa
FP64_MATH_SDPA_VENV_DIR = pathlib.Path("/home/david/workspace/PPoPP26-FlashAttentionTensor/extra_benchmarks/precision/.venv")
# the python venv for running fp16 math sdpa
FP16_MATH_SDPA_VENV_DIR = pathlib.Path("/home/david/workspace/PPoPP26-FlashAttentionTensor/extra_benchmarks/precision/.venv")
# the python venv for running fp16 flash attention sdpa
FP16_FA_SDPA_VENV_DIR = pathlib.Path("/home/david/workspace/PPoPP26-FlashAttentionTensor/extra_benchmarks/precision/.venv")
# the python venv for running fp16 fullsimt attention sdpa
FP16_FULLSIMT_SDPA_VENV_DIR = None
# the python venv for running fp16 fullmma attention sdpa
FP16_FULLMMA_SDPA_VENV_DIR = pathlib.Path("/home/david/workspace/fa2-accuracy/pytorch-v270-custom-flash-attn2/.venv-py311-torchv270-07b2-attempt")
# the python venv for running fp16 ilph attention sdpa
FP16_ILPH_SDPA_VENV_DIR = None
# the python venv for running fp16 ilpv attention sdpa
FP16_ILPV_SDPA_VENV_DIR = None

SDPA_DICT = {
    SDPAVariant.FP64_MATH: (FP64_MATH_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp64_math_sdpa.py"),
    SDPAVariant.FP16_MATH: (FP16_MATH_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_math_sdpa.py"),
    SDPAVariant.FP16_FA: (FP16_FA_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_fa_sdpa.py"),
    # SDPAVariant.FP16_FULLSIMT: (FP16_FULLSIMT_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_fa_sdpa.py"),
    SDPAVariant.FP16_FULLMMA: (FP16_FULLMMA_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_fa_sdpa.py"),
    # SDPAVariant.FP16_ILPH: (FP16_ILPH_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_fa_sdpa.py"),
    # SDPAVariant.FP16_ILPV: (FP16_ILPV_SDPA_VENV_DIR, _THIS_SCRIPT_DIR / "run_fp16_fa_sdpa.py"),
}

# list of (ref_sdpa_variant, target_sdpa_variant)
ORIG_COMPARE_LIST = [
    (SDPAVariant.FP64_MATH, SDPAVariant.FP16_MATH),
    (SDPAVariant.FP64_MATH, SDPAVariant.FP16_FA),
    (SDPAVariant.FP64_MATH, SDPAVariant.FP16_FULLMMA)
]

OUR_COMPARE_LIST = [
    (SDPAVariant.FP16_FA, SDPAVariant.FP16_FULLMMA)
]


COMPARE_LIST = ORIG_COMPARE_LIST + OUR_COMPARE_LIST

def get_venv_activate_script(venv_dir: pathlib.Path) -> pathlib.Path:
    activate_script = venv_dir / "bin" / "activate"
    if not activate_script.exists():
        raise FileNotFoundError(f"Venv activate script not found: {activate_script}")
    return activate_script

def call_tensor_gen(
    tensorgen_dtype_str: str,
    batchsize: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    base_variance: float,
    outlier_variance: float,
    tensor_save_dir: pathlib.Path,
    seed: int
):
    venv_activate_script = get_venv_activate_script(TENSOR_GEN_VENV_DIR)
    command_activate = f"source {venv_activate_script}"
    command_tensor_gen = f"python {str(_THIS_SCRIPT_DIR / 'tensor_gen.py')}"
    command_tensor_gen_args = [
        tensorgen_dtype_str,
        str(batchsize),
        str(seqlen),
        str(nheads),
        str(headdim),
        str(base_variance),
        str(outlier_variance),
        str(tensor_save_dir),
        str(seed)
    ]
    command_tensor_gen_full = f"{command_activate} && {command_tensor_gen} {' '.join(command_tensor_gen_args)}"
    print(f"--> Running tensor_gen with command: {command_tensor_gen_full}")
    result = subprocess.run(command_tensor_gen_full, shell=True, check=True, executable='/bin/bash')
    if result.returncode != 0:
        raise RuntimeError(f"Tensor generation failed with return code {result.returncode}")
    print("--> Tensor generation completed successfully.")

    q_tensor_filename = namescheme.get_tensor_name(namescheme.TensorType.Q, tensorgen_dtype_str, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    k_tensor_filename = namescheme.get_tensor_name(namescheme.TensorType.K, tensorgen_dtype_str, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    v_tensor_filename = namescheme.get_tensor_name(namescheme.TensorType.V, tensorgen_dtype_str, batchsize, seqlen, nheads, headdim, base_variance, outlier_variance, seed)
    q_tensor_path = tensor_save_dir / f"{q_tensor_filename}.pt"
    k_tensor_path = tensor_save_dir / f"{k_tensor_filename}.pt"
    v_tensor_path = tensor_save_dir / f"{v_tensor_filename}.pt"
    assert q_tensor_path.exists(), f"It seems like Q tensor generation failed: {q_tensor_path}"
    assert k_tensor_path.exists(), f"It seems like K tensor generation failed: {k_tensor_path}"
    assert v_tensor_path.exists(), f"It seems like V tensor generation failed: {v_tensor_path}"

    return (q_tensor_path, k_tensor_path, v_tensor_path)


def get_output_sdpa_tensor_filepath(
    output_dir: pathlib.Path,
    tensorgen_dtype_str: str,
    batchsize: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    base_variance: float,
    outlier_variance: float,
    seed: int,
    sdpa_variant: SDPAVariant
):
    TENSOR_NAME_TEMPLATE = "tensor{}:{}:bs{}:s{}:nh{}:hd{}:bv{}:ov{}:seed{}"
    base_name = TENSOR_NAME_TEMPLATE.format(
        namescheme.TensorType.O.value,
        ("tensorgen{}".format(tensorgen_dtype_str)),
        batchsize,
        seqlen,
        nheads,
        headdim,
        int(base_variance * 100),  # Convert to integer percentage
        int(outlier_variance * 100),  # Convert to integer percentage
        seed
    )
    prefix_sdpa_type_str = "sdpa_{}".format(sdpa_variant.value)
    output_tensor_name = f"{prefix_sdpa_type_str}-{base_name}.pt"
    output_tensor_filepath = output_dir / output_tensor_name
    return output_tensor_filepath

def call_sdpa(
    sdpa_venv_dir: pathlib.Path,
    sdpa_script: pathlib.Path,
    q_tensor_filepath: pathlib.Path,
    k_tensor_filepath: pathlib.Path,
    v_tensor_filepath: pathlib.Path,
    output_tensor_filepath: pathlib.Path,
    sdpa_variant: SDPAVariant
):
    venv_activate_script = get_venv_activate_script(sdpa_venv_dir)

    command_activate = f"source {venv_activate_script}"
    command_sdpa_run = f"python {str(sdpa_script)}"
    command_sdpa_args = [
        str(q_tensor_filepath),
        str(k_tensor_filepath),
        str(v_tensor_filepath),
        str(output_tensor_filepath)
    ]
    command_sdpa_full = f"{command_activate} && {command_sdpa_run} {' '.join(command_sdpa_args)}"
    print(f"--> Running SDPA Variant {sdpa_variant.value} with command: {command_sdpa_full}")
    result = subprocess.run(command_sdpa_full, shell=True, check=True, executable='/bin/bash')
    if result.returncode != 0:
        raise RuntimeError(f"SDPA run failed with return code {result.returncode}")
    print(f"--> SDPA Variant {sdpa_variant.value} completed successfully")
    assert output_tensor_filepath.exists(), f"Output tensor file not found: {output_tensor_filepath}"
    return output_tensor_filepath


def compare_tensors(
    ref_tensor_sdpa_variant: SDPAVariant,
    ref_tensor_filepath: pathlib.Path,
    target_tensor_sdpa_variant: SDPAVariant,
    target_tensor_filepath: pathlib.Path
):
    print(f"Comparing tensors for SDPA variants: REF | {ref_tensor_sdpa_variant.value} vs TARGET | {target_tensor_sdpa_variant.value}")
    ref_tensor = torch.load(ref_tensor_filepath)
    target_tensor = torch.load(target_tensor_filepath)

    assert ref_tensor.shape == target_tensor.shape, "Tensors must have the same shape"

    mse = tcompare.calc_mse(ref_tensor, target_tensor)
    rmse = tcompare.calc_rmse(ref_tensor, target_tensor)

    print(f"Mean Squared Error (MSE) : {mse:.10f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.10f}")
    return mse, rmse
    


def main():
    argparser = argparse.ArgumentParser(description="Run precision test")
    argparser.add_argument(
        "tensorgen_dtype",
        type=str,
        choices=["fp16", "fp64"],
    )
    argparser.add_argument(
        "batchsize",
        type=int,
        help="Batch size for the tensors."
    )
    argparser.add_argument(
        "seqlen",
        type=int,
        help="Sequence length for the tensors."
    )
    argparser.add_argument(
        "nheads",
        type=int,
        help="Number of attention heads."
    )
    argparser.add_argument(
        "headdim",
        type=int,
        help="Dimension of each attention head."
    )
    argparser.add_argument(
        "base_variance",
        type=float,
        help="Base variance for the tensor generation."
    )
    argparser.add_argument(
        "outlier_variance",
        type=float,
        help="Outlier variance for the tensor generation."
    )
    argparser.add_argument(
        "seed",
        type=int,
        help="Random seed for reproducibility."
    )
    
    args = argparser.parse_args()
    tensorgen_dtype_str = args.tensorgen_dtype
    batchsize = args.batchsize
    seqlen = args.seqlen
    nheads = args.nheads
    headdim = args.headdim
    base_variance = args.base_variance
    outlier_variance = args.outlier_variance
    seed = args.seed

    cur_time_string = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    tensorgen_save_dir = _THIS_SCRIPT_DIR / "tensors.d" / cur_time_string
    tensor_output_dir = _THIS_SCRIPT_DIR / "tensors.d" / cur_time_string
    json_output_dir = _THIS_SCRIPT_DIR / "tensors.d" / cur_time_string
    json_output_filepath = json_output_dir / "sdpa_precision_results.json"

    input_tensor_pathes = call_tensor_gen(
        tensorgen_dtype_str,
        batchsize,
        seqlen,
        nheads,
        headdim,
        base_variance,
        outlier_variance,
        tensorgen_save_dir,
        seed
    )
    q_tensor_filepath, k_tensor_filepath, v_tensor_filepath = input_tensor_pathes

    sdpa_variants_output_tensor_filepath_dict = {}

    for sdpa_variant, dictentry in SDPA_DICT.items():
        output_tensor_filepath = get_output_sdpa_tensor_filepath(
            tensor_output_dir,
            tensorgen_dtype_str,
            batchsize,
            seqlen,
            nheads,
            headdim,
            base_variance,
            outlier_variance,
            seed,
            sdpa_variant
        )

        # Ensure the output directory exists
        output_tensor_filepath.parent.mkdir(parents=True, exist_ok=True)

        sdpa_venv_dir, sdpa_script = dictentry
        if not sdpa_script.exists():
            raise FileNotFoundError(f"SDPA script not found: {sdpa_script}")
        # Call the SDPA function
        _ret = call_sdpa(
            sdpa_venv_dir,
            sdpa_script,
            q_tensor_filepath,
            k_tensor_filepath,
            v_tensor_filepath,
            output_tensor_filepath,
            sdpa_variant
        )
        assert _ret == output_tensor_filepath, f"SDPA output tensor file mismatch: {_ret} != {output_tensor_filepath}"
        sdpa_variants_output_tensor_filepath_dict[sdpa_variant] = output_tensor_filepath

    compare_result_entires = []

    for compare_tuple in COMPARE_LIST:
        ref_sdpa_variant, target_sdpa_variant = compare_tuple
        ref_tensor_filepath = sdpa_variants_output_tensor_filepath_dict[ref_sdpa_variant]
        target_tensor_filepath = sdpa_variants_output_tensor_filepath_dict[target_sdpa_variant]

        print(f"------------BEGIN------------")
        print(f"==> Run mode summary: tensorgen dtype={tensorgen_dtype_str}, batchsize={batchsize}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}, base_variance={base_variance}, outlier_variance={outlier_variance}, seed={seed}")
        mse,rmse = compare_tensors(ref_sdpa_variant, ref_tensor_filepath, target_sdpa_variant, target_tensor_filepath)
        print(f"------------END------------")

        compare_result_entires.append(
            SDPACompareResultEntry(
                ref_method=ref_sdpa_variant,
                target_method=target_sdpa_variant,
                mse=mse,
                rmse=rmse
            )
        )

    precision_result = PrecisionTestResult(
        tensorgen_dtype=tensorgen_dtype_str,
        batchsize=batchsize,
        seqlen=seqlen,
        nheads=nheads,
        headdim=headdim,
        base_variance=base_variance,
        outlier_variance=outlier_variance,
        seed=seed,
        compare_results=compare_result_entires
    )
    with open(json_output_filepath, 'w') as json_file:
        json.dump(precision_result.to_json(), json_file, indent=2)



if __name__ == "__main__":
    main()