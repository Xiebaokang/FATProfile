from __future__ import annotations

from pathlib import Path
from typing import Literal

from run_utils import BenchResult, run
from solver import Mode, mix_wgmma_solve


Config = dict[str, int]
Shape = tuple[int, int, int, int]
DType = Literal["fp8", "fp16"]


def get_configs(
    shape: Shape,
    dtype: DType,
    causal: bool,
    bn_limit_n: int = 0,
    smem_limit: int = 232_448,
    reg_limit: int = 65_536,
    num_consumer_limit: int = 3,
    buf_use_rate: float = 0.8,
    mode: Mode = "radical",
) -> list[Config]:
    """Generate all stage-1/2/3 configs accepted by the FA3 solver."""
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("shape must be a positive (B, H, S, D) tuple")
    if dtype not in ("fp8", "fp16"):
        raise ValueError("dtype must be 'fp8' or 'fp16'")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")

    _, _, _, head_dim = shape
    elem_width = 1 if dtype == "fp8" else 2
    configs: list[Config] = []

    for stage in range(1, 4):
        configs.extend(
            mix_wgmma_solve(
                HD=head_dim,
                stage=stage,
                elem_width=elem_width,
                bn_limit_n=bn_limit_n,
                smem_limit=smem_limit,
                reg_limit=reg_limit,
                num_consumer_limit=num_consumer_limit,
                buf_use_rate=buf_use_rate,
                causal=causal,
                mode=mode,
            )
        )

    return configs


def tune(
    shape: Shape,
    dtype: DType,
    causal: bool,
    bn_limit_n: int = 0,
    smem_limit: int = 232_448,
    reg_limit: int = 65_536,
    num_consumer_limit: int = 3,
    buf_use_rate: float = 0.3,
    arch: str = "90a",
    jobs: int = 16,
    rank: int = 15,
    result_dir: str | Path | None = None,
    coarse_register_usage_level: int = 5,
    final_register_usage_level: int = 10,
    structure_top_n: int = 3,
    benchmark_timeout_seconds: float = 120.0,
    result_tag: str | None = None,
    mode: Mode = "radical",
) -> list[BenchResult]:
    """Generate configs and tune them with coarse and final compile passes."""
    configs = get_configs(
        shape=shape,
        dtype=dtype,
        causal=causal,
        bn_limit_n=bn_limit_n,
        smem_limit=smem_limit,
        reg_limit=reg_limit,
        num_consumer_limit=num_consumer_limit,
        buf_use_rate=buf_use_rate,
        mode=mode,
    )
    if not configs:
        raise ValueError("no config satisfies the solver constraints")
    
    configs = [
        cfg for cfg in configs
        # if cfg["kBlockM"] == 128
        # and cfg["kBlockN"] == 80
        # and cfg["num_consumer"] == 2
        # and cfg["p_smem_k_tiles"] == 0
        # and cfg["q_reg_k_tiles"] == 0
        if cfg["p_smem_k_tiles"] != 0
        or cfg["q_reg_k_tiles"] != 0
    ]

    print(f"generated {len(configs)} configs")
    return run(
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=configs,
        arch=arch,
        jobs=jobs,
        rank=rank,
        result_dir=result_dir,
        coarse_register_usage_level=coarse_register_usage_level,
        final_register_usage_level=final_register_usage_level,
        structure_top_n=structure_top_n,
        benchmark_timeout_seconds=benchmark_timeout_seconds,
        result_tag=result_tag,
    )

if __name__ == "__main__":
    shapes = [(1, 16, 32768, 256)]
    for shape in shapes:
        tune(
            shape=shape, 
            dtype="fp16", 
            causal=False, 
            bn_limit_n=0, 
        )
