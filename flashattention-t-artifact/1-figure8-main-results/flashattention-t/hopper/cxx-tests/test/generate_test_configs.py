#!/usr/bin/env python3
"""Generate FA3 test sources and compile them into executables."""

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "test.cu"
DEFAULT_OUTPUT = HERE / "generated_cu"
CXX_TESTS_DIR = HERE.parent
OLD_CMAKE_CACHE = CXX_TESTS_DIR / "fwd_bench" / "build" / "CMakeCache.txt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", nargs=4, type=int, required=True,
                        metavar=("B", "H", "S", "D"))
    parser.add_argument("--num-consumer", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp8", "fp16"), required=True)
    parser.add_argument("--causal", action="store_true")

    # Pass more than one value to sweep that field. Defaults generate one file.
    parser.add_argument("--block-m", nargs="+", type=int, default=[128])
    parser.add_argument("--block-n", nargs="+", type=int, default=[128])
    parser.add_argument("--stage", nargs="+", type=int, default=[2])
    parser.add_argument("--producer-reg", nargs="+", type=int, default=[24])
    parser.add_argument("--consumer-reg", nargs="+", type=int, default=[240])
    parser.add_argument("--p-smem-k-tiles", nargs="+", type=int, default=[0])
    parser.add_argument("--q-reg-k-tiles", nargs="+", type=int, default=[0])

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arch", default="90a")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cmake", type=Path)
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--no-build", action="store_true")
    return parser.parse_args()


def make_configs(args):
    names = (
        "kBlockM", "kBlockN", "kStage", "producer_reg_dealloc",
        "consumer_reg_alloc", "p_smem_k_tiles", "q_reg_k_tiles",
    )
    values = (
        args.block_m, args.block_n, args.stage, args.producer_reg,
        args.consumer_reg, args.p_smem_k_tiles, args.q_reg_k_tiles,
    )
    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(names, combination))
        config["num_consumer"] = args.num_consumer
        configs.append(config)
    return configs


def replace_one(text, pattern, replacement, name):
    text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"cannot replace {name} in test.cu")
    return text


def render(template, config, shape, dtype, causal):
    source = template

    match = re.search(r"struct\s+TConfig\s*\{.*?\};", source, re.DOTALL)
    if match is None:
        raise RuntimeError("cannot find TConfig in test.cu")
    struct = match.group(0)
    for name, value in config.items():
        struct = replace_one(
            struct,
            rf"(\bint\s+{name}\s*=\s*)-?\d+(\s*;)",
            rf"\g<1>{value}\g<2>",
            name,
        )
    source = source[:match.start()] + struct + source[match.end():]

    batch, heads, seqlen, head_dim = shape
    constants = {
        "kBatch": batch,
        "kSeqlen": seqlen,
        "kNumHeads": heads,
        "kHeadDim": head_dim,
    }
    for name, value in constants.items():
        source = replace_one(
            source,
            rf"(constexpr\s+int\s+{name}\s*=\s*)\d+(\s*;)",
            rf"\g<1>{value}\g<2>",
            name,
        )

    calls = {
        ("fp8", False): 'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, false>("FP8");',
        ("fp8", True): 'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, true>("FP8");',
        ("fp16", False): 'benchmark<cutlass::half_t, at::kHalf, false>("FP16");',
        ("fp16", True): 'benchmark<cutlass::half_t, at::kHalf, true>("FP16");',
    }
    main = re.search(r"int\s+main\s*\(\s*\)\s*\{.*?^\}", source,
                     re.DOTALL | re.MULTILINE)
    if main is None:
        raise RuntimeError("cannot find main() in test.cu")
    main_text = re.sub(r"^\s*benchmark<.*?\);\s*$", "", main.group(0),
                       flags=re.MULTILINE)
    main_text = main_text.replace("  return 0;", f"  {calls[(dtype, causal)]}\n\n  return 0;")
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)
    return source[:main.start()] + main_text + source[main.end():]


def source_name(config, shape, dtype, causal):
    batch, heads, seqlen, head_dim = shape
    mode = "causal" if causal else "noncausal"
    return (
        f"test_b{batch}_h{heads}_s{seqlen}_d{head_dim}_{dtype}_{mode}"
        f"_bm{config['kBlockM']}_bn{config['kBlockN']}_st{config['kStage']}"
        f"_prd{config['producer_reg_dealloc']}_cra{config['consumer_reg_alloc']}"
        f"_p{config['p_smem_k_tiles']}_q{config['q_reg_k_tiles']}"
        f"_nc{config['num_consumer']}.cu"
    )


def cache_value(name):
    if not OLD_CMAKE_CACHE.exists():
        return None
    match = re.search(rf"^{name}(?::[^=]+)?=(.+)$",
                      OLD_CMAKE_CACHE.read_text(), re.MULTILINE)
    return match.group(1).strip() if match else None


def find_cmake(explicit):
    if explicit:
        return str(explicit.resolve())
    command = shutil.which("cmake") or cache_value("CMAKE_COMMAND")
    if not command:
        raise RuntimeError("cmake not found; activate the torch environment or pass --cmake")
    return command


def find_venv(explicit):
    candidates = [explicit] if explicit else [
        os.environ.get("VIRTUAL_ENV"),
        os.environ.get("CONDA_PREFIX"),
        cache_value("FA_VENV_DIR"),
    ]
    for value in candidates:
        if value:
            path = Path(value).resolve()
            torch_cmake = list(path.glob("lib/python*/site-packages/torch/share/cmake/Torch"))
            if torch_cmake:
                return path
    raise RuntimeError("PyTorch environment not found; pass --venv")


def write_cmake(output_dir, sources):
    cmake_utils = (CXX_TESTS_DIR / "cmake").resolve().as_posix()
    source_lines = "\n".join(
        f'add_single_source_executable("{source.name}")' for source in sources
    )
    content = f"""cmake_minimum_required(VERSION 3.26)
set(PROJ_NAME generated_fa3_tests)
set(CMAKE_UTILS_DIR "{cmake_utils}")
include(${{CMAKE_UTILS_DIR}}/cmakePrologue.cmake)
include(${{CMAKE_UTILS_DIR}}/utilFuncs.cmake)
project(${{PROJ_NAME}} LANGUAGES CXX CUDA)
include(${{CMAKE_UTILS_DIR}}/flashVenv.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")

{source_lines}
"""
    (output_dir / "CMakeLists.txt").write_text(content)


def build_all(args, output_dir):
    cmake = find_cmake(args.cmake)
    venv = find_venv(args.venv)
    build_dir = output_dir / "build"
    configure = [
        cmake, "-S", str(output_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
        f"-DFA_VENV_DIR={venv}",
    ]
    build = [cmake, "--build", str(build_dir), "--parallel", str(args.jobs)]
    print("+", " ".join(configure), flush=True)
    subprocess.run(configure, check=True)
    print("+", " ".join(build), flush=True)
    subprocess.run(build, check=True)
    print(f"executables: {build_dir / 'bin'}")


def main():
    args = parse_args()
    if any(value <= 0 for value in (*args.shape, args.num_consumer, args.jobs)):
        raise ValueError("shape, num-consumer and jobs must be positive")

    configs = make_configs(args)
    template = TEMPLATE.read_text()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = []
    manifest = []
    for config in configs:
        path = output_dir / source_name(config, args.shape, args.dtype, args.causal)
        path.write_text(render(template, config, args.shape, args.dtype, args.causal))
        sources.append(path)
        manifest.append({"source": path.name, "executable": path.stem, "config": config})

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_cmake(output_dir, sources)
    print(f"generated {len(sources)} CUDA sources in {output_dir}")

    if not args.no_build:
        build_all(args, output_dir)


if __name__ == "__main__":
    main()
