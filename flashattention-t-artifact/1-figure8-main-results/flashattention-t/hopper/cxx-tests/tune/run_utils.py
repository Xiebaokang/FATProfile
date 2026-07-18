#!/usr/bin/env python3
"""Generate FA3 test sources and compile them into executables."""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Literal, TypedDict


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "test.cu"
DEFAULT_OUTPUT = HERE / "generated_cu"

Shape = tuple[int, int, int, int]
Config = dict[str, int]
DType = Literal["fp8", "fp16"]
BuildResult = dict[str, Any]
BenchResult = dict[str, Any]
TCONFIG_FIELDS = {
    "kBlockM",
    "kBlockN",
    "kStage",
    "producer_reg_dealloc",
    "consumer_reg_alloc",
    "p_smem_k_tiles",
    "q_reg_k_tiles",
    "num_consumer",
    "use_scheduler_barrier",
}
STRUCTURE_FIELDS = (
    "kBlockM",
    "kBlockN",
    "kStage",
    "p_smem_k_tiles",
    "q_reg_k_tiles",
    "num_consumer",
)


class CompiledEntry(TypedDict):
    shape: list[int]
    dtype: DType
    causal: bool
    config: Config
    executable: str

PTXAS_PERF_LOSS_RE = re.compile(
    r"ptxas\s+info\s*:\s*\(C75\d{2}\).*Potential Performance Loss[^\r\n]*",
    re.IGNORECASE,
)


def replace_one(text: str, patterns: list[str], replacement: str, name: str) -> str:
    for pattern in patterns:
        text, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if count == 1:
            return text
    if count != 1:
        raise RuntimeError(f"cannot replace {name} in test.cu")
    


def render(
    template: str,
    config: Config,
    shape: Shape,
    dtype: DType,
    causal: bool,
) -> str:
    source = template

    match = re.search(r"struct\s+TConfig\s*\{.*?\};", source, re.DOTALL)
    if match is None:
        raise RuntimeError("cannot find TConfig in test.cu")
    struct = match.group(0)
    for name, value in config.items():
        # print(name)
        if name not in TCONFIG_FIELDS:
            continue
        struct = replace_one(
            struct,
            [rf"(\bint\s+{name}\s*=\s*)-?\d+(\s*;)", 
            rf"(\buint32_t\s+{name}\s*=\s*)-?\d+(\s*;)"],
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
            [rf"(constexpr\s+int\s+{name}\s*=\s*)\d+(\s*;)"],
            rf"\g<1>{value}\g<2>",
            name,
        )

    calls = {
        ("fp8", False): 'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, false>("FP8", options);',
        ("fp8", True): 'benchmark<cute::float_e4m3_t, at::kFloat8_e4m3fn, true>("FP8", options);',
        ("fp16", False): 'benchmark<cutlass::half_t, at::kHalf, false>("FP16", options);',
        ("fp16", True): 'benchmark<cutlass::half_t, at::kHalf, true>("FP16", options);',
    }
    main = re.search(r"int\s+main\s*\([^)]*\)\s*\{.*?^\}", source,
                     re.DOTALL | re.MULTILINE)
    if main is None:
        raise RuntimeError("cannot find main() in test.cu")
    main_text = re.sub(r"^\s*benchmark<.*?\);\s*$", "", main.group(0),
                       flags=re.MULTILINE)
    main_text = main_text.replace("  return 0;", f"  {calls[(dtype, causal)]}\n\n  return 0;")
    main_text = re.sub(r"\n{3,}", "\n\n", main_text)
    return source[:main.start()] + main_text + source[main.end():]


def source_name(
    config: Config,
    shape: Shape,
    dtype: DType,
    causal: bool,
    register_usage_level: int,
) -> str:
    batch, heads, seqlen, head_dim = shape
    mode = "causal" if causal else "noncausal"
    return (
        f"test_b{batch}_h{heads}_s{seqlen}_d{head_dim}_{dtype}_{mode}"
        f"_bm{config['kBlockM']}_bn{config['kBlockN']}_st{config['kStage']}"
        f"_prd{config['producer_reg_dealloc']}_cra{config['consumer_reg_alloc']}"
        f"_p{config['p_smem_k_tiles']}_q{config['q_reg_k_tiles']}"
        f"_nc{config['num_consumer']}_sb{config['use_scheduler_barrier']}"
        f"_rl{register_usage_level}.cu"
    )


def write_cmake(
    output_dir: Path,
    sources: list[Path],
    register_usage_level: int,
) -> None:
    source_lines = "\n".join(
        f'add_single_source_executable("{source.name}" {register_usage_level})'
        for source in sources
    )
    content = f"""cmake_minimum_required(VERSION 3.26)
set(PROJ_NAME generated_fa3_tests)
set(CMAKE_UTILS_DIR "${{CMAKE_CURRENT_LIST_DIR}}/../../cmake")
include(${{CMAKE_UTILS_DIR}}/cmakePrologue.cmake)
include(${{CMAKE_UTILS_DIR}}/utilFuncs.cmake)
project(${{PROJ_NAME}} LANGUAGES CXX CUDA)
include(${{CMAKE_UTILS_DIR}}/flashVenv.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/bin")

{source_lines}
"""
    cmake_path = output_dir / "CMakeLists.txt"
    if not cmake_path.exists() or cmake_path.read_text() != content:
        cmake_path.write_text(content)


def get_build_dir() -> Path:
    """Use a path-specific build tree, avoiding stale caches after a rename."""
    cache_path = DEFAULT_OUTPUT / "build" / "CMakeCache.txt"
    if not cache_path.exists():
        return DEFAULT_OUTPUT / "build"

    match = re.search(
        r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$",
        cache_path.read_text(errors="replace"),
        re.MULTILINE,
    )
    if match and Path(match.group(1)).resolve() == DEFAULT_OUTPUT.resolve():
        return DEFAULT_OUTPUT / "build"

    suffix = sha256(str(DEFAULT_OUTPUT.resolve()).encode()).hexdigest()[:8]
    return DEFAULT_OUTPUT / f"build_{suffix}"


def build_all(
    arch: str,
    targets: list[str],
    jobs: int = 2,
    previous_results: dict[str, BuildResult] | None = None,
) -> tuple[dict[str, BuildResult], Path]:
    if jobs <= 0:
        raise ValueError("jobs must be positive")

    build_dir = get_build_dir()
    configure = [
        "cmake", "-S", str(DEFAULT_OUTPUT), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
    ]
    print("+", " ".join(configure), flush=True)
    subprocess.run(configure, check=True)

    # Build the common scheduler first, so its output cannot be mistaken for a
    # warning emitted by one of the generated config kernels.
    common_build = [
        "cmake", "--build", str(build_dir),
        "--target", "fa3_prepare_scheduler",
        "--parallel", "1",
    ]
    print("+", " ".join(common_build), flush=True)
    subprocess.run(common_build, check=True)

    previous_results = previous_results or {}
    bin_dir = build_dir / "bin"

    def build_one(target: str) -> tuple[str, BuildResult, str]:
        executable = bin_dir / target
        old_mtime = executable.stat().st_mtime_ns if executable.exists() else None
        build = [
            "cmake", "--build", str(build_dir),
            "--target", target,
            "--parallel", "1",
        ]
        process = subprocess.run(build, text=True, capture_output=True, check=False)
        output = "\n".join(
            part for part in (process.stdout.strip(), process.stderr.strip()) if part
        )
        warnings = [match.group(0) for match in PTXAS_PERF_LOSS_RE.finditer(output)]
        if process.returncode != 0:
            result = {
                "status": "compile_failed",
                "error": f"cmake build exited with code {process.returncode}",
                "warnings": warnings,
            }
        elif warnings:
            result = {
                "status": "performance_warning",
                "warnings": warnings,
            }
        elif executable.is_file():
            new_mtime = executable.stat().st_mtime_ns
            previous = previous_results.get(target)
            if (
                old_mtime == new_mtime
                and previous
                and previous.get("status") == "performance_warning"
            ):
                result = {
                    "status": "performance_warning",
                    "warnings": previous.get("warnings", []),
                    "cached": True,
                }
            else:
                result = {"status": "success", "warnings": []}
        else:
            result = {
                "status": "compile_failed",
                "error": "build succeeded but executable is missing",
                "warnings": [],
            }
        return target, result, output

    build_results = {}
    worker_count = min(jobs, len(targets))
    print(f"building {len(targets)} configs with {worker_count} parallel jobs")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(build_one, target): target for target in targets}
        completed = 0
        for future in as_completed(futures):
            target, result, output = future.result()
            completed += 1
            print(f"[{completed}/{len(targets)}] {target}", flush=True)
            if output:
                print(output)
            if result["status"] == "compile_failed":
                print(f"  excluded: {result['error']}")
            elif result["status"] == "performance_warning":
                print("  excluded: ptxas potential performance loss warning")
            build_results[target] = result

    return build_results, build_dir


def compile_interface(
    shape: Shape,
    dtype: DType,
    causal: bool,
    configs: Iterable[Config],
    arch: str = "90a",
    jobs: int = 2,
    register_usage_level: int = 10,
) -> list[CompiledEntry]:
    if not 0 <= register_usage_level <= 10:
        raise ValueError("register_usage_level must be in [0, 10]")
    configs = list(configs)
    if not configs:
        raise ValueError("config list is empty")
    template = TEMPLATE.read_text()
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = []
    manifest = []
    generated_count = 0
    for config in configs:
        path = DEFAULT_OUTPUT / source_name(
            config, shape, dtype, causal, register_usage_level
        )
        rendered = render(template, config, shape, dtype, causal)
        if path.exists() and path.read_text() == rendered:
            print(f"cached source: {path.name}")
        else:
            path.write_text(rendered)
            generated_count += 1
        sources.append(path)
        manifest.append({"source": path.name, "executable": path.stem, "config": config})

    (DEFAULT_OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_cmake(DEFAULT_OUTPUT, sources, register_usage_level)
    print(
        f"CUDA sources: {generated_count} generated, "
        f"{len(sources) - generated_count} cached"
    )

    report_path = DEFAULT_OUTPUT / f"compile_report_rl{register_usage_level}.json"
    previous_results = {}
    if report_path.exists():
        for item in json.loads(report_path.read_text()):
            previous_results[item["executable"]] = item

    targets = [source.stem for source in sources]
    build_results, build_dir = build_all(
        arch, targets, jobs, previous_results=previous_results
    )
    bin_dir = build_dir / "bin"

    compiled = []
    for source, config in zip(sources, configs):
        build_result = build_results[source.stem]
        executable = (bin_dir / source.stem).resolve()
        if build_result["status"] == "success" and executable.is_file():
            compiled.append({
                "shape": list(shape),
                "dtype": dtype,
                "causal": causal,
                "config": config,
                "executable": str(executable),
            })
        elif build_result["status"] == "success":
            print(f"missing executable after build: {executable}")

    compile_report = []
    for source, config in zip(sources, configs):
        compile_report.append({
            "source": source.name,
            "executable": source.stem,
            "config": config,
            "register_usage_level": register_usage_level,
            **build_results[source.stem],
        })
    report_path.write_text(
        json.dumps(compile_report, indent=2) + "\n"
    )

    print(f"compiled {len(compiled)}/{len(configs)} configs successfully")
    print(f"compile report: {report_path}")
    return compiled


def bench_interface(
    compiled: list[CompiledEntry],
    rank: int = 15,
    result_dir: str | Path | None = None,
    timeout_seconds: float = 120.0,
    result_tag: str | None = None,
) -> list[BenchResult]:
    """Run generated binaries and save the top-rank configurations as JSON."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not compiled:
        raise ValueError("compiled executable list is empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    result_dir = Path(result_dir) if result_dir else HERE / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    shape = compiled[0]["shape"]
    dtype = compiled[0]["dtype"]
    causal = compiled[0]["causal"]
    for item in compiled:
        if (item["shape"], item["dtype"], item["causal"]) != (shape, dtype, causal):
            raise ValueError("all compiled entries must use the same shape, dtype and causal mode")
    batch, heads, seqlen, head_dim = shape
    mode = "causal" if causal else "noncausal"
    tag_suffix = ""
    if result_tag is not None:
        sanitized_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_tag).strip("._-")
        if not sanitized_tag:
            raise ValueError("result_tag must contain at least one filename-safe character")
        tag_suffix = f"_{sanitized_tag}"
    result_path = result_dir / (
        f"top{rank}_b{batch}_h{heads}_s{seqlen}_d{head_dim}_{dtype}_{mode}{tag_suffix}.json"
    )
    output_pattern = re.compile(
        r"time\s*=\s*([0-9.eE+-]+)\s*ms,\s*"
        r"throughput\s*=\s*([0-9.eE+-]+)\s*TFLOPS"
    )

    successful = []
    failed = []

    def save_results() -> list[BenchResult]:
        top_results = sorted(
            successful, key=lambda item: item["tflops"], reverse=True
        )[:rank]
        for index, item in enumerate(top_results, 1):
            item["rank"] = index
        payload = {
            "shape": {"B": batch, "H": heads, "S": seqlen, "D": head_dim},
            "dtype": dtype,
            "causal": causal,
            "requested_rank": rank,
            "benchmark_timeout_seconds": timeout_seconds,
            "result_tag": result_tag,
            "total_configs": len(compiled),
            "completed": len(successful) + len(failed),
            "successful": len(successful),
            "failed": len(failed),
            "top_results": top_results,
            "failed_results": failed,
        }
        result_path.write_text(json.dumps(payload, indent=2) + "\n")
        return top_results

    for index, item in enumerate(compiled, 1):
        config = item["config"]
        executable = Path(item["executable"])
        executable_name = executable.name
        print(f"[{index}/{len(compiled)}] {executable_name}", flush=True)

        if not executable.is_file():
            failed.append({
                "config": config,
                "executable": str(executable),
                "error": "executable not found",
            })
            save_results()
            continue

        try:
            process = subprocess.run(
                [str(executable)],
                cwd=executable.parent,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout_seconds,
            )
            output = "\n".join(
                part for part in (process.stdout.strip(), process.stderr.strip()) if part
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"exit code {process.returncode}: {output or 'no output'}"
                )

            matches = output_pattern.findall(output)
            if not matches:
                raise RuntimeError(f"cannot parse benchmark output: {output!r}")
            time_ms, tflops = map(float, matches[-1])
            successful.append({
                "config": config,
                "executable": executable_name,
                "time_ms": time_ms,
                "tflops": tflops,
            })
            print(f"  time={time_ms:.3f} ms, tflops={tflops:.2f}")
        except Exception as error:
            failed.append({
                "config": config,
                "executable": executable_name,
                "error": str(error),
            })
            print(f"  failed: {error}")

        save_results()

    top_results = save_results()
    print(f"saved top {len(top_results)} results to {result_path}")
    return top_results

def run(
    shape: Shape,
    dtype: DType,
    causal: bool,
    configs: Iterable[Config],
    arch: str = "90a",
    jobs: int = 2,
    rank: int = 15,
    result_dir: str | Path | None = None,
    coarse_register_usage_level: int = 5,
    final_register_usage_level: int = 10,
    structure_top_n: int = 3,
    benchmark_timeout_seconds: float = 120.0,
    result_tag: str | None = None,
) -> list[BenchResult]:
    """Rank structures with scheduler barriers off, then tune both choices."""
    configs = list(configs)
    if structure_top_n <= 0:
        raise ValueError("structure_top_n must be positive")

    coarse_configs = [dict(config, use_scheduler_barrier=0) for config in configs]
    representatives = []
    seen_structures = set()
    for config in coarse_configs:
        key = tuple(config[field] for field in STRUCTURE_FIELDS)
        if key not in seen_structures:
            seen_structures.add(key)
            representatives.append(config)

    print(
        f"coarse pass: {len(representatives)} representatives for "
        f"{len(seen_structures)} structures at register-usage-level "
        f"{coarse_register_usage_level}"
    )
    coarse_compiled = compile_interface(
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=representatives,
        arch=arch,
        jobs=jobs,
        register_usage_level=coarse_register_usage_level,
    )
    if not len(coarse_compiled):
        print("[NONE]: there is no kernel that meets the execution conditions.")
        return []
    coarse_results = bench_interface(
        compiled=coarse_compiled,
        rank=len(coarse_compiled),
        result_dir=result_dir,
        timeout_seconds=benchmark_timeout_seconds,
        result_tag=result_tag,
    )
    selected_structures = {
        tuple(result["config"][field] for field in STRUCTURE_FIELDS)
        for result in coarse_results[:structure_top_n]
    }
    finalist_bases = [
        config for config in configs
        if tuple(config[field] for field in STRUCTURE_FIELDS) in selected_structures
    ]
    if not finalist_bases:
        raise ValueError("no structure passed the coarse benchmark")
    finalists = [
        dict(config, use_scheduler_barrier=use_scheduler_barrier)
        for config in finalist_bases
        for use_scheduler_barrier in (0, 1)
    ]

    print(
        f"final pass: {len(finalists)} configs (sb0/sb1) from "
        f"{len(selected_structures)} structures at register-usage-level "
        f"{final_register_usage_level}"
    )
    final_compiled = compile_interface(
        shape=shape,
        dtype=dtype,
        causal=causal,
        configs=finalists,
        arch=arch,
        jobs=jobs,
        register_usage_level=final_register_usage_level,
    )
    if not final_compiled:
        print("[NONE]: no finalist compiled without a performance warning.")
        return []
    return bench_interface(
        compiled=final_compiled,
        rank=rank,
        result_dir=result_dir,
        timeout_seconds=benchmark_timeout_seconds,
        result_tag=result_tag,
    )

# if __name__ == "__main__":
    # main()
