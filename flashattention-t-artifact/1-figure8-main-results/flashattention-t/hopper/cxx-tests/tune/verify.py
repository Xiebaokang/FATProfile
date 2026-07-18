#!/usr/bin/env python3
"""Compile and numerically verify tuned FA3 configurations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_utils import CompiledEntry, compile_interface


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile configurations from a tune result JSON and compare each "
            "custom kernel against LibTorch scaled-dot-product attention."
        )
    )
    parser.add_argument("result_json", type=Path, help="top*.json produced by bench.py")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--index",
        type=int,
        default=1,
        help="1-based top_results index to verify (default: 1)",
    )
    selection.add_argument(
        "--all", action="store_true", help="verify every entry in top_results"
    )
    parser.add_argument("--arch", default="90a")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--register-usage-level", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument(
        "--seqlen", type=int,
        help="override S from the result JSON for boundary/tail tests",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_cases(path: Path, index: int, verify_all: bool) -> tuple[tuple[int, int, int, int], str, bool, list[dict[str, int]]]:
    payload: dict[str, Any] = json.loads(path.read_text())
    shape_obj = payload["shape"]
    shape = (
        int(shape_obj["B"]),
        int(shape_obj["H"]),
        int(shape_obj["S"]),
        int(shape_obj["D"]),
    )
    dtype = str(payload["dtype"])
    if dtype not in ("fp8", "fp16"):
        raise ValueError(f"unsupported dtype in {path}: {dtype!r}")
    causal_value = payload["causal"]
    if not isinstance(causal_value, bool):
        raise ValueError(f"causal in {path} must be a JSON boolean")
    causal = causal_value
    entries = payload.get("top_results", [])
    if not entries:
        raise ValueError(f"{path} contains no top_results")
    if verify_all:
        selected = entries
    else:
        if index <= 0 or index > len(entries):
            raise ValueError(f"--index must be in [1, {len(entries)}]")
        selected = [entries[index - 1]]
    return shape, dtype, causal, [dict(entry["config"]) for entry in selected]


def verify_executables(
    compiled: list[CompiledEntry], seed: int, atol: float | None,
    rtol: float | None, timeout: float,
) -> bool:
    passed = True
    for position, item in enumerate(compiled, 1):
        executable = Path(item["executable"])
        command = [str(executable), "--verify", f"--seed={seed}"]
        if atol is not None:
            command.append(f"--atol={atol}")
        if rtol is not None:
            command.append(f"--rtol={rtol}")
        print(f"[{position}/{len(compiled)}] {' '.join(command)}", flush=True)
        try:
            process = subprocess.run(
                command,
                cwd=executable.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            passed = False
            print(f"  verification timed out after {timeout:g} seconds")
            continue
        if process.stdout:
            print(process.stdout, end="" if process.stdout.endswith("\n") else "\n")
        if process.returncode != 0:
            passed = False
            print(f"  verification exited with code {process.returncode}")
    return passed


def main() -> int:
    args = parse_args()
    try:
        shape, dtype, causal, configs = load_cases(
            args.result_json.resolve(), args.index, args.all
        )
        if args.seqlen is not None:
            if args.seqlen <= 0:
                raise ValueError("--seqlen must be positive")
            shape = (shape[0], shape[1], args.seqlen, shape[3])
        if args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        compiled = compile_interface(
            shape=shape,
            dtype=dtype,  # type: ignore[arg-type]
            causal=causal,
            configs=configs,
            arch=args.arch,
            jobs=args.jobs,
            register_usage_level=args.register_usage_level,
        )
        if len(compiled) != len(configs):
            print(
                f"only {len(compiled)}/{len(configs)} configurations compiled",
                file=sys.stderr,
            )
            return 1
        return 0 if verify_executables(
            compiled, args.seed, args.atol, args.rtol, args.timeout
        ) else 1
    except (
        KeyError,
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"verify.py: {error}", file=sys.stderr)
        return 2

# python verify.py results/top15_b1_h16_s32768_d64_fp16_noncausal.json
if __name__ == "__main__":
    raise SystemExit(main())
