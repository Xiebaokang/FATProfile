import argparse
import csv
import itertools
import math
import shlex
import subprocess
import sys
from pathlib import Path


NCU_METRICS = (
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
)
CSV_FIELDS = ["B", "H", "S", "D", "causal", "tc_utilization"]

DTYPE_CHOICES = (
    "fp16",
    "fp8",
)
SCRIPT_DTYPE_CHOICES = {
    "fa3.py": {"fp16", "fp8"},
    "flex.py": {"fp16", "fp8"},
    "finfer.py": {"fp16"},
}


def parse_int_list(value):
    return [int(item) for item in value.replace(" ", ",").split(",") if item.strip()]


def parse_float(value):
    value = value.strip().strip('"').replace(",", "")
    if not value or value.lower() in {"n/a", "nan"}:
        return math.nan
    return float(value)


def metric_value_from_row(row, metric_name):
    for idx, cell in enumerate(row):
        if cell.strip().strip('"') != metric_name:
            continue

        for value_idx in range(len(row) - 1, idx, -1):
            try:
                value = parse_float(row[value_idx])
            except ValueError:
                continue
            unit = row[value_idx - 1] if value_idx - 1 > idx else ""
            return value, unit
    return None


def metric_value_from_text_line(line, metric_name):
    line = line.strip()
    if not line.startswith(metric_name):
        return None

    fields = line[len(metric_name):].strip().split()
    if not fields:
        return None

    try:
        value = parse_float(fields[-1])
    except ValueError:
        return None

    unit = fields[-2] if len(fields) >= 2 else ""
    return value, unit


def parse_ncu_output(output):
    tc_utilization = math.nan

    for line in output.splitlines():
        tc_result = metric_value_from_text_line(
            line, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    for row in csv.reader(output.splitlines()):
        if not row:
            continue

        tc_result = metric_value_from_row(
            row, "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
        if tc_result is not None:
            value, _unit = tc_result
            tc_utilization = value

    return tc_utilization


def format_number(value):
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def median_finite(values):
    finite_values = sorted(value for value in values if not math.isnan(value))
    if not finite_values:
        return math.nan
    mid = len(finite_values) // 2
    if len(finite_values) % 2:
        return finite_values[mid]
    return 0.5 * (finite_values[mid - 1] + finite_values[mid])


def print_subprocess_output(output, file=None):
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    print(output, file=file, end="" if output.endswith("\n") else "\n")


def build_ncu_command(args, script, B, H, S, D):
    script_path = args.script_dir / script
    cmd = [
        args.ncu,
        "--metrics",
        NCU_METRICS,
        "--profile-from-start",
        args.profile_from_start,
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        args.python,
        str(script_path),
        "--B",
        str(B),
        "--H",
        str(H),
        "--S",
        str(S),
        "--D",
        str(D),
        "--warmup",
        str(args.warmup),
        "--causal",
        str(args.causal).lower(),
    ]
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])

    kernel_name = args.kernel_name
    if kernel_name is None:
        kernel_name = default_kernel_name(script)
    if kernel_name:
        cmd[1:1] = [
            "--kernel-name-base",
            "function",
            "--kernel-name",
            kernel_name,
        ]
    if args.ncu_csv:
        cmd[1:1] = ["--csv", "--page", "raw"]
    return cmd


def default_kernel_name(script):
    if script == "fa3.py":
        return "regex:.*FlashAttnFwd.*"
    if script == "flex.py":
        return "regex:.*triton_tem_fused_2.*"
    if script == "finfer.py":
        return "regex:.*PrefillWithKVCacheKernel.*"
    return ""


def iter_shapes(args):
    return itertools.product(args.batch_sizes, args.num_heads, args.seq_lens, args.head_dims)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def shape_filter(B, H, S, D):
    return B * S == 16384 and H * D == 2048


def skipped_rows(args, shapes):
    rows = []
    for B, H, S, D in shapes:
        if shape_filter(B, H, S, D):
            rows.append(
                {
                    "B": B,
                    "H": H,
                    "S": S,
                    "D": D,
                    "causal": int(args.causal),
                    "tc_utilization": "nan",
                }
            )
    return rows


def profile_shape(args, script, B, H, S, D):
    cmd = build_ncu_command(args, script, B, H, S, D)
    print(shlex.join(cmd), flush=True)

    if args.dry_run:
        return math.nan

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        print_subprocess_output(exc.stdout, file=sys.stderr)
        print(
            f"ncu timed out after {args.timeout_sec}s for {script} "
            f"B={B} H={H} S={S} D={D}",
            file=sys.stderr,
        )
        return math.nan
    if args.print_ncu_output:
        print_subprocess_output(completed.stdout)
    tc_utilization = parse_ncu_output(completed.stdout)

    if completed.returncode != 0 or math.isnan(tc_utilization):
        print_subprocess_output(completed.stdout, file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"ncu failed for B={B} H={H} S={S} D={D} "
            f"with return code {completed.returncode}",
            file=sys.stderr,
        )
    elif math.isnan(tc_utilization):
        print(
            f"failed to parse tensor-core utilization for {script} "
            f"B={B} H={H} S={S} D={D}",
            file=sys.stderr,
        )

    return tc_utilization


def profile_shape_repeated(args, script, B, H, S, D):
    values = []
    for repeat_idx in range(1, args.ncu_repeats + 1):
        if args.ncu_repeats > 1:
            print(f"repeat {repeat_idx}/{args.ncu_repeats}", flush=True)
        values.append(profile_shape(args, script, B, H, S, D))
    if args.ncu_repeats > 1:
        formatted = ", ".join(format_number(value) for value in values)
        print(f"tc_utilization repeats = [{formatted}]", flush=True)
    return median_finite(values)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sweep B/H/S/D shapes, collect tensor-core utilization with ncu, and save CSV."
    )
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--num-heads", type=parse_int_list, default=[16, 32])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--head-dims", type=parse_int_list, default=[64, 128])
    # parser.add_argument("--batch-sizes", type=parse_int_list, default=[8])
    # parser.add_argument("--num-heads", type=parse_int_list, default=[16])
    # parser.add_argument("--seq-lens", type=parse_int_list, default=[8192])
    # parser.add_argument("--head-dims", type=parse_int_list, default=[128])
    parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="fp16",
        help="Forward dtype to fa3.py/flex.py/finfer.py. If omitted, each script keeps its own default.",
    )
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--profile-from-start", choices=("on", "off"), default="off")
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--ncu-repeats", type=int, default=1)
    parser.add_argument("--kernel-name", default=None)
    parser.add_argument("--fa3-csv", default="./result/fa3_ncu_sweep.csv")
    parser.add_argument("--flex-csv", default="./result/flex_ncu_sweep.csv")
    parser.add_argument("--finfer-csv", default="./result/finfer_ncu_sweep.csv")
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--ncu-csv", action="store_true")
    parser.add_argument(
        "--print-ncu-output",
        action="store_true",
        help="Print raw ncu output before parsing tc_utilization for debugging.",
    )
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=300.0,
        help="Per-shape subprocess timeout in seconds. Use 0 to disable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_sweep(args, script, csv_path, shapes):
    print(f"script     = {script}")
    print(f"num shapes = {len(shapes)}")
    print(f"csv        = {csv_path}")
    print(f"dtype      = {args.dtype or 'script default'}")
    print(f"causal     = {args.causal}")
    print(f"kernel name= {args.kernel_name or default_kernel_name(script)}")
    print(f"profile fs = {args.profile_from_start}")
    print(f"launch skip= {args.launch_skip}")
    print(f"launch cnt = {args.launch_count}")
    print(f"ncu repeats= {args.ncu_repeats}")
    print(f"timeout s  = {args.timeout_sec}")
    print(f"print ncu  = {args.print_ncu_output}")
    print(f"dry run    = {args.dry_run}")

    rows = []
    if args.dtype not in SCRIPT_DTYPE_CHOICES[script]:
        print(f"skip {script}: dtype {args.dtype} is not supported")
        rows = skipped_rows(args, shapes)
        if args.dry_run:
            print(f"dry-run: skipped CSV write to {csv_path}")
        else:
            write_rows(csv_path, rows)
            print(f"saved rows = {len(rows)} to {csv_path}")
        return

    for idx, (B, H, S, D) in enumerate(shapes, 1):
        if not shape_filter(B, H, S, D):
            continue
        print(f"[{idx}/{len(shapes)}] B={B} H={H} S={S} D={D}", flush=True)
        tc_utilization = profile_shape_repeated(args, script, B, H, S, D)
        row = {
            "B": B,
            "H": H,
            "S": S,
            "D": D,
            "causal": int(args.causal),
            "tc_utilization": format_number(tc_utilization),
        }
        rows.append(row)
        print(row, flush=True)

    if args.dry_run:
        print(f"dry-run: skipped CSV write to {csv_path}")
    else:
        write_rows(csv_path, rows)
        print(f"saved rows = {len(rows)} to {csv_path}")


def main():
    args = build_parser().parse_args()
    if args.timeout_sec <= 0:
        args.timeout_sec = None
    if args.ncu_repeats < 1:
        raise ValueError("--ncu-repeats must be >= 1")
    script_dir = Path(__file__).resolve().parent
    args.script_dir = script_dir
    args.fa3_csv = resolve_output_path(script_dir, args.fa3_csv)
    args.flex_csv = resolve_output_path(script_dir, args.flex_csv)
    args.finfer_csv = resolve_output_path(script_dir, args.finfer_csv)
    shapes = list(iter_shapes(args))
    run_sweep(args, "fa3.py", args.fa3_csv, shapes)
    run_sweep(args, "flex.py", args.flex_csv, shapes)
    run_sweep(args, "finfer.py", args.finfer_csv, shapes)


def resolve_output_path(script_dir, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return script_dir / path


if __name__ == "__main__":
    main()
