"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton

import flashinfer
import argparse
import pathlib
import csv
from typing import Optional, Tuple


def bench_batch_ragged_prefill(batch_size, num_heads, seq_len, causal, head_dim):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    sm80_wrapper, sm90_wrapper = (
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"),
            kv_layout="NHD",
            backend=backend,
        )
        for backend in ["fa2", "fa3"]
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()

    for wrapper in [sm80_wrapper, sm90_wrapper]:
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            causal=causal,
        )

    sm80_ms, sm90_ms = (
        triton.testing.do_bench(
            lambda: wrapper.run(q, k, v),
            warmup=100,
            rep=1000,
        )
        for wrapper in [sm80_wrapper, sm90_wrapper]
    )

    def flops(ms):
        if causal:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 2 / ms / 1e9
            )
        else:
            return (
                batch_size * seq_len * seq_len * num_qo_heads * head_dim * 4 / ms / 1e9
            )

    print(
        f"bench_batch_ragged_prefill (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, causal={causal}, head_dim={head_dim}), fa2-template: {flops(sm80_ms):.3f} TFLOPs/s, fa3-template: {flops(sm90_ms):.3f} TFLOPs/s"
    )

def bench_batch_ragged_prefill(
    batch_size, num_heads, seq_len, causal, head_dim
):
    num_qo_heads = num_kv_heads = num_heads
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=torch.half, device="cuda"
    )

    backend = "fa2"

    sm80_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"),
        kv_layout="NHD",
        backend=backend
    )

    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    
    sm80_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=causal,
        use_fp16_qk_reduction=False
    )

    sm80_ms = triton.testing.do_bench(
        lambda: sm80_wrapper.run(q, k, v),
        warmup=100,
        rep=1000,
    )

    return sm80_ms


def run_custom_batch_ragged_prefill_bench(output_csv: Optional[pathlib.Path] = None):
    TOTAL_TOKS = 16384
    TOTAL_HEADS = 2048

    seqlens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    headdims = [64, 128]
    causals = [False, True]

    # return: (b, n, s, h)
    def get_full_shape(seqlen:int, headdim:int):
        assert TOTAL_TOKS % seqlen == 0, "invalid seqlen"
        assert TOTAL_HEADS % headdim == 0, "invalid headdim"
        batchsize = TOTAL_TOKS // seqlen
        nheads = TOTAL_HEADS // headdim
        return (batchsize, nheads, seqlen, headdim)

    rows = []
    rows.append(["DataType", "Comment", "batchsize", "nheads", "seqlen", "headdim", "is_causal", "time_ms"])
    

    # bench_batch_ragged_prefill(batch_size, num_heads, seq_len, causal, head_dim):
    for is_causal in causals:
        for headdim in headdims:
            for seqlen in seqlens:
                batchsize, nheads, _, _ = get_full_shape(seqlen, headdim)
                time_ms = bench_batch_ragged_prefill(batchsize, nheads, seqlen, is_causal, headdim)
                print(f"batch_size: {batchsize:<10d}, seqlen: {seqlen:<10d}, num_heads: {nheads:<10d}, head_dim: {headdim:<10d}, causal: {is_causal:<10d}, avg_time_ms: {time_ms:.6f}")

                row = ["FP16-FP32", "flashinfer", batchsize, nheads, seqlen, headdim, int(is_causal), time_ms]
                rows.append(row)

    if output_csv is not None:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

def main():
    argparser = argparse.ArgumentParser(description="Benchmark FlashInfer attention prefill [For Ampere machine (A100/Orin)]")
    argparser.add_argument("output_csv", type=str, default=None, nargs='?')
    args = argparser.parse_args()
    print("=== BENCHMARK FP16-FP32 ===")

    run_custom_batch_ragged_prefill_bench(pathlib.Path(args.output_csv) if args.output_csv else None)


if __name__ == "__main__":
    main()
