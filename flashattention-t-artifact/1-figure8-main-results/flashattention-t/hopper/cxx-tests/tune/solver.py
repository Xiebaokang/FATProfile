"""Resource-constrained solver for mixed SS/RS Hopper attention configs."""

from __future__ import annotations

from typing import Literal

Config = dict[str, int]
Mode = Literal["keep", "radical"]

MMA_M = 64
MAX_MMA_N = 256
WARP_GROUP_THREADS = 128

# Candidate (producer, consumer) register allocations. The first entry for
# each consumer count is FlashAttnFwdSm90's default. The remaining entries
# trade producer registers against the largest 8-register-aligned consumer
# allocation that still fits in Hopper's 65536-register file.
REGISTER_ALLOCATION = {
    1: (
        (56, 256),
        (64, 256),
    ),
    2: (
        (24, 240),
        (32, 240),
        (24, 232),
        (32, 232),
        (24, 224),
        (32, 224),
        (24, 216),
        (24, 208),
        (24, 200),
    ),
    3: (
        (32, 160),
        (40, 152),
        (48, 152),
        (56, 152),
        (64, 144),
    ),
    4: (
        (24, 120),
        (32, 120),
        (40, 112),
        (48, 112),
        (56, 112),
        (64, 112),
    ),
}


def ceil_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _mma_k(elem_width: int) -> int:
    """Return the SM90 WGMMA K tile for the selected input type."""
    if elem_width not in (1, 2):
        raise ValueError("elem_width must be 1 (FP8) or 2 (FP16/BF16)")
    return 32 if elem_width == 1 else 16


def _validate_inputs(
    hd: int,
    stage: int,
    elem_width: int,
    bn_limit_n: int,
    smem_limit: int,
    reg_limit: int,
    num_consumer_limit: int,
    buf_use_rate: float,
    causal: bool,
) -> None:
    mma_k = _mma_k(elem_width)
    if hd <= 0 or hd % mma_k != 0:
        raise ValueError(f"HD must be a positive multiple of {mma_k}")
    if hd > 256:
        raise ValueError("USE_MIX_WGMMA requires HD <= 256")
    if stage <= 0:
        raise ValueError("stage must be positive")
    if isinstance(bn_limit_n, bool) or not isinstance(bn_limit_n, int):
        raise TypeError("bn_limit_n must be int")
    if bn_limit_n < 0:
        raise ValueError("bn_limit_n must be non-negative")
    if smem_limit <= 0:
        raise ValueError("smem_limit must be positive")
    if reg_limit <= 0:
        raise ValueError("reg_limit must be positive")
    if num_consumer_limit <= 0:
        raise ValueError("num_consumer_limit must be positive")
    if num_consumer_limit > max(REGISTER_ALLOCATION):
        raise ValueError("USE_MIX_WGMMA supports at most 4 MMA consumer warpgroups")
    if not 0.0 < buf_use_rate <= 1.0:
        raise ValueError("buf_use_rate must be in (0, 1]")
    if not isinstance(causal, bool):
        raise TypeError("causal must be bool")


def _fa3_pipeline_storage_bytes(stage: int) -> int:
    """Size of FlashAttnFwdSm90::PipelineStorage for the simple TMA path.

    There are three standalone barriers, five kStages TMA pipelines (K, V,
    Vt, K-new and V-new), and one scheduler int. Every member is alignas(16).
    Causal and non-causal schedulers both use an int as SharedStorage.
    """
    standalone_barriers = 3 * 16
    one_tma_pipeline = 2 * stage * 8
    scheduler_storage = 16
    return standalone_barriers + 5 * one_tma_pipeline + scheduler_storage


def _fa3_swizzle_alignment(elements: int, elem_width: int) -> int:
    """Return CUTLASS alignment_for_swizzle for the selected GMMA layout.

    ``ss_smem_selector`` picks the largest SW128/SW64/SW32 atom dividing the
    contiguous byte width. Their CUTE Swizzle alignments are respectively
    1024/512/256 bytes; the INTER fallback is 128-byte aligned.
    """
    byte_width = elements * elem_width
    if byte_width % 128 == 0:
        return 1024
    if byte_width % 64 == 0:
        return 512
    if byte_width % 32 == 0:
        return 256
    return 128


def _fa3_smem_size_bytes(
    bm: int,
    bn: int,
    hd: int,
    stage: int,
    elem_width: int,
    p_smem_k_tiles: int,
    q_reg_k_tiles: int,
    causal: bool,
) -> int:
    """Model FlashAttnFwdSm90::SharedStorage for fixed-length Q/K/V.

    This follows CollectiveMainloopFwdSm90::TensorStorage and the kernel-level
    mainloop/epilogue union. FP8 row-major V owns both smem_v and smem_vt;
    its output is BF16, so output element width is always two bytes here.

    p_smem_k_tiles and q_reg_k_tiles describe this tree's mixed-WGMMA P and Q
    partitions; both are expressed in the input type's MMA-K tiles.
    """
    del causal  # Both scheduler variants have the same SharedStorage size.
    mma_k = _mma_k(elem_width)
    q_smem_cols = hd - q_reg_k_tiles * mma_k
    p_smem_cols = p_smem_k_tiles * mma_k
    v_tma_alignment = _fa3_swizzle_alignment(hd, elem_width)
    v_mma_alignment = (
        _fa3_swizzle_alignment(bn, elem_width)
        if elem_width == 1
        else v_tma_alignment
    )
    p_alignment = (
        # SmemLayoutP is selected from the full BN width even though only a
        # prefix is allocated for the mixed SS part.
        _fa3_swizzle_alignment(bn, elem_width)
        if p_smem_cols > 0
        else 128
    )
    mainloop_alignment = max(
        128, v_tma_alignment, v_mma_alignment, p_alignment
    )
    offset = 0

    def add_array(size: int, alignment: int) -> None:
        nonlocal offset
        if size == 0:
            return
        offset = align_up(offset, alignment)
        # sizeof(cute::array_aligned<T, N, Alignment>) is itself rounded to
        # Alignment, not merely aligned at its starting address.
        offset += align_up(size, alignment)

    v_storage = bn * hd * stage * elem_width
    add_array(v_storage, v_mma_alignment)                # smem_v
    if elem_width == 1:
        add_array(v_storage, v_tma_alignment)            # FP8 smem_vt
    add_array(bm * q_smem_cols * elem_width, 128)       # smem_q
    add_array(bn * hd * stage * elem_width, 128)        # smem_k
    # SmemQv_t is cute::array<Element, 0> in the requested simple path. CUDA
    # gives that empty member sizeof 1, which affects the enclosing rounding.
    offset += 1
    add_array(
        bm * p_smem_cols * elem_width,
        p_alignment,
    )                                                    # mixed-WGMMA smem_p
    mainloop_storage = align_up(offset, mainloop_alignment)

    # FlashAttnFwdSm90 overlaps epilogue smem_o only with smem_v. The nested
    # anonymous mainloop struct contains a zero-length padding array; when the
    # required padding is zero, that member still has sizeof 1 and pushes the
    # aligned Mainloop::TensorStorage to the next alignment boundary.
    output_storage = align_up(bm * hd * 2, 128)          # FP8 output is BF16
    smem_v_member_size = align_up(v_storage, v_mma_alignment)
    epilogue_padding = max(0, output_storage - smem_v_member_size)
    padding_member_size = epilogue_padding or 1
    mainloop_offset = align_up(padding_member_size, mainloop_alignment)
    mainloop_union_member = align_up(
        mainloop_offset + mainloop_storage,
        mainloop_alignment,
    )
    tensor_storage = align_up(
        max(output_storage, mainloop_union_member),
        mainloop_alignment,
    )

    return align_up(
        tensor_storage + _fa3_pipeline_storage_bytes(stage),
        mainloop_alignment,
    )


def _consumer_registers_per_thread(
    bm: int,
    hd: int,
    bn: int,
    elem_width: int,
    p_smem_k_tiles: int,
    q_reg_k_tiles: int,
    num_consumer: int,
    mode: Mode,
) -> int:
    """Count source-visible 32-bit fragment registers per consumer thread.

    FA3 constructs acc-O and acc-S with ``partition_fragment_C``. Their FP32
    element counts are the corresponding matrix element counts divided by
    ``NumMmaThreads``. P packs 2 FP16/BF16 or 4 FP8 elements into each
    32-bit register.

    ``keep`` charges the complete converted P fragment. ``radical`` charges
    only the persistent FP8/FP16 register suffix after the SMEM prefix; the
    C++ path converts each prefix tile through one short-lived temporary.

    This is a static fragment lower bound, not ptxas's final register count:
    address arithmetic, pipeline state and mask/control temporaries are chosen
    by the compiler and must be checked from each compiled kernel's ptxas info.
    """
    num_mma_threads = WARP_GROUP_THREADS * num_consumer
    acc_o_elements = bm * hd
    acc_s_elements = bm * bn
    if acc_o_elements % num_mma_threads != 0:
        raise ValueError("BM * HD must be divisible by NumMmaThreads")
    if acc_s_elements % num_mma_threads != 0:
        raise ValueError("BM * BN must be divisible by NumMmaThreads")

    # tOrO and tSrS are FP32 C fragments: one element is one register.
    acc_o = acc_o_elements // num_mma_threads
    acc_s = acc_s_elements // num_mma_threads

    # Q register tiles are M64 x MMA-K A fragments owned by one 128-thread
    # consumer WG. In radical mode, only the P columns not mapped to SMEM
    # remain persistently live as the RS operand.
    mma_k = _mma_k(elem_width)
    p_reg_cols = bn - p_smem_k_tiles * mma_k
    if p_reg_cols < 0:
        raise ValueError("P SMEM tiles exceed the PV K dimension")
    p_live_cols = p_reg_cols if mode == "radical" else bn
    p_elements = bm * p_live_cols // num_mma_threads
    q_reg_elements = MMA_M * (q_reg_k_tiles * mma_k) // WARP_GROUP_THREADS
    p_regs = ceil_div(p_elements * elem_width, 4)
    q_regs = ceil_div(q_reg_elements * elem_width, 4)

    # flash::Softmax<kNRows> owns row_max[kNRows], row_sum[kNRows] and one
    # scale. max_get_scale also has scores_scale and scores_max_prev fragments.
    k_n_rows = 2 * (2 * bm // num_mma_threads)
    softmax_peak_regs = 4 * k_n_rows + 1

    always_live = acc_o + q_regs + softmax_peak_regs
    return always_live + acc_s + p_regs


def _mix_wgmma_solve(
    hd: int,
    stage: int,
    elem_width: int,
    bn_limit_n: int,
    smem_limit: int,
    reg_limit: int,
    num_consumer_limit: int,
    buf_use_rate: float,
    causal: bool,
    mode: Mode,
) -> list[Config]:
    _validate_inputs(
        hd, stage, elem_width, bn_limit_n, smem_limit, reg_limit,
        num_consumer_limit, buf_use_rate, causal,
    )

    solutions: list[Config] = []
    mma_k = _mma_k(elem_width)
    q_total_tiles = hd // mma_k

    for num_consumer in range(2, num_consumer_limit + 1):
        # The requested mapping is one m64 WGMMA tile per consumer warpgroup.
        bm = MMA_M * num_consumer

        # K and V are unavoidable, so they give a safe finite BN upper bound.
        # FP8 has K + V + transposed-V; FP16/BF16 has K + V.
        staged_tensor_count = 3 if elem_width == 1 else 2
        kv_bytes_per_bn = staged_tensor_count * hd * stage * elem_width
        bn_max = smem_limit // kv_bytes_per_bn
        bn_max = bn_max // mma_k * mma_k
        bn_max = min(MAX_MMA_N, bn_max)
        bn_min = (
            mma_k
            if bn_limit_n == 0
            else max(mma_k, bn_max - (bn_limit_n - 1) * mma_k)
        )

        for bn in range(bn_min, bn_max + 1, mma_k):
            # FP8 row-major V takes the Transpose_V path. If HD is not a
            # multiple of 64, the C++ transpose layout requires BN % 64 == 0.
            if elem_width == 1 and hd % 64 != 0 and bn % 64 != 0:
                continue

            # if bn < 112:
            #     continue

            # print(bm, bn)
            p_total_tiles = bn // mma_k

            for p_smem_k_tiles in range(p_total_tiles + 1):
                for q_reg_k_tiles in range(q_total_tiles + 1):
                    smem_size = _fa3_smem_size_bytes(
                        bm, bn, hd, stage, elem_width,
                        p_smem_k_tiles, q_reg_k_tiles, causal,
                    )
                    
                    if smem_size > smem_limit:
                        continue
                    
                    if smem_size / smem_limit < buf_use_rate:
                        continue
                    
                    estimated_regs = _consumer_registers_per_thread(
                        bm, hd, bn, elem_width, p_smem_k_tiles,
                        q_reg_k_tiles, num_consumer, mode,
                    )
                    for (
                        producer_reg_dealloc,
                        consumer_reg_alloc,
                    ) in REGISTER_ALLOCATION[num_consumer]:
                        if estimated_regs > consumer_reg_alloc:
                            continue

                        allocated_regs = WARP_GROUP_THREADS * (
                            producer_reg_dealloc
                            + num_consumer * consumer_reg_alloc
                        )
                        if allocated_regs > reg_limit:
                            continue
                        if allocated_regs / reg_limit < buf_use_rate:
                            continue

                        solutions.append({
                            "kBlockM": bm,
                            "kBlockN": bn,
                            "kStage": stage,
                            "producer_reg_dealloc": producer_reg_dealloc,
                            "consumer_reg_alloc": consumer_reg_alloc,
                            "p_smem_k_tiles": p_smem_k_tiles,
                            "q_reg_k_tiles": q_reg_k_tiles,
                            "num_consumer": num_consumer,
                            # The coarse structure pass always benchmarks with
                            # scheduler barriers disabled. run_utils expands
                            # each selected finalist to both sb0 and sb1.
                            "use_scheduler_barrier": 0,
                            "smem_size": smem_size,
                            "estimated_consumer_regs_per_thread": estimated_regs,
                            "estimated_consumer_fragment_regs": (
                                estimated_regs * num_consumer * WARP_GROUP_THREADS
                            ),
                            "allocated_registers_per_cta": allocated_regs,
                        })

    return solutions


def mix_wgmma_solve(
    HD: int,
    stage: int = 2,
    elem_width: int = 2,
    bn_limit_n: int = 0,
    smem_limit: int = 232_448,
    reg_limit: int = 65_536,
    num_consumer_limit: int = 4,
    buf_use_rate: float = 0.8,
    causal: bool = False,
    mode: Mode = "radical",
) -> list[Config]:
    """Return feasible mixed-WGMMA configurations.

    ``bn_limit_n == 0`` enumerates every BN candidate. A positive value ``n``
    keeps at most ``n`` candidates, descending by one input MMA-K tile
    (32 columns for FP8 and 16 for FP16/BF16). A sufficiently large ``n``
    enumerates all aligned BN values. ``mode="keep"`` conservatively charges
    the full converted P fragment. ``mode="radical"`` follows the compact
    FP8/FP16 register suffix implemented by the mixed-WGMMA C++ path.
    """
    if mode not in ("keep", "radical"):
        raise ValueError("mode must be 'keep' or 'radical'")
    return _mix_wgmma_solve(
        HD, stage, elem_width, bn_limit_n, smem_limit, reg_limit,
        num_consumer_limit, buf_use_rate, causal=causal, mode=mode,
    )


if "__main__" == __name__:
    for stage in range(2, 4):
        test_configs = mix_wgmma_solve(
            HD=64,
            stage=stage,
            bn_limit_n=0,
            elem_width=2,  # FP8; use 2 for FP16/BF16
            smem_limit=232_448,
            reg_limit=65_536,
            num_consumer_limit=3,
            buf_use_rate=0.3,
            causal=False,
        )

        print(f"found {len(test_configs)} configs")
        for config in test_configs:
            if config["p_smem_k_tiles"] or config["q_reg_k_tiles"]:
                print(config)
