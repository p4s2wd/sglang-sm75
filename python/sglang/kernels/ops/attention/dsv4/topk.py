from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_xpu

from .candidate_table import CANDIDATE_BLOCK_SIZE
from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    from sglang.kernels.jit.utils.occupancy import (
        NoSchedulableClustersError,
        get_max_active_clusters,
    )

    args = make_cpp_args(is_arch_support_pdl())
    # Enable each cluster path only when its occupancy probe reports capacity.
    extra_cuda_cflags = []
    if is_arch_support_pdl():  # set the persistent cluster size after hopper
        for cluster_size, occupancy in ((8, 2), (16, 1)):
            try:
                max_active_clusters = get_max_active_clusters(
                    cluster_size, occupancy=occupancy
                )
            except NoSchedulableClustersError:
                max_active_clusters = 0
            extra_cuda_cflags.append(
                f"-DSGL_TOPK_V2_MAX_C{cluster_size}_OCC{occupancy}={max_active_clusters}"
            )
    kernel = f"TopKKernel<{args}>"
    wrappers = [
        ("topk_transform_paged", f"{kernel}::transform_paged"),
        ("topk_transform_ragged", f"{kernel}::transform_ragged"),
        ("topk_plan", f"{kernel}::plan"),
    ]
    if is_hip_runtime():
        # transform_packed only exists under USE_ROCM, see topk_v2.cuh
        wrappers.append(("topk_transform_packed", f"{kernel}::transform_packed"))
    return load_jit(
        make_name("topk_v2"),
        *args,
        extra_cuda_cflags=extra_cuda_cflags,
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=wrappers,
    )


@cache_once
def _jit_topk_bf16_small_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_bf16_small"),
        *args,
        cuda_files=["deepseek_v4/topk_bf16_small.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKBF16Kernel<{args}>::transform")],
    )


def topk_transform_bf16_small(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    """bf16 top-k for rows of at most 16384 scores (the DeepSeek-V4.1 sparse
    indexer's consumer rows), fused with a page-table transform.

    Row ``b`` selects the ``k = out_page_indices.shape[1]`` best of its first
    ``seq_lens[b]`` scores (``k`` at most 2048); a selected index ``i`` is
    written as ``page_table[b, i // page_size] * page_size + i % page_size``,
    in no particular order, and ``-1`` fills the slots past
    ``min(k, seq_lens[b])``. Selection is exact (two radix passes over the raw
    bf16 bytes locate the k-th largest value); which of the elements equal to
    it fill the last slots is arbitrary. NaN scores are not supported.
    """
    _jit_topk_bf16_small_module().topk_transform(
        scores, seq_lens, page_table, out_page_indices, page_size
    )


@triton.jit
def _topk_transform_paged_triton_kernel(
    scores_ptr,
    seq_lens_ptr,
    page_tables_ptr,
    out_page_indices_ptr,
    out_raw_indices_ptr,
    max_seq_len,
    stride_scores,
    page_table_width,
    stride_page_tables,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WRITE_RAW: tl.constexpr,
):
    row = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + row)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for start in range(0, seq_len, BLOCK_N):
        cols = start + offs_n
        valid = cols < seq_len
        score = tl.load(
            scores_ptr + row * stride_scores + cols,
            mask=valid,
            other=float("-inf"),
        )
        score_bits = score.to(tl.uint32, bitcast=True)
        sign = tl.full(score_bits.shape, 0x80000000, tl.uint32)
        key = tl.where(
            (score_bits & sign) != 0,
            ~score_bits,
            score_bits ^ sign,
        )
        packed = (key.to(tl.uint64) << 32) | cols.to(tl.uint64)
        candidate = tl.topk(packed, K_POW2, dim=0)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(candidate, K_POW2, dim=0))
    acc = tl.sort(acc, descending=True)
    offs_k = tl.arange(0, K_POW2)
    raw = (acc & 0xFFFFFFFF).to(tl.int32)
    valid = (offs_k < K) & (offs_k < seq_len)
    page_ids = raw // PAGE_SIZE
    page_ids = tl.where(valid, page_ids, 0)
    # page_table_width is a parameter precisely so this bound exists. `raw` is only
    # known to be < seq_len, which does not imply raw // PAGE_SIZE < width -- the
    # documented form of this computation (page_table[i, j // page_size]) requires
    # it. Without it an out-of-range page_id reads past the row and the garbage page
    # id flows into extra_indices, where the sparse-attention gather -- which only
    # masks raw >= 0 and is never handed num_pages -- turns it into an illegal
    # memory access.
    in_table = valid & (page_ids >= 0) & (page_ids < page_table_width)
    pages = tl.load(
        page_tables_ptr + row * stride_page_tables + page_ids,
        mask=in_table,
        other=0,
    )
    page_indices = pages * PAGE_SIZE + raw % PAGE_SIZE
    page_indices = tl.where(in_table, page_indices, -1).to(tl.int32)
    tl.store(
        out_page_indices_ptr + row * K + offs_k,
        page_indices,
        mask=offs_k < K,
    )
    if WRITE_RAW:
        raw = tl.where(valid, raw, -1)
        tl.store(
            out_raw_indices_ptr + row * K + offs_k,
            raw,
            mask=offs_k < K,
        )


@triton.jit
def _topk_slab_kernel(
    scores_ptr,
    seq_lens_ptr,
    part_ptr,
    stride_scores,
    n_partial,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SLAB: tl.constexpr,
):
    """Top-K_POW2 of one slab of one row, written out for the merge.

    Identical arithmetic to _topk_transform_paged_triton_kernel -- the same
    bit-cast, sign-flip and bitonic-merge recurrence -- so each slab's result is
    exactly the top-K_POW2 of the scores it covers. The recurrence's
    accumulator is uint64, which matters: roughly half of all real scores pack
    to a negative int64, and a signed comparison would select the smallest.
    """
    pid = tl.program_id(0)
    row = pid // n_partial
    s = pid % n_partial
    seq_len = tl.load(seq_lens_ptr + row)
    start0 = s * SLAB
    offs_k = tl.arange(0, K_POW2)
    # Slabs past the end of a short row must still write, and must write
    # zeros: zero is below every real packed key, so it sorts last and cannot
    # displace a winner.
    if start0 >= seq_len:
        tl.store(
            part_ptr + pid * K_POW2 + offs_k,
            tl.zeros((K_POW2,), tl.uint64).to(tl.int64, bitcast=True),
        )
        return
    end = tl.minimum(start0 + SLAB, seq_len)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((K_POW2,), dtype=tl.uint64)
    for start in range(start0, end, BLOCK_N):
        cols = start + offs_n
        score = tl.load(
            scores_ptr + row * stride_scores + cols,
            mask=cols < end,
            other=float("-inf"),
        )
        score_bits = score.to(tl.uint32, bitcast=True)
        sign = tl.full(score_bits.shape, 0x80000000, tl.uint32)
        key = tl.where(
            (score_bits & sign) != 0,
            ~score_bits,
            score_bits ^ sign,
        )
        packed = (key.to(tl.uint64) << 32) | cols.to(tl.uint64)
        candidate = tl.topk(packed, K_POW2, dim=0)
        acc = tl.bitonic_merge(acc)
        acc = tl.maximum(acc, tl.topk(candidate, K_POW2, dim=0))
    acc = tl.sort(acc, descending=True)
    tl.store(part_ptr + pid * K_POW2 + offs_k, acc.to(tl.int64, bitcast=True))


@triton.jit
def _topk_merge_sort_kernel(
    in_ptr,
    out_ptr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    FANIN: tl.constexpr,
    N_OUT: tl.constexpr,
    N_IN: tl.constexpr,
):
    """out[g] = top-K of in[g*FANIN : (g+1)*FANIN], by sorting the concatenation.

    A full sort is exact by construction, which the bitonic recurrence used
    inside a slab is not -- it happens to agree on raw score blocks, but it is
    not a merge of pre-reduced lists. The sort runs on uint64 for the same
    reason the slab accumulator does.

    FANIN is capped at 2 because tl.sort's shared-memory use grows with the
    sort width and SM75 has 64 KB: FANIN=8 (4096 int64) already asks for more
    than that.
    """
    pid = tl.program_id(0)
    row = pid // N_OUT
    g = pid % N_OUT
    offs = tl.arange(0, FANIN * K_POW2)
    vals = tl.load(in_ptr + (row * N_IN + g * FANIN) * K_POW2 + offs)
    srt = tl.sort(vals.to(tl.uint64, bitcast=True), descending=True)
    # The K largest of the concatenation are its leading K_POW2 lanes, so the
    # store is masked rather than a slice; Triton cannot slice a tensor, and
    # tl.sort returns one rather than a pointer.
    tl.store(
        out_ptr + (row * N_OUT + g) * K_POW2 + offs,
        srt.to(tl.int64, bitcast=True),
        mask=offs < K_POW2,
    )


@triton.jit
def _topk_merge_tail_paged_kernel(
    in_ptr,
    seq_lens_ptr,
    page_tables_ptr,
    out_page_indices_ptr,
    out_raw_indices_ptr,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    page_table_width,
    stride_page_tables,
    WRITE_RAW: tl.constexpr,
):
    """Page-table transform of the final merge level.

    Fusing keeps the parallel path to one kernel per level; left unfused it
    costs about six small torch ops per token, which is real money against a
    130 us kernel.

    There is deliberately no sort here. The last merge level already reduced to
    a single K_POW2 group per row, so this reads exactly K_POW2 lanes at a
    K_POW2 stride. Reading a wider window would overlap the next row's answer,
    which is invisible at one row and silently swaps rows at two.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, K_POW2)
    best = tl.load(in_ptr + row * K_POW2 + offs).to(tl.uint64, bitcast=True)
    raw = (best & 0xFFFFFFFF).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + row)
    valid = (offs < K) & (offs < seq_len)
    page_ids = raw // PAGE_SIZE
    page_ids = tl.where(valid, page_ids, 0)
    # Same bound, and for the same reason, as the single-program kernel: raw is
    # only known to be < seq_len, which does not imply raw // PAGE_SIZE is a
    # valid column of the page table.
    in_table = valid & (page_ids >= 0) & (page_ids < page_table_width)
    pages = tl.load(
        page_tables_ptr + row * stride_page_tables + page_ids,
        mask=in_table,
        other=0,
    )
    page_indices = pages * PAGE_SIZE + raw % PAGE_SIZE
    page_indices = tl.where(in_table, page_indices, -1).to(tl.int32)
    tl.store(
        out_page_indices_ptr + row * K + offs,
        page_indices,
        mask=offs < K,
    )
    if WRITE_RAW:
        raw = tl.where(valid, raw, -1)
        tl.store(
            out_raw_indices_ptr + row * K + offs,
            raw,
            mask=offs < K,
        )


# Below this allocated row width a single program is already cheap and
# splitting it costs more than it saves: measured on RTX 2080 Ti, the parallel
# path runs at 0.84x of the single-program kernel at 6144 and 1.66x at 8192.
_PARALLEL_MIN_WIDTH = 8192
# Only for few rows. The single-program kernel already runs one program per row,
# so it has parallelism of its own; splitting each row is what pays, and that
# stops paying as rows grow. Measured speedup at width 37500: 5.9x at 1 row,
# 1.7x at 16, 0.93x at 32. At width 262144: 25.8x at 1, 3.4x at 16, 1.6x at 32,
# 0.80x at 128. Capping at 16 keeps every win that matters for decode and no
# regression, and bounds the buffers at 8 MB per shape.
_PARALLEL_MAX_ROWS = 16
# 32 partials win below 65536, 64 above it. Both are powers of two so the merge
# level count stays fixed per branch.
_PARALLEL_N_PARTIAL_SMALL = 32
_PARALLEL_N_PARTIAL_LARGE = 64
_PARALLEL_N_PARTIAL_SPLIT = 65536
_PARALLEL_FANIN = 2
# Ceiling on the scratch the cache may hold. Entries are never freed (see
# below), so without this an unusual sequence of graph buckets would keep
# growing resident memory on a pipeline that has under a gigabyte of headroom.
_PARALLEL_BUFFER_BUDGET_ELEMS = 8 * 1024 * 1024

# Kill switch, so the parallel path can be turned off without a rebuild. It is
# the only way to attribute a decode regression to this change or to whatever
# else moved, which is not always decidable from the outside.
_PARALLEL_ENABLED = os.environ.get("SGLANG_OPT_SM75_PARALLEL_TOPK", "1") != "0"

# One-shot, env-gated trace of which path each shape took and how wide the
# dispatch actually sees things. /proc/<pid>/environ is not evidence for a
# forked scheduler -- it renames itself, so the env it reports is stale -- so
# the only trustworthy record of what a worker did is its own log.
_PARALLEL_DIAG = os.environ.get("SGLANG_OPT_SM75_TOPK_DIAG", "0") == "1"
_PARALLEL_DIAG_SEEN: set = set()


def _diag(msg):
    import logging

    logging.getLogger(__name__).warning("[dsv4-topk-diag] %s", msg)

# Buffers are keyed by shape and never released. A captured CUDA graph bakes the
# pointers in, so reallocating under a live graph would silently make it read
# someone else's memory. Falling back to the single-program kernel is the safe
# way to run out of budget.
_PARALLEL_BUFFERS: dict = {}
_PARALLEL_BUFFER_ELEMS = 0


def _parallel_buffers(rows, n_partial, fanin, K_POW2, device):
    """Cached scratch for one shape, or None if the budget is exhausted.

    Cached rather than pooled on purpose: a captured CUDA graph holds the
    pointers, so a buffer freed and reallocated at a different address would
    turn a graph replay into a read of unrelated memory. Holding it is the only
    way to keep the capture valid.
    """
    global _PARALLEL_BUFFER_ELEMS
    key = (rows, n_partial, fanin, K_POW2, device.index)
    bufs = _PARALLEL_BUFFERS.get(key)
    if bufs is not None:
        return bufs
    part_elems = rows * n_partial * K_POW2
    level_elems = 0
    cur_n = n_partial
    while cur_n > 1:
        cur_n = triton.cdiv(cur_n, fanin)
        level_elems += rows * cur_n * fanin * K_POW2
    if _PARALLEL_BUFFER_ELEMS + part_elems + level_elems > \
            _PARALLEL_BUFFER_BUDGET_ELEMS:
        return None
    part = torch.zeros(part_elems, dtype=torch.int64, device=device)
    levels = []
    cur_n = n_partial
    while cur_n > 1:
        nxt = triton.cdiv(cur_n, fanin)
        levels.append(
            torch.zeros(
                rows * nxt * fanin * K_POW2, dtype=torch.int64, device=device
            )
        )
        cur_n = nxt
    bufs = (part, levels)
    _PARALLEL_BUFFERS[key] = bufs
    _PARALLEL_BUFFER_ELEMS += part_elems + level_elems
    return bufs


def _topk_transform_paged_parallel(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    raw_indices: torch.Tensor,
    write_raw: bool,
    page_size: int,
    K: int,
    K_POW2: int,
    BLOCK_N: int,
) -> bool:
    """Returns False if it declined to run, having done nothing."""
    rows, width = scores.shape
    fanin = _PARALLEL_FANIN
    n_partial = (
        _PARALLEL_N_PARTIAL_LARGE
        if width >= _PARALLEL_N_PARTIAL_SPLIT
        else _PARALLEL_N_PARTIAL_SMALL
    )
    cached = _parallel_buffers(rows, n_partial, fanin, K_POW2, scores.device)
    if cached is None:
        return False
    part, levels = cached
    n_blocks = triton.cdiv(width, BLOCK_N)
    slab = triton.cdiv(n_blocks, n_partial) * BLOCK_N
    _topk_slab_kernel[(rows * n_partial,)](
        scores,
        seq_lens,
        part,
        scores.stride(0),
        n_partial,
        K=K,
        K_POW2=K_POW2,
        BLOCK_N=BLOCK_N,
        SLAB=slab,
        num_warps=4,
        num_stages=1,
    )
    cur, cur_n = part, n_partial
    for lvl, dst in enumerate(levels):
        nxt_n = triton.cdiv(cur_n, fanin)
        _topk_merge_sort_kernel[(rows * nxt_n,)](
            cur,
            dst,
            K=K,
            K_POW2=K_POW2,
            FANIN=fanin,
            N_OUT=nxt_n,
            N_IN=cur_n,
            num_warps=4,
            num_stages=1,
        )
        cur, cur_n = dst, nxt_n
    _topk_merge_tail_paged_kernel[(rows,)](
        cur,
        seq_lens,
        page_tables,
        out_page_indices,
        raw_indices,
        K=K,
        K_POW2=K_POW2,
        PAGE_SIZE=page_size,
        page_table_width=page_tables.shape[1],
        stride_page_tables=page_tables.stride(0),
        WRITE_RAW=write_raw,
        num_warps=4,
        num_stages=1,
    )
    return True


def topk_transform_paged_triton(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    assert scores.ndim == 2
    assert scores.dtype == torch.float32
    assert scores.stride(1) == 1
    assert scores.stride(0) > 0
    assert seq_lens.ndim == 1 and seq_lens.shape[0] == scores.shape[0]
    assert page_tables.ndim == 2 and page_tables.shape[0] == scores.shape[0]
    assert page_tables.stride(1) == 1
    assert out_page_indices.ndim == 2
    assert out_page_indices.shape[0] == scores.shape[0]
    assert out_page_indices.dtype == torch.int32
    K = out_page_indices.shape[1]
    K_POW2 = triton.next_power_of_2(K)
    assert 0 < K <= 1024
    assert page_size > 0
    assert page_size & (page_size - 1) == 0
    if out_raw_indices is None:
        raw_indices = out_page_indices
        write_raw = False
    else:
        raw_indices = out_raw_indices
        assert raw_indices.shape == out_page_indices.shape
        assert raw_indices.dtype == torch.int32
        write_raw = True
    BLOCK_N = max(256, min(K_POW2, 1024))
    if _PARALLEL_DIAG:
        key = (scores.shape[1], scores.shape[0])
        if key not in _PARALLEL_DIAG_SEEN:
            _PARALLEL_DIAG_SEEN.add(key)
            _diag(
                f"enabled={_PARALLEL_ENABLED} width={key[0]} rows={key[1]} "
                f"K={K} "
                f"path={'parallel' if (_PARALLEL_ENABLED and key[0] >= _PARALLEL_MIN_WIDTH and key[1] <= _PARALLEL_MAX_ROWS) else 'single'}"
            )
    # Dispatch on the allocated row width and row count, not on seq_lens:
    # inside a captured CUDA graph seq_lens is a device buffer whose values
    # change per replay, so it cannot select a launch. Width and row count are
    # fixed for a graph bucket, and the parallel path is correct for any ragged
    # lengths under the width.
    if (
        _PARALLEL_ENABLED
        and scores.shape[1] >= _PARALLEL_MIN_WIDTH
        and scores.shape[0] <= _PARALLEL_MAX_ROWS
        and _topk_transform_paged_parallel(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            raw_indices,
            write_raw,
            page_size,
            K,
            K_POW2,
            BLOCK_N,
        )
    ):
        return
    grid = (scores.shape[0],)
    _topk_transform_paged_triton_kernel[grid](
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        raw_indices,
        scores.shape[1],
        scores.stride(0),
        page_tables.shape[1],
        page_tables.stride(0),
        K=K,
        K_POW2=K_POW2,
        BLOCK_N=BLOCK_N,
        PAGE_SIZE=page_size,
        WRITE_RAW=write_raw,
        num_warps=4,
        num_stages=1,
    )


def topk_transform_paged(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    elif is_xpu():
        torch.ops.sgl_kernel.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


def topk_transform_paged_torch(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """The torch ``topk_transform_paged``: top-``k`` (``k = out_page_indices.shape[1]``)
    of each row within its first ``seq_lens[b]`` columns, ascending, as pool slots
    through ``page_tables`` and as positions when given; ``-1`` padded."""
    topk = out_page_indices.shape[1]
    columns = torch.arange(scores.shape[1], device=seq_lens.device)
    lens_c = seq_lens.unsqueeze(-1)
    # Columns past a row's length hold garbage.
    s = scores.masked_fill(columns[None, :] >= lens_c, -torch.inf)
    idx = s.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    reach = idx < lens_c
    slots = page_tables.gather(-1, idx // page_size) * page_size + (idx % page_size)
    out_page_indices.copy_(torch.where(reach, slots, -1).to(torch.int32))
    if out_raw_indices is not None:
        out_raw_indices.copy_(torch.where(reach, idx, -1).to(torch.int32))


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = -1) -> torch.Tensor:
    """
    Preprocess the per-batch routing plan for :func:`topk_transform_paged_v2`.
    NOTE: every entry of ``seq_lens`` must be NON-NEGATIVE.

    :param static_threshold: If a batch item has `seq_len` > `static_threshold`,
                             prefer the cluster implementation.
                             Negative number means internal heuristic.
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_v2_plan_is_written(seq_lens: torch.Tensor) -> bool:
    """Whether :func:`plan_topk_v2` writes a plan for these lengths. Small
    batches and devices without clusters leave the plan buffer untouched."""
    probe = torch.full(
        (seq_lens.shape[0] + 1, _PLAN_METADATA_INTS_PER_BATCH),
        -1,
        dtype=torch.int32,
        device=seq_lens.device,
    )
    _jit_topk_v2_module().topk_plan(seq_lens, probe, -1)
    return probe[0, 1].item() != -1


def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """Ragged (prefill) fused top-k for a contiguous-KV score matrix.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks =
    row_starts[i]``, 0 when ``row_starts`` is omitted) and writes
    ``selected_position + out_offsets[i]`` into ``out_indices``, ``-1`` padded.
    With the production convention ``out_offsets == row_starts`` that is the
    column index itself, i.e. the token's slot in the batch's flattened KV.

    Unlike :func:`topk_transform_paged_v2` this needs no page table and no plan
    (the cluster path only pays off for very few rows, and prefill has many).

    NOTE: ``scores`` is written in place -- the <= 3 columns ahead of each
    row's window that the 16-byte-aligned read base pulls in are masked out.
    They are invalid for that row and the buffer must have no other consumer.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.
    """
    if is_xpu():
        torch.ops.sgl_kernel.topk_transform_ragged(
            scores,
            seq_lens,
            out_indices,
            out_offsets,
            row_starts,
        )
        return
    module = _jit_topk_v2_module()
    module.topk_transform_ragged(scores, seq_lens, row_starts, out_offsets, out_indices)


def topk_transform_paged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + optional page-table transform (DeepSeek-V4 top-k v2 kernel).

    Output mode is chosen from ``page_tables`` and ``out_raw_indices`` and
    resolved to a device-side template parameter, so an unused page-table gather
    is compiled out rather than skipped at runtime:

    * ``page_tables=None`` -- ``out_page_indices`` receives the raw selected
      indices and no page table is read.
    * ``page_tables`` given -- ``out_page_indices`` receives the page-table
      transform of them.
    * Both outputs given -- ``out_page_indices`` receives the page-table
      transform and ``out_raw_indices`` receives the selected raw indices.

    For the packed (DSA extend prefill) layout see
    :func:`topk_transform_packed_v2`.

    NOTE: every entry of `seq_lens` must be NON-NEGATIVE, and `metadata` must
    come from :func:`plan_topk_v2` over the same `seq_lens` values.
    A length of 0 is the valid way to express "no tokens": the row takes the
    trivial path and the output is guaranteed to be all -1.
    """
    if is_xpu():
        if out_raw_indices is not None:
            topk_transform_paged(
                scores,
                seq_lens,
                page_tables,
                out_page_indices,
                page_size,
                out_raw_indices,
            )
            return
        torch.ops.sgl_kernel.topk_transform_paged(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            metadata,
        )
        return
    module = _jit_topk_v2_module()
    module.topk_transform_paged(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )


def topk_transform_packed_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    *,
    row_starts: torch.Tensor,
    row_to_batch: Optional[torch.Tensor] = None,
) -> None:
    """Packed (DSA extend prefill) fused top-k + page-table transform.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]``
    (``ks = row_starts[i]``) and writes the page-table transform of the selected
    row-local positions into ``out_page_indices``, ``-1`` padded. Prefill expands
    one request into many query-token rows, so ``row_to_batch[i]`` (optional,
    ``(rows,)`` int32) names the ``page_tables`` row of the request row ``i``
    belongs to; omitting it indexes the table by score row. ``row_to_batch`` is
    not range-checked.

    This is :func:`topk_transform_ragged_v2` with a page-table output instead of
    an additive offset. Like ragged, it dispatches the implementation per row at
    runtime, so it needs no plan and no :func:`plan_topk_v2` metadata.

    NOTE: ``scores`` is MODIFIED IN PLACE -- the <= 3 columns ahead of each row's
    window that the 16-byte-aligned read base pulls in are masked out. They are
    invalid for that row and the buffer must have no other consumer, so do not
    pass a view with overlapping rows.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.

    ROCm only: the kernel is compiled under ``USE_ROCM`` so that CUDA and XPU
    builds are untouched. Nothing in it is AMD-specific -- no non-ROCm caller
    produces this layout today.
    """
    assert is_hip_runtime(), "topk_transform_packed_v2 is compiled under USE_ROCM only"
    module = _jit_topk_v2_module()
    module.topk_transform_packed(
        scores,
        seq_lens,
        row_starts,
        page_tables,
        out_page_indices,
        page_size,
        row_to_batch,
    )


def topk_transform_sparse(
    logits: torch.Tensor,
    valid_lens: torch.Tensor,
    blocks: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """Top-``k`` (``k = out_indices.shape[1]``) of each row of the bf16 sparse
    ``logits`` within its first ``valid_lens[b]`` columns, ``-1`` padded, unordered;
    column ``j`` is written as ``blocks[b, j // 8] * 8 + j % 8``: pool slots for the
    published blocks as pool slots / 8, compressed positions for logical ids."""
    topk_transform_bf16_small(
        logits, valid_lens, blocks, out_indices, CANDIDATE_BLOCK_SIZE
    )
