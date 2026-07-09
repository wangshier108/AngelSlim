"""Triton implementation of the STEM sparse attention pipeline.

Implements:
  - stem_oam_prep_paged_kv: Prepare K_flat and V_bias from paged FP8 KV cache
  - stem_oam_prep_varlen_q: Prepare Q_flat from packed FP8 Q tensor
  - stem_oam_gemm: Block-level scoring GEMM with fused causal mask
  - stem_tpd: Top-k Policy Denoising mask generation
  - stem_paged_kv: End-to-end fused pipeline
    - attention_with_kvcache_blocksparse_prefill_fp8: Paged FP8 attention with
            optional block-sparse mask (dense/sparse prefill)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# Block-sparse tile granularity for prefill attention.
BSA_TILE = 128


# ===========================================================================
# Triton Kernel: K_flat computation (one group per program)
# ===========================================================================


@triton.jit
def _stem_prep_kflat_kernel(
    # Pointers
    Kcache,          # [total_blocks, kv_block_size, num_kv_heads, dim_qk] fp8
    Kscale,          # [1] fp32 (per-tensor scale)
    KvIndices,       # [num_batch, max_blocks_per_req] int32
    KvSeqLens,       # [num_batch] int32
    Kflat,           # [num_batch, num_kv_heads, max_Kb, flat_dim] bf16
    # Strides for Kcache: [total_blocks, kv_block_size, num_kv_heads, dim_qk]
    stride_kc_block, stride_kc_token, stride_kc_head,
    # Strides for KvIndices
    stride_idx_batch,
    # Strides for Kflat
    stride_kf_batch, stride_kf_head, stride_kf_kb,
    # Dimensions
    kv_block_size: tl.constexpr,
    dim_qk: tl.constexpr,
    stem_block_size: tl.constexpr,
    stem_stride: tl.constexpr,
    max_Kb: tl.constexpr,
    samples_per_group: tl.constexpr,
):
    """Compute K_flat: group-sum K vectors with reversed group order.

    Grid: (max_Kb * stem_stride, num_kv_heads, num_batch)
    Each program computes one group of one stem-block for one head/batch.
    """
    pid_kg = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    kb_idx = pid_kg // stem_stride
    group_local = pid_kg % stem_stride

    # Get the actual KV sequence length for this batch
    kv_len = tl.load(KvSeqLens + batch_idx)
    actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size

    # Skip if this block is beyond the actual range
    if kb_idx >= actual_Kb:
        return

    # Load the per-tensor K scale
    k_scale_val = tl.load(Kscale)

    # Reversed group index for anti-diagonal scoring
    reversed_group = stem_stride - 1 - group_local

    # Accumulate interleaved group samples: g, g+stride, g+2*stride, ...
    offs_d = tl.arange(0, dim_qk)  # [dim_qk]
    acc = tl.zeros([dim_qk], dtype=tl.float32)

    for t in range(samples_per_group):
        token_in_block = group_local + t * stem_stride
        global_token = kb_idx * stem_block_size + token_in_block

        if global_token < kv_len:
            page_idx = global_token // kv_block_size
            token_in_page = global_token % kv_block_size

            phys_block = tl.load(KvIndices + batch_idx * stride_idx_batch + page_idx)

            k_ptrs = (
                Kcache
                + phys_block * stride_kc_block
                + token_in_page * stride_kc_token
                + head_idx * stride_kc_head
                + offs_d
            )
            k_vals = tl.load(k_ptrs).to(tl.float32)
            acc += k_vals * k_scale_val

    # Store to Kflat with reversed group ordering
    out_ptrs = (
        Kflat
        + batch_idx * stride_kf_batch
        + head_idx * stride_kf_head
        + kb_idx * stride_kf_kb
        + reversed_group * dim_qk
        + offs_d
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16))


# ===========================================================================
# Triton Kernel: V_bias computation
# ===========================================================================


@triton.jit
def _stem_prep_vbias_kernel(
    # Pointers
    Vcache,          # [total_blocks, kv_block_size, num_kv_heads, dim_v] fp8
    Vscale,          # [1] fp32
    KvIndices,       # [num_batch, max_blocks_per_req] int32
    KvSeqLens,       # [num_batch] int32
    Vbias,           # [num_batch, num_kv_heads, max_Kb] fp32
    # Strides
    stride_vc_block, stride_vc_token, stride_vc_head,
    stride_idx_batch,
    stride_vb_batch, stride_vb_head,
    # Dimensions
    kv_block_size: tl.constexpr,
    dim_v: tl.constexpr,
    stem_block_size: tl.constexpr,
    stem_stride: tl.constexpr,
    max_Kb: tl.constexpr,
    groups_per_block: tl.constexpr,
):
    """Compute per-block max V L2-norm for V_bias.

    Grid: (max_Kb, num_kv_heads, num_batch)
    """
    kb_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    kv_len = tl.load(KvSeqLens + batch_idx)
    actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size

    if kb_idx >= actual_Kb:
        tl.store(Vbias + batch_idx * stride_vb_batch + head_idx * stride_vb_head + kb_idx, 0.0)
        return

    v_scale_val = tl.load(Vscale)
    offs_d = tl.arange(0, dim_v)

    max_norm_sq = 0.0

    for g in range(groups_per_block):
        for t in range(stem_stride):
            token_in_block = g * stem_stride + t
            global_token = kb_idx * stem_block_size + token_in_block

            if global_token < kv_len:
                page_idx = global_token // kv_block_size
                token_in_page = global_token % kv_block_size
                phys_block = tl.load(KvIndices + batch_idx * stride_idx_batch + page_idx)

                v_ptrs = (
                    Vcache
                    + phys_block * stride_vc_block
                    + token_in_page * stride_vc_token
                    + head_idx * stride_vc_head
                    + offs_d
                )
                v_vals = tl.load(v_ptrs).to(tl.float32) * v_scale_val
                norm_sq = tl.sum(v_vals * v_vals)
                max_norm_sq = tl.maximum(max_norm_sq, norm_sq)

    max_norm = tl.sqrt(max_norm_sq)
    log_norm = tl.log(max_norm + 1e-6)
    tl.store(Vbias + batch_idx * stride_vb_batch + head_idx * stride_vb_head + kb_idx, log_norm)


# ===========================================================================
# Triton Kernel: Q_flat prep (one group per program)
# ===========================================================================


@triton.jit
def _stem_prep_qflat_kernel(
    # Pointers
    Qfp8,            # [total_tokens, num_q_heads, dim_qk] fp8
    Qscale,          # [num_batch, num_q_heads, max_seq_q_pad] fp32
    QSeqLens,        # [num_batch] int32
    CuSeqlensQ,      # [num_batch + 1] int32
    Qflat,           # [num_batch, num_q_heads, max_Qb, flat_dim] bf16
    # Strides for Q
    stride_q_token, stride_q_head,
    # Strides for Qscale
    stride_qs_batch, stride_qs_head, stride_qs_seq,
    # Strides for Qflat
    stride_qf_batch, stride_qf_head, stride_qf_qb,
    # Dimensions
    dim_qk: tl.constexpr,
    stem_block_size: tl.constexpr,
    stem_stride: tl.constexpr,
    max_Qb: tl.constexpr,
    samples_per_group: tl.constexpr,
):
    """Compute Q_flat: weighted group-sum of Q tokens (natural group order).

    Grid: (max_Qb * stem_stride, num_q_heads, num_batch)
    """
    pid_qg = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    qb_idx = pid_qg // stem_stride
    group_local = pid_qg % stem_stride

    q_len = tl.load(QSeqLens + batch_idx)
    actual_Qb = (q_len + stem_block_size - 1) // stem_block_size

    if qb_idx >= actual_Qb:
        return

    cu_offset = tl.load(CuSeqlensQ + batch_idx)
    offs_d = tl.arange(0, dim_qk)

    acc = tl.zeros([dim_qk], dtype=tl.float32)

    for t in range(samples_per_group):
        token_in_block = group_local + t * stem_stride
        global_token = qb_idx * stem_block_size + token_in_block

        if global_token < q_len:
            global_token_packed = cu_offset + global_token

            q_ptrs = (
                Qfp8
                + global_token_packed * stride_q_token
                + head_idx * stride_q_head
                + offs_d
            )
            q_vals = tl.load(q_ptrs).to(tl.float32)

            q_scale_val = tl.load(
                Qscale
                + batch_idx * stride_qs_batch
                + head_idx * stride_qs_head
                + global_token * stride_qs_seq
            )

            acc += q_vals * q_scale_val

    # Store in natural group order
    out_ptrs = (
        Qflat
        + batch_idx * stride_qf_batch
        + head_idx * stride_qf_head
        + qb_idx * stride_qf_qb
        + group_local * dim_qk
        + offs_d
    )
    tl.store(out_ptrs, acc.to(tl.bfloat16))


# ===========================================================================
# Triton Kernel: OAM GEMM (Qflat @ Kflat^T + Vbias)
# ===========================================================================


@triton.jit
def _stem_oam_gemm_kernel(
    # Pointers
    Qflat,           # [num_batch, num_q_heads, max_Qb, flat_dim] bf16
    Kflat,           # [num_batch, num_kv_heads, max_Kb, flat_dim] bf16
    Vbias,           # [num_batch, num_kv_heads, max_Kb] fp32
    QSeqLens,        # [num_batch] int32
    KvSeqLens,       # [num_batch] int32
    BlockLogits,     # [num_batch, num_q_heads, max_Qb, max_Kb] bf16
    # Strides for Qflat
    stride_qf_batch, stride_qf_head, stride_qf_qb,
    # Strides for Kflat
    stride_kf_batch, stride_kf_head, stride_kf_kb,
    # Strides for Vbias
    stride_vb_batch, stride_vb_head,
    # Strides for BlockLogits
    stride_bl_batch, stride_bl_head, stride_bl_qb,
    # Dimensions
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_Qb: tl.constexpr,
    max_Kb: tl.constexpr,
    flat_dim: tl.constexpr,
    stem_stride: tl.constexpr,
    stem_block_size: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute block_logits = FrobScale * (Qflat @ Kflat^T) + Vbias.

    Grid: (ceil(max_Qb/BLOCK_M), ceil(max_Kb/BLOCK_N), num_batch * num_q_heads)
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)
    batch_idx = pid_bh // num_q_heads
    head_q_idx = pid_bh % num_q_heads
    # GQA head mapping
    heads_per_group = num_q_heads // num_kv_heads
    head_kv_idx = head_q_idx // heads_per_group

    q_len = tl.load(QSeqLens + batch_idx)
    kv_len = tl.load(KvSeqLens + batch_idx)
    actual_Qb = (q_len + stem_block_size - 1) // stem_block_size
    actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    samples_per_group = stem_block_size // stem_stride
    frob_scale = 1.0 / (samples_per_group * samples_per_group)

    # Compute dot product tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, flat_dim, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load Q tile [BLOCK_M, BLOCK_K]
        q_ptrs = (
            Qflat
            + batch_idx * stride_qf_batch
            + head_q_idx * stride_qf_head
            + offs_m[:, None] * stride_qf_qb
            + offs_k[None, :]
        )
        q_mask = (offs_m[:, None] < actual_Qb) & (offs_k[None, :] < flat_dim)
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Load K tile [BLOCK_K, BLOCK_N]
        k_ptrs = (
            Kflat
            + batch_idx * stride_kf_batch
            + head_kv_idx * stride_kf_head
            + offs_n[None, :] * stride_kf_kb
            + offs_k[:, None]
        )
        k_mask = (offs_n[None, :] < actual_Kb) & (offs_k[:, None] < flat_dim)
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc += tl.dot(q_tile, k_tile)

    acc = acc * frob_scale

    # Add Vbias
    vbias_ptrs = Vbias + batch_idx * stride_vb_batch + head_kv_idx * stride_vb_head + offs_n
    vbias_mask = offs_n < actual_Kb
    vbias_vals = tl.load(vbias_ptrs, mask=vbias_mask, other=0.0)
    acc = acc + vbias_vals[None, :]

    # Causal mask
    if causal:
        block_offset = actual_Kb - actual_Qb
        causal_mask = offs_n[None, :] > (offs_m[:, None] + block_offset)
        acc = tl.where(causal_mask, float("-inf"), acc)

    # Out-of-range mask
    valid_mask = (offs_m[:, None] < actual_Qb) & (offs_n[None, :] < actual_Kb)
    acc = tl.where(valid_mask, acc, float("-inf"))

    # Store
    out_ptrs = (
        BlockLogits
        + batch_idx * stride_bl_batch
        + head_q_idx * stride_bl_head
        + offs_m[:, None] * stride_bl_qb
        + offs_n[None, :]
    )
    out_mask = (offs_m[:, None] < max_Qb) & (offs_n[None, :] < max_Kb)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# ===========================================================================
# Triton Kernel: TPD (Top-k Policy Denoising) mask generation
# ===========================================================================


@triton.jit
def _stem_tpd_row_kernel(
    # Pointers
    BlockLogits,     # [num_batch, num_q_heads, max_Qb, max_Kb] bf16
    QSeqLens,        # [num_batch] int32
    KvSeqLens,       # [num_batch] int32
    NumPromptTokens, # [num_batch] int32
    Mask,            # [num_batch, num_q_heads, max_Qb, max_Kb] uint8
    # Strides
    stride_bl_batch, stride_bl_head, stride_bl_qb,
    stride_m_batch, stride_m_head, stride_m_qb,
    # Dimensions & params
    max_Kb: tl.constexpr,
    block_size: tl.constexpr,
    alpha: tl.constexpr,
    initial_blocks: tl.constexpr,
    window_size: tl.constexpr,
    k_block_num_rate_medium: tl.constexpr,
    k_block_num_bias_medium: tl.constexpr,
    k_block_num_rate_large: tl.constexpr,
    k_block_num_bias_large: tl.constexpr,
):
    """Generate one row of the sparse block mask via top-k.

    Grid: (max_Qb, num_q_heads, num_batch)
    """
    qb_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    kv_len = tl.load(KvSeqLens + batch_idx)
    q_len = tl.load(QSeqLens + batch_idx)
    prompt_tokens = tl.load(NumPromptTokens + batch_idx)

    actual_Qb = (q_len + block_size - 1) // block_size
    actual_Kb = (kv_len + block_size - 1) // block_size

    offs_kb = tl.arange(0, max_Kb)

    # Zero out row if out of range
    if qb_idx >= actual_Qb:
        out_ptrs = (
            Mask + batch_idx * stride_m_batch + head_idx * stride_m_head
            + qb_idx * stride_m_qb + offs_kb
        )
        tl.store(out_ptrs, tl.zeros([max_Kb], dtype=tl.uint8), mask=offs_kb < max_Kb)
        return

    # Budget from CUDA k_schedule + linear decay (chunked-prefill aligned).
    prompt_Kb = (prompt_tokens + block_size - 1) // block_size
    k_val = prompt_Kb
    if prompt_Kb >= 160:
        k_val = (k_block_num_rate_large * prompt_Kb + k_block_num_bias_large).to(tl.int32)
    elif prompt_Kb >= 56:
        k_val = (k_block_num_rate_medium * prompt_Kb + k_block_num_bias_medium).to(tl.int32)

    k_val = tl.maximum(k_val, 1)

    kb_offset = actual_Kb - actual_Qb
    q_pos = qb_idx + kb_offset
    decay_len = prompt_Kb - k_val
    k_end = k_val.to(tl.float32) * alpha

    use_decay = (q_pos >= k_val) & (decay_len > 1)
    t = (q_pos - k_val).to(tl.float32) / (decay_len - 1).to(tl.float32)
    budget_decay = tl.math.floor(k_val.to(tl.float32) + t * (k_end - k_val.to(tl.float32))).to(
        tl.int32
    )
    row_budget = tl.where(use_decay, budget_decay, k_val)
    row_budget = tl.maximum(1, tl.minimum(row_budget, k_val))

    # Load entire row of block_logits
    logit_ptrs = (
        BlockLogits + batch_idx * stride_bl_batch + head_idx * stride_bl_head
        + qb_idx * stride_bl_qb + offs_kb
    )
    logits = tl.load(logit_ptrs, mask=offs_kb < max_Kb, other=float("-inf")).to(tl.float32)

    # Select among finite logits only, for columns within actual_Kb.
    finite_mask = (offs_kb < actual_Kb) & tl.isfinite(logits)
    total_finite = tl.sum(finite_mask.to(tl.int32), axis=0)

    # Binary search threshold only when budget does not cover all finite entries.
    inf = float("inf")
    ninf = float("-inf")
    lo = tl.min(tl.where(finite_mask, logits, inf), axis=0)
    hi = tl.max(tl.where(finite_mask, logits, ninf), axis=0)

    for _ in range(32):
        mid = (lo + hi) * 0.5
        count = tl.sum((finite_mask & (logits >= mid)).to(tl.int32), axis=0)
        lo = tl.where(count >= row_budget, mid, lo)
        hi = tl.where(count >= row_budget, hi, mid)

    threshold = lo
    selected = tl.where(total_finite <= row_budget, finite_mask, finite_mask & (logits >= threshold))
    selected = selected.to(tl.uint8)

    # Always keep initial blocks.
    initial_mask = (offs_kb < initial_blocks) & (offs_kb < actual_Kb)
    selected = tl.where(initial_mask, tl.full([max_Kb], 1, dtype=tl.uint8), selected)

    # Always keep recent window and diagonal aligned to KV index space.
    diag_col = qb_idx + kb_offset
    window_mask = (offs_kb <= diag_col) & (offs_kb > (diag_col - window_size)) & (offs_kb < actual_Kb)
    selected = tl.where(window_mask, tl.full([max_Kb], 1, dtype=tl.uint8), selected)
    diag_mask = (offs_kb == diag_col) & (offs_kb < actual_Kb)
    selected = tl.where(diag_mask, tl.full([max_Kb], 1, dtype=tl.uint8), selected)

    # Zero positions beyond valid KV range.
    out_valid = offs_kb < actual_Kb
    selected = tl.where(out_valid, selected, tl.zeros([max_Kb], dtype=tl.uint8))

    # Store
    out_ptrs = (
        Mask + batch_idx * stride_m_batch + head_idx * stride_m_head
        + qb_idx * stride_m_qb + offs_kb
    )
    tl.store(out_ptrs, selected, mask=offs_kb < max_Kb)


# ===========================================================================
# Python wrapper functions
# ===========================================================================


@triton.jit
def _blocksparse_paged_attn_kernel(
    # Pointers
    Q,               # [total_seq, num_head_q, dim_qk] fp8
    Kcache,          # [num_pages, page_size, num_head_kv, dim_qk] fp8
    Vcache,          # [num_pages, page_size, num_head_kv, dim_v] fp8
    Qscale,          # [num_batch, num_head_q, max_seq_q_pad] fp32
    Kscale,          # [1] fp32
    Vscale,          # [1] fp32
    CuSeqlensQ,      # [num_batch + 1] int32
    BlockIds,        # [num_batch, max_blocks_per_req] int32
    SeqlensKvcache,  # [num_batch] int32
    BlockMask,       # [num_batch, num_head_q, max_tile_m, max_tile_kv] uint8
    Output,          # [total_seq, num_head_q, dim_v] bf16
    # Strides
    stride_q_seq, stride_q_head,
    stride_kc_page, stride_kc_token, stride_kc_head,
    stride_vc_page, stride_vc_token, stride_vc_head,
    stride_qs_batch, stride_qs_head, stride_qs_seq,
    stride_bi_batch,
    stride_bm_batch, stride_bm_head, stride_bm_qb,
    stride_o_seq, stride_o_head,
    # Dimensions
    num_head_q: tl.constexpr,
    num_head_kv: tl.constexpr,
    dim_qk: tl.constexpr,
    dim_v: tl.constexpr,
    page_size: tl.constexpr,
    max_num_pages: tl.constexpr,
    has_block_mask: tl.constexpr,
    BSA_TILE_SIZE: tl.constexpr,
):
    """Block-sparse paged attention kernel.

    Grid: (max_seqlens_q, num_head_q, num_batch)
    Each program handles one Q token for one head in one batch.
    """
    q_local_idx = tl.program_id(0)
    head_q_idx = tl.program_id(1)
    batch_idx = tl.program_id(2)

    heads_per_group = num_head_q // num_head_kv
    head_kv_idx = head_q_idx // heads_per_group

    cu_q_start = tl.load(CuSeqlensQ + batch_idx)
    cu_q_end = tl.load(CuSeqlensQ + batch_idx + 1)
    q_len = cu_q_end - cu_q_start
    kv_history_len = tl.load(SeqlensKvcache + batch_idx)
    total_kv_len = kv_history_len + q_len

    if q_local_idx >= q_len:
        return

    global_q_pos = kv_history_len + q_local_idx
    q_tile_idx = q_local_idx // BSA_TILE_SIZE

    q_global_pos = cu_q_start + q_local_idx
    offs_d = tl.arange(0, dim_qk)
    q_ptrs = Q + q_global_pos * stride_q_seq + head_q_idx * stride_q_head + offs_d
    q_vec = tl.load(q_ptrs).to(tl.float32)

    q_scale_val = tl.load(
        Qscale + batch_idx * stride_qs_batch + head_q_idx * stride_qs_head + q_local_idx * stride_qs_seq
    )
    k_scale_val = tl.load(Kscale)
    v_scale_val = tl.load(Vscale)

    rsqrt_d = 1.0 / tl.sqrt(float(dim_qk))
    attn_scale = q_scale_val * k_scale_val * rsqrt_d

    m_i = float("-inf")
    l_i = 0.0
    offs_v = tl.arange(0, dim_v)
    acc = tl.zeros([dim_v], dtype=tl.float32)

    for page_idx in range(max_num_pages):
        page_kv_start = page_idx * page_size
        if page_kv_start < total_kv_len:
            kv_tile_of_page = page_kv_start // BSA_TILE_SIZE

            mask_ok = True
            if has_block_mask:
                bm_val = tl.load(
                    BlockMask
                    + batch_idx * stride_bm_batch
                    + head_q_idx * stride_bm_head
                    + q_tile_idx * stride_bm_qb
                    + kv_tile_of_page
                )
                mask_ok = bm_val != 0

                next_tile = (page_kv_start + page_size - 1) // BSA_TILE_SIZE
                if next_tile != kv_tile_of_page:
                    bm_val2 = tl.load(
                        BlockMask
                        + batch_idx * stride_bm_batch
                        + head_q_idx * stride_bm_head
                        + q_tile_idx * stride_bm_qb
                        + next_tile
                    )
                    mask_ok = mask_ok | (bm_val2 != 0)

            if mask_ok:
                phys_page = tl.load(BlockIds + batch_idx * stride_bi_batch + page_idx)

                for t in range(page_size):
                    global_kv_pos = page_kv_start + t
                    valid = (global_kv_pos < total_kv_len) & (global_kv_pos <= global_q_pos)

                    if has_block_mask:
                        token_tile = global_kv_pos // BSA_TILE_SIZE
                        token_bm = tl.load(
                            BlockMask
                            + batch_idx * stride_bm_batch
                            + head_q_idx * stride_bm_head
                            + q_tile_idx * stride_bm_qb
                            + token_tile
                        )
                        valid = valid & (token_bm != 0)

                    if valid:
                        k_ptrs = (
                            Kcache
                            + phys_page * stride_kc_page
                            + t * stride_kc_token
                            + head_kv_idx * stride_kc_head
                            + offs_d
                        )
                        k_vec = tl.load(k_ptrs).to(tl.float32)
                        score = tl.sum(q_vec * k_vec) * attn_scale

                        m_new = tl.maximum(m_i, score)
                        alpha = tl.exp(m_i - m_new)
                        p = tl.exp(score - m_new)

                        l_i = l_i * alpha + p
                        acc = acc * alpha

                        v_ptrs = (
                            Vcache
                            + phys_page * stride_vc_page
                            + t * stride_vc_token
                            + head_kv_idx * stride_vc_head
                            + offs_v
                        )
                        v_vec = tl.load(v_ptrs).to(tl.float32)
                        acc = acc + p * v_vec

                        m_i = m_new

    l_i_safe = tl.where(l_i > 0.0, l_i, 1.0)
    output_vec = acc * (v_scale_val / l_i_safe)

    o_ptrs = (
        Output
        + (cu_q_start + q_local_idx) * stride_o_seq
        + head_q_idx * stride_o_head
        + offs_v
    )
    tl.store(o_ptrs, output_vec.to(tl.bfloat16))


def attention_with_kvcache_blocksparse_prefill_fp8(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paged FP8 attention prefill with optional block-sparse mask.

    When block_mask is None this runs dense attention; otherwise only block
    tiles set to 1 are computed.
    """
    total_seq = q.shape[0]
    num_head_q = q.shape[1]
    dim_qk = q.shape[2]
    num_head_kv = kcache.shape[2]
    dim_v = vcache.shape[3]
    page_size = kcache.shape[1]
    device = q.device
    num_batch = cu_seqlens_q.shape[0] - 1

    if output is None:
        output = torch.zeros(total_seq, num_head_q, dim_v, dtype=torch.bfloat16, device=device)

    has_block_mask = block_mask is not None
    if has_block_mask:
        bm_stride_batch = block_mask.stride(0)
        bm_stride_head = block_mask.stride(1)
        bm_stride_qb = block_mask.stride(2)
    else:
        block_mask = torch.empty(1, dtype=torch.uint8, device=device)
        bm_stride_batch = 0
        bm_stride_head = 0
        bm_stride_qb = 0

    grid = (max_seqlens_q, num_head_q, num_batch)

    max_kv_history = int(seqlens_kvcache.max().item()) if num_batch > 0 else 0
    max_total_kv = max_kv_history + max_seqlens_q
    max_num_pages = (max_total_kv + page_size - 1) // page_size

    _blocksparse_paged_attn_kernel[grid](
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        block_mask,
        output,
        q.stride(0), q.stride(1),
        kcache.stride(0), kcache.stride(1), kcache.stride(2),
        vcache.stride(0), vcache.stride(1), vcache.stride(2),
        qscale.stride(0), qscale.stride(1), qscale.stride(2),
        block_ids.stride(0),
        bm_stride_batch, bm_stride_head, bm_stride_qb,
        output.stride(0), output.stride(1),
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        dim_qk=dim_qk,
        dim_v=dim_v,
        page_size=page_size,
        max_num_pages=max_num_pages,
        has_block_mask=has_block_mask,
        BSA_TILE_SIZE=BSA_TILE,
    )

    return output


def stem_oam_prep_paged_kv(
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute K_flat and V_bias from paged FP8 KV cache using Triton.

    Args:
        kcache: Paged K cache [num_blocks, kv_block_size, num_kv_heads, dim_qk], fp8
        vcache: Paged V cache [num_blocks, kv_block_size, num_kv_heads, dim_v], fp8
        kscale: K FP8 dequantization scale [1], fp32
        vscale: V FP8 dequantization scale [1], fp32
        kv_indices: Block table [num_batch, max_blocks_per_req], int32
        kv_seq_lens: KV seq length per request [num_batch], int32
        lambda_mag: V-bias scaling coefficient (default 0.3)
        stem_block_size: Stem block size (default 128)
        stem_stride: Downsampling stride (default 16)

    Returns:
        kflat: [num_batch, num_kv_heads, max_Kb, stem_stride * dim_qk], bf16
        vbias: [num_batch, num_kv_heads, max_Kb], fp32
    """
    num_batch = kv_seq_lens.shape[0]
    num_kv_heads = kcache.shape[2]
    dim_qk = kcache.shape[3]
    kv_block_size = kcache.shape[1]
    max_blocks_per_req = kv_indices.shape[1]

    max_kv_len = max_blocks_per_req * kv_block_size
    max_Kb = (max_kv_len + stem_block_size - 1) // stem_block_size
    flat_dim = stem_stride * dim_qk
    samples_per_group = stem_block_size // stem_stride
    dim_v = vcache.shape[3]

    device = kcache.device

    kflat = torch.zeros(
        (num_batch, num_kv_heads, max_Kb, flat_dim), dtype=torch.bfloat16, device=device
    )
    vbias = torch.zeros(
        (num_batch, num_kv_heads, max_Kb), dtype=torch.float32, device=device
    )

    # Launch K_flat kernel
    grid_kflat = (max_Kb * stem_stride, num_kv_heads, num_batch)
    _stem_prep_kflat_kernel[grid_kflat](
        kcache, kscale, kv_indices, kv_seq_lens, kflat,
        kcache.stride(0), kcache.stride(1), kcache.stride(2),
        kv_indices.stride(0),
        kflat.stride(0), kflat.stride(1), kflat.stride(2),
        kv_block_size=kv_block_size,
        dim_qk=dim_qk,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        max_Kb=max_Kb,
        samples_per_group=samples_per_group,
    )

    # Compute V_bias with the same semantics as CUDA vbias_reduce:
    # 1) build v_norm_down from per-sample-per-block max L2 norms
    # 2) log-normalize globally per (batch, head)
    # 3) ReLU + lambda scale + per-block averaging
    v_scale_val = float(vscale.flatten()[0].item())
    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded_len = ((kv_len + stem_block_size - 1) // stem_block_size) * stem_block_size
        k_down_len = k_padded_len // stem_stride
        actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size
        if actual_Kb == 0:
            continue

        for h in range(num_kv_heads):
            v_norm_down = torch.zeros((k_down_len,), dtype=torch.float32, device=device)

            for iblock in range(actual_Kb):
                for isample in range(samples_per_group):
                    max_norm = 0.0
                    for t in range(stem_stride):
                        global_token = iblock * stem_block_size + isample * stem_stride + t
                        if global_token < kv_len:
                            page_idx = global_token // kv_block_size
                            token_in_page = global_token % kv_block_size
                            phys_block = kv_indices[b, page_idx].item()
                            v_val = (
                                vcache[phys_block, token_in_page, h, :].to(torch.float32)
                                * v_scale_val
                            )
                            norm = torch.norm(v_val).item()
                            if norm > max_norm:
                                max_norm = norm

                    down_idx = iblock * samples_per_group + isample
                    if down_idx < k_down_len:
                        v_norm_down[down_idx] = max_norm

            log_vals = torch.log(v_norm_down + 1e-6)
            v_mean = log_vals.mean()
            v_std = log_vals.std(unbiased=True) if k_down_len > 1 else torch.tensor(0.0, device=device)
            inv_std = 1.0 / (v_std + 1e-6)

            for iblock in range(actual_Kb):
                block_sum = 0.0
                for isample in range(samples_per_group):
                    idx = iblock * samples_per_group + isample
                    if idx < k_down_len:
                        normalized = (log_vals[idx] - v_mean) * inv_std
                        block_sum += lambda_mag * max(0.0, float(normalized.item()))
                vbias[b, h, iblock] = block_sum / samples_per_group

    return kflat, vbias


def stem_oam_prep_varlen_q(
    q_fp8: torch.Tensor,
    qscale: torch.Tensor,
    q_seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
) -> torch.Tensor:
    """Precompute Q_flat from packed FP8 Q tensor using Triton.

    Args:
        q_fp8: Per-token query vectors [total_tokens, num_q_heads, dim_qk], fp8
        qscale: Q FP8 dequantization scale [num_batch, num_q_heads, max_seq_q_pad], fp32
        q_seq_lens: Q seq length per request [num_batch], int32
        cu_seqlens_q: Cumulative Q seq lengths [num_batch + 1], int32
        stem_block_size: Stem block size (default 128)
        stem_stride: Downsampling stride (default 16)

    Returns:
        qflat: [num_batch, num_q_heads, max_Qb, stem_stride * dim_qk], bf16
    """
    num_batch = q_seq_lens.shape[0]
    num_q_heads = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    device = q_fp8.device

    max_q_len = q_seq_lens.max().item()
    max_Qb = (max_q_len + stem_block_size - 1) // stem_block_size
    flat_dim = stem_stride * dim_qk
    samples_per_group = stem_block_size // stem_stride

    qflat = torch.zeros(
        (num_batch, num_q_heads, max_Qb, flat_dim), dtype=torch.bfloat16, device=device
    )

    grid = (max_Qb * stem_stride, num_q_heads, num_batch)
    _stem_prep_qflat_kernel[grid](
        q_fp8, qscale, q_seq_lens, cu_seqlens_q, qflat,
        q_fp8.stride(0), q_fp8.stride(1),
        qscale.stride(0), qscale.stride(1), qscale.stride(2),
        qflat.stride(0), qflat.stride(1), qflat.stride(2),
        dim_qk=dim_qk,
        stem_block_size=stem_block_size,
        stem_stride=stem_stride,
        max_Qb=max_Qb,
        samples_per_group=samples_per_group,
    )

    return qflat


def stem_oam_gemm(
    qflat: torch.Tensor,
    kflat: torch.Tensor,
    vbias: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
) -> torch.Tensor:
    """Compute block_logits via OAM GEMM with fused causal mask using Triton.

    Args:
        qflat: [num_batch, num_q_heads, max_Qb, flat_dim], bf16
        kflat: [num_batch, num_kv_heads, max_Kb, flat_dim], bf16
        vbias: [num_batch, num_kv_heads, max_Kb], fp32
        q_seq_lens: [num_batch], int32
        kv_seq_lens: [num_batch], int32
        stem_block_size: Stem block size (default 128)
        stem_stride: Downsampling stride (default 16)
        causal: Whether to apply causal masking (default True)

    Returns:
        block_logits: [num_batch, num_q_heads, max_Qb, max_Kb], bf16
    """
    num_batch = qflat.shape[0]
    num_q_heads = qflat.shape[1]
    max_Qb = qflat.shape[2]
    flat_dim = qflat.shape[3]
    num_kv_heads = kflat.shape[1]
    max_Kb = kflat.shape[2]
    device = qflat.device

    block_logits = torch.full(
        (num_batch, num_q_heads, max_Qb, max_Kb),
        float("-inf"),
        dtype=torch.bfloat16,
        device=device,
    )

    # Choose tile sizes (powers of 2, at least 16 for tl.dot)
    BLOCK_M = min(32, triton.next_power_of_2(max_Qb))
    BLOCK_N = min(32, triton.next_power_of_2(max_Kb))
    BLOCK_K = min(128, triton.next_power_of_2(flat_dim))
    # tl.dot requires at least 16
    BLOCK_M = max(16, BLOCK_M)
    BLOCK_N = max(16, BLOCK_N)
    BLOCK_K = max(16, BLOCK_K)

    grid = (
        (max_Qb + BLOCK_M - 1) // BLOCK_M,
        (max_Kb + BLOCK_N - 1) // BLOCK_N,
        num_batch * num_q_heads,
    )

    _stem_oam_gemm_kernel[grid](
        qflat, kflat, vbias, q_seq_lens, kv_seq_lens, block_logits,
        qflat.stride(0), qflat.stride(1), qflat.stride(2),
        kflat.stride(0), kflat.stride(1), kflat.stride(2),
        vbias.stride(0), vbias.stride(1),
        block_logits.stride(0), block_logits.stride(1), block_logits.stride(2),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        max_Qb=max_Qb,
        max_Kb=max_Kb,
        flat_dim=flat_dim,
        stem_stride=stem_stride,
        stem_block_size=stem_block_size,
        causal=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return block_logits


def stem_tpd(
    block_logits: torch.Tensor,
    q_seq_lens: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    block_size: int = 128,
    alpha: float = 1.0,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
) -> torch.Tensor:
    """Generate sparse block mask via top-k policy denoising.

    Args:
        block_logits: [num_batch, num_q_heads, max_Qb, max_Kb], bf16
        q_seq_lens: [num_batch], int32
        kv_seq_lens: [num_batch], int32
        num_prompt_tokens: [num_batch], int32
        block_size: Stem block size (default 128)
        alpha: Budget decay factor (default 1.0)
        initial_blocks: Leading blocks always retained (default 4)
        window_size: Recent blocks always retained (default 4)
        k_block_num_rate_medium: Rate for medium regime (default 0.2)
        k_block_num_bias_medium: Bias for medium regime (default 30)
        k_block_num_rate_large: Rate for large regime (default 0.1)
        k_block_num_bias_large: Bias for large regime (default 30)

    Returns:
        mask: [num_batch, num_q_heads, max_Qb, max_Kb], uint8
    """
    num_batch, num_q_heads, max_Qb, max_Kb = block_logits.shape
    device = block_logits.device
    mask = torch.zeros((num_batch, num_q_heads, max_Qb, max_Kb), dtype=torch.uint8, device=device)

    k_small_seq_max = 56
    k_medium_seq_max = 160

    for b in range(num_batch):
        qi_blocks = int((int(q_seq_lens[b].item()) + block_size - 1) // block_size)
        ki_blocks = int((int(kv_seq_lens[b].item()) + block_size - 1) // block_size)
        prompt_kv_blocks = int((int(num_prompt_tokens[b].item()) + block_size - 1) // block_size)

        if qi_blocks == 0 or ki_blocks == 0:
            continue

        if prompt_kv_blocks < k_small_seq_max:
            k_val = prompt_kv_blocks
        elif prompt_kv_blocks < k_medium_seq_max:
            k_val = int(prompt_kv_blocks * k_block_num_rate_medium) + k_block_num_bias_medium
        else:
            k_val = int(prompt_kv_blocks * k_block_num_rate_large) + k_block_num_bias_large
        k_val = max(1, min(k_val, prompt_kv_blocks))

        kb_offset = ki_blocks - qi_blocks
        cols = torch.arange(ki_blocks, device=device)

        for q_row in range(qi_blocks):
            q_pos = q_row + kb_offset
            decay_len = prompt_kv_blocks - k_val
            if q_pos < k_val or decay_len <= 1:
                budget = k_val
            else:
                k_end = k_val * alpha
                t = float(q_pos - k_val) / float(decay_len - 1)
                budget = int(math.floor(k_val + t * (k_end - k_val)))
                budget = max(1, min(budget, k_val))

            diag_col = q_row + kb_offset
            row = block_logits[b, :, q_row, :ki_blocks]
            finite = torch.isfinite(row)
            selected = torch.zeros_like(row, dtype=torch.bool)

            for h in range(num_q_heads):
                vals = row[h]
                finite_h = finite[h]
                num_finite = int(finite_h.sum().item())

                if num_finite > 0:
                    if budget >= num_finite:
                        topk_sel = finite_h
                    else:
                        finite_vals = vals[finite_h]
                        kth = torch.topk(finite_vals, k=budget, largest=True, sorted=True).values[-1]
                        topk_sel = finite_h & (vals >= kth)
                else:
                    topk_sel = torch.zeros_like(finite_h)

                forced = cols < initial_blocks
                forced |= (cols <= diag_col) & (cols > diag_col - window_size)
                forced |= cols == diag_col

                selected[h] = topk_sel | forced

            mask[b, :, q_row, :ki_blocks] = selected.to(torch.uint8)

    return mask


def stem_paged_kv(
    q_fp8: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    kv_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    num_prompt_tokens: torch.Tensor,
    lambda_mag: float = 0.3,
    alpha: float = 1.0,
    stem_block_size: int = 128,
    stem_stride: int = 16,
    causal: bool = True,
    initial_blocks: int = 4,
    window_size: int = 4,
    k_block_num_rate_medium: float = 0.2,
    k_block_num_bias_medium: int = 30,
    k_block_num_rate_large: float = 0.1,
    k_block_num_bias_large: int = 30,
) -> torch.Tensor:
    """End-to-end Stem sparse mask generation for paged FP8 KV cache.

    Pipeline: prep_paged_kv -> prep_varlen_q -> oam_gemm -> tpd

    Args:
        q_fp8: [total_tokens, num_q_heads, dim_qk], fp8
        kcache: [num_blocks, kv_block_size, num_kv_heads, dim_qk], fp8
        vcache: [num_blocks, kv_block_size, num_kv_heads, dim_v], fp8
        qscale: [num_batch, num_q_heads, max_seq_q_pad], fp32
        kscale: [1], fp32
        vscale: [1], fp32
        kv_indices: [num_batch, max_blocks_per_req], int32
        cu_seqlens_q: [num_batch + 1], int32
        kv_seq_lens: [num_batch], int32
        num_prompt_tokens: [num_batch], int32

    Returns:
        mask: [num_batch, num_q_heads, max_Qb, max_Kb], uint8
    """
    q_seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)

    kflat, vbias = stem_oam_prep_paged_kv(
        kcache, vcache, kscale, vscale, kv_indices, kv_seq_lens,
        lambda_mag=lambda_mag, stem_block_size=stem_block_size, stem_stride=stem_stride,
    )

    qflat = stem_oam_prep_varlen_q(
        q_fp8, qscale, q_seq_lens, cu_seqlens_q,
        stem_block_size=stem_block_size, stem_stride=stem_stride,
    )

    block_logits = stem_oam_gemm(
        qflat, kflat, vbias, q_seq_lens, kv_seq_lens,
        stem_block_size=stem_block_size, stem_stride=stem_stride, causal=causal,
    )

    mask = stem_tpd(
        block_logits, q_seq_lens, kv_seq_lens, num_prompt_tokens,
        block_size=stem_block_size, alpha=alpha,
        initial_blocks=initial_blocks, window_size=window_size,
        k_block_num_rate_medium=k_block_num_rate_medium,
        k_block_num_bias_medium=k_block_num_bias_medium,
        k_block_num_rate_large=k_block_num_rate_large,
        k_block_num_bias_large=k_block_num_bias_large,
    )

    return mask

