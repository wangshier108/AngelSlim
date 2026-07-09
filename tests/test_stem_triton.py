"""Correctness tests for Triton STEM implementation vs CUDA kernel implementation.

Verifies all five Triton kernels against both PyTorch reference implementations
and CUDA kernel (hpc-ops) implementations:
  1. stem_oam_prep_paged_kv  — K_flat and V_bias shape/dtype/correctness
  2. stem_oam_prep_varlen_q  — Q_flat shape/dtype/correctness
  3. stem_oam_gemm           — Block logits shape/finite/correctness
  4. stem_tpd                — Mask shape/dtype/invariants
  5. stem_paged_kv           — End-to-end pipeline mask correctness
  6. attention_with_kvcache_blocksparse_prefill_fp8 — Dense/block-sparse
         paged FP8 attention correctness and Triton-vs-CUDA parity

NOTE: The CUDA kernel implementation (hpc-ops) uses Hopper (SM90) specific
features including TMA (Tensor Memory Accelerator) and WGMMA (Warp Group
Matrix-Multiply Accumulate). These tests require:
  - An NVIDIA H100/H200 GPU (compute capability >= 9.0)
  - The hpc-ops library compiled and installed
"""

import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Add the module directly to the path to avoid loading the full package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "angelslim", "compressor", "sparsity", "stem", "ops"))
from stem_triton import (
    stem_oam_prep_paged_kv,
    stem_oam_prep_varlen_q,
    stem_oam_gemm,
    stem_tpd,
    stem_paged_kv,
    attention_with_kvcache_blocksparse_prefill_fp8,
)

# Try to import CUDA kernel implementation (hpc-ops)
# NOTE: hpc-ops must be installed via: make wheel && pip install dist/*.whl
try:
    import hpc
    from hpc.stem import (
        stem_oam_prep_paged_kv as cuda_stem_oam_prep_paged_kv,
        stem_oam_prep_varlen_q as cuda_stem_oam_prep_varlen_q,
        stem_oam_gemm as cuda_stem_oam_gemm,
        stem_tpd as cuda_stem_tpd,
        stem_paged_kv as cuda_stem_paged_kv,
    )
    from hpc.attention import QuantType
    HPC_AVAILABLE = True
except (ImportError, AssertionError, OSError) as e:
    print(f"import hpc error!!!!!!!!!!! {e}")
    HPC_AVAILABLE = False
    _HPC_IMPORT_ERROR = str(e)


def _is_hopper_gpu():
    """Check if current GPU is Hopper architecture (SM90+)."""
    if not torch.cuda.is_available():
        print("hopper check error")
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 9


# Skip markers
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_hopper_gpu(),
    reason="Hopper GPU (SM90+) required for CUDA kernel tests"
)
requires_hpc = pytest.mark.skipif(
    not HPC_AVAILABLE,
    reason=f"hpc-ops library not available: {_HPC_IMPORT_ERROR if not HPC_AVAILABLE else ''}"
)


# ===========================================================================
# Test Constants
# ===========================================================================

STEM_BLOCK_SIZE = 128
STEM_STRIDE = 16
DIM_QK = 128
LAMBDA_MAG = 0.3
ALPHA = 1.0
INITIAL_BLOCKS = 4
WINDOW_SIZE = 4
K_BLOCK_NUM_RATE_MEDIUM = 0.2
K_BLOCK_NUM_BIAS_MEDIUM = 30
K_BLOCK_NUM_RATE_LARGE = 0.1
K_BLOCK_NUM_BIAS_LARGE = 30

# Tolerances for Triton vs CUDA comparison
# CUDA uses Hopper WGMMA (BF16 accumulation paths) and TMA which may produce
# slightly different rounding compared to Triton's standard FP32 accumulation.
PREP_KV_ATOL = 1e-2      # K_flat: BF16 group-sum accumulation
PREP_KV_RTOL = 5e-2
PREP_Q_ATOL = 1e-2       # Q_flat: BF16 group-sum accumulation
PREP_Q_RTOL = 5e-2
GEMM_ATOL = 5e-2         # GEMM: Hopper WGMMA vs Triton dot product
GEMM_RTOL = 1e-1
VBIAS_ATOL = 1e-2        # V_bias: FP32 norm computation
VBIAS_RTOL = 5e-2

BSA_BLOCK = 128
ATTN_ATOL = 2e-1
ATTN_RTOL = 3e-1


# ===========================================================================
# Shared helpers
# ===========================================================================


def _setup_paged_fp8_data(
    num_batch, seq_len, num_head_q, num_head_kv, dim_qk=128, dim_v=128, kv_block_size=64
):
    """Create random paged FP8 data for STEM pipeline testing."""
    T_fp8 = torch.float8_e4m3fn
    device = "cuda"
    total_tokens = num_batch * seq_len

    q_fp8 = (
        torch.randn(total_tokens, num_head_q, dim_qk, dtype=torch.bfloat16, device=device)
        / math.sqrt(dim_qk)
    ).to(T_fp8)

    num_blocks_per_req = (seq_len + kv_block_size - 1) // kv_block_size
    total_blocks = num_batch * num_blocks_per_req

    kcache_fp8 = (
        torch.randn(
            total_blocks, kv_block_size, num_head_kv, dim_qk, dtype=torch.bfloat16, device=device
        )
        / math.sqrt(dim_qk)
    ).to(T_fp8)

    vcache_fp8 = torch.randn(
        total_blocks, kv_block_size, num_head_kv, dim_v, dtype=torch.bfloat16, device=device,
    ).to(T_fp8)

    kv_indices = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(
        num_batch, num_blocks_per_req
    )

    max_seqlen_pad128 = ((seq_len + 127) // 128) * 128
    qscale = torch.ones(
        num_batch, num_head_q, max_seqlen_pad128, dtype=torch.float32, device=device
    )
    kscale = torch.ones(1, dtype=torch.float32, device=device)
    vscale = torch.ones(1, dtype=torch.float32, device=device)

    seqlens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
    cu_q_seqlens = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_q_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    kv_seqlens = seqlens.clone()

    return dict(
        q_fp8=q_fp8,
        kcache_fp8=kcache_fp8,
        vcache_fp8=vcache_fp8,
        qscale=qscale,
        kscale=kscale,
        vscale=vscale,
        kv_indices=kv_indices,
        cu_q_seqlens=cu_q_seqlens,
        kv_seqlens=kv_seqlens,
    )


# ===========================================================================
# Reference implementations (pure PyTorch for verification)
# ===========================================================================


def _ref_prep_kflat(kcache_fp8, kscale, kv_indices, kv_seq_lens, stem_block_size=128, stem_stride=16):
    """Reference: compute K_flat from paged KV cache.

    Uses interleaved grouping: num_groups = stem_stride, samples_per_group = stem_block_size // stem_stride.
    Group g sums tokens at positions {g, g+stride, g+2*stride, ...} within each stem block.
    Output stored in reversed group order for anti-diagonal scoring.
    """
    num_batch = kv_seq_lens.shape[0]
    num_kv_heads = kcache_fp8.shape[2]
    dim_qk = kcache_fp8.shape[3]
    kv_block_size = kcache_fp8.shape[1]
    max_blocks_per_req = kv_indices.shape[1]
    max_kv_len = max_blocks_per_req * kv_block_size
    max_Kb = (max_kv_len + stem_block_size - 1) // stem_block_size
    flat_dim = stem_stride * dim_qk
    num_groups = stem_stride
    samples_per_group = stem_block_size // stem_stride

    device = kcache_fp8.device
    kflat = torch.zeros((num_batch, num_kv_heads, max_Kb, flat_dim), dtype=torch.bfloat16, device=device)

    k_scale_val = kscale.item()

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size

        for h in range(num_kv_heads):
            for kb in range(actual_Kb):
                for g in range(num_groups):
                    reversed_g = num_groups - 1 - g
                    acc = torch.zeros(dim_qk, dtype=torch.float32, device=device)

                    for t in range(samples_per_group):
                        global_token = kb * stem_block_size + g + t * stem_stride
                        if global_token < kv_len:
                            page_idx = global_token // kv_block_size
                            token_in_page = global_token % kv_block_size
                            phys_block = kv_indices[b, page_idx].item()
                            k_val = kcache_fp8[phys_block, token_in_page, h, :].to(torch.float32)
                            acc += k_val * k_scale_val

                    kflat[b, h, kb, reversed_g * dim_qk:(reversed_g + 1) * dim_qk] = acc.to(torch.bfloat16)

    return kflat


def _ref_prep_vbias(vcache_fp8, vscale, kv_indices, kv_seq_lens, lambda_mag=0.3,
                    stem_block_size=128, stem_stride=16):
    """Reference: compute V_bias from paged V cache.

    Two-stage computation matching the CUDA kernel:
    1. Per-group (stem_stride consecutive tokens) max L2 norm → v_norm_down
    2. Global log-normalize + per-block average → vbias
    """
    num_batch = kv_seq_lens.shape[0]
    num_kv_heads = vcache_fp8.shape[2]
    kv_block_size = vcache_fp8.shape[1]
    max_blocks_per_req = kv_indices.shape[1]
    max_kv_len = max_blocks_per_req * kv_block_size
    max_Kb = (max_kv_len + stem_block_size - 1) // stem_block_size
    samples_per_group = stem_block_size // stem_stride

    device = vcache_fp8.device
    v_scale_val = vscale.item()

    vbias_raw = torch.zeros((num_batch, num_kv_heads, max_Kb), dtype=torch.float32, device=device)

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded_len = ((kv_len + stem_block_size - 1) // stem_block_size) * stem_block_size
        k_down_len = k_padded_len // stem_stride
        num_stem_blocks = k_padded_len // stem_block_size
        if k_down_len == 0:
            continue

        for h in range(num_kv_heads):
            # Stage 1: compute per-group max norms (v_norm_down)
            v_norm_down = []
            for iblock in range(num_stem_blocks):
                for iwarp in range(samples_per_group):
                    max_norm = 0.0
                    for t in range(stem_stride):
                        global_token = iblock * stem_block_size + iwarp * stem_stride + t
                        if global_token < kv_len:
                            page_idx = global_token // kv_block_size
                            token_in_page = global_token % kv_block_size
                            phys_block = kv_indices[b, page_idx].item()
                            v_val = vcache_fp8[phys_block, token_in_page, h, :].to(torch.float32) * v_scale_val
                            norm = torch.norm(v_val).item()
                            max_norm = max(max_norm, norm)
                    v_norm_down.append(max_norm)

            # Stage 2: global log-normalize + per-block average
            log_vals = [math.log(v + 1e-6) for v in v_norm_down]
            v_mean = sum(log_vals) / k_down_len
            if k_down_len > 1:
                v_var = sum((lv - v_mean) ** 2 for lv in log_vals) / (k_down_len - 1)
                v_std = math.sqrt(v_var)
            else:
                v_std = 0.0
            inv_std = 1.0 / (v_std + 1e-6)

            for iblock in range(num_stem_blocks):
                block_sum = 0.0
                for isample in range(samples_per_group):
                    idx = iblock * samples_per_group + isample
                    if idx < k_down_len:
                        normalized = (log_vals[idx] - v_mean) * inv_std
                        block_sum += lambda_mag * max(0.0, normalized)
                vbias_raw[b, h, iblock] = block_sum / samples_per_group

    return vbias_raw


def _ref_prep_qflat(q_fp8, qscale, q_seq_lens, cu_seqlens_q, stem_block_size=128, stem_stride=16):
    """Reference: compute Q_flat from packed FP8 Q.

    Uses interleaved grouping: num_groups = stem_stride, samples_per_group = stem_block_size // stem_stride.
    Group g sums tokens at positions {g, g+stride, g+2*stride, ...} within each stem block.
    Output stored in natural group order.
    """
    num_batch = q_seq_lens.shape[0]
    num_q_heads = q_fp8.shape[1]
    dim_qk = q_fp8.shape[2]
    max_q_len = q_seq_lens.max().item()
    max_Qb = (max_q_len + stem_block_size - 1) // stem_block_size
    flat_dim = stem_stride * dim_qk
    num_groups = stem_stride
    samples_per_group = stem_block_size // stem_stride

    device = q_fp8.device
    qflat = torch.zeros((num_batch, num_q_heads, max_Qb, flat_dim), dtype=torch.bfloat16, device=device)

    for b in range(num_batch):
        q_len = q_seq_lens[b].item()
        actual_Qb = (q_len + stem_block_size - 1) // stem_block_size
        cu_offset = cu_seqlens_q[b].item()

        for h in range(num_q_heads):
            for qb in range(actual_Qb):
                for g in range(num_groups):
                    acc = torch.zeros(dim_qk, dtype=torch.float32, device=device)
                    for t in range(samples_per_group):
                        global_token = qb * stem_block_size + g + t * stem_stride
                        if global_token < q_len:
                            packed_idx = cu_offset + global_token
                            q_val = q_fp8[packed_idx, h, :].to(torch.float32)
                            q_s = qscale[b, h, global_token].item()
                            acc += q_val * q_s

                    qflat[b, h, qb, g * dim_qk:(g + 1) * dim_qk] = acc.to(torch.bfloat16)

    return qflat


def _ref_oam_gemm(qflat, kflat, vbias, q_seq_lens, kv_seq_lens, stem_block_size=128, stem_stride=16, causal=True):
    """Reference: compute block_logits = FrobScale * (Qflat @ Kflat^T) + Vbias."""
    num_batch = qflat.shape[0]
    num_q_heads = qflat.shape[1]
    max_Qb = qflat.shape[2]
    num_kv_heads = kflat.shape[1]
    max_Kb = kflat.shape[2]
    device = qflat.device

    samples_per_group = stem_block_size // stem_stride
    frob_scale = 1.0 / (samples_per_group * samples_per_group)
    block_logits = torch.full((num_batch, num_q_heads, max_Qb, max_Kb), float("-inf"),
                              dtype=torch.bfloat16, device=device)

    for b in range(num_batch):
        q_len = q_seq_lens[b].item()
        kv_len = kv_seq_lens[b].item()
        actual_Qb = (q_len + stem_block_size - 1) // stem_block_size
        actual_Kb = (kv_len + stem_block_size - 1) // stem_block_size

        for h_q in range(num_q_heads):
            h_kv = h_q * num_kv_heads // num_q_heads
            for qb in range(actual_Qb):
                for kb in range(actual_Kb):
                    if causal and kb > qb:
                        continue
                    # Dot product
                    q_vec = qflat[b, h_q, qb, :].to(torch.float32)
                    k_vec = kflat[b, h_kv, kb, :].to(torch.float32)
                    dot = (q_vec * k_vec).sum().item()
                    logit = dot * frob_scale + vbias[b, h_kv, kb].item()
                    block_logits[b, h_q, qb, kb] = logit

    return block_logits


# ===========================================================================
# Tests: Triton vs PyTorch Reference (original tests)
# ===========================================================================


@requires_cuda
class TestStemOamPrepPagedKv:
    """Test stem_oam_prep_paged_kv."""

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [256, 512])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_shape_dtype(self, num_batch, seq_len, num_head_q, num_head_kv):
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        kflat, vbias = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        max_kv_padded = ((seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE) * STEM_BLOCK_SIZE
        max_kb = max_kv_padded // STEM_BLOCK_SIZE

        assert kflat.dim() == 4
        assert kflat.shape[0] == num_batch
        assert kflat.shape[1] == num_head_kv
        assert kflat.shape[2] >= max_kb
        assert kflat.shape[3] == STEM_STRIDE * DIM_QK
        assert kflat.dtype == torch.bfloat16

        assert vbias.dim() == 3
        assert vbias.shape[0] == num_batch
        assert vbias.shape[1] == num_head_kv
        assert vbias.shape[2] >= max_kb
        assert vbias.dtype == torch.float32

    def test_correctness(self):
        """Verify K_flat matches reference implementation."""
        torch.manual_seed(42)
        num_batch, seq_len, num_head_q, num_head_kv = 1, 256, 2, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        kflat, vbias = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        kflat_ref = _ref_prep_kflat(
            d["kcache_fp8"], d["kscale"], d["kv_indices"], d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE, stem_stride=STEM_STRIDE,
        )

        # Compare K_flat (exact match expected since both use same fp8 data)
        actual_Kb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        kflat_triton = kflat[:, :, :actual_Kb, :].float()
        kflat_ref_trimmed = kflat_ref[:, :, :actual_Kb, :].float()

        max_diff = (kflat_triton - kflat_ref_trimmed).abs().max().item()
        assert max_diff < 1e-3, f"K_flat max diff too large: {max_diff:.6f}"


@requires_cuda
class TestStemOamPrepVarlenQ:
    """Test stem_oam_prep_varlen_q."""

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [256, 512])
    @pytest.mark.parametrize("num_head_q", [2, 4])
    def test_shape_dtype(self, num_batch, seq_len, num_head_q):
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, 1)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        qflat = stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        max_q_padded = ((seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE) * STEM_BLOCK_SIZE
        max_qb = max_q_padded // STEM_BLOCK_SIZE

        assert qflat.dim() == 4
        assert qflat.shape[0] == num_batch
        assert qflat.shape[1] == num_head_q
        assert qflat.shape[2] == max_qb
        assert qflat.shape[3] == STEM_STRIDE * DIM_QK
        assert qflat.dtype == torch.bfloat16

    def test_correctness(self):
        """Verify Q_flat matches reference implementation."""
        torch.manual_seed(42)
        num_batch, seq_len, num_head_q = 1, 256, 2
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, 1)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        qflat = stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        qflat_ref = _ref_prep_qflat(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE, stem_stride=STEM_STRIDE,
        )

        actual_Qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        qflat_triton = qflat[:, :, :actual_Qb, :].float()
        qflat_ref_trimmed = qflat_ref[:, :, :actual_Qb, :].float()

        max_diff = (qflat_triton - qflat_ref_trimmed).abs().max().item()
        assert max_diff < 1e-3, f"Q_flat max diff too large: {max_diff:.6f}"


@requires_cuda
class TestStemOamGemm:
    """Test stem_oam_gemm."""

    @pytest.mark.parametrize("num_batch", [1])
    @pytest.mark.parametrize("seq_len", [256, 512])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_shape_finite(self, num_batch, seq_len, num_head_q, num_head_kv):
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        kflat, vbias = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )
        qflat = stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        block_logits = stem_oam_gemm(
            qflat, kflat, vbias, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        assert block_logits.dim() == 4
        assert block_logits.shape[0] == num_batch
        assert block_logits.shape[1] == num_head_q
        assert block_logits.dtype == torch.bfloat16

        # Should have some finite values (lower triangle for causal)
        finite_ratio = torch.isfinite(block_logits).float().mean().item()
        assert finite_ratio > 0.1, f"Too few finite logits: {finite_ratio:.2%}"

    def test_causal_mask(self):
        """Verify causal masking: block_logits[qb, kb] = -inf when kb > qb."""
        torch.manual_seed(42)
        num_batch, seq_len, num_head_q, num_head_kv = 1, 512, 2, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        kflat, vbias = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )
        qflat = stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        block_logits = stem_oam_gemm(
            qflat, kflat, vbias, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        max_Qb = block_logits.shape[2]
        max_Kb = block_logits.shape[3]
        actual_Kb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE

        # Check that above-diagonal positions within valid range are -inf
        for qb in range(min(max_Qb, actual_Kb)):
            for kb in range(qb + 1, actual_Kb):
                val = block_logits[0, 0, qb, kb].item()
                assert val == float("-inf") or math.isinf(val) and val < 0, \
                    f"Expected -inf at [{qb}, {kb}], got {val}"


@requires_cuda
class TestStemTpd:
    """Test stem_tpd."""

    @pytest.mark.parametrize("num_batch", [1])
    @pytest.mark.parametrize("seq_len", [512, 1024])
    @pytest.mark.parametrize("num_head_q", [2, 4])
    def test_shape_dtype(self, num_batch, seq_len, num_head_q):
        torch.manual_seed(42)
        device = "cuda"
        max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        max_kb = max_qb

        block_logits = torch.randn(
            num_batch, num_head_q, max_qb, max_kb, dtype=torch.bfloat16, device=device
        )
        # Apply causal mask
        qr = torch.arange(max_qb, device=device).unsqueeze(1)
        kr = torch.arange(max_kb, device=device).unsqueeze(0)
        causal = kr > qr
        block_logits.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        q_seq_lens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
        kv_seq_lens = q_seq_lens.clone()

        mask = stem_tpd(
            block_logits, q_seq_lens, kv_seq_lens, kv_seq_lens,
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        assert mask.shape == (num_batch, num_head_q, max_qb, max_kb)
        assert mask.dtype == torch.uint8

    def test_invariants(self):
        """Verify TPD invariants: diagonal, initial_blocks, window always selected."""
        torch.manual_seed(42)
        device = "cuda"
        num_batch, seq_len, num_head_q = 1, 2048, 4
        max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        max_kb = max_qb

        block_logits = torch.randn(
            num_batch, num_head_q, max_qb, max_kb, dtype=torch.bfloat16, device=device
        )
        qr = torch.arange(max_qb, device=device).unsqueeze(1)
        kr = torch.arange(max_kb, device=device).unsqueeze(0)
        causal = kr > qr
        block_logits.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        q_seq_lens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
        kv_seq_lens = q_seq_lens.clone()

        mask = stem_tpd(
            block_logits, q_seq_lens, kv_seq_lens, kv_seq_lens,
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # Diagonal should always be selected
        diag = mask[:, :, range(min(max_qb, max_kb)), range(min(max_qb, max_kb))]
        assert (diag == 1).all(), "Diagonal blocks must be selected"

        # Initial blocks should be selected for all Q blocks (within causal range)
        if max_kb >= INITIAL_BLOCKS:
            for qb in range(INITIAL_BLOCKS, max_qb):
                init = mask[:, :, qb, :INITIAL_BLOCKS]
                assert (init == 1).all(), f"Initial blocks not selected at qb={qb}"

    def test_sparsity(self):
        """Verify mask is non-trivial (not all 1s) for long sequences."""
        torch.manual_seed(42)
        device = "cuda"
        num_batch, seq_len, num_head_q = 1, 4096, 4
        max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        max_kb = max_qb

        block_logits = torch.randn(
            num_batch, num_head_q, max_qb, max_kb, dtype=torch.bfloat16, device=device
        )
        qr = torch.arange(max_qb, device=device).unsqueeze(1)
        kr = torch.arange(max_kb, device=device).unsqueeze(0)
        causal = kr > qr
        block_logits.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        q_seq_lens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
        kv_seq_lens = q_seq_lens.clone()

        mask = stem_tpd(
            block_logits, q_seq_lens, kv_seq_lens, kv_seq_lens,
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # For 4096 tokens (32 blocks), with rate 0.1 and bias 30, budget should cap
        # density should be between 0 and 1 (actually sparse)
        density = mask.float().mean().item()
        assert 0.0 < density < 1.0, f"Mask density {density:.4f} is degenerate"


@requires_cuda
class TestStemPagedKv:
    """Test stem_paged_kv end-to-end."""

    @pytest.mark.parametrize("num_batch", [1])
    @pytest.mark.parametrize("seq_len", [512, 1024])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1)])
    def test_e2e_shape(self, num_batch, seq_len, num_head_q, num_head_kv):
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        mask = stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        assert mask.dim() == 4
        assert mask.shape[0] == num_batch
        assert mask.shape[1] == num_head_q
        assert mask.dtype == torch.uint8

        # Diagonal should be selected
        min_dim = min(mask.shape[2], mask.shape[3])
        diag = mask[:, :, range(min_dim), range(min_dim)]
        assert (diag == 1).all(), "Diagonal blocks must be selected"

    def test_e2e_long_sequence(self):
        """Test end-to-end on a longer sequence to verify sparsity behavior."""
        torch.manual_seed(42)
        num_batch, seq_len, num_head_q, num_head_kv = 1, 2048, 4, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        mask = stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # Should produce non-trivial sparse mask
        density = mask.float().mean().item()
        assert 0.0 < density < 1.0, f"Mask density {density:.4f} is degenerate"
        print(f"  [E2E] seq_len={seq_len}, mask density={density:.4f}")


# ===========================================================================
# Tests: Triton vs CUDA Kernel (hpc-ops) Precision Comparison
#
# These tests require:
#   1. Hopper GPU (SM90+) — CUDA kernels use TMA and WGMMA
#   2. hpc-ops library built and installed
#
# The CUDA implementation leverages Hopper-specific hardware:
#   - TMA (Tensor Memory Accelerator) for efficient global→shared memory loads
#   - WGMMA (Warp Group MMA) with SM90_64x128x16_F32BF16BF16_SS for GEMM
#   - FP8 E4M3 native dequantization paths
#   - 4-stage async pipeline with persistent kernel work-stealing
#
# Expected precision differences:
#   - prep_kv/prep_q: Both accumulate in FP32 then cast to BF16. Differences
#     arise from CUDA using fused multiply-add (FMA) vs Triton's sequential ops.
#   - oam_gemm: Hopper WGMMA uses BF16×BF16→FP32 structured MMA which may
#     differ in intermediate rounding from Triton's tl.dot accumulation.
#   - tpd: Integer/comparison logic — should produce identical masks given
#     identical block_logits input.
# ===========================================================================


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaPrepPagedKv:
    """Compare stem_oam_prep_paged_kv: Triton vs CUDA kernel."""

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2), (8, 2)])
    def test_kflat_precision(self, num_batch, seq_len, num_head_q, num_head_kv):
        """Verify K_flat outputs match between Triton and CUDA implementations."""
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        # Triton implementation
        kflat_triton, vbias_triton = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        # CUDA kernel implementation
        kflat_cuda, vbias_cuda = cuda_stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

        # Compare shapes
        assert kflat_triton.shape == kflat_cuda.shape, \
            f"K_flat shape mismatch: Triton {kflat_triton.shape} vs CUDA {kflat_cuda.shape}"
        assert kflat_triton.dtype == kflat_cuda.dtype == torch.bfloat16

        # Compare values (within valid range)
        actual_Kb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        kt = kflat_triton[:, :, :actual_Kb, :].float()
        kc = kflat_cuda[:, :, :actual_Kb, :].float()

        max_diff = (kt - kc).abs().max().item()
        mean_diff = (kt - kc).abs().mean().item()

        # Use relative tolerance for non-zero elements
        nonzero_mask = kc.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = ((kt - kc).abs() / (kc.abs() + 1e-8))[nonzero_mask].max().item()
        else:
            rel_diff = 0.0

        assert max_diff < PREP_KV_ATOL or rel_diff < PREP_KV_RTOL, \
            f"K_flat Triton vs CUDA: max_abs_diff={max_diff:.6f}, " \
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"

        print(f"  [K_flat] batch={num_batch}, seq={seq_len}, heads=({num_head_q},{num_head_kv}): "
              f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f}")

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_vbias_precision(self, num_batch, seq_len, num_head_q, num_head_kv):
        """Verify V_bias outputs match between Triton and CUDA implementations."""
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        # Triton implementation
        _, vbias_triton = stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        # CUDA kernel implementation
        _, vbias_cuda = cuda_stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

        # Compare shapes and dtype
        assert vbias_triton.shape == vbias_cuda.shape, \
            f"V_bias shape mismatch: Triton {vbias_triton.shape} vs CUDA {vbias_cuda.shape}"
        assert vbias_triton.dtype == vbias_cuda.dtype == torch.float32

        # Compare values
        actual_Kb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        vt = vbias_triton[:, :, :actual_Kb]
        vc = vbias_cuda[:, :, :actual_Kb]

        max_diff = (vt - vc).abs().max().item()
        mean_diff = (vt - vc).abs().mean().item()

        nonzero_mask = vc.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = ((vt - vc).abs() / (vc.abs() + 1e-8))[nonzero_mask].max().item()
        else:
            rel_diff = 0.0

        assert max_diff < VBIAS_ATOL or rel_diff < VBIAS_RTOL, \
            f"V_bias Triton vs CUDA: max_abs_diff={max_diff:.6f}, " \
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"

        print(f"  [V_bias] batch={num_batch}, seq={seq_len}: "
              f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f}")


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaPrepVarlenQ:
    """Compare stem_oam_prep_varlen_q: Triton vs CUDA kernel."""

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    @pytest.mark.parametrize("num_head_q", [2, 4, 8])
    def test_qflat_precision(self, num_batch, seq_len, num_head_q):
        """Verify Q_flat outputs match between Triton and CUDA implementations."""
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, 1)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        # Triton implementation
        qflat_triton = stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        # CUDA kernel implementation
        qflat_cuda = cuda_stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        # Compare shapes
        assert qflat_triton.shape == qflat_cuda.shape, \
            f"Q_flat shape mismatch: Triton {qflat_triton.shape} vs CUDA {qflat_cuda.shape}"
        assert qflat_triton.dtype == qflat_cuda.dtype == torch.bfloat16

        # Compare values
        actual_Qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        qt = qflat_triton[:, :, :actual_Qb, :].float()
        qc = qflat_cuda[:, :, :actual_Qb, :].float()

        max_diff = (qt - qc).abs().max().item()
        mean_diff = (qt - qc).abs().mean().item()

        nonzero_mask = qc.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = ((qt - qc).abs() / (qc.abs() + 1e-8))[nonzero_mask].max().item()
        else:
            rel_diff = 0.0

        assert max_diff < PREP_Q_ATOL or rel_diff < PREP_Q_RTOL, \
            f"Q_flat Triton vs CUDA: max_abs_diff={max_diff:.6f}, " \
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"

        print(f"  [Q_flat] batch={num_batch}, seq={seq_len}, heads={num_head_q}: "
              f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f}")


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaOamGemm:
    """Compare stem_oam_gemm: Triton vs CUDA kernel (Hopper WGMMA).

    The CUDA GEMM uses SM90_64x128x16_F32BF16BF16_SS WGMMA instruction
    which performs BF16×BF16→FP32 matrix multiply-accumulate with structured
    sparsity-ready tiling. The intermediate FP32 accumulation then adds V_bias
    and applies causal masking in the epilogue before casting to BF16 output.

    Differences vs Triton stem from:
      - WGMMA's specific rounding behavior for BF16 inputs
      - TMA copy-box alignment padding (zero-filled OOB)
      - Fused epilogue (bias + mask) vs sequential operations
    """

    @pytest.mark.parametrize("num_batch", [1])
    @pytest.mark.parametrize("seq_len", [256, 512, 1024])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_block_logits_precision(self, num_batch, seq_len, num_head_q, num_head_kv):
        """Verify block_logits match between Triton and CUDA WGMMA implementations."""
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        # Use CUDA prep outputs as shared input to isolate GEMM differences
        kflat_cuda, vbias_cuda = cuda_stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )
        qflat_cuda = cuda_stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        # Triton GEMM with CUDA-prepared inputs (isolate GEMM logic)
        logits_triton = stem_oam_gemm(
            qflat_cuda, kflat_cuda, vbias_cuda, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        # CUDA GEMM with same inputs
        logits_cuda = cuda_stem_oam_gemm(
            qflat_cuda, kflat_cuda, vbias_cuda, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        # Compare only finite (valid) positions
        assert logits_triton.shape == logits_cuda.shape
        finite_mask = torch.isfinite(logits_cuda) & torch.isfinite(logits_triton)
        assert finite_mask.any(), "No finite logits to compare"

        lt = logits_triton[finite_mask].float()
        lc = logits_cuda[finite_mask].float()

        max_diff = (lt - lc).abs().max().item()
        mean_diff = (lt - lc).abs().mean().item()

        nonzero_mask = lc.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = ((lt - lc).abs() / (lc.abs() + 1e-8))[nonzero_mask].max().item()
        else:
            rel_diff = 0.0

        assert max_diff < GEMM_ATOL or rel_diff < GEMM_RTOL, \
            f"GEMM Triton vs CUDA: max_abs_diff={max_diff:.6f}, " \
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"

        print(f"  [GEMM] batch={num_batch}, seq={seq_len}, heads=({num_head_q},{num_head_kv}): "
              f"max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f}")

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    def test_causal_mask_agreement(self, seq_len):
        """Verify both implementations agree on causal masking positions."""
        torch.manual_seed(42)
        num_batch, num_head_q, num_head_kv = 1, 4, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        kflat_cuda, vbias_cuda = cuda_stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )
        qflat_cuda = cuda_stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )

        logits_triton = stem_oam_gemm(
            qflat_cuda, kflat_cuda, vbias_cuda, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )
        logits_cuda = cuda_stem_oam_gemm(
            qflat_cuda, kflat_cuda, vbias_cuda, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        # Causal mask positions (where output is -inf) must match exactly
        triton_masked = torch.isinf(logits_triton) & (logits_triton < 0)
        cuda_masked = torch.isinf(logits_cuda) & (logits_cuda < 0)

        mask_agreement = (triton_masked == cuda_masked).all().item()
        assert mask_agreement, \
            f"Causal mask disagreement: Triton has {triton_masked.sum().item()} masked, " \
            f"CUDA has {cuda_masked.sum().item()} masked"


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaTpd:
    """Compare stem_tpd: Triton vs CUDA kernel.

    TPD (Top-k Policy Denoising) is primarily integer/comparison logic.
    Given identical block_logits input, both implementations should produce
    bit-identical masks. Any difference indicates a logic bug.
    """

    @pytest.mark.parametrize("seq_len", [512, 1024, 2048, 4096])
    @pytest.mark.parametrize("num_head_q", [2, 4])
    def test_mask_exact_match(self, seq_len, num_head_q):
        """Verify TPD masks are bit-identical between Triton and CUDA."""
        torch.manual_seed(42)
        device = "cuda"
        num_batch = 1
        max_qb = (seq_len + STEM_BLOCK_SIZE - 1) // STEM_BLOCK_SIZE
        max_kb = max_qb

        # Create shared block_logits input
        block_logits = torch.randn(
            num_batch, num_head_q, max_qb, max_kb, dtype=torch.bfloat16, device=device
        )
        qr = torch.arange(max_qb, device=device).unsqueeze(1)
        kr = torch.arange(max_kb, device=device).unsqueeze(0)
        causal = kr > qr
        block_logits.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        q_seq_lens = torch.full((num_batch,), seq_len, dtype=torch.int32, device=device)
        kv_seq_lens = q_seq_lens.clone()

        # Triton TPD
        mask_triton = stem_tpd(
            block_logits, q_seq_lens, kv_seq_lens, kv_seq_lens,
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # CUDA TPD
        mask_cuda = cuda_stem_tpd(
            block_logits, q_seq_lens, kv_seq_lens, kv_seq_lens,
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # Shapes must match
        assert mask_triton.shape == mask_cuda.shape, \
            f"TPD mask shape mismatch: Triton {mask_triton.shape} vs CUDA {mask_cuda.shape}"
        assert mask_triton.dtype == mask_cuda.dtype == torch.uint8

        # Masks should be bit-identical (both are integer logic)
        mismatches = (mask_triton != mask_cuda).sum().item()
        total = mask_triton.numel()

        assert mismatches == 0, \
            f"TPD mask mismatch: {mismatches}/{total} positions differ " \
            f"({mismatches/total*100:.4f}%)"

        print(f"  [TPD] seq={seq_len}, heads={num_head_q}: EXACT MATCH "
              f"(density={mask_triton.float().mean().item():.4f})")

    @pytest.mark.parametrize("seq_len", [1024, 2048])
    def test_tpd_with_real_logits(self, seq_len):
        """Test TPD agreement using real GEMM-produced block_logits."""
        torch.manual_seed(42)
        num_batch, num_head_q, num_head_kv = 1, 4, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)
        q_seqlens = (d["cu_q_seqlens"][1:] - d["cu_q_seqlens"][:-1]).to(torch.int32)

        # Use CUDA pipeline to generate block_logits
        kflat, vbias = cuda_stem_oam_prep_paged_kv(
            d["kcache_fp8"], d["vcache_fp8"], d["kscale"], d["vscale"],
            d["kv_indices"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )
        qflat = cuda_stem_oam_prep_varlen_q(
            d["q_fp8"], d["qscale"], q_seqlens, d["cu_q_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
        )
        block_logits = cuda_stem_oam_gemm(
            qflat, kflat, vbias, q_seqlens, d["kv_seqlens"],
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
        )

        # Both TPD implementations on same logits
        mask_triton = stem_tpd(
            block_logits, q_seqlens, d["kv_seqlens"], d["kv_seqlens"],
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        mask_cuda = cuda_stem_tpd(
            block_logits, q_seqlens, d["kv_seqlens"], d["kv_seqlens"],
            block_size=STEM_BLOCK_SIZE,
            alpha=ALPHA,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        mismatches = (mask_triton != mask_cuda).sum().item()
        assert mismatches == 0, \
            f"TPD with real logits: {mismatches} mask differences"


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaE2E:
    """End-to-end comparison: Triton vs CUDA kernel full pipeline.

    This tests the complete stem_paged_kv pipeline including cumulative
    precision differences from all stages. The final mask comparison uses
    IoU (Intersection over Union) to account for borderline blocks where
    small logit differences may flip the top-k selection.
    """

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("seq_len", [512, 1024, 2048])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (8, 2)])
    def test_e2e_mask_agreement(self, num_batch, seq_len, num_head_q, num_head_kv):
        """Verify end-to-end pipeline masks have high agreement."""
        torch.manual_seed(42)
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        # Triton end-to-end
        mask_triton = stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        # CUDA end-to-end
        mask_cuda = cuda_stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

        # Shapes must match
        assert mask_triton.shape == mask_cuda.shape, \
            f"E2E mask shape mismatch: Triton {mask_triton.shape} vs CUDA {mask_cuda.shape}"

        # Compute IoU (Intersection over Union) per head
        mt = mask_triton.float()
        mc = mask_cuda.float()

        intersection = (mt * mc).sum(dim=(-2, -1))  # [batch, heads]
        union = ((mt + mc) > 0).float().sum(dim=(-2, -1))

        iou = (intersection / (union + 1e-8)).mean().item()

        # Also compute exact match rate
        exact_match = (mask_triton == mask_cuda).float().mean().item()

        # For practical use, we expect high IoU (>0.9) since most blocks
        # are clearly above/below threshold. Borderline blocks may differ
        # due to accumulated FP precision differences.
        MIN_IOU = 0.90
        MIN_EXACT = 0.95

        assert iou >= MIN_IOU, \
            f"E2E IoU too low: {iou:.4f} < {MIN_IOU} " \
            f"(exact_match={exact_match:.4f})"

        print(f"  [E2E] batch={num_batch}, seq={seq_len}, "
              f"heads=({num_head_q},{num_head_kv}): "
              f"IoU={iou:.4f}, exact={exact_match:.4f}")

    def test_e2e_invariants_preserved(self):
        """Verify structural invariants hold for both implementations."""
        torch.manual_seed(42)
        num_batch, seq_len, num_head_q, num_head_kv = 1, 2048, 4, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        mask_triton = stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        mask_cuda = cuda_stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

        # Both must satisfy structural invariants
        for name, mask in [("Triton", mask_triton), ("CUDA", mask_cuda)]:
            min_dim = min(mask.shape[2], mask.shape[3])

            # Diagonal always selected
            diag = mask[:, :, range(min_dim), range(min_dim)]
            assert (diag == 1).all(), f"{name}: diagonal not all selected"

            # Initial blocks always selected (for rows beyond initial_blocks)
            max_qb = mask.shape[2]
            if mask.shape[3] >= INITIAL_BLOCKS:
                for qb in range(INITIAL_BLOCKS, max_qb):
                    init = mask[:, :, qb, :INITIAL_BLOCKS]
                    assert (init == 1).all(), \
                        f"{name}: initial blocks not selected at qb={qb}"

    @pytest.mark.parametrize("seq_len", [4096, 8192])
    def test_e2e_long_sequence_density(self, seq_len):
        """Compare sparsity density between implementations on long sequences."""
        torch.manual_seed(42)
        num_batch, num_head_q, num_head_kv = 1, 4, 1
        d = _setup_paged_fp8_data(num_batch, seq_len, num_head_q, num_head_kv)

        mask_triton = stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
        )

        mask_cuda = cuda_stem_paged_kv(
            d["q_fp8"], d["kcache_fp8"], d["vcache_fp8"],
            d["qscale"], d["kscale"], d["vscale"],
            d["kv_indices"], d["cu_q_seqlens"], d["kv_seqlens"], d["kv_seqlens"],
            lambda_mag=LAMBDA_MAG,
            alpha=ALPHA,
            stem_block_size=STEM_BLOCK_SIZE,
            stem_stride=STEM_STRIDE,
            causal=True,
            initial_blocks=INITIAL_BLOCKS,
            window_size=WINDOW_SIZE,
            k_block_num_rate_medium=K_BLOCK_NUM_RATE_MEDIUM,
            k_block_num_bias_medium=K_BLOCK_NUM_BIAS_MEDIUM,
            k_block_num_rate_large=K_BLOCK_NUM_RATE_LARGE,
            k_block_num_bias_large=K_BLOCK_NUM_BIAS_LARGE,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        )

        density_triton = mask_triton.float().mean().item()
        density_cuda = mask_cuda.float().mean().item()

        # Densities should be very close (same budget logic)
        density_diff = abs(density_triton - density_cuda)
        assert density_diff < 0.05, \
            f"Density divergence: Triton={density_triton:.4f}, CUDA={density_cuda:.4f}, " \
            f"diff={density_diff:.4f}"

        print(f"  [E2E Long] seq={seq_len}: "
              f"density_triton={density_triton:.4f}, density_cuda={density_cuda:.4f}")


# ===========================================================================
# Block-sparse Attention: reference helpers
# ===========================================================================


def _ref_blocksparse_attn_with_kvcache(
    q,
    kcache,
    vcache,
    qscale,
    kscale,
    vscale,
    cu_seqlens_q,
    block_ids,
    seqlens_kvcache,
    block_mask=None,
    causal=True,
):
    """Naive reference: paged FP8 attention with optional block-sparse mask."""
    num_batch = cu_seqlens_q.shape[0] - 1
    num_head_q = q.shape[1]
    dim_qk = q.shape[2]
    num_head_kv = kcache.shape[2]
    dim_v = vcache.shape[3]
    page_size = kcache.shape[1]
    num_group = num_head_q // num_head_kv
    device = q.device

    outputs = []

    # CUDA indexing on FP8 tensors is not supported in some torch versions.
    kcache_f = kcache.float()
    vcache_f = vcache.float()

    for i in range(num_batch):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        q_len = q_end - q_start
        kv_history = seqlens_kvcache[i].item()
        total_kv = kv_history + q_len

        bq = q[q_start:q_end].float()

        num_pages_needed = (total_kv + page_size - 1) // page_size
        page_ids = block_ids[i, :num_pages_needed]
        k_gathered = kcache_f[page_ids].reshape(-1, num_head_kv, dim_qk)[:total_kv]
        v_gathered = vcache_f[page_ids].reshape(-1, num_head_kv, dim_v)[:total_kv]

        k_gathered = k_gathered.repeat_interleave(num_group, dim=1)
        v_gathered = v_gathered.repeat_interleave(num_group, dim=1)

        bq_t = bq.transpose(0, 1)
        bk_t = k_gathered.transpose(0, 1)
        bv_t = v_gathered.transpose(0, 1)

        scores = torch.matmul(bq_t, bk_t.transpose(-2, -1))
        qs = qscale[i, :, :q_len].unsqueeze(-1)
        scores = scores * qs * kscale[0].item() / math.sqrt(dim_qk)

        if block_mask is not None:
            bm = block_mask[i]
            elem_mask = bm.repeat_interleave(BSA_BLOCK, dim=-2)[:, :q_len, :]
            elem_mask = elem_mask.repeat_interleave(BSA_BLOCK, dim=-1)[:, :, :total_kv]
            scores = scores.masked_fill(~elem_mask.bool(), float("-inf"))

        if causal:
            q_positions = torch.arange(kv_history, kv_history + q_len, device=device).unsqueeze(1)
            kv_positions = torch.arange(total_kv, device=device).unsqueeze(0)
            causal_mask = kv_positions > q_positions
            scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        out = torch.matmul(attn_weights, bv_t)
        out = out * vscale[0].item()
        outputs.append(out.transpose(0, 1))

    return torch.cat(outputs, dim=0).to(torch.bfloat16)


def _generate_block_sparse_mask(batch, heads, nrow, ncol, skip_ratio, causal=True, device="cuda"):
    """Generate a random block-sparse mask."""
    mask = torch.rand(batch, heads, nrow, ncol, device=device) >= skip_ratio

    row_idx = torch.arange(nrow, device=device).view(nrow, 1)
    col_idx = torch.arange(ncol, device=device).view(1, ncol)

    if causal:
        causal_boundary = row_idx + (ncol - nrow)
        valid_causal = col_idx <= causal_boundary
        mask = mask & valid_causal
        diag_col = torch.clamp(causal_boundary, max=ncol - 1)
        mask = mask | (col_idx == diag_col)

    return mask


def _setup_blocksparse_fp8_data(
    num_batch,
    num_seq,
    num_head_q,
    num_head_kv,
    head_dim=128,
    page_size=64,
):
    """Setup paged FP8 attention data for blocksparse tests."""
    t_fp8 = torch.float8_e4m3fn
    t_bf16 = torch.bfloat16
    device = "cuda"

    total_seq = num_batch * num_seq
    num_seq_q_pad = ((num_seq + 127) // 128) * 128

    q = (
        torch.randn(total_seq, num_head_q, head_dim, dtype=t_bf16, device=device)
        / math.sqrt(head_dim)
    ).to(t_fp8)

    qscale = (
        torch.abs(torch.randn(num_batch, num_head_q, num_seq_q_pad, dtype=torch.float32, device=device)) / 10
        + 0.01
    )
    kscale = torch.rand(1, dtype=torch.float32, device=device) + 0.5
    vscale = torch.rand(1, dtype=torch.float32, device=device) + 0.5

    kvcache_blocks = (num_seq + page_size - 1) // page_size
    total_pages = num_batch * kvcache_blocks

    kcache = (
        torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device)
        / math.sqrt(head_dim)
    ).to(t_fp8)
    vcache = torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device).to(
        t_fp8
    )

    block_ids = torch.arange(total_pages, dtype=torch.int32, device=device).reshape(num_batch, kvcache_blocks)

    seqlens_q = torch.full((num_batch,), num_seq, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.zeros(num_batch + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    seqlens_kvcache = torch.zeros(num_batch, dtype=torch.int32, device=device)

    return {
        "q": q,
        "kcache": kcache,
        "vcache": vcache,
        "qscale": qscale,
        "kscale": kscale,
        "vscale": vscale,
        "cu_seqlens_q": cu_seqlens_q,
        "block_ids": block_ids,
        "seqlens_kvcache": seqlens_kvcache,
    }


# ===========================================================================
# Block-sparse Attention: Triton vs Reference
# ===========================================================================


@requires_cuda
class TestBlocksparsePagedAttnFP8:
    """Test attention_with_kvcache_blocksparse_prefill_fp8."""

    @pytest.mark.parametrize("num_batch", [1, 2])
    @pytest.mark.parametrize("num_seq", [128, 256])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_dense_shape_dtype(self, num_batch, num_seq, num_head_q, num_head_kv):
        torch.manual_seed(42)
        d = _setup_blocksparse_fp8_data(num_batch, num_seq, num_head_q, num_head_kv)

        out = attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            max_seqlens_q=num_seq,
            block_mask=None,
        )

        assert out.shape == (num_batch * num_seq, num_head_q, 128)
        assert out.dtype == torch.bfloat16
        assert out.abs().sum() > 0

    @pytest.mark.parametrize("num_seq", [128, 256])
    def test_dense_correctness(self, num_seq):
        torch.manual_seed(42)
        num_batch, num_head_q, num_head_kv = 1, 2, 1
        d = _setup_blocksparse_fp8_data(num_batch, num_seq, num_head_q, num_head_kv)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            max_seqlens_q=num_seq,
            block_mask=None,
        )

        out_ref = _ref_blocksparse_attn_with_kvcache(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            block_mask=None,
            causal=True,
        )

        diff = (out_triton.float() - out_ref.float()).abs()
        ref_norm = out_ref.float().abs().clamp(min=1e-6)
        rel_err = (diff / ref_norm).mean().item()
        assert rel_err < 0.1, f"Dense attention relative error too high: {rel_err:.4f}"

    @pytest.mark.parametrize("skip_ratio", [0.3, 0.5])
    def test_sparse_correctness(self, skip_ratio):
        torch.manual_seed(42)
        num_batch, num_seq, num_head_q, num_head_kv = 1, 256, 2, 1
        d = _setup_blocksparse_fp8_data(num_batch, num_seq, num_head_q, num_head_kv)

        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = _generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device="cuda"
        ).to(torch.uint8)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            max_seqlens_q=num_seq,
            block_mask=block_mask,
        )

        out_ref = _ref_blocksparse_attn_with_kvcache(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            block_mask=block_mask,
            causal=True,
        )

        diff = (out_triton.float() - out_ref.float()).abs()
        ref_norm = out_ref.float().abs().clamp(min=1e-6)
        rel_err = (diff / ref_norm).mean().item()
        assert rel_err < 0.1, f"Sparse attention relative error too high: {rel_err:.4f}"

    def test_with_history(self):
        torch.manual_seed(42)
        device = "cuda"
        t_fp8 = torch.float8_e4m3fn
        t_bf16 = torch.bfloat16

        num_batch = 1
        q_len = 128
        kv_history = 128
        total_kv = kv_history + q_len
        num_head_q, num_head_kv = 2, 1
        head_dim = 128
        page_size = 64

        q = (
            torch.randn(num_batch * q_len, num_head_q, head_dim, dtype=t_bf16, device=device)
            / math.sqrt(head_dim)
        ).to(t_fp8)
        qscale = (
            torch.abs(
                torch.randn(
                    num_batch,
                    num_head_q,
                    ((q_len + 127) // 128) * 128,
                    dtype=torch.float32,
                    device=device,
                )
            )
            / 10
            + 0.01
        )
        kscale = torch.ones(1, dtype=torch.float32, device=device)
        vscale = torch.ones(1, dtype=torch.float32, device=device)

        total_pages = (total_kv + page_size - 1) // page_size
        kcache = (
            torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device)
            / math.sqrt(head_dim)
        ).to(t_fp8)
        vcache = torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device).to(
            t_fp8
        )

        block_ids = torch.arange(total_pages, dtype=torch.int32, device=device).unsqueeze(0)
        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        seqlens_kvcache = torch.tensor([kv_history], dtype=torch.int32, device=device)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q=q_len,
            block_mask=None,
        )

        out_ref = _ref_blocksparse_attn_with_kvcache(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            block_mask=None,
            causal=True,
        )

        diff = (out_triton.float() - out_ref.float()).abs()
        ref_norm = out_ref.float().abs().clamp(min=1e-6)
        rel_err = (diff / ref_norm).mean().item()
        assert rel_err < 0.1, f"History attention relative error too high: {rel_err:.4f}"


# ===========================================================================
# Block-sparse Attention: Triton vs CUDA
# ===========================================================================


@requires_cuda
@requires_hopper
@requires_hpc
class TestTritonVsCudaBlocksparsePrefillFP8:
    """Compare Triton block-sparse paged FP8 attention vs CUDA kernel."""

    @pytest.mark.parametrize("num_batch", [1])
    @pytest.mark.parametrize("num_seq", [128, 256])
    @pytest.mark.parametrize("num_head_q,num_head_kv", [(4, 1), (2, 2)])
    def test_dense_precision(self, num_batch, num_seq, num_head_q, num_head_kv):
        torch.manual_seed(42)
        d = _setup_blocksparse_fp8_data(num_batch, num_seq, num_head_q, num_head_kv)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            max_seqlens_q=num_seq,
            block_mask=None,
        )

        out_cuda = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            num_seq,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            block_mask=None,
        )

        ot = out_triton.float()
        oc = out_cuda.float()
        diff = (ot - oc).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        nonzero = oc.abs() > 1e-6
        rel_diff = ((diff / (oc.abs() + 1e-8))[nonzero].max().item()) if nonzero.any() else 0.0

        assert max_diff < ATTN_ATOL or rel_diff < ATTN_RTOL, (
            f"Dense Triton vs CUDA mismatch: max_abs_diff={max_diff:.6f}, "
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    @pytest.mark.parametrize("skip_ratio", [0.3, 0.5])
    def test_sparse_precision(self, skip_ratio):
        torch.manual_seed(42)
        num_batch, num_seq, num_head_q, num_head_kv = 1, 256, 2, 1
        d = _setup_blocksparse_fp8_data(num_batch, num_seq, num_head_q, num_head_kv)

        ntiles = (num_seq + BSA_BLOCK - 1) // BSA_BLOCK
        block_mask = _generate_block_sparse_mask(
            num_batch, num_head_q, ntiles, ntiles, skip_ratio, causal=True, device="cuda"
        ).to(torch.uint8)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            max_seqlens_q=num_seq,
            block_mask=block_mask,
        )

        out_cuda = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
            d["q"],
            d["kcache"],
            d["vcache"],
            d["qscale"],
            d["kscale"],
            d["vscale"],
            d["cu_seqlens_q"],
            d["block_ids"],
            d["seqlens_kvcache"],
            num_seq,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            block_mask=block_mask,
        )

        ot = out_triton.float()
        oc = out_cuda.float()
        diff = (ot - oc).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        nonzero = oc.abs() > 1e-6
        rel_diff = ((diff / (oc.abs() + 1e-8))[nonzero].max().item()) if nonzero.any() else 0.0

        assert max_diff < ATTN_ATOL or rel_diff < ATTN_RTOL, (
            f"Sparse Triton vs CUDA mismatch: max_abs_diff={max_diff:.6f}, "
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"
        )

    def test_with_history_precision(self):
        torch.manual_seed(42)
        device = "cuda"
        t_fp8 = torch.float8_e4m3fn
        t_bf16 = torch.bfloat16

        num_batch = 1
        q_len = 128
        kv_history = 128
        total_kv = kv_history + q_len
        num_head_q, num_head_kv = 2, 1
        head_dim = 128
        page_size = 64

        q = (
            torch.randn(num_batch * q_len, num_head_q, head_dim, dtype=t_bf16, device=device)
            / math.sqrt(head_dim)
        ).to(t_fp8)
        qscale = (
            torch.abs(
                torch.randn(
                    num_batch,
                    num_head_q,
                    ((q_len + 127) // 128) * 128,
                    dtype=torch.float32,
                    device=device,
                )
            )
            / 10
            + 0.01
        )
        kscale = torch.ones(1, dtype=torch.float32, device=device)
        vscale = torch.ones(1, dtype=torch.float32, device=device)

        total_pages = (total_kv + page_size - 1) // page_size
        kcache = (
            torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device)
            / math.sqrt(head_dim)
        ).to(t_fp8)
        vcache = torch.randn(total_pages, page_size, num_head_kv, head_dim, dtype=t_bf16, device=device).to(
            t_fp8
        )

        block_ids = torch.arange(total_pages, dtype=torch.int32, device=device).unsqueeze(0)
        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        seqlens_kvcache = torch.tensor([kv_history], dtype=torch.int32, device=device)

        out_triton = attention_with_kvcache_blocksparse_prefill_fp8(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            max_seqlens_q=q_len,
            block_mask=None,
        )

        out_cuda = hpc.attention_with_kvcache_blocksparse_prefill_fp8(
            q,
            kcache,
            vcache,
            qscale,
            kscale,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            q_len,
            quant_type=QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
            block_mask=None,
        )

        ot = out_triton.float()
        oc = out_cuda.float()
        diff = (ot - oc).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        nonzero = oc.abs() > 1e-6
        rel_diff = ((diff / (oc.abs() + 1e-8))[nonzero].max().item()) if nonzero.any() else 0.0

        assert (max_diff < 3.0 and mean_diff < 0.2) or rel_diff < ATTN_RTOL, (
            f"History Triton vs CUDA mismatch: max_abs_diff={max_diff:.6f}, "
            f"max_rel_diff={rel_diff:.6f}, mean_diff={mean_diff:.6f}"
        )


# ===========================================================================
# Standalone execution
# ===========================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("STEM Triton Implementation - Correctness Verification")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Part 1: Triton vs PyTorch Reference
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Part 1: Triton vs PyTorch Reference")
    print("-" * 70)

    print("\n[1/5] Testing stem_oam_prep_paged_kv...")
    test_kv = TestStemOamPrepPagedKv()
    test_kv.test_shape_dtype(1, 256, 4, 1)
    test_kv.test_correctness()
    print("  PASSED")

    print("\n[2/5] Testing stem_oam_prep_varlen_q...")
    test_q = TestStemOamPrepVarlenQ()
    test_q.test_shape_dtype(1, 256, 4)
    test_q.test_correctness()
    print("  PASSED")

    print("\n[3/5] Testing stem_oam_gemm...")
    test_gemm = TestStemOamGemm()
    test_gemm.test_shape_finite(1, 512, 4, 1)
    test_gemm.test_causal_mask()
    print("  PASSED")

    print("\n[4/5] Testing stem_tpd...")
    test_tpd = TestStemTpd()
    test_tpd.test_shape_dtype(1, 1024, 4)
    test_tpd.test_invariants()
    test_tpd.test_sparsity()
    print("  PASSED")

    print("\n[5/5] Testing stem_paged_kv (end-to-end)...")
    test_e2e = TestStemPagedKv()
    test_e2e.test_e2e_shape(1, 512, 4, 1)
    test_e2e.test_e2e_long_sequence()
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Part 2: Triton vs CUDA Kernel (requires Hopper + hpc-ops)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Part 2: Triton vs CUDA Kernel (hpc-ops)")
    print("-" * 70)

    if not HPC_AVAILABLE:
        print(f"\n  SKIPPED: hpc-ops library not available ({_HPC_IMPORT_ERROR})")
    elif not _is_hopper_gpu():
        cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  SKIPPED: Hopper GPU required (current: {gpu_name}, SM{cap[0]}{cap[1]})")
        print("  CUDA kernels use TMA and WGMMA which require SM90+")
    else:
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")

        print("\n[1/5] Comparing prep_paged_kv...")
        test_vs = TestTritonVsCudaPrepPagedKv()
        test_vs.test_kflat_precision(1, 512, 4, 1)
        test_vs.test_vbias_precision(1, 512, 4, 1)
        print("  PASSED")

        print("\n[2/5] Comparing prep_varlen_q...")
        test_vs_q = TestTritonVsCudaPrepVarlenQ()
        test_vs_q.test_qflat_precision(1, 512, 4)
        print("  PASSED")

        print("\n[3/5] Comparing oam_gemm...")
        test_vs_gemm = TestTritonVsCudaOamGemm()
        test_vs_gemm.test_block_logits_precision(1, 512, 4, 1)
        test_vs_gemm.test_causal_mask_agreement(512)
        print("  PASSED")

        print("\n[4/5] Comparing tpd...")
        test_vs_tpd = TestTritonVsCudaTpd()
        test_vs_tpd.test_mask_exact_match(1024, 4)
        test_vs_tpd.test_tpd_with_real_logits(1024)
        print("  PASSED")

        print("\n[5/5] Comparing end-to-end...")
        test_vs_e2e = TestTritonVsCudaE2E()
        test_vs_e2e.test_e2e_mask_agreement(1, 1024, 4, 1)
        test_vs_e2e.test_e2e_invariants_preserved()
        test_vs_e2e.test_e2e_long_sequence_density(4096)
        print("  PASSED")

    # -----------------------------------------------------------------------
    # Part 3: Block-sparse Triton vs PyTorch Reference
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Part 3: Block-sparse Triton vs PyTorch Reference")
    print("-" * 70)

    test_bs = TestBlocksparsePagedAttnFP8()

    print("\n[1/4] Dense attention shape/dtype...")
    test_bs.test_dense_shape_dtype(1, 128, 4, 1)
    print("  PASSED")

    print("\n[2/4] Dense attention correctness...")
    test_bs.test_dense_correctness(128)
    print("  PASSED")

    print("\n[3/4] Sparse attention correctness...")
    test_bs.test_sparse_correctness(0.3)
    print("  PASSED")

    print("\n[4/4] History attention correctness...")
    test_bs.test_with_history()
    print("  PASSED")

    # -----------------------------------------------------------------------
    # Part 4: Block-sparse Triton vs CUDA
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Part 4: Block-sparse Triton vs CUDA (hpc-ops)")
    print("-" * 70)

    if not HPC_AVAILABLE:
        print(f"\n  SKIPPED: hpc-ops library not available ({_HPC_IMPORT_ERROR})")
    elif not _is_hopper_gpu():
        cap = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n  SKIPPED: Hopper GPU required (current: {gpu_name}, SM{cap[0]}{cap[1]})")
        print("  CUDA kernels use TMA and WGMMA which require SM90+")
    else:
        test_bs_vs = TestTritonVsCudaBlocksparsePrefillFP8()

        print("\n[1/3] Dense output precision...")
        test_bs_vs.test_dense_precision(1, 256, 4, 1)
        print("  PASSED")

        print("\n[2/3] Sparse output precision...")
        test_bs_vs.test_sparse_precision(0.3)
        print("  PASSED")

        print("\n[3/3] History output precision...")
        test_bs_vs.test_with_history_precision()
        print("  PASSED")

    print("\n" + "=" * 70)
    print("All correctness tests PASSED!")
    print("=" * 70)

