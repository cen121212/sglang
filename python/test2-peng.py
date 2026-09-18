"""Compare the target-verify FIA v2 call with CANN Flash MLA on an Ascend NPU.

From the sglang repository, with torch, torch_npu and the matching
CANN/custom-op libraries installed:

    python test/manual/attention/test_npu_flash_mla_with_kvcache.py --tp-size 8

Total Query heads are 96. --tp-size selects 1/2/4/8 (default: 1), so each
operator runs with 96/48/24/12 local Query heads and one replicated KV head.
This is a single-device comparison of one TP rank's operator geometry; it
does not launch distributed workers or test TP communication.
Only verify-mixed-lengths is tested: query_len=4, KV lengths=[4, 127, 513].
For each dtype, inputs are prepared once and each attention operator runs
10 times. Flash MLA metadata is regenerated in every iteration before attention.
The final outputs are checked against each other and FP32.

If cann_ops_transformer is not installed, the unpacked package under the
repository's python directory is used automatically.

The source call is hardware_backend/npu/attention/ascend_backend.py:2374.
This tests its ND paged-cache path (not PA_NZ or quantized caches). All KV
lengths include the current query tokens. Only attention output is compared;
the source call discards LSE. Missing dependencies or a failed case exit nonzero.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tp-size", "--tp_size", type=int, choices=(1, 2, 4, 8), default=1,
        help="Tensor parallel size; local Query heads = 96 / tp_size (default: 1)",
    )
    return parser.parse_args()


try:
    import torch
    import torch_npu
except ImportError as exc:
    raise SystemExit(f"Required NPU dependency could not be loaded: {exc}") from exc
if not torch.npu.is_available():
    raise SystemExit("An Ascend NPU is required to run this comparison.")

if importlib.util.find_spec("cann_ops_transformer") is None:
    package_dir = (
        Path(__file__).resolve().parents[3]
        / "python"
        / "cann_ops_transformer-1.0.0-py3-none-any"
    )
    if not (package_dir / "cann_ops_transformer").is_dir():
        raise SystemExit("Install cann_ops_transformer before running this script.")
    sys.path.insert(0, str(package_dir))

# Once an NPU is present, package/operator loading errors must fail the test.
from cann_ops_transformer.ops.attention.flash_mla_with_kvcache import (
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_metadata,
)


HEAD_NUM = 96
NUM_ITERATIONS = 10
HEAD_DIM_V = 512
HEAD_DIM_ROPE = 64
PAGE_SIZE = 128
FULL_ATTENTION_WINDOW = 2147483647


def _reference(q, cache, block_table, kv_lengths, scale):
    """Independent FP32 CPU reference, with bottom-right causal alignment."""
    batch, query_len, heads, _ = q.shape
    output = torch.empty(batch, query_len, heads, HEAD_DIM_V, dtype=torch.float32)
    q = q.float()
    cache = cache.float()
    for b, kv_len in enumerate(kv_lengths):
        pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
        ids = block_table[b, :pages].long()
        kv = cache[ids].reshape(-1, HEAD_DIM_V + HEAD_DIM_ROPE)[:kv_len]
        logits = torch.einsum("shd,kd->hsk", q[b], kv) * scale
        last_visible = kv_len - query_len + torch.arange(query_len)
        masked = torch.arange(kv_len)[None, :] > last_visible[:, None]
        probs = logits.masked_fill(masked[None], float("-inf")).softmax(dim=-1)
        output[b] = torch.einsum("hsk,kd->shd", probs, kv[:, :HEAD_DIM_V])
    return output


def _assert_close(actual, expected, *, label, atol, rtol):
    assert actual.shape == expected.shape, label
    assert torch.isfinite(actual).all(), f"{label}: non-finite output"
    assert torch.isfinite(expected).all(), f"{label}: non-finite reference"
    diff = (actual.float() - expected.float()).abs()
    print(
        f"{label}: max_abs={diff.max().item():.6g}, "
        f"mean_abs={diff.mean().item():.6g}, atol={atol}, rtol={rtol}"
    )
    torch.testing.assert_close(
        actual.float(), expected.float(), atol=atol, rtol=rtol, msg=label
    )


@torch.inference_mode()
def test_flash_mla_matches_fia_v2(dtype, query_len, kv_lengths, num_heads):
    # Build on CPU for a reproducible reference using the exact rounded inputs.
    rng = torch.Generator().manual_seed(20260916)
    batch = len(kv_lengths)
    max_kv_len = max(kv_lengths)
    max_pages = (max_kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = batch * max_pages
    q_cpu = torch.randn(
        batch, query_len, num_heads, HEAD_DIM_V + HEAD_DIM_ROPE, generator=rng
    ).to(dtype)
    cache_cpu = torch.randn(
        num_blocks, PAGE_SIZE, 1, HEAD_DIM_V + HEAD_DIM_ROPE, generator=rng
    ).to(dtype)
    # Nonsequential physical pages expose block-table/layout mapping errors.
    table_cpu = torch.randperm(num_blocks, generator=rng).reshape(batch, max_pages)
    table_cpu = table_cpu.to(torch.int32)
    scale = (128 + HEAD_DIM_ROPE) ** -0.5  # Explicit DeepSeek-style layer scale.
    expected = _reference(q_cpu, cache_cpu, table_cpu, kv_lengths, scale)

    device = torch.device("npu", torch.npu.current_device())
    q = q_cpu.to(device)
    cache = cache_cpu.to(device)
    block_table = table_cpu.to(device)
    cache_seqlens = torch.tensor(kv_lengths, dtype=torch.int32, device=device)
    seqused_q = torch.full((batch,), query_len, dtype=torch.int32, device=device)
    # Same compressed causal-mask convention as AscendAttentionBackend.mtp_mask.
    attn_mask = torch.ones(2048, 2048, dtype=torch.bool).triu_(1).to(device)
    # Flash MLA requires INT8; FIA v2 uses BOOL. Casting preserves the same
    # mask values (0 = visible, 1 = masked) without changing causal alignment.
    mla_attn_mask = attn_mask.to(dtype=torch.int8)

    q_nope = q[..., :HEAD_DIM_V].transpose(1, 2).contiguous()
    q_rope = q[..., HEAD_DIM_V:].transpose(1, 2).contiguous()
    c_kv_cache = cache[..., :HEAD_DIM_V].transpose(1, 2).contiguous()
    k_rope_cache = cache[..., HEAD_DIM_V:].transpose(1, 2).contiguous()

    actual_seq_qlen = [query_len] * batch
    actual_seq_kvlen = list(kv_lengths)
    torch.npu.synchronize()

    # Reuse prepared inputs for ten executions; compare the final outputs.
    for _ in range(NUM_ITERATIONS):
        fia_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q_nope,
            c_kv_cache,
            c_kv_cache,
            query_rope=q_rope,
            key_rope=k_rope_cache,
            num_query_heads=num_heads,
            num_key_value_heads=1,
            input_layout="BNSD",
            softmax_scale=scale,
            block_table=block_table,
            block_size=PAGE_SIZE,
            sparse_mode=3,
            atten_mask=attn_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            pre_tokens=FULL_ATTENTION_WINDOW,
            next_tokens=0,
        )
    torch.npu.synchronize()

    for _ in range(NUM_ITERATIONS):
        metadata = flash_mla_with_kvcache_metadata(
            cache_seqlens=cache_seqlens,
            num_heads_q=num_heads,
            num_heads_kv=1,
            seqused_q=seqused_q,
            max_seqlen_q=query_len,
            max_seqlen_kv=max_kv_len,
            head_dim_qk=HEAD_DIM_V + HEAD_DIM_ROPE,
            head_dim_v=HEAD_DIM_V,
            mask_mode=3,
            layout_q="BSND",
        )
        mla_output, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            seqused_q=seqused_q,
            attn_mask=mla_attn_mask,
            metadata=metadata,
            head_dim_v=HEAD_DIM_V,
            softmax_scale=scale,
            mask_mode=3,
            max_seqlen_q=query_len,
            max_seqlen_kv=max_kv_len,
            layout_q="BSND",
            layout_kv="PA_BBND",
            layout_out="BSND",
            return_softmax_lse=False,
        )
    torch.npu.synchronize()

    assert fia_output.dtype == mla_output.dtype == dtype
    # Normalize both outputs to the backend's final [B*S, N, 512] layout.
    shape = (batch * query_len, num_heads, HEAD_DIM_V)
    assert tuple(fia_output.shape) == (batch, num_heads, query_len, HEAD_DIM_V)
    assert tuple(mla_output.shape) == (batch, query_len, num_heads, HEAD_DIM_V)
    fia = fia_output.transpose(1, 2).contiguous().reshape(shape).cpu()
    mla = mla_output.reshape(shape).cpu()
    expected = expected.reshape(shape)
    # Numerical equivalence, not bitwise equality across different kernels.
    atol, rtol = (2e-3, 2e-3) if dtype == torch.float16 else (1e-2, 1e-2)
    _assert_close(mla, fia, label="Flash MLA vs FIA v2", atol=atol, rtol=rtol)
    _assert_close(fia, expected, label="FIA v2 vs FP32", atol=atol, rtol=rtol)
    _assert_close(mla, expected, label="Flash MLA vs FP32", atol=atol, rtol=rtol)


def main():
    # num_heads = HEAD_NUM // tp_size
    # xx = 24 * 8
    # print(
    #     f"Single-rank operator comparison: head_num={HEAD_NUM}, "
    #     f"tp_size={tp_size}, local_query_heads={num_heads}, kv_heads=1",
    #     flush=True,
    # )
    cases = [
        ("16-12", 4, (131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072,), 12),
        ("8-24", 4, (131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072,), 24),
        ("4-48", 4, (131072, 131072, 131072, 131072,), 48),
        ("2-96", 4, (131072, 131072,), 96),
        ("16-12", 4, (131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072,), 12),
        ("8-24", 4, (131072, 131072, 131072, 131072,), 24),
        ("4-48", 4, (131072, 131072,), 48),
        ("2-96", 4, (131072,), 96),
        ("2-96", 4, (131072,), 96),
        ("8-96", 4, (131072, 131072, 131072, 131072, 131072, 131072, 131072, 131072,), 96),


    ]
    passed = 0
    dtype = torch.bfloat16
    for name, query_len, kv_lengths, num_heads in cases:
        label = (
            f"{name}, dtype={dtype}, num_heads={num_heads}, "
            f"local_query_heads={num_heads}, iterations={NUM_ITERATIONS}"
        )
        print(f"\n[RUN] {label}", flush=True)
        try:
            test_flash_mla_matches_fia_v2(
                dtype, query_len, kv_lengths, num_heads
            )
        except Exception:
            print(f"[FAIL] {label}", file=sys.stderr, flush=True)
            raise
        passed += 1
        print(f"[PASS] {label}", flush=True)
    print(f"\nAll {passed} cases passed.", flush=True)


if __name__ == "__main__":
    # args = parse_args()
    main()
