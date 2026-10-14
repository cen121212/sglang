"""
Unit tests for the NPU prefill context parallel paths in
sglang.srt.hardware_backend.npu.attention.ascend_backend.

These guard the two defects left behind when the CP v1 runtime was removed:

* the KV all-gather for prefill CP must go through the CP v2 strategy, not
  through the deleted ``cp_all_gather_rerange_kv_cache`` shim;
* ``do_cp_attn_fia`` must hand FIA a query whose row count matches the
  cumulative ``actual_seq_lengths`` it advertises -- the model runner pads the
  CP shard to ``per_rank_actual_token[0]`` rows, which previously made FIA
  tiling fail with error 561002.

Everything runs on CPU tensors, so the checks are hardware independent.
"""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# Mock NPU-only modules before importing the source module.
for _ in (
    "torch_npu",
    "torch_npu.contrib",
    "sgl_kernel_npu",
    "sgl_kernel_npu.attention",
    "sgl_kernel_npu.attention.sinks_attention",
    "sglang.srt.speculative",
    "sglang.srt.speculative.decoupled_spec_io",
    "sglang.srt.speculative.spec_info",
    "sglang.srt.speculative.eagle_info",
):
    sys.modules.setdefault(_, unittest.mock.MagicMock())

from sglang.srt.hardware_backend.npu.attention import ascend_backend
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    DSACPSegmentMetadata,
    _cp_allgather_and_save_kv_npu,
)
from sglang.srt.layers.attention.dsa import dsa_npu_indexer


class _RecordingPool:
    def __init__(self):
        self.calls = []

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        self.calls.append((tuple(k.shape), tuple(v.shape)))


class _RecordingStrategy:
    def __init__(self, cp_size=2):
        self.cp_size = cp_size
        self.kv_calls = []

    def materialize_full_kv(self, forward_batch, layer, k, v, swa_loc=None):
        self.kv_calls.append((tuple(k.shape), tuple(v.shape)))


def _cp_batch(out_cache_loc):
    return SimpleNamespace(out_cache_loc=out_cache_loc, encoder_out_cache_loc=None)


def _attn_layer():
    return SimpleNamespace(is_cross_attention=False, k_scale=None, v_scale=None)


def _fia_layer():
    return SimpleNamespace(
        tp_q_head_num=16,
        tp_k_head_num=8,
        tp_v_head_num=8,
        qk_head_dim=128,
        v_head_dim=128,
        scaling=0.08838834764831845,
    )


class TestCPKVWrite(unittest.TestCase):
    """`_cp_allgather_and_save_kv_npu` must use the CP v2 strategy API."""

    def setUp(self):
        self.k = torch.zeros(3, 8, 128, dtype=torch.bfloat16)
        self.v = torch.zeros(3, 8, 128, dtype=torch.bfloat16)
        self.batch = _cp_batch(torch.arange(6))
        self.layer = _attn_layer()

    def test_uses_cp_v2_strategy(self):
        strategy = _RecordingStrategy(cp_size=2)
        pool = _RecordingPool()
        with patch.object(ascend_backend, "get_cp_strategy", return_value=strategy):
            _cp_allgather_and_save_kv_npu(
                self.batch, self.layer, self.k, self.v, 2, pool, swa_loc=None
            )
        self.assertEqual(len(strategy.kv_calls), 1)
        self.assertEqual(strategy.kv_calls[0], ((3, 8, 128), (3, 8, 128)))
        # the strategy owns the pool write; the helper must not bypass it
        self.assertEqual(pool.calls, [])

    def test_falls_back_to_plain_write_without_cp(self):
        strategy = _RecordingStrategy(cp_size=1)
        pool = _RecordingPool()
        with patch.object(ascend_backend, "get_cp_strategy", return_value=strategy):
            _cp_allgather_and_save_kv_npu(
                self.batch, self.layer, self.k, self.v, 1, pool, swa_loc=None
            )
        self.assertEqual(strategy.kv_calls, [])
        self.assertEqual(pool.calls, [((3, 8, 128), (3, 8, 128))])

    def test_deleted_cp_v1_helper_is_not_referenced(self):
        with open(ascend_backend.__file__) as handle:
            source = handle.read()
        self.assertNotIn("cp_all_gather_rerange_kv_cache", source)


class TestZigzagSparseAttention(unittest.TestCase):
    @staticmethod
    def _segments(meta, tables):
        split = meta.total_q_prev_tokens
        return (
            DSACPSegmentMetadata(
                0, split, meta.cu_seqlens_q_prev_tensor[1:],
                meta.kv_len_prev_tensor, tables,
            ),
            DSACPSegmentMetadata(
                split, split + meta.total_q_next_tokens,
                meta.cu_seqlens_q_next_tensor[1:], meta.kv_len_next_tensor, tables,
            ),
        )

    def test_prepare_zigzag_batched_segments(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        tables = torch.tensor([[11, 12], [21, 22]], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(block_tables=tables)
        # CP2 rank0: A length10/prefix3, B length14/prefix5.
        meta = SimpleNamespace(
            bs=2, total_seq_lens=24, per_rank_logical_token=None,
            total_q_prev_tokens=7, total_q_next_tokens=5,
            cu_seqlens_q_prev_tensor=torch.tensor([0, 3, 7]),
            cu_seqlens_q_next_tensor=torch.tensor([0, 2, 5]),
            kv_len_prev_tensor=torch.tensor([6, 9]),
            kv_len_next_tensor=torch.tensor([13, 19]),
        )
        batch = SimpleNamespace(
            attn_cp_metadata=meta,
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: True,
                is_target_verify=lambda: False,
            ),
            positions=torch.cat([torch.arange(3, 13), torch.arange(5, 19)]),
            extend_seq_lens_cpu=[10, 14],
        )
        strategy = SimpleNamespace(
            cp_size=2,
            local_q_indices=lambda *_: torch.tensor(
                [0, 1, 2, 10, 11, 12, 13, 8, 9, 21, 22, 23]
            ),
        )
        with (
            patch.object(ascend_backend, "get_cp_strategy", return_value=strategy),
            patch.object(ascend_backend, "is_zigzag", return_value=True),
        ):
            backend.prepare_dsa_cp_metadata(batch)
            backend.prepare_dsa_cp_metadata(batch)
        fm = backend.forward_metadata
        self.assertIs(fm.block_tables, tables)
        self.assertIs(fm.dsa_cp_request_block_tables, tables)
        prev, next_ = fm.dsa_cp_segments
        self.assertEqual((prev.start, prev.end, next_.start, next_.end), (0, 7, 7, 12))
        self.assertEqual(prev.actual_seq_lengths_q.tolist(), [3, 7])
        self.assertEqual(next_.actual_seq_lengths_q.tolist(), [2, 5])
        self.assertEqual(prev.actual_seq_lengths_kv.tolist(), [6, 9])
        self.assertEqual(next_.actual_seq_lengths_kv.tolist(), [13, 19])
        self.assertIs(prev.block_tables, tables)
        self.assertIs(next_.block_tables, tables)

    def test_prepare_retains_request_tables_for_attention(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        tables = torch.tensor([[11, 12]], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(block_tables=tables)
        batch = SimpleNamespace(
            attn_cp_metadata=SimpleNamespace(
                total_seq_lens=8, per_rank_logical_token=None
            ),
            forward_mode=SimpleNamespace(
                is_context_parallel_extend=lambda: True,
                is_target_verify=lambda: False,
            ),
            positions=torch.arange(8),
            extend_seq_lens_cpu=[8],
        )
        strategy = SimpleNamespace(
            cp_size=2,
            local_q_indices=lambda *_: torch.tensor([0, 1, 6, 7]),
        )
        with (
            patch.object(ascend_backend, "get_cp_strategy", return_value=strategy),
            patch.object(ascend_backend, "is_zigzag", return_value=False),
        ):
            backend.prepare_dsa_cp_metadata(batch)
            # Preparation is idempotent: never save the expanded table as original.
            backend.prepare_dsa_cp_metadata(batch)
        fm = backend.forward_metadata
        self.assertIs(fm.dsa_cp_request_block_tables, tables)
        self.assertEqual(fm.block_tables.shape, (4, 2))
        self.assertEqual(fm.actual_seq_lengths_q.tolist(), [1, 2, 3, 4])
        self.assertEqual(fm.actual_seq_lengths_kv.tolist(), [1, 2, 7, 8])

    def test_two_segments_match_per_token_causal_reference(self):
        """Uneven front/back blocks, two requests, prefixes and physical pad."""
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        request_tables = torch.tensor([[0], [1]], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            dsa_cp_request_block_tables=request_tables,
            # Deliberately different: attention must not use per-token tables.
            block_tables=torch.tensor([[0], [1], [1], [0], [0], [1], [0]]),
        )
        # Request 0: front at 2, back at 7/8; request 1: front at 3/4,
        # back at 9. Rows are [all fronts, all backs, padding].
        positions = [2, 3, 4, 7, 8, 9]
        request_ids = [0, 1, 1, 0, 0, 1]
        q = torch.arange(7 * 2, dtype=torch.float32).reshape(7, 1, 2) / 10
        kv = torch.arange(2 * 12 * 2, dtype=torch.float32).reshape(2, 12, 1, 2) / 20
        rope = torch.zeros(7, 1, 1)
        indices = torch.arange(12).expand(7, 12).clone()
        meta = SimpleNamespace(
            total_q_prev_tokens=3,
            total_q_next_tokens=3,
            cu_seqlens_q_prev_tensor=torch.tensor([0, 1, 3]),
            cu_seqlens_q_next_tensor=torch.tensor([0, 2, 3]),
            kv_len_prev_tensor=torch.tensor([3, 5]),
            kv_len_next_tensor=torch.tensor([9, 10]),
        )
        backend.forward_metadata.dsa_cp_segments = self._segments(meta, request_tables)
        calls = []

        def reference_row(query, request_id, end):
            keys = kv[request_id, :end]
            scores = (keys * query).sum(-1)
            return (scores.softmax(dim=0).unsqueeze(-1) * keys).sum(0)

        def sparse_attention(**kwargs):
            calls.append(kwargs)
            self.assertIs(kwargs["block_table"], request_tables)
            self.assertIs(kwargs["key"], kv)
            self.assertIs(kwargs["value"], kv)
            query = kwargs["query"]
            cumulative = kwargs["actual_seq_lengths_query"].tolist()
            lengths = kwargs["actual_seq_lengths_kv"].tolist()
            self.assertEqual(cumulative[-1], query.shape[0])
            rows, start = [], 0
            for request_id, (end, kv_len) in enumerate(zip(cumulative, lengths)):
                for row in range(start, end):
                    # Bottom-right causal alignment of a contiguous Q block.
                    visible = kv_len - (end - start) + row - start + 1
                    rows.append(reference_row(query[row], request_id, visible))
                start = end
            return torch.stack(rows), None, None

        with patch.object(
            ascend_backend.torch_npu,
            "npu_sparse_flash_attention",
            side_effect=sparse_attention,
        ):
            out = backend.do_cp_zigzag_attn(
                q,
                kv,
                rope,
                torch.zeros_like(kv[..., :1]),
                indices,
                SimpleNamespace(scaling=1.0),
                SimpleNamespace(attn_cp_metadata=meta),
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["actual_seq_lengths_query"].tolist(), [1, 3])
        self.assertEqual(calls[1]["actual_seq_lengths_query"].tolist(), [2, 3])
        for call, start, end in ((calls[0], 0, 3), (calls[1], 3, 6)):
            torch.testing.assert_close(call["query"], q[start:end])
            torch.testing.assert_close(call["query_rope"], rope[start:end])
            self.assertEqual(call["sparse_indices"].shape[0], end - start)
        expected = torch.stack(
            [
                reference_row(q[row], request_id, position + 1)
                for row, (request_id, position) in enumerate(zip(request_ids, positions))
            ]
        )
        torch.testing.assert_close(out[:6], expected)
        torch.testing.assert_close(out[6:], torch.zeros_like(out[6:]))

    def test_split_uses_metadata_not_physical_midpoint(self):
        backend = AscendAttnBackend.__new__(AscendAttnBackend)
        backend.forward_metadata = SimpleNamespace(
            dsa_cp_request_block_tables=torch.zeros(1, 1, dtype=torch.int32)
        )
        meta = SimpleNamespace(
            total_q_prev_tokens=2,
            total_q_next_tokens=1,
            cu_seqlens_q_prev_tensor=torch.tensor([0, 2]),
            cu_seqlens_q_next_tensor=torch.tensor([0, 1]),
            kv_len_prev_tensor=torch.tensor([4]),
            kv_len_next_tensor=torch.tensor([9]),
        )
        backend.forward_metadata.dsa_cp_segments = self._segments(
            meta, backend.forward_metadata.dsa_cp_request_block_tables
        )
        q = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
        calls = []

        def sparse_attention(**kwargs):
            calls.append(kwargs)
            return kwargs["query"].clone(), None, None

        with patch.object(
            ascend_backend.torch_npu,
            "npu_sparse_flash_attention",
            side_effect=sparse_attention,
        ):
            out = backend.do_cp_zigzag_attn(
                q,
                q,
                q,
                q,
                torch.zeros(3, 1, dtype=torch.int32),
                SimpleNamespace(scaling=1.0),
                SimpleNamespace(attn_cp_metadata=meta),
            )
        self.assertEqual([c["query"].shape[0] for c in calls], [2, 1])
        torch.testing.assert_close(out[:3], q[:3])
        self.assertEqual(out.shape, q.shape)
        torch.testing.assert_close(out[3:], torch.zeros_like(q[3:]))

    def test_indexer_segments_align_query_weights_and_topk(self):
        indexer = dsa_npu_indexer.DSANPUIndexerMixin()
        indexer.index_topk = 2
        tables = torch.tensor([[0], [1]], dtype=torch.int32)
        meta = SimpleNamespace(
            total_q_prev_tokens=3, total_q_next_tokens=2,
            cu_seqlens_q_prev_tensor=torch.tensor([0, 1, 3]),
            cu_seqlens_q_next_tensor=torch.tensor([0, 1, 2]),
            kv_len_prev_tensor=torch.tensor([3, 5]),
            kv_len_next_tensor=torch.tensor([9, 10]),
        )
        backend = SimpleNamespace(forward_metadata=SimpleNamespace(
            dsa_cp_segments=self._segments(meta, tables)
        ))
        q = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
        weights = torch.arange(8, dtype=torch.float32).view(8, 1)
        key = torch.zeros(2, 12, 1, 1)
        calls = []

        def lightning_indexer(**kwargs):
            calls.append(kwargs)
            query = kwargs["query"]
            # Encode each Q row into its TopK to verify concatenation order.
            indices = query.to(torch.int32).expand(-1, 1, 2).clone()
            return (indices,)

        fake_npu = SimpleNamespace(npu_lightning_indexer=lightning_indexer)
        with (
            patch.object(dsa_npu_indexer, "get_attn_backend", return_value=backend),
            patch.object(dsa_npu_indexer, "torch_npu", fake_npu, create=True),
        ):
            out = indexer.do_npu_cp_zigzag_indexer(q, key, weights)
        self.assertEqual(len(calls), 2)
        for call, start, end, q_lengths in (
            (calls[0], 0, 3, [1, 3]), (calls[1], 3, 5, [1, 2])
        ):
            self.assertIs(call["block_table"], tables)
            self.assertIs(call["key"], key)
            torch.testing.assert_close(call["query"], q[start:end])
            torch.testing.assert_close(call["weights"], weights[start:end])
            self.assertEqual(call["actual_seq_lengths_query"].tolist(), q_lengths)
        self.assertEqual(out.shape, (5, 2))
        self.assertEqual(out[:, 0].tolist(), [0, 1, 2, 3, 4])


class TestDoCpAttnFia(unittest.TestCase):
    """`do_cp_attn_fia` must trim and re-append the physical pad rows."""

    PHYSICAL_ROWS = 4

    def setUp(self):
        self.backend = AscendAttnBackend.__new__(AscendAttnBackend)
        self.backend.page_size = 128
        self.backend.device = torch.device("cpu")
        self.backend.fia_mask = torch.zeros(2048, 2048, dtype=torch.bool)
        self.backend.forward_metadata = SimpleNamespace(
            block_tables=torch.zeros(1, 1, dtype=torch.int32)
        )
        self.layer = _fia_layer()
        self.q = torch.randn(self.PHYSICAL_ROWS, 16 * 128, dtype=torch.bfloat16)
        self.k_cache = torch.zeros(2163 * 128, 8 * 128, dtype=torch.bfloat16)
        self.v_cache = torch.zeros(2163 * 128, 8 * 128, dtype=torch.bfloat16)
        self.calls = []

    def _fia(self, query, key, value, **kwargs):
        record = dict(kwargs)
        record["query"] = query
        self.calls.append(record)
        return (
            torch.zeros(query.shape[0], query.shape[1], 128, dtype=query.dtype),
            None,
        )

    def _run(self, total_q_prev, total_q_next, prev_len, next_len):
        cp_meta = SimpleNamespace(
            total_q_prev_tokens=total_q_prev,
            total_q_next_tokens=total_q_next,
            actual_seq_q_prev_list=[prev_len],
            actual_seq_q_next_list=[next_len],
            kv_len_prev_list=[prev_len],
            kv_len_next_list=[total_q_prev + total_q_next],
        )
        forward_batch = SimpleNamespace(attn_cp_metadata=cp_meta)
        fake_ops = SimpleNamespace(
            npu=SimpleNamespace(npu_fused_infer_attention_score=self._fia)
        )
        with patch.object(torch, "ops", fake_ops):
            return self.backend.do_cp_attn_fia(
                self.q, self.k_cache, self.v_cache, self.layer, forward_batch
            )

    def test_pad_rows_are_trimmed_before_fia(self):
        # logical tokens are 2 + 1 = 3 of the 4 physical rows
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["query"].shape[0], 2)
        self.assertEqual(self.calls[1]["query"].shape[0], 1)

    def test_query_rows_match_cumulative_actual_seq_lengths(self):
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        for call in self.calls:
            self.assertEqual(call["query"].shape[0], sum(call["actual_seq_lengths"]))

    def test_kv_lengths_are_forwarded(self):
        self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(self.calls[0]["actual_seq_lengths_kv"], [2])
        self.assertEqual(self.calls[1]["actual_seq_lengths_kv"], [3])

    def test_output_keeps_physical_row_count(self):
        out = self._run(total_q_prev=2, total_q_next=1, prev_len=2, next_len=1)
        self.assertEqual(out.shape[0], self.PHYSICAL_ROWS)
        self.assertEqual(out.shape[1], 16 * 128)

    def test_no_padding_keeps_all_rows(self):
        # 2 + 2 == physical rows, so nothing is trimmed
        self._run(total_q_prev=2, total_q_next=2, prev_len=2, next_len=2)
        self.assertEqual(self.calls[0]["query"].shape[0], 2)
        self.assertEqual(self.calls[1]["query"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
