"""Tests for SetAttentionBlock and PoolingByMultiheadAttention."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.components import PoolingByMultiheadAttention, SetAttentionBlock


# ---------------------------------------------------------------------------
# SetAttentionBlock
# ---------------------------------------------------------------------------


def test_sab_output_shape():
    sab = SetAttentionBlock(dim_model=64, num_heads=4, ff_dim=128, dropout=0.0)
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10)
    mask[:, -2:] = 0
    out = sab(x, key_padding_mask=mask)
    assert out.shape == (4, 10, 64)
    assert not torch.isnan(out).any()


def test_sab_respects_padding():
    """Output at non-padded positions must be invariant to padded-position noise."""
    torch.manual_seed(0)
    sab = SetAttentionBlock(dim_model=32, num_heads=4, ff_dim=64, dropout=0.0).eval()
    x = torch.randn(3, 8, 32)
    mask = torch.ones(3, 8)
    mask[:, -3:] = 0
    out_a = sab(x, key_padding_mask=mask)[:, :5, :]

    x2 = x.clone()
    x2[:, -3:, :] = torch.randn(3, 3, 32) * 50.0
    out_b = sab(x2, key_padding_mask=mask)[:, :5, :]

    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)


def test_sab_no_mask_still_works():
    sab = SetAttentionBlock(dim_model=16, num_heads=2, ff_dim=32, dropout=0.0)
    x = torch.randn(2, 5, 16)
    out = sab(x)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# PoolingByMultiheadAttention
# ---------------------------------------------------------------------------


def test_pma_output_shape():
    pma = PoolingByMultiheadAttention(dim_model=64, num_heads=4, dropout=0.0)
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10)
    out = pma(x, key_padding_mask=mask)
    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()


def test_pma_attention_saved_and_correct_shape():
    pma = PoolingByMultiheadAttention(dim_model=64, num_heads=4, dropout=0.0)
    x = torch.randn(4, 10, 64)
    _ = pma(x)
    assert pma.last_attention_weights is not None
    # (B, num_heads, 1, N) when average_attn_weights=False
    assert pma.last_attention_weights.shape == (4, 4, 1, 10)


def test_pma_respects_padding():
    torch.manual_seed(1)
    pma = PoolingByMultiheadAttention(dim_model=32, num_heads=4, dropout=0.0).eval()
    x = torch.randn(3, 8, 32)
    mask = torch.ones(3, 8)
    mask[:, -3:] = 0
    out_a = pma(x, key_padding_mask=mask)
    x2 = x.clone()
    x2[:, -3:, :] = torch.randn(3, 3, 32) * 50.0
    out_b = pma(x2, key_padding_mask=mask)
    torch.testing.assert_close(out_a, out_b, atol=1e-5, rtol=1e-5)
