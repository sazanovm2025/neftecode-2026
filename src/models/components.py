"""Building blocks for the Set-Transformer.

Both blocks are pre-norm variants for stability on small training sets
(167 scenarios total) and support a ``key_padding_mask`` where the mask
uses **1 for real elements** and **0 for padding** — the same convention
as :func:`src.data.dataset.collate_fn`. Internally we invert to the
``True = padding`` convention expected by :class:`torch.nn.MultiheadAttention`.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


def _to_mha_pad_mask(key_padding_mask: Optional[Tensor]) -> Optional[Tensor]:
    """Convert our 1=real / 0=pad mask to PyTorch MHA's True=pad convention.

    Also returns ``None`` if no padding — letting MHA take a faster path.
    """
    if key_padding_mask is None:
        return None
    # Pad positions are those with mask == 0.
    pad = key_padding_mask == 0
    if not pad.any():
        return None
    return pad


# ---------------------------------------------------------------------------
# Set Attention Block (pre-norm, self-attention)
# ---------------------------------------------------------------------------


class SetAttentionBlock(nn.Module):
    """Pre-norm self-attention block with a feed-forward residual.

    Sequence::

        h = LN(x);  a = MHA(h, h, h);  x = x + a
        h = LN(x);  f = FF(h);         x = x + f
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim_model)
        self.ff = nn.Sequential(
            nn.Linear(dim_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        h = self.norm1(x)
        pad_mask = _to_mha_pad_mask(key_padding_mask)
        a, _ = self.attn(
            h, h, h, key_padding_mask=pad_mask, need_weights=False
        )
        x = x + a
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


# ---------------------------------------------------------------------------
# Pooling by Multihead Attention (PMA with 1 seed vector)
# ---------------------------------------------------------------------------


class PoolingByMultiheadAttention(nn.Module):
    """PMA with a single learned seed vector.

    The seed acts as a query attending over the set elements. We expose
    the attention weights as ``self.last_attention_weights`` so downstream
    analysis / interpretability code can inspect which components the
    model is weighting.
    """

    def __init__(self, dim_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, dim_model))
        nn.init.xavier_uniform_(self.seed)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim_model)
        self.last_attention_weights: Optional[Tensor] = None

    def forward(
        self, x: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        B = x.shape[0]
        q = self.seed.expand(B, -1, -1)  # (B, 1, D)
        pad_mask = _to_mha_pad_mask(key_padding_mask)
        out, attn_w = self.attn(
            q,
            x,
            x,
            key_padding_mask=pad_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_w: (B, num_heads, 1, N)  when average_attn_weights=False
        self.last_attention_weights = attn_w.detach()
        pooled = self.norm(out.squeeze(1))  # (B, D)
        return pooled
