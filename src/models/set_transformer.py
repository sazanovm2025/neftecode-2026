"""Set-Transformer variant for compositional regression.

Architecture overview::

    component_features (B, N, 50)
        │ split by schema: cols 0,1 = comp_id_idx, type_idx; cols 2..49 = 48 floats
        │
        │  comp_emb + type_emb concat with 48 floats → (B, N, 48 + 16 + 8 = 72)
        │  Linear(72 → 64) + LN + GELU     (component projection)
        │
        │  SetAttentionBlock × 2
        │
        │  masked_mean_pool    → (B, 64)
        │  PoolingByMultiheadAttention → (B, 64)   (PMA with 1 seed)
        │  concat → (B, 128)
        ▼
    component_pool (B, 128)

    scenario_features (B, 40)
        │ col 0 = mode_id_idx → Embedding(11, 8)
        │ cols 1..39 = 39 floats  → concat with mode_emb → (B, 47)
        │ Linear(47 → 64) + LN + GELU + Dropout
        ▼
    scenario_emb (B, 64)

    fused = concat(component_pool, scenario_emb) → (B, 192)
    Linear(192 → 128) + LN + GELU + Dropout
        ├─ regression_head: Linear(128 → 2)
        └─ sign_head:       Linear(128 → 1)

The regression head predicts in z-transformed target space; sign head
predicts probability that raw Delta KV100 is strictly positive.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .components import PoolingByMultiheadAttention, SetAttentionBlock


class SetTransformerModel(nn.Module):
    def __init__(
        self,
        n_components: int,
        n_types: int,
        n_modes: int,
        d_comp_numeric: int = 48,
        d_scen_other: int = 39,
        comp_emb_dim: int = 16,
        type_emb_dim: int = 8,
        mode_emb_dim: int = 8,
        dim_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_encoder_blocks: int = 2,
        fusion_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # -- component path ------------------------------------------------
        self.comp_emb = nn.Embedding(n_components + 1, comp_emb_dim)
        self.type_emb = nn.Embedding(n_types + 1, type_emb_dim)
        self.component_proj = nn.Sequential(
            nn.Linear(d_comp_numeric + comp_emb_dim + type_emb_dim, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(),
        )
        self.encoder = nn.ModuleList(
            [
                SetAttentionBlock(
                    dim_model=dim_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_encoder_blocks)
            ]
        )
        self.pma = PoolingByMultiheadAttention(
            dim_model=dim_model, num_heads=num_heads, dropout=dropout
        )

        # -- scenario path -------------------------------------------------
        self.mode_emb = nn.Embedding(n_modes + 1, mode_emb_dim)
        self.scenario_mlp = nn.Sequential(
            nn.Linear(d_scen_other + mode_emb_dim, dim_model),
            nn.LayerNorm(dim_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # -- fusion + heads ------------------------------------------------
        fusion_in = dim_model * 2 + dim_model  # mean + pma + scenario
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(fusion_hidden, 2)
        self.sign_head = nn.Linear(fusion_hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        component_features: Tensor,
        component_mask: Tensor,
        scenario_features: Tensor,
    ) -> dict[str, Tensor]:
        # component_features: (B, N, 50). Per component_feature_schema:
        # col 0 = component_idx, col 1 = type_idx, cols 2..49 = 48 floats.
        comp_idx = component_features[..., 0].long()
        type_idx = component_features[..., 1].long()
        numeric = component_features[..., 2:]

        c_emb = self.comp_emb(comp_idx)       # (B, N, comp_emb_dim)
        t_emb = self.type_emb(type_idx)       # (B, N, type_emb_dim)
        x = torch.cat([numeric, c_emb, t_emb], dim=-1)
        x = self.component_proj(x)            # (B, N, dim_model)

        for block in self.encoder:
            x = block(x, key_padding_mask=component_mask)

        # masked mean pool — the ONLY pooling weighted by mask, never by share_pct
        mask_f = component_mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = mask_f.sum(dim=1).clamp(min=1e-8)
        pool_mean = (x * mask_f).sum(dim=1) / denom        # (B, dim_model)
        pool_pma = self.pma(x, key_padding_mask=component_mask)  # (B, dim_model)
        comp_pool = torch.cat([pool_mean, pool_pma], dim=-1)     # (B, 2*dim_model)

        # -- scenario path
        mode_idx = scenario_features[:, 0].long()
        other = scenario_features[:, 1:]
        m_emb = self.mode_emb(mode_idx)
        scen_emb = self.scenario_mlp(torch.cat([other, m_emb], dim=-1))  # (B, dim_model)

        # -- fusion
        fused = torch.cat([comp_pool, scen_emb], dim=-1)  # (B, 3*dim_model)
        fused = self.fusion_mlp(fused)                     # (B, fusion_hidden)

        return {
            "reg_out": self.regression_head(fused),
            "sign_logit": self.sign_head(fused),
            "component_attention": self.pma.last_attention_weights,
        }
