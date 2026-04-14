"""CompositionalMLP — scenario-only baseline.

Consumes ONLY ``scenario_features``. The component path is completely
ignored by design: this model serves as a lower-bound for any more
complex architecture. If a Set-Transformer over components can't beat
a flat MLP over scenario-level aggregates, we have a problem.

``scenario_features`` layout (per :func:`src.data.features.scenario_feature_schema`):
- position 0 : ``mode_id_idx`` (int cast to float, categorical)
- positions 1..39 : flat float vector (one-hots, class histograms,
  KV100/TBN aggregates, etc.) — already z-scored or binary passthrough
  by :class:`FeatureNormalizer` for the numeric positions.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class CompositionalMLP(nn.Module):
    """Scenario-only two-head regressor + sign classifier.

    ~20K parameters. Acts as the architectural null hypothesis.
    """

    def __init__(
        self,
        n_modes: int,
        d_scen_other: int = 39,
        mode_emb_dim: int = 8,
        hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # ``n_modes`` is the count of distinct mode ids we've seen in train.
        # We allocate +1 slots so that unseen mode ids would fall into an
        # unused UNK slot rather than crash (shouldn't happen on this data).
        self.mode_emb = nn.Embedding(n_modes + 1, mode_emb_dim)
        self.scenario_mlp = nn.Sequential(
            nn.Linear(d_scen_other + mode_emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden, 2)
        self.sign_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, scenario_features: Tensor) -> dict[str, Tensor]:
        # scenario_features: (B, D_scen) — col 0 is mode_id_idx (float cast
        # from int), cols 1.. are floats.
        mode_idx = scenario_features[:, 0].long()
        other = scenario_features[:, 1:]
        m_emb = self.mode_emb(mode_idx)
        x = torch.cat([other, m_emb], dim=-1)
        h = self.scenario_mlp(x)
        return {
            "reg_out": self.regression_head(h),
            "sign_logit": self.sign_head(h),
        }
