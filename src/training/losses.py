"""Joint loss for the two-head regressor + sign classifier."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class DaimlerLoss(nn.Module):
    """SmoothL1 on z-transformed regression targets + BCE on the sign head.

    Total is a weighted sum::

        total = reg_loss + sign_weight * sign_loss

    Parameters
    ----------
    huber_delta
        ``delta`` passed to :class:`torch.nn.SmoothL1Loss`. The default of
        1.0 gives the "classic" Huber — quadratic below 1, linear above.
    sign_weight
        Weight on the auxiliary sign classification loss. 0 disables the
        auxiliary task entirely.
    """

    def __init__(self, huber_delta: float = 1.0, sign_weight: float = 0.2):
        super().__init__()
        self.reg_loss_fn = nn.SmoothL1Loss(beta=huber_delta)
        self.sign_loss_fn = nn.BCEWithLogitsLoss()
        self.sign_weight = float(sign_weight)

    def forward(
        self,
        pred_reg: Tensor,
        pred_sign_logit: Tensor,
        target_reg_z: Tensor,
        target_sign: Tensor,
    ) -> dict[str, Tensor]:
        reg = self.reg_loss_fn(pred_reg, target_reg_z)
        # sign_head emits (B, 1); target_sign is (B,). Align shapes.
        sign = self.sign_loss_fn(
            pred_sign_logit.view(-1), target_sign.view(-1).to(pred_sign_logit.dtype)
        )
        total = reg + self.sign_weight * sign
        return {
            "total": total,
            "reg_loss": reg.detach(),
            "sign_loss": sign.detach(),
        }
