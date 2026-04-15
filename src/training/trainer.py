"""PyTorch Lightning module wrapping both model types.

The same trainer class handles both ``compositional_mlp`` and
``set_transformer`` — the only difference is which tensors from the
batch are fed to ``self.model`` in the steps.

``target_transformer`` and ``raw_target_stats`` are kept OUT of Lightning's
hparam serialization — they must be passed as kwargs to
``load_from_checkpoint`` at inference time.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models.compositional_mlp import CompositionalMLP
from ..models.set_transformer import SetTransformerModel
from ..utils.metrics import normalized_mae
from ..utils.transforms import TargetTransformer
from .losses import DaimlerLoss


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


class DaimlerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_type: str,
        # vocab sizes
        n_components: int,
        n_types: int,
        n_modes: int,
        # feature dims
        d_comp_numeric: int = 48,
        d_scen_other: int = 39,
        # shared hparams
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-3,
        epochs: int = 80,
        huber_delta: float = 1.0,
        sign_weight: float = 0.2,
        # set-transformer only
        dim_model: int = 64,
        num_heads: int = 4,
        ff_dim: int = 128,
        num_encoder_blocks: int = 2,
        fusion_hidden: int = 128,
        comp_emb_dim: int = 16,
        type_emb_dim: int = 8,
        mode_emb_dim: int = 8,
        # compositional_mlp only
        mlp_hidden: int = 128,
        # non-hparam state
        target_transformer: TargetTransformer | None = None,
        raw_target_stats: dict | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["target_transformer", "raw_target_stats"])

        self.model_type = model_type
        # The dispatch accepts two families:
        # - "compositional_mlp"           → CompositionalMLP
        # - "set_transformer{,_*}"        → SetTransformerModel with config-driven hparams
        # Variants ``set_transformer_wide`` / ``_deep`` / ``_highdrop`` are the same
        # class with different hyperparameters — ``model_type`` is kept distinct so
        # ``aggregate_cv.py`` can tell them apart in the summary CSV.
        if model_type == "compositional_mlp":
            self.model = CompositionalMLP(
                n_modes=n_modes,
                d_scen_other=d_scen_other,
                mode_emb_dim=mode_emb_dim,
                hidden=mlp_hidden,
                dropout=dropout,
            )
        elif model_type == "set_transformer" or model_type.startswith("set_transformer_"):
            self.model = SetTransformerModel(
                n_components=n_components,
                n_types=n_types,
                n_modes=n_modes,
                d_comp_numeric=d_comp_numeric,
                d_scen_other=d_scen_other,
                comp_emb_dim=comp_emb_dim,
                type_emb_dim=type_emb_dim,
                mode_emb_dim=mode_emb_dim,
                dim_model=dim_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_encoder_blocks=num_encoder_blocks,
                fusion_hidden=fusion_hidden,
                dropout=dropout,
            )
        else:
            raise ValueError(f"unknown model_type '{model_type}'")

        self.loss_fn = DaimlerLoss(huber_delta=huber_delta, sign_weight=sign_weight)
        self.target_transformer = target_transformer
        self.raw_target_stats = raw_target_stats
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        # accumulators for validation metrics
        self._val_pred_z: list[np.ndarray] = []
        self._val_target_raw: list[np.ndarray] = []
        self._val_sign_logit: list[np.ndarray] = []
        self._val_sign_target: list[np.ndarray] = []

    # -- forward branching --------------------------------------------------

    def _forward_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.model_type == "compositional_mlp":
            return self.model(batch["scenario_features"])
        # any set_transformer variant consumes the full component path
        return self.model(
            batch["component_features"],
            batch["component_mask"],
            batch["scenario_features"],
        )

    # -- training -----------------------------------------------------------

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred = self._forward_batch(batch)
        losses = self.loss_fn(
            pred["reg_out"],
            pred["sign_logit"],
            batch["targets_transformed"],
            batch["sign_target"],
        )
        bs = pred["reg_out"].size(0)
        self.log("train/total", losses["total"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/reg", losses["reg_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/sign", losses["sign_loss"], on_step=False, on_epoch=True, batch_size=bs)
        return losses["total"]

    # -- validation ---------------------------------------------------------

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred = self._forward_batch(batch)
        losses = self.loss_fn(
            pred["reg_out"],
            pred["sign_logit"],
            batch["targets_transformed"],
            batch["sign_target"],
        )
        bs = pred["reg_out"].size(0)
        self.log("val/total", losses["total"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/reg", losses["reg_loss"], on_step=False, on_epoch=True, batch_size=bs)
        self.log("val/sign", losses["sign_loss"], on_step=False, on_epoch=True, batch_size=bs)

        self._val_pred_z.append(pred["reg_out"].detach().cpu().numpy())
        self._val_target_raw.append(batch["targets"].detach().cpu().numpy())
        self._val_sign_logit.append(pred["sign_logit"].detach().cpu().numpy())
        self._val_sign_target.append(batch["sign_target"].detach().cpu().numpy())
        return losses["total"]

    def on_validation_epoch_end(self) -> None:
        if not self._val_pred_z:
            return
        pred_z = np.concatenate(self._val_pred_z, axis=0)          # (N, 2)
        target_raw = np.concatenate(self._val_target_raw, axis=0)  # (N, 2)
        sign_logit = np.concatenate(self._val_sign_logit, axis=0).reshape(-1)
        sign_target = np.concatenate(self._val_sign_target, axis=0).reshape(-1)

        if self.target_transformer is not None and self.raw_target_stats is not None:
            pred_raw_dict = self.target_transformer.inverse_transform(
                {"target_dkv": pred_z[:, 0], "target_eot": pred_z[:, 1]}
            )
            pred_raw = np.stack(
                [pred_raw_dict["target_dkv"], pred_raw_dict["target_eot"]], axis=1
            )
            mae_dkv, mae_eot, mae_mean = normalized_mae(
                target_raw, pred_raw, self.raw_target_stats
            )
            self.log("val/mae_dkv", float(mae_dkv), prog_bar=True)
            self.log("val/mae_eot", float(mae_eot), prog_bar=True)
            self.log("val/mae_mean", float(mae_mean), prog_bar=True)

        # sign accuracy (threshold 0.5)
        sign_prob = 1.0 / (1.0 + np.exp(-sign_logit))
        sign_pred = (sign_prob > 0.5).astype(np.int64)
        sign_acc = float((sign_pred == sign_target.astype(np.int64)).mean())
        self.log("val/sign_acc", sign_acc, prog_bar=True)

        self._val_pred_z.clear()
        self._val_target_raw.clear()
        self._val_sign_logit.clear()
        self._val_sign_target.clear()

    # -- optim --------------------------------------------------------------

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = CosineAnnealingLR(opt, T_max=self.epochs)
        return [opt], [sched]
