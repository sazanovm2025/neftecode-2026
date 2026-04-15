#!/usr/bin/env python3
"""Train one (fold, seed) of either ``compositional_mlp`` or ``set_transformer``.

Example::

    python scripts/train.py \\
        --config configs/set_transformer.yaml \\
        --fold 0 --seed 42 \\
        --output-dir artifacts/set_transformer

Output layout::

    artifacts/<model_type>/fold_<fold>_seed_<seed>/
        ├── normalizer.pkl
        ├── target_transformer.pkl
        ├── metrics.json
        ├── checkpoints/best.ckpt
        └── logs/version_0/metrics.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    DaimlerDataset,
    FeatureNormalizer,
    PropertyResolver,
    Vocab,
    build_scenario_record,
    collate_fn,
    component_feature_dim,
    component_numeric_mask,
    load_raw,
    scenario_feature_dim,
    scenario_numeric_mask,
)
from src.training.cv import scenario_based_kfold  # noqa: E402
from src.training.trainer import DaimlerLightningModule  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.transforms import TargetTransformer  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train one (fold, seed) of a model")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="artifacts")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override config 'epochs' — useful for smoke runs")
    p.add_argument("--accelerator", type=str, default="auto")
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.max_epochs is not None:
        cfg["epochs"] = int(args.max_epochs)

    seed_everything(args.seed)
    pl.seed_everything(args.seed, workers=True)

    model_type = str(cfg["model_type"])
    run_dir = Path(args.output_dir) / f"fold_{args.fold}_seed_{args.seed}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    # ---- data ------------------------------------------------------------
    train_df, test_df, props = load_raw(args.data_dir)
    vocab = Vocab().build(train_df, test_df)
    resolver = PropertyResolver.build(props, train_df, test_df, vocab, k=15)

    train_records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=True)
        for sid, grp in train_df.groupby("scenario_id", sort=False)
    ]

    folds = scenario_based_kfold(
        train_records, n_splits=5, seed=args.seed, rare_mode_threshold=6
    )
    if args.fold < 0 or args.fold >= len(folds):
        raise ValueError(f"fold {args.fold} out of range")
    tr_idx, va_idx = folds[args.fold]
    tr_recs = [train_records[i] for i in tr_idx]
    va_recs = [train_records[i] for i in va_idx]
    print(f"[{model_type} fold={args.fold} seed={args.seed}] "
          f"train={len(tr_recs)} val={len(va_recs)}")

    # ---- per-fold normalizer + target transformer (fit on TRAIN only) ---
    normalizer = FeatureNormalizer().fit(
        tr_recs,
        component_numeric_mask(resolver),
        scenario_numeric_mask(vocab, resolver),
    )
    target_transformer = TargetTransformer().fit(
        pd.DataFrame(
            {
                "target_dkv": [float(r["targets"][0]) for r in tr_recs],
                "target_eot": [float(r["targets"][1]) for r in tr_recs],
            }
        )
    )
    raw_target_stats = target_transformer.raw_stats()
    normalizer.save(run_dir / "normalizer.pkl")
    target_transformer.save(run_dir / "target_transformer.pkl")

    # ---- datasets + loaders ----------------------------------------------
    component_dropout_p = float(cfg.get("component_dropout", 0.0))
    if model_type == "compositional_mlp":
        component_dropout_p = 0.0  # no effect anyway, keep it clean
    # any set_transformer variant (base, _wide, _deep, _highdrop, …) keeps
    # its configured component_dropout value.

    tr_ds = DaimlerDataset(
        tr_recs, normalizer=normalizer, target_transformer=target_transformer,
        training=True, component_dropout_p=component_dropout_p,
    )
    va_ds = DaimlerDataset(
        va_recs, normalizer=normalizer, target_transformer=target_transformer,
        training=False, component_dropout_p=0.0,
    )
    tr_loader = DataLoader(
        tr_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    va_loader = DataLoader(
        va_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    # ---- model -----------------------------------------------------------
    d_comp = component_feature_dim(resolver)
    d_scen = scenario_feature_dim(vocab, resolver)
    module = DaimlerLightningModule(
        model_type=model_type,
        n_components=vocab.n_components,
        n_types=vocab.n_types,
        n_modes=vocab.n_modes,
        d_comp_numeric=d_comp - 2,   # minus component_idx + type_idx
        d_scen_other=d_scen - 1,     # minus mode_id_idx
        dropout=float(cfg.get("dropout", 0.2)),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        epochs=int(cfg["epochs"]),
        huber_delta=float(cfg.get("huber_delta", 1.0)),
        sign_weight=float(cfg.get("sign_weight", 0.2)),
        dim_model=int(cfg.get("dim_model", 64)),
        num_heads=int(cfg.get("num_heads", 4)),
        ff_dim=int(cfg.get("ff_dim", 128)),
        # Accept both historical name ``num_encoder_blocks`` (base config)
        # and the newer ``num_sab_blocks`` alias used in diversification configs.
        num_encoder_blocks=int(
            cfg.get("num_sab_blocks", cfg.get("num_encoder_blocks", 2))
        ),
        fusion_hidden=int(cfg.get("fusion_hidden", 128)),
        comp_emb_dim=int(cfg.get("comp_emb_dim", 16)),
        type_emb_dim=int(cfg.get("type_emb_dim", 8)),
        mode_emb_dim=int(cfg.get("mode_emb_dim", 8)),
        mlp_hidden=int(cfg.get("mlp_hidden", 128)),
        target_transformer=target_transformer,
        raw_target_stats=raw_target_stats,
    )

    n_params = sum(p.numel() for p in module.model.parameters())
    print(f"[{model_type}] trainable params: {n_params:,}")

    # ---- callbacks + logger ----------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best",
        monitor="val/mae_mean",
        mode="min",
        save_top_k=1,
        save_weights_only=False,
    )
    early = EarlyStopping(monitor="val/mae_mean", mode="min", patience=15)
    logger = CSVLogger(save_dir=str(run_dir / "logs"), name="")

    trainer = pl.Trainer(
        max_epochs=int(cfg["epochs"]),
        callbacks=[ckpt_cb, early],
        logger=logger,
        accelerator=args.accelerator,
        devices=1,
        gradient_clip_val=1.0,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    trainer.fit(module, tr_loader, va_loader)

    # ---- persist final metrics -------------------------------------------
    best_score = (
        float(ckpt_cb.best_model_score.item())
        if ckpt_cb.best_model_score is not None
        else float("nan")
    )
    final = {
        k: float(v.item()) if hasattr(v, "item") else float(v)
        for k, v in trainer.logged_metrics.items()
    }
    metrics_out = {
        "model_type": model_type,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "best_val_mae_mean": best_score,
        "n_params": int(n_params),
        "final": final,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[{model_type} fold={args.fold} seed={args.seed}] "
          f"best val/mae_mean = {best_score:.4f}")
    print(f"  checkpoint → {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
