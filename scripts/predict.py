#!/usr/bin/env python3
"""Ensemble predict over all trained (fold, seed) checkpoints.

Example::

    python scripts/predict.py \\
        --checkpoints-dir artifacts \\
        --output predictions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    load_raw,
)
from src.training.trainer import DaimlerLightningModule  # noqa: E402
from src.utils.submission import DKV_COL, EOT_COL, validate_submission  # noqa: E402
from src.utils.transforms import TargetTransformer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble predict over all run directories")
    p.add_argument("--checkpoints-dir", type=str, default="artifacts")
    p.add_argument("--output", type=str, default="predictions.csv")
    p.add_argument("--data-dir", type=str, default="data/raw")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="optional comma-separated floats matching the order of discovered runs",
    )
    return p.parse_args()


def _discover_runs(root: Path) -> list[Path]:
    """Return every ``fold_*_seed_*`` directory under any model folder."""
    runs: list[Path] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(model_dir.glob("fold_*_seed_*")):
            ckpts = list((run_dir / "checkpoints").glob("*.ckpt"))
            if not ckpts:
                continue
            if not (run_dir / "normalizer.pkl").exists():
                continue
            if not (run_dir / "target_transformer.pkl").exists():
                continue
            runs.append(run_dir)
    return runs


@torch.no_grad()
def _predict_run(
    run_dir: Path,
    test_records: list,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    normalizer = FeatureNormalizer.load(run_dir / "normalizer.pkl")
    target_transformer = TargetTransformer.load(run_dir / "target_transformer.pkl")
    raw_target_stats = target_transformer.raw_stats()
    ckpt_path = next((run_dir / "checkpoints").glob("*.ckpt"))

    module = DaimlerLightningModule.load_from_checkpoint(
        str(ckpt_path),
        target_transformer=target_transformer,
        raw_target_stats=raw_target_stats,
        map_location=device,
    )
    module.eval().to(device)

    ds = DaimlerDataset(
        test_records,
        normalizer=normalizer,
        target_transformer=None,
        training=False,
        component_dropout_p=0.0,
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    chunks = []
    for batch in loader:
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        pred = module._forward_batch(batch)
        chunks.append(pred["reg_out"].cpu().numpy())
    pred_z = np.concatenate(chunks, axis=0)
    pred_raw_dict = target_transformer.inverse_transform(
        {"target_dkv": pred_z[:, 0], "target_eot": pred_z[:, 1]}
    )
    return np.stack(
        [pred_raw_dict["target_dkv"], pred_raw_dict["target_eot"]], axis=1
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_df, test_df, props = load_raw(args.data_dir)
    vocab = Vocab().build(train_df, test_df)
    resolver = PropertyResolver.build(props, train_df, test_df, vocab, k=15)
    test_records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=False)
        for sid, grp in test_df.groupby("scenario_id", sort=False)
    ]
    scenario_ids = [r["scenario_id"] for r in test_records]

    root = Path(args.checkpoints_dir)
    runs = _discover_runs(root)
    if not runs:
        raise RuntimeError(f"no runnable checkpoints under {root}")

    print(f"ensembling {len(runs)} runs:")
    for r in runs:
        print(f"  {r.relative_to(root)}")

    preds = []
    for run_dir in runs:
        preds.append(_predict_run(run_dir, test_records, args.batch_size, device))

    preds_arr = np.stack(preds, axis=0)  # (K, n_test, 2)
    if args.ensemble_weights:
        w = np.array([float(x) for x in args.ensemble_weights.split(",")], dtype=np.float64)
        if w.shape[0] != preds_arr.shape[0]:
            raise ValueError(
                f"got {w.shape[0]} weights but {preds_arr.shape[0]} runs"
            )
        w = w / w.sum()
        mean_pred = (preds_arr * w[:, None, None]).sum(axis=0)
    else:
        mean_pred = preds_arr.mean(axis=0)

    submission = pd.DataFrame(
        {
            "scenario_id": scenario_ids,
            DKV_COL: mean_pred[:, 0],
            EOT_COL: mean_pred[:, 1],
        }
    )
    validate_submission(submission, scenario_ids)
    submission.to_csv(args.output, index=False)
    print(f"wrote {args.output}  (runs averaged: {len(runs)})")


if __name__ == "__main__":
    main()
