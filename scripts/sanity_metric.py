#!/usr/bin/env python3
"""Sanity-check the ``val/mae`` metric against trivial baselines and manual math.

Runs on the ``set_transformer fold=0 seed=42`` checkpoint. Three blocks:

1. Dumb baselines (mean / median / per-mode-mean) on val_part of fold 0 →
   shows how much signal the trained model actually captures.
2. Hand-computed z-errors on every val record → must match the
   Lightning-reported ``val/mae_dkv`` / ``val/mae_eot`` within epsilon.
3. Leakage check: the pickled ``TargetTransformer`` must contain
   train_part stats, NOT full-train stats.

Prerequisite::

    python scripts/train.py --config configs/set_transformer.yaml \\
        --fold 0 --seed 42 --output-dir artifacts/set_transformer
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data import (  # noqa: E402
    DaimlerDataset,
    FeatureNormalizer,
    PropertyResolver,
    Vocab,
    build_scenario_record,
    collate_fn,
    load_raw,
)
from src.training.cv import scenario_based_kfold  # noqa: E402
from src.training.trainer import DaimlerLightningModule  # noqa: E402
from src.utils.metrics import normalized_mae  # noqa: E402
from src.utils.transforms import TargetTransformer  # noqa: E402


RUN_DIR = ROOT / "artifacts" / "set_transformer" / "fold_0_seed_42"


def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------


def main() -> None:
    if not (RUN_DIR / "checkpoints").exists():
        raise FileNotFoundError(
            f"expected run dir {RUN_DIR}. run:\n"
            "  python scripts/train.py --config configs/set_transformer.yaml "
            "--fold 0 --seed 42 --output-dir artifacts/set_transformer"
        )

    # ------- 0. rebuild data pipeline exactly like train.py ----------------
    train_df, test_df, props = load_raw(ROOT / "data" / "raw")
    vocab = Vocab().build(train_df, test_df)
    resolver = PropertyResolver.build(props, train_df, test_df, vocab, k=15)
    train_records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=True)
        for sid, grp in train_df.groupby("scenario_id", sort=False)
    ]

    folds = scenario_based_kfold(
        train_records, n_splits=5, seed=42, rare_mode_threshold=6
    )
    tr_idx, va_idx = folds[0]
    tr_recs = [train_records[i] for i in tr_idx]
    va_recs = [train_records[i] for i in va_idx]

    tr_dkv = np.array([float(r["targets"][0]) for r in tr_recs])
    tr_eot = np.array([float(r["targets"][1]) for r in tr_recs])
    va_dkv = np.array([float(r["targets"][0]) for r in va_recs])
    va_eot = np.array([float(r["targets"][1]) for r in va_recs])

    raw_stats_manual = {
        "target_dkv": {
            "mean": float(tr_dkv.mean()),
            "std": float(tr_dkv.std(ddof=0)),
        },
        "target_eot": {
            "mean": float(tr_eot.mean()),
            "std": float(tr_eot.std(ddof=0)),
        },
    }
    print(f"train_part: {len(tr_recs)} records   val_part: {len(va_recs)} records")
    print("raw_target_stats from train_part (what normalized_mae uses):")
    for col, s in raw_stats_manual.items():
        print(f"  {col}: mean={s['mean']:+.4f}  std={s['std']:.4f}")

    # ======================================================================
    _hr("BLOCK 1 — trivial baselines on val_part")
    # ======================================================================

    y_true = {"target_dkv": va_dkv, "target_eot": va_eot}

    # a) global mean
    mean_dkv = float(tr_dkv.mean())
    mean_eot = float(tr_eot.mean())
    y_mean = {
        "target_dkv": np.full(len(va_recs), mean_dkv),
        "target_eot": np.full(len(va_recs), mean_eot),
    }
    mae_a = normalized_mae(y_true, y_mean, raw_stats_manual)

    # b) global median
    median_dkv = float(np.median(tr_dkv))
    median_eot = float(np.median(tr_eot))
    y_med = {
        "target_dkv": np.full(len(va_recs), median_dkv),
        "target_eot": np.full(len(va_recs), median_eot),
    }
    mae_b = normalized_mae(y_true, y_med, raw_stats_manual)

    # c) per-mode mean
    def mode_of(rec) -> int:
        return int(rec["scenario_features"][0])

    groups: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"dkv": [], "eot": []}
    )
    for r in tr_recs:
        mid = mode_of(r)
        groups[mid]["dkv"].append(float(r["targets"][0]))
        groups[mid]["eot"].append(float(r["targets"][1]))
    mode_mean: dict[int, tuple[float, float]] = {}
    for mid, vals in groups.items():
        if len(vals["dkv"]) >= 2:  # rare-mode guard
            mode_mean[mid] = (
                float(np.mean(vals["dkv"])),
                float(np.mean(vals["eot"])),
            )

    pred_dkv_m: list[float] = []
    pred_eot_m: list[float] = []
    fallback = 0
    for r in va_recs:
        mid = mode_of(r)
        if mid in mode_mean:
            d, e = mode_mean[mid]
        else:
            d, e = mean_dkv, mean_eot
            fallback += 1
        pred_dkv_m.append(d)
        pred_eot_m.append(e)
    y_perm = {
        "target_dkv": np.asarray(pred_dkv_m),
        "target_eot": np.asarray(pred_eot_m),
    }
    mae_c = normalized_mae(y_true, y_perm, raw_stats_manual)
    print(f"per-mode fallback to global mean for {fallback}/{len(va_recs)} val scenarios")

    # smoke baseline ("our" SetTransformer) — we need the BEST-epoch metrics,
    # not the last-epoch ones. ``metrics.json`` only stores last epoch, so
    # parse the CSV logger output and pick the row with min val/mae_mean.
    import pandas as pd

    csv_path = RUN_DIR / "logs" / "version_0" / "metrics.csv"
    csv = pd.read_csv(csv_path)
    val_rows = csv[csv["val/mae_mean"].notna()]
    best_row = val_rows.loc[val_rows["val/mae_mean"].idxmin()]
    our_dkv = float(best_row["val/mae_dkv"])
    our_eot = float(best_row["val/mae_eot"])
    our_mean = float(best_row["val/mae_mean"])

    with open(RUN_DIR / "metrics.json") as f:
        run_metrics = json.load(f)
    our_best = float(run_metrics["best_val_mae_mean"])
    print(
        f"\n  best epoch (from metrics.csv, epoch={int(best_row.get('epoch', -1))}): "
        f"dkv={our_dkv:.4f} eot={our_eot:.4f} mean={our_mean:.4f}"
    )

    print(
        f"\n  {'baseline':<22s} | {'mae_dkv':>9s} | {'mae_eot':>9s} | {'mae_mean':>9s}"
    )
    print(f"  {'-'*22}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
    print(f"  {'mean predictor':<22s} | {mae_a[0]:>9.4f} | {mae_a[1]:>9.4f} | {mae_a[2]:>9.4f}")
    print(f"  {'median predictor':<22s} | {mae_b[0]:>9.4f} | {mae_b[1]:>9.4f} | {mae_b[2]:>9.4f}")
    print(f"  {'per-mode mean':<22s} | {mae_c[0]:>9.4f} | {mae_c[1]:>9.4f} | {mae_c[2]:>9.4f}")
    print(f"  {'our SetTransformer':<22s} | {our_dkv:>9.4f} | {our_eot:>9.4f} | {our_mean:>9.4f}")

    best_dkv = min(mae_a[0], mae_b[0], mae_c[0])
    best_eot = min(mae_a[1], mae_b[1], mae_c[1])
    best_mean = min(mae_a[2], mae_b[2], mae_c[2])
    print("\n  improvement over best trivial baseline:")
    print(f"    dkv :  {best_dkv:.4f} → {our_dkv:.4f}   ({best_dkv/max(our_dkv,1e-8):.2f}x)")
    print(f"    eot :  {best_eot:.4f} → {our_eot:.4f}   ({best_eot/max(our_eot,1e-8):.2f}x)")
    print(f"    mean:  {best_mean:.4f} → {our_mean:.4f}   ({best_mean/max(our_mean,1e-8):.2f}x)")
    print(f"  best_val_mae_mean (ModelCheckpoint): {our_best:.4f}")

    # ======================================================================
    _hr("BLOCK 2 — manual metric verification (val_part, first 5 + mean)")
    # ======================================================================

    normalizer = FeatureNormalizer.load(RUN_DIR / "normalizer.pkl")
    tt = TargetTransformer.load(RUN_DIR / "target_transformer.pkl")
    raw_stats_from_tt = tt.raw_stats()
    ckpt_path = next((RUN_DIR / "checkpoints").glob("*.ckpt"))

    module = DaimlerLightningModule.load_from_checkpoint(
        str(ckpt_path),
        target_transformer=tt,
        raw_target_stats=raw_stats_from_tt,
        map_location="cpu",
    )
    module.eval()

    ds = DaimlerDataset(
        va_recs,
        normalizer=normalizer,
        target_transformer=tt,
        training=False,
        component_dropout_p=0.0,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    pred_z_parts: list[np.ndarray] = []
    target_raw_parts: list[np.ndarray] = []
    sid_parts: list[str] = []
    with torch.no_grad():
        for batch in loader:
            pred = module._forward_batch(batch)
            pred_z_parts.append(pred["reg_out"].cpu().numpy())
            target_raw_parts.append(batch["targets"].cpu().numpy())
            sid_parts.extend(batch["scenario_ids"])
    pred_z = np.concatenate(pred_z_parts, axis=0)           # (N_val, 2)
    target_raw = np.concatenate(target_raw_parts, axis=0)    # (N_val, 2)

    pred_raw_dict = tt.inverse_transform(
        {"target_dkv": pred_z[:, 0], "target_eot": pred_z[:, 1]}
    )
    pred_dkv = pred_raw_dict["target_dkv"]
    pred_eot = pred_raw_dict["target_eot"]

    print(f"\nval_part scenario_ids ({len(sid_parts)} total):")
    print(f"  {sid_parts}")

    print(f"\nfirst 5 predictions (raw space):")
    print(
        f"  {'scenario_id':<10s} | {'pred_dkv':>10s} | {'true_dkv':>10s} |"
        f" {'pred_eot':>10s} | {'true_eot':>10s}"
    )
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(min(5, len(sid_parts))):
        print(
            f"  {sid_parts[i]:<10s} | {pred_dkv[i]:>10.4f} | {target_raw[i,0]:>10.4f} |"
            f" {pred_eot[i]:>10.4f} | {target_raw[i,1]:>10.4f}"
        )

    std_dkv = raw_stats_from_tt["target_dkv"]["std"]
    std_eot = raw_stats_from_tt["target_eot"]["std"]
    z_err_dkv = np.abs(pred_dkv - target_raw[:, 0]) / std_dkv
    z_err_eot = np.abs(pred_eot - target_raw[:, 1]) / std_eot

    print(f"\nfirst 5 z-errors:")
    print(f"  {'scenario_id':<10s} | {'z_err_dkv':>10s} | {'z_err_eot':>10s}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(min(5, len(sid_parts))):
        print(f"  {sid_parts[i]:<10s} | {z_err_dkv[i]:>10.4f} | {z_err_eot[i]:>10.4f}")

    manual_mae_dkv = float(z_err_dkv.mean())
    manual_mae_eot = float(z_err_eot.mean())
    manual_mae_mean = (manual_mae_dkv + manual_mae_eot) / 2
    print(f"\nmanual (mean over all {len(sid_parts)} val):")
    print(f"  mae_dkv  = {manual_mae_dkv:.6f}")
    print(f"  mae_eot  = {manual_mae_eot:.6f}")
    print(f"  mae_mean = {manual_mae_mean:.6f}")
    print(f"Lightning reported (metrics.json 'final'):")
    print(f"  mae_dkv  = {our_dkv:.6f}")
    print(f"  mae_eot  = {our_eot:.6f}")
    print(f"  mae_mean = {our_mean:.6f}")

    eps = 1e-3
    dkv_match = abs(manual_mae_dkv - our_dkv) < eps
    eot_match = abs(manual_mae_eot - our_eot) < eps
    print(f"\ndelta (manual - lightning):")
    print(f"  dkv:  {manual_mae_dkv - our_dkv:+.6e}  within {eps}? {dkv_match}")
    print(f"  eot:  {manual_mae_eot - our_eot:+.6e}  within {eps}? {eot_match}")

    # ======================================================================
    _hr("BLOCK 3 — leakage check on TargetTransformer")
    # ======================================================================

    full_dkv = np.array([float(r["targets"][0]) for r in train_records])
    full_eot = np.array([float(r["targets"][1]) for r in train_records])

    print("\nTargetTransformer stored raw stats (from pickle):")
    for col in ("target_dkv", "target_eot"):
        print(
            f"  {col}: mean={tt.raw_mean_[col]:+.6f}  std={tt.raw_std_[col]:.6f}"
        )

    print(f"\nComputed from train_part of fold 0 ({len(tr_recs)} records):")
    tp_stats = {
        "target_dkv": {"mean": float(tr_dkv.mean()), "std": float(tr_dkv.std(ddof=0))},
        "target_eot": {"mean": float(tr_eot.mean()), "std": float(tr_eot.std(ddof=0))},
    }
    for col, s in tp_stats.items():
        print(f"  {col}: mean={s['mean']:+.6f}  std={s['std']:.6f}")

    print(f"\nComputed from FULL train ({len(train_records)} records):")
    ft_stats = {
        "target_dkv": {"mean": float(full_dkv.mean()), "std": float(full_dkv.std(ddof=0))},
        "target_eot": {"mean": float(full_eot.mean()), "std": float(full_eot.std(ddof=0))},
    }
    for col, s in ft_stats.items():
        print(f"  {col}: mean={s['mean']:+.6f}  std={s['std']:.6f}")

    leak_eps = 1e-6

    def _close(a: float, b: float) -> bool:
        return abs(a - b) < leak_eps

    matches_tp = all(
        _close(tt.raw_mean_[c], tp_stats[c]["mean"])
        and _close(tt.raw_std_[c], tp_stats[c]["std"])
        for c in ("target_dkv", "target_eot")
    )
    matches_ft = all(
        _close(tt.raw_mean_[c], ft_stats[c]["mean"])
        and _close(tt.raw_std_[c], ft_stats[c]["std"])
        for c in ("target_dkv", "target_eot")
    )
    print(f"\nstats match train_part (expected): {matches_tp}")
    print(f"stats match full train (bad!):     {matches_ft}")

    # ======================================================================
    _hr("VERDICT")
    # ======================================================================

    metric_ok = dkv_match and eot_match
    leakage_ok = matches_tp and not matches_ft
    beats_dkv = our_dkv < best_dkv
    beats_mean = our_mean < best_mean

    def _badge(ok: bool) -> str:
        return "✅" if ok else "❌"

    print()
    print(f"  {_badge(metric_ok)} manual z-error mean matches Lightning "
          f"val/mae_dkv and val/mae_eot (within {eps})")
    print(f"  {_badge(leakage_ok)} TargetTransformer fit on train_part only "
          f"(no leakage into full train stats)")
    print(f"  {_badge(beats_dkv)} SetTransformer beats best trivial baseline "
          f"on val/mae_dkv ({best_dkv:.4f} → {our_dkv:.4f})")
    print(f"  {_badge(beats_mean)} SetTransformer beats best trivial baseline "
          f"on val/mae_mean ({best_mean:.4f} → {our_mean:.4f})")

    print()
    if metric_ok and leakage_ok and beats_dkv and beats_mean:
        print("  ✅ Метрика считается корректно. val/mae_dkv — реальный сигнал.")
    else:
        print("  ❌ Найдены проблемы (см. ❌ выше).")


if __name__ == "__main__":
    main()
