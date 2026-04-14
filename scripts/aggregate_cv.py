#!/usr/bin/env python3
"""Aggregate every ``metrics.json`` under ``artifacts/`` into a summary table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


METRICS_KEYS = [
    "val/mae_dkv",
    "val/mae_eot",
    "val/mae_mean",
    "val/sign_acc",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts-dir", type=str, default="artifacts")
    p.add_argument("--output", type=str, default="artifacts/cv_summary.csv")
    return p.parse_args()


def _collect(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        with open(metrics_path) as f:
            m = json.load(f)
        final = m.get("final", {})
        row = {
            "model": m.get("model_type", metrics_path.parents[1].name),
            "fold": int(m.get("fold", -1)),
            "seed": int(m.get("seed", -1)),
            "n_params": int(m.get("n_params", 0)),
            "best_val_mae_mean": float(m.get("best_val_mae_mean", float("nan"))),
        }
        for k in METRICS_KEYS:
            row[k.replace("val/", "")] = float(final.get(k, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame, group: str) -> pd.DataFrame:
    agg = (
        df.groupby(group)[[
            "mae_dkv", "mae_eot", "mae_mean", "sign_acc", "best_val_mae_mean",
        ]]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    return agg


def main() -> None:
    args = parse_args()
    root = Path(args.artifacts_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    df = _collect(root)
    if df.empty:
        print("no metrics.json found")
        return

    df = df.sort_values(["model", "fold", "seed"]).reset_index(drop=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} rows)")

    with pd.option_context("display.max_columns", 50, "display.width", 220):
        print("\n=== per-model (mean ± std, n runs) ===")
        print(_summary(df, "model"))
        print("\n=== per-(model, fold) ===")
        print(_summary(df, ["model", "fold"]))
        print("\n=== per-(model, seed) ===")
        print(_summary(df, ["model", "seed"]))
        print("\n=== top-5 runs by val/mae_mean ===")
        print(df.nsmallest(5, "mae_mean").to_string(index=False))
        print("\n=== bottom-5 runs by val/mae_mean ===")
        print(df.nlargest(5, "mae_mean").to_string(index=False))


if __name__ == "__main__":
    main()
