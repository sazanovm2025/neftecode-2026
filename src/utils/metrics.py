"""Competition metric: normalized MAE.

The competition scores per-target MAE after subtracting the train-set
mean and dividing by the train-set std of the **raw** targets (so both
cells end up on comparable scales before averaging). Predictions and
targets MUST be supplied in raw space — the transformer inversion should
happen before calling this function.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


TARGET_COLS: tuple[str, str] = ("target_dkv", "target_eot")


def normalized_mae(
    y_true,
    y_pred,
    target_stats: Mapping[str, Mapping[str, float]],
) -> tuple[float, float, float]:
    """Per-target + mean normalized MAE.

    Parameters
    ----------
    y_true, y_pred
        Either a mapping ``{col: 1D-array}`` or a 2D array of shape
        ``(n, 2)`` whose columns are ordered as :data:`TARGET_COLS`.
        Both must be supplied in **raw** (un-transformed) target space.
    target_stats
        ``{col: {"mean": float, "std": float}}`` computed on the raw train
        targets (e.g. :meth:`TargetTransformer.raw_stats`).

    Returns
    -------
    (mae_dkv, mae_eot, mean_mae)
        ``mean_mae`` is the plain average of the two per-target MAEs.
    """
    yt = _to_col_dict(y_true)
    yp = _to_col_dict(y_pred)

    per_target: dict[str, float] = {}
    for col in TARGET_COLS:
        if col not in target_stats:
            raise KeyError(f"target_stats missing '{col}'")
        std = float(target_stats[col]["std"])
        if std <= 0:
            raise ValueError(f"std for '{col}' must be positive, got {std}")
        mean = float(target_stats[col]["mean"])

        if col not in yt or col not in yp:
            raise KeyError(f"'{col}' missing from y_true or y_pred")
        t = np.asarray(yt[col], dtype=np.float64)
        p = np.asarray(yp[col], dtype=np.float64)
        if t.shape != p.shape:
            raise ValueError(
                f"shape mismatch for '{col}': y_true={t.shape} vs y_pred={p.shape}"
            )
        if t.size == 0:
            raise ValueError(f"empty arrays for '{col}'")

        # Normalize both sides identically — shift cancels out, so we only
        # need the scale, but we compute both for explicitness/symmetry.
        per_target[col] = float(
            np.mean(np.abs((t - mean) / std - (p - mean) / std))
        )

    mae_dkv = per_target["target_dkv"]
    mae_eot = per_target["target_eot"]
    return mae_dkv, mae_eot, (mae_dkv + mae_eot) / 2.0


def _to_col_dict(y) -> dict[str, np.ndarray]:
    if isinstance(y, Mapping):
        return {k: np.asarray(v, dtype=np.float64) for k, v in y.items()}
    a = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(
            f"expected 2D array with 2 columns ordered as {TARGET_COLS}, got shape {a.shape}"
        )
    return {TARGET_COLS[0]: a[:, 0], TARGET_COLS[1]: a[:, 1]}
