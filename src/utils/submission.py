"""Submission format validator.

The competition expects a CSV with exactly three columns, in this exact
order and with exact (long Russian) header strings matching the train
file. Anything else — extra column, reordered column, missing id, extra
id, duplicate id, NaN/inf target — is a hard reject.

Use :func:`validate_submission` right before writing the final CSV so we
never submit a silently-broken file.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DKV_COL: str = (
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %"
)
EOT_COL: str = "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm"

EXPECTED_COLUMNS: list[str] = ["scenario_id", DKV_COL, EOT_COL]


def validate_submission(
    df: pd.DataFrame, expected_scenario_ids: Iterable[str]
) -> pd.DataFrame:
    """Validate a submission dataframe in place and return it on success.

    Raises :class:`ValueError` with an actionable message on any failure.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"expected pandas.DataFrame, got {type(df).__name__}")

    _check_schema(df)
    _check_scenario_ids(df, list(expected_scenario_ids))
    _check_targets(df)
    return df


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _check_schema(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    if cols == EXPECTED_COLUMNS:
        return

    missing = [c for c in EXPECTED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in EXPECTED_COLUMNS]
    parts: list[str] = []
    if missing:
        parts.append(f"missing columns: {missing}")
    if extra:
        parts.append(f"unexpected columns: {extra}")
    if not parts:
        # Same set but wrong order.
        parts.append(f"wrong column order: got {cols}, expected {EXPECTED_COLUMNS}")
    raise ValueError("submission schema invalid — " + "; ".join(parts))


def _check_scenario_ids(df: pd.DataFrame, expected: list[str]) -> None:
    sid = df["scenario_id"]

    if sid.isna().any():
        raise ValueError("scenario_id column contains NaN values")

    dupes = sorted(set(sid[sid.duplicated()].tolist()))
    if dupes:
        raise ValueError(f"duplicate scenario_id rows: {dupes}")

    expected_set = set(expected)
    got_set = set(sid.tolist())

    missing_ids = sorted(expected_set - got_set)
    if missing_ids:
        raise ValueError(f"missing scenario_id in submission: {missing_ids}")

    extra_ids = sorted(got_set - expected_set)
    if extra_ids:
        raise ValueError(f"unexpected scenario_id in submission: {extra_ids}")


def _check_targets(df: pd.DataFrame) -> None:
    for col in (DKV_COL, EOT_COL):
        vals = df[col]
        if vals.isna().any():
            n_nan = int(vals.isna().sum())
            raise ValueError(f"column '{col}' has {n_nan} NaN value(s)")
        arr = np.asarray(vals.to_numpy(), dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"column '{col}' has non-finite (inf) value(s)")
