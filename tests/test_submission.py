"""Tests for src.utils.submission.validate_submission."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.submission import (
    DKV_COL,
    EOT_COL,
    EXPECTED_COLUMNS,
    validate_submission,
)


EXPECTED_IDS = ["test_1", "test_2", "test_3"]


def _good(ids=EXPECTED_IDS) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": list(ids),
            DKV_COL: [1.5, -2.0, 3.3][: len(ids)],
            EOT_COL: [10.0, 20.0, 30.0][: len(ids)],
        }
    )


# ---------------------------------------------------------------------------
# positive case
# ---------------------------------------------------------------------------


def test_valid_submission_passes_and_returns_df():
    df = _good()
    out = validate_submission(df, EXPECTED_IDS)
    assert out is df


def test_valid_submission_accepts_any_row_order():
    df = _good(ids=["test_3", "test_1", "test_2"])
    # Permuted rows are legal — scoring joins on scenario_id.
    validate_submission(df, EXPECTED_IDS)


def test_expected_columns_constant_matches_header():
    assert EXPECTED_COLUMNS[0] == "scenario_id"
    assert EXPECTED_COLUMNS[1] == DKV_COL
    assert EXPECTED_COLUMNS[2] == EOT_COL


# ---------------------------------------------------------------------------
# negative cases — the 5 mandated + a couple of extras
# ---------------------------------------------------------------------------


def test_rejects_missing_column():
    df = _good().drop(columns=[EOT_COL])
    with pytest.raises(ValueError, match="missing columns"):
        validate_submission(df, EXPECTED_IDS)


def test_rejects_extra_scenario_id():
    df = pd.DataFrame(
        {
            "scenario_id": ["test_1", "test_2", "test_3", "test_999"],
            DKV_COL: [1.0, 2.0, 3.0, 4.0],
            EOT_COL: [1.0, 2.0, 3.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="unexpected scenario_id"):
        validate_submission(df, EXPECTED_IDS)


def test_rejects_missing_scenario_id():
    df = _good()
    with pytest.raises(ValueError, match="missing scenario_id"):
        validate_submission(df, EXPECTED_IDS + ["test_4"])


def test_rejects_duplicate_scenario_id():
    df = pd.DataFrame(
        {
            "scenario_id": ["test_1", "test_2", "test_2"],
            DKV_COL: [1.0, 2.0, 3.0],
            EOT_COL: [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        validate_submission(df, ["test_1", "test_2"])


def test_rejects_nan_in_dkv_target():
    df = _good()
    df.loc[1, DKV_COL] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_submission(df, EXPECTED_IDS)


# extra coverage beyond the mandated 5 — cheap to add, catches real bugs

def test_rejects_inf_in_eot_target():
    df = _good()
    df.loc[0, EOT_COL] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_submission(df, EXPECTED_IDS)


def test_rejects_wrong_column_order():
    df = _good()[["scenario_id", EOT_COL, DKV_COL]]
    with pytest.raises(ValueError, match="wrong column order"):
        validate_submission(df, EXPECTED_IDS)


def test_rejects_non_dataframe_input():
    with pytest.raises(ValueError, match="DataFrame"):
        validate_submission({"scenario_id": ["test_1"]}, EXPECTED_IDS)  # type: ignore[arg-type]
