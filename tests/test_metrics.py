"""Tests for src.utils.metrics.normalized_mae."""
from __future__ import annotations

import numpy as np
import pytest

from src.utils.metrics import TARGET_COLS, normalized_mae


STATS = {
    "target_dkv": {"mean": 0.0, "std": 10.0},
    "target_eot": {"mean": 100.0, "std": 50.0},
}


def test_perfect_prediction_gives_zero():
    y = {
        "target_dkv": np.array([0.0, 5.0, -5.0]),
        "target_eot": np.array([100.0, 150.0, 50.0]),
    }
    mae_dkv, mae_eot, mean = normalized_mae(y, y, STATS)
    assert mae_dkv == 0.0
    assert mae_eot == 0.0
    assert mean == 0.0


def test_known_errors_unit_step():
    # DKV: const error of 10 on std=10 -> normalized |err| = 1.0
    # EOT: const error of 50 on std=50 -> normalized |err| = 1.0
    yt = {
        "target_dkv": np.array([0.0, 0.0]),
        "target_eot": np.array([100.0, 100.0]),
    }
    yp = {
        "target_dkv": np.array([10.0, -10.0]),
        "target_eot": np.array([150.0, 50.0]),
    }
    mae_dkv, mae_eot, mean = normalized_mae(yt, yp, STATS)
    assert mae_dkv == pytest.approx(1.0)
    assert mae_eot == pytest.approx(1.0)
    assert mean == pytest.approx(1.0)


def test_half_unit_errors():
    yt = {"target_dkv": np.array([0.0]), "target_eot": np.array([100.0])}
    yp = {"target_dkv": np.array([5.0]), "target_eot": np.array([125.0])}
    mae_dkv, mae_eot, mean = normalized_mae(yt, yp, STATS)
    assert mae_dkv == pytest.approx(0.5)
    assert mae_eot == pytest.approx(0.5)
    assert mean == pytest.approx(0.5)


def test_mean_is_average_of_two_targets():
    # Asymmetric: dkv perfect, eot off by 100 (= 2 std)
    yt = {"target_dkv": np.array([0.0]), "target_eot": np.array([100.0])}
    yp = {"target_dkv": np.array([0.0]), "target_eot": np.array([200.0])}
    mae_dkv, mae_eot, mean = normalized_mae(yt, yp, STATS)
    assert mae_dkv == 0.0
    assert mae_eot == pytest.approx(2.0)
    assert mean == pytest.approx(1.0)


def test_accepts_2d_array_in_target_cols_order():
    yt = np.array([[0.0, 100.0], [0.0, 100.0]])
    yp = np.array([[10.0, 150.0], [-10.0, 50.0]])
    _, _, mean = normalized_mae(yt, yp, STATS)
    assert mean == pytest.approx(1.0)
    assert TARGET_COLS == ("target_dkv", "target_eot")


def test_shape_mismatch_raises():
    yt = {"target_dkv": np.zeros(3), "target_eot": np.zeros(3)}
    yp = {"target_dkv": np.zeros(2), "target_eot": np.zeros(3)}
    with pytest.raises(ValueError, match="shape mismatch"):
        normalized_mae(yt, yp, STATS)


def test_missing_target_in_stats_raises():
    yt = {"target_dkv": np.zeros(1), "target_eot": np.zeros(1)}
    with pytest.raises(KeyError):
        normalized_mae(yt, yt, {"target_dkv": {"mean": 0.0, "std": 1.0}})


def test_zero_std_in_stats_raises():
    yt = {"target_dkv": np.zeros(1), "target_eot": np.zeros(1)}
    bad_stats = {
        "target_dkv": {"mean": 0.0, "std": 0.0},
        "target_eot": {"mean": 0.0, "std": 1.0},
    }
    with pytest.raises(ValueError, match="std"):
        normalized_mae(yt, yt, bad_stats)


def test_bad_2d_shape_raises():
    with pytest.raises(ValueError, match="2 columns"):
        normalized_mae(np.zeros((3, 3)), np.zeros((3, 3)), STATS)
