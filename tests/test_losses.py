"""Tests for DaimlerLoss."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src.training.losses import DaimlerLoss


def test_components_at_zero():
    loss_fn = DaimlerLoss(huber_delta=1.0, sign_weight=0.2)
    pred_reg = torch.zeros(4, 2)
    target_reg = torch.zeros(4, 2)
    pred_sign = torch.zeros(4, 1)
    target_sign = torch.ones(4)  # all positive

    out = loss_fn(pred_reg, pred_sign, target_reg, target_sign)

    assert out["reg_loss"].item() == pytest.approx(0.0)
    # BCEWithLogits(0, 1) = log(1 + e^0) = log(2)
    assert out["sign_loss"].item() == pytest.approx(math.log(2.0), rel=1e-5)
    assert out["total"].item() == pytest.approx(0.2 * math.log(2.0), rel=1e-5)


def test_reg_loss_quadratic_zone():
    loss_fn = DaimlerLoss(huber_delta=1.0, sign_weight=0.0)
    pred_reg = torch.zeros(4, 2)
    target_reg = torch.full((4, 2), 0.5)
    # SmoothL1 with beta=1: 0.5 * |d|^2 / beta = 0.5 * 0.25 = 0.125
    out = loss_fn(pred_reg, torch.zeros(4, 1), target_reg, torch.zeros(4))
    assert out["reg_loss"].item() == pytest.approx(0.125, rel=1e-5)


def test_reg_loss_linear_zone():
    loss_fn = DaimlerLoss(huber_delta=1.0, sign_weight=0.0)
    pred_reg = torch.zeros(4, 2)
    target_reg = torch.full((4, 2), 3.0)
    # SmoothL1 with beta=1 for |d|=3: |d| - 0.5 * beta = 3.0 - 0.5 = 2.5
    out = loss_fn(pred_reg, torch.zeros(4, 1), target_reg, torch.zeros(4))
    assert out["reg_loss"].item() == pytest.approx(2.5, rel=1e-5)


def test_sign_weight_scales_total():
    common = dict(
        pred_reg=torch.zeros(2, 2),
        pred_sign_logit=torch.zeros(2, 1),
        target_reg_z=torch.zeros(2, 2),
        target_sign=torch.ones(2),
    )
    low = DaimlerLoss(sign_weight=0.0)(**common)["total"].item()
    mid = DaimlerLoss(sign_weight=0.5)(**common)["total"].item()
    high = DaimlerLoss(sign_weight=1.0)(**common)["total"].item()
    assert low == pytest.approx(0.0)
    assert low < mid < high


def test_sign_loss_accepts_int_target():
    loss_fn = DaimlerLoss()
    pred_reg = torch.zeros(3, 2)
    pred_sign = torch.zeros(3, 1)
    target_reg = torch.zeros(3, 2)
    target_sign = torch.tensor([0, 1, 1], dtype=torch.int64)
    out = loss_fn(pred_reg, pred_sign, target_reg, target_sign)
    assert torch.isfinite(out["total"])
