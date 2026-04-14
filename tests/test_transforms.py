"""Tests for src.utils.transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.transforms import (
    TargetTransformer,
    inverse_log1p,
    inverse_signed_log1p,
    log1p,
    signed_log1p,
)


# ---------------------------------------------------------------------------
# elementwise transforms
# ---------------------------------------------------------------------------


def test_signed_log1p_zero_is_zero():
    assert float(signed_log1p(0.0)) == 0.0
    assert float(inverse_signed_log1p(0.0)) == 0.0


def test_signed_log1p_sign_preserved():
    x = np.array([-42.0, -1.0, 1.0, 42.0])
    y = signed_log1p(x)
    assert np.all(np.sign(y) == np.sign(x))


def test_signed_log1p_roundtrip_numpy():
    x = np.array([-1000.0, -5.5, -0.1, 0.0, 0.1, 5.5, 1000.0])
    back = inverse_signed_log1p(signed_log1p(x))
    np.testing.assert_allclose(back, x, rtol=1e-10, atol=1e-10)


def test_log1p_roundtrip_numpy():
    x = np.array([0.0, 0.01, 1.0, 100.0, 1e6])
    np.testing.assert_allclose(inverse_log1p(log1p(x)), x, rtol=1e-10)


def test_signed_log1p_monotonic():
    x = np.linspace(-100, 100, 201)
    y = signed_log1p(x)
    assert np.all(np.diff(y) > 0)


def test_signed_log1p_torch_roundtrip():
    torch = pytest.importorskip("torch")
    x = torch.tensor([-1000.0, -0.5, 0.0, 0.5, 1000.0], dtype=torch.float64)
    y = signed_log1p(x)
    assert isinstance(y, torch.Tensor)
    back = inverse_signed_log1p(y)
    assert torch.allclose(back, x, atol=1e-10)


def test_log1p_torch_roundtrip():
    torch = pytest.importorskip("torch")
    x = torch.tensor([0.0, 1.0, 100.0], dtype=torch.float64)
    back = inverse_log1p(log1p(x))
    assert torch.allclose(back, x, atol=1e-10)


# ---------------------------------------------------------------------------
# TargetTransformer
# ---------------------------------------------------------------------------


def _make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "target_dkv": rng.normal(0.0, 30.0, size=n),
            "target_eot": np.abs(rng.normal(50.0, 40.0, size=n)) + 1.0,
        }
    )


def test_transformer_fit_makes_stats_standardized():
    df = _make_df()
    tt = TargetTransformer().fit(df)
    z = tt.transform(df)
    assert set(z.keys()) == {"target_dkv", "target_eot"}
    for col, arr in z.items():
        assert abs(float(arr.mean())) < 1e-9, col
        assert abs(float(arr.std()) - 1.0) < 1e-9, col


def test_transformer_inverse_is_exact_roundtrip():
    df = _make_df()
    tt = TargetTransformer().fit(df)
    z = tt.transform(df)
    back = tt.inverse_transform(z)
    for col in ("target_dkv", "target_eot"):
        np.testing.assert_allclose(
            back[col], df[col].to_numpy(), rtol=1e-10, atol=1e-10
        )


def test_transformer_raw_stats_match_pandas():
    df = _make_df()
    tt = TargetTransformer().fit(df)
    stats = tt.raw_stats()
    for col in ("target_dkv", "target_eot"):
        assert stats[col]["mean"] == pytest.approx(float(df[col].mean()))
        assert stats[col]["std"] == pytest.approx(float(df[col].std(ddof=0)))


def test_transformer_save_load_identity(tmp_path):
    df = _make_df(n=50)
    tt = TargetTransformer().fit(df)
    p = tmp_path / "tt.pkl"
    tt.save(p)
    tt2 = TargetTransformer.load(p)
    assert tt2.raw_mean_ == tt.raw_mean_
    assert tt2.raw_std_ == tt.raw_std_
    assert tt2.config == tt.config
    z1 = tt.transform(df)
    z2 = tt2.transform(df)
    for col in z1:
        np.testing.assert_allclose(z1[col], z2[col])


def test_transformer_rejects_unfit_transform():
    tt = TargetTransformer()
    with pytest.raises(RuntimeError):
        tt.transform(pd.DataFrame({"target_dkv": [1.0], "target_eot": [1.0]}))


def test_transformer_rejects_nan_in_fit():
    df = pd.DataFrame({"target_dkv": [1.0, np.nan], "target_eot": [1.0, 2.0]})
    with pytest.raises(ValueError, match="non-finite"):
        TargetTransformer().fit(df)


def test_transformer_rejects_missing_column():
    df = pd.DataFrame({"target_dkv": [1.0, 2.0]})
    with pytest.raises(KeyError):
        TargetTransformer().fit(df)
