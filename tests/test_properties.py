"""Tests for src.data.properties: parser + 3-level resolver fallback."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
import numpy as np

from src.data.properties import PropertyResolver, parse_property_value
from src.data.vocab import Vocab


# --- parse_property_value -------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("13.74", 13.74),
        ("13,74", 13.74),
        ("-1.5", -1.5),
        ("10-20", 15.0),
        ("10–20", 15.0),  # en-dash
        ("<0.5", 0.5),
        (">100", 100.0),
        ("≤5", 5.0),
        ("", float("nan")),
        (None, float("nan")),
        ("абра-кадабра", float("nan")),
    ],
)
def test_parse_property_value(raw, expected):
    got = parse_property_value(raw)
    if isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


# --- PropertyResolver.resolve fallback ------------------------------------

def _build_tiny_resolver():
    """A 2-component × 2-batch × 2-property universe we can reason about exactly."""
    props = pd.DataFrame(
        [
            # A_1 has a measured value for prop_X at batch '32' only
            {"component_id": "A_1", "batch_id": "32", "property_name": "prop_X", "unit": "u", "value_raw": "5.0"},
            # A_1 has a typical value for prop_X AND prop_Y
            {"component_id": "A_1", "batch_id": "typical", "property_name": "prop_X", "unit": "u", "value_raw": "1.0"},
            {"component_id": "A_1", "batch_id": "typical", "property_name": "prop_Y", "unit": "u", "value_raw": "7.0"},
            # B_1 has only a measured value for prop_X
            {"component_id": "B_1", "batch_id": "55", "property_name": "prop_X", "unit": "u", "value_raw": "3.0"},
        ]
    )
    train = pd.DataFrame(
        {
            "scenario_id": ["train_1"] * 2,
            "component_id": ["A_1", "B_1"],
            "batch_id": ["32", "55"],
            "share_pct": [60.0, 40.0],
            "T_C": [160, 160],
            "t_h": [168, 168],
            "biofuel_pct": [0, 0],
            "catalyst_cat": [1, 1],
            "target_dkv": [10.0, 10.0],
            "target_eot": [50.0, 50.0],
        }
    )
    test = pd.DataFrame(
        {
            "scenario_id": ["test_1"],
            "component_id": ["A_1"],
            "batch_id": ["999"],  # new batch for A_1 — typical fallback
            "share_pct": [100.0],
            "T_C": [160],
            "t_h": [168],
            "biofuel_pct": [0],
            "catalyst_cat": [1],
        }
    )
    vocab = Vocab().build(train, test)
    resolver = PropertyResolver.build(props, train, test, vocab, k=2)
    return resolver


def test_resolver_level_1_measured_wins_over_typical():
    r = _build_tiny_resolver()
    # A_1 at batch '32' should return measured 5.0 for prop_X, not typical 1.0
    out = r.resolve("A_1", "32")
    assert out["prop_X"] == 5.0
    assert out["prop_X__is_measured"] == 1.0


def test_resolver_level_2_typical_fallback():
    r = _build_tiny_resolver()
    # A_1 at batch '999' has no measured → fall back to typical 1.0
    out = r.resolve("A_1", "999")
    assert out["prop_X"] == 1.0
    assert out["prop_X__is_measured"] == 0.0


def test_resolver_level_3_nan_when_nothing_found():
    r = _build_tiny_resolver()
    # B_1 at batch '77' has neither measured nor typical
    out = r.resolve("B_1", "77")
    assert np.isnan(out["prop_X"])
    assert out["prop_X__is_measured"] == 0.0


def test_resolver_class_median_imputes_nan():
    r = _build_tiny_resolver()
    out = r.resolve("B_1", "77")
    # inject a class and impute — class 'b' has median(prop_X) computed on train
    r.impute_class_median("B_1", out)
    # B_1's class is 'b' (strip trailing _1, lowercase); only B_1 in train → median 3.0
    assert out["prop_X"] == 3.0


def test_resolver_feature_properties_limit():
    r = _build_tiny_resolver()
    # k=2 → both prop_X and prop_Y get picked up (there are exactly 2 props)
    assert set(r.feature_properties) == {"prop_X", "prop_Y"}


def test_resolver_pickle_roundtrip(tmp_path):
    r = _build_tiny_resolver()
    p = tmp_path / "resolver.pkl"
    r.save(p)
    r2 = PropertyResolver.load(p)
    assert r2.feature_properties == r.feature_properties
    out1 = r.resolve("A_1", "32")
    out2 = r2.resolve("A_1", "32")
    assert out1 == out2


def test_applicability_on_tiny_resolver():
    r = _build_tiny_resolver()
    # A_1: measured for prop_X at batch '32' → class 'a' applicable for prop_X
    # A_1 has NO measured prop_Y (only typical)   → class 'a' NOT applicable for prop_Y
    vec_a = r.get_applicability_vector("A_1")
    feat = r.feature_properties
    px = feat.index("prop_X")
    py = feat.index("prop_Y")
    assert vec_a[px] == 1.0
    assert vec_a[py] == 0.0
    # B_1: measured for prop_X at batch '55' → class 'b' applicable for prop_X
    vec_b = r.get_applicability_vector("B_1")
    assert vec_b[px] == 1.0
    assert vec_b[py] == 0.0


def test_applicability_pickle_roundtrip(tmp_path):
    r = _build_tiny_resolver()
    p = tmp_path / "resolver.pkl"
    r.save(p)
    r2 = PropertyResolver.load(p)
    v1 = r.get_applicability_vector("A_1")
    v2 = r2.get_applicability_vector("A_1")
    np.testing.assert_array_equal(v1, v2)
    # class_applicability dict itself survives too
    assert r2.class_applicability == r.class_applicability


def test_applicability_unknown_class_fallback():
    r = _build_tiny_resolver()
    vec = r.get_applicability_vector("Совсем_новый_класс_1")
    assert vec.shape == (len(r.feature_properties),)
    # unknown class → all 1 (let the model decide)
    assert float(vec.sum()) == float(len(r.feature_properties))
