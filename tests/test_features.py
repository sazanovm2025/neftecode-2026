"""Tests for src.data.features: scenario/component feature dims + structure."""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
import numpy as np

from src.data.features import (
    build_component_features,
    build_scenario_features,
    build_scenario_record,
    component_feature_dim,
    component_feature_schema,
    component_numeric_mask,
    scenario_feature_dim,
    scenario_feature_schema,
    scenario_numeric_mask,
)
from src.data.loader import load_raw
from src.data.properties import PropertyResolver
from src.data.vocab import Vocab

REPO = Path(__file__).resolve().parent.parent


def _fixtures():
    props = pd.DataFrame(
        [
            {"component_id": "Базовое_масло_1", "batch_id": "32",
             "property_name": "Кинематическая вязкость, при 100°C, ASTM D445",
             "unit": "мм²/с", "value_raw": "16.68"},
            {"component_id": "Антиоксидант_5", "batch_id": "32",
             "property_name": "Щелочное число, ASTM D2896",
             "unit": "мг KOH/г", "value_raw": "25.0"},
            {"component_id": "Базовое_масло_1", "batch_id": "typical",
             "property_name": "Кинематическая вязкость, при 100°C, ASTM D445",
             "unit": "мм²/с", "value_raw": "14.0"},
        ]
    )
    train = pd.DataFrame(
        {
            "scenario_id": ["train_1", "train_1", "train_2", "train_2"],
            "component_id": ["Базовое_масло_1", "Антиоксидант_5", "Базовое_масло_1", "Детергент_2"],
            "batch_id": ["32", "32", "32", "unknown"],
            "share_pct": [800.0, 100.0, 500.0, 150.0],
            "T_C": [160, 160, 150, 150],
            "t_h": [168, 168, 216, 216],
            "biofuel_pct": [0, 0, 7, 7],
            "catalyst_cat": [1, 1, 2, 2],
            "target_dkv": [10.0, 10.0, -5.0, -5.0],
            "target_eot": [80.0, 80.0, 95.0, 95.0],
        }
    )
    test = pd.DataFrame(
        {
            "scenario_id": ["test_1", "test_1"],
            "component_id": ["Базовое_масло_1", "Антиоксидант_5"],
            "batch_id": ["32", "32"],
            "share_pct": [700.0, 120.0],
            "T_C": [160, 160],
            "t_h": [168, 168],
            "biofuel_pct": [0, 0],
            "catalyst_cat": [1, 1],
        }
    )
    vocab = Vocab().build(train, test)
    resolver = PropertyResolver.build(props, train, test, vocab, k=2)
    return train, test, vocab, resolver


def test_schemas_agree_with_dim():
    _, _, vocab, resolver = _fixtures()
    assert sum(w for _, w in component_feature_schema(resolver)) == component_feature_dim(resolver)
    assert sum(w for _, w in scenario_feature_schema(vocab, resolver)) == scenario_feature_dim(vocab, resolver)


def test_component_features_shape():
    train, _, vocab, resolver = _fixtures()
    scen1 = train[train["scenario_id"] == "train_1"]
    X = build_component_features(scen1, resolver, vocab)
    assert X.shape == (len(scen1), component_feature_dim(resolver))
    assert X.dtype == np.float32


def test_scenario_features_shape():
    train, _, vocab, resolver = _fixtures()
    scen1 = train[train["scenario_id"] == "train_1"]
    v = build_scenario_features(scen1, resolver, vocab)
    assert v.shape == (scenario_feature_dim(vocab, resolver),)
    assert v.dtype == np.float32


def test_scenario_record_includes_targets_on_train():
    train, _, vocab, resolver = _fixtures()
    scen1 = train[train["scenario_id"] == "train_1"]
    rec = build_scenario_record("train_1", scen1, resolver, vocab, has_targets=True)
    assert rec["scenario_id"] == "train_1"
    assert rec["component_features"].shape[0] == rec["component_mask"].shape[0]
    assert rec["targets"].shape == (2,)
    assert rec["sign_target"] in (0, 1)


def test_scenario_record_omits_targets_on_test():
    _, test, vocab, resolver = _fixtures()
    scen = test[test["scenario_id"] == "test_1"]
    rec = build_scenario_record("test_1", scen, resolver, vocab, has_targets=False)
    assert rec["targets"] is None
    assert rec["sign_target"] is None


def test_mode_idx_is_first_scenario_feature():
    train, _, vocab, resolver = _fixtures()
    scen1 = train[train["scenario_id"] == "train_1"]
    v = build_scenario_features(scen1, resolver, vocab)
    # first position is mode_id_idx (categorical, kept as float)
    expected = float(vocab.mode_idx(160.0, 168.0, 0.0, 1))
    assert v[0] == expected


def test_numeric_masks_exclude_categorical_positions():
    _, _, vocab, resolver = _fixtures()
    cmask = component_numeric_mask(resolver)
    smask = scenario_numeric_mask(vocab, resolver)
    k = len(resolver.feature_properties)
    # comp positions 0 (component_idx) and 1 (type_idx) are categorical
    assert cmask[0] == False
    assert cmask[1] == False
    assert cmask[2] == True  # share_pct
    assert cmask[3] == True  # rank_in_class
    # batch_known (pos 4) — binary indicator, NOT normalized
    assert cmask[4] == False
    # properties block (pos 5 .. 5+k) — numeric
    for i in range(5, 5 + k):
        assert cmask[i] == True, f"properties[{i - 5}] should be numeric"
    # is_measured block — all False
    for i in range(5 + k, 5 + 2 * k):
        assert cmask[i] == False, f"is_measured[{i - 5 - k}] must not be z-scored"
    # is_applicable block — all False
    for i in range(5 + 2 * k, 5 + 3 * k):
        assert cmask[i] == False, f"is_applicable[{i - 5 - 2 * k}] must not be z-scored"
    # scen position 0 (mode_id_idx) is categorical
    assert smask[0] == False
    assert smask[1] == True  # first T_onehot bit


def test_component_feature_dim_includes_applicability():
    _, _, _, resolver = _fixtures()
    k = len(resolver.feature_properties)
    # 5 (idx/type/share/rank/known) + 3*k (props + is_measured + is_applicable)
    assert component_feature_dim(resolver) == 5 + 3 * k


@pytest.fixture(scope="module")
def real_fixtures():
    """Load the real data pipeline once for applicability tests."""
    if not (REPO / "data" / "raw" / "daimler_mixtures_train.csv").exists():
        pytest.skip("real data not present in data/raw/")
    train, test, props = load_raw(REPO / "data" / "raw")
    vocab = Vocab().build(train, test)
    resolver = PropertyResolver.build(props, train, test, vocab, k=15)
    return train, test, vocab, resolver


def test_applicability_base_oil_has_broad_coverage(real_fixtures):
    _, _, _, resolver = real_fixtures
    vec = resolver.get_applicability_vector("Базовое_масло_7")
    assert len(vec) == 15
    # base oil class should have almost all top-15 properties applicable
    # (only TBN is not measured on base oils)
    assert int(vec.sum()) >= 12, f"base oil applicability too low: {int(vec.sum())}"


def test_applicability_moly_is_mostly_zero(real_fixtures):
    _, _, _, resolver = real_fixtures
    vec = resolver.get_applicability_vector("Соединение_молибдена_3")
    assert len(vec) == 15
    # moly compound class is narrow (~4 components in train) and only a
    # handful of properties are ever *measured* (not typical) on it.
    # Note: the diagnostic coverage matrix shows higher coverage because it
    # counts the measured ∨ typical fallback; applicability is stricter.
    assert int(vec.sum()) <= 3, f"moly applicability too high: {int(vec.sum())}"


def test_applicability_unknown_class_falls_back_to_ones(real_fixtures):
    _, _, _, resolver = real_fixtures
    vec = resolver.get_applicability_vector("Абсолютно_новый_класс_99")
    assert vec.shape == (15,)
    assert float(vec.sum()) == 15.0


def test_d_comp_is_50_on_real_data(real_fixtures):
    _, _, _, resolver = real_fixtures
    assert component_feature_dim(resolver) == 50


def test_batch_known_survives_normalization():
    """batch_known=1/0 values must be preserved verbatim through the normalizer."""
    from src.data import FeatureNormalizer, build_scenario_record

    train, _, vocab, resolver = _fixtures()
    records = [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=True)
        for sid, grp in train.groupby("scenario_id", sort=False)
    ]
    norm = FeatureNormalizer().fit(
        records,
        component_numeric_mask(resolver),
        scenario_numeric_mask(vocab, resolver),
    )
    # batch_known is at component feature position 4
    for rec in records:
        before = rec["component_features"][:, 4].copy()
        transformed = norm.transform_record(rec)
        after = transformed["component_features"][:, 4]
        np.testing.assert_array_equal(before, after)
        # also confirm every value is 0 or 1 (not some z-scored garbage)
        assert set(np.unique(after).tolist()).issubset({0.0, 1.0})


def test_rank_in_class_is_zero_for_singletons():
    _, _, vocab, resolver = _fixtures()
    # scenario with 1 component per class → all ranks == 0
    df = pd.DataFrame(
        {
            "scenario_id": ["train_solo"] * 2,
            "component_id": ["Базовое_масло_1", "Антиоксидант_5"],
            "batch_id": ["32", "32"],
            "share_pct": [700.0, 80.0],
            "T_C": [160, 160],
            "t_h": [168, 168],
            "biofuel_pct": [0, 0],
            "catalyst_cat": [1, 1],
            "target_dkv": [0.0, 0.0],
            "target_eot": [0.0, 0.0],
        }
    )
    X = build_component_features(df, resolver, vocab)
    # rank_in_class is at position 3
    assert float(X[0, 3]) == 0.0
    assert float(X[1, 3]) == 0.0
