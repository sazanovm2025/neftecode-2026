"""Tests for src.training.cv — runs on the real 167 train scenarios."""
from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
import numpy as np

from src.data import (
    PropertyResolver,
    Vocab,
    build_scenario_record,
    load_raw,
)
from src.training.cv import (
    assert_mode_coverage,
    get_fold_summary,
    scenario_based_kfold,
)


REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def train_records():
    if not (REPO / "data" / "raw" / "daimler_mixtures_train.csv").exists():
        pytest.skip("real data not present in data/raw/")
    train, test, props = load_raw(REPO / "data" / "raw")
    vocab = Vocab().build(train, test)
    resolver = PropertyResolver.build(props, train, test, vocab, k=15)
    return [
        build_scenario_record(sid, grp, resolver, vocab, has_targets=True)
        for sid, grp in train.groupby("scenario_id", sort=False)
    ]


def test_five_folds_no_overlap(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    assert len(folds) == 5
    for tr, va in folds:
        tr_ids = {train_records[i]["scenario_id"] for i in tr}
        va_ids = {train_records[i]["scenario_id"] for i in va}
        assert not (tr_ids & va_ids), "train/val scenario overlap"


def test_folds_cover_entire_dataset(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    all_ids = {r["scenario_id"] for r in train_records}
    for tr, va in folds:
        covered = {train_records[i]["scenario_id"] for i in tr} | {
            train_records[i]["scenario_id"] for i in va
        }
        assert covered == all_ids


def test_every_val_scenario_appears_exactly_once_across_folds(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    seen: list[str] = []
    for _, va in folds:
        seen.extend(train_records[i]["scenario_id"] for i in va)
    assert len(seen) == len(train_records)
    assert len(set(seen)) == len(train_records)


def test_mode_coverage_holds_for_every_fold(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    for fi, (tr, _) in enumerate(folds):
        # should not raise; singleton modes are skipped by design
        assert_mode_coverage(tr, train_records, min_coverage=1)


def test_fold_summary_shape_and_val_size(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    df = get_fold_summary(folds, train_records)
    assert set(df.columns) == {"fold", "split", "n_scenarios", "mode_counts"}
    assert len(df) == 10  # 5 folds × 2 splits
    val_sizes = df[df["split"] == "val"]["n_scenarios"].tolist()
    # 167 / 5 ≈ 33; allow moderate slack
    for vs in val_sizes:
        assert 25 <= vs <= 45


def test_mode_6_present_in_every_train_and_val(train_records):
    folds = scenario_based_kfold(train_records, n_splits=5, seed=42)
    df = get_fold_summary(folds, train_records)
    for _, row in df.iterrows():
        assert 6 in row["mode_counts"], (
            f"mode_id=6 missing from {row['split']} of fold {row['fold']}"
        )


def test_rare_mode_threshold_collapses_small_strata(train_records):
    # With threshold=6 on real data: modes 1,2,4,5 (counts 1,1,2,2) collapse.
    folds_strict = scenario_based_kfold(train_records, n_splits=5, seed=42, rare_mode_threshold=6)
    assert len(folds_strict) == 5
    # Smoke: the builder should not raise StratifiedKFold's "not enough members" error
