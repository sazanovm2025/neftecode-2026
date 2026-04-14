"""Tests for src.data.vocab: extract_component_type + Vocab build/roundtrip."""
from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from src.data.vocab import UNK, Vocab, extract_component_type


# --- extract_component_type -----------------------------------------------

@pytest.mark.parametrize(
    "component_id, expected",
    [
        ("Антиоксидант_5", "антиоксидант"),
        ("Базовое_масло_10", "базовое_масло"),
        ("Соединение_молибдена_3", "соединение_молибдена"),
        ("Противоизносная_присадка_20", "противоизносная_присадка"),
        ("Детергент_1", "детергент"),
        ("Дисперсант_4", "дисперсант"),
        ("Модификатор_трения_2", "модификатор_трения"),
        ("Антипенная_присадка_1", "антипенная_присадка"),
        ("Депрессор_7", "депрессор"),
        ("  Антиоксидант_12  ", "антиоксидант"),  # whitespace tolerance
    ],
)
def test_extract_component_type_examples(component_id, expected):
    assert extract_component_type(component_id) == expected


def test_extract_component_type_handles_no_trailing_number():
    # pathological input — no trailing number, still lowercased
    assert extract_component_type("Базовое_масло") == "базовое_масло"


# --- Vocab.build + lookup -------------------------------------------------

def _toy_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["train_1"] * 3 + ["train_2"] * 2,
            "component_id": [
                "Базовое_масло_1",
                "Антиоксидант_5",
                "Детергент_2",
                "Базовое_масло_1",
                "Антиоксидант_5",
            ],
            "batch_id": ["32", "32", "б/н", "32", "32"],
            "share_pct": [50.0, 10.0, 5.0, 60.0, 20.0],
            "T_C": [160, 160, 160, 150, 150],
            "t_h": [168, 168, 168, 216, 216],
            "biofuel_pct": [0, 0, 0, 7, 7],
            "catalyst_cat": [1, 1, 1, 2, 2],
            "target_dkv": [10.0, 10.0, 10.0, -5.0, -5.0],
            "target_eot": [80.0, 80.0, 80.0, 95.0, 95.0],
        }
    )


def _toy_test() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scenario_id": ["test_1"] * 2,
            "component_id": ["Базовое_масло_2", "Антиоксидант_5"],
            "batch_id": ["01050.22", "32"],
            "share_pct": [70.0, 15.0],
            "T_C": [160, 160],
            "t_h": [168, 168],
            "biofuel_pct": [0, 0],
            "catalyst_cat": [1, 1],
        }
    )


def test_vocab_build_sizes():
    v = Vocab().build(_toy_train(), _toy_test())
    # 4 unique components: _1, _5, Детергент_2, _2 — plus UNK
    assert v.n_components == 4 + 1
    # 3 unique types: базовое_масло, антиоксидант, детергент — plus UNK
    assert v.n_types == 3 + 1
    # 2 unique modes in train
    assert v.n_modes == 2
    assert v.component_id_to_idx[UNK] == 0
    assert v.component_type_to_idx[UNK] == 0


def test_vocab_unknown_component_falls_back_to_unk():
    v = Vocab().build(_toy_train(), _toy_test())
    assert v.component_idx("НикогдаНеВидел_99") == 0
    assert v.type_idx("НикогдаНеВидел_99") == 0


def test_vocab_mode_lookup():
    v = Vocab().build(_toy_train(), _toy_test())
    idx = v.mode_idx(160.0, 168.0, 0.0, 1)
    assert 0 <= idx < v.n_modes


def test_vocab_rejects_new_test_mode():
    train = _toy_train()
    test = _toy_test()
    test.loc[0, "T_C"] = 999  # unseen in train
    with pytest.raises(ValueError, match="not seen in train"):
        Vocab().build(train, test)


def test_vocab_pickle_roundtrip(tmp_path):
    v = Vocab().build(_toy_train(), _toy_test())
    p = tmp_path / "vocab.pkl"
    v.save(p)
    v2 = Vocab.load(p)
    assert v2.component_id_to_idx == v.component_id_to_idx
    assert v2.component_type_to_idx == v.component_type_to_idx
    assert v2.mode_id_to_idx == v.mode_id_to_idx
    assert v2.T_cats == v.T_cats
    assert v2.n_modes == v.n_modes
