"""Load raw CSVs + rename Russian columns to canonical tech names.

The raw files use long Russian headers. We collapse them to short machine
names (``component_id``, ``batch_id``, ``share_pct``, ``T_C``, …) used
throughout the rest of the pipeline.

The alias map intentionally accepts TWO flavors of each column:
- the exact headers present in the current raw files;
- the shorter aliases from the task spec (e.g. ``"Температура, °C"``).

This keeps the loader robust to minor header changes in future data drops
from the organizers — any column that doesn't match either flavor is
simply left with its original name (not dropped) and will trip the
required-columns check below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# --- column alias maps ----------------------------------------------------

MIX_COLUMN_ALIASES: dict[str, str] = {
    # actual raw-file headers
    "Компонент": "component_id",
    "Наименование партии": "batch_id",
    "Массовая доля, %": "share_pct",
    "Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C": "T_C",
    "Время испытания | - Daimler Oxidation Test (DOT), ч": "t_h",
    "Количество биотоплива | - Daimler Oxidation Test (DOT), % масс": "biofuel_pct",
    "Дозировка катализатора, категория": "catalyst_cat",
    "Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %": "target_dkv",
    "Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm": "target_eot",
    # task-spec short aliases (fallback)
    "Идентификатор компонента": "component_id",
    "Идентификатор партии": "batch_id",
    "Доля компонента, %": "share_pct",
    "Температура, °C": "T_C",
    "Время, ч": "t_h",
    "Доля биотоплива, %": "biofuel_pct",
    "Дозировка катализатора": "catalyst_cat",
}

PROP_COLUMN_ALIASES: dict[str, str] = {
    "Компонент": "component_id",
    "Наименование партии": "batch_id",
    "Наименование показателя": "property_name",
    "Единица измерения_по_партиям": "unit",
    "Значение показателя": "value_raw",
}


MIX_REQUIRED_COLS: list[str] = [
    "scenario_id", "component_id", "batch_id", "share_pct",
    "T_C", "t_h", "biofuel_pct", "catalyst_cat",
]
TARGET_COLS: list[str] = ["target_dkv", "target_eot"]


# --- public entrypoint ----------------------------------------------------

def load_raw(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (train, test, properties) dataframes with canonical columns.

    Parameters
    ----------
    data_dir
        Directory containing ``daimler_mixtures_train.csv``,
        ``daimler_mixtures_test.csv``, ``daimler_component_properties.csv``.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "daimler_mixtures_train.csv", encoding="utf-8-sig")
    test = pd.read_csv(data_dir / "daimler_mixtures_test.csv", encoding="utf-8-sig")
    props = pd.read_csv(data_dir / "daimler_component_properties.csv", encoding="utf-8-sig")

    train = _rename_columns(train, MIX_COLUMN_ALIASES)
    test = _rename_columns(test, MIX_COLUMN_ALIASES)
    props = _rename_columns(props, PROP_COLUMN_ALIASES)

    # batch_id is always a string (values include '32', '13799.21', 'б/н',
    # 'nan', 'typical'). component_id and scenario_id also coerced to strings.
    for df in (train, test, props):
        df["batch_id"] = df["batch_id"].astype(str).str.strip()
        df["component_id"] = df["component_id"].astype(str).str.strip()
    for df in (train, test):
        df["scenario_id"] = df["scenario_id"].astype(str).str.strip()

    _validate_mixtures(train, kind="train")
    _validate_mixtures(test, kind="test")
    return train, test, props


# --- internals ------------------------------------------------------------

def _rename_columns(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    renames = {k: v for k, v in alias_map.items() if k in df.columns}
    return df.rename(columns=renames)


def _validate_mixtures(df: pd.DataFrame, kind: str) -> None:
    missing = [c for c in MIX_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{kind}: missing required columns {missing}")

    for c in MIX_REQUIRED_COLS:
        if df[c].isna().any():
            raise ValueError(f"{kind}: NaN values in key column '{c}'")

    if kind == "train":
        for c in TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"train: missing target column '{c}'")
            if df[c].isna().any():
                raise ValueError(f"train: NaN in target column '{c}'")

    prefix = kind + "_"
    bad = df[~df["scenario_id"].str.startswith(prefix)]
    if len(bad):
        sample = bad["scenario_id"].iloc[0]
        raise ValueError(
            f"{kind}: scenario_id must start with '{prefix}' — found e.g. {sample!r}"
        )
