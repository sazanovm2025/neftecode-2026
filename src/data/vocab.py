"""Vocabulary for categorical features: component_id, type, mode.

A ``Vocab`` holds three dictionaries and is built from (train, test) and
frozen. Unknown keys fall back to an UNK token at index 0 for component_id
and component_type. Modes are stricter — a test mode that isn't present
in train is a hard error, because the whole condition space is enumerable
and a new mode would indicate a data issue.

Persistence: pickle the whole dataclass; reload before inference.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd


UNK: str = "<UNK>"

_TRAILING_NUM_RE = re.compile(r"_\d+$")


def extract_component_type(component_id: str) -> str:
    """Strip trailing ``_<digits>``, lowercase, replace spaces with ``_``.

    Examples
    --------
    >>> extract_component_type("Антиоксидант_5")
    'антиоксидант'
    >>> extract_component_type("Базовое_масло_10")
    'базовое_масло'
    >>> extract_component_type("Соединение_молибдена_3")
    'соединение_молибдена'
    """
    s = str(component_id).strip()
    s = _TRAILING_NUM_RE.sub("", s)
    return s.lower().replace(" ", "_")


Mode = Tuple[float, float, float, int]  # (T, t, biofuel, catalyst)


@dataclass
class Vocab:
    component_id_to_idx: dict[str, int] = field(default_factory=dict)
    component_type_to_idx: dict[str, int] = field(default_factory=dict)
    mode_id_to_idx: dict[Mode, int] = field(default_factory=dict)
    # per-axis category lists (sorted numerically) used for scenario one-hots
    T_cats: list[float] = field(default_factory=list)
    t_cats: list[float] = field(default_factory=list)
    biofuel_cats: list[float] = field(default_factory=list)
    catalyst_cats: list[int] = field(default_factory=list)
    _built: bool = False

    # -- derived sizes ---------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self.component_id_to_idx)

    @property
    def n_types(self) -> int:
        return len(self.component_type_to_idx)

    @property
    def n_modes(self) -> int:
        return len(self.mode_id_to_idx)

    # -- build -----------------------------------------------------------

    def build(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> "Vocab":
        # component_id vocabulary (covers both splits so test has no unknowns)
        all_comps = sorted(
            set(train_df["component_id"].tolist())
            | set(test_df["component_id"].tolist())
        )
        self.component_id_to_idx = {UNK: 0}
        for c in all_comps:
            self.component_id_to_idx[c] = len(self.component_id_to_idx)

        # component_type vocabulary (extracted from component_id)
        all_types = sorted({extract_component_type(c) for c in all_comps})
        self.component_type_to_idx = {UNK: 0}
        for t in all_types:
            self.component_type_to_idx[t] = len(self.component_type_to_idx)

        # per-axis condition categories (sorted numerically)
        self.T_cats = sorted({float(v) for v in train_df["T_C"].unique()})
        self.t_cats = sorted({float(v) for v in train_df["t_h"].unique()})
        self.biofuel_cats = sorted({float(v) for v in train_df["biofuel_pct"].unique()})
        self.catalyst_cats = sorted({int(v) for v in train_df["catalyst_cat"].unique()})

        # modes (4-tuples) — built from train, test must be a subset
        train_modes = self._unique_modes(train_df)
        self.mode_id_to_idx = {m: i for i, m in enumerate(sorted(train_modes))}

        test_modes = self._unique_modes(test_df)
        new_modes = test_modes - set(self.mode_id_to_idx)
        if new_modes:
            raise ValueError(
                f"test contains {len(new_modes)} mode(s) not seen in train: "
                f"{sorted(new_modes)}"
            )

        self._built = True
        return self

    @staticmethod
    def _unique_modes(df: pd.DataFrame) -> set[Mode]:
        sub = df[["T_C", "t_h", "biofuel_pct", "catalyst_cat"]].drop_duplicates()
        return {
            (float(r.T_C), float(r.t_h), float(r.biofuel_pct), int(r.catalyst_cat))
            for r in sub.itertuples(index=False)
        }

    # -- lookup ----------------------------------------------------------

    def component_idx(self, component_id: str) -> int:
        return self.component_id_to_idx.get(str(component_id), 0)

    def type_idx(self, component_id: str) -> int:
        return self.component_type_to_idx.get(extract_component_type(component_id), 0)

    def mode_idx(self, T, t, biofuel, catalyst) -> int:
        key: Mode = (float(T), float(t), float(biofuel), int(catalyst))
        if key not in self.mode_id_to_idx:
            raise KeyError(f"unknown mode {key} (must be a subset of train modes)")
        return self.mode_id_to_idx[key]

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(obj).__name__}")
        return obj
