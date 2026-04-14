"""Z-score normalizer for component + scenario feature matrices.

Categorical positions (``component_idx``, ``type_idx``, ``mode_id_idx``)
are NOT normalized — their column masks come from
:func:`src.data.features.component_numeric_mask` and
:func:`src.data.features.scenario_numeric_mask`.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class FeatureNormalizer:
    comp_mean: np.ndarray | None = None
    comp_std: np.ndarray | None = None
    comp_numeric: np.ndarray | None = None  # bool mask (D_comp,)
    scen_mean: np.ndarray | None = None
    scen_std: np.ndarray | None = None
    scen_numeric: np.ndarray | None = None  # bool mask (D_scen,)
    fitted_: bool = False

    def fit(
        self,
        records: Iterable[dict],
        comp_numeric_mask: np.ndarray,
        scen_numeric_mask: np.ndarray,
    ) -> "FeatureNormalizer":
        recs = list(records)
        if not recs:
            raise ValueError("cannot fit FeatureNormalizer on empty records")
        self.comp_numeric = np.asarray(comp_numeric_mask, dtype=bool)
        self.scen_numeric = np.asarray(scen_numeric_mask, dtype=bool)

        all_comp = np.concatenate(
            [r["component_features"] for r in recs], axis=0
        ).astype(np.float64)
        all_scen = np.stack(
            [r["scenario_features"] for r in recs], axis=0
        ).astype(np.float64)

        self.comp_mean = all_comp.mean(axis=0)
        self.comp_std = all_comp.std(axis=0, ddof=0)
        self.comp_std[self.comp_std == 0] = 1.0

        self.scen_mean = all_scen.mean(axis=0)
        self.scen_std = all_scen.std(axis=0, ddof=0)
        self.scen_std[self.scen_std == 0] = 1.0

        self.fitted_ = True
        return self

    def transform_record(self, rec: dict) -> dict:
        if not self.fitted_:
            raise RuntimeError("FeatureNormalizer is not fitted")
        cf = rec["component_features"].astype(np.float32, copy=True)
        sf = rec["scenario_features"].astype(np.float32, copy=True)

        if self.comp_numeric is not None and self.comp_numeric.any():
            m = self.comp_numeric
            cf[:, m] = (cf[:, m] - self.comp_mean[m]) / self.comp_std[m]  # type: ignore[index]
        if self.scen_numeric is not None and self.scen_numeric.any():
            m = self.scen_numeric
            sf[m] = (sf[m] - self.scen_mean[m]) / self.scen_std[m]  # type: ignore[index]

        out = dict(rec)
        out["component_features"] = cf
        out["scenario_features"] = sf
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(obj).__name__}")
        return obj
