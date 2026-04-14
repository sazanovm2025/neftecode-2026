"""Component properties: value parsing, wide matrix, resolver + imputation.

Pipeline:
1. Parse raw string values (``'13,74'``, ``'10-20'``, ``'<0.5'``).
2. Pivot into a wide matrix indexed by ``(component_id, batch_id)`` with one
   column per ``property_name``.
3. Select the top-K most-covered properties over the universe of
   ``(component, batch)`` pairs that actually appear in train+test mixtures.
4. Build class-median lookup on train for post-resolution NaN imputation.

At inference time, ``resolve(component_id, batch_id)`` applies the
three-level fallback: measured → typical → NaN. A separate
``impute_class_median`` step then fills remaining NaNs with the
per-class median computed from train.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .vocab import Vocab, extract_component_type


FEATURE_PROPERTIES_K: int = 15

# fuzzy markers used to discover the KV100 and TBN columns by substring
KV100_MARKERS: tuple[str, ...] = ("Кинематическая вязкость", "100")
TBN_MARKERS: tuple[str, ...] = ("Щелочное число",)


# --- raw value parsing ----------------------------------------------------

_RANGE_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)$")
_THRESH_RE = re.compile(r"^[<>≤≥]\s*=?\s*(-?\d+(?:\.\d+)?)$")


def parse_property_value(raw) -> float:
    """Parse a raw string property value to ``float``; return ``NaN`` on failure.

    Supported forms:
    - plain numbers, optional leading sign
    - comma as decimal separator (``'13,74'`` -> ``13.74``)
    - ranges (``'10-20'`` -> ``15.0``) with hyphen / en-dash / em-dash
    - thresholds (``'<0.5'`` / ``'>=100'`` -> ``0.5`` / ``100``)
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, float) and np.isnan(raw):
        return float("nan")
    s = str(raw).strip().replace(",", ".").replace(" ", "")
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        pass
    m = _RANGE_RE.match(s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = _THRESH_RE.match(s)
    if m:
        return float(m.group(1))
    return float("nan")


# --- PropertyResolver -----------------------------------------------------

@dataclass
class PropertyResolver:
    wide: pd.DataFrame = None  # MultiIndex(component_id, batch_id) -> columns=property_name
    feature_properties: list[str] = field(default_factory=list)
    class_medians: dict[str, dict[str, float]] = field(default_factory=dict)
    kv100_col: str | None = None
    tbn_col: str | None = None
    # Frozen set of (component_id, batch_id) pairs from the TRAIN mixtures.
    # Used by the ``batch_known`` feature: tells the model whether a given
    # test pair has been seen at train time (generalization signal), NOT
    # whether we have measured properties for it (that's ``is_measured``).
    train_pairs: set = field(default_factory=set)
    # Per-class property applicability (physical relevance signal).
    # ``class_applicability[class_name][property_name]`` is True iff at
    # least one component of that class had a MEASURED (non-typical)
    # value of that property in train-mixture pairs. Used to tell the
    # model "this property is physically 0 / not applicable" vs
    # "unknown, we're imputing with class median". See ``get_applicability_vector``.
    class_applicability: dict[str, dict[str, bool]] = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        properties_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        vocab: Vocab,
        k: int = FEATURE_PROPERTIES_K,
    ) -> "PropertyResolver":
        p = properties_df.copy()
        p["value"] = p["value_raw"].apply(parse_property_value)
        p = p[
            p["property_name"].notna()
            & (p["property_name"].astype(str).str.strip() != "")
        ]
        wide = (
            p.dropna(subset=["value"])
            .groupby(["component_id", "batch_id", "property_name"])["value"]
            .mean()
            .unstack("property_name")
            .sort_index()
        )

        mix_pairs = (
            pd.concat(
                [train_df[["component_id", "batch_id"]], test_df[["component_id", "batch_id"]]],
                ignore_index=True,
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        feat_props = _select_top_properties(wide, mix_pairs, k)

        kv = next((c for c in wide.columns if _matches(c, KV100_MARKERS)), None)
        tbn = next((c for c in wide.columns if _matches(c, TBN_MARKERS)), None)

        train_pairs = {
            (str(r.component_id), str(r.batch_id))
            for r in train_df[["component_id", "batch_id"]]
            .drop_duplicates()
            .itertuples(index=False)
        }

        resolver = cls(
            wide=wide,
            feature_properties=feat_props,
            kv100_col=kv,
            tbn_col=tbn,
            train_pairs=train_pairs,
        )
        resolver._compute_class_medians(train_df)
        resolver._compute_class_applicability(train_df)
        return resolver

    def _compute_class_applicability(self, train_df: pd.DataFrame) -> None:
        """For each class, record which top-K properties ever have a MEASURED
        (non-typical) value on at least one train (component, batch) pair."""
        applicable: dict[str, set[str]] = {}
        train_pairs_df = train_df[["component_id", "batch_id"]].drop_duplicates()
        for row in train_pairs_df.itertuples(index=False):
            cid = str(row.component_id)
            bid = str(row.batch_id)
            if bid == "typical":
                continue  # defensive: mixtures should never reference 'typical'
            klass = extract_component_type(cid)
            applicable.setdefault(klass, set())
            if (cid, bid) not in self.wide.index:
                continue
            row_vals = self.wide.loc[(cid, bid)]
            for prop in self.feature_properties:
                if prop not in self.wide.columns:
                    continue
                v = row_vals[prop]
                if not _is_nan(v):
                    applicable[klass].add(prop)

        self.class_applicability = {
            klass: {prop: (prop in props_set) for prop in self.feature_properties}
            for klass, props_set in applicable.items()
        }

    def get_applicability_vector(self, component_id: str) -> np.ndarray:
        """Return a ``(K,)`` 0/1 vector over :attr:`feature_properties`.

        Fallback: classes unknown to train → all-ones (let the model decide).
        """
        klass = extract_component_type(component_id)
        if klass not in self.class_applicability:
            return np.ones(len(self.feature_properties), dtype=np.float32)
        d = self.class_applicability[klass]
        return np.array(
            [1.0 if d.get(p, False) else 0.0 for p in self.feature_properties],
            dtype=np.float32,
        )

    def _compute_class_medians(self, train_df: pd.DataFrame) -> None:
        """Per-class, per-property medians on train rows (measured-or-typical)."""
        props_needed = list(self.feature_properties)
        for extra in (self.kv100_col, self.tbn_col):
            if extra and extra not in props_needed:
                props_needed.append(extra)

        train_pairs = train_df[["component_id", "batch_id"]].drop_duplicates()
        records: list[dict] = []
        for row in train_pairs.itertuples(index=False):
            vals = self._raw_lookup(row.component_id, row.batch_id, props_needed)
            vals["_class"] = extract_component_type(row.component_id)
            records.append(vals)
        df = pd.DataFrame(records)

        medians: dict[str, dict[str, float]] = {}
        for prop in props_needed:
            if prop not in df.columns:
                continue
            grouped = df.groupby("_class")[prop].median()
            medians[prop] = {
                k: float(v) for k, v in grouped.dropna().to_dict().items()
            }
        self.class_medians = medians

    # -- lookups ---------------------------------------------------------

    def _raw_lookup(
        self, component_id: str, batch_id: str, props: Iterable[str]
    ) -> dict[str, float]:
        """Two-level lookup (measured → typical) without class imputation."""
        measured_key = (component_id, batch_id)
        typical_key = (component_id, "typical")
        has_m = measured_key in self.wide.index
        has_t = typical_key in self.wide.index
        out: dict[str, float] = {}
        for prop in props:
            val = float("nan")
            if has_m and prop in self.wide.columns:
                v = self.wide.at[measured_key, prop]
                if not _is_nan(v):
                    val = float(v)
            if np.isnan(val) and has_t and prop in self.wide.columns:
                v = self.wide.at[typical_key, prop]
                if not _is_nan(v):
                    val = float(v)
            out[prop] = val
        return out

    def resolve(self, component_id: str, batch_id: str) -> dict[str, float]:
        """Three-level fallback for the frozen top-K feature properties.

        Returns ``{prop: value, prop__is_measured: 0/1, …}``.
        ``value`` may be NaN — use :meth:`impute_class_median` afterwards
        to fill with class-median on train.
        """
        measured_key = (component_id, batch_id)
        typical_key = (component_id, "typical")
        has_m = measured_key in self.wide.index
        has_t = typical_key in self.wide.index

        out: dict[str, float] = {}
        for prop in self.feature_properties:
            val = float("nan")
            is_measured = 0
            if has_m and prop in self.wide.columns:
                v = self.wide.at[measured_key, prop]
                if not _is_nan(v):
                    val = float(v)
                    is_measured = 1
            if np.isnan(val) and has_t and prop in self.wide.columns:
                v = self.wide.at[typical_key, prop]
                if not _is_nan(v):
                    val = float(v)
            out[prop] = val
            out[prop + "__is_measured"] = float(is_measured)
        return out

    def impute_class_median(
        self, component_id: str, props_dict: dict[str, float]
    ) -> dict[str, float]:
        klass = extract_component_type(component_id)
        for prop in self.feature_properties:
            if np.isnan(props_dict[prop]):
                med = self.class_medians.get(prop, {}).get(klass, float("nan"))
                props_dict[prop] = med
        return props_dict

    # -- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "PropertyResolver":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(obj).__name__}")
        return obj


# --- helpers --------------------------------------------------------------

def _is_nan(v) -> bool:
    return isinstance(v, float) and np.isnan(v)


def _matches(name: str, markers: tuple[str, ...]) -> bool:
    return all(m in name for m in markers)


def _select_top_properties(
    wide: pd.DataFrame, mix_pairs: pd.DataFrame, k: int
) -> list[str]:
    """Top-K properties by coverage (measured ∨ typical fallback) over mix_pairs."""
    if wide is None or wide.empty or len(mix_pairs) == 0:
        return []

    pairs_idx = pd.MultiIndex.from_frame(mix_pairs, names=["component_id", "batch_id"])
    measured = wide.reindex(pairs_idx)

    # typical-only slice, indexed by component_id
    typical_mask = wide.index.get_level_values(1) == "typical"
    typical_only = wide[typical_mask].copy()
    typical_only.index = typical_only.index.get_level_values(0)

    typical_for_pairs = typical_only.reindex(mix_pairs["component_id"].to_numpy())
    typical_for_pairs.index = pairs_idx

    combined = measured.fillna(typical_for_pairs)
    coverage = combined.notna().mean(axis=0).sort_values(ascending=False)
    return coverage.head(k).index.tolist()
