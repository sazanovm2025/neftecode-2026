"""Component-level and scenario-level feature builders.

A ``scenario record`` is the atomic unit the downstream dataset operates on:

    {
        "scenario_id": str,
        "component_features": np.ndarray (N, D_comp),
        "component_mask":     np.ndarray (N,),
        "scenario_features":  np.ndarray (D_scen,),
        "targets":            np.ndarray (2,) | None,
        "sign_target":        int | None,
    }

All dimensions are derived from ``Vocab`` + ``PropertyResolver`` so there
are no magic numbers. Callers can inspect :func:`component_feature_schema`
and :func:`scenario_feature_schema` for ordered (name, width) lists and
:func:`component_numeric_mask` / :func:`scenario_numeric_mask` for per-position
boolean masks telling the ``FeatureNormalizer`` which columns to z-score.

NOTE ON ``share_pct``: per task spec, the raw shares are in a
"transformed" representation (sums to 200-950%). ``share_pct`` is used
only as a per-component feature — NEVER as a pooling weight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .properties import PropertyResolver
from .vocab import Vocab, extract_component_type


# ---------------------------------------------------------------------------
# schemas & dims
# ---------------------------------------------------------------------------


def component_feature_schema(resolver: PropertyResolver) -> list[tuple[str, int]]:
    k = len(resolver.feature_properties)
    return [
        ("component_idx", 1),   # categorical — skip from normalization
        ("type_idx", 1),        # categorical — skip from normalization
        ("share_pct", 1),
        ("rank_in_class", 1),
        ("batch_known", 1),     # binary, skip from normalization
        ("properties", k),
        ("is_measured", k),     # binary, skip from normalization
        ("is_applicable", k),   # binary, skip from normalization — physical relevance flag
    ]


def component_feature_dim(resolver: PropertyResolver) -> int:
    return sum(w for _, w in component_feature_schema(resolver))


def component_numeric_mask(resolver: PropertyResolver) -> np.ndarray:
    """Boolean mask of shape ``(D_comp,)`` — True where z-score normalization applies.

    Positions excluded from normalization (mask = False) — all binary /
    categorical features that would otherwise be corrupted by z-score:
    - ``component_idx``, ``type_idx`` — categorical indices
    - ``batch_known``                 — binary, 1 for all train rows
    - ``is_measured``                 — binary flags (per property)
    - ``is_applicable``               — binary flags (per property)
    """
    dim = component_feature_dim(resolver)
    mask = np.ones(dim, dtype=bool)
    binary_or_cat = {
        "component_idx",
        "type_idx",
        "batch_known",
        "is_measured",
        "is_applicable",
    }
    offset = 0
    for name, width in component_feature_schema(resolver):
        if name in binary_or_cat:
            mask[offset : offset + width] = False
        offset += width
    return mask


def scenario_feature_schema(
    vocab: Vocab, resolver: PropertyResolver
) -> list[tuple[str, int]]:
    n_types = vocab.n_types
    return [
        ("mode_id_idx", 1),     # categorical — skip from normalization
        ("T_onehot", len(vocab.T_cats)),
        ("t_onehot", len(vocab.t_cats)),
        ("biofuel_onehot", len(vocab.biofuel_cats)),
        ("catalyst_onehot", len(vocab.catalyst_cats)),
        ("has_class", n_types),
        ("mean_share_class", n_types),
        ("n_components", 1),
        ("kv100_mean", 1),
        ("kv100_std", 1),
        ("kv100_min", 1),
        ("kv100_max", 1),
        ("tbn_mean", 1),
        ("has_moly_and_antiox", 1),
        ("avg_antiox_share", 1),
        ("avg_deterg_share", 1),
    ]


def scenario_feature_dim(vocab: Vocab, resolver: PropertyResolver) -> int:
    return sum(w for _, w in scenario_feature_schema(vocab, resolver))


def scenario_numeric_mask(vocab: Vocab, resolver: PropertyResolver) -> np.ndarray:
    dim = scenario_feature_dim(vocab, resolver)
    mask = np.ones(dim, dtype=bool)
    mask[0] = False  # mode_id_idx — categorical
    return mask


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_component_features(
    scen_df: pd.DataFrame, resolver: PropertyResolver, vocab: Vocab
) -> np.ndarray:
    """Build the (N, D_comp) feature matrix for one scenario.

    ``scen_df`` is a slice of the mixtures dataframe containing the rows for
    a single ``scenario_id`` (in any order). Duplicate ``(component_id,
    batch_id)`` rows are NOT removed — per current project decision, each
    mixture row is treated as a distinct component-slot.
    """
    n = len(scen_df)
    D = component_feature_dim(resolver)
    X = np.zeros((n, D), dtype=np.float32)

    classes: list[str] = [extract_component_type(c) for c in scen_df["component_id"]]
    shares: np.ndarray = scen_df["share_pct"].to_numpy(dtype=np.float32)

    rank_arr = _rank_in_class(classes, shares)

    k_props = len(resolver.feature_properties)
    for i, row in enumerate(scen_df.itertuples(index=False)):
        cid = str(row.component_id)
        bid = str(row.batch_id)

        comp_idx = float(vocab.component_idx(cid))
        type_idx = float(vocab.type_idx(cid))
        share = float(row.share_pct)
        # "Have we seen this exact (component, batch) pair at train time?"
        # 1 for every train row by construction; 0 for test rows referencing
        # batches / components that never appeared in train. This is the
        # generalization signal — NOT a data-quality flag. Whether a batch
        # has measured properties is captured separately by ``is_measured_*``.
        batch_known = 1.0 if (cid, bid) in resolver.train_pairs else 0.0

        props = resolver.resolve(cid, bid)
        resolver.impute_class_median(cid, props)

        prop_vals = np.asarray(
            [props[p] for p in resolver.feature_properties], dtype=np.float32
        )
        is_measured = np.asarray(
            [props[p + "__is_measured"] for p in resolver.feature_properties],
            dtype=np.float32,
        )
        is_applicable = resolver.get_applicability_vector(cid)
        # any lingering NaN (class median missing) → 0; kept conservative
        prop_vals = np.nan_to_num(prop_vals, nan=0.0)

        X[i, 0] = comp_idx
        X[i, 1] = type_idx
        X[i, 2] = share
        X[i, 3] = rank_arr[i]
        X[i, 4] = batch_known
        X[i, 5 : 5 + k_props] = prop_vals
        X[i, 5 + k_props : 5 + 2 * k_props] = is_measured
        X[i, 5 + 2 * k_props : 5 + 3 * k_props] = is_applicable
    return X


def build_scenario_features(
    scen_df: pd.DataFrame, resolver: PropertyResolver, vocab: Vocab
) -> np.ndarray:
    """Build the 1D scenario feature vector."""
    first = scen_df.iloc[0]
    T = float(first["T_C"])
    t = float(first["t_h"])
    bf = float(first["biofuel_pct"])
    cat = int(first["catalyst_cat"])
    mode_idx = float(vocab.mode_idx(T, t, bf, cat))

    type_list = list(vocab.component_type_to_idx.keys())  # stable insertion order
    n_types = len(type_list)

    classes_in_scen: list[str] = [
        extract_component_type(c) for c in scen_df["component_id"]
    ]
    shares = scen_df["share_pct"].to_numpy(dtype=np.float32)

    has_class = np.zeros(n_types, dtype=np.float32)
    mean_share_class = np.zeros(n_types, dtype=np.float32)
    for i, ty in enumerate(type_list):
        idxs = [j for j, c in enumerate(classes_in_scen) if c == ty]
        if idxs:
            has_class[i] = 1.0
            mean_share_class[i] = float(shares[idxs].mean())

    kv_vals, tbn_vals = _collect_kv_tbn(scen_df, resolver)
    if kv_vals.size:
        kv_feat = np.array(
            [kv_vals.mean(), kv_vals.std(ddof=0), kv_vals.min(), kv_vals.max()],
            dtype=np.float32,
        )
    else:
        kv_feat = np.zeros(4, dtype=np.float32)
    tbn_mean = float(tbn_vals.mean()) if tbn_vals.size else 0.0

    moly_class = _find_class_containing(type_list, "молибден")
    antiox_class = _find_class_containing(type_list, "антиоксидант")
    deterg_class = _find_class_containing(type_list, "детергент")
    has_moly = any(c == moly_class for c in classes_in_scen) if moly_class else False
    has_antiox = any(c == antiox_class for c in classes_in_scen) if antiox_class else False

    def avg_share(target_class: str | None) -> float:
        if not target_class:
            return 0.0
        sel = [shares[i] for i, c in enumerate(classes_in_scen) if c == target_class]
        return float(np.mean(sel)) if sel else 0.0

    parts: list[np.ndarray] = [
        np.array([mode_idx], dtype=np.float32),
        _one_hot(T, vocab.T_cats),
        _one_hot(t, vocab.t_cats),
        _one_hot(bf, vocab.biofuel_cats),
        _one_hot(cat, vocab.catalyst_cats),
        has_class,
        mean_share_class,
        np.array([float(len(scen_df))], dtype=np.float32),
        kv_feat,
        np.array([tbn_mean], dtype=np.float32),
        np.array([1.0 if (has_moly and has_antiox) else 0.0], dtype=np.float32),
        np.array([avg_share(antiox_class)], dtype=np.float32),
        np.array([avg_share(deterg_class)], dtype=np.float32),
    ]
    vec = np.concatenate(parts).astype(np.float32)

    expected_dim = scenario_feature_dim(vocab, resolver)
    if vec.shape[0] != expected_dim:
        raise RuntimeError(
            f"scenario feature size mismatch: built {vec.shape[0]}, expected {expected_dim}"
        )
    return vec


def build_scenario_record(
    scenario_id: str,
    scen_df: pd.DataFrame,
    resolver: PropertyResolver,
    vocab: Vocab,
    has_targets: bool = True,
) -> dict:
    """Assemble a full scenario record. ``has_targets`` should be False for test data."""
    comp = build_component_features(scen_df, resolver, vocab)
    scen = build_scenario_features(scen_df, resolver, vocab)
    rec: dict = {
        "scenario_id": scenario_id,
        "component_features": comp,
        "component_mask": np.ones(len(scen_df), dtype=np.float32),
        "scenario_features": scen,
        "targets": None,
        "sign_target": None,
    }
    if has_targets and "target_dkv" in scen_df.columns:
        dkv = float(scen_df["target_dkv"].iloc[0])
        eot = float(scen_df["target_eot"].iloc[0])
        rec["targets"] = np.array([dkv, eot], dtype=np.float32)
        rec["sign_target"] = 1 if dkv > 0 else 0
    return rec


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _rank_in_class(classes: Sequence[str], shares: np.ndarray) -> np.ndarray:
    """Rank each component within its own class (by share desc), normalized.

    Rank is in ``[0, 1]``. Singletons get 0. Ties break by first-seen order —
    stable but arbitrary for equal shares.
    """
    out = np.zeros(len(classes), dtype=np.float32)
    by_class: dict[str, list[int]] = {}
    for i, c in enumerate(classes):
        by_class.setdefault(c, []).append(i)
    for _, idxs in by_class.items():
        k = len(idxs)
        if k <= 1:
            continue
        order = sorted(idxs, key=lambda j: -float(shares[j]))
        for rank, j in enumerate(order):
            out[j] = rank / (k - 1)
    return out


def _one_hot(value: float, cats: Sequence[float]) -> np.ndarray:
    v = np.zeros(len(cats), dtype=np.float32)
    for i, c in enumerate(cats):
        if float(c) == float(value):
            v[i] = 1.0
            return v
    return v


def _find_class_containing(type_list: Iterable[str], marker: str) -> str | None:
    for t in type_list:
        if marker in t:
            return t
    return None


def _collect_kv_tbn(
    scen_df: pd.DataFrame, resolver: PropertyResolver
) -> tuple[np.ndarray, np.ndarray]:
    targets = [p for p in (resolver.kv100_col, resolver.tbn_col) if p]
    if not targets:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    kv: list[float] = []
    tbn: list[float] = []
    for row in scen_df.itertuples(index=False):
        looked = resolver._raw_lookup(str(row.component_id), str(row.batch_id), targets)
        if resolver.kv100_col:
            v = looked.get(resolver.kv100_col, float("nan"))
            if not np.isnan(v):
                kv.append(v)
        if resolver.tbn_col:
            v = looked.get(resolver.tbn_col, float("nan"))
            if not np.isnan(v):
                tbn.append(v)
    return np.array(kv, dtype=np.float32), np.array(tbn, dtype=np.float32)
