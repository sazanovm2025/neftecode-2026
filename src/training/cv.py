"""Cross-validation for scenario-level records.

Why StratifiedGroupKFold: the dataset has a massively skewed mode
distribution (``mode_id=6`` owns 115/167 scenarios). A plain KFold would
shuffle this noise around randomly — we want to keep the mode histogram
roughly equal across folds so each fold's metrics are comparable.

Why "soft" stratification: several modes have only 1-2 scenarios each —
strict StratifiedKFold on per-mode strata is mathematically impossible
(you can't split 1 sample across 5 folds). We collapse modes with fewer
than ``rare_mode_threshold`` occurrences into a single ``"rare"`` stratum
so the stratifier has enough material to work with while still keeping
the dominant modes balanced.

Assumptions:
- Each scenario record has ``scenario_features[0] == mode_id_idx``. This
  is enforced by the scenario feature schema (see
  :func:`src.data.features.scenario_feature_schema`) where
  ``"mode_id_idx"`` is always the first entry.
- Records are 1:1 with scenarios (one record per ``scenario_id``), so
  groups used by :class:`StratifiedGroupKFold` are effectively identity.
  We still pass them — it costs nothing and documents intent.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


MODE_ID_POSITION = 0  # scenario_features[0] per the schema


def _extract_mode_ids(records: Sequence[dict]) -> np.ndarray:
    return np.array(
        [int(r["scenario_features"][MODE_ID_POSITION]) for r in records],
        dtype=int,
    )


def _extract_scenario_ids(records: Sequence[dict]) -> np.ndarray:
    return np.array([str(r["scenario_id"]) for r in records])


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def scenario_based_kfold(
    records: Sequence[dict],
    n_splits: int = 5,
    seed: int = 42,
    rare_mode_threshold: int = 6,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold split with mode-id stratification + rare-mode collapsing.

    Parameters
    ----------
    records
        Scenario records produced by
        :func:`src.data.features.build_scenario_record`. Must expose
        ``scenario_features`` (ndarray, ``[0] == mode_id_idx``) and
        ``scenario_id`` (str).
    n_splits
        Number of folds. Must be ``<=`` the smallest stratum size.
    seed
        Random state forwarded to ``StratifiedGroupKFold``.
    rare_mode_threshold
        Modes whose total train-count is strictly less than this value
        are collapsed into a single ``"rare"`` stratum. Set to 0 to
        disable rare-mode collapsing.

    Returns
    -------
    list of ``(train_idx, val_idx)`` — numpy integer arrays, one per fold.
    """
    if len(records) == 0:
        raise ValueError("scenario_based_kfold: empty records")

    mode_ids = _extract_mode_ids(records)
    groups = _extract_scenario_ids(records)
    counts = Counter(mode_ids)

    strata = np.array(
        [
            f"mode_{mid}" if counts[mid] >= rare_mode_threshold else "rare"
            for mid in mode_ids
        ],
        dtype=object,
    )

    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    x_dummy = np.zeros(len(records))
    folds = [
        (np.asarray(tr, dtype=int), np.asarray(va, dtype=int))
        for tr, va in cv.split(x_dummy, strata, groups=groups)
    ]
    return folds


def assert_mode_coverage(
    train_indices: Iterable[int],
    records: Sequence[dict],
    min_coverage: int = 1,
) -> None:
    """Verify every mode that CAN be in train is in train ``min_coverage`` times.

    Modes with ``total_count < 2`` are skipped: a singleton mode can't
    possibly be in both train and val of the same fold, so failing on it
    would be a tautology. This matters for rare modes like
    ``mode_id=1`` / ``mode_id=2`` with only one scenario each.

    Raises
    ------
    AssertionError
        When at least one non-singleton mode is under-represented. The
        message lists every offending mode with (got, total) counts.
    """
    train_indices = list(train_indices)
    if not train_indices:
        raise ValueError("assert_mode_coverage: empty train_indices")

    all_mode_ids = _extract_mode_ids(records)
    total_counts = Counter(all_mode_ids.tolist())
    train_counts = Counter(all_mode_ids[np.asarray(train_indices, dtype=int)].tolist())

    missing: list[tuple[int, int, int]] = []
    for mode_id, total in total_counts.items():
        if total < 2:
            continue  # unavoidable
        got = train_counts.get(mode_id, 0)
        if got < min_coverage:
            missing.append((mode_id, got, total))

    if missing:
        lines = [
            f"  mode_id={mid}: {got} in train (total in dataset: {tot})"
            for mid, got, tot in sorted(missing)
        ]
        raise AssertionError(
            f"mode coverage below min_coverage={min_coverage}:\n" + "\n".join(lines)
        )


def get_fold_summary(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    records: Sequence[dict],
) -> pd.DataFrame:
    """Per-fold, per-split summary as a long DataFrame.

    Columns: ``fold``, ``split`` ("train"/"val"), ``n_scenarios``,
    ``mode_counts`` (dict ``mode_id -> n_scenarios``).
    """
    all_mode_ids = _extract_mode_ids(records)
    rows: list[dict] = []
    for fi, (tr, va) in enumerate(folds):
        for split_name, idxs in (("train", tr), ("val", va)):
            idxs_arr = np.asarray(idxs, dtype=int)
            mode_counts = Counter(all_mode_ids[idxs_arr].tolist())
            rows.append(
                {
                    "fold": fi,
                    "split": split_name,
                    "n_scenarios": int(len(idxs_arr)),
                    "mode_counts": dict(sorted(mode_counts.items())),
                }
            )
    return pd.DataFrame(rows)
