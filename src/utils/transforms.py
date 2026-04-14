"""Target transformations for DOT regression.

Two targets with very different distributions:
- Delta KV100 (``target_dkv``): signed, heavy-tailed, balanced sign → ``signed_log1p``
- Oxidation EOT (``target_eot``): strictly positive, moderately skewed → ``log1p``

``TargetTransformer`` applies a per-target nonlinearity and then standardizes
to zero-mean / unit-variance in transformed space. It stores both the
transformed-space stats (used by the model head) and the raw-space stats
(used by :func:`src.utils.metrics.normalized_mae`).

The class is picklable so a fitted transformer can be shipped with model
weights.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping

import numpy as np

try:  # torch is optional for feature-engineering-only contexts
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


TransformName = Literal["identity", "log1p", "signed_log1p"]


# ---------------------------------------------------------------------------
# Elementwise transforms (numpy OR torch, chosen by type of input)
# ---------------------------------------------------------------------------


def signed_log1p(x):
    """``sign(x) * log1p(|x|)`` — elementwise, numpy or torch."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch.sign(x) * torch.log1p(torch.abs(x))
    arr = np.asarray(x, dtype=np.float64)
    return np.sign(arr) * np.log1p(np.abs(arr))


def inverse_signed_log1p(y):
    """Inverse of :func:`signed_log1p`: ``sign(y) * (exp(|y|) - 1)``."""
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        return torch.sign(y) * torch.expm1(torch.abs(y))
    arr = np.asarray(y, dtype=np.float64)
    return np.sign(arr) * np.expm1(np.abs(arr))


def log1p(x):
    """``log(1 + x)`` — numpy or torch wrapper."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch.log1p(x)
    return np.log1p(np.asarray(x, dtype=np.float64))


def inverse_log1p(y):
    """Inverse of :func:`log1p`."""
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        return torch.expm1(y)
    return np.expm1(np.asarray(y, dtype=np.float64))


_FORWARD: dict[str, Any] = {
    "identity": lambda x: x if _HAS_TORCH and isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64),
    "log1p": log1p,
    "signed_log1p": signed_log1p,
}
_INVERSE: dict[str, Any] = {
    "identity": lambda x: x if _HAS_TORCH and isinstance(x, torch.Tensor) else np.asarray(x, dtype=np.float64),
    "log1p": inverse_log1p,
    "signed_log1p": inverse_signed_log1p,
}


# ---------------------------------------------------------------------------
# TargetTransformer
# ---------------------------------------------------------------------------


DEFAULT_CONFIG: dict[str, TransformName] = {
    "target_dkv": "signed_log1p",
    "target_eot": "log1p",
}


@dataclass
class TargetTransformer:
    """Fit per-target nonlinearity + standardization on train data.

    Parameters
    ----------
    config
        Mapping ``target_column_name -> transform_name``. The transform names
        must be one of ``identity``, ``log1p``, ``signed_log1p``. Defaults
        match the DOT task: ``target_dkv -> signed_log1p``,
        ``target_eot -> log1p``.
    """

    config: dict[str, TransformName] = field(default_factory=lambda: dict(DEFAULT_CONFIG))
    mean_: dict[str, float] = field(default_factory=dict)
    std_: dict[str, float] = field(default_factory=dict)
    raw_mean_: dict[str, float] = field(default_factory=dict)
    raw_std_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    # -- fitting ----------------------------------------------------------

    def fit(self, df) -> "TargetTransformer":
        """Compute raw-space stats AND transformed-space stats.

        ``df`` is any object that supports ``df[col].to_numpy()`` (pandas
        DataFrame is the intended input). Raises if any target column is
        missing or contains non-finite values.
        """
        for col, tname in self.config.items():
            if tname not in _FORWARD:
                raise ValueError(f"unknown transform '{tname}' for column '{col}'")
            if col not in df.columns:
                raise KeyError(f"target column '{col}' not in dataframe")

            raw = np.asarray(df[col].to_numpy(), dtype=np.float64)
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"non-finite values in target '{col}' during fit")

            self.raw_mean_[col] = float(raw.mean())
            raw_std = float(raw.std(ddof=0))
            self.raw_std_[col] = raw_std if raw_std > 0 else 1.0

            t = np.asarray(_FORWARD[tname](raw), dtype=np.float64)
            self.mean_[col] = float(t.mean())
            s = float(t.std(ddof=0))
            self.std_[col] = s if s > 0 else 1.0

        self.fitted_ = True
        return self

    # -- transform / inverse ---------------------------------------------

    def transform(self, df) -> dict[str, np.ndarray]:
        """Map raw targets to standardized transformed space.

        Returns a dict ``{col: np.ndarray}`` so callers can decide how to
        stack them (dict avoids committing to a column order).
        """
        self._check_fitted()
        out: dict[str, np.ndarray] = {}
        for col, tname in self.config.items():
            raw = np.asarray(df[col].to_numpy(), dtype=np.float64)
            t = np.asarray(_FORWARD[tname](raw), dtype=np.float64)
            out[col] = (t - self.mean_[col]) / self.std_[col]
        return out

    def inverse_transform(
        self, arr_by_col: Mapping[str, Any]
    ) -> dict[str, np.ndarray]:
        """Map standardized transformed predictions back to raw space."""
        self._check_fitted()
        out: dict[str, np.ndarray] = {}
        for col, tname in self.config.items():
            if col not in arr_by_col:
                raise KeyError(f"column '{col}' missing from inverse_transform input")
            a = np.asarray(arr_by_col[col], dtype=np.float64)
            t = a * self.std_[col] + self.mean_[col]
            out[col] = np.asarray(_INVERSE[tname](t), dtype=np.float64)
        return out

    # -- accessors --------------------------------------------------------

    def raw_stats(self) -> dict[str, dict[str, float]]:
        """``{col: {'mean': ..., 'std': ...}}`` in **raw** space.

        Consumed by :func:`src.utils.metrics.normalized_mae`.
        """
        self._check_fitted()
        return {
            col: {"mean": self.raw_mean_[col], "std": self.raw_std_[col]}
            for col in self.config
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "TargetTransformer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    # -- internals --------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.fitted_:
            raise RuntimeError(
                "TargetTransformer is not fitted; call fit() on train targets first"
            )
