"""PyTorch Dataset + collate_fn for scenario records.

- ``DaimlerDataset`` applies the ``FeatureNormalizer`` on-the-fly, optionally
  pre-transforms targets with a ``TargetTransformer``, and supports
  component_dropout augmentation during training.
- ``collate_fn`` pads variable-length component matrices along axis 0 to the
  batch max and produces an attention mask (1 = real component, 0 = padding).

``torch`` is imported lazily so that the module can be parsed/tested in a
pure-numpy environment; calling ``collate_fn`` or instantiating ``DaimlerDataset``
without torch installed raises a clear error.
"""
from __future__ import annotations

import random
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

    class _TorchDataset:  # type: ignore[no-redef]
        """Stub base class so the module imports without torch."""


class DaimlerDataset(_TorchDataset):
    def __init__(
        self,
        records: Sequence[dict],
        normalizer=None,
        target_transformer=None,
        training: bool = True,
        component_dropout_p: float = 0.0,
    ):
        if not _HAS_TORCH:
            raise RuntimeError(
                "DaimlerDataset requires torch; install torch to use this class"
            )
        self.records = list(records)
        self.normalizer = normalizer
        self.target_transformer = target_transformer
        self.training = training
        self.component_dropout_p = float(component_dropout_p)

        self._transformed_targets: dict[str, np.ndarray] | None = None
        if (
            target_transformer is not None
            and self.records
            and self.records[0].get("targets") is not None
        ):
            import pandas as pd

            df = pd.DataFrame(
                {
                    "target_dkv": [float(r["targets"][0]) for r in self.records],
                    "target_eot": [float(r["targets"][1]) for r in self.records],
                }
            )
            self._transformed_targets = target_transformer.transform(df)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        if self.normalizer is not None:
            rec = self.normalizer.transform_record(rec)

        comp = rec["component_features"]
        mask = rec["component_mask"]

        if self.training and self.component_dropout_p > 0.0 and comp.shape[0] > 1:
            if random.random() < self.component_dropout_p:
                drop = random.randrange(comp.shape[0])
                comp = np.delete(comp, drop, axis=0)
                mask = np.delete(mask, drop, axis=0)

        item: dict[str, Any] = {
            "scenario_id": rec["scenario_id"],
            "component_features": comp,
            "component_mask": mask,
            "scenario_features": rec["scenario_features"],
        }
        if rec.get("targets") is not None:
            item["targets"] = rec["targets"]
        if rec.get("sign_target") is not None:
            item["sign_target"] = int(rec["sign_target"])
        if self._transformed_targets is not None:
            item["targets_transformed"] = np.array(
                [
                    self._transformed_targets["target_dkv"][idx],
                    self._transformed_targets["target_eot"][idx],
                ],
                dtype=np.float32,
            )
        return item


def collate_fn(batch: Iterable[dict]) -> dict[str, Any]:
    """Pad component tensors to batch-max length; return a torch-friendly dict."""
    if not _HAS_TORCH:
        raise RuntimeError("collate_fn requires torch; install torch to use it")

    batch = list(batch)
    if not batch:
        raise ValueError("empty batch")

    max_n = max(item["component_features"].shape[0] for item in batch)
    D_comp = batch[0]["component_features"].shape[1]
    D_scen = batch[0]["scenario_features"].shape[0]
    bs = len(batch)

    comp_arr = np.zeros((bs, max_n, D_comp), dtype=np.float32)
    mask_arr = np.zeros((bs, max_n), dtype=np.float32)
    scen_arr = np.zeros((bs, D_scen), dtype=np.float32)
    for i, item in enumerate(batch):
        n = item["component_features"].shape[0]
        comp_arr[i, :n] = item["component_features"]
        mask_arr[i, :n] = item["component_mask"]
        scen_arr[i] = item["scenario_features"]

    out: dict[str, Any] = {
        "component_features": torch.from_numpy(comp_arr),
        "component_mask": torch.from_numpy(mask_arr),
        "scenario_features": torch.from_numpy(scen_arr),
        "scenario_ids": [item["scenario_id"] for item in batch],
    }
    if "targets" in batch[0]:
        out["targets"] = torch.from_numpy(
            np.stack([item["targets"] for item in batch], axis=0)
        )
    if "targets_transformed" in batch[0]:
        out["targets_transformed"] = torch.from_numpy(
            np.stack([item["targets_transformed"] for item in batch], axis=0)
        )
    if "sign_target" in batch[0]:
        out["sign_target"] = torch.tensor(
            [int(item["sign_target"]) for item in batch], dtype=torch.float32
        )
    return out
