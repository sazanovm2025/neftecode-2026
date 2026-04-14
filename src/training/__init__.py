"""Training primitives: cross-validation, losses, trainer module, …"""
from .cv import (
    assert_mode_coverage,
    get_fold_summary,
    scenario_based_kfold,
)

__all__ = [
    "scenario_based_kfold",
    "assert_mode_coverage",
    "get_fold_summary",
]
