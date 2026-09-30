"""The contract every arm in the bake-off implements.

Keeping this deliberately small is what lets the agent treat retrieval models,
a transformer and a two-stage ranker as interchangeable arms, and lets the
evaluator guarantee that every arm is scored on identical candidate sets.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class Recommender(ABC):
    """Scores a fixed candidate set for a batch of users.

    ``score_users`` returns an ``(n_users, n_candidates)`` matrix. Higher is
    better. The evaluator masks already-seen items itself, so implementations
    must not do their own filtering - that keeps the masking rule identical
    across arms.
    """

    name: str = "recommender"
    family: str = "base"
    #: Set by subclasses that consume within-session context. The evaluator
    #: routes rows through ``score_rows`` for these.
    session_aware: bool = False

    def __init__(self, **params: Any):
        self.params: Dict[str, Any] = params
        self.data = None

    # ------------------------------------------------------------------ fit
    @abstractmethod
    def fit(self, data) -> "Recommender":
        ...

    @abstractmethod
    def score_users(self, users: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        ...

    # Session-aware arms override this to see the prefix of the current
    # session. ``rows`` carries one row per evaluation target.
    def score_rows(self, rows: pd.DataFrame, candidates: np.ndarray) -> np.ndarray:
        return self.score_users(rows["u"].to_numpy(), candidates)

    # ------------------------------------------------------------------ meta
    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "family": self.family, "params": dict(self.params)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.__class__.__name__}({self.params})"


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-10)
