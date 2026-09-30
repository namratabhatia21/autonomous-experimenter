"""The agent's action space.

Everything the planner is allowed to try lives here. Two reasons that matters:

  * The planner - rule-based or LLM-driven - can only emit a ``kind`` from
    this table plus keyword arguments. A model spec that is not in the table
    is rejected before anything is trained, so a hallucinated suggestion
    cannot reach the runner.

  * Constructing a model from a plain dict is what makes a run replayable. The
    whole experiment log serialises to JSON and can be re-executed.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from .models.base import Recommender
from .models.classical import BPRMatrixFactorization, ItemKNN, PopularityRecommender
from .models.graph import CrossVerticalBridge, GraphWalkRetriever
from .models.hybrid import TwoStageRanker
from .models.sequential import SASRec, SessionAdaptiveSASRec

#: kind -> (constructor, human-readable purpose)
CATALOG: Dict[str, Dict[str, Any]] = {
    "popularity": {
        "ctor": PopularityRecommender,
        "role": "non-personalised floor",
    },
    "item_knn": {
        "ctor": ItemKNN,
        "role": "classical item-item collaborative filtering",
    },
    "bpr_mf": {
        "ctor": BPRMatrixFactorization,
        "role": "matrix factorisation with a pairwise ranking loss (BPR)",
    },
    "graph_walk": {
        "ctor": GraphWalkRetriever,
        "role": "graph retrieval: random walk with restart over item + KG edges",
    },
    "cross_vertical_bridge": {
        "ctor": CrossVerticalBridge,
        "role": "explicit, inspectable category-level cross-vertical transfer",
    },
    "sasrec": {
        "ctor": SASRec,
        "role": "self-attentive sequential recommendation (transformer)",
    },
    "sasrec_session": {
        "ctor": SessionAdaptiveSASRec,
        "role": "transformer re-encoded with in-session events (streaming)",
    },
    "two_stage": {
        "ctor": TwoStageRanker,
        "role": "learned fusion of retrieval sources (second-stage ranker)",
    },
}

#: Kinds whose constructor needs already-fitted retrievers passed in.
COMPOSITE = {"two_stage"}


def build(kind: str, params: Dict[str, Any] | None = None,
          retrievers: List[Recommender] | None = None) -> Recommender:
    if kind not in CATALOG:
        raise KeyError(
            f"Unknown model kind {kind!r}. The planner may only choose from: "
            f"{sorted(CATALOG)}"
        )
    params = dict(params or {})
    ctor = CATALOG[kind]["ctor"]
    if kind in COMPOSITE:
        if not retrievers:
            raise ValueError(f"{kind} requires retrievers to fuse")
        return ctor(retrievers=retrievers, **params)
    return ctor(**params)


def describe_catalog() -> str:
    """Rendered into the planner prompt so the LLM sees its real action space."""
    return "\n".join(f"- {k}: {v['role']}" for k, v in CATALOG.items())
