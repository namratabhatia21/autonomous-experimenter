"""The agent's experiment vocabulary.

A ``Hypothesis`` is a claim with the comparison that decides it. A ``Trial`` is
one configured arm. A ``RoundPlan`` is what the planner emits each round. All
plain dataclasses, so a whole run serialises to JSON and can be audited or
replayed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: _asdict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj


@dataclass
class Hypothesis:
    """A falsifiable statement the agent commits to *before* seeing the result.

    ``treatment`` and ``control`` name the two arms whose difference decides
    the claim; ``slice`` names the sub-population it applies to. Writing the
    comparison down in advance is what stops the agent from finding a slice
    after the fact where the number happened to look good.

    ``treatment``/``control`` may be a placeholder like ``__best_retriever__``,
    resolved by the orchestrator once the round's results are in.
    """
    id: str
    round: int
    statement: str
    rationale: str            # why the agent believes it, grounded in evidence
    metric: str               # metric the hypothesis is judged on
    treatment: str = ""       # arm expected to win (or a placeholder)
    control: str = ""         # arm it must beat ("" = absolute threshold)
    slice: str = "all"        # population the claim is about
    target_delta: float = 0.0  # minimum *relative* improvement to count as support
    direction: str = "increase"   # increase | no_harm
    baseline: Optional[float] = None
    verdict: Optional[str] = None      # supported | refuted | inconclusive
    observed: Optional[float] = None
    evidence: str = ""
    comparison: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trial:
    """One configured pipeline the agent wants to evaluate."""
    id: str
    round: int
    family: str               # logistic_regression, hist_gradient_boosting, ...
    label: str                # human-readable name shown in the leaderboard
    params: Dict[str, Any] = field(default_factory=dict)
    hypothesis_id: Optional[str] = None
    motivation: str = ""


@dataclass
class RoundPlan:
    round: int
    goal: str
    hypotheses: List[Hypothesis] = field(default_factory=list)
    trials: List[Trial] = field(default_factory=list)
    reflection: str = ""      # what the agent concluded from the previous round
    actions: List[str] = field(default_factory=list)
    judgements: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
