"""The agent's experimental reasoning.

A grid search tries everything. An experimenter states what it expects, runs
the smallest comparison that could falsify it, reads the result, and lets that
decide what to run next. This module is the second thing.

Each round the planner emits:

  * **Hypotheses** - written down *before* the trials run, each with the metric
    it will be judged on, the slice it applies to, and the minimum effect that
    counts as support. A hypothesis that cannot be refuted is not allowed.
  * **Trials** - the smallest set of arms that can decide those hypotheses,
    preferring matched pairs that differ in exactly one factor.

After the runner returns results, ``judge`` scores every open hypothesis
against its own stated threshold *and* the paired bootstrap CI, then
``plan_round`` uses those verdicts - not a fixed schedule - to choose what to
do next. Refuted transfer means the next round stops spending budget on
transfer. A family that trails badly is pruned. A collapsed catalogue raises a
guardrail instead of being ignored.

An LLM may be put in the driver's seat for the *choice of next action*
(``llm_planner=True``); it selects from the same action table and its choice
is schema-validated before anything is trained. The rule-based planner is
always present underneath, so the run is reproducible without an API key.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .evaluate import PRIMARY_METRIC, EvalResult, paired_comparison
from .registry import CATALOG, describe_catalog
from .types import Hypothesis, RoundPlan, Trial

#: Relative improvement below which a round is considered to have plateaued.
PLATEAU_EPS = 0.01
#: A family this far below the leader stops earning budget.
PRUNE_RATIO = 0.75
#: Below this many rows that either arm ranked at all, a slice comparison is
#: reported as underpowered rather than as a null result.
MIN_INFORMATIVE_ROWS = 30


@dataclass
class Evidence:
    """Everything the planner knows when choosing the next round."""
    profile: Dict[str, Any]
    results: Dict[str, EvalResult] = field(default_factory=dict)
    hypotheses: List[Hypothesis] = field(default_factory=list)
    pruned_families: set = field(default_factory=set)
    notes: List[str] = field(default_factory=list)
    champion: Optional[str] = None

    def best(self, exclude: Sequence[str] = ()) -> Optional[EvalResult]:
        pool = [r for n, r in self.results.items() if n not in exclude and not r.error]
        return max(pool, key=lambda r: r.primary) if pool else None

    def by_family(self, family: str) -> List[EvalResult]:
        return [r for r in self.results.values() if r.family == family and not r.error]

    def slice_score(self, name: str, slice_name: str,
                    metric: str = PRIMARY_METRIC) -> Optional[float]:
        r = self.results.get(name)
        if r is None or slice_name not in r.slice_metrics:
            return None
        return r.slice_metrics[slice_name].get(metric)

    def verdicts(self) -> Dict[str, str]:
        return {h.id: (h.verdict or "open") for h in self.hypotheses}


class Planner:
    def __init__(self, max_rounds: int = 5, llm_planner: bool = False,
                 llm_client=None, seed: int = 0):
        self.max_rounds = max_rounds
        self.llm_planner = llm_planner
        self.llm_client = llm_client
        self.seed = seed
        self.decisions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- judging
    def judge(self, evidence: Evidence, round_no: int) -> List[str]:
        """Score every open hypothesis for this round and record the reasoning."""
        lines: List[str] = []
        for h in evidence.hypotheses:
            if h.round != round_no or h.verdict is not None:
                continue
            treat = evidence.results.get(h.treatment)
            ctrl = evidence.results.get(h.control) if h.control else None
            if treat is None or (h.control and ctrl is None):
                h.verdict, h.evidence = "inconclusive", "an arm failed to run"
                lines.append(f"{h.id}: inconclusive (missing arm)")
                continue

            if ctrl is None:
                observed = _slice_value(treat, h.slice, h.metric)
                h.observed = observed
                h.verdict = "supported" if observed >= h.target_delta else "refuted"
                h.evidence = f"{h.metric} on {h.slice} = {observed:.4f}"
            else:
                cmp = paired_comparison(
                    treat, ctrl, metric=h.metric, seed=self.seed,
                    slice_name=None if h.slice == "all" else h.slice,
                )
                h.baseline = cmp.get("baseline")
                h.observed = cmp.get("delta")
                h.comparison = cmp
                rel = cmp.get("rel_lift_pct", float("nan"))
                info = cmp.get("n_informative", cmp.get("n", 0))

                if info < MIN_INFORMATIVE_ROWS:
                    # The slice is too hard for this comparison to resolve
                    # anything: almost every row scores zero for both arms, so
                    # the difference is 0.0 by construction. Saying
                    # "inconclusive" hides that; saying "underpowered" tells the
                    # reader what to fix.
                    h.verdict = "underpowered"
                    h.evidence = (
                        f"only {info} of {cmp['n']} rows on {h.slice} were ranked in "
                        f"the top-10 by either arm, so this comparison could not have "
                        f"detected the {h.target_delta * 100:.0f}% effect it was "
                        f"looking for. Needs a larger slice, a deeper cutoff, or a "
                        f"metric that scores the whole ranking."
                    )
                else:
                    # Support requires both a big enough effect and a CI that
                    # excludes zero. Either alone is how teams ship noise.
                    big_enough = rel >= h.target_delta * 100
                    if cmp.get("significant") and big_enough:
                        h.verdict = "supported"
                    elif not cmp.get("significant"):
                        h.verdict = ("inconclusive" if abs(rel) < h.target_delta * 100
                                     else "refuted")
                    else:
                        h.verdict = "refuted"
                    h.evidence = (
                        f"{h.metric} on {h.slice}: {cmp['delta']:+.4f} "
                        f"({rel:+.1f}%), 95% CI [{cmp['ci_low']:+.4f}, {cmp['ci_high']:+.4f}], "
                        f"n={cmp['n']} ({info} informative), "
                        f"required >= {h.target_delta * 100:.0f}%"
                    )
            lines.append(f"{h.id}: {h.verdict.upper()} - {h.evidence}")
        return lines

    # ------------------------------------------------------------- pruning
    def prune(self, evidence: Evidence) -> List[str]:
        """Stop spending budget on families that are clearly out of contention."""
        msgs = []
        best = evidence.best()
        if best is None or best.primary <= 0:
            return msgs
        for fam in {r.family for r in evidence.results.values() if not r.error}:
            if fam in evidence.pruned_families or fam == best.family:
                continue
            fam_best = max(evidence.by_family(fam), key=lambda r: r.primary, default=None)
            if fam_best is None:
                continue
            if fam_best.primary < PRUNE_RATIO * best.primary:
                evidence.pruned_families.add(fam)
                msgs.append(
                    f"pruned '{fam}' - best {PRIMARY_METRIC} {fam_best.primary:.4f} is "
                    f"{100 * (1 - fam_best.primary / best.primary):.0f}% below the leader"
                )
        return msgs

    # -------------------------------------------------------------- guards
    def guardrails(self, evidence: Evidence) -> List[str]:
        """Catalogue-health checks that accuracy alone will never surface."""
        out = []
        best = evidence.best()
        if best is None:
            return out
        cov = best.metrics.get("coverage@10", 1.0)
        pop = evidence.results.get("popularity")
        floor = pop.metrics.get("coverage@10", 0.0) if pop else 0.0
        if cov < max(0.05, floor * 1.2):
            out.append(
                f"GUARDRAIL: the leader covers only {100 * cov:.1f}% of the catalogue in "
                f"its top-10 (popularity floor {100 * floor:.1f}%). A win bought by "
                f"collapsing onto head items is a merchandising risk, not a ranking win."
            )
        return out

    # ------------------------------------------------------------ planning
    def plan_round(self, round_no: int, evidence: Evidence) -> Optional[RoundPlan]:
        if round_no > self.max_rounds:
            return None
        actions = self._choose_actions(round_no, evidence)
        if not actions:
            return None
        plan = RoundPlan(round=round_no, goal="", reflection="")
        goals, reflections = [], []
        for action in actions:
            built = ACTIONS[action](round_no, evidence)
            if built is None:
                continue
            goals.append(built["goal"])
            plan.trials.extend(built["trials"])
            plan.hypotheses.extend(built["hypotheses"])
            reflections.append(built.get("why", ""))
        if not plan.trials:
            return None
        plan.goal = "; ".join(goals)
        plan.reflection = " ".join(r for r in reflections if r)
        plan.actions = actions
        return plan

    def _choose_actions(self, round_no: int, evidence: Evidence) -> List[str]:
        rule = self._rule_actions(round_no, evidence)
        if self.llm_planner and self.llm_client is not None and round_no > 1:
            llm = self._llm_actions(round_no, evidence, rule)
            if llm:
                self.decisions.append(
                    {"round": round_no, "rule_based": rule, "llm_chosen": llm}
                )
                return llm
        self.decisions.append({"round": round_no, "rule_based": rule, "llm_chosen": None})
        return rule

    def _rule_actions(self, round_no: int, evidence: Evidence) -> List[str]:
        """The deterministic policy. Every branch is a stated experimental rule."""
        p = evidence.profile
        v = evidence.verdicts()

        if round_no == 1:
            return ["baselines"]

        if round_no == 2:
            # Only worth asking about transfer if the population is actually
            # multi-surface. Otherwise go straight to sequence modelling.
            if p.get("multi_vertical_user_pct", 0) >= 15:
                return ["cross_vertical_ablation"]
            evidence.notes.append(
                "skipped the cross-vertical ablation: only "
                f"{p.get('multi_vertical_user_pct', 0):.1f}% of users touch 2+ verticals, "
                "so there is no population for transfer to help."
            )
            return ["sequential"]

        if round_no == 3:
            acts = []
            # Transfer confirmed -> push on it (KG edges, walk depth).
            if v.get("H2-transfer") == "supported":
                acts.append("graph_depth_and_kg")
            # Sequences long enough to be worth attention? Say so either way.
            if p.get("median_history", 0) >= 4 and "sasrec" not in evidence.pruned_families:
                acts.append("sequential")
            return acts or ["graph_depth_and_kg"]

        if round_no == 4:
            acts = ["fusion"]
            if p.get("has_sessions") and "sasrec" not in evidence.pruned_families:
                acts.append("session_adaptation")
            return acts

        return []

    # ------------------------------------------------------------- LLM path
    def _llm_actions(self, round_no: int, evidence: Evidence,
                     rule_default: List[str]) -> Optional[List[str]]:
        """Let Claude pick the next action from the same constrained table.

        The model sees the profile, the leaderboard, and every hypothesis
        verdict so far. It returns action names only - never free-form code or
        hyperparameters - and anything outside the table is discarded in favour
        of the rule-based choice. That is what makes an LLM safe to put in a
        planning loop: a small, typed, validated action space.
        """
        board = "\n".join(
            f"  {n}: {PRIMARY_METRIC}={r.primary:.4f} "
            f"(transfer_only={r.slice_metrics.get('transfer_only', {}).get(PRIMARY_METRIC, float('nan')):.4f}, "
            f"coverage@10={r.metrics.get('coverage@10', 0):.3f})"
            for n, r in sorted(evidence.results.items(), key=lambda kv: -kv[1].primary)
            if not r.error
        )
        verdicts = "\n".join(
            f"  {h.id} [{h.verdict or 'open'}] {h.statement} -- {h.evidence}"
            for h in evidence.hypotheses
        )
        prompt = f"""You are planning round {round_no} of an automated recommender-system experiment.

DATASET PROFILE
{json.dumps(evidence.profile, indent=2, default=str)}

LEADERBOARD SO FAR ({PRIMARY_METRIC})
{board or '  (none yet)'}

HYPOTHESIS VERDICTS
{verdicts or '  (none yet)'}

PRUNED FAMILIES: {sorted(evidence.pruned_families) or 'none'}

AVAILABLE ACTIONS (choose 1-2, by name only):
{chr(10).join(f'- {k}: {d}' for k, d in ACTION_DOCS.items())}

The rule-based planner would choose: {rule_default}

Choose the actions that would most reduce uncertainty about whether
cross-vertical personalization is worth shipping. Do not repeat an action
whose hypothesis has already been decisively settled.

Reply with JSON only: {{"actions": ["..."], "reasoning": "one sentence"}}"""

        try:
            text = self.llm_client.complete(prompt, max_tokens=400)
            blob = text[text.index("{"): text.rindex("}") + 1]
            parsed = json.loads(blob)
            chosen = [a for a in parsed.get("actions", []) if a in ACTIONS]
            if chosen:
                evidence.notes.append(
                    f"round {round_no} planned by LLM: {parsed.get('reasoning', '')}"
                )
                return chosen[:2]
        except Exception as exc:                        # noqa: BLE001
            evidence.notes.append(f"LLM planner unavailable ({exc}); used rules.")
        return None

    # ------------------------------------------------------------- stopping
    def should_stop(self, evidence: Evidence, history: List[float]) -> Optional[str]:
        if len(history) >= 2:
            prev, cur = history[-2], history[-1]
            if prev > 0 and (cur - prev) / prev < PLATEAU_EPS:
                return (
                    f"plateau: best {PRIMARY_METRIC} moved {100 * (cur - prev) / prev:+.1f}% "
                    f"({prev:.4f} -> {cur:.4f}), below the {100 * PLATEAU_EPS:.0f}% "
                    f"threshold for continuing to spend compute"
                )
        return None


# ===================================================================== actions
def _t(round_no: int, idx: int, kind: str, params: Dict[str, Any],
       motivation: str, hypothesis_id: Optional[str] = None) -> Trial:
    return Trial(
        id=f"r{round_no}t{idx}", round=round_no, family=kind,
        label=kind, params=params, motivation=motivation, hypothesis_id=hypothesis_id,
    )


def _action_baselines(round_no: int, ev: Evidence) -> Dict[str, Any]:
    trials = [
        _t(round_no, 1, "popularity", {}, "the floor every result is reported against"),
        _t(round_no, 2, "item_knn", {"top_k": 150},
           "strongest cheap baseline on sparse implicit logs"),
        _t(round_no, 3, "bpr_mf", {"factors": 64, "epochs": 30},
           "classical latent-factor control, trained with a ranking loss"),
        _t(round_no, 4, "graph_walk",
           {"cross_vertical": False, "use_metadata": False, "walk_length": 3},
           "graph retrieval restricted to a single vertical - the control arm "
           "for next round's transfer test"),
    ]
    hyps = [
        Hypothesis(
            id="H1-personalization", round=round_no,
            statement="Personalised retrieval beats non-personalised popularity by at "
                      "least 10% relative NDCG@10.",
            rationale="If this fails, nothing downstream is worth building and the "
                      "honest recommendation is to ship the popularity baseline.",
            metric=PRIMARY_METRIC, slice="all", target_delta=0.10,
            treatment="__best_non_popularity__", control="popularity",
        ),
    ]
    return {
        "goal": "establish the floor and the classical baselines",
        "trials": trials, "hypotheses": hyps,
        "why": "Round 1 buys the reference points every later claim is measured against.",
    }


def _action_cross_vertical(round_no: int, ev: Evidence) -> Dict[str, Any]:
    trials = [
        _t(round_no, 1, "graph_walk",
           {"cross_vertical": True, "use_metadata": False, "walk_length": 3},
           "identical to the round-1 within-vertical walk except that the walk may "
           "traverse cross-vertical co-occurrence edges - a one-factor ablation",
           "H2-transfer"),
        _t(round_no, 2, "cross_vertical_bridge", {},
           "an explicit, inspectable category-level transfer map, as a second and "
           "differently-shaped test of the same idea", "H2-transfer"),
    ]
    hyps = [
        Hypothesis(
            id="H2-transfer", round=round_no,
            statement="Letting retrieval traverse cross-vertical edges improves NDCG@10 "
                      "by at least 5% relative for users who are cold on the surface "
                      "being ranked but active elsewhere.",
            rationale="This is the team's core thesis stated as a falsifiable claim. "
                      "The two arms differ in exactly one factor, so a difference is "
                      "attributable to cross-vertical edges and nothing else.",
            metric=PRIMARY_METRIC, slice="transfer_only", target_delta=0.05,
            treatment="graph_walk(xvert,L=3)", control="graph_walk(within,L=3)",
        ),
        Hypothesis(
            id="H2b-no-harm", round=round_no,
            statement="Cross-vertical retrieval does not degrade NDCG@10 for users "
                      "active in only one vertical.",
            rationale="A transfer win that taxes single-surface users is not shippable. "
                      "This is the guardrail half of the same experiment.",
            metric=PRIMARY_METRIC, slice="single_vertical_user", target_delta=-0.02,
            treatment="graph_walk(xvert,L=3)", control="graph_walk(within,L=3)",
            direction="no_harm",
        ),
    ]
    return {
        "goal": "test cross-vertical transfer as a one-factor ablation",
        "trials": trials, "hypotheses": hyps,
        "why": f"{ev.profile.get('multi_vertical_user_pct', 0):.0f}% of users are active in "
               f"two or more verticals, so there is a real population for transfer to serve.",
    }


def _action_graph_depth(round_no: int, ev: Evidence) -> Dict[str, Any]:
    trials = [
        _t(round_no, 1, "graph_walk",
           {"cross_vertical": True, "use_metadata": True, "walk_length": 3},
           "adds knowledge-graph edges (category / price band / vertical) on top of "
           "the confirmed cross-vertical walk", "H3-kg"),
        _t(round_no, 2, "graph_walk",
           {"cross_vertical": True, "use_metadata": True, "walk_length": 5},
           "a longer walk reaches further through the graph at the cost of "
           "sharpness - tests whether depth or edges is doing the work"),
    ]
    hyps = [
        Hypothesis(
            id="H3-kg", round=round_no,
            statement="Adding knowledge-graph (attribute) edges improves NDCG@10 on "
                      "tail items by at least 5% relative.",
            rationale="Tail items have sparse or empty co-occurrence rows, so an "
                      "attribute hop is the only path a walk can take to reach them. "
                      "If KG edges help anywhere, it is here.",
            metric=PRIMARY_METRIC, slice="tail_target", target_delta=0.05,
            treatment="graph_walk(xvert+kg,L=3)", control="graph_walk(xvert,L=3)",
        ),
    ]
    return {
        "goal": "separate the contribution of KG edges from walk depth",
        "trials": trials, "hypotheses": hyps,
        "why": "Cross-vertical transfer was supported, so the next question is which "
               "graph structure carries it.",
    }


def _action_sequential(round_no: int, ev: Evidence) -> Dict[str, Any]:
    med = ev.profile.get("median_history", 0)
    trials = [
        _t(round_no, 1, "sasrec",
           {"epochs": 12, "d": 64, "use_vertical": True, "max_len": 50},
           "self-attention over the user's cross-vertical sequence, with an "
           "embedding for which vertical each event happened on", "H4-attention"),
        _t(round_no, 2, "sasrec",
           {"epochs": 12, "d": 64, "use_vertical": False, "max_len": 50},
           "identical transformer with the vertical embedding removed - isolates "
           "whether the cross-vertical *context* helps, not just the sequence",
           "H4-attention"),
    ]
    hyps = [
        Hypothesis(
            id="H4-attention", round=round_no,
            statement="A self-attentive sequential model beats the best non-sequential "
                      "retriever by at least 3% relative NDCG@10.",
            rationale="Attention should pay off where order carries information. With "
                      f"a median history of {med} events it may simply be underpowered, "
                      "and that is a finding worth reporting rather than hiding.",
            metric=PRIMARY_METRIC, slice="all", target_delta=0.03,
            treatment="sasrec(d=64,L=2,+vert)", control="__best_retriever__",
        ),
        Hypothesis(
            id="H4b-vertical-context", round=round_no,
            statement="Giving the transformer an explicit vertical embedding improves "
                      "NDCG@10 on the transfer_only slice by at least 3% relative.",
            rationale="Isolates cross-vertical context from raw sequence modelling: "
                      "both arms see the same events in the same order.",
            metric=PRIMARY_METRIC, slice="transfer_only", target_delta=0.03,
            treatment="sasrec(d=64,L=2,+vert)", control="sasrec(d=64,L=2,no-vert)",
        ),
    ]
    return {
        "goal": "test whether attention over the cross-vertical sequence pays for itself",
        "trials": trials, "hypotheses": hyps,
        "why": f"Median history is {med} events.",
    }


def _action_fusion(round_no: int, ev: Evidence) -> Dict[str, Any]:
    trials = [
        _t(round_no, 1, "two_stage", {"use_transfer_features": True},
           "second-stage ranker fusing the surviving retrievers, with "
           "cross-vertical history as explicit features", "H5-fusion"),
        _t(round_no, 2, "two_stage", {"use_transfer_features": False},
           "the same ranker with the cross-vertical features removed - turns "
           "transfer from a model into a measurable feature contribution",
           "H5-fusion"),
    ]
    hyps = [
        Hypothesis(
            id="H5-fusion", round=round_no,
            statement="A learned second stage beats the best single retriever by at "
                      "least 2% relative NDCG@10.",
            rationale="Fusion is only worth its serving cost if it beats its own best "
                      "input. Blending weight is chosen on validation, so alpha=0 "
                      "(pure retrieval) is inside the search space and fusion has to earn it.",
            metric=PRIMARY_METRIC, slice="all", target_delta=0.02,
            treatment="__best_two_stage__", control="__best_retriever__",
        ),
        Hypothesis(
            id="H5b-transfer-features", round=round_no,
            statement="Cross-vertical history features contribute at least 2% relative "
                      "NDCG@10 inside the ranker on the transfer_only slice.",
            rationale="If the feature ablation is flat, transfer belongs in retrieval "
                      "only and the ranker should not carry the extra features.",
            metric=PRIMARY_METRIC, slice="transfer_only", target_delta=0.02,
            treatment="two_stage(+transfer)", control="two_stage(no-transfer)",
        ),
    ]
    return {
        "goal": "fuse the surviving retrievers and measure the transfer features directly",
        "trials": trials, "hypotheses": hyps,
        "why": "Retrieval questions are settled; the remaining question is how to combine them.",
    }


def _action_session(round_no: int, ev: Evidence) -> Dict[str, Any]:
    trials = [
        _t(round_no, 1, "sasrec_session",
           {"epochs": 12, "d": 64, "use_vertical": True, "max_len": 50},
           "identical weights to the batch transformer, but the user representation "
           "is recomputed per request including in-session events", "H6-session"),
    ]
    hyps = [
        Hypothesis(
            id="H6-session", round=round_no,
            statement="Re-encoding the user within the session improves NDCG@10 by at "
                      "least 2% relative over a representation frozen at batch time.",
            rationale="Separates within-session adaptation from retraining, which are "
                      "usually argued as one thing. Model weights are identical across "
                      "both arms; only the inference-time sequence differs, so this "
                      "measures the cheap, deployable end of online learning.",
            metric=PRIMARY_METRIC, slice="all", target_delta=0.02,
            treatment="sasrec_session(d=64,L=2,+vert)", control="sasrec(d=64,L=2,+vert)",
        ),
    ]
    return {
        "goal": "measure within-session adaptation separately from retraining",
        "trials": trials, "hypotheses": hyps,
        "why": "The log carries session ids, so in-session prefixes are reconstructable.",
    }


ACTIONS = {
    "baselines": _action_baselines,
    "cross_vertical_ablation": _action_cross_vertical,
    "graph_depth_and_kg": _action_graph_depth,
    "sequential": _action_sequential,
    "fusion": _action_fusion,
    "session_adaptation": _action_session,
}

ACTION_DOCS = {
    "baselines": "popularity floor + classical CF baselines",
    "cross_vertical_ablation": "one-factor test of cross-vertical graph edges",
    "graph_depth_and_kg": "knowledge-graph edges and walk depth",
    "sequential": "transformer sequential models, with/without vertical context",
    "fusion": "learned second-stage ranker over the surviving retrievers",
    "session_adaptation": "within-session re-encoding vs a batch-frozen representation",
}


def _slice_value(r: EvalResult, slice_name: str, metric: str) -> float:
    if slice_name == "all":
        return float(r.metrics.get(metric, float("nan")))
    return float(r.slice_metrics.get(slice_name, {}).get(metric, float("nan")))
