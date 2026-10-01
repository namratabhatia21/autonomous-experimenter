"""The loop: profile, plan, run, judge, adapt, decide, narrate.

This is the part that makes the system an experimenter rather than a
benchmark script. Each round it asks the planner what to run, runs only that,
scores the round's hypotheses against thresholds written down beforehand,
prunes what is out of contention, and feeds the verdicts back into the next
round's plan. It stops when the leaderboard plateaus rather than when a fixed
schedule runs out.

When the rounds end, the champion is put through the decision layer -
interleaving, a simulated A/B where a ground-truth click model exists, and a
power analysis - because an offline NDCG win is not a launch recommendation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import registry
from .abtest import ABResult, interleaving_test, power_analysis, simulated_ab_test
from .evaluate import PRIMARY_METRIC, EvalResult, evaluate_model, leaderboard, paired_comparison
from .planner import Evidence, Planner
from .types import Hypothesis, RoundPlan, Trial

#: Families that count as "retrieval" when a hypothesis asks for the best retriever.
RETRIEVAL_FAMILIES = {"item_knn", "bpr_mf", "graph_walk",
                      "cross_vertical_bridge", "sasrec", "sasrec_session"}


@dataclass
class RunResult:
    run_id: str
    dataset: str
    profile: Dict[str, Any]
    rounds: List[RoundPlan]
    results: Dict[str, EvalResult]
    champion: Optional[str]
    baseline: str
    decision: Dict[str, Any]
    hypotheses: List[Hypothesis]
    notes: List[str]
    planner_decisions: List[Dict[str, Any]]
    elapsed: float
    stopped_because: str
    narrative: str = ""
    narrator_engine: str = "none"

    def leaderboard(self) -> pd.DataFrame:
        return leaderboard(list(self.results.values()))

    def to_json(self) -> Dict[str, Any]:
        from dataclasses import asdict

        def clean(x):
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, (np.bool_,)):
                return bool(x)
            return x

        return {
            "run_id": self.run_id,
            "dataset": self.dataset,
            "profile": self.profile,
            "elapsed_seconds": round(self.elapsed, 1),
            "stopped_because": self.stopped_because,
            "champion": self.champion,
            "baseline": self.baseline,
            "narrator_engine": self.narrator_engine,
            "rounds": [
                {
                    "round": r.round, "goal": r.goal, "actions": r.actions,
                    "reflection": r.reflection,
                    "trials": [{"id": t.id, "kind": t.family, "params": t.params,
                                "motivation": t.motivation} for t in r.trials],
                    "judgements": r.judgements, "decisions": r.decisions,
                }
                for r in self.rounds
            ],
            "hypotheses": [
                {k: clean(v) for k, v in asdict(h).items()} for h in self.hypotheses
            ],
            "leaderboard": json.loads(self.leaderboard().to_json(orient="records")),
            "decision": self.decision,
            "planner_decisions": self.planner_decisions,
            "notes": self.notes,
            "narrative": self.narrative,
        }


class AutonomousExperimenter:
    def __init__(self, data, max_rounds: int = 4, time_budget: float = 1800.0,
                 llm_planner: bool = False, llm_client=None, seed: int = 0,
                 verbose: bool = True):
        self.data = data
        self.planner = Planner(max_rounds=max_rounds, llm_planner=llm_planner,
                               llm_client=llm_client, seed=seed)
        self.time_budget = time_budget
        self.seed = seed
        self.verbose = verbose
        self.models: Dict[str, Any] = {}

    # ------------------------------------------------------------------ util
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def profile(self) -> Dict[str, Any]:
        s = self.data.summary()
        ev = self.data.eval_frame()
        s["has_sessions"] = "session_id" in self.data.events.columns
        s["eval_rows"] = int(len(ev))
        for col in ("cold_in_vertical", "transfer_only", "cross_vertical_user", "tail_target"):
            s[f"pct_{col}"] = round(100 * float(ev[col].mean()), 1)
        s["source"] = self.data.source
        if self.data.ground_truth:
            s["ground_truth"] = self.data.ground_truth
        return s

    # -------------------------------------------------------------- resolve
    def _resolve(self, name: str, ev: Evidence) -> Optional[str]:
        """Map a hypothesis arm name onto an actual result key.

        Hypotheses are written before the models exist, so they refer to arms
        by intent - ``two_stage(+transfer)`` or ``__best_retriever__`` - and
        this resolves that to whatever the runner actually produced.
        """
        if name in ev.results:
            return name
        if name == "popularity":
            return "popularity" if "popularity" in ev.results else None
        if name == "__best_non_popularity__":
            pool = [r for n, r in ev.results.items() if n != "popularity" and not r.error]
            return max(pool, key=lambda r: r.primary).model if pool else None
        if name == "__best_retriever__":
            pool = [r for r in ev.results.values()
                    if r.family in RETRIEVAL_FAMILIES and not r.error]
            return max(pool, key=lambda r: r.primary).model if pool else None
        if name == "__best_two_stage__":
            pool = [r for r in ev.results.values() if r.family == "two_stage" and not r.error]
            return max(pool, key=lambda r: r.primary).model if pool else None
        # Fall back to a token-subset match: "two_stage(+transfer)" should find
        # "two_stage(3src,+transfer)" without the planner needing to know how
        # many sources survived pruning.
        tokens = [t for t in name.replace("(", ",").replace(")", "").split(",") if t]
        hits = [n for n in ev.results if all(t in n for t in tokens)]
        return hits[0] if len(hits) == 1 else None

    # ------------------------------------------------------------------ run
    def run(self) -> RunResult:
        t0 = time.monotonic()
        run_id = time.strftime("%Y%m%d-%H%M%S")
        prof = self.profile()
        ev = Evidence(profile=prof)
        eval_df = self.data.eval_frame()

        self._log(f"\n{'=' * 74}\nAUTONOMOUS EXPERIMENTER  |  dataset: {self.data.name}")
        self._log(f"{prof['n_users']:,} users x {prof['n_items']:,} items, "
                  f"{prof['n_interactions']:,} interactions, "
                  f"{prof['multi_vertical_user_pct']}% multi-vertical")
        self._log(f"eval rows {prof['eval_rows']:,} | "
                  f"transfer_only {prof['pct_transfer_only']}% | "
                  f"tail {prof['pct_tail_target']}%\n{'=' * 74}")

        rounds: List[RoundPlan] = []
        best_history: List[float] = []
        stopped = "completed all planned rounds"

        for round_no in range(1, self.planner.max_rounds + 1):
            if time.monotonic() - t0 > self.time_budget:
                stopped = f"time budget of {self.time_budget:.0f}s exhausted"
                break

            plan = self.planner.plan_round(round_no, ev)
            if plan is None:
                stopped = "the planner had no further hypothesis worth testing"
                break

            self._log(f"\n--- ROUND {round_no}: {plan.goal}")
            if plan.reflection:
                self._log(f"    why: {plan.reflection}")
            for h in plan.hypotheses:
                self._log(f"    [{h.id}] {h.statement}")
            ev.hypotheses.extend(plan.hypotheses)

            self._run_trials(plan, ev, eval_df)
            self._resolve_hypotheses(plan, ev)

            plan.judgements = self.planner.judge(ev, round_no)
            for j in plan.judgements:
                self._log(f"    -> {j}")

            plan.decisions = self.planner.prune(ev) + self.planner.guardrails(ev)
            for d in plan.decisions:
                self._log(f"    ! {d}")

            rounds.append(plan)
            best = ev.best()
            if best:
                best_history.append(best.primary)
                ev.champion = best.model
                self._log(f"    leader: {best.model}  {PRIMARY_METRIC}={best.primary:.4f}")

            stop = self.planner.should_stop(ev, best_history, round_no + 1)
            if stop:
                stopped = stop
                self._log(f"    STOP: {stop}")
                break

        champion = ev.champion
        decision = self._decide(champion, ev, eval_df)

        return RunResult(
            run_id=run_id, dataset=self.data.name, profile=prof, rounds=rounds,
            results=ev.results, champion=champion,
            baseline="popularity" if "popularity" in ev.results else "",
            decision=decision, hypotheses=ev.hypotheses, notes=ev.notes,
            planner_decisions=self.planner.decisions,
            elapsed=time.monotonic() - t0, stopped_because=stopped,
        )

    # --------------------------------------------------------------- trials
    def _run_trials(self, plan: RoundPlan, ev: Evidence, eval_df: pd.DataFrame) -> None:
        for trial in plan.trials:
            if trial.family in ev.pruned_families:
                ev.notes.append(f"{trial.id} skipped: family '{trial.family}' was pruned")
                continue
            try:
                t0 = time.perf_counter()
                retr = self._surviving_retrievers(ev) if trial.family == "two_stage" else None
                model = registry.build(trial.family, trial.params, retrievers=retr)
                model.fit(self.data)
                fit_s = time.perf_counter() - t0

                result = evaluate_model(model, self.data, eval_df)
                result.fit_seconds = fit_s
                ev.results[model.name] = result
                self.models[model.name] = model
                trial.label = model.name

                tr = result.slice_metrics.get("transfer_only", {}).get(PRIMARY_METRIC)
                self._log(
                    f"    {model.name:34s} {PRIMARY_METRIC}={result.primary:.4f}"
                    + (f"  transfer_only={tr:.4f}" if tr is not None else "")
                    + f"  cov={result.metrics.get('coverage@10', 0):.3f}  {fit_s:.0f}s"
                )
            except Exception as exc:                       # noqa: BLE001
                # A failed arm is data, not a crash: record it and keep going,
                # so one bad configuration cannot take down the whole run.
                ev.results[f"{trial.family}#{trial.id}"] = EvalResult(
                    model=f"{trial.family}#{trial.id}", metrics={}, slice_metrics={},
                    per_row=pd.DataFrame(), family=trial.family, error=str(exc),
                )
                ev.notes.append(f"{trial.id} ({trial.family}) failed: {exc}")
                self._log(f"    {trial.family:34s} FAILED: {exc}")

    def _surviving_retrievers(self, ev: Evidence, top_k: int = 3) -> List[Any]:
        """Fuse the best *diverse* retrievers: one per family, best first.

        Taking the global top-3 would often pick three variants of the same
        graph walk, which gives the ranker three copies of one opinion.
        """
        by_family: Dict[str, EvalResult] = {}
        for r in ev.results.values():
            if r.error or r.family not in RETRIEVAL_FAMILIES:
                continue
            if r.family in ev.pruned_families:
                continue
            if r.family not in by_family or r.primary > by_family[r.family].primary:
                by_family[r.family] = r
        chosen = sorted(by_family.values(), key=lambda r: -r.primary)[:top_k]
        return [self.models[r.model] for r in chosen if r.model in self.models]

    def _resolve_hypotheses(self, plan: RoundPlan, ev: Evidence) -> None:
        for h in plan.hypotheses:
            if h.treatment:
                h.treatment = self._resolve(h.treatment, ev) or h.treatment
            if h.control:
                h.control = self._resolve(h.control, ev) or h.control

    # -------------------------------------------------------------- decision
    def _decide(self, champion: Optional[str], ev: Evidence,
                eval_df: pd.DataFrame) -> Dict[str, Any]:
        """Turn the offline leaderboard into something a team can act on."""
        out: Dict[str, Any] = {}
        if champion is None or champion not in self.models:
            return {"error": "no champion produced"}

        incumbent = "popularity" if "popularity" in self.models else None
        best_classical = max(
            (r for r in ev.results.values()
             if r.family in {"item_knn", "bpr_mf"} and not r.error),
            key=lambda r: r.primary, default=None,
        )
        comparators = [n for n in
                       {incumbent, best_classical.model if best_classical else None}
                       if n and n != champion]

        out["champion"] = champion
        out["offline"] = {}
        for c in comparators:
            out["offline"][c] = paired_comparison(
                ev.results[champion], ev.results[c], seed=self.seed
            )
            out["offline"][c]["power"] = power_analysis(
                ev.results[champion], ev.results[c]
            )

        if comparators:
            ref = comparators[0]
            self._log(f"\n--- DECISION LAYER: {champion} vs {ref}")
            try:
                il = interleaving_test(self.models[champion], self.models[ref],
                                       self.data, eval_df, seed=self.seed)
                out["interleaving"] = il.to_dict()
                self._log(f"    interleaving: {il.abs_lift:+.4f} "
                          f"CI[{il.ci_low:+.4f},{il.ci_high:+.4f}] sig={il.significant}")
            except Exception as exc:                        # noqa: BLE001
                out["interleaving"] = {"error": str(exc)}

            sim = getattr(self.data, "sim", None)
            if sim is not None:
                try:
                    ab = simulated_ab_test(self.models[champion], self.models[ref],
                                           self.data, sim, eval_df, seed=self.seed)
                    out["simulated_ab"] = ab.to_dict()
                    self._log(f"    simulated A/B CTR: {ab.control_mean:.4f} -> "
                              f"{ab.treatment_mean:.4f} ({ab.rel_lift_pct:+.1f}%) "
                              f"sig={ab.significant}, needs {ab.required_n_per_arm:,}/arm")
                except Exception as exc:                    # noqa: BLE001
                    out["simulated_ab"] = {"error": str(exc)}
            else:
                out["simulated_ab"] = {
                    "skipped": "no ground-truth click model on a real dataset; "
                               "only the offline paired estimate and the power "
                               "analysis are reported."
                }

        out["conflicts"] = _reconcile(out)
        for c in out["conflicts"]:
            self._log(f"    ?? {c}")

        # The interpretable transfer artefact, if that model was ever built.
        for name, m in self.models.items():
            if m.family == "cross_vertical_bridge" and hasattr(m, "top_transfers"):
                try:
                    out["top_cross_vertical_signals"] = [
                        {"from_category": a, "from_vertical": va,
                         "to_category": b, "to_vertical": vb, "log_lift": round(w, 3)}
                        for a, va, b, vb, w in m.top_transfers(10)
                    ]
                except Exception:                           # noqa: BLE001
                    pass

        champ_model = self.models.get(champion)
        if champ_model is not None and hasattr(champ_model, "feature_importance"):
            try:
                fi = champ_model.feature_importance()
                out["feature_importance"] = json.loads(
                    fi.head(12).to_json(orient="records")
                )
            except Exception:                               # noqa: BLE001
                pass
        return out


def _reconcile(decision: Dict[str, Any]) -> List[str]:
    """Flag the cases where two measurements of the same change disagree.

    A readout that reports an offline win and an online loss without naming the
    contradiction is how a team ships a regression. On this project the two
    genuinely diverged: the graph walk won NDCG@10 by a wide margin and *lost*
    simulated CTR, because NDCG scores where a single held-out item landed while
    CTR scores all ten slots. Surfacing it is the whole point of having both.
    """
    msgs: List[str] = []
    il = decision.get("interleaving") or {}
    ab = decision.get("simulated_ab") or {}
    off = decision.get("offline") or {}

    if il and "error" not in il and ab and "error" not in ab and "skipped" not in ab:
        if np.sign(il.get("abs_lift", 0)) != np.sign(ab.get("abs_lift", 0)):
            msg = (
                "CONFLICT: interleaving and the simulated A/B disagree in sign "
                f"(preference {il.get('abs_lift', 0):+.4f} vs CTR "
                f"{ab.get('rel_lift_pct', 0):+.1f}%)."
            )
            su_c, su_t = ab.get("slate_utility_control"), ab.get("slate_utility_treatment")
            if su_c is not None and su_t is not None and su_t < su_c:
                msg += (
                    f" Mean true utility of the full slate is lower for the challenger "
                    f"({su_t:+.4f} vs {su_c:+.4f}), so the champion is better at placing "
                    f"the single next item near the top and worse at filling the rest of "
                    f"the slate. Ranking-metric wins do not transfer to slate-level "
                    f"engagement automatically; do not launch on the offline number alone."
                )
            msgs.append(msg)

    for ref, c in off.items():
        if c.get("significant") and ab and "error" not in ab and "skipped" not in ab:
            if not ab.get("significant") and np.sign(c.get("delta", 0)) == np.sign(ab.get("abs_lift", 0)):
                msgs.append(
                    f"UNDERPOWERED: the offline gain over `{ref}` is significant, but the "
                    f"online estimate is not at this sample size. A real test needs "
                    f"~{ab.get('required_n_per_arm', 0):,} users per arm."
                )
    return msgs
