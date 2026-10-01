"""Fast checks on the parts that silently produce plausible-looking nonsense.

Every assertion here corresponds to a bug that actually shipped during
development and cost real debugging time. They run in a few seconds:

    python tests_smoke.py
"""
from __future__ import annotations

import numpy as np

from autoexp.datasets import load
from autoexp.evaluate import _rank_of_target, evaluate_model, mask_seen, paired_comparison
from autoexp.models.classical import ItemKNN, PopularityRecommender
from autoexp.models.graph import GraphWalkRetriever

FAILS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def test_ranking_is_pessimistic_on_ties():
    """A model with no signal must not score a perfect NDCG."""
    zeros = np.zeros((1, 50))
    check("all-tied row gets the worst rank, not rank 1",
          _rank_of_target(zeros, np.array([0]))[0] == 50)
    s = np.array([[9.0, 1.0, 2.0]])
    check("unique best gets rank 1", _rank_of_target(s, np.array([0]))[0] == 1)
    s = np.array([[5.0, 5.0, 5.0, 1.0]])
    check("3-way tie ranks last within the tie group",
          _rank_of_target(s, np.array([0]))[0] == 3)


def test_masking_keeps_target_rankable():
    d = load("careem_sim", n_users=600)
    v = d.verticals[0]
    cand, pos_map = d.candidates[v], d.cand_pos[v]
    rows = d.test[d.test.vertical == v].head(20)
    users = rows["u"].to_numpy()
    tgt = np.array([pos_map[int(i)] for i in rows["i"]])
    scores = np.random.default_rng(0).random((len(rows), len(cand)))
    out = mask_seen(scores, users, d, pos_map, tgt)
    check("target is never masked to -inf",
          np.isfinite(out[np.arange(len(rows)), tgt]).all())
    seen_masked = []
    for r, u in enumerate(users):
        for it in d.seqs.get(int(u), []):
            if int(it) in pos_map and pos_map[int(it)] != tgt[r]:
                seen_masked.append(out[r, pos_map[int(it)]] == -np.inf)
    check("every already-seen non-target item is masked",
          all(seen_masked), f"{sum(seen_masked)}/{len(seen_masked)}")


def test_transfer_ablation_is_monotone_control():
    """transfer_weight=0 must be *identical* to cross_vertical=False."""
    d = load("careem_sim", n_users=800)
    a = GraphWalkRetriever(cross_vertical=False, use_metadata=False).fit(d)
    b = GraphWalkRetriever(cross_vertical=True, transfer_weight=0.0,
                           use_metadata=False).fit(d)
    v = d.verticals[0]
    cand = d.candidates[v]
    users = d.test[d.test.vertical == v]["u"].to_numpy()[:100]
    check("cross_vertical=False == transfer_weight=0 (a true control arm)",
          np.allclose(a.score_users(users, cand), b.score_users(users, cand)))


def test_no_signal_users_are_not_rewarded():
    """A user with no in-vertical history must not beat a real model."""
    d = load("careem_sim", n_users=800)
    within = GraphWalkRetriever(cross_vertical=False, use_metadata=False).fit(d)
    ev = d.eval_frame()
    r = evaluate_model(within, d, ev)
    check("within-vertical arm scores below 1.0 overall", 0 < r.primary < 1.0,
          f"ndcg@10={r.primary}")
    check("no row achieves a perfect rank-1 on an empty score vector",
          (r.per_row["rank"] > 1).any())


def test_kg_edges_carry_the_requested_mass():
    d = load("careem_sim", n_users=800)
    w = 0.35
    g = GraphWalkRetriever(use_metadata=True, meta_weight=w).fit(d)
    ni = d.n_items
    share = np.asarray(g.T_.tocsr()[:ni][:, ni:].sum(1)).ravel().mean()
    check(f"meta_weight={w} really sends ~{w} of outgoing mass to KG nodes",
          abs(share - w) < 0.02, f"measured {share:.3f}")


def test_paired_comparison_slices():
    d = load("careem_sim", n_users=800)
    ev = d.eval_frame()
    a = evaluate_model(ItemKNN(top_k=50).fit(d), d, ev)
    b = evaluate_model(PopularityRecommender().fit(d), d, ev)
    full = paired_comparison(a, b)
    sub = paired_comparison(a, b, slice_name="transfer_only")
    check("paired comparison is paired (equal n to eval rows)", full["n"] == len(ev),
          f"{full['n']} vs {len(ev)}")
    check("slice restricts the comparison", 0 < sub["n"] < full["n"],
          f"slice n={sub['n']}, full n={full['n']}")


def test_nonfinite_scores_are_refused():
    class Broken(PopularityRecommender):
        def score_users(self, users, candidates):
            out = super().score_users(users, candidates)
            out[0, 0] = np.nan
            return out

    d = load("careem_sim", n_users=600)
    m = Broken().fit(d)
    m.name = "broken"
    try:
        evaluate_model(m, d, d.eval_frame())
        check("evaluator refuses NaN scores", False, "it accepted them")
    except ValueError as exc:
        check("evaluator refuses NaN scores", "non-finite" in str(exc))


def test_no_harm_hypothesis_is_not_refuted_by_improvement():
    """A no-harm claim must not be refuted by the metric moving *up*."""
    from autoexp.evaluate import PRIMARY_METRIC
    from autoexp.planner import Evidence, Planner
    from autoexp.types import Hypothesis

    pl = Planner()
    h = Hypothesis(
        id="H-noharm", round=1, statement="does not degrade", rationale="guardrail",
        metric=PRIMARY_METRIC, slice="all", target_delta=-0.02,
        treatment="t", control="c", direction="no_harm",
    )
    # +3.7% with an interval entirely above the -2% bar.
    cmp = {"n": 4104, "n_informative": 61, "delta": 0.0003, "rel_lift_pct": 3.7,
           "ci_low": 0.0, "ci_high": 0.0007, "rel_ci_low": 0.1, "rel_ci_high": 8.0,
           "significant": False, "baseline": 0.008}
    verdict = _verdict_for(pl, h, cmp)
    check("+3.7% against a -2% no-harm bar is SUPPORTED, not refuted",
          verdict == "supported", f"got {verdict}")

    # A genuine regression: the whole interval sits below the bar.
    cmp2 = dict(cmp, rel_ci_low=-30.0, rel_ci_high=-12.0, rel_lift_pct=-20.0)
    check("a real regression past the bar is REFUTED",
          _verdict_for(pl, h, cmp2) == "refuted")

    # Improvement claim whose interval straddles the bar.
    h2 = Hypothesis(id="H-up", round=1, statement="improves", rationale="",
                    metric=PRIMARY_METRIC, slice="all", target_delta=0.05,
                    treatment="t", control="c")
    cmp3 = dict(cmp, rel_ci_low=-4.3, rel_ci_high=2.1, rel_lift_pct=-1.9)
    check("an interval entirely below a +5% bar is REFUTED",
          _verdict_for(pl, h2, cmp3) == "refuted")
    cmp4 = dict(cmp, rel_ci_low=-1.0, rel_ci_high=12.0, rel_lift_pct=5.5)
    check("an interval straddling the bar is INCONCLUSIVE",
          _verdict_for(pl, h2, cmp4) == "inconclusive")


def _verdict_for(planner, h, cmp):
    """Drive Planner.judge against a canned comparison."""
    import autoexp.planner as P
    from autoexp.evaluate import EvalResult
    import pandas as pd

    ev = P.Evidence(profile={})
    ev.results = {
        "t": EvalResult("t", {}, {}, pd.DataFrame(), family="x"),
        "c": EvalResult("c", {}, {}, pd.DataFrame(), family="y"),
    }
    h.verdict = None
    ev.hypotheses = [h]
    real = P.paired_comparison
    P.paired_comparison = lambda *a, **k: cmp
    try:
        planner.judge(ev, h.round)
    finally:
        P.paired_comparison = real
    return h.verdict


def test_pruning_protects_future_control_arms():
    """A weak arm in round 1 can still be a later experiment's control."""
    from autoexp.evaluate import EvalResult
    from autoexp.planner import ACTION_REQUIRES, Evidence, Planner

    pl = Planner(max_rounds=4)
    pl.executed_actions = {"baselines"}          # round 1 has run, nothing else
    ev = Evidence(profile={})
    ev.results = {
        "item_knn(k=150)": EvalResult("item_knn(k=150)", {"ndcg@10": 0.0126}, {},
                                      __import__("pandas").DataFrame(), family="item_knn"),
        "graph_walk(within,L=3)": EvalResult("graph_walk(within,L=3)", {"ndcg@10": 0.0072}, {},
                                             __import__("pandas").DataFrame(), family="graph_walk"),
    }
    pl.prune(ev)
    check("graph_walk survives round-1 pruning while its own test is pending",
          "graph_walk" not in ev.pruned_families,
          f"pruned: {ev.pruned_families}")

    pl.executed_actions |= {"cross_vertical_ablation", "graph_depth_and_kg"}
    pl.prune(ev)
    check("graph_walk becomes prunable once its experiments have run",
          "graph_walk" in ev.pruned_families)


if __name__ == "__main__":
    for fn in [
        test_ranking_is_pessimistic_on_ties,
        test_masking_keeps_target_rankable,
        test_transfer_ablation_is_monotone_control,
        test_no_signal_users_are_not_rewarded,
        test_kg_edges_carry_the_requested_mass,
        test_paired_comparison_slices,
        test_nonfinite_scores_are_refused,
        test_pruning_protects_future_control_arms,
        test_no_harm_hypothesis_is_not_refuted_by_improvement,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    raise SystemExit(1 if FAILS else 0)
