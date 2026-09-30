"""Turning an offline win into a shippable decision.

An offline NDCG delta is not a launch decision. Three things stand between
them, and this module does all three:

1. **Interleaving.** Team-draft interleaving (Chapelle et al.) puts both
   rankers in front of the *same* user in the *same* slate, so the comparison
   is within-user and immune to the traffic-split variance that makes
   small-effect A/B tests so expensive. It is the standard first online test
   for a ranker change, and it typically needs an order of magnitude less
   traffic than a split test.

2. **A simulated online A/B.** On the simulator we know each user's true
   latent utility and the position-bias curve, so we can actually *serve* both
   rankers, simulate clicks, and measure realised CTR - the metric the
   business cares about - rather than assuming offline NDCG translates.

3. **Power analysis.** The number that decides whether the experiment is worth
   running at all: given the observed effect and variance, how many users are
   needed, and how long that takes at a stated traffic level.

On real data (Amazon) there is no ground-truth click model, so only the
paired offline estimate and the power analysis are reported - and the report
says so rather than dressing a simulation up as a measured result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .evaluate import EvalResult, mask_seen


@dataclass
class ABResult:
    champion: str
    challenger: str
    method: str
    metric: str
    control_mean: float
    treatment_mean: float
    abs_lift: float
    rel_lift_pct: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool
    n_units: int
    required_n_per_arm: Optional[int] = None
    days_to_significance: Optional[float] = None
    #: Mean *true* latent utility of the whole top-k slate, where ground truth
    #: exists. NDCG measures where one held-out item landed; this measures how
    #: good the other nine slots were. They can disagree, and when they do the
    #: disagreement is the finding.
    slate_utility_control: Optional[float] = None
    slate_utility_treatment: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}


# ------------------------------------------------------------------ power
def required_sample_size(effect: float, sd: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Per-arm n for a two-sided two-sample test of a difference in means."""
    from scipy import stats

    if effect == 0 or not np.isfinite(effect) or sd <= 0:
        return -1
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_a + z_b) ** 2) * (sd ** 2) / (effect ** 2)))


def power_analysis(a: EvalResult, b: EvalResult, metric: str = "ndcg@10",
                   daily_users: int = 200_000, exposure_frac: float = 1.0,
                   alpha: float = 0.05, power: float = 0.8) -> Dict[str, object]:
    """Size the online test implied by an offline paired difference."""
    key = ["u", "i"]
    m = a.per_row[key + [metric]].merge(b.per_row[key + [metric]], on=key,
                                        suffixes=("_a", "_b"))
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    if len(d) == 0:
        return {"error": "no overlapping rows"}
    effect = float(d.mean())
    # Unpaired sizing is the conservative choice: a split test does not get the
    # within-user pairing that makes the offline delta look so precise.
    sd = float(np.std(np.concatenate([m[f"{metric}_a"], m[f"{metric}_b"]])))
    n = required_sample_size(effect, sd, alpha, power)
    daily_exposed = max(daily_users * exposure_frac, 1)
    return {
        "effect": effect,
        "pooled_sd": sd,
        "required_n_per_arm": n,
        "days_to_significance": round(2 * n / daily_exposed, 2) if n > 0 else None,
        "assumed_daily_users": daily_users,
        "alpha": alpha,
        "power": power,
    }


# ----------------------------------------------------------- interleaving
def team_draft_interleave(scores_a: np.ndarray, scores_b: np.ndarray,
                          k: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Build one interleaved slate; return (items, which ranker contributed)."""
    order_a = np.argsort(-scores_a)
    order_b = np.argsort(-scores_b)
    pa = pb = 0
    chosen: List[int] = []
    owner: List[int] = []
    seen = set()
    while len(chosen) < k:
        # Coin flip decides who picks next; ties in team size are broken fairly.
        pick_a = rng.random() < 0.5
        if pick_a:
            while pa < len(order_a) and order_a[pa] in seen:
                pa += 1
            if pa >= len(order_a):
                break
            chosen.append(int(order_a[pa])); owner.append(0); seen.add(int(order_a[pa])); pa += 1
        else:
            while pb < len(order_b) and order_b[pb] in seen:
                pb += 1
            if pb >= len(order_b):
                break
            chosen.append(int(order_b[pb])); owner.append(1); seen.add(int(order_b[pb])); pb += 1
    return np.array(chosen), np.array(owner)


def interleaving_test(model_a, model_b, data, eval_df: pd.DataFrame, k: int = 10,
                      seed: int = 0, n_boot: int = 2000) -> ABResult:
    """Offline interleaving: credit whichever ranker contributed the item the
    user actually went on to interact with.

    The per-user outcome is +1 if the held-out target came from A's team, -1 if
    from B's, 0 if the target was not in the interleaved slate. Bootstrapping
    that per-user score gives a within-user comparison with a CI.
    """
    rng = np.random.default_rng(seed)
    outcomes: List[float] = []

    for vertical, grp in eval_df.groupby("vertical", sort=False):
        cand = data.candidates[vertical]
        pos_map = data.cand_pos[vertical]
        grp = grp.reset_index(drop=True)
        users = grp["u"].to_numpy()
        tgt = np.array([pos_map[int(i)] for i in grp["i"]])

        sa = np.asarray(model_a.score_rows(grp, cand) if model_a.session_aware
                        else model_a.score_users(users, cand), dtype=np.float64)
        sb = np.asarray(model_b.score_rows(grp, cand) if model_b.session_aware
                        else model_b.score_users(users, cand), dtype=np.float64)
        sa = mask_seen(sa, users, data, pos_map, tgt)
        sb = mask_seen(sb, users, data, pos_map, tgt)

        for r in range(len(grp)):
            items, owner = team_draft_interleave(sa[r], sb[r], k, rng)
            hit = np.where(items == tgt[r])[0]
            outcomes.append(0.0 if len(hit) == 0 else (1.0 if owner[hit[0]] == 0 else -1.0))

    out = np.array(outcomes)
    boots = out[rng.integers(0, len(out), size=(n_boot, len(out)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    wins = float((out > 0).sum())
    losses = float((out < 0).sum())
    return ABResult(
        champion=model_b.name, challenger=model_a.name, method="team_draft_interleaving",
        metric="per-user preference (+1 challenger / -1 champion)",
        control_mean=0.0, treatment_mean=float(out.mean()),
        abs_lift=float(out.mean()),
        rel_lift_pct=float(100 * (wins - losses) / max(wins + losses, 1)),
        ci_low=float(lo), ci_high=float(hi), p_value=float(min(1.0, p)),
        significant=bool(lo > 0 or hi < 0), n_units=int(len(out)),
        notes=[
            f"{int(wins)} slates credited to the challenger, {int(losses)} to the champion, "
            f"{int((out == 0).sum())} ties (target not in the interleaved top-{k}).",
            "Interleaving is within-user, so it needs far less traffic than a split test "
            "to reach the same power - but it measures preference, not absolute CTR.",
        ],
    )


# ------------------------------------------------- simulated online A/B test
def simulated_ab_test(model_a, model_b, data, sim, eval_df: pd.DataFrame,
                      k: int = 10, seed: int = 0, n_boot: int = 2000,
                      daily_users: int = 200_000) -> ABResult:
    """Serve both rankers to randomly assigned users and simulate clicks.

    Only valid on the simulator, where the true latent utility and the
    position-bias curve are known. Users - not requests - are the randomisation
    unit, which is what a real assignment would do.
    """
    rng = np.random.default_rng(seed)

    rows = eval_df.reset_index(drop=True)
    assign = rng.random(len(rows)) < 0.5      # True -> treatment (model_a)
    ctr = {0: [], 1: []}
    slate_util = {0: [], 1: []}

    # Encoded id -> simulator index, built once. Doing this with .loc per row
    # made the test take minutes instead of seconds.
    u_to_sim = {int(u): int(str(uid)[1:])
                for u, uid in data.events.drop_duplicates("u")[["u", "user_id"]].to_numpy()}
    i_to_sim = {int(i): int(str(iid)[1:])
                for i, iid in data.items[["i", "item_id"]].to_numpy()}

    for vertical, grp in rows.groupby("vertical", sort=False):
        cand = data.candidates[vertical]
        pos_map = data.cand_pos[vertical]
        idx = grp.index.to_numpy()
        users = grp["u"].to_numpy()
        tgt = np.array([pos_map[int(i)] for i in grp["i"]])

        sa = np.asarray(model_a.score_rows(grp, cand) if model_a.session_aware
                        else model_a.score_users(users, cand), dtype=np.float64)
        sb = np.asarray(model_b.score_rows(grp, cand) if model_b.session_aware
                        else model_b.score_users(users, cand), dtype=np.float64)
        sa = mask_seen(sa, users, data, pos_map, tgt)
        sb = mask_seen(sb, users, data, pos_map, tgt)

        for r in range(len(grp)):
            arm = 1 if assign[idx[r]] else 0
            scores = sa[r] if arm == 1 else sb[r]
            slate_local = np.argsort(-scores)[:k]
            slate_items = cand[slate_local]
            sim_u = u_to_sim[int(users[r])]
            sim_items = np.array([i_to_sim[int(x)] for x in slate_items])
            rel = sim._relevance(sim_u, sim_items, vertical)
            # Exactly the click model the log was generated from - imported,
            # not re-implemented, so the two can never drift apart.
            clicks = (rng.random(len(rel)) < sim.click_probabilities(rel)).sum()
            ctr[arm].append(float(clicks) / k)
            slate_util[arm].append(float(rel.mean()))

    a = np.array(ctr[1]); b = np.array(ctr[0])
    su_t = float(np.mean(slate_util[1])) if slate_util[1] else None
    su_c = float(np.mean(slate_util[0])) if slate_util[0] else None
    d_boot = np.array([
        a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    p = 2 * min((d_boot <= 0).mean(), (d_boot >= 0).mean())
    effect = float(a.mean() - b.mean())
    sd = float(np.std(np.concatenate([a, b])))
    n_req = required_sample_size(effect, sd)
    return ABResult(
        champion=model_b.name, challenger=model_a.name,
        method="simulated_online_ab (user-randomised, position-biased click model)",
        metric="CTR@%d" % k,
        control_mean=float(b.mean()), treatment_mean=float(a.mean()),
        abs_lift=effect,
        rel_lift_pct=float(100 * effect / b.mean()) if b.mean() > 0 else float("nan"),
        ci_low=float(lo), ci_high=float(hi), p_value=float(min(1.0, p)),
        significant=bool(lo > 0 or hi < 0),
        n_units=int(len(a) + len(b)),
        required_n_per_arm=n_req,
        days_to_significance=round(2 * n_req / daily_users, 2) if n_req > 0 else None,
        slate_utility_control=su_c, slate_utility_treatment=su_t,
        notes=[
            (f"Mean true utility of the whole top-{k} slate: "
             f"{su_c:+.4f} (control) vs {su_t:+.4f} (challenger). NDCG scores where "
             f"one held-out item landed; this scores all {k} slots.")
            if (su_c is not None and su_t is not None) else "",
            "Clicks are simulated from the generator's true latent utility and its "
            "position-bias curve. This is an upper bound on what an online test would "
            "show: the click model is exactly the one the data came from.",
            f"Randomisation unit: user ({len(a)} treatment / {len(b)} control).",
        ],
    )
