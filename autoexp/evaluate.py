"""Ranking evaluation, slices, and paired statistics.

Three commitments here, because they are what separate a real bake-off from a
leaderboard screenshot:

1. **Every arm sees identical candidate sets and identical masking.** The
   evaluator - not the model - excludes already-seen items, so no arm can win
   by filtering more aggressively.

2. **Per-row metrics are retained, not just means.** Every comparison is
   therefore *paired* at the user level, which is both far more powerful than
   comparing two means and the only correct way to test two rankers scored on
   the same users.

3. **Accuracy is not the only axis.** Coverage and novelty are reported
   alongside NDCG, because a model that wins NDCG by serving the same 50 head
   items to everyone is a catalogue-collapse risk, not a win.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

K_VALUES = (5, 10, 20)
PRIMARY_METRIC = "ndcg@10"

#: Slices the agent breaks every result down by. ``transfer_only`` is the one
#: that matters most for the cross-vertical thesis.
SLICES = {
    "all": lambda df: np.ones(len(df), dtype=bool),
    "cold_in_vertical": lambda df: df["cold_in_vertical"].to_numpy(),
    "warm_in_vertical": lambda df: ~df["cold_in_vertical"].to_numpy(),
    "transfer_only": lambda df: df["transfer_only"].to_numpy(),
    "single_vertical_user": lambda df: ~df["cross_vertical_user"].to_numpy(),
    "tail_target": lambda df: df["tail_target"].to_numpy(),
}


@dataclass
class EvalResult:
    model: str
    metrics: Dict[str, float]                       # overall means
    slice_metrics: Dict[str, Dict[str, float]]      # slice -> metric -> mean
    per_row: pd.DataFrame                           # u, vertical, rank, ndcg@10, ...
    fit_seconds: float = 0.0
    score_seconds: float = 0.0
    params: Dict[str, object] = field(default_factory=dict)
    family: str = ""
    error: str = ""

    @property
    def primary(self) -> float:
        return float(self.metrics.get(PRIMARY_METRIC, np.nan))


# --------------------------------------------------------------------- core
def mask_seen(scores: np.ndarray, users: np.ndarray, data,
              pos_map: Dict[int, int], target_pos: np.ndarray) -> np.ndarray:
    """Remove already-seen items from contention, keeping the target rankable.

    This lives in exactly one place on purpose. An earlier version of the
    two-stage ranker did its own model selection without this masking, and the
    two protocols silently disagreed: a random walk restarts on the user's own
    history, so unmasked it returns items the user already has and looks like
    the *worst* retriever, while masked it is the best. Validation picked the
    wrong prior on the strength of that discrepancy. Any code that ranks
    anything must call this.
    """
    scores = np.array(scores, dtype=np.float64, copy=True)
    saved = scores[np.arange(len(scores)), target_pos].copy()
    for r, u in enumerate(users):
        seq = data.seqs.get(int(u))
        if seq is None or len(seq) == 0:
            continue
        hits = [pos_map[int(x)] for x in seq if int(x) in pos_map]
        if hits:
            scores[r, hits] = -np.inf
    scores[np.arange(len(scores)), target_pos] = saved
    return scores


def ndcg_at_k(scores: np.ndarray, target_pos: np.ndarray, k: int = 10) -> np.ndarray:
    rank = _rank_of_target(scores, target_pos)
    return np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)


def _rank_of_target(scores: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """1-based rank of each target within its row of ``scores``.

    Ties are broken **pessimistically**: the target is placed below everything
    it ties with, so ``rank = (strictly better) + (size of its tie group)``.

    Counting only strictly-better candidates instead - ``(scores > tgt) + 1`` -
    is optimistic, and silently rewards exactly the models that deserve it
    least. A model that returns an all-zero score row (a user it has no signal
    for) ties with every candidate and would be handed **rank 1, a perfect
    NDCG**. That inflated the popularity baseline, which has large tie groups
    by construction, and it inflated the transfer control arm on every user it
    had no in-vertical history for. Pessimistic handling gives those rows the
    worst rank in the tie group, which is what "no signal" actually means.
    """
    tgt = scores[np.arange(len(scores)), target_pos]
    strictly_better = (scores > tgt[:, None]).sum(1)
    tie_group = (scores == tgt[:, None]).sum(1)      # includes the target itself
    return strictly_better + tie_group


def evaluate_model(model, data, eval_df: pd.DataFrame,
                   ks: Sequence[int] = K_VALUES,
                   batch_rows: int = 4000) -> EvalResult:
    """Score one arm across every vertical and return per-row metrics."""
    import time

    rows_out = []
    topk_items: List[np.ndarray] = []
    t0 = time.perf_counter()

    for vertical, grp in eval_df.groupby("vertical", sort=False):
        cand = data.candidates[vertical]
        pos_map = data.cand_pos[vertical]
        grp = grp.reset_index(drop=True)

        for b in range(0, len(grp), batch_rows):
            chunk = grp.iloc[b:b + batch_rows]
            if model.session_aware:
                scores = model.score_rows(chunk, cand)
            else:
                scores = model.score_users(chunk["u"].to_numpy(), cand)
            scores = np.asarray(scores, dtype=np.float64)
            # A model that emits NaN (diverged training, degenerate embedding)
            # would otherwise beat every honest arm: NaN fails every `>`
            # comparison, so the target lands at rank 1 and scores a perfect
            # NDCG. Fail loudly instead of crowning a broken model.
            if not np.isfinite(scores).all():
                bad = int((~np.isfinite(scores)).sum())
                raise ValueError(
                    f"{model.name} produced {bad} non-finite scores "
                    f"({bad / scores.size:.1%} of the batch). Refusing to rank."
                )

            target_pos = np.array([pos_map[int(i)] for i in chunk["i"]])
            scores = mask_seen(scores, chunk["u"].to_numpy(), data, pos_map, target_pos)

            rank = _rank_of_target(scores, target_pos)
            kmax = max(ks)
            top = np.argpartition(-scores, min(kmax, scores.shape[1] - 1), axis=1)[:, :kmax]
            topk_items.append(cand[top])

            out = chunk[["u", "i", "vertical"]].copy()
            out["rank"] = rank
            for k in ks:
                out[f"hit@{k}"] = (rank <= k).astype(float)
                out[f"ndcg@{k}"] = np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)
            out["mrr"] = 1.0 / rank
            for col in ("cold_in_vertical", "warm", "transfer_only",
                        "cross_vertical_user", "tail_target"):
                if col in chunk.columns:
                    out[col] = chunk[col].to_numpy()
            rows_out.append(out)

    per_row = pd.concat(rows_out, ignore_index=True)
    score_seconds = time.perf_counter() - t0

    metric_cols = [c for c in per_row.columns
                   if c.startswith(("hit@", "ndcg@")) or c == "mrr"]
    metrics = {c: float(per_row[c].mean()) for c in metric_cols}
    metrics["mean_rank"] = float(per_row["rank"].mean())

    # Catalogue-health guardrails, computed over the union of all top-10 lists.
    all_top10 = np.concatenate([t[:, :10].ravel() for t in topk_items])
    metrics["coverage@10"] = float(len(np.unique(all_top10)) / data.n_items)
    share = data.item_pop / max(data.item_pop.sum(), 1.0)
    metrics["novelty@10"] = float(np.mean(-np.log2(np.maximum(share[all_top10], 1e-12))))

    slice_metrics: Dict[str, Dict[str, float]] = {}
    joined = per_row.merge(
        eval_df[["u", "i", "cold_in_vertical", "transfer_only",
                 "cross_vertical_user", "tail_target"]].drop_duplicates(["u", "i"]),
        on=["u", "i"], how="left", suffixes=("", "_ev"),
    )
    for sname, fn in SLICES.items():
        try:
            mask = fn(joined)
        except KeyError:
            continue
        if mask.sum() < 30:            # too small to report responsibly
            continue
        sub = joined[mask]
        slice_metrics[sname] = {c: float(sub[c].mean()) for c in metric_cols}
        slice_metrics[sname]["n"] = int(len(sub))

    return EvalResult(
        model=model.name, metrics=metrics, slice_metrics=slice_metrics,
        per_row=per_row, score_seconds=score_seconds,
        params=dict(getattr(model, "params", {})), family=getattr(model, "family", ""),
    )


# --------------------------------------------------------------- statistics
def paired_comparison(a: EvalResult, b: EvalResult, metric: str = PRIMARY_METRIC,
                      n_boot: int = 2000, seed: int = 0,
                      slice_name: Optional[str] = None) -> Dict[str, float]:
    """Paired user-level bootstrap of ``a - b`` on ``metric``.

    Bootstrap rather than a t-test because per-user NDCG is a spike-at-zero
    mixture - most users score exactly 0 - and is nowhere near normal. The
    bootstrap makes no distributional assumption and gives a CI a PM can read
    directly.
    """
    key = ["u", "i"]
    cols = key + [metric]
    extra = [c for c in ("cold_in_vertical", "transfer_only",
                         "cross_vertical_user", "tail_target")
             if c in a.per_row.columns]
    m = a.per_row[cols + extra].merge(
        b.per_row[cols], on=key, suffixes=("_a", "_b")
    )
    if slice_name and slice_name != "all":
        fn = SLICES.get(slice_name)
        if fn is None:
            return {"n": 0, "delta": float("nan"), "error": f"unknown slice {slice_name}"}
        try:
            m = m[fn(m)]
        except KeyError:
            return {"n": 0, "delta": float("nan"),
                    "error": f"slice {slice_name} unavailable"}
    if len(m) == 0:
        return {"n": 0, "delta": float("nan")}

    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)

    base = float(m[f"{metric}_b"].mean())
    delta = float(d.mean())
    # Rows where *either* arm scored above zero. On a hard slice almost every
    # row is 0 for both arms, which makes the mean difference exactly 0.0 with a
    # degenerate CI - not evidence of no effect, but evidence the comparison
    # could not have detected one. Callers use this to say which it is.
    informative = int(
        ((m[f"{metric}_a"] > 0) | (m[f"{metric}_b"] > 0)).sum()
    )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # Two-sided bootstrap p-value: how often the resampled mean crosses zero.
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {
        "n": int(len(d)),
        "n_informative": informative,
        "metric": metric,
        "baseline": base,
        "delta": delta,
        "rel_lift_pct": float(100 * delta / base) if base > 0 else float("nan"),
        "ci_low": float(lo),
        "ci_high": float(hi),
        # The CI expressed as relative lift, so a verdict can be decided by
        # comparing the whole interval against the threshold the hypothesis
        # committed to in advance.
        "rel_ci_low": float(100 * lo / base) if base > 0 else float("nan"),
        "rel_ci_high": float(100 * hi / base) if base > 0 else float("nan"),
        "p_value": float(min(1.0, p)),
        "significant": bool(lo > 0 or hi < 0),
        "win_rate": float((d > 0).mean()),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def leaderboard(results: Sequence[EvalResult], metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.error:
            continue
        row = {"model": r.model, "family": r.family}
        row.update({k: v for k, v in r.metrics.items()})
        for sname in ("transfer_only", "cold_in_vertical", "tail_target"):
            if sname in r.slice_metrics:
                row[f"{metric}|{sname}"] = r.slice_metrics[sname].get(metric)
        row["fit_s"] = round(r.fit_seconds, 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(metric, ascending=False).reset_index(drop=True) if len(df) else df
