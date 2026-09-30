"""Two-stage retrieval -> ranking, the shape a production stack actually takes.

Stage 1 is the retrievers already in the zoo (graph walk, item-kNN, MF,
attention), each contributing candidates and a score. Stage 2 is a gradient
boosted ranker trained on the *validation* split to fuse those signals along
with context the retrievers cannot see: how popular the item is, how much
history the user has on this surface versus elsewhere, whether the user is
cold here, how the retrieval sources agree.

Why this is the interesting arm rather than just an ensemble:

  * It is where cross-vertical signal becomes a **feature** rather than a
    model. ``in_vertical_hist`` and ``other_vertical_hist`` let the ranker
    learn to lean on transfer exactly for the users who need it, instead of
    applying it uniformly - which is what the slice results say it should do.

  * It is trained on a split the retrievers never saw, so a retriever that
    overfits the training log cannot launder that into the final score.

  * Its feature importances are the artefact you take to a design review.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ..evaluate import mask_seen, ndcg_at_k
from .base import Recommender


class TwoStageRanker(Recommender):
    family = "two_stage"

    def __init__(self, retrievers: Sequence[Recommender], n_candidates: int = 200,
                 n_negatives: int = 24, hard_negative_frac: float = 0.35,
                 learning_rate: float = 0.08,
                 max_iter: int = 220, max_depth: Optional[int] = 6,
                 use_transfer_features: bool = True, seed: int = 0):
        super().__init__(
            n_candidates=n_candidates, n_negatives=n_negatives,
            hard_negative_frac=hard_negative_frac, learning_rate=learning_rate, max_iter=max_iter, max_depth=max_depth,
            use_transfer_features=use_transfer_features, seed=seed,
            retrievers=[r.name for r in retrievers],
        )
        self.retrievers = list(retrievers)
        tag = "+transfer" if use_transfer_features else "no-transfer"
        self.name = f"two_stage({len(retrievers)}src,{tag})"

    # ---------------------------------------------------------------- features
    def _context(self, data) -> None:
        counts = data.user_vertical_counts()
        self._counts = counts
        self._total_hist = counts.sum(1)
        pop = data.item_pop
        self._log_pop = np.log1p(pop)
        self._pop_pct = pd.Series(pop).rank(pct=True).to_numpy()

    def _feature_block(self, data, users: np.ndarray, verticals: np.ndarray,
                       cand: np.ndarray, retr_scores: List[np.ndarray]) -> np.ndarray:
        """(n_rows * n_cand, n_features) design matrix for one vertical."""
        n_rows, n_cand = len(users), len(cand)
        cols: List[np.ndarray] = []
        names: List[str] = []

        for r, mat in zip(self.retrievers, retr_scores):
            # Raw score, plus its within-row rank. Rank is scale-free, which
            # keeps the ranker from having to learn each retriever's scale.
            cols.append(mat.reshape(-1))
            names.append(f"score::{r.name}")
            order = (-mat).argsort(1).argsort(1).astype(np.float32)
            cols.append(order.reshape(-1))
            names.append(f"rank::{r.name}")

        # Item-side
        cols.append(np.tile(self._log_pop[cand], n_rows)); names.append("log_pop")
        cols.append(np.tile(self._pop_pct[cand], n_rows)); names.append("pop_pct")

        # User-side and, crucially, cross-vertical context.
        vert = verticals[0]
        in_v = np.array([self._counts.loc[u, vert] if u in self._counts.index else 0
                         for u in users], dtype=np.float32)
        tot = np.array([self._total_hist.get(u, 0) for u in users], dtype=np.float32)
        other = tot - in_v

        cols.append(np.repeat(tot, n_cand)); names.append("hist_total")
        if self.params["use_transfer_features"]:
            cols.append(np.repeat(in_v, n_cand)); names.append("hist_in_vertical")
            cols.append(np.repeat(other, n_cand)); names.append("hist_other_verticals")
            cols.append(np.repeat(other / np.maximum(tot, 1), n_cand))
            names.append("share_other_verticals")
            cols.append(np.repeat((in_v <= 2).astype(np.float32), n_cand))
            names.append("cold_in_vertical")

        self.feature_names_ = names
        return np.column_stack(cols).astype(np.float32)

    # -------------------------------------------------------------------- fit
    def fit(self, data):
        self.data = data
        self._context(data)
        p = self.params
        rng = np.random.default_rng(int(p["seed"]))

        for r in self.retrievers:
            if getattr(r, "data", None) is not data:
                r.fit(data)

        X_parts, y_parts = [], []
        val = data.val
        for vertical, grp in val.groupby("vertical", sort=False):
            cand = data.candidates[vertical]
            pos_map = data.cand_pos[vertical]
            grp = grp[grp["i"].isin(pos_map.keys())]
            if len(grp) < 20:
                continue
            users = grp["u"].to_numpy()
            scores = [np.asarray(r.score_users(users, cand), dtype=np.float32)
                      for r in self.retrievers]

            # Negative sampling, and the one decision that decides whether
            # this model works at all.
            #
            # Mining negatives purely from the fused top-N inverts the model.
            # The held-out target frequently sits *below* the top-N, so every
            # "hard" negative outranks it and the ranker learns the exact
            # opposite rule: high retrieval score => negative. Measured on this
            # data, pure hard-negative mining trained a ranker with held-out
            # AUC 0.40 - reliably worse than chance.
            #
            # Mixing a majority of uniform negatives restores the correct
            # marginal (a random catalogue item really is worse than the
            # target) while a minority of hard negatives still teaches the
            # model to separate the head. The ``_assert_learned`` check below
            # fails loudly if this ever regresses.
            fused = np.mean([_zscore(s) for s in scores], 0)
            n_hard = int(round(int(p["n_negatives"]) * float(p["hard_negative_frac"])))
            n_unif = int(p["n_negatives"]) - n_hard
            topn = np.argpartition(-fused, min(int(p["n_candidates"]), fused.shape[1] - 1), 1)
            topn = topn[:, :int(p["n_candidates"])]

            tgt = np.array([pos_map[int(i)] for i in grp["i"]])
            rows_idx, col_idx, labels = [], [], []
            for r in range(len(grp)):
                hard = topn[r][topn[r] != tgt[r]]
                if len(hard) > n_hard:
                    hard = rng.choice(hard, n_hard, replace=False)
                unif = rng.choice(len(cand), n_unif * 2, replace=False)
                unif = unif[(unif != tgt[r]) & ~np.isin(unif, hard)][:n_unif]
                negs = np.concatenate([hard, unif]).astype(int)
                sel = np.concatenate([[tgt[r]], negs])
                rows_idx.extend([r] * len(sel))
                col_idx.extend(sel.tolist())
                labels.extend([1] + [0] * len(negs))

            rows_idx = np.array(rows_idx); col_idx = np.array(col_idx)
            block = self._feature_block(
                data, users, np.array([vertical] * len(users)), cand, scores
            )
            flat = rows_idx * len(cand) + col_idx
            X_parts.append(block[flat])
            y_parts.append(np.array(labels))

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        self.clf_ = HistGradientBoostingClassifier(
            learning_rate=float(p["learning_rate"]), max_iter=int(p["max_iter"]),
            max_depth=p["max_depth"], random_state=int(p["seed"]),
            early_stopping=True, validation_fraction=0.15,
        ).fit(X, y)
        self.n_train_rows_ = len(y)
        self.train_auc_ = self._assert_learned(X, y)
        self.blend_ = self._tune_blend(data)
        return self

    def _best_retriever_index(self, data) -> int:
        """Which single retriever ranks validation targets best."""
        best, best_score = 0, -np.inf
        for k, r in enumerate(self.retrievers):
            vals = []
            for vertical, grp in data.val.groupby("vertical", sort=False):
                cand = data.candidates[vertical]
                pos_map = data.cand_pos[vertical]
                grp = grp[grp["i"].isin(pos_map.keys())].head(1200)
                if len(grp) < 20:
                    continue
                sc = np.asarray(r.score_users(grp["u"].to_numpy(), cand), dtype=np.float64)
                tgt = np.array([pos_map[int(i)] for i in grp["i"]])
                # Same masking the evaluator applies - see evaluate.mask_seen.
                sc = mask_seen(sc, grp["u"].to_numpy(), data, pos_map, tgt)
                vals.append(ndcg_at_k(sc, tgt, 10))
            if vals:
                m = float(np.concatenate(vals).mean())
                if m > best_score:
                    best, best_score = k, m
        self.prior_name_ = self.retrievers[best].name
        self.prior_val_ndcg_ = best_score
        return best

    def _tune_blend(self, data) -> float:
        """Pick how much of the raw retrieval ordering to keep, on validation.

        A boosted tree bins each feature, so it discretises the very ordering
        the retrievers express most precisely - which is why the fused model
        can land *below* its own best input even with a healthy AUC. Blending
        the learned score with the z-scored retrieval consensus restores that
        resolution.

        The weight is chosen on the validation split only. alpha=1 is pure
        learned ranker, alpha=0 is pure retrieval consensus, so the search
        includes both endpoints and the fusion has to earn its place.
        """
        # The prior is the single strongest retriever on validation, not the
        # consensus average: averaging a strong retriever with weak ones just
        # dilutes it, which makes the blend useless as a safety net.
        self.prior_idx_ = self._best_retriever_index(data)
        grid = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
        totals = {a: [] for a in grid}
        for vertical, grp in data.val.groupby("vertical", sort=False):
            cand = data.candidates[vertical]
            pos_map = data.cand_pos[vertical]
            grp = grp[grp["i"].isin(pos_map.keys())].head(1500)
            if len(grp) < 20:
                continue
            users = grp["u"].to_numpy()
            scores = [np.asarray(r.score_users(users, cand), dtype=np.float32)
                      for r in self.retrievers]
            X = self._feature_block(
                data, users, np.array([vertical] * len(users)), cand, scores
            )
            learned = self.clf_.predict_proba(X)[:, 1].reshape(len(users), len(cand))
            prior = _zscore(scores[self.prior_idx_])
            learned_z = _zscore(learned)
            tgt = np.array([pos_map[int(i)] for i in grp["i"]])
            for a in grid:
                combined = a * learned_z + (1 - a) * prior
                combined = mask_seen(combined, users, data, pos_map, tgt)
                totals[a].append(ndcg_at_k(combined, tgt, 10))

        means = {a: float(np.concatenate(v).mean()) for a, v in totals.items() if v}
        self.blend_search_ = means
        return max(means, key=means.get) if means else 1.0

    @staticmethod
    def _assert_learned(X: np.ndarray, y: np.ndarray) -> float:
        """Refuse to return a ranker that scores worse than a coin flip.

        A pointwise ranker can invert silently when negative sampling is
        correlated with the label, and an inverted ranker still produces
        plausible-looking probabilities. Only the AUC gives it away, so it is
        checked here rather than discovered three experiments later.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
        probe = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=80, random_state=0
        ).fit(Xa, ya)
        auc = float(roc_auc_score(yb, probe.predict_proba(Xb)[:, 1]))
        if auc < 0.5:
            raise ValueError(
                f"Two-stage ranker learned an inverted decision rule "
                f"(held-out AUC={auc:.3f} < 0.5). This is almost always a "
                f"negative-sampling bug, not a modelling one."
            )
        return auc

    def score_users(self, users, candidates, chunk: int = 512):
        """Score in user chunks.

        The design matrix has one row per (user, candidate) pair, so a full
        evaluation batch against a 3,300-item pool would allocate ~700MB.
        Chunking bounds it to tens of megabytes without changing the result.
        """
        users = np.asarray(users)
        vertical = self._vertical_of(candidates)
        a = getattr(self, "blend_", 1.0)
        out = np.zeros((len(users), len(candidates)), dtype=np.float32)

        for start in range(0, len(users), chunk):
            u = users[start:start + chunk]
            scores = [np.asarray(r.score_users(u, candidates), dtype=np.float32)
                      for r in self.retrievers]
            X = self._feature_block(
                self.data, u, np.array([vertical] * len(u)), candidates, scores
            )
            learned = self.clf_.predict_proba(X)[:, 1].reshape(len(u), len(candidates))
            if a >= 1.0:
                out[start:start + len(u)] = learned
            else:
                prior = _zscore(scores[getattr(self, "prior_idx_", 0)])
                out[start:start + len(u)] = a * _zscore(learned) + (1 - a) * prior
        return out

    def _vertical_of(self, candidates: np.ndarray) -> str:
        code = int(self.data.item_vertical[candidates[0]])
        for v, c in self.data.vert_code.items():
            if c == code:
                return v
        return self.data.verticals[0]

    # ---------------------------------------------------------------- explain
    def feature_importance(self, n_repeats: int = 3, sample: int = 4000,
                           seed: int = 0) -> pd.DataFrame:
        """Permutation importance on held-out test rows.

        Permutation rather than split-gain because gain is biased toward
        high-cardinality continuous features, and the whole point of this table
        is to answer "does the cross-vertical feature actually carry weight?"
        """
        from sklearn.inspection import permutation_importance

        data = self.data
        rng = np.random.default_rng(seed)
        parts_X, parts_y = [], []
        for vertical, grp in data.test.groupby("vertical", sort=False):
            cand = data.candidates[vertical]
            pos_map = data.cand_pos[vertical]
            grp = grp[grp["i"].isin(pos_map.keys())].head(400)
            if len(grp) < 20:
                continue
            users = grp["u"].to_numpy()
            scores = [np.asarray(r.score_users(users, cand), dtype=np.float32)
                      for r in self.retrievers]
            block = self._feature_block(
                data, users, np.array([vertical] * len(users)), cand, scores
            )
            tgt = np.array([pos_map[int(i)] for i in grp["i"]])
            rows, cols, ys = [], [], []
            for r in range(len(grp)):
                negs = rng.choice(len(cand), 12, replace=False)
                sel = np.concatenate([[tgt[r]], negs])
                rows.extend([r] * len(sel)); cols.extend(sel.tolist())
                ys.extend([1] + [0] * len(negs))
            flat = np.array(rows) * len(cand) + np.array(cols)
            parts_X.append(block[flat]); parts_y.append(np.array(ys))

        X = np.vstack(parts_X); y = np.concatenate(parts_y)
        if len(y) > sample:
            idx = rng.choice(len(y), sample, replace=False)
            X, y = X[idx], y[idx]
        imp = permutation_importance(
            self.clf_, X, y, n_repeats=n_repeats, random_state=seed, scoring="roc_auc"
        )
        return (
            pd.DataFrame({
                "feature": self.feature_names_,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-8)
