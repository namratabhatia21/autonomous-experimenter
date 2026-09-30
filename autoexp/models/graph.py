"""Graph-based retrieval: random walk with restart over a heterogeneous
item-attribute graph.

The candidate-generation arm, and the one that tests the Personalization team's
core thesis directly.

**How transfer is modelled, and why it is modelled this way.** Ranking surface
V, the walk is seeded twice: once from the user's history *on V*, and once from
their history on *other* verticals. The second contribution is added with
weight ``transfer_weight``:

    score_V(u) = walk(history_in_V) + transfer_weight * walk(history_elsewhere)

Setting ``cross_vertical=False`` zeroes the second term, so the ablation is a
genuine one-factor comparison of "does behaviour on other surfaces add anything
on top of what this surface already knows?" - which is the actual product
question.

An earlier version instead toggled whether the *graph* contained cross-vertical
edges, with a single seed over the whole history. That measured -47% NDCG@10 on
the transfer slice, and the mechanism explains why: enabling cross-vertical
edges let walk mass leak out of the target vertical through high-degree items,
so transfer arrived as popularity-biased noise layered on top of - and partly
replacing - the within-vertical signal. Catalogue coverage rose to 0.75 while
accuracy fell, which is the signature of dilution rather than of transfer. That
framing also made the ablation non-monotone: turning transfer "on" could only
displace existing signal, so it could never measure transfer's incremental
value. Decomposing the seed fixes both problems, and ``transfer_weight=0`` is
exactly the control.

**Knowledge-graph edges.** With ``use_metadata`` the graph also carries
item -> category and item -> price-band nodes, so a walk can hop between items
that never co-occur but share an attribute. That is the cheap version of a KG
pipeline, and it is what rescues cold-start items whose co-occurrence row is
empty.

Scores come from power iteration on a sparse transition matrix, batched over
users, so the whole thing stays O(nnz) and runs on CPU.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender


class GraphWalkRetriever(Recommender):
    family = "graph_walk"

    def __init__(self, walk_length: int = 3, restart: float = 0.25,
                 cross_vertical: bool = True, transfer_weight: float = 0.6,
                 use_metadata: bool = True, cooc_top_k: int = 150,
                 batch_size: int = 1024, recency_halflife: float = 0.0,
                 meta_weight: float = 0.35):
        super().__init__(
            walk_length=walk_length, restart=restart, cross_vertical=cross_vertical,
            transfer_weight=transfer_weight if cross_vertical else 0.0,
            use_metadata=use_metadata, cooc_top_k=cooc_top_k,
            recency_halflife=recency_halflife, meta_weight=meta_weight,
        )
        self.batch_size = batch_size
        tag = "xvert" if cross_vertical else "within"
        meta = "+kg" if use_metadata else ""
        self.name = f"graph_walk({tag}{meta},L={walk_length})"

    # ------------------------------------------------------------------ graph
    def fit(self, data):
        self.data = data
        p = self.params

        X = data.ui.astype(np.float32)
        # Down-weight blockbuster items so the walk is not dragged to the head
        # of the catalogue (the standard popularity correction for RWR).
        deg = np.asarray(X.sum(0)).ravel()
        damp = sp.diags((1.0 / np.maximum(deg, 1.0) ** 0.5).astype(np.float32))
        C = (damp @ (X.T @ X) @ damp).tocsr()
        C.setdiag(0.0)
        C.eliminate_zeros()

        # The graph is always the full one. Whether cross-vertical *history* is
        # used is decided at scoring time by ``transfer_weight``, which is what
        # keeps the ablation one-factor and monotone.
        C = _truncate_rows(C, int(p["cooc_top_k"]))

        C = _row_normalize(C)

        if p["use_metadata"]:
            M = self._metadata_incidence(data)
            self.n_meta_ = M.shape[1]
            w = float(p["meta_weight"])
            # ``meta_weight`` is the *share of outgoing mass* a walk sends to
            # attribute nodes, not a raw edge weight. Scaling the raw incidence
            # instead leaves KG edges carrying whatever fraction the
            # co-occurrence row sums happen to leave them: measured at 10% of
            # outgoing mass for meta_weight=0.3, which made the KG ablation come
            # back at exactly +0.0000 - the edges were there but inert, so the
            # experiment could not answer its own question. Normalising both
            # blocks first makes w mean what it says.
            top = sp.hstack([(1.0 - w) * C, (w * _row_normalize(M)).tocsr()])
            bottom = sp.hstack([_row_normalize(M.T.tocsr()),
                                sp.csr_matrix((self.n_meta_, self.n_meta_))])
            self.T_ = sp.vstack([top, bottom]).tocsr()
        else:
            self.n_meta_ = 0
            self.T_ = C

        self.n_nodes_ = self.T_.shape[0]
        return self

    def _metadata_incidence(self, data) -> sp.csr_matrix:
        """item -> {category, price_band, vertical} incidence, the KG layer."""
        cols, rows = [], []
        offset = 0
        for attr in ("category", "price_band", "vertical"):
            if attr not in data.items.columns:
                continue
            codes, uniq = pd.factorize(data.items[attr])
            rows.extend(data.items["i"].to_numpy().tolist())
            cols.extend((codes + offset).tolist())
            offset += len(uniq)
        return sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(data.n_items, max(offset, 1)),
        )

    # ------------------------------------------------------------------ seeds
    def _seeds(self, users: np.ndarray, target_vertical: int):
        """Two seed vectors per user: history on the target surface, and history
        everywhere else.

        Each is normalised independently, so the transfer term's magnitude does
        not depend on how lopsided the user's activity happens to be - a user
        with 40 Food events and 2 Shops events should not get a weaker Shops
        signal purely because the Food total dominates the denominator.
        """
        hl = self.params["recency_halflife"]
        S_in = np.zeros((len(users), self.n_nodes_), dtype=np.float32)
        S_out = np.zeros((len(users), self.n_nodes_), dtype=np.float32)

        for r, u in enumerate(users):
            seq = self.data.seqs.get(int(u))
            if seq is None or len(seq) == 0:
                continue
            if hl:
                w = (0.5 ** (np.arange(len(seq))[::-1] / hl)).astype(np.float32)
            else:
                w = np.ones(len(seq), dtype=np.float32)
            same = self.data.item_vertical[seq] == target_vertical
            if same.any():
                np.add.at(S_in[r], seq[same], w[same])
                S_in[r] /= S_in[r].sum()
            if (~same).any():
                np.add.at(S_out[r], seq[~same], w[~same])
                S_out[r] /= S_out[r].sum()
        return S_in, S_out

    def _walk(self, seed: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        alpha = 1.0 - float(self.params["restart"])
        state = seed.copy()
        acc = np.zeros_like(seed)
        for _ in range(int(self.params["walk_length"])):
            state = alpha * (state @ self.T_) + (1 - alpha) * seed
            acc += state
        return acc[:, candidates]

    def score_users(self, users, candidates):
        users = np.asarray(users)
        candidates = np.asarray(candidates)
        target_vertical = int(self.data.item_vertical[candidates[0]])
        beta = float(self.params["transfer_weight"])
        out = np.zeros((len(users), len(candidates)), dtype=np.float32)

        for start in range(0, len(users), self.batch_size):
            chunk = users[start:start + self.batch_size]
            s_in, s_out = self._seeds(chunk, target_vertical)
            block = self._walk(s_in, candidates)
            if beta > 0:
                block = block + beta * self._walk(s_out, candidates)
            out[start:start + len(chunk)] = block
        return out


class CrossVerticalBridge(Recommender):
    """An explicit, inspectable transfer model: learn a category-to-category
    affinity map, then project a user's history in *other* verticals onto the
    target vertical's catalogue.

    Where ``GraphWalkRetriever`` transfers implicitly through walk edges, this
    arm makes the transfer legible - you can print the matrix and show a PM
    which Food categories predict which Shops categories. That interpretability
    is what makes it shippable as a feature rather than a black box, even when
    its standalone accuracy is modest.
    """

    family = "cross_vertical_bridge"

    def __init__(self, smoothing: float = 5.0, self_weight: float = 1.0,
                 popularity_prior: float = 1.0):
        super().__init__(smoothing=smoothing, self_weight=self_weight,
                         popularity_prior=popularity_prior)
        self.name = "cross_vertical_bridge"

    def fit(self, data):
        self.data = data
        cat_codes, cats = pd.factorize(data.items["category"])
        self.n_cat_ = len(cats)
        self.cat_names_ = list(cats)
        item_cat = np.zeros(data.n_items, dtype=np.int32)
        item_cat[data.items["i"].to_numpy()] = cat_codes
        self.item_cat_ = item_cat

        tr = data.train
        uc = sp.csr_matrix(
            (np.ones(len(tr), dtype=np.float32),
             (tr["u"].to_numpy(), item_cat[tr["i"].to_numpy()])),
            shape=(data.n_users, self.n_cat_),
        )
        uc.sum_duplicates()

        # Category-to-category affinity = within-user co-occurrence normalised
        # by what independence would predict, i.e. lift. Raw co-occurrence would
        # just rediscover which categories are large.
        co = (uc.T @ uc).toarray().astype(np.float64)
        np.fill_diagonal(co, 0.0)
        marg = co.sum(1, keepdims=True)
        total = max(co.sum(), 1.0)
        expected = (marg @ marg.T) / total
        sm = self.params["smoothing"]
        self.affinity_ = np.log((co + sm) / (expected + sm))
        self.user_cat_ = uc
        self.pop_ = data.item_pop
        return self

    def top_transfers(self, n: int = 12):
        """The human-readable artefact: strongest cross-vertical category pulls."""
        vert_of_cat = dict(zip(self.data.items["category"], self.data.items["vertical"]))
        pairs = []
        for a in range(self.n_cat_):
            va = vert_of_cat.get(self.cat_names_[a])
            for b in range(self.n_cat_):
                vb = vert_of_cat.get(self.cat_names_[b])
                if a != b and va != vb:
                    pairs.append((self.cat_names_[a], va, self.cat_names_[b], vb,
                                  float(self.affinity_[a, b])))
        pairs.sort(key=lambda t: -t[4])
        return pairs[:n]

    def score_users(self, users, candidates):
        users = np.asarray(users)
        uc = np.asarray(self.user_cat_[users].todense())
        uc = uc / np.maximum(uc.sum(1, keepdims=True), 1e-9)
        cat_score = uc @ self.affinity_
        cat_score += self.params["self_weight"] * uc
        # Category affinity alone cannot order items *within* a category, so
        # every candidate there would tie. A popularity prior breaks those ties;
        # the transfer signal still decides which categories rise.
        prior = self.params["popularity_prior"] * np.log1p(self.pop_[candidates])
        return cat_score[:, self.item_cat_[candidates]] + prior[None, :]


# ------------------------------------------------------------------- helpers
def _row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    s = np.asarray(A.sum(1)).ravel()
    return sp.diags((1.0 / np.maximum(s, 1e-9)).astype(np.float32)) @ A.tocsr()


def _truncate_rows(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest entries per row, so downstream products stay cheap."""
    S = S.tocsr()
    rows, cols, vals = [], [], []
    for r in range(S.shape[0]):
        s, e = S.indptr[r], S.indptr[r + 1]
        if e - s == 0:
            continue
        d, idx = S.data[s:e], S.indices[s:e]
        if e - s > k:
            top = np.argpartition(-d, k)[:k]
            d, idx = d[top], idx[top]
        rows.extend([r] * len(idx)); cols.extend(idx.tolist()); vals.extend(d.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=S.shape, dtype=np.float32)
