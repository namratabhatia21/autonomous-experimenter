"""Classical baselines.

These exist to keep the agent honest. A transformer that cannot beat
item-kNN on a sparse log is not a win, and popularity is the floor every
recommendation result should be reported against.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .base import Recommender, l2_normalize


class PopularityRecommender(Recommender):
    """Non-personalised floor: rank by training frequency.

    Reported for every experiment. A lift over this is the only number a
    product decision can actually be based on.
    """

    family = "popularity"

    def __init__(self, damping: float = 0.0):
        super().__init__(damping=damping)
        self.name = "popularity"

    def fit(self, data):
        self.data = data
        pop = data.item_pop.astype(np.float64)
        self.scores_ = np.log1p(pop) if self.params["damping"] else pop
        return self

    def score_users(self, users, candidates):
        return np.tile(self.scores_[candidates], (len(users), 1))


class ItemKNN(Recommender):
    """Item-item collaborative filtering with cosine similarity.

    Still the strongest cheap baseline on sparse implicit logs. ``top_k``
    truncation keeps the similarity matrix sparse enough for a large catalogue.
    """

    family = "item_knn"

    def __init__(self, top_k: int = 200, shrink: float = 10.0, recency_halflife: float = 0.0):
        super().__init__(top_k=top_k, shrink=shrink, recency_halflife=recency_halflife)
        self.name = f"item_knn(k={top_k})"

    def fit(self, data):
        self.data = data
        X = data.ui.astype(np.float32)          # users x items
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())
        inv = 1.0 / np.maximum(norms + self.params["shrink"], 1e-8)
        Xn = X @ sp.diags(inv.astype(np.float32))
        S = (Xn.T @ Xn).tocsr()                 # items x items
        S.setdiag(0.0)
        S.eliminate_zeros()
        self.S_ = _truncate_rows(S, int(self.params["top_k"]))
        return self

    def _user_profile(self, users: np.ndarray) -> sp.csr_matrix:
        hl = self.params["recency_halflife"]
        if not hl:
            return self.data.ui[users]
        # Exponentially discount older interactions: the same kNN model, but
        # weighted toward what the user did recently.
        rows, cols, vals = [], [], []
        for r, u in enumerate(users):
            seq = self.data.seqs.get(int(u))
            if seq is None or len(seq) == 0:
                continue
            age = np.arange(len(seq))[::-1]
            w = 0.5 ** (age / hl)
            rows.extend([r] * len(seq)); cols.extend(seq.tolist()); vals.extend(w.tolist())
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(len(users), self.data.n_items), dtype=np.float32
        )

    def score_users(self, users, candidates):
        profile = self._user_profile(np.asarray(users))
        return np.asarray((profile @ self.S_[:, candidates]).todense())


class BPRMatrixFactorization(Recommender):
    """Matrix factorisation trained with the Bayesian Personalised Ranking loss.

    The classical latent-factor control the attention model has to justify its
    cost against. BPR rather than iALS for a concrete reason: this environment's
    LAPACK solves a 64x64 system in ~2.7ms, so per-row alternating least
    squares spent 108s on a 2,840-user log and batching the solves did not help
    (the batched path was just as slow). BPR needs no linear solves at all -
    only embedding lookups and a sigmoid - so it trains in seconds on the same
    data and is, on implicit feedback, the stronger baseline anyway because its
    loss is a ranking loss rather than a regression one.
    """

    family = "bpr_mf"

    def __init__(self, factors: int = 64, lr: float = 0.05, reg: float = 2e-5,
                 epochs: int = 30, batch_size: int = 8192, seed: int = 0,
                 device: str = "cpu"):
        super().__init__(factors=factors, lr=lr, reg=reg, epochs=epochs,
                         batch_size=batch_size, seed=seed)
        self.device = device
        self.name = f"bpr_mf(d={factors})"

    def fit(self, data):
        import torch

        self.data = data
        p = self.params
        f = int(p["factors"])
        torch.manual_seed(int(p["seed"]))
        rng = np.random.default_rng(int(p["seed"]))

        coo = data.ui.tocoo()
        users = torch.from_numpy(coo.row.astype(np.int64))
        items = torch.from_numpy(coo.col.astype(np.int64))
        n_obs = len(users)

        U = torch.nn.Embedding(data.n_users, f)
        V = torch.nn.Embedding(data.n_items, f)
        bias = torch.nn.Embedding(data.n_items, 1)
        torch.nn.init.normal_(U.weight, std=0.05)
        torch.nn.init.normal_(V.weight, std=0.05)
        torch.nn.init.zeros_(bias.weight)
        opt = torch.optim.Adam(
            list(U.parameters()) + list(V.parameters()) + list(bias.parameters()),
            lr=float(p["lr"]), weight_decay=float(p["reg"]),
        )

        bs = int(p["batch_size"])
        self.history_ = []
        for _ in range(int(p["epochs"])):
            perm = torch.from_numpy(rng.permutation(n_obs))
            total = 0.0
            for b in range(0, n_obs, bs):
                idx = perm[b:b + bs]
                u, i = users[idx], items[idx]
                # Uniform negative sampling. A collision with a true positive is
                # possible and harmless at this density; checking every draw
                # against the sparse matrix would cost more than it buys.
                j = torch.from_numpy(rng.integers(0, data.n_items, len(idx)))
                eu = U(u)
                x_ui = (eu * V(i)).sum(-1) + bias(i).squeeze(-1)
                x_uj = (eu * V(j)).sum(-1) + bias(j).squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(x_ui - x_uj).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss) * len(idx)
            self.history_.append(total / max(n_obs, 1))

        self.U_ = U.weight.detach().numpy()
        self.V_ = V.weight.detach().numpy()
        self.b_ = bias.weight.detach().numpy().ravel()
        return self

    def score_users(self, users, candidates):
        return (self.U_[np.asarray(users)] @ self.V_[candidates].T
                + self.b_[candidates][None, :])


def _truncate_rows(S: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep only the k largest entries per row, so downstream products stay cheap."""
    S = S.tocsr()
    rows, cols, vals = [], [], []
    for r in range(S.shape[0]):
        s, e = S.indptr[r], S.indptr[r + 1]
        if e - s == 0:
            continue
        d = S.data[s:e]
        idx = S.indices[s:e]
        if e - s > k:
            top = np.argpartition(-d, k)[:k]
            d, idx = d[top], idx[top]
        rows.extend([r] * len(idx)); cols.extend(idx.tolist()); vals.extend(d.tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=S.shape, dtype=np.float32)
