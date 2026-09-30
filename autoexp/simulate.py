"""A Careem-shaped multi-vertical marketplace simulator.

Why simulate? No public dataset contains food-delivery, quick-commerce and
retail interactions for the *same* users, which is precisely the setting the
Personalization team operates in. The simulator gives us two things a public
log cannot:

  1. Ground truth for cross-vertical transfer strength (rho), so we can check
     whether the agent's conclusions are actually correct, not merely
     internally consistent.
  2. A known logging policy with position bias, so off-policy evaluation and
     the simulated A/B test are honest rather than hand-waved.

The simulator is never the only evidence: the agent runs the identical
protocol against a real cross-vertical slice of Amazon-Reviews-2023 (see
datasets.py), and the report shows both. A conclusion that holds in one and
not the other is reported as exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

VERTICALS = ["food", "quik", "shops"]

# Shared semantic concepts that deliberately span verticals: a user who orders
# spicy food should also buy chilli paste in Quik and a tagine dish in Shops.
CONCEPTS = [
    "spicy", "healthy", "budget", "premium", "family", "late_night",
    "breakfast", "sweet", "beverages", "household",
]

VERTICAL_CATEGORIES = {
    "food": ["biryani", "burgers", "shawarma", "sushi", "pizza", "salads", "desserts", "coffee"],
    "quik": ["fresh_produce", "dairy", "snacks", "beverages", "household", "bakery", "frozen", "personal_care"],
    "shops": ["electronics", "apparel", "home", "beauty", "toys", "sports", "books", "pharmacy"],
}


@dataclass
class SimConfig:
    n_users: int = 4000
    items_per_vertical: int = 500
    n_sessions_mean: float = 11.0
    slate_size: int = 12
    embed_dim: int = 12
    rho: float = 0.55             # ground-truth cross-vertical taste correlation
    position_bias: float = 0.85   # geometric decay of exposure down the slate
    popularity_alpha: float = 0.45  # how hard the logging policy chases popularity
    attract_scale: float = 1.5     # click model: utility -> attractiveness slope
    attract_bias: float = -2.05    # click model: intercept, sets the base CTR
    explore_noise: float = 1.3     # exploration in the logging policy; without
                                   # it the log only ever contains head items
    seed: int = 42
    days: int = 60


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class MultiVerticalSimulator:
    """Generates an event log from a known generative process."""

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def _build_items(self) -> pd.DataFrame:
        cfg, rng = self.cfg, self.rng
        rows = []
        item_id = 0
        for v in VERTICALS:
            cats = VERTICAL_CATEGORIES[v]
            for _ in range(cfg.items_per_vertical):
                cat = cats[rng.integers(len(cats))]
                k = int(rng.integers(1, 4))
                concepts = list(rng.choice(CONCEPTS, size=k, replace=False))
                rows.append({
                    "item_id": f"{v[:1]}{item_id:04d}",
                    "vertical": v,
                    "category": cat,
                    "concepts": ";".join(concepts),
                    "price_band": int(rng.integers(1, 5)),
                })
                item_id += 1
        items = pd.DataFrame(rows)

        n = len(items)
        concept_mat = np.zeros((n, len(CONCEPTS)))
        for i, cs in enumerate(items["concepts"]):
            for c in cs.split(";"):
                concept_mat[i, CONCEPTS.index(c)] = 1.0
        concept_mat /= np.maximum(concept_mat.sum(1, keepdims=True), 1)
        private = rng.normal(0, 1, (n, cfg.embed_dim))
        self.item_concept = concept_mat
        self.item_private = private / np.linalg.norm(private, axis=1, keepdims=True)

        # Long-tail popularity (Zipf) - the bias every recommender must fight.
        pop = 1.0 / np.power(np.arange(1, n + 1), 1.1)
        rng.shuffle(pop)
        items["base_popularity"] = pop / pop.sum()
        return items

    def _build_users(self) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        cfg, rng = self.cfg, self.rng
        n = cfg.n_users

        # Shared concept taste: this is the signal that *should* transfer.
        shared = rng.normal(0, 1, (n, len(CONCEPTS)))
        concept_taste = {}
        for v in VERTICALS:
            private = rng.normal(0, 1, (n, len(CONCEPTS)))
            concept_taste[v] = np.sqrt(cfg.rho) * shared + np.sqrt(1 - cfg.rho) * private

        private_taste = rng.normal(0, 1, (n, cfg.embed_dim))

        # Many users are active in only 1-2 verticals: a realistic
        # cross-vertical cold-start population.
        props = rng.dirichlet(np.array([1.1, 0.9, 0.7]), size=n)
        mask = rng.random((n, 3)) < np.array([0.92, 0.70, 0.45])
        mask[np.arange(n), props.argmax(1)] = True
        props = props * mask
        props = props / props.sum(1, keepdims=True)

        users = pd.DataFrame({
            "user_id": [f"u{i:05d}" for i in range(n)],
            "home_vertical": [VERTICALS[i] for i in props.argmax(1)],
            "night_owl": rng.random(n) < 0.3,
        })
        self.vertical_prop = props
        return users, private_taste, concept_taste

    def click_probabilities(self, rel: np.ndarray) -> np.ndarray:
        """P(click) per slate position = P(examine) x P(attractive).

        Attractiveness is a function of *absolute* utility, deliberately. An
        earlier version used a softmax over the slate, which normalises to 1
        and therefore makes total clicks almost independent of slate quality -
        a simulated A/B built on it compares two rankers and finds nothing,
        because there is nothing there to find. Position bias handles
        examination; attractiveness has to carry the quality signal.
        """
        cfg = self.cfg
        pos_bias = np.power(cfg.position_bias, np.arange(len(rel)))
        attract = _sigmoid(cfg.attract_scale * rel + cfg.attract_bias)
        return np.clip(pos_bias * attract, 0.0, 0.95)

    def _relevance(self, u: int, item_idx: np.ndarray, vertical: str) -> np.ndarray:
        """True latent utility - never exposed to any model."""
        c = self.concept_taste[vertical][u] @ self.item_concept[item_idx].T
        p = self.private_taste[u] @ self.item_private[item_idx].T
        return 1.6 * c + 0.7 * p

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg, rng = self.cfg, self.rng
        items = self._build_items()
        users, self.private_taste, self.concept_taste = self._build_users()

        vert_idx = {v: np.where(items["vertical"].to_numpy() == v)[0] for v in VERTICALS}
        pop = items["base_popularity"].to_numpy()
        item_ids = items["item_id"].to_numpy()
        uids = users["user_id"].to_numpy()
        owls = users["night_owl"].to_numpy()
        events: List[dict] = []
        impressions: List[dict] = []
        t0 = pd.Timestamp("2025-06-01")

        for u in range(cfg.n_users):
            n_sessions = 1 + rng.poisson(cfg.n_sessions_mean)
            starts = np.sort(rng.random(n_sessions)) * cfg.days
            for s, day_offset in enumerate(starts):
                v = VERTICALS[rng.choice(3, p=self.vertical_prop[u])]
                hour = int(rng.normal(21 if owls[u] else 13, 3)) % 24
                ts = t0 + pd.Timedelta(days=float(day_offset)) + pd.Timedelta(hours=hour)

                cand = vert_idx[v]
                # Logging policy: popularity^alpha plus exploration noise.
                logit = (cfg.popularity_alpha * np.log(pop[cand] + 1e-12)
                         + rng.normal(0, cfg.explore_noise, len(cand)))
                slate = cand[np.argsort(-logit)[: cfg.slate_size]]

                rel = self._relevance(u, slate, v)
                clicked = rng.random(len(slate)) < self.click_probabilities(rel)

                sid = f"{uids[u]}_s{s}"
                for rank, (ii, ck) in enumerate(zip(slate, clicked)):
                    impressions.append({
                        "user_id": uids[u], "item_id": item_ids[ii], "vertical": v,
                        "rank": rank, "ts": ts, "clicked": int(ck), "session_id": sid,
                    })
                for ii in slate[clicked]:
                    events.append({
                        "user_id": uids[u], "item_id": item_ids[ii], "vertical": v,
                        "ts": ts, "session_id": sid, "hour": hour, "event_type": "order",
                    })

        ev = pd.DataFrame(events).sort_values("ts").reset_index(drop=True)
        imp = pd.DataFrame(impressions)
        return ev, items, users, imp
