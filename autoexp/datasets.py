"""Dataset assembly: a single container every model and metric speaks to.

Two sources ship with the repo, and the agent runs the *same* experiment
protocol against both:

  * ``amazon_xvert`` - a cross-vertical slice of **McAuley-Lab/Amazon-Reviews-2023**
    (the most-liked recommendation dataset on the Hugging Face Hub, 361 likes).
    Three product categories are mapped onto Careem's three verticals:

        Grocery_and_Gourmet_Food -> food
        Health_and_Household     -> quik
        Beauty_and_Personal_Care -> shops

    26% of the 1.46M users in these categories buy in two or more of them and
    6.5% buy in all three, which is a genuine multi-surface population - the
    thing the Personalization team's thesis depends on and that no
    single-domain benchmark can test.

  * ``careem_sim`` - the multi-vertical simulator. It is not a substitute for
    the real log; it exists because it ships two things the real log cannot:
    ground-truth cross-vertical transfer strength (so the agent's conclusion
    can be checked for *correctness*, not just consistency) and a known
    logging policy with position bias (so the simulated A/B is honest).

Evaluation protocol, identical for every arm so comparisons are apples to
apples: per-user temporal leave-one-out. The last interaction is the test
target, the second-to-last is validation, everything earlier is training.
Candidates are ranked against the target vertical's catalogue (capped at
``max_candidates`` by popularity, always including the target), minus items
the user already interacted with in training - which mirrors ranking a surface
the user is actually standing on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HF_REPO = "McAuley-Lab/Amazon-Reviews-2023"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/benchmark/5core/rating_only"

#: Amazon category -> Careem vertical. Chosen because these three have the
#: highest shared-user overlap of any triple in the release, and because the
#: product semantics line up (gourmet food, household essentials, retail).
AMAZON_VERTICALS = {
    "Grocery_and_Gourmet_Food": "food",
    "Health_and_Household": "quik",
    "Beauty_and_Personal_Care": "shops",
}

MIN_INTERACTIONS = 5   # users below this cannot support a train/val/test split


@dataclass
class RecData:
    """Encoded interaction log plus the fixed train/val/test split."""

    name: str
    description: str
    source: str
    events: pd.DataFrame            # u, i, vertical, ts (+ raw ids)
    items: pd.DataFrame             # i, item_id, vertical, category, price_band
    n_users: int
    n_items: int
    verticals: List[str]
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    impressions: Optional[pd.DataFrame] = None
    ground_truth: Dict[str, float] = field(default_factory=dict)
    max_candidates: int = 3000

    def __post_init__(self) -> None:
        self.vert_code = {v: k for k, v in enumerate(self.verticals)}
        self.item_vertical = np.zeros(self.n_items, dtype=np.int16)
        self.item_vertical[self.items["i"].to_numpy()] = (
            self.items["vertical"].map(self.vert_code).to_numpy()
        )
        self._build_sequences()
        self._build_matrix()
        self._build_candidates()

    # ---------------------------------------------------------------- derived
    def _build_sequences(self) -> None:
        tr = self.train.sort_values("ts")
        self.seqs: Dict[int, np.ndarray] = {
            int(u): g["i"].to_numpy() for u, g in tr.groupby("u", sort=False)
        }
        self.seq_verticals: Dict[int, np.ndarray] = {
            int(u): g["vertical"].map(self.vert_code).to_numpy()
            for u, g in tr.groupby("u", sort=False)
        }

    def _build_matrix(self) -> None:
        tr = self.train
        self.ui = sp.csr_matrix(
            (np.ones(len(tr), dtype=np.float32), (tr["u"].to_numpy(), tr["i"].to_numpy())),
            shape=(self.n_users, self.n_items),
        )
        self.ui.sum_duplicates()
        self.ui.data[:] = 1.0
        self.item_pop = np.asarray(self.ui.sum(0)).ravel()

    def _build_candidates(self) -> None:
        """One candidate pool per vertical, shared by every arm.

        For a large catalogue, ranking against every item is both slow and
        unrealistic - production retrieval never considers the full tail. We
        cap the pool at the ``max_candidates`` most popular items per vertical
        and then force-include every held-out target, so no arm is ever asked
        to rank an item that is absent from its pool.
        """
        self.candidates: Dict[str, np.ndarray] = {}
        targets = set(self.test["i"].tolist()) | set(self.val["i"].tolist())
        for v, code in self.vert_code.items():
            pool = np.where(self.item_vertical == code)[0]
            if len(pool) > self.max_candidates:
                order = np.argsort(-self.item_pop[pool])
                kept = set(pool[order[: self.max_candidates]].tolist())
                kept |= {int(i) for i in pool if int(i) in targets}
                pool = np.array(sorted(kept), dtype=np.int64)
            self.candidates[v] = pool
        self.cand_pos = {
            v: {int(it): k for k, it in enumerate(pool)} for v, pool in self.candidates.items()
        }

    # ----------------------------------------------------------------- slices
    def user_vertical_counts(self) -> pd.DataFrame:
        return (
            self.train.groupby(["u", "vertical"]).size().unstack(fill_value=0)
        ).reindex(columns=self.verticals, fill_value=0)

    def eval_frame(self, split: str = "test") -> pd.DataFrame:
        """One row per ranking target, carrying the slice labels the agent uses
        to break results down: cold-start-in-vertical, cross-vertical user,
        tail item. Slices are where recommendation wins or dies, so they are
        first-class, not an afterthought."""
        df = (self.test if split == "test" else self.val).copy()
        counts = self.user_vertical_counts()
        hist_len = self.train.groupby("u").size()
        n_verts = (counts > 0).sum(1)

        df["hist_len"] = df["u"].map(hist_len).fillna(0).astype(int)
        df["n_hist_verticals"] = df["u"].map(n_verts).fillna(0).astype(int)
        stacked = counts.stack()
        df["in_vertical_hist"] = [
            int(stacked.get((u, v), 0)) for u, v in zip(df["u"], df["vertical"])
        ]
        df["cold_in_vertical"] = df["in_vertical_hist"] <= 2
        df["cross_vertical_user"] = df["n_hist_verticals"] >= 2
        # The sharpest test of the team's thesis: no history on this surface,
        # but history elsewhere. Only transfer can help these users.
        df["transfer_only"] = df["cold_in_vertical"] & df["cross_vertical_user"]
        pop_rank = pd.Series(self.item_pop).rank(pct=True)
        df["target_pop_pct"] = df["i"].map(pop_rank).fillna(0.0)
        df["tail_target"] = df["target_pop_pct"] < 0.70
        df = df[df["hist_len"] > 0]
        return df.reset_index(drop=True)

    def summary(self) -> Dict[str, object]:
        counts = self.user_vertical_counts()
        return {
            "dataset": self.name,
            "n_users": int(self.n_users),
            "n_items": int(self.n_items),
            "n_interactions": int(len(self.events)),
            "verticals": self.verticals,
            "density_pct": round(100 * len(self.train) / (self.n_users * self.n_items), 4),
            "median_history": int(self.train.groupby("u").size().median()),
            "multi_vertical_user_pct": round(100 * float(((counts > 0).sum(1) >= 2).mean()), 1),
            "gini_popularity": round(_gini(self.item_pop), 3),
            "candidates_per_vertical": {v: int(len(c)) for v, c in self.candidates.items()},
            "interactions_per_vertical": {
                str(k): int(v) for k, v in self.train.groupby("vertical").size().items()
            },
        }


def _gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------------- split
def _encode_and_split(
    events: pd.DataFrame, items: pd.DataFrame, name: str, description: str,
    source: str, impressions: Optional[pd.DataFrame] = None,
    ground_truth: Optional[Dict[str, float]] = None, max_candidates: int = 3000,
) -> RecData:
    events = events.sort_values("ts").reset_index(drop=True)

    keep = events.groupby("user_id").size()
    events = events[events["user_id"].isin(keep[keep >= MIN_INTERACTIONS].index)]
    items = items[items["item_id"].isin(set(events["item_id"]))].reset_index(drop=True)
    events = events[events["item_id"].isin(set(items["item_id"]))]

    uidx = {u: k for k, u in enumerate(sorted(events["user_id"].unique()))}
    iidx = {i: k for k, i in enumerate(sorted(items["item_id"].unique()))}
    events = events.assign(u=events["user_id"].map(uidx), i=events["item_id"].map(iidx))
    items = items.assign(i=items["item_id"].map(iidx))

    events = events.sort_values(["u", "ts"], kind="mergesort").reset_index(drop=True)
    rank_desc = events.groupby("u").cumcount(ascending=False)
    test = events[rank_desc == 0]
    val = events[rank_desc == 1]
    train = events[rank_desc >= 2]

    # A held-out item never seen in training cannot be ranked by any
    # collaborative arm; keeping it would add identical noise to every arm and
    # depress all metrics without changing the comparison.
    trained = set(train["i"].unique())
    test = test[test["i"].isin(trained)]
    val = val[val["i"].isin(trained)]

    return RecData(
        name=name, description=description, source=source,
        events=events, items=items,
        n_users=len(uidx), n_items=len(iidx),
        verticals=sorted(items["vertical"].unique().tolist()),
        train=train.reset_index(drop=True), val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        impressions=impressions, ground_truth=ground_truth or {},
        max_candidates=max_candidates,
    )


def _induce_categories(train: pd.DataFrame, items: pd.DataFrame,
                       n_per_vertical: int = 30, seed: int = 0) -> pd.DataFrame:
    """The 5-core release ships no taxonomy, so induce one behaviourally.

    Truncated SVD on the item-user matrix, then k-means per vertical. The
    resulting clusters are what the knowledge-graph edges and the
    cross-vertical bridge use as "category" nodes. This is also what you would
    do against a real marketplace catalogue whose taxonomy is inconsistent
    across verticals.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD

    n_items = items["i"].max() + 1
    X = sp.csr_matrix(
        (np.ones(len(train), dtype=np.float32), (train["i"].to_numpy(), train["u"].to_numpy())),
        shape=(n_items, train["u"].max() + 1),
    )
    k = min(64, min(X.shape) - 1)
    emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(X)

    cats = np.full(n_items, -1, dtype=np.int32)
    offset = 0
    for v, grp in items.groupby("vertical"):
        idx = grp["i"].to_numpy()
        nc = min(n_per_vertical, max(2, len(idx) // 20))
        km = MiniBatchKMeans(n_clusters=nc, random_state=seed, n_init=3, batch_size=1024)
        cats[idx] = km.fit_predict(emb[idx]) + offset
        offset += nc
    items = items.copy()
    items["category"] = [f"c{c}" for c in cats[items["i"].to_numpy()]]
    return items


# -------------------------------------------------------------------- loaders
def load_careem_sim(n_users: int = 4000, rho: float = 0.55, seed: int = 42) -> RecData:
    from .simulate import MultiVerticalSimulator, SimConfig

    cfg = SimConfig(n_users=n_users, rho=rho, seed=seed)
    sim = MultiVerticalSimulator(cfg)
    ev, items, users, imp = sim.generate()
    data = _encode_and_split(
        ev, items,
        name="careem_sim",
        description=(
            f"Simulated Careem-style log: {n_users} users across Food / Quik / Shops, "
            f"ground-truth cross-vertical taste correlation rho={rho}, "
            f"logging policy with geometric position bias."
        ),
        source=f"autoexp.simulate.MultiVerticalSimulator (deterministic, seed={seed})",
        impressions=imp,
        ground_truth={"rho": rho, "position_bias": cfg.position_bias},
    )
    # Keep the generator: the simulated A/B test needs its latent utilities and
    # position-bias curve to produce clicks. Real datasets have no equivalent,
    # which is exactly why that test is only run here.
    data.sim = sim
    return data


def _download_amazon(category: str) -> Path:
    import urllib.request

    dest = DATA_DIR / "amazon2023" / f"{category}.csv"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{category}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": "autoexp"})
    with urllib.request.urlopen(req, timeout=900) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    return dest


def load_amazon_xvert(
    n_cross_users: int = 22000, n_single_users: int = 8000,
    min_item_count: int = 10, max_candidates: int = 3000, seed: int = 0,
) -> RecData:
    """Cross-vertical slice of Amazon-Reviews-2023.

    Sampling is deliberate, not convenience: we keep a large cohort of users
    active in two or more verticals (the population the transfer hypothesis is
    about) plus a control cohort of single-vertical users, so the agent can
    measure transfer benefit *and* check it does not cost anything for users
    who only ever touch one surface.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for cat, vert in AMAZON_VERTICALS.items():
        path = _download_amazon(cat)
        df = pd.read_csv(
            path, usecols=["user_id", "parent_asin", "rating", "timestamp"],
            dtype={"user_id": "string", "parent_asin": "string", "rating": "float32"},
        )
        df["vertical"] = vert
        frames.append(df)
    ev = pd.concat(frames, ignore_index=True)
    del frames

    # Implicit positive feedback only.
    ev = ev[ev["rating"] >= 4.0].drop(columns=["rating"])

    nv = ev.groupby("user_id")["vertical"].nunique()
    cross = nv.index[nv >= 2].to_numpy()
    single = nv.index[nv == 1].to_numpy()
    pick = np.concatenate([
        rng.choice(cross, size=min(n_cross_users, len(cross)), replace=False),
        rng.choice(single, size=min(n_single_users, len(single)), replace=False),
    ])
    ev = ev[ev["user_id"].isin(set(pick.tolist()))].copy()

    counts = ev.groupby("parent_asin").size()
    ev = ev[ev["parent_asin"].isin(counts[counts >= min_item_count].index)]

    ev = ev.rename(columns={"parent_asin": "item_id"})
    ev["ts"] = pd.to_datetime(ev["timestamp"], unit="ms")
    ev["session_id"] = ev["user_id"]
    ev = ev[["user_id", "item_id", "vertical", "ts", "session_id"]]

    items = ev.drop_duplicates("item_id")[["item_id", "vertical"]].copy()
    items["category"] = items["vertical"]      # replaced by induced clusters below
    items["price_band"] = 1

    data = _encode_and_split(
        ev, items,
        name="amazon_xvert",
        description=(
            "Cross-vertical slice of McAuley-Lab/Amazon-Reviews-2023 (5-core, "
            "ratings >= 4 as implicit positives). Grocery -> food, "
            "Health & Household -> quik, Beauty -> shops."
        ),
        source=f"https://huggingface.co/datasets/{HF_REPO}",
        max_candidates=max_candidates,
    )
    # Induce a taxonomy now that ids are encoded, then rebuild the container so
    # metadata-dependent structures see the categories.
    data.items = _induce_categories(data.train, data.items, seed=seed)
    return data


REGISTRY = {
    "careem_sim": load_careem_sim,
    "amazon_xvert": load_amazon_xvert,
}


def load(name: str, **kwargs) -> RecData:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)
