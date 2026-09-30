"""Self-attentive sequential recommendation (SASRec-style), in PyTorch.

This is the "beyond classical ML" arm. The architecture follows Kang & McAuley
(2018): item embeddings + learned positional embeddings, a stack of causally
masked self-attention blocks, and a binary cross-entropy loss over one sampled
negative per position.

Two departures from the paper, both aimed at the cross-vertical question:

  * **Vertical embeddings.** Each position adds an embedding of the vertical
    the interaction happened on, so the model can represent "this user is in a
    Food mood right now" and attend differently. Toggled by ``use_vertical``,
    which is what makes the vertical signal an *ablation* rather than an
    assumption.

  * **Vertical-restricted scoring.** At evaluation the model scores only the
    target vertical's candidate pool, matching how it would be served behind a
    surface-specific retrieval call.

Sized to train on CPU in a couple of minutes. It is deliberately small: the
point of the experiment is whether attention over a cross-vertical sequence
beats item-kNN and a graph walk on *this* data, not to chase a leaderboard.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .base import Recommender


class _SASRecNet(nn.Module):
    def __init__(self, n_items: int, n_verticals: int, d: int = 64, n_heads: int = 2,
                 n_blocks: int = 2, max_len: int = 50, dropout: float = 0.2,
                 use_vertical: bool = True):
        super().__init__()
        self.use_vertical = use_vertical
        self.d = d
        self.max_len = max_len
        # index 0 is the padding slot, hence n_items + 1
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.vert_emb = nn.Embedding(n_verticals + 1, d, padding_idx=0) if use_vertical else None
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d)

        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, seq: torch.Tensor, vert: Optional[torch.Tensor]) -> torch.Tensor:
        B, L = seq.shape
        x = self.item_emb(seq) * math.sqrt(self.d)
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)
        if self.use_vertical and vert is not None:
            x = x + self.vert_emb(vert)
        x = self.drop(x)

        # NOTE: no src_key_padding_mask. Sequences are left-padded and the
        # attention is causal, so an early pad position would be able to attend
        # to nothing at all; softmax over an all-masked row returns NaN, which
        # then propagates through LayerNorm and silently destroys the run. The
        # pad embedding is exactly zero (padding_idx=0) and we re-zero pad
        # positions on the way out, which achieves the same thing safely.
        causal = torch.triu(torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=causal)
        h = self.norm(h)
        return h * (seq != 0).unsqueeze(-1).to(h.dtype)


class SASRec(Recommender):
    family = "sasrec"

    def __init__(self, d: int = 64, n_heads: int = 2, n_blocks: int = 2, max_len: int = 50,
                 dropout: float = 0.2, lr: float = 1e-3, epochs: int = 12,
                 batch_size: int = 256, use_vertical: bool = True, weight_decay: float = 1e-5,
                 seed: int = 0, device: str = "cpu", max_seconds: float = 600.0):
        super().__init__(
            d=d, n_heads=n_heads, n_blocks=n_blocks, max_len=max_len, dropout=dropout,
            lr=lr, epochs=epochs, batch_size=batch_size, use_vertical=use_vertical,
            weight_decay=weight_decay, seed=seed,
        )
        self.device = device
        self.max_seconds = max_seconds
        tag = "+vert" if use_vertical else "no-vert"
        self.name = f"sasrec(d={d},L={n_blocks},{tag})"

    # ------------------------------------------------------------------ data
    def _build_tensors(self, data):
        L = int(self.params["max_len"])
        users = sorted(data.seqs.keys())
        seq = np.zeros((len(users), L), dtype=np.int64)
        vert = np.zeros((len(users), L), dtype=np.int64)
        pos = np.zeros((len(users), L), dtype=np.int64)
        for r, u in enumerate(users):
            s = data.seqs[u][-(L + 1):]
            v = data.seq_verticals[u][-(L + 1):]
            # inputs are s[:-1], targets are s[1:]; +1 shifts past the pad id
            inp, tgt = s[:-1] + 1, s[1:] + 1
            iv = v[:-1] + 1
            seq[r, L - len(inp):] = inp
            pos[r, L - len(tgt):] = tgt
            vert[r, L - len(iv):] = iv
        self.user_index_ = {u: r for r, u in enumerate(users)}
        return (torch.from_numpy(seq), torch.from_numpy(vert), torch.from_numpy(pos))

    # ------------------------------------------------------------------- fit
    def fit(self, data):
        self.data = data
        p = self.params
        torch.manual_seed(int(p["seed"]))
        np.random.seed(int(p["seed"]))
        torch.set_num_threads(max(1, torch.get_num_threads()))

        seq, vert, pos = self._build_tensors(data)
        self.seq_, self.vert_ = seq, vert

        net = _SASRecNet(
            n_items=data.n_items, n_verticals=len(data.verticals), d=int(p["d"]),
            n_heads=int(p["n_heads"]), n_blocks=int(p["n_blocks"]),
            max_len=int(p["max_len"]), dropout=float(p["dropout"]),
            use_vertical=bool(p["use_vertical"]),
        ).to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=float(p["lr"]),
                                weight_decay=float(p["weight_decay"]),
                                betas=(0.9, 0.98))
        bce = nn.BCEWithLogitsLoss(reduction="none")

        n = seq.shape[0]
        bs = int(p["batch_size"])
        rng = np.random.default_rng(int(p["seed"]))
        start = time.time()
        self.history_ = []

        for epoch in range(int(p["epochs"])):
            net.train()
            perm = rng.permutation(n)
            total, count = 0.0, 0
            for b in range(0, n, bs):
                idx = torch.from_numpy(perm[b:b + bs])
                s, v, t = seq[idx].to(self.device), vert[idx].to(self.device), pos[idx].to(self.device)
                h = net.encode(s, v)

                # one uniformly sampled negative per position (SASRec's loss)
                neg = torch.from_numpy(
                    rng.integers(1, data.n_items + 1, size=tuple(t.shape))
                ).to(self.device)
                mask = (t != 0).float()

                pe = net.item_emb(t)
                ne = net.item_emb(neg)
                pos_logit = (h * pe).sum(-1)
                neg_logit = (h * ne).sum(-1)

                loss = (bce(pos_logit, torch.ones_like(pos_logit)) * mask).sum()
                loss = loss + (bce(neg_logit, torch.zeros_like(neg_logit)) * mask).sum()
                loss = loss / mask.sum().clamp(min=1.0)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                total += float(loss) * int(mask.sum())
                count += int(mask.sum())
            self.history_.append(total / max(count, 1))
            if time.time() - start > self.max_seconds:
                self.history_.append(f"stopped early at epoch {epoch + 1} (time budget)")
                break

        net.eval()
        self.net_ = net
        with torch.no_grad():
            self.user_repr_ = self._encode_all()
        return self

    def _encode_all(self) -> torch.Tensor:
        """Final-position hidden state for every user: the user's current intent."""
        out = []
        bs = 512
        for b in range(0, self.seq_.shape[0], bs):
            s = self.seq_[b:b + bs].to(self.device)
            v = self.vert_[b:b + bs].to(self.device)
            h = self.net_.encode(s, v)[:, -1, :]
            out.append(h.cpu())
        return torch.cat(out, 0)

    # ----------------------------------------------------------------- score
    def score_users(self, users, candidates):
        rows = np.array([self.user_index_.get(int(u), -1) for u in users])
        item_emb = self.net_.item_emb.weight.detach()
        cand_emb = item_emb[torch.from_numpy(np.asarray(candidates) + 1)]
        H = torch.zeros((len(rows), self.user_repr_.shape[1]))
        ok = rows >= 0
        if ok.any():
            H[torch.from_numpy(np.where(ok)[0])] = self.user_repr_[torch.from_numpy(rows[ok])]
        return (H @ cand_emb.T).numpy()


class SessionAdaptiveSASRec(SASRec):
    """Streaming variant: re-encode the sequence *including* what the user has
    done so far in the current session, instead of serving a representation
    frozen at the last batch refresh.

    This is the cheap, deployable end of "online learning": model weights stay
    fixed, but the user representation is recomputed per request from a
    sequence that now includes in-session events. Comparing this arm against
    plain ``SASRec`` isolates the value of within-session adaptation from the
    value of retraining, which are usually - and wrongly - argued as one thing.
    """

    family = "sasrec_session"
    session_aware = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = self.name.replace("sasrec(", "sasrec_session(")

    def score_rows(self, rows: pd.DataFrame, candidates: np.ndarray) -> np.ndarray:
        L = int(self.params["max_len"])
        data = self.data
        vert_code = data.vert_code

        seq = np.zeros((len(rows), L), dtype=np.int64)
        vert = np.zeros((len(rows), L), dtype=np.int64)
        prefixes = rows["session_prefix"] if "session_prefix" in rows.columns else None

        for r, (u, v) in enumerate(zip(rows["u"].to_numpy(), rows["vertical"].to_numpy())):
            base = data.seqs.get(int(u), np.array([], dtype=np.int64))
            base_v = data.seq_verticals.get(int(u), np.array([], dtype=np.int64))
            if prefixes is not None:
                extra = prefixes.iloc[r]
                if len(extra):
                    base = np.concatenate([base, np.asarray(extra, dtype=np.int64)])
                    base_v = np.concatenate([base_v, np.full(len(extra), vert_code[v])])
            s = (base[-L:] + 1).astype(np.int64)
            sv = (base_v[-L:] + 1).astype(np.int64)
            seq[r, L - len(s):] = s
            vert[r, L - len(sv):] = sv

        with torch.no_grad():
            H = []
            bs = 512
            for b in range(0, len(rows), bs):
                h = self.net_.encode(
                    torch.from_numpy(seq[b:b + bs]).to(self.device),
                    torch.from_numpy(vert[b:b + bs]).to(self.device),
                )[:, -1, :]
                H.append(h.cpu())
            H = torch.cat(H, 0)
            cand_emb = self.net_.item_emb.weight.detach()[
                torch.from_numpy(np.asarray(candidates) + 1)
            ]
            return (H @ cand_emb.T).numpy()
