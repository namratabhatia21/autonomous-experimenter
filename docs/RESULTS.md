# Results — does behaviour on one surface make the others smarter?

An agent pre-registered nine hypotheses about cross-vertical recommendation, ran the
ablations that could refute them, and judged each against a threshold it committed to
**before** the trials ran. Three held.

| | |
|---|---|
| Hypotheses | 9 — **3 held, 5 refuted, 1 underpowered** |
| Rounds | 4, self-paced (the agent chose each from the last round's verdicts) |
| Models evaluated | 13 |
| Wall clock | 10 min 09 s (`careem_sim`, CPU only) |
| Evaluation bugs found and fixed | 5 — see [Things that went wrong](#things-that-went-wrong) |

> A designed version of this page is at [`docs/ledger.html`](ledger.html), with the
> confidence intervals drawn against their thresholds. Enable GitHub Pages on `/docs`
> to view it in a browser.

---

## The setup

Three verticals — **Food**, **Quik**, **Shops** — and one question that decides whether a
shared personalization layer is worth building: when a user is cold on the surface being
ranked but active elsewhere, does their other behaviour help?

**Protocol, identical for every arm.** Per-user temporal leave-one-out; the last interaction
is the test target, the second-to-last is validation, everything earlier trains. Every arm
ranks the *same* candidate pool with the *same* already-seen masking, applied by the
evaluator rather than the model, so no arm can win by filtering harder. Primary metric
NDCG@10; every comparison paired at user level with a 2,000-sample bootstrap 95% CI.

**How a verdict is decided.** The whole confidence interval is compared against the bar the
hypothesis committed to in advance. Entirely past it is support; entirely short of it is
refutation; straddling it means the data cannot separate the two. One rule handles both
improvement claims and no-harm guardrails with no direction special-casing.

---

## The verdict ledger

`careem_sim` — 5,720 users × 1,499 items, 91,548 interactions, 72.6% of users active in
two or more verticals.

| Hypothesis | Claim | Slice | Bar | Effect | 95% CI | Verdict |
|---|---|---|---|---|---|---|
| `H1-personalization` | Personalised retrieval beats the popularity floor | `all` | +10% | **+46.1%** | [+41.8%, +50.5%] | ✅ **HELD** |
| `H2-transfer` | Cross-vertical history helps users cold on the ranked surface | `transfer_only` | +5% | **+15.7%** | [+9.0%, +23.2%] | ✅ **HELD** |
| `H2b-no-harm` | Transfer does not degrade single-vertical users | `single_vertical_user` | −2% | **+3.3%** | [+1.8%, +5.1%] | ✅ **HELD** |
| `H3-kg` | Knowledge-graph attribute edges rescue tail items | `tail_target` | +5% | — | — | ⚠️ **UNDERPOWERED** |
| `H4-attention` | A self-attentive sequential model beats the best retriever | `all` | +3% | −35.0% | [−38.0%, −32.1%] | ❌ **REFUTED** |
| `H4b-vertical-context` | A vertical embedding helps the transformer transfer | `transfer_only` | +3% | −1.4% | [−4.3%, +1.6%] | ❌ **REFUTED** |
| `H5-fusion` | A learned second stage beats its own best input | `all` | +2% | −0.7% | [−1.4%, −0.0%] | ❌ **REFUTED** |
| `H5b-transfer-features` | Transfer features earn their place inside the ranker | `transfer_only` | +2% | −0.5% | [−2.6%, +1.5%] | ❌ **REFUTED** |
| `H6-session` | In-session re-encoding beats a batch-frozen representation | `all` | +2% | −0.3% | [−1.1%, +0.5%] | ❌ **REFUTED** |

Most claims did not survive. That is the expected shape of an honest bake-off.

`H3-kg` is reported as **underpowered rather than null** because only 25 of 1,382 tail rows
were ranked in the top-10 by either arm — the comparison could not have detected the effect
it was looking for, which is a different statement from "there is no effect". Reporting that
as "inconclusive" with a CI of [0.0000, 0.0000] would have hidden it.

---

## The headline: where the two datasets disagree

The same hypothesis, the same code, two logs.

| | `careem_sim` (simulated, ρ = 0.55 known) | `amazon_xvert` (real, 35,229 users) |
|---|---|---|
| Effect on `transfer_only` | **+15.7%** | **−1.9%** |
| 95% CI | [+9.0%, +23.2%] | [−7.3%, +3.3%] |
| Bar | +5% | +5% |
| n | 679 | 12,715 |
| Verdict | ✅ **HELD** | ❌ **REFUTED** |

With transfer genuinely present in the generative process, the method recovers it — that
validates the **method**, not user behaviour. On real Amazon cross-category purchases the
interval **excludes the +5% bar entirely**: graph-walk transfer does not deliver the lift it
delivers in simulation.

Both runs cleared the no-harm guardrail, so transfer costs single-surface users nothing — it
simply does not pay on this real log. The honest reading is that the mechanism works when the
signal is there, and on Amazon cross-category co-purchase the signal is too weak to clear a
5% bar. A marketplace with genuinely shared intent across verticals is a different question,
and the simulator shows what the answer would look like if it were.

**Scope of the Amazon run:** rounds 1–2 completed (baselines and the transfer ablation,
35,207 evaluation rows). Rounds 3–4 — the transformer, the fusion ranker and session
adaptation — were cut short on CPU budget, so those four hypotheses carry simulator evidence
only. Stated here rather than left for a reader to notice.

---

## Leaderboard

`careem_sim`. Coverage sits next to accuracy on purpose: a model that wins by serving the
same head items to everyone is a catalogue-collapse risk, not a ranking win.

| Model | Family | NDCG@10 | Hit@10 | transfer_only | cold_in_vertical | Coverage@10 | Fit |
|---|---|---|---|---|---|---|---|
| **graph_walk(xvert,L=3)** | graph_walk | **0.2283** | 0.3091 | 0.1502 | 0.1397 | 0.703 | 0.7s |
| graph_walk(xvert+kg,L=5) | graph_walk | 0.2277 | 0.3075 | 0.1475 | 0.1386 | 0.688 | 0.9s |
| graph_walk(xvert+kg,L=3) | graph_walk | 0.2272 | 0.3080 | 0.1437 | 0.1328 | 0.774 | 0.8s |
| two_stage(2src,+transfer) | two_stage | 0.2267 | 0.3093 | 0.1562 | 0.1463 | 0.463 | 40s |
| two_stage(2src,no-transfer) | two_stage | 0.2264 | 0.3101 | 0.1570 | 0.1476 | 0.450 | 42s |
| graph_walk(within,L=3) | graph_walk | 0.2240 | 0.3017 | 0.1299 | 0.1078 | 0.514 | 0.8s |
| bpr_mf(d=64) | bpr_mf | 0.1563 | 0.2656 | 0.1410 | 0.1354 | 0.244 | 28s |
| popularity | popularity | 0.1533 | 0.2642 | 0.1501 | 0.1446 | 0.064 | 0.0s |
| sasrec(d=64,L=2,no-vert) | sasrec | 0.1502 | 0.2594 | 0.1477 | 0.1410 | 0.075 | 110s |
| sasrec(d=64,L=2,+vert) | sasrec | 0.1483 | 0.2566 | 0.1456 | 0.1404 | 0.075 | 115s |
| sasrec_session(d=64,L=2,+vert) | sasrec_session | 0.1479 | 0.2573 | 0.1415 | 0.1370 | 0.074 | 112s |
| item_knn(k=150) | item_knn | 0.1419 | 0.2622 | 0.1442 | 0.1390 | 0.203 | 0.4s |
| cross_vertical_bridge | cross_vertical_bridge | 0.1084 | 0.2054 | 0.1362 | 0.1330 | 0.108 | 0.0s |

The transformer lands **below the popularity floor** on overall NDCG. With a median history
of 13 events there is little sequence for attention to exploit, and its top-10 covers 7.5% of
the catalogue against the graph walk's 70%. Reported as a refutation, not omitted.

---

## How the agent spent its budget

Rounds are not a fixed schedule — each is chosen from the previous round's verdicts.

**Round 1 — establish the floor and the classical baselines** (`baselines`)
Buys the reference points every later claim is measured against.
→ pruned `item_knn` (37% below the leader), `popularity` (32%), `bpr_mf` (30%)

**Round 2 — test cross-vertical transfer as a one-factor ablation** (`cross_vertical_ablation`)
72.6% of users are active in two or more verticals, so there is a real population for
transfer to serve. The two arms differ in exactly one term.
→ pruned `cross_vertical_bridge` (53% below the leader)

**Round 3 — separate KG edges from walk depth, and test attention** (`graph_depth_and_kg`, `sequential`)
Transfer was supported, so the next question is which graph structure carries it — and
whether attention over the cross-vertical sequence pays for itself.
→ leaderboard did not move, but two mechanisms remained untested, so the run continued

**Round 4 — fuse the surviving retrievers; isolate within-session adaptation** (`fusion`, `session_adaptation`)
Retrieval questions are settled; what remains is how to combine them, and whether in-session
re-encoding is worth it separately from retraining.
→ pruned `sasrec` (34%), `sasrec_session` (35%); stopped: reached the 4-round budget

### The stopping rule, in its own words

> "Round 3 did not move the leaderboard, but `['fusion', 'session_adaptation']` still test
> untested mechanisms, so the run continued."

A plateau rule alone would have ended the run at round 3 with the fusion and
session-adaptation hypotheses never tested. A hypothesis-driven agent stops when it runs out
of **questions**, not when a number stops moving.

---

## From offline win to launch decision

| Comparison | Δ NDCG@10 | Relative | 95% CI | Significant | Users / arm |
|---|---|---|---|---|---|
| champion vs popularity floor | +0.0750 | +48.9% | [+0.0686, +0.0815] | yes | 323 |
| champion vs BPR-MF | +0.0721 | +46.1% | [+0.0655, +0.0785] | yes | 353 |
| team-draft interleaving | +0.0991 | per-slate preference | [+0.0850, +0.1133] | yes | 5,720 slates |
| simulated online A/B (CTR@10) | 0.1225 → 0.1146 | −6.4% | [−0.0142, −0.0008] | yes | 3,031 |

### The conflict the agent raised itself

Interleaving and the simulated A/B **disagree in sign**: +0.0991 per-slate preference against
−6.4% CTR. Mean true utility of the full slate is lower for the challenger (+0.0067 against
+0.0549).

The champion is better at placing the **single next item** near the top and worse at filling
the other nine slots. **NDCG@10 on one held-out item is not slate quality** — and a readout
that reported the offline win without naming this would have shipped a regression.

---

## Things that went wrong

Each was found by a result that looked plausible and was not. All five are regression-tested
in [`tests_smoke.py`](../tests_smoke.py).

**1. Optimistic tie-breaking scored no-signal users as perfect.**
Rank was counted as strictly-better + 1. A model returning an all-zero score row — a user it
has no signal for — ties with every candidate and was handed **rank 1, a perfect NDCG**. This
inflated the popularity floor, which has large tie groups by construction, and the transfer
control arm. *This single bug was flipping the sign of the headline result.*

**2. The transfer ablation measured dilution, not transfer.**
Toggling cross-vertical graph edges with one seed over the whole history let walk mass leak
out of the target vertical through high-degree items, so transfer arrived as popularity-biased
noise *replacing* good signal (coverage rose to 0.75 while accuracy fell). It was also
non-monotone, so it could never measure transfer's incremental value. *−47% → +15.7% after
decomposing the seed.*

**3. Two evaluation protocols silently disagreed.**
The two-stage ranker chose its blend prior without masking already-seen items while the
evaluator masked them. A random walk restarts on the user's own history, so unmasked it looks
like the *worst* retriever and masked it is the best — validation picked the wrong prior on
that discrepancy. *Masking now lives in one function every ranking path calls.*

**4. Hard-negative mining inverted the ranker.**
Mining negatives from the fused top-200 trained a ranker with held-out **AUC 0.40** — worse
than chance. The held-out target often sits below the top-200, so every "hard" negative
outranked it and the model learned *high retrieval score ⇒ negative*. *AUC 0.40 → 0.75, plus
an assertion that refuses to return an inverted ranker.*

**5. Budget discipline deleted the experiment.**
Pruning families on round-1 rank removed the within-vertical graph walk on the Amazon log —
which is the **control arm** of the cross-vertical hypothesis. The headline question came back
as "an arm failed to run". *Pruning now protects any family whose own experiment has not
happened yet: a weak arm can still be the right control.*

Also fixed along the way: NaN scores won (fully-masked attention rows produced NaN, which
fails every `>` comparison, so a diverged transformer scored a perfect NDCG@10 of 1.0000 — the
evaluator now refuses non-finite scores); a no-harm hypothesis was refuted by a **+3.7%**
result (the verdict rule conflated "effect exceeds the bar in magnitude" with "effect in the
wrong direction"); KG edges carried only 10% of walk mass when asked for 35%, making the KG
ablation return exactly +0.0000; and iALS took 108s where BPR-MF takes 5.3s because this
environment's LAPACK solves a 64×64 system in ~2.7ms.

---

## Reproduce

```bash
pip install -r requirements.txt

python -m autoexp --dataset careem_sim     # ~10 min, no downloads
python -m autoexp --dataset amazon_xvert   # real HF data, ~1GB once
python -m autoexp --dataset both --llm     # + LLM-written readout

streamlit run app.py                       # browse any completed run
python tests_smoke.py                      # the five bug regressions
```

Each run writes `report.md`, `run.json` (every hypothesis, verdict and statistic — the
auditable log), `leaderboard.csv` and `narrative.md` into `runs/<dataset>-<timestamp>/`.

The LLM narrator is an upgrade, not a dependency: with no API key a deterministic narrator
renders the same facts from the same payload, which also makes the model's output diffable
against a known-correct version of itself.

---

**Data:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(5-core, rating-only; Grocery → food, Health & Household → quik, Beauty → shops) plus a
bundled multi-vertical simulator with known transfer strength. 25.8% of the 1.46M users in
those categories buy across two or more of them.
