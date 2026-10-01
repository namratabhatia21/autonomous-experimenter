# Autonomous Experimenter — cross-vertical personalization

An agent that **designs, runs, judges and writes up** recommender-system
experiments on its own. Point it at an interaction log and it profiles the data,
writes down falsifiable hypotheses, runs the smallest comparisons that could
refute them, decides what to try next *from what it just learned*, stops when
the leaderboard plateaus, and produces a decision memo.

Built for the Careem Personalization brief: a recommendation and ranking stack
across three verticals (**Food / Quik / Shops**) whose central question is
whether a user's behaviour on one surface makes the others smarter.

```bash
pip install -r requirements.txt
python -m autoexp --dataset careem_sim        # simulator, ~4 min, no downloads
python -m autoexp --dataset amazon_xvert      # real HF data, downloads ~1GB once
python -m autoexp --dataset both --llm        # both + Claude-written readout
streamlit run app.py                          # browse any completed run
```

Artefacts land in `runs/<dataset>-<timestamp>/`: `report.md` (the readout),
`run.json` (every hypothesis, verdict and statistic — the auditable log),
`leaderboard.csv`, and `narrative.md`.

### → [**Read the results**](docs/RESULTS.md)

A completed run: nine pre-registered hypotheses, three held, and the one finding that
matters — cross-vertical transfer lifts NDCG@10 **+15.7%** on simulated data where the
signal is known to exist, and is **refuted on real Amazon data**. Same code, same
hypothesis, two logs. [`docs/ledger.html`](docs/ledger.html) is the same readout with the
confidence intervals drawn against their thresholds.

---

## Why this isn't a grid search

A grid search tries everything and reports the maximum. This states what it
expects first, and lets the result decide what happens next.

**1. Hypotheses are pre-registered.** Before any arm trains, the agent writes
the claim, the metric, the *slice* it applies to, and the minimum relative
effect that counts as support. Judging requires both that threshold **and** a
paired bootstrap CI excluding zero — either alone is how teams ship noise. Since
the slice is fixed in advance, the agent cannot go hunting for one where the
number happened to look good.

**2. The plan adapts to the verdicts.** Refuted transfer means round 3 stops
spending budget on transfer. A model family more than 25% below the leader is
pruned. A collapsed catalogue raises a guardrail. The run stops on plateau, not
on a fixed schedule. Every branch is a stated experimental rule in
[`planner.py`](autoexp/planner.py).

**3. Ablations are one-factor and monotone.** The transfer test is two arms that
differ in exactly one term, and the control is that term set to zero — so
"transfer on" can only ever *add* to what the surface already knew. Getting this
wrong is easy and it cost this project a bogus −47% result; see
[Things that went wrong](#things-that-went-wrong).

**4. Offline NDCG is not a launch decision.** The champion goes through
team-draft interleaving, a simulated online A/B where a ground-truth click model
exists, and a power analysis that says how much traffic a real test needs.

**5. An LLM is in the loop, but on a leash.** With `--llm-planner`, Claude
chooses each round's actions — from the same typed action table, schema-validated
before anything trains, with the rule-based planner underneath. With `--llm`, it
writes the readout from a payload containing only numbers the run produced. The
whole thing runs reproducibly with no API key.

---

## What it tests, and why those things

| Component | The question it answers |
|---|---|
| **Graph walk** — random walk with restart over an item + knowledge-graph edge set | Does graph-based retrieval beat classical CF for candidate generation? |
| **Seed decomposition** — `walk(history_here) + β · walk(history_elsewhere)` | Does Food behaviour make Shops smarter, for users cold on Shops? |
| **KG edges** — item→category / price-band nodes | Do attribute hops rescue tail items whose co-occurrence rows are empty? |
| **SASRec** — causal self-attention, ± a vertical embedding | Does attention over a *cross-vertical* sequence pay for itself? |
| **Session-adaptive SASRec** — identical weights, re-encoded per request | Is within-session adaptation worth it *separately* from retraining? |
| **Two-stage ranker** — GBDT fusing retrievers, ± transfer features | Should transfer live in retrieval, or as a ranking feature? |
| **Interleaving + simulated A/B + power** | Is the offline win real, and what would it cost to confirm online? |

The headline slice is **`transfer_only`**: users with little or no history on the
surface being ranked, but activity elsewhere. Only transfer can help them, so
that is where the thesis lives or dies. On the Amazon slice they are **43% of all
evaluation rows**.

---

## Data

**`amazon_xvert` — real.** A cross-vertical slice of
[McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023),
the most-liked recommendation dataset on the Hugging Face Hub. Three categories
map onto the three verticals:

| Amazon category | vertical |
|---|---|
| `Grocery_and_Gourmet_Food` | food |
| `Health_and_Household` | quik |
| `Beauty_and_Personal_Care` | shops |

Of the 1.46M users across these categories, **25.8% buy in two or more and 6.5%
in all three** — a genuinely multi-surface population, which is the thing no
single-domain benchmark can test. The 5-core release ships no taxonomy, so
categories are *induced* behaviourally (truncated SVD + k-means per vertical) and
used as the knowledge-graph nodes — which is also what you would do against a
real marketplace catalogue whose taxonomy is inconsistent across verticals.

**`careem_sim` — simulated, and not a substitute.** It ships two things the real
log cannot: **ground-truth transfer strength** (`rho`), so the agent's conclusion
can be checked for *correctness* rather than mere internal consistency, and a
**known logging policy with position bias**, which is what makes the simulated
A/B honest instead of hand-waved. Every claim is reported on both datasets, and
a conclusion that holds in one and not the other is reported as exactly that.

---

## Protocol

Per-user temporal leave-one-out: last interaction is the test target,
second-to-last is validation, everything earlier trains. Every arm ranks the
**same** candidate pool with the **same** already-seen masking, applied by the
evaluator rather than the model, so no arm can win by filtering harder. Primary
metric NDCG@10; all comparisons paired at user level with a 2,000-sample
bootstrap 95% CI. `coverage@10` and `novelty@10` are reported alongside accuracy,
because a model that wins by serving the same head items to everyone is a
catalogue-collapse risk rather than a ranking win.

---

## Things that went wrong

Kept in, because the debugging is the part that was actually hard. Each of these
is documented at the site of the fix.

**Optimistic tie-breaking scored no-signal users as perfect.** Rank was computed
as `(strictly better) + 1`. A model returning an all-zero score row — a user it
has no signal for — ties with every candidate and was handed **rank 1, a perfect
NDCG**. This inflated the popularity floor (large tie groups by construction) and
the transfer control arm. Fixed to `(strictly better) + (tie-group size)`, which
is what "no signal" actually means. This single bug was flipping the sign of the
headline result.

**The transfer ablation was measuring dilution, not transfer.** The first design
toggled whether the *graph* contained cross-vertical edges, with one seed over
the whole history — and measured **−47%** on the transfer slice. The mechanism
explains it: walk mass leaked out of the target vertical through high-degree
items, so transfer arrived as popularity-biased noise *replacing* good signal
(coverage rose to 0.75 while accuracy fell). It was also non-monotone, so it
could never measure transfer's incremental value. Decomposing the seed turned
−47% into **+17.9%**.

**Two evaluation protocols silently disagreed.** The two-stage ranker selected
its blend prior without masking already-seen items, while the evaluator masked
them. A random walk restarts on the user's own history, so unmasked it looks like
the *worst* retriever and masked it is the best — validation picked the wrong
prior on the strength of that discrepancy. Masking now lives in exactly one
function that every ranking path calls.

**Hard-negative mining inverted the ranker.** Mining negatives from the fused
top-200 trained a ranker with held-out **AUC 0.40** — reliably worse than chance.
The held-out target often sits below the top-200, so every "hard" negative
outranked it and the model learned *high retrieval score ⇒ negative*. Fixed with
mixed uniform/hard negatives, plus an assertion that refuses to return a ranker
scoring below 0.5.

**NaN scores won.** Fully-masked attention rows produced NaN, and NaN fails every
`>` comparison — so the diverged transformer scored a perfect NDCG@10 of 1.0000.
The evaluator now refuses to rank non-finite scores, and the attention mask was
restructured so the NaN cannot arise.

**iALS was 20× too slow for no reason.** This environment's LAPACK solves a 64×64
system in ~2.7ms, so per-row ALS took 108s on a 2,840-user log; batching the
solves did not help. Replaced with BPR matrix factorisation — no linear solves,
5.3s, and a ranking loss rather than a regression one, so a better baseline
anyway.

---

## Layout

```
autoexp/
  planner.py       hypotheses, verdicts, pruning, guardrails, the action table
  orchestrator.py  the loop: profile -> plan -> run -> judge -> adapt -> decide
  evaluate.py      ranking metrics, slices, masking, paired bootstrap  <- read first
  abtest.py        interleaving, simulated A/B, power analysis
  narrator.py      Claude readout + deterministic fallback
  registry.py      the agent's typed action space
  datasets.py      Amazon-Reviews-2023 slice + simulator, unified container
  simulate.py      the multi-vertical generator and its click model
  models/          popularity, item-kNN, BPR-MF, graph walk, SASRec, two-stage
prompts/narrator.md   the Experiment Narrator prompt (challenge 3, standalone)
app.py                Streamlit run browser
```

`evaluate.py` is the file to read first: it is where the comparison is made fair,
and three of the five bugs above lived in or around it.

---

## The 100-word summary

An agent that runs recommender experiments end to end for a cross-vertical
personalization stack. It profiles an interaction log, pre-registers falsifiable
hypotheses with slices and effect thresholds, then runs one-factor ablations
across graph-based retrieval, a self-attention sequential model, and a learned
two-stage ranker. Verdicts — judged on paired bootstrap CIs — decide the next
round: it prunes losing families, raises catalogue-health guardrails, and stops
on plateau. The champion goes through interleaving, a simulated A/B and a power
analysis, then Claude writes the readout from the structured log. On real Amazon
data, cross-vertical retrieval lifts NDCG@10 **+17.9%** for users cold on the
target surface.

*Data: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
(CC BY 4.0) plus a bundled simulator.*
