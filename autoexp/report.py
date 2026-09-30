"""Run artefacts: a JSON log, a leaderboard CSV, and a readable report.

The JSON log is the source of truth - every hypothesis, verdict, parameter and
statistic, enough to reconstruct or audit the run. The Markdown report is what
a person reads. Both come from the same object, so they cannot disagree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluate import PRIMARY_METRIC

VERDICT_MARK = {
    "supported": "**HELD**",
    "refuted": "**REFUTED**",
    "inconclusive": "INCONCLUSIVE",
    "underpowered": "UNDERPOWERED",
    None: "not evaluated",
}


def save_run(run, out_dir: Path | str = "runs") -> Path:
    """Write every artefact for one run into ``runs/<dataset>-<run_id>/``."""
    out = Path(out_dir) / f"{run.dataset}-{run.run_id}"
    out.mkdir(parents=True, exist_ok=True)

    (out / "run.json").write_text(
        json.dumps(run.to_json(), indent=2, default=_json_default), encoding="utf-8"
    )
    lb = run.leaderboard()
    if len(lb):
        lb.to_csv(out / "leaderboard.csv", index=False)
    (out / "report.md").write_text(render_markdown(run), encoding="utf-8")
    if run.narrative:
        (out / "narrative.md").write_text(run.narrative, encoding="utf-8")

    # Per-row scores for the champion, so a reader can audit any single user.
    if run.champion and run.champion in run.results:
        pr = run.results[run.champion].per_row
        if len(pr):
            pr.head(5000).to_csv(out / "champion_per_row_sample.csv", index=False)
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ------------------------------------------------------------------ markdown
def render_markdown(run) -> str:
    p = run.profile
    L: List[str] = []

    L.append(f"# Autonomous Experimenter — `{run.dataset}`\n")
    L.append(f"Run `{run.run_id}` · {len(run.rounds)} rounds · "
             f"{run.elapsed:.0f}s · narrator: {run.narrator_engine}\n")

    # ---- dataset
    L.append("## Dataset\n")
    L.append(f"- **Source**: {p.get('source')}")
    L.append(f"- **Scale**: {p['n_users']:,} users × {p['n_items']:,} items, "
             f"{p['n_interactions']:,} interactions "
             f"(density {p.get('density_pct')}%)")
    L.append(f"- **Verticals**: {', '.join(p['verticals'])} — "
             f"{p['multi_vertical_user_pct']}% of users active in 2 or more")
    L.append(f"- **Median training history**: {p.get('median_history')} events; "
             f"popularity Gini {p.get('gini_popularity')}")
    L.append(f"- **Evaluation rows**: {p.get('eval_rows'):,} "
             f"(transfer-only {p.get('pct_transfer_only')}%, "
             f"cold-in-vertical {p.get('pct_cold_in_vertical')}%, "
             f"tail target {p.get('pct_tail_target')}%)")
    if p.get("ground_truth"):
        L.append(f"- **Simulated with known parameters**: {p['ground_truth']} — "
                 f"results here test whether the method recovers a signal that is "
                 f"genuinely present, which is a claim about the method, not about "
                 f"real users.")
    L.append("")

    L.append("### Protocol\n")
    L.append("Per-user temporal leave-one-out: the last interaction is the test "
             "target, the second-to-last is validation, everything earlier trains. "
             "Every arm ranks the *same* candidate pool (the target vertical's "
             "catalogue) with the *same* already-seen masking applied by the "
             "evaluator, so no arm can win by filtering more aggressively. "
             f"Primary metric **{PRIMARY_METRIC}**; all comparisons are paired at "
             "the user level with a 2,000-sample bootstrap 95% CI.\n")

    # ---- the loop
    L.append("## What the agent did\n")
    for r in run.rounds:
        L.append(f"### Round {r.round} — {r.goal}\n")
        if r.reflection:
            L.append(f"*Why this round:* {r.reflection}\n")
        if r.actions:
            L.append(f"*Actions chosen:* `{'`, `'.join(r.actions)}`\n")
        L.append("| trial | model | params | why |")
        L.append("|---|---|---|---|")
        for t in r.trials:
            prm = ", ".join(f"{k}={v}" for k, v in t.params.items()) or "defaults"
            L.append(f"| `{t.id}` | `{t.label}` | `{prm}` | {t.motivation} |")
        L.append("")
        if r.judgements:
            L.append("**Verdicts**\n")
            for j in r.judgements:
                L.append(f"- {j}")
            L.append("")
        if r.decisions:
            L.append("**Decisions taken**\n")
            for d in r.decisions:
                L.append(f"- {d}")
            L.append("")

    # ---- hypotheses
    L.append("## Hypotheses\n")
    L.append("Each was written down *before* its trials ran, with the slice it "
             "applies to and the minimum relative effect that counts as support.\n")
    L.append("| id | claim | slice | required | verdict | evidence |")
    L.append("|---|---|---|---|---|---|")
    for h in run.hypotheses:
        L.append(
            f"| `{h.id}` | {h.statement} | `{h.slice}` | "
            f"{h.target_delta * 100:+.0f}% | {VERDICT_MARK.get(h.verdict, h.verdict)} | "
            f"{h.evidence or '—'} |"
        )
    L.append("")

    # ---- leaderboard
    lb = run.leaderboard()
    if len(lb):
        L.append("## Leaderboard\n")
        cols = [c for c in ["model", "family", PRIMARY_METRIC, "hit@10", "mrr",
                            f"{PRIMARY_METRIC}|transfer_only",
                            f"{PRIMARY_METRIC}|cold_in_vertical",
                            f"{PRIMARY_METRIC}|tail_target",
                            "coverage@10", "novelty@10", "fit_s"] if c in lb.columns]
        L.append(_md_table(lb[cols]))
        L.append("")
        L.append("`coverage@10` is the share of the catalogue that appears in any "
                 "top-10, and `novelty@10` the mean negative log popularity of "
                 "recommended items. They are reported next to accuracy because a "
                 "model that wins by serving the same head items to everyone is a "
                 "catalogue-collapse risk, not a ranking win.\n")

    # ---- decision layer
    d = run.decision or {}
    L.append("## Decision layer\n")
    if d.get("error"):
        L.append(f"_{d['error']}_\n")
    else:
        L.append(f"Champion: **`{d.get('champion')}`**\n")
        off = d.get("offline") or {}
        if off:
            L.append("### Offline, paired\n")
            L.append("| vs | Δ ndcg@10 | relative | 95% CI | p | significant | "
                     "users needed per arm |")
            L.append("|---|---|---|---|---|---|---|")
            for ref, c in off.items():
                pw = c.get("power", {})
                n = pw.get("required_n_per_arm")
                L.append(
                    f"| `{ref}` | {c.get('delta', float('nan')):+.4f} | "
                    f"{c.get('rel_lift_pct', float('nan')):+.1f}% | "
                    f"[{c.get('ci_low', float('nan')):+.4f}, "
                    f"{c.get('ci_high', float('nan')):+.4f}] | "
                    f"{c.get('p_value', float('nan')):.4f} | "
                    f"{'yes' if c.get('significant') else 'no'} | "
                    f"{n:,}" if isinstance(n, int) and n > 0 else "n/a"
                )
            L.append("")

        il = d.get("interleaving") or {}
        if il and "error" not in il:
            L.append("### Team-draft interleaving\n")
            L.append(f"Per-slate preference **{il['abs_lift']:+.4f}** "
                     f"(95% CI [{il['ci_low']:+.4f}, {il['ci_high']:+.4f}], "
                     f"p={il['p_value']:.4f}, "
                     f"{'significant' if il['significant'] else 'not significant'}) "
                     f"over `{il['champion']}` across {il['n_units']:,} slates.")
            for n in il.get("notes", []):
                L.append(f"- {n}")
            L.append("")

        ab = d.get("simulated_ab") or {}
        if ab.get("skipped"):
            L.append(f"### Online estimate\n\n_Not available: {ab['skipped']}_\n")
        elif ab and "error" not in ab:
            L.append("### Simulated online A/B\n")
            L.append(f"CTR@10 **{ab['control_mean']:.4f} → {ab['treatment_mean']:.4f}** "
                     f"({ab['rel_lift_pct']:+.1f}%), 95% CI "
                     f"[{ab['ci_low']:+.5f}, {ab['ci_high']:+.5f}], "
                     f"{'significant' if ab['significant'] else '**not** significant'} "
                     f"at n={ab['n_units']:,}.")
            if ab.get("required_n_per_arm", 0) > 0:
                L.append(f"A real split test would need ≈**{ab['required_n_per_arm']:,} "
                         f"users per arm** for 80% power at α=0.05 "
                         f"({ab.get('days_to_significance')} days at 200k daily users).")
            for n in ab.get("notes", []):
                L.append(f"- {n}")
            L.append("")

        conflicts = d.get("conflicts") or []
        if conflicts:
            L.append("### Where the measurements disagree\n")
            for c in conflicts:
                L.append(f"> **{c}**\n")

        fi = d.get("feature_importance")
        if fi:
            L.append("### What the ranker actually used\n")
            L.append("Permutation importance (AUC) on held-out rows.\n")
            L.append(_md_table(pd.DataFrame(fi)))
            L.append("")

        ts = d.get("top_cross_vertical_signals")
        if ts:
            L.append("### Strongest cross-vertical signals\n")
            L.append("Category pairs in *different* verticals whose co-purchase "
                     "exceeds independence by the largest margin — the "
                     "interpretable form of the transfer effect.\n")
            L.append(_md_table(pd.DataFrame(ts)))
            L.append("")

    # ---- narrative
    if run.narrative:
        L.append("---\n")
        L.append(f"## Readout (generated by {run.narrator_engine})\n")
        L.append(run.narrative)
        L.append("")

    # ---- reproducibility
    L.append("---\n")
    L.append("## Run metadata\n")
    L.append(f"- Stopped because: {run.stopped_because}")
    L.append(f"- Wall clock: {run.elapsed:.0f}s")
    if run.planner_decisions:
        L.append("- Planner decisions per round:")
        for pd_ in run.planner_decisions:
            chosen = pd_.get("llm_chosen") or pd_.get("rule_based")
            src = "LLM" if pd_.get("llm_chosen") else "rules"
            L.append(f"  - round {pd_['round']}: `{chosen}` ({src})")
    if run.notes:
        L.append("- Notes:")
        for n in run.notes:
            L.append(f"  - {n}")
    return "\n".join(L)


def _md_table(df: pd.DataFrame) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, rule] + rows)
