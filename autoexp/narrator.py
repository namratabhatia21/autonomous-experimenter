"""Turning the experiment log into a readout a team can act on.

The narrator is given the structured run - profile, hypotheses and their
verdicts, leaderboard, slice breakdowns, guardrails, and the decision-layer
statistics - and asked to write the conclusion. Three design choices matter:

  * **It is only ever given numbers the run actually produced.** The prompt
    carries no free-text summary for the model to embroider, and it is told
    explicitly not to introduce figures that are not in the payload. An LLM
    that invents a lift number in an experiment readout is worse than no
    readout at all.

  * **The hypothesis verdicts are the spine of the narrative**, not the
    leaderboard. That keeps the write-up about what was learned rather than
    which row came top.

  * **There is a deterministic fallback.** ``TemplateNarrator`` produces the
    same structure from the same payload with no API key, so the repository
    runs end-to-end for anyone and the LLM is an upgrade, not a dependency.

``prompts/narrator.md`` holds the prompt on its own - it is also the answer to
challenge 3 (Experiment Narrator) as a standalone artefact.
"""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "narrator.md"

DEFAULT_MODEL = "claude-opus-5"


class ClaudeClient:
    """Thin wrapper so the planner and the narrator share one client."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        from anthropic import Anthropic

        self.model = model
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str, max_tokens: int = 2000,
                 system: Optional[str] = None) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a senior data scientist writing an experiment readout.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


# --------------------------------------------------------------- payload
def build_payload(run) -> Dict[str, Any]:
    """The only facts the narrator is allowed to use."""
    lb = run.leaderboard()
    keep = [c for c in ["model", "family", "ndcg@10", "hit@10", "mrr", "coverage@10",
                        "novelty@10", "ndcg@10|transfer_only", "ndcg@10|cold_in_vertical",
                        "ndcg@10|tail_target", "fit_s"] if c in lb.columns]
    champ = run.results.get(run.champion) if run.champion else None
    base = run.results.get(run.baseline) if run.baseline else None

    payload: Dict[str, Any] = {
        "dataset": {
            "name": run.dataset,
            "source": run.profile.get("source"),
            "n_users": run.profile.get("n_users"),
            "n_items": run.profile.get("n_items"),
            "n_interactions": run.profile.get("n_interactions"),
            "verticals": run.profile.get("verticals"),
            "multi_vertical_user_pct": run.profile.get("multi_vertical_user_pct"),
            "median_history": run.profile.get("median_history"),
            "pct_transfer_only": run.profile.get("pct_transfer_only"),
            "pct_tail_target": run.profile.get("pct_tail_target"),
            "ground_truth": run.profile.get("ground_truth"),
        },
        "protocol": {
            "split": "per-user temporal leave-one-out (last=test, second-last=validation)",
            "primary_metric": "ndcg@10",
            "candidate_pool": "target vertical's catalogue, already-seen items masked",
            "significance": "paired user-level bootstrap, 2000 resamples, 95% CI",
        },
        "rounds": [
            {
                "round": r.round, "goal": r.goal, "actions": r.actions,
                "why_this_round": r.reflection,
                "judgements": r.judgements, "decisions": r.decisions,
            } for r in run.rounds
        ],
        "hypotheses": [
            {
                "id": h.id, "statement": h.statement, "rationale": h.rationale,
                "slice": h.slice, "required_relative_lift": h.target_delta,
                "treatment": h.treatment, "control": h.control,
                "verdict": h.verdict, "evidence": h.evidence,
            } for h in run.hypotheses
        ],
        "leaderboard": json.loads(lb[keep].to_json(orient="records")) if len(lb) else [],
        "champion": run.champion,
        "baseline": run.baseline,
        "decision_layer": run.decision,
        "stopped_because": run.stopped_because,
        "elapsed_seconds": round(run.elapsed, 1),
        "planner_decisions": run.planner_decisions,
        "notes": run.notes,
    }
    if champ and base:
        payload["champion_vs_baseline_slices"] = {
            s: {
                "champion": champ.slice_metrics.get(s, {}).get("ndcg@10"),
                "baseline": base.slice_metrics.get(s, {}).get("ndcg@10"),
                "n": champ.slice_metrics.get(s, {}).get("n"),
            }
            for s in champ.slice_metrics
        }
    return payload


def load_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return _FALLBACK_PROMPT


_FALLBACK_PROMPT = """Write an experiment readout from the JSON below.
Use only numbers present in the JSON. Structure: headline, what was tested,
what held and what did not, what to ship, what to run next, risks."""


# -------------------------------------------------------------- narrators
def narrate(run, client: Optional[ClaudeClient] = None) -> tuple[str, str]:
    """Return ``(markdown, engine)``. Falls back to the template narrator."""
    payload = build_payload(run)
    if client is not None:
        try:
            prompt = load_prompt().replace("{{RUN_JSON}}", json.dumps(payload, indent=2, default=str))
            text = client.complete(prompt, max_tokens=3000)
            if text and len(text.strip()) > 200:
                return text.strip(), f"claude:{client.model}"
        except Exception as exc:                            # noqa: BLE001
            run.notes.append(f"LLM narrator unavailable ({exc}); used the template narrator.")
    return TemplateNarrator(payload).render(), "template (deterministic)"


class TemplateNarrator:
    """Deterministic narrator. Same payload, same structure, no API key.

    It exists so the repository is reproducible and so the LLM's output can be
    diffed against a known-correct rendering of the same facts - which is the
    cheapest available check that the model did not invent anything.
    """

    def __init__(self, payload: Dict[str, Any]):
        self.p = payload

    def render(self) -> str:
        p = self.p
        d = p["dataset"]
        parts: List[str] = []

        champ = p.get("champion")
        base = p.get("baseline")
        lb = {r["model"]: r for r in p["leaderboard"]}
        c_ndcg = lb.get(champ, {}).get("ndcg@10")
        b_ndcg = lb.get(base, {}).get("ndcg@10")
        lift = (100 * (c_ndcg - b_ndcg) / b_ndcg) if (c_ndcg and b_ndcg) else None

        parts.append("## Headline\n")
        if lift is not None:
            parts.append(
                f"On **{d['name']}** ({d['n_users']:,} users, {d['n_interactions']:,} "
                f"interactions across {', '.join(d['verticals'])}), the agent ran "
                f"{len(p['rounds'])} rounds and settled on **{champ}**, which scores "
                f"NDCG@10 {c_ndcg:.4f} against the popularity floor's {b_ndcg:.4f} "
                f"— a {lift:+.1f}% relative lift.\n"
            )
        parts.append(f"The run stopped because {p['stopped_because']}.\n")

        parts.append("\n## What was tested, and what held\n")
        for h in p["hypotheses"]:
            mark = {"supported": "HELD", "refuted": "REFUTED",
                    "inconclusive": "INCONCLUSIVE"}.get(h["verdict"], "OPEN")
            parts.append(f"- **[{mark}] {h['id']}** — {h['statement']}")
            parts.append(f"  - Evidence: {h['evidence'] or 'not evaluated'}")

        sup = [h for h in p["hypotheses"] if h["verdict"] == "supported"]
        ref = [h for h in p["hypotheses"] if h["verdict"] == "refuted"]

        parts.append("\n## Recommendation\n")
        if sup:
            parts.append(f"Ship **{champ}**. The claims that survived testing were: "
                         + "; ".join(h["id"] for h in sup) + ".")
        else:
            parts.append(f"Do not ship yet. No hypothesis cleared its pre-registered "
                         f"threshold, so the honest position is that {champ} is not "
                         f"distinguishable from the baseline on this evidence.")
        if ref:
            parts.append(f"\nStop investing in: " + "; ".join(h["id"] for h in ref) + ".")

        dl = p.get("decision_layer", {})
        il = dl.get("interleaving") or {}
        ab = dl.get("simulated_ab") or {}
        if il and "error" not in il:
            parts.append(
                f"\nInterleaving against {il.get('champion')}: "
                f"{il.get('abs_lift', 0):+.4f} per-slate preference, 95% CI "
                f"[{il.get('ci_low', 0):+.4f}, {il.get('ci_high', 0):+.4f}], "
                f"{'significant' if il.get('significant') else 'not significant'}."
            )
        if ab and "error" not in ab and "skipped" not in ab:
            parts.append(
                f"Simulated online A/B: CTR {ab.get('control_mean', 0):.4f} -> "
                f"{ab.get('treatment_mean', 0):.4f} ({ab.get('rel_lift_pct', 0):+.1f}%), "
                f"{'significant' if ab.get('significant') else 'not significant'} at this "
                f"sample size; a real test needs ~{ab.get('required_n_per_arm', 0):,} "
                f"users per arm."
            )
        elif ab.get("skipped"):
            parts.append(f"\nNo online estimate: {ab['skipped']}")

        parts.append("\n## Risks and what to run next\n")
        guard = [x for r in p["rounds"] for x in r.get("decisions", []) if "GUARDRAIL" in x]
        for g in guard:
            parts.append(f"- {g}")
        parts.append(
            "- Offline NDCG is a proxy. The interleaving result and the power analysis "
            "size the online test; nothing here replaces running it."
        )
        if d.get("ground_truth"):
            parts.append(
                f"- This dataset is simulated with known parameters "
                f"({d['ground_truth']}), so the transfer result is a check that the "
                f"method recovers a signal that is genuinely there — not evidence "
                f"about real user behaviour."
            )
        return "\n".join(parts)
