"""Streamlit browser for completed runs.

    streamlit run app.py

Reads the `run.json` written by `autoexp.report.save_run`, so it shows exactly
what the agent recorded - no recomputation, nothing that can drift from the
report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RUNS = Path("runs")
PRIMARY = "ndcg@10"

VERDICT_STYLE = {
    "supported": ("HELD", "#0f7b3f", "#e6f4ec"),
    "refuted": ("REFUTED", "#a3232b", "#fbeaea"),
    "inconclusive": ("INCONCLUSIVE", "#8a6d1f", "#fdf4e0"),
    "underpowered": ("UNDERPOWERED", "#4a4a8a", "#ececf6"),
}

st.set_page_config(page_title="Autonomous Experimenter", layout="wide")


@st.cache_data(show_spinner=False)
def find_runs() -> list[Path]:
    return sorted(RUNS.glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)


@st.cache_data(show_spinner=False)
def load_run(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


runs = find_runs()
if not runs:
    st.title("Autonomous Experimenter")
    st.warning("No runs found. Generate one first:")
    st.code("python -m autoexp --dataset careem_sim", language="bash")
    st.stop()

with st.sidebar:
    st.header("Runs")
    choice = st.radio(
        "Completed runs", runs, format_func=lambda p: p.parent.name, label_visibility="collapsed"
    )
    st.caption("Newest first. Each is a full experiment log.")

run = load_run(str(choice))
prof = run["profile"]

st.title("Autonomous Experimenter")
st.caption(
    f"`{run['dataset']}` · run `{run['run_id']}` · {len(run['rounds'])} rounds · "
    f"{run['elapsed_seconds']:.0f}s · narrator: {run['narrator_engine']}"
)

# ----------------------------------------------------------------- headline
lb = pd.DataFrame(run["leaderboard"])
champ, base = run.get("champion"), run.get("baseline")
c_val = b_val = None
if len(lb) and PRIMARY in lb.columns:
    if champ in set(lb["model"]):
        c_val = float(lb.loc[lb["model"] == champ, PRIMARY].iloc[0])
    if base in set(lb["model"]):
        b_val = float(lb.loc[lb["model"] == base, PRIMARY].iloc[0])

k = st.columns(5)
k[0].metric("Champion", champ or "—")
k[1].metric(f"{PRIMARY}", f"{c_val:.4f}" if c_val is not None else "—",
            f"{100 * (c_val - b_val) / b_val:+.1f}% vs floor"
            if (c_val and b_val) else None)
k[2].metric("Users x items", f"{prof['n_users']:,} x {prof['n_items']:,}")
k[3].metric("Multi-vertical users", f"{prof['multi_vertical_user_pct']}%")
k[4].metric("Transfer-only rows", f"{prof.get('pct_transfer_only', 0)}%")

st.info(f"**Stopped because:** {run['stopped_because']}")

tabs = st.tabs(["Hypotheses", "Leaderboard", "The loop", "Decision layer", "Readout", "Raw log"])

# --------------------------------------------------------------- hypotheses
with tabs[0]:
    st.caption(
        "Each claim was written down **before** its trials ran, with the slice it "
        "applies to and the minimum relative effect that counts as support. "
        "Support requires that threshold *and* a bootstrap CI excluding zero."
    )
    for h in run["hypotheses"]:
        label, fg, bg = VERDICT_STYLE.get(h["verdict"], ("OPEN", "#555", "#eee"))
        st.markdown(
            f"<div style='background:{bg};border-left:4px solid {fg};"
            f"padding:10px 14px;margin-bottom:10px;border-radius:4px'>"
            f"<b style='color:{fg}'>{label}</b> &nbsp;<code>{h['id']}</code>"
            f"<div style='margin-top:6px'>{h['statement']}</div>"
            f"<div style='margin-top:6px;font-size:0.85em;color:#444'>"
            f"slice <code>{h['slice']}</code> · required "
            f"{h['required_relative_lift'] * 100 if 'required_relative_lift' in h else h.get('target_delta', 0) * 100:+.0f}% · "
            f"{h['treatment']} vs {h['control'] or 'absolute threshold'}</div>"
            f"<div style='margin-top:6px;font-size:0.85em'><i>{h['evidence'] or 'not evaluated'}</i></div>"
            f"<div style='margin-top:6px;font-size:0.8em;color:#666'>{h['rationale']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# -------------------------------------------------------------- leaderboard
with tabs[1]:
    if len(lb):
        slice_cols = [c for c in lb.columns if "|" in c]
        show = [c for c in ["model", "family", PRIMARY, "hit@10", "mrr"] + slice_cols
                + ["coverage@10", "novelty@10", "fit_s"] if c in lb.columns]
        st.dataframe(
            lb[show].style.background_gradient(subset=[PRIMARY], cmap="Greens"),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "`coverage@10` is the share of the catalogue appearing in any top-10; "
            "`novelty@10` the mean negative log popularity of recommended items. "
            "They sit next to accuracy because a model that wins by serving the same "
            "head items to everyone is a catalogue-collapse risk, not a ranking win."
        )
        if slice_cols:
            st.subheader("By slice")
            m = lb.set_index("model")[[PRIMARY] + slice_cols]
            m.columns = [c.split("|")[-1] if "|" in c else "overall" for c in m.columns]
            st.bar_chart(m)

# ------------------------------------------------------------------- loop
with tabs[2]:
    for r in run["rounds"]:
        with st.expander(f"Round {r['round']} — {r['goal']}", expanded=r["round"] <= 2):
            if r.get("reflection"):
                st.markdown(f"*Why this round:* {r['reflection']}")
            if r.get("actions"):
                st.markdown("Actions: " + " ".join(f"`{a}`" for a in r["actions"]))
            st.dataframe(
                pd.DataFrame(r["trials"])[["id", "kind", "params", "motivation"]],
                use_container_width=True, hide_index=True,
            )
            if r.get("judgements"):
                st.markdown("**Verdicts**")
                for j in r["judgements"]:
                    st.markdown(f"- {j}")
            if r.get("decisions"):
                st.markdown("**Decisions taken**")
                for d in r["decisions"]:
                    st.warning(d) if "GUARDRAIL" in d else st.markdown(f"- {d}")
    if run.get("planner_decisions"):
        st.subheader("Who chose each round")
        st.dataframe(pd.DataFrame(run["planner_decisions"]),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------- decision
with tabs[3]:
    d = run.get("decision") or {}
    if d.get("error"):
        st.warning(d["error"])
    off = d.get("offline") or {}
    if off:
        st.subheader("Offline, paired at user level")
        rows = []
        for ref, c in off.items():
            pw = c.get("power", {})
            rows.append({
                "vs": ref, f"delta {PRIMARY}": c.get("delta"),
                "relative %": c.get("rel_lift_pct"),
                "CI low": c.get("ci_low"), "CI high": c.get("ci_high"),
                "p": c.get("p_value"), "significant": c.get("significant"),
                "users needed / arm": pw.get("required_n_per_arm"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    il = d.get("interleaving") or {}
    with c1:
        st.subheader("Team-draft interleaving")
        if il and "error" not in il:
            st.metric("Per-slate preference", f"{il['abs_lift']:+.4f}",
                      "significant" if il["significant"] else "not significant")
            st.caption(f"95% CI [{il['ci_low']:+.4f}, {il['ci_high']:+.4f}], "
                       f"p={il['p_value']:.4f}, {il['n_units']:,} slates "
                       f"vs `{il['champion']}`")
            for n in il.get("notes", []):
                st.caption(f"· {n}")
        else:
            st.caption(il.get("error", "not run"))

    ab = d.get("simulated_ab") or {}
    with c2:
        st.subheader("Simulated online A/B")
        if ab.get("skipped"):
            st.caption(f"Not available: {ab['skipped']}")
        elif ab and "error" not in ab:
            st.metric("CTR@10", f"{ab['treatment_mean']:.4f}",
                      f"{ab['rel_lift_pct']:+.1f}% vs {ab['control_mean']:.4f}")
            st.caption(f"95% CI [{ab['ci_low']:+.5f}, {ab['ci_high']:+.5f}] · "
                       + ("significant" if ab["significant"] else "**not** significant")
                       + f" at n={ab['n_units']:,}")
            if ab.get("required_n_per_arm", 0) > 0:
                st.caption(f"A real split test needs ~{ab['required_n_per_arm']:,} "
                           f"users per arm for 80% power.")
            for n in ab.get("notes", []):
                st.caption(f"· {n}")
        else:
            st.caption(ab.get("error", "not run"))

    if d.get("feature_importance"):
        st.subheader("What the ranker actually used")
        fi = pd.DataFrame(d["feature_importance"])
        st.bar_chart(fi.set_index("feature")["importance"])
    if d.get("top_cross_vertical_signals"):
        st.subheader("Strongest cross-vertical signals")
        st.caption("Category pairs in different verticals whose co-purchase exceeds "
                   "independence by the largest margin.")
        st.dataframe(pd.DataFrame(d["top_cross_vertical_signals"]),
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- readout
with tabs[4]:
    if run.get("narrative"):
        st.caption(f"Generated by {run['narrator_engine']}")
        st.markdown(run["narrative"])
    else:
        st.caption("No narrative in this run.")
    rp = Path(choice).parent / "report.md"
    if rp.exists():
        st.download_button("Download report.md", rp.read_text(encoding="utf-8"),
                           file_name="report.md")

with tabs[5]:
    st.json(run, expanded=False)
