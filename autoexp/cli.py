"""Command line entry point.

    python -m autoexp --dataset careem_sim
    python -m autoexp --dataset amazon_xvert --rounds 4 --llm
    python -m autoexp --dataset both
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from . import datasets
from .narrator import ClaudeClient, narrate
from .orchestrator import AutonomousExperimenter
from .report import save_run


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="autoexp",
        description="An agent that plans, runs and judges recommender experiments, "
                    "then writes the readout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default="careem_sim",
                    choices=sorted(datasets.REGISTRY) + ["both"],
                    help="which log to experiment on (default: careem_sim)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="maximum experiment rounds (the agent may stop earlier)")
    ap.add_argument("--time-budget", type=float, default=2400.0,
                    help="seconds before the agent stops planning new rounds")
    ap.add_argument("--out", default="runs", help="output directory for artefacts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", action="store_true",
                    help="use Claude for the narrative readout (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--llm-planner", action="store_true",
                    help="also let Claude choose each round's actions, from the "
                         "validated action table (falls back to rules on any error)")
    ap.add_argument("--sim-users", type=int, default=6000,
                    help="simulated users, for --dataset careem_sim")
    ap.add_argument("--rho", type=float, default=0.55,
                    help="ground-truth cross-vertical taste correlation in the simulator")
    ap.add_argument("--amazon-users", type=int, default=40000,
                    help="cross-vertical users to sample from Amazon-Reviews-2023")
    ap.add_argument("--quiet", action="store_true")
    return ap


def load_one(name: str, args) -> "datasets.RecData":
    if name == "careem_sim":
        return datasets.load("careem_sim", n_users=args.sim_users,
                             rho=args.rho, seed=args.seed or 42)
    if name == "amazon_xvert":
        return datasets.load("amazon_xvert", n_cross_users=args.amazon_users,
                             n_single_users=max(args.amazon_users // 3, 1000),
                             seed=args.seed)
    return datasets.load(name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    client = None
    if args.llm or args.llm_planner:
        if not ClaudeClient.available():
            print("ANTHROPIC_API_KEY is not set — falling back to the deterministic "
                  "template narrator and the rule-based planner.", file=sys.stderr)
        else:
            try:
                client = ClaudeClient()
            except Exception as exc:                        # noqa: BLE001
                print(f"Could not initialise the Claude client ({exc}); "
                      f"continuing without it.", file=sys.stderr)

    names = sorted(datasets.REGISTRY) if args.dataset == "both" else [args.dataset]
    out_paths = []
    failures = 0

    for name in names:
        print(f"\n### loading {name} ...", flush=True)
        try:
            data = load_one(name, args)
        except Exception as exc:                            # noqa: BLE001
            print(f"!! could not load {name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures += 1
            continue

        agent = AutonomousExperimenter(
            data, max_rounds=args.rounds, time_budget=args.time_budget,
            llm_planner=args.llm_planner and client is not None,
            llm_client=client, seed=args.seed, verbose=not args.quiet,
        )
        run = agent.run()
        run.narrative, run.narrator_engine = narrate(run, client if args.llm else None)
        path = save_run(run, args.out)
        out_paths.append(path)
        print(f"\nartefacts -> {path}")
        print(f"  report.md, run.json, leaderboard.csv"
              + (", narrative.md" if run.narrative else ""))

    if out_paths:
        print("\n" + "=" * 74)
        print("done. open the report with:")
        for p in out_paths:
            print(f"  {p / 'report.md'}")
    return 1 if failures and not out_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
