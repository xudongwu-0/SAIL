#!/usr/bin/env python
"""Evaluate trained online-RPL adapters on the held-out U10 prompts.

Runs three stages per (run, tag) tuple:

  1. ``pipelines/generate.py``  - sample one response per prompt
  2. ``pipelines/evalreward.py``- score with openbmb/Eurus-RM-7b
  3. ``pipelines/eval_ranking.py`` (optional) - top-1 / pairwise / Kendall tau / NDCG@4
                                                on the raw 4-way eval split

Optional GPT-as-judge stage requires ``configs/services/openai.yaml`` to be
populated with a usable endpoint and is invoked only with ``--gpt``.

Examples:
    # Default: evaluate the four SF=1 main-table cells, gen+reward only
    python scripts/online/evaluate.py

    # Custom set of (run, tag) pairs and GPT-judge as well
    python scripts/online/evaluate.py --runs 'A|Q0.5B' 'B|freshinit' --gpt

    # Reward-only (skip generation, dataset already on the Hub)
    python scripts/online/evaluate.py --skip-gen
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUNS = [
    "RPL_Q0.5B_U10_beta0.10rho0.00K2_sf1.00|Q0.5B",
    "RPL_Q0.5B_U10_beta0.10rho0.00K4_sf1.00|Q0.5B",
    "RPL_Q0.5B_U10_beta0.10rho0.05K2_sf1.00|Q0.5B",
    "RPL_Q0.5B_U10_beta0.10rho0.05K4_sf1.00|Q0.5B",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                   help="space-separated 'run|tag' tokens")
    p.add_argument("--n", type=int, default=256, help="number of held-out prompts")
    p.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3"),
                   help="CUDA_VISIBLE_DEVICES for accelerate")
    p.add_argument("--nprocs", type=int, default=None,
                   help="accelerate processes (defaults to len(gpus))")
    p.add_argument("--gen-port", type=int, default=63871)
    p.add_argument("--rw-port", type=int, default=63872)
    p.add_argument("--gen-batch", type=int, default=8)
    p.add_argument("--rw-batch", type=int, default=8)
    p.add_argument("--gpt", action="store_true", help="also run GPT-as-judge")
    p.add_argument("--ranking", action="store_true",
                   help="also run offline ranking accuracy on the raw 4-way split")
    p.add_argument("--ref-repo", default="xudongwu/SFT_Q0.5B_U10",
                   help="reference policy repo for ranking eval")
    p.add_argument("--ref-rev", default="Q0.5B",
                   help="reference policy revision for ranking eval")
    p.add_argument("--skip-gen", action="store_true",
                   help="skip generation stage (re-score an existing dataset)")
    p.add_argument("--skip-reward", action="store_true",
                   help="skip reward-model scoring stage")
    p.add_argument("--gpt-ver", default="gpt-35-turbo",
                   help="key in configs/services/openai.yaml")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> None:
    args = parse_args()
    nprocs = args.nprocs or max(1, len(args.gpus.split(",")))
    env_gpus = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus,
                "TOKENIZERS_PARALLELISM": "false"}

    for token in args.runs:
        if "|" not in token:
            print(f"[skip] bad token (expected 'run|tag'): {token!r}", file=sys.stderr)
            continue
        run_name, tag = token.split("|", 1)
        print(f"\n==================== {run_name} @ {tag} ====================")

        if not args.skip_gen:
            cmd = ["accelerate", "launch", f"--num_processes={nprocs}",
                   "--main_process_port", str(args.gen_port),
                   "pipelines/generate.py",
                   "--run", run_name, "--tag", tag,
                   "--eval_limit", str(args.n),
                   "--per_device_generation_batch_size", str(args.gen_batch),
                   "--use_flash_attn", "False"]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env_gpus)

        if not args.skip_reward:
            cmd = ["accelerate", "launch", f"--num_processes={nprocs}",
                   "--main_process_port", str(args.rw_port),
                   "pipelines/evalreward.py",
                   "--run_name", run_name, "--tag", tag,
                   "--per_device_evalreward_batch_size", str(args.rw_batch)]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env_gpus)

        if args.ranking:
            cmd = ["python", "pipelines/eval_ranking.py",
                   "--policy_repo", f"xudongwu/{run_name}",
                   "--policy_revision", tag,
                   "--reference_repo", args.ref_repo,
                   "--reference_revision", args.ref_rev,
                   "--n_prompts", str(args.n),
                   "--tag", run_name]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env_gpus)

        if args.gpt:
            cmd = ["python", "cli.py", "evalgpt",
                   "-p", "RPL", "-m", "Q0.5B", "-d", "U10", "-t", tag,
                   "--gpt_ver", args.gpt_ver]
            subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    print("\n[evaluate] done.")


if __name__ == "__main__":
    main()
