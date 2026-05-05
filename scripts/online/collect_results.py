#!/usr/bin/env python
"""Aggregate online-RPL evaluation results into a single JSON + Markdown table.

Pulls each ``xudongwu/<run>@<tag>`` dataset from the Hub and reports:
  - ``reward_mean``, ``reward_median``, ``reward_std`` from
    ``pipelines/evalreward.py``;
  - ``gpt_score`` win/tie/loss vs the dataset ``chosen`` answer
    (only if the GPT-judge stage was run);
  - per-run ranking metrics (top-1 / pairwise / Kendall tau / NDCG@4) by
    reading ``results/ranking/<tag>.json`` if present.

The output mirrors the SF=1 table in docs/online_reproduction.md and is
written to ``outputs/online_results.json`` and ``outputs/online_results.md``.

Examples:
    python scripts/online/collect_results.py
    python scripts/online/collect_results.py --runs 'A|Q0.5B' 'B|freshinit'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

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
    p.add_argument("--ranking-dir", default="results/ranking",
                   help="directory containing per-run <tag>.json files from eval_ranking")
    p.add_argument("--outdir", default="outputs", help="where to write outputs")
    return p.parse_args()


def summarise_one(run: str, tag: str, ranking_dir: Path) -> dict:
    out: dict = {"run": run, "tag": tag}

    try:
        from datasets import load_dataset
        import numpy as np
    except ImportError as e:
        return {**out, "error": f"missing dep: {e}"}

    try:
        ds = load_dataset(f"xudongwu/{run}", tag, split="default")
    except Exception:
        try:
            ds = load_dataset(f"xudongwu/{run}", split="default")
        except Exception as e:
            return {**out, "error": f"load_dataset: {type(e).__name__}: {e}"}

    cols = set(ds.column_names)

    rcol: Optional[str] = (
        "reward_score" if "reward_score" in cols
        else ("reward" if "reward" in cols else None)
    )
    if rcol is not None:
        rs = np.asarray([r for r in ds[rcol] if r is not None], dtype=np.float64)
        if rs.size:
            out["n"] = int(rs.size)
            out["reward_mean"] = float(rs.mean())
            out["reward_std"] = float(rs.std())
            out["reward_median"] = float(np.median(rs))

    if "gpt_score" in cols:
        arr = np.asarray([s for s in ds["gpt_score"] if s is not None], dtype=np.float64)
        if arr.size:
            wins = int((arr > 0).sum())
            losses = int((arr < 0).sum())
            ties = int((arr == 0).sum())
            out["gpt"] = {
                "n": int(arr.size),
                "win": wins,
                "tie": ties,
                "loss": losses,
                "win_plus_half_tie": (wins + 0.5 * ties) / arr.size,
            }

    rfile = ranking_dir / f"{run}.json"
    if rfile.exists():
        try:
            out["ranking"] = json.loads(rfile.read_text())
        except Exception as e:
            out["ranking_error"] = f"{type(e).__name__}: {e}"

    return out


def render_markdown(rows: list[dict]) -> str:
    header = (
        "| run | tag | n | reward_mean | reward_median | "
        "GPT W/T/L | GPT win+1/2tie | top-1 | pairwise | Kendall tau | NDCG@4 |"
    )
    sep = "|" + "|".join(["---"] * 11) + "|"
    out = [header, sep]
    for r in rows:
        rk = r.get("ranking") or {}
        gpt = r.get("gpt") or {}
        wtl = (
            f"{gpt['win']}/{gpt['tie']}/{gpt['loss']}"
            if gpt else "-"
        )
        whalf = f"{gpt['win_plus_half_tie']:.3f}" if gpt else "-"
        cells = [
            r["run"], r["tag"], str(r.get("n", "-")),
            f"{r.get('reward_mean', float('nan')):.2f}" if "reward_mean" in r else "-",
            f"{r.get('reward_median', float('nan')):.2f}" if "reward_median" in r else "-",
            wtl, whalf,
            f"{rk.get('top1', float('nan')):.3f}" if "top1" in rk else "-",
            f"{rk.get('pairwise', float('nan')):.3f}" if "pairwise" in rk else "-",
            f"{rk.get('kendall_tau', float('nan')):.3f}" if "kendall_tau" in rk else "-",
            f"{rk.get('ndcg', float('nan')):.3f}" if "ndcg" in rk else "-",
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    args = parse_args()
    outdir = REPO_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    ranking_dir = (REPO_ROOT / args.ranking_dir).resolve()

    rows: list[dict] = []
    for token in args.runs:
        if "|" not in token:
            print(f"[skip] bad token: {token!r}", file=sys.stderr)
            continue
        run_name, tag = token.split("|", 1)
        print(f"[collect] {run_name} @ {tag}", flush=True)
        rows.append(summarise_one(run_name, tag, ranking_dir))

    json_path = outdir / "online_results.json"
    md_path = outdir / "online_results.md"
    json_path.write_text(json.dumps(rows, indent=2))
    md_path.write_text(render_markdown(rows))
    print(f"[collect] wrote {json_path}")
    print(f"[collect] wrote {md_path}")


if __name__ == "__main__":
    main()
