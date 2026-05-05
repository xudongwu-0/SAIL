# Online Reproduction — Robust Listwise Preference Optimization (RPL)

This document covers the **online policy-induced** track of the paper:
candidates are sampled from the *current* policy, ranked by an external
reward model (Eurus-RM-7b), and aligned with a robust Plackett–Luce loss.
It explains how to reproduce the K x rho main table with the
score-function (SF) correction enabled.

## 1. What this experiment is

For each prompt we

1. sample `K` responses from the current policy (`K in {2, 4}`),
2. score them with the reward model `openbmb/Eurus-RM-7b`,
3. compute the robust Plackett–Luce loss with robustness coefficient
   `rho in {0.0, 0.05}`,
4. add the policy-gradient score-function term so that `K=2, rho=0`
   reduces to the on-policy DPO objective (`lambda_sf=1`,
   cross-process mean baseline).

The resulting policy is evaluated on a held-out 256-prompt slice of the
training set with three judges:

- **Reward model (Eurus-RM-7b)** — `reward_mean`, win-rate vs SFT.
- **GPT-as-judge** — single-rating, response vs dataset `chosen`
  (SAIL-style win/tie/loss).
- **Offline ranking** — top-1, pairwise, Kendall tau, NDCG@4 on the raw
  4-way UltraFeedback eval split.

## 2. Setup

```bash
# Python 3.10, CUDA 12.x
pip install -e .

# Fill in your own service credentials (these files are .gitignore'd):
#   configs/services/huggingface.yaml   - HF token + 'xudongwu/...'-style prefixes
#   configs/services/wandb.yaml         - WandB team/key/project (or set report_to=none)
#   configs/services/openai.yaml        - only required for the GPT-judge stage
```

### Data and reward model

- Prompt source dataset: `xudongwu/U10` (UltraFeedback subset; loaded
  automatically by `pipelines/rpl.py`).
- Reward model: `openbmb/Eurus-RM-7b` (downloaded on first run).
- SFT reference adapter (used as the policy init for the SFT-init rows):
  `xudongwu/SFT_Q0.5B_U10@Q0.5B`.
- Base policy: `Qwen/Qwen1.5-0.5B`.

## 3. Quick smoke test (single GPU, ~2 minutes)

```bash
GPU=0 bash scripts/online/run_quick_test.sh
```

Verifies that data loading, the Plackett–Luce loss
(`cdpo/listwise_losses.py`), the online RPL trainer
(`cdpo/online_rpl_trainer.py`) and the LoRA forward pass all wire up.
It does NOT push to the Hub.

## 4. Train the main table (K x rho with SF=1)

The wrapper trains the four cells sequentially. On 4 x RTX 4090
each cell takes ~1.5 hours.

```bash
GPUS=0,1,2,3 NPROCS=4 NUM_STEPS=200 bash scripts/online/run_main_table.sh
# Add PUSH=1 to push the resulting LoRA adapters to the HF Hub.
```

What it runs (see [scripts/online/run_main_table.sh](../scripts/online/run_main_table.sh)):

| K | rho | run name |
|---|------|----------|
| 2 | 0.00 | `RPL_Q0.5B_U10_beta0.10rho0.00K2_sf1.00` |
| 2 | 0.05 | `RPL_Q0.5B_U10_beta0.10rho0.05K2_sf1.00` |
| 4 | 0.00 | `RPL_Q0.5B_U10_beta0.10rho0.00K4_sf1.00` |
| 4 | 0.05 | `RPL_Q0.5B_U10_beta0.10rho0.05K4_sf1.00` |

Common hyper-parameters: `beta=0.1`, `lambda_sf=1.0`,
`sf_baseline=mean`, `learning_rate=5e-6`, 200 optimiser steps,
prompt batch = 1 / device x 4 accum x N_GPU,
`max_prompt_length=384`, `max_target_length=128`, sampling `T=0.9`.

## 5. Evaluate generated outputs

```bash
# Default: gen + Eurus-RM reward score on the 4 main-table cells.
GPUS=0,1,2,3 python scripts/online/evaluate.py

# Add the offline ranking metrics (single GPU, ~3 min/model):
GPUS=0 python scripts/online/evaluate.py --ranking

# Add the GPT-as-judge stage (requires configs/services/openai.yaml).
python scripts/online/evaluate.py --gpt --gpt-ver gpt-35-turbo

# Custom set of (run, tag) pairs:
python scripts/online/evaluate.py \
    --runs 'RPL_Q0.5B_U10_beta0.10rho0.05K4_sf1.00|Q0.5B' \
           'SFT_Q0.5B_U10|Q0.5B'
```

Internally calls
[pipelines/generate.py](../pipelines/generate.py) (contrastive search,
deterministic, 256 prompts), then
[pipelines/evalreward.py](../pipelines/evalreward.py)
(`openbmb/Eurus-RM-7b`), and optionally
[pipelines/eval_ranking.py](../pipelines/eval_ranking.py) and
`cli.py evalgpt`.

Per-run results are written back to the same `xudongwu/<run>@<tag>`
dataset on the Hub (columns `response`, `reward`/`reward_score`,
`gpt_score`).

## 6. Collect the table

```bash
python scripts/online/collect_results.py
# -> outputs/online_results.json
# -> outputs/online_results.md
```

Pulls each `xudongwu/<run>@<tag>` dataset, computes
`reward_mean / reward_median`, GPT win/tie/loss vs `chosen`, and
ranking metrics (from `results/ranking/<run>.json` if present), then
writes a JSON dump and a Markdown table.

## 7. Expected outputs

| location | produced by |
|----------|-------------|
| `cache/checkpoints/<run>_<HASH>/` | `pipelines/rpl.py` — LoRA adapter weights |
| `xudongwu/<run>@<tag>` (Hub model) | `pipelines/rpl.py` (with `PUSH=1`) |
| `xudongwu/<run>@<tag>` (Hub dataset) | `pipelines/generate.py` + `pipelines/evalreward.py` |
| `results/ranking/<run>.json` | `pipelines/eval_ranking.py` |
| `outputs/online_results.json` | `scripts/online/collect_results.py` |
| `outputs/online_results.md` | `scripts/online/collect_results.py` |
| `logs/<timestamp>_RPL_*.log` | `scripts/online/run_main_table.sh` |

## 8. Hardware and runtime

- Smoke test: 1 x consumer GPU >= 24 GB, ~2 minutes.
- Main table (4 cells, 200 steps each, K=4 path):
  4 x RTX 4090 (24 GB), ~1.5 h / cell, ~6 h total.
- Reward eval (256 prompts, 4 GPUs): ~10 minutes per model.
- GPT-as-judge: API only, ~5 minutes per 256-prompt model under
  `gpt-3.5-turbo` rate limits.
- Memory: bf16 LoRA training fits in 24 GB / GPU with the defaults
  (`use_flash_attn=False`, `gradient_checkpointing=False`).

## 9. Notes / caveats

- `pipelines/rpl.py` requires `--lambda_sf 1.0` for the math-correct
  fully online gradient. `lambda_sf=0` falls back to the
  behavior-policy approximation and is intentionally NOT the default
  for the main table.
- Numerical results depend on the reward-model checkpoint pinned at the
  time of writing (`openbmb/Eurus-RM-7b`); upstream model updates may
  shift absolute reward scores.
- Service credential YAMLs under `configs/services/` are git-ignored;
  fill in your own before any Hub upload / GPT-judge run.
