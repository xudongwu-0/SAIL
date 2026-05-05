#!/usr/bin/env bash
# Single-GPU smoke test for the online RPL pipeline.
# Verifies data path + loss + reward model + LoRA forward all wire up.
# Runs in a few minutes; does NOT push to the Hub.
#
# Usage:
#   GPU=0 bash scripts/online/run_quick_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p cache/models cache/datasets cache/checkpoints logs

python pipelines/rpl.py \
    --model Q0.5B \
    --dataset U10 \
    --tag online_smoke \
    --beta 0.1 \
    --rho 0.05 \
    --K 4 \
    --lambda_sf 1.0 \
    --sf_baseline mean \
    --num_train_steps 5 \
    --per_device_prompt_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --per_device_evalreward_batch_size 4 \
    --max_prompt_length 384 \
    --max_target_length 96 \
    --learning_rate 5e-6 \
    --warmup_steps 1 \
    --logging_steps 1 \
    --n_eval_prompts 8 \
    --push_to_hub_after_train False \
    --report_to none \
    --use_flash_attn False
