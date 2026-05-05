#!/usr/bin/env bash
# Train the four cells of the online RPL main table:
#   K in {2, 4} x rho in {0.0, 0.05}, all with score-function correction (SF=1).
#
# Each cell runs sequentially on the same GPUs. Adjust GPUS/NPROCS for your
# hardware. Reproduces the SF=1 SFT-init rows of the online table reported
# in docs/online_reproduction.md.
#
# Required environment:
#   GPUS       comma-separated CUDA_VISIBLE_DEVICES (default: 0,1,2,3)
#   NPROCS     number of accelerate processes (default: matches GPUS)
# Optional:
#   TAG        SFT adapter revision on the Hub (default: Q0.5B)
#   NUM_STEPS  optimization steps per cell (default: 200)
#   PUSH       "1" to push the resulting adapters to HF Hub (default: 0)
#   MAIN_PORT  starting DDP master port (default: 29610)
#
# Usage (single-node, 4 GPU):
#   GPUS=0,1,2,3 NPROCS=4 bash scripts/online/run_main_table.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

GPUS="${GPUS:-0,1,2,3}"
NPROCS="${NPROCS:-$(awk -F, '{print NF}' <<<"$GPUS")}"
TAG="${TAG:-Q0.5B}"
NUM_STEPS="${NUM_STEPS:-200}"
PUSH="${PUSH:-0}"
MAIN_PORT="${MAIN_PORT:-29610}"

mkdir -p cache/models cache/datasets cache/checkpoints logs outputs

export CUDA_VISIBLE_DEVICES="$GPUS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

PUSH_FLAG="False"
if [[ "$PUSH" == "1" ]]; then PUSH_FLAG="True"; fi

train_cell() {
    local K="$1" RHO="$2" PORT="$3"
    local STAMP
    STAMP="$(date +%Y%m%d-%H%M%S)"
    local LOG="logs/${STAMP}_RPL_K${K}_rho${RHO}_sf1.log"
    echo "[run_main_table] K=$K rho=$RHO port=$PORT log=$LOG"
    accelerate launch \
        --config_file configs/accelerate/rpl/ddp.yaml \
        --num_processes "$NPROCS" \
        --main_process_port "$PORT" \
        pipelines/rpl.py \
            --model Q0.5B \
            --dataset U10 \
            --tag "$TAG" \
            --beta 0.1 \
            --rho "$RHO" \
            --K "$K" \
            --lambda_sf 1.0 \
            --sf_baseline mean \
            --sft_tag "$TAG" \
            --push_to_hub_after_train "$PUSH_FLAG" \
            --num_train_steps "$NUM_STEPS" \
            --per_device_prompt_batch_size 1 \
            --gradient_accumulation_steps 4 \
            --per_device_evalreward_batch_size 4 \
            --max_prompt_length 384 \
            --max_target_length 128 \
            --learning_rate 5e-6 \
            --warmup_steps 10 \
            --logging_steps 5 \
            --n_eval_prompts 64 \
            --report_to none \
            --use_flash_attn False \
            2>&1 | tee "$LOG"
}

PORT="$MAIN_PORT"
for K in 2 4; do
    for RHO in 0.0 0.05; do
        train_cell "$K" "$RHO" "$PORT"
        PORT=$((PORT + 1))
    done
done

echo "[run_main_table] all cells complete."
