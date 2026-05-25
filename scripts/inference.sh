#!/usr/bin/env bash
# RTDMD: inference launcher (single GPU or distributed via torchrun).
#
# The YAML under `configs/inference/` selects the model (sd3m, sd35m,
# flux1_dev, flux2_4b, flux2_9b). The same YAML supports three LoRA
# regimes — toggled by `lora_paths` length — and reward eval — toggled
# by `eval_reward`. Use `--override key=value` to flip these on the CLI.
#
# Usage:
#   bash scripts/inference.sh <NUM_GPUS> <CONFIG_YAML> [EXTRA_ARGS...]
#
# Examples:
#   # Single GPU, plain inference (override disables eval)
#   bash scripts/inference.sh 1 configs/inference/sd35m.yaml \
#       --override eval_reward=false --prompt "a cute cat"
#
#   # 8-GPU distributed reward eval on drawbench (the YAML default)
#   bash scripts/inference.sh 8 configs/inference/flux2_4b.yaml \
#       --override distributed=true
#
#   # No LoRA, plain pretrained model
#   bash scripts/inference.sh 1 configs/inference/sd35m.yaml \
#       --override lora_paths=
#
#   # Distilled-only LoRA (override the stacked list with a single path)
#   bash scripts/inference.sh 1 configs/inference/sd35m.yaml \
#       --override lora_paths=/path/to/cold_start_ckpt/checkpoint-15000/generator_ema.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${1:?usage: bash scripts/inference.sh <NUM_GPUS> <CONFIG_YAML> [...]}"
CONFIG="${2:?missing CONFIG yaml path}"
EXTRA_ARGS=("${@:3}")
set --

if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG}" >&2
  exit 1
fi

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${NNODES}" -gt 1 && "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] Multi-node launch detected but MASTER_ADDR is localhost." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false

TOTAL_GPUS=$((NUM_GPUS * NNODES))

echo "=============================================="
echo "RTDMD Inference"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  Time:        $(date)"
echo "=============================================="

if [[ "${NUM_GPUS}" -le 1 && "${NNODES}" -le 1 ]]; then
  python inference.py "${CONFIG}" "${EXTRA_ARGS[@]}"
else
  torchrun \
      --nnodes="${NNODES}" \
      --node_rank="${NODE_RANK}" \
      --nproc_per_node="${NUM_GPUS}" \
      --master_addr="${MASTER_ADDR}" \
      --master_port="${MASTER_PORT}" \
      inference.py \
      "${CONFIG}" \
      --override distributed=true \
      "${EXTRA_ARGS[@]}"
fi
