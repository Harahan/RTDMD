#!/usr/bin/env bash
# RTDMD: AC-DMD cold-start training launcher.
#
# Single-node or multi-node `torchrun` wrapper for the `ac_dmd` trainer.
# The yaml under `configs/cold_start/` selects the model
# (sd3m, sd35m, flux1_dev, flux2_4b, flux2_9b).
#
# Usage:
#   bash scripts/cold_start.sh <NUM_GPUS> <CONFIG_YAML> [TRAINER] [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/cold_start.sh 8 configs/cold_start/sd35m.yaml
#   bash scripts/cold_start.sh 8 configs/cold_start/flux2_4b.yaml ac_dmd \
#       --override train.seed=123
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT on every node.
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash scripts/cold_start.sh 8 configs/cold_start/flux2_9b.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${1:?usage: bash scripts/cold_start.sh <NUM_GPUS> <CONFIG_YAML> [TRAINER] [...]}"
CONFIG="${2:?missing CONFIG yaml path}"
TRAINER="${3:-ac_dmd}"
EXTRA_ARGS=("${@:4}")
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
echo "RTDMD AC-DMD cold-start"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  ${TOTAL_GPUS}"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG}"
echo "  Trainer:     ${TRAINER}"
echo "  Time:        $(date)"
echo "=============================================="

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    main.py \
    "${CONFIG}" \
    --trainer "${TRAINER}" \
    "${EXTRA_ARGS[@]}"
