#!/usr/bin/env bash
# RTDMD: GRPO + AC-DMD/BP auxiliary-loss training launcher.
#
# Single-node or multi-node `torchrun` wrapper for the `rtdmd` trainer.
# The yaml under `configs/rtdmd/` selects the model
# (sd3m, sd35m, flux1_dev, flux2_4b, flux2_9b).
#
# Usage:
#   bash scripts/rtdmd.sh <NUM_GPUS> <CONFIG_YAML> [EXTRA_ARGS...]
#
# Examples:
#   bash scripts/rtdmd.sh 8 configs/rtdmd/sd35m.yaml
#   bash scripts/rtdmd.sh 8 configs/rtdmd/flux2_4b.yaml \
#       --override train.seed=123
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT on every node.
# When the scheduler exports CHIEF_IP / INDEX / HOST_NUM / HOST_GPU_NUM these
# are used as defaults (so the script works as-is on MPI-style platforms).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS="${1:-${HOST_GPU_NUM:?usage: bash scripts/rtdmd.sh <NUM_GPUS> <CONFIG_YAML> [...]}}"
CONFIG_PATH="${2:?missing CONFIG yaml path}"
EXTRA_ARGS=("${@:3}")
set --

NUM_GPUS="${NUM_GPUS%%.*}"

NNODES="${NNODES:-${HOST_NUM:-1}}"
NODE_RANK="${NODE_RANK:-${INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${CHIEF_IP:-localhost}}"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES%%.*}"
NODE_RANK="${NODE_RANK%%.*}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] Config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi
if [[ "${NNODES}" -gt 1 && "${MASTER_ADDR}" == "localhost" ]]; then
  echo "[ERROR] Multi-node launch detected but MASTER_ADDR is localhost." >&2
  echo "        Ensure CHIEF_IP is available or set MASTER_ADDR manually." >&2
  exit 1
fi

# NCCL tuning
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT_SEC="${NCCL_TIMEOUT_SEC:-10800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"

# Multi-node IB/RDMA tuning (active only when NNODES > 1)
if [[ "${NNODES}" -gt 1 ]]; then
  export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
  export NCCL_IB_SL="${NCCL_IB_SL:-3}"
  export NCCL_CHECK_DISABLE="${NCCL_CHECK_DISABLE:-1}"
  export NCCL_LL_THRESHOLD="${NCCL_LL_THRESHOLD:-16384}"
  export NCCL_IB_CUDA_SUPPORT="${NCCL_IB_CUDA_SUPPORT:-1}"
  export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6}"
  export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-0}"
  export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
  export NCCL_IB_TC="${NCCL_IB_TC:-160}"
  export NCCL_PXN_DISABLE="${NCCL_PXN_DISABLE:-1}"
fi

echo "=============================================="
echo "RTDMD GRPO + AC-DMD/BP aux loss"
echo "  Nodes:       ${NNODES} (this node: ${NODE_RANK})"
echo "  GPUs/node:   ${NUM_GPUS}"
echo "  Total GPUs:  $((NNODES * NUM_GPUS))"
echo "  Master:      ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Config:      ${CONFIG_PATH}"
echo "  Trainer:     rtdmd"
echo "  Time:        $(date)"
echo "=============================================="

RDZV_TIMEOUT_SEC="${RDZV_TIMEOUT_SEC:-900}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --rdzv-conf="timeout=${RDZV_TIMEOUT_SEC}" \
  main.py \
  "${CONFIG_PATH}" \
  --trainer rtdmd \
  "${EXTRA_ARGS[@]}"
