#!/usr/bin/env bash
set -euo pipefail

# Short 2.9B context/memory profile.  Set CTX_LEN and DATA_FILE per prepared
# binidx dataset; the profile does not save a checkpoint or use wandb.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
source "$REPO_DIR/scripts/setup_rwkv_cuda_env.sh"

MODEL_TYPE="${MODEL_TYPE:-x070}"
N_LAYER="${N_LAYER:-32}"
N_EMBD="${N_EMBD:-2560}"
DIM_FFN="${DIM_FFN:-10240}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
HEAD_SIZE="${HEAD_SIZE:-64}"
D_DECAY_LORA="${D_DECAY_LORA:-96}"
D_AAA_LORA="${D_AAA_LORA:-96}"
D_MV_LORA="${D_MV_LORA:-64}"
D_GATE_LORA="${D_GATE_LORA:-320}"
CTX_LEN="${CTX_LEN:-192512}"
LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1j-2.9b-20260831-ctx16384.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/Datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B}"
PROFILE_ROOT="${PROFILE_ROOT:-${REPO_DIR}/outs/3b-sft-profile}"
PROJ_DIR="${PROJ_DIR:-${PROFILE_ROOT}/ctx${CTX_LEN}}"
LR_INIT="${LR_INIT:-4e-6}"
LR_FINAL="${LR_FINAL:-4e-7}"

if (( CTX_LEN <= 0 || CTX_LEN % 16 != 0 )); then
  echo "CTX_LEN must be a positive multiple of 16 for the current x070 CUDA kernel: ${CTX_LEN}" >&2
  exit 2
fi
if [[ ! -f "$LOAD_MODEL" ]]; then
  echo "Missing LOAD_MODEL: $LOAD_MODEL" >&2
  exit 1
fi

"$PYTHON_BIN" "$REPO_DIR/scripts/check_rwkv_checkpoint.py" "$LOAD_MODEL" \
  --expect "n_layer=${N_LAYER}" --expect "n_embd=${N_EMBD}" --expect "dim_ffn=${DIM_FFN}" \
  --expect "vocab_size=${VOCAB_SIZE}" --expect "head_size=${HEAD_SIZE}" \
  --expect "d_decay_lora=${D_DECAY_LORA}" --expect "d_aaa_lora=${D_AAA_LORA}" \
  --expect "d_mv_lora=${D_MV_LORA}" --expect "d_gate_lora=${D_GATE_LORA}" >/dev/null

export PYTHON_BIN MODEL_TYPE N_LAYER N_EMBD DIM_FFN VOCAB_SIZE HEAD_SIZE
export D_DECAY_LORA D_AAA_LORA D_MV_LORA D_GATE_LORA CTX_LEN LOAD_MODEL DATA_FILE PROJ_DIR LR_INIT LR_FINAL
exec bash "$REPO_DIR/run_13b_sft_profile.sh"
