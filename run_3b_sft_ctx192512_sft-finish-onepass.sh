#!/usr/bin/env bash
set -euo pipefail

# RWKV7 G1J 2.9B SFT finish stage.
#
# Run after run_3b_sft_ctx192512_pt.sh.  The PT step checkpoint has already
# consumed the first part of the deterministic shuffled stream with mask=1;
# this stage resumes that position and trains the remaining documents once with
# the normal assistant-only SFT mask.

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
LOAD_MODEL="${LOAD_MODEL:-${REPO_DIR}/outs/3b-sft-ctx192512-pt/rwkv-step-1000.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/Datasets/SFT_RWKV7_13B_ctx192512/results/SFT_RWKV7_13B}"
PROJ_DIR="${PROJ_DIR:-${REPO_DIR}/outs/3b-sft-ctx192512-sft-finish}"
SFT_MASK_FILE="${SFT_MASK_FILE:-}"

SFT_ONE_PASS="${SFT_ONE_PASS:-1}"
EPOCH_STEPS="${EPOCH_STEPS:-1}" # replaced by train.py when SFT_ONE_PASS=1
EPOCH_COUNT="${EPOCH_COUNT:-1}"
EPOCH_BEGIN="${EPOCH_BEGIN:-0}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"
ESTIMATED_STEP_SECONDS="${ESTIMATED_STEP_SECONDS:-88}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-500}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-8}"

SFT_EVAL_TAIL_RATIO="${SFT_EVAL_TAIL_RATIO:-0.005}"
SFT_EVAL_TAIL_DOCS="${SFT_EVAL_TAIL_DOCS:-0}"
SFT_EVAL_INCLUDE_IN_TRAIN="${SFT_EVAL_INCLUDE_IN_TRAIN:-0}"
SFT_EVAL_EVERY_N_STEPS="${SFT_EVAL_EVERY_N_STEPS:-$SAVE_EVERY_N_STEPS}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-5}"
SFT_TRAIN_SHUFFLE="${SFT_TRAIN_SHUFFLE:-1}"
SFT_TRAIN_SHUFFLE_SEED="${SFT_TRAIN_SHUFFLE_SEED:-1234}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
MICRO_BSZ="${MICRO_BSZ:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"

STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
GRAD_CP="${GRAD_CP:-1}"
HEAD_CHUNK="${HEAD_CHUNK:-0}"
SFT_MASKED_CE_CHUNK="${SFT_MASKED_CE_CHUNK:-0}"
SFT_MASKED_FUSED_CE_CHUNK="${SFT_MASKED_FUSED_CE_CHUNK:-4096}"
DS_BUCKET_MB="${DS_BUCKET_MB:-64}"
DS_OFFLOAD_PIN_MEMORY="${DS_OFFLOAD_PIN_MEMORY:-1}"
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD="${DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD:-0}"
DS_STAGE3_PREFETCH_BUCKET_SIZE="${DS_STAGE3_PREFETCH_BUCKET_SIZE:-5000000}"
DS_STAGE3_MAX_LIVE_PARAMETERS="${DS_STAGE3_MAX_LIVE_PARAMETERS:-200000000}"
KERNEL="${KERNEL:-@rwkv3}"

LR_INIT="${LR_INIT:-4e-6}"
LR_FINAL="${LR_FINAL:-4e-7}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
REQUESTED_LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-}"

# Empty means no wandb login or network logging.
WANDB_PROJECT="${WANDB_PROJECT:-RWKV7-3B-SFT}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29502}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${REPO_DIR}/.torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-16}"
export MASTER_ADDR MASTER_PORT

if (( CTX_LEN <= 0 || CTX_LEN % 16 != 0 )); then
  echo "CTX_LEN must be a positive multiple of 16: $CTX_LEN" >&2
  exit 2
fi
if [[ ! -e "$LOAD_MODEL" ]]; then
  echo "Missing PT checkpoint: $LOAD_MODEL" >&2
  echo "Run run_3b_sft_ctx192512_pt.sh for one day first." >&2
  exit 1
fi
for suffix in .bin .idx .mask.bin .mask.idx; do
  if [[ ! -f "${DATA_FILE}${suffix}" ]]; then
    echo "Missing SFT dataset file: ${DATA_FILE}${suffix}" >&2
    exit 1
  fi
done

if [[ -d "$LOAD_MODEL" ]]; then
  if [[ "$STRATEGY" != *deepspeed* ]]; then
    echo "PT checkpoint is a DeepSpeed directory; use a deepspeed strategy to resume it." >&2
    exit 2
  fi
  if [[ ! -f "$LOAD_MODEL/latest" ]] && ! compgen -G "$LOAD_MODEL/*_model_states.pt" >/dev/null; then
    echo "PT checkpoint directory is not a recognized DeepSpeed checkpoint: $LOAD_MODEL" >&2
    exit 1
  fi
else
  "$PYTHON_BIN" "$REPO_DIR/scripts/check_rwkv_checkpoint.py" "$LOAD_MODEL" \
    --expect "n_layer=${N_LAYER}" --expect "n_embd=${N_EMBD}" --expect "dim_ffn=${DIM_FFN}" \
    --expect "vocab_size=${VOCAB_SIZE}" --expect "head_size=${HEAD_SIZE}" \
    --expect "d_decay_lora=${D_DECAY_LORA}" --expect "d_aaa_lora=${D_AAA_LORA}" \
    --expect "d_mv_lora=${D_MV_LORA}" --expect "d_gate_lora=${D_GATE_LORA}" >/dev/null
fi

if ! DOC_COUNT="$($PYTHON_BIN - "$DATA_FILE" <<'PY'
import sys
from src.binidx import MMapIndexedDataset, index_file_path

print(len(MMapIndexedDataset.Index(index_file_path(sys.argv[1]))))
PY
)"; then
  echo "Could not read document count from ${DATA_FILE}.idx" >&2
  exit 1
fi
if ! [[ "$DOC_COUNT" =~ ^[0-9]+$ ]] || (( DOC_COUNT <= 0 )); then
  echo "Invalid SFT document count: $DOC_COUNT" >&2
  exit 1
fi

REAL_BSZ=$((N_NODE * GPU_PER_NODE * MICRO_BSZ))
EFFECTIVE_BSZ=$((REAL_BSZ * ACCUMULATE_GRAD_BATCHES))
if (( REAL_BSZ <= 0 || EFFECTIVE_BSZ <= 0 )); then
  echo "Batch-size settings must produce a positive batch size" >&2
  exit 2
fi

if ! SFT_STATS="$($PYTHON_BIN - "$DOC_COUNT" "$SFT_EVAL_TAIL_RATIO" "$EFFECTIVE_BSZ" <<'PY'
import math
import sys

document_count = int(sys.argv[1])
tail_ratio = float(sys.argv[2])
effective_bsz = int(sys.argv[3])
eval_count = max(1, math.ceil(document_count * tail_ratio)) if tail_ratio > 0 else 0
train_count = document_count - eval_count
total_steps = (train_count + effective_bsz - 1) // effective_bsz
decay_steps = max(1, (total_steps * 25 + 99) // 100)
print(eval_count, train_count, total_steps, decay_steps)
PY
)"; then
  echo "Could not calculate SFT one-pass schedule" >&2
  exit 1
fi
read -r EVAL_DOC_COUNT TRAIN_DOC_COUNT TOTAL_STEPS COMPUTED_LR_WSD_DECAY_ITERS <<< "$SFT_STATS"
if [[ -n "$REQUESTED_LR_WSD_DECAY_ITERS" ]]; then
  LR_WSD_DECAY_ITERS="$REQUESTED_LR_WSD_DECAY_ITERS"
else
  LR_WSD_DECAY_ITERS="$COMPUTED_LR_WSD_DECAY_ITERS"
fi

mkdir -p "$PROJ_DIR" "$TORCH_EXTENSIONS_DIR"
EXTRA_ARGS=()
if [[ -n "$SFT_MASK_FILE" ]]; then
  EXTRA_ARGS+=(--sft_mask_file "$SFT_MASK_FILE")
fi

echo "SFT checkpoint:           $LOAD_MODEL"
echo "SFT dataset:              $DATA_FILE"
echo "SFT output:               $PROJ_DIR"
echo "Documents:                $DOC_COUNT total / $TRAIN_DOC_COUNT train / $EVAL_DOC_COUNT eval"
echo "Effective batch size:     $EFFECTIVE_BSZ"
echo "One-pass optimizer steps: $TOTAL_STEPS"
echo "LR WSD decay steps:       $LR_WSD_DECAY_ITERS (last 25%)"
echo "Checkpoint interval:      $SAVE_EVERY_N_STEPS (~12.2h at ${ESTIMATED_STEP_SECONDS}s/optimizer step)"
echo "wandb:                    ${WANDB_PROJECT:-disabled}"

"$PYTHON_BIN" train.py --load_model "$LOAD_MODEL" --wandb "$WANDB_PROJECT" --proj_dir "$PROJ_DIR" --my_testing "$MODEL_TYPE" \
  --ctx_len "$CTX_LEN" --train_stage 0 --epoch_steps "$EPOCH_STEPS" --epoch_count "$EPOCH_COUNT" --epoch_begin "$EPOCH_BEGIN" \
  --sft_one_pass "$SFT_ONE_PASS" \
  --sft_eval_tail_ratio "$SFT_EVAL_TAIL_RATIO" --sft_eval_tail_docs "$SFT_EVAL_TAIL_DOCS" --sft_eval_include_in_train "$SFT_EVAL_INCLUDE_IN_TRAIN" \
  --sft_eval_every_n_steps "$SFT_EVAL_EVERY_N_STEPS" --sft_eval_steps "$SFT_EVAL_STEPS" \
  --sft_train_shuffle "$SFT_TRAIN_SHUFFLE" --sft_train_shuffle_seed "$SFT_TRAIN_SHUFFLE_SEED" \
  --data_file "$DATA_FILE" --sft_masked_ce_chunk "$SFT_MASKED_CE_CHUNK" --sft_masked_fused_ce_chunk "$SFT_MASKED_FUSED_CE_CHUNK" \
  --num_nodes "$N_NODE" --micro_bsz "$MICRO_BSZ" --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES" \
  --n_layer "$N_LAYER" --n_embd "$N_EMBD" --dim_ffn "$DIM_FFN" --kernel "$KERNEL" \
  --lr_init "$LR_INIT" --lr_final "$LR_FINAL" --lr_wsd_decay_iters "$LR_WSD_DECAY_ITERS" --lr_wsd_decay_style "$LR_WSD_DECAY_STYLE" \
  --warmup_steps "$WARMUP_STEPS" --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
  --data_type sft_binidx --vocab_size "$VOCAB_SIZE" --weight_decay "$WEIGHT_DECAY" \
  --epoch_save "$EPOCH_SAVE" --save_every_n_steps "$SAVE_EVERY_N_STEPS" --keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
  --head_size "$HEAD_SIZE" --head_chunk "$HEAD_CHUNK" \
  --accelerator gpu --devices "$GPU_PER_NODE" --precision bf16 --strategy "$STRATEGY" --grad_cp "$GRAD_CP" \
  --enable_progress_bar True --ds_bucket_mb "$DS_BUCKET_MB" \
  --ds_offload_pin_memory "$DS_OFFLOAD_PIN_MEMORY" \
  --ds_stage3_param_persistence_threshold "$DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD" \
  --ds_stage3_prefetch_bucket_size "$DS_STAGE3_PREFETCH_BUCKET_SIZE" \
  --ds_stage3_max_live_parameters "$DS_STAGE3_MAX_LIVE_PARAMETERS" --master_port "$MASTER_PORT" \
  --d_decay_lora "$D_DECAY_LORA" --d_aaa_lora "$D_AAA_LORA" --d_mv_lora "$D_MV_LORA" --d_gate_lora "$D_GATE_LORA" \
  "${EXTRA_ARGS[@]}"
