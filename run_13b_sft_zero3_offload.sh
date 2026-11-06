#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
source "$REPO_DIR/scripts/setup_rwkv_cuda_env.sh"

#######################################################################################################################
#
# RWKV7 G1F 13.3B SFT launcher for binidx + mask-sidecar data.
#
# Prepare data first. For example, for CTX_LEN=8192:
#   python data/make_sft_binidx.py /path/to/sft_jsonl_dir \
#     --out-prefix /mnt/data/datasets/sft_train_ctx8192 \
#     --ctx-len 8192 \
#     --pack \
#     --pack-strategy best-fit-decreasing \
#     --pack-shard-group-size 8 \
#     --num-workers 32 \
#     --shuffle
#
# The resulting training prefix must have:
#   DATA_FILE.bin / DATA_FILE.idx / DATA_FILE.mask.bin / DATA_FILE.mask.idx
#
#######################################################################################################################

MODEL_TYPE="${MODEL_TYPE:-x070}"

N_LAYER="${N_LAYER:-61}"
N_EMBD="${N_EMBD:-4096}"
DIM_FFN="${DIM_FFN:-16384}"
VOCAB_SIZE="${VOCAB_SIZE:-65536}"
HEAD_SIZE="${HEAD_SIZE:-64}"

D_DECAY_LORA="${D_DECAY_LORA:-192}"
D_AAA_LORA="${D_AAA_LORA:-192}"
D_MV_LORA="${D_MV_LORA:-128}"
D_GATE_LORA="${D_GATE_LORA:-384}"

CTX_LEN="${CTX_LEN:-8192}"
LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/datasets/sft_train_ctx8192}"
PROJ_DIR="${PROJ_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-zero3-offload}"

#######################################################################################################################
#
# SFT schedule.
#
# EPOCH_STEPS is optimizer steps per SFT epoch. If each packed document is one sample, a common starting point is:
#   ceil(num_documents / (NUM_NODES * GPU_PER_NODE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES))
# If EPOCH_STEPS * effective_bsz is larger than the document count, the SFT dataset wraps around deterministically.
#
# Set SFT_ONE_PASS=1 when you want train.py to compute that one-pass schedule automatically from DATA_FILE.idx.
# In that mode EPOCH_STEPS / EPOCH_COUNT are still passed as integer placeholders, but train.py overrides them with:
#   epoch_steps = ceil(num_documents / effective_bsz), epoch_count = 1
#
#######################################################################################################################

EPOCH_STEPS="${EPOCH_STEPS:-1000}"
EPOCH_COUNT="${EPOCH_COUNT:-1}"
SFT_ONE_PASS="${SFT_ONE_PASS:-0}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-3}"
SFT_EVAL_TAIL_RATIO="${SFT_EVAL_TAIL_RATIO:-0}"
SFT_EVAL_TAIL_DOCS="${SFT_EVAL_TAIL_DOCS:-0}"
SFT_EVAL_INCLUDE_IN_TRAIN="${SFT_EVAL_INCLUDE_IN_TRAIN:-0}"
SFT_EVAL_EVERY_N_STEPS="${SFT_EVAL_EVERY_N_STEPS:-0}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-0}"
SFT_TRAIN_SHUFFLE="${SFT_TRAIN_SHUFFLE:-0}"
SFT_TRAIN_SHUFFLE_SEED="${SFT_TRAIN_SHUFFLE_SEED:-1234}"

MICRO_BSZ="${MICRO_BSZ:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
LR_INIT="${LR_INIT:-1e-5}"
LR_FINAL="${LR_FINAL:-1e-6}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-0}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
GRAD_CP="${GRAD_CP:-1}"
HEAD_CHUNK="${HEAD_CHUNK:-0}"
# Keep this at 0 for production; the current positive chunked SFT CE path can timeout under 13B ZeRO-3.
SFT_MASKED_CE_CHUNK="${SFT_MASKED_CE_CHUNK:-0}"
SFT_MASKED_FUSED_CE_CHUNK="${SFT_MASKED_FUSED_CE_CHUNK:-4096}"
KERNEL="${KERNEL:-@rwkv3}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
DS_BUCKET_MB="${DS_BUCKET_MB:-64}"
DS_OFFLOAD_PIN_MEMORY="${DS_OFFLOAD_PIN_MEMORY:-1}"
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD="${DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD:-0}"
DS_STAGE3_PREFETCH_BUCKET_SIZE="${DS_STAGE3_PREFETCH_BUCKET_SIZE:-5000000}"
DS_STAGE3_MAX_LIVE_PARAMETERS="${DS_STAGE3_MAX_LIVE_PARAMETERS:-200000000}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/.torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-16}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

if [[ ! -e "$LOAD_MODEL" ]]; then
  echo "Missing LOAD_MODEL: $LOAD_MODEL" >&2
  exit 1
fi

for suffix in .bin .idx .mask.bin .mask.idx; do
  if [[ ! -f "${DATA_FILE}${suffix}" ]]; then
    echo "Missing SFT dataset file: ${DATA_FILE}${suffix}" >&2
    exit 1
  fi
done

mkdir -p "$PROJ_DIR"
mkdir -p "$TORCH_EXTENSIONS_DIR"

EXTRA_ARGS=()
if [[ -n "${SFT_MASK_FILE:-}" ]]; then
  EXTRA_ARGS+=(--sft_mask_file "$SFT_MASK_FILE")
fi

"$PYTHON_BIN" train.py --load_model "$LOAD_MODEL" --wandb "${WANDB_PROJECT:-}" --proj_dir "$PROJ_DIR" --my_testing "$MODEL_TYPE" \
 --ctx_len "$CTX_LEN" --train_stage 0 --epoch_steps "$EPOCH_STEPS" --epoch_count "$EPOCH_COUNT" --epoch_begin "${EPOCH_BEGIN:-0}" \
 --sft_one_pass "$SFT_ONE_PASS" \
 --sft_eval_tail_ratio "$SFT_EVAL_TAIL_RATIO" --sft_eval_tail_docs "$SFT_EVAL_TAIL_DOCS" --sft_eval_include_in_train "$SFT_EVAL_INCLUDE_IN_TRAIN" \
 --sft_eval_every_n_steps "$SFT_EVAL_EVERY_N_STEPS" --sft_eval_steps "$SFT_EVAL_STEPS" \
 --sft_train_shuffle "$SFT_TRAIN_SHUFFLE" --sft_train_shuffle_seed "$SFT_TRAIN_SHUFFLE_SEED" \
 --data_file "$DATA_FILE" --sft_masked_ce_chunk "$SFT_MASKED_CE_CHUNK" --sft_masked_fused_ce_chunk "$SFT_MASKED_FUSED_CE_CHUNK" \
 --num_nodes "$N_NODE" --micro_bsz "$MICRO_BSZ" --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES" --n_layer "$N_LAYER" --n_embd "$N_EMBD" --dim_ffn "$DIM_FFN" --kernel "$KERNEL" \
 --lr_init "$LR_INIT" --lr_final "$LR_FINAL" --lr_wsd_decay_iters "$LR_WSD_DECAY_ITERS" --lr_wsd_decay_style "$LR_WSD_DECAY_STYLE" --warmup_steps "$WARMUP_STEPS" --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
 --data_type "sft_binidx" --vocab_size "$VOCAB_SIZE" \
 --weight_decay "$WEIGHT_DECAY" --epoch_save "$EPOCH_SAVE" --save_every_n_steps "$SAVE_EVERY_N_STEPS" --keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
 --head_size "$HEAD_SIZE" --head_chunk "$HEAD_CHUNK" \
 --accelerator gpu --devices "$GPU_PER_NODE" --precision bf16 --strategy "$STRATEGY" --grad_cp "$GRAD_CP" --enable_progress_bar True --ds_bucket_mb "$DS_BUCKET_MB" \
 --ds_offload_pin_memory "$DS_OFFLOAD_PIN_MEMORY" --ds_stage3_param_persistence_threshold "$DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD" \
 --ds_stage3_prefetch_bucket_size "$DS_STAGE3_PREFETCH_BUCKET_SIZE" --ds_stage3_max_live_parameters "$DS_STAGE3_MAX_LIVE_PARAMETERS" \
 --master_port "$MASTER_PORT" \
 --d_decay_lora "$D_DECAY_LORA" --d_aaa_lora "$D_AAA_LORA" --d_mv_lora "$D_MV_LORA" --d_gate_lora "$D_GATE_LORA" \
 "${EXTRA_ARGS[@]}"
