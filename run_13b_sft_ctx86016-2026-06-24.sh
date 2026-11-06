#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="x070"

N_LAYER="61"
N_EMBD="4096"
DIM_FFN="16384"
VOCAB_SIZE="65536"
HEAD_SIZE="64"

D_DECAY_LORA="192"
D_AAA_LORA="192"
D_MV_LORA="128"
D_GATE_LORA="384"

CTX_LEN="86016"
LOAD_MODEL="/mnt/data/Models/RWKV-7/rwkv7-g1g-13.3b-20260523-ctx8192.pth"
DATA_FILE="/mnt/data/Datasets/SFT_RWKV7_13B_2026_06_23_mask_1/results/SFT_RWKV7_13B"
PROJ_DIR="/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-ctx86016-onepass-2026-06-23"
SFT_MASK_FILE=""

# 关闭 one-pass，手动按 14 天预算控制 optimizer step
SFT_ONE_PASS="0"
EPOCH_STEPS="5000"
EPOCH_COUNT="1"
EPOCH_BEGIN="0"

# accum=4 配合 14 天预算手：
#   EPOCH_STEPS=10000 (按 ~123s/step × 14天)
#   LR_WSD_DECAY_ITERS=3500 (decay 占总长 35%)
#   SAVE/EVAL=350 (~12h/次)
#   WARMUP=100
EPOCH_SAVE="1"
SAVE_EVERY_N_STEPS="350"
KEEP_LAST_N_CHECKPOINTS="8"

SFT_EVAL_TAIL_RATIO="0.005"
SFT_EVAL_TAIL_DOCS="0"
SFT_EVAL_INCLUDE_IN_TRAIN="0"
SFT_EVAL_EVERY_N_STEPS="350"
SFT_EVAL_STEPS="10"
SFT_TRAIN_SHUFFLE="1"
SFT_TRAIN_SHUFFLE_SEED="1234"

N_NODE="1"
GPU_PER_NODE="8"
MICRO_BSZ="1"
ACCUMULATE_GRAD_BATCHES="4"

STRATEGY="deepspeed_stage_3_offload"
GRAD_CP="1"
HEAD_CHUNK="0"
SFT_MASKED_CE_CHUNK="0"
SFT_MASKED_FUSED_CE_CHUNK="4096"

DS_BUCKET_MB="64"
DS_OFFLOAD_PIN_MEMORY="1"
DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD="0"
DS_STAGE3_PREFETCH_BUCKET_SIZE="5000000"
DS_STAGE3_MAX_LIVE_PARAMETERS="200000000"

KERNEL="@rwkv3"

LR_INIT="5e-6"
LR_FINAL="5e-7"
LR_WSD_DECAY_ITERS="1750"
LR_WSD_DECAY_STYLE="cosine"
WARMUP_STEPS="100"
WEIGHT_DECAY="0.001"

WANDB_PROJECT="RWKV7-13B-SFT"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29501"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NCCL_DEBUG="WARN"
CUDA_DEVICE_MAX_CONNECTIONS="1"
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
TORCH_EXTENSIONS_DIR="/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/.torch_extensions"
TORCH_CUDA_ARCH_LIST="9.0"
MAX_JOBS="16"

export CUDA_VISIBLE_DEVICES
export NCCL_DEBUG
export CUDA_DEVICE_MAX_CONNECTIONS
export PYTORCH_CUDA_ALLOC_CONF
export TORCH_EXTENSIONS_DIR
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export MASTER_ADDR
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
if [[ -n "$SFT_MASK_FILE" ]]; then
  EXTRA_ARGS+=(--sft_mask_file "$SFT_MASK_FILE")
fi

python train.py --load_model "$LOAD_MODEL" --wandb "$WANDB_PROJECT" --proj_dir "$PROJ_DIR" --my_testing "$MODEL_TYPE" \
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
 --data_type "sft_binidx" --vocab_size "$VOCAB_SIZE" \
 --weight_decay "$WEIGHT_DECAY" --epoch_save "$EPOCH_SAVE" --save_every_n_steps "$SAVE_EVERY_N_STEPS" --keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
 --head_size "$HEAD_SIZE" --head_chunk "$HEAD_CHUNK" \
 --accelerator gpu --devices "$GPU_PER_NODE" --precision bf16 --strategy "$STRATEGY" --grad_cp "$GRAD_CP" \
 --enable_progress_bar True --ds_bucket_mb "$DS_BUCKET_MB" \
 --ds_offload_pin_memory "$DS_OFFLOAD_PIN_MEMORY" \
 --ds_stage3_param_persistence_threshold "$DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD" \
 --ds_stage3_prefetch_bucket_size "$DS_STAGE3_PREFETCH_BUCKET_SIZE" \
 --ds_stage3_max_live_parameters "$DS_STAGE3_MAX_LIVE_PARAMETERS" \
 --master_port "$MASTER_PORT" \
 --d_decay_lora "$D_DECAY_LORA" --d_aaa_lora "$D_AAA_LORA" --d_mv_lora "$D_MV_LORA" --d_gate_lora "$D_GATE_LORA" \
 "${EXTRA_ARGS[@]}"