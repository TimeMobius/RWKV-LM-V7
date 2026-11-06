#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# Self-contained production RWKV7 G1F 13.3B SFT launcher for ctx_len=86016, one full pass.
#
# Expected dataset prefix:
#   DATA_FILE.bin / DATA_FILE.idx / DATA_FILE.mask.bin / DATA_FILE.mask.idx
#
# Override any value by exporting it before the command, for example:
#   LOAD_MODEL=/path/to/rwkv7-g1f-13.3b.pth DATA_FILE=/path/to/SFT_RWKV7_13B bash run_13b_sft_ctx86016_onepass.sh
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

CTX_LEN="${CTX_LEN:-86016}"
LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B}"
PROJ_DIR="${PROJ_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-ctx86016-onepass}"

SFT_ONE_PASS="${SFT_ONE_PASS:-1}"
EPOCH_STEPS="${EPOCH_STEPS:-1}"
EPOCH_COUNT="${EPOCH_COUNT:-1}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-0}"
KEEP_LAST_N_CHECKPOINTS="${KEEP_LAST_N_CHECKPOINTS:-6}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-4}"
MICRO_BSZ="${MICRO_BSZ:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

STRATEGY="${STRATEGY:-deepspeed_stage_3_offload}"
GRAD_CP="${GRAD_CP:-1}"
HEAD_CHUNK="${HEAD_CHUNK:-0}"
SFT_MASKED_CE_CHUNK="${SFT_MASKED_CE_CHUNK:-4096}"
DS_BUCKET_MB="${DS_BUCKET_MB:-64}"
KERNEL="${KERNEL:-@rwkv3}"

LR_INIT="${LR_INIT:-5e-6}"
LR_FINAL="${LR_FINAL:-5e-7}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-0}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"

WANDB_PROJECT="${WANDB_PROJECT:-RWKV7-13B-SFT}"
MASTER_PORT="${MASTER_PORT:-29501}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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

python train.py --load_model "$LOAD_MODEL" --wandb "$WANDB_PROJECT" --proj_dir "$PROJ_DIR" --my_testing "$MODEL_TYPE" \
 --ctx_len "$CTX_LEN" --train_stage 0 --epoch_steps "$EPOCH_STEPS" --epoch_count "$EPOCH_COUNT" --epoch_begin "${EPOCH_BEGIN:-0}" \
 --sft_one_pass "$SFT_ONE_PASS" \
 --data_file "$DATA_FILE" --sft_masked_ce_chunk "$SFT_MASKED_CE_CHUNK" \
 --num_nodes "$N_NODE" --micro_bsz "$MICRO_BSZ" --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES" --n_layer "$N_LAYER" --n_embd "$N_EMBD" --dim_ffn "$DIM_FFN" --kernel "$KERNEL" \
 --lr_init "$LR_INIT" --lr_final "$LR_FINAL" --lr_wsd_decay_iters "$LR_WSD_DECAY_ITERS" --lr_wsd_decay_style "$LR_WSD_DECAY_STYLE" --warmup_steps "$WARMUP_STEPS" --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
 --data_type "sft_binidx" --vocab_size "$VOCAB_SIZE" \
 --weight_decay "$WEIGHT_DECAY" --epoch_save "$EPOCH_SAVE" --save_every_n_steps "$SAVE_EVERY_N_STEPS" --keep_last_n_checkpoints "$KEEP_LAST_N_CHECKPOINTS" \
 --head_size "$HEAD_SIZE" --head_chunk "$HEAD_CHUNK" \
 --accelerator gpu --devices "$GPU_PER_NODE" --precision bf16 --strategy "$STRATEGY" --grad_cp "$GRAD_CP" --enable_progress_bar True --ds_bucket_mb "$DS_BUCKET_MB" --master_port "$MASTER_PORT" \
 --d_decay_lora "$D_DECAY_LORA" --d_aaa_lora "$D_AAA_LORA" --d_mv_lora "$D_MV_LORA" --d_gate_lora "$D_GATE_LORA" \
 "${EXTRA_ARGS[@]}"
