#!/usr/bin/env bash
set -euo pipefail

#######################################################################################################################
#
# Short 13.3B SFT profiling launcher for ctx_len=86016.
#
# This is a diagnostic entry point, not the production long-run script. It runs a small number of optimizer steps,
# disables wandb/checkpoint saving, records train.py output, samples GPU state, and can optionally wrap the run in
# Nsight Systems.
#
# Typical usage:
#   PROFILE_STEPS=8 PROFILE_MODE=nsys NCCL_DEBUG=INFO bash run_13b_sft_profile.sh
#
#######################################################################################################################

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
source "$REPO_DIR/scripts/setup_rwkv_cuda_env.sh"

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
LOAD_MODEL="${LOAD_MODEL:-/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b.pth}"
DATA_FILE="${DATA_FILE:-/mnt/data/Datasets/SFT_RWKV7_13B/results/SFT_RWKV7_13B}"
PROFILE_ROOT="${PROFILE_ROOT:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/13b-sft-profile}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
PROJ_DIR="${PROJ_DIR:-${PROFILE_ROOT}/${RUN_TAG}}"

PROFILE_STEPS="${PROFILE_STEPS:-8}"
PROFILE_MODE="${PROFILE_MODE:-none}" # none | nsys
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas}"
NSYS_STATS="${NSYS_STATS:-1}"
SFT_EVAL_TAIL_RATIO="${SFT_EVAL_TAIL_RATIO:-0}"
SFT_EVAL_TAIL_DOCS="${SFT_EVAL_TAIL_DOCS:-0}"
SFT_EVAL_INCLUDE_IN_TRAIN="${SFT_EVAL_INCLUDE_IN_TRAIN:-0}"
SFT_EVAL_EVERY_N_STEPS="${SFT_EVAL_EVERY_N_STEPS:-0}"
SFT_EVAL_STEPS="${SFT_EVAL_STEPS:-0}"
SFT_TRAIN_SHUFFLE="${SFT_TRAIN_SHUFFLE:-0}"
SFT_TRAIN_SHUFFLE_SEED="${SFT_TRAIN_SHUFFLE_SEED:-1234}"

GPU_MONITOR="${GPU_MONITOR:-1}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-2}"
CPU_MONITOR="${CPU_MONITOR:-1}"
CPU_MONITOR_INTERVAL="${CPU_MONITOR_INTERVAL:-2}"

N_NODE="${N_NODE:-1}"
GPU_PER_NODE="${GPU_PER_NODE:-8}"
MICRO_BSZ="${MICRO_BSZ:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"

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

LR_INIT="${LR_INIT:-5e-6}"
LR_FINAL="${LR_FINAL:-5e-7}"
LR_WSD_DECAY_ITERS="${LR_WSD_DECAY_ITERS:-0}"
LR_WSD_DECAY_STYLE="${LR_WSD_DECAY_STYLE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"

MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/.torch_extensions}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export MAX_JOBS="${MAX_JOBS:-16}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,NET}"
export DEEPSPEED_LOG_LEVEL="${DEEPSPEED_LOG_LEVEL:-info}"

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

export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${PROJ_DIR}/nccl.%h.%p.log}"

GPU_MONITOR_PID=""
CPU_MONITOR_PID=""

cleanup() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then
    kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$CPU_MONITOR_PID" ]]; then
    kill "$CPU_MONITOR_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "$GPU_MONITOR" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  (
    echo "wall_time,timestamp,index,utilization.gpu [%],utilization.memory [%],memory.used [MiB],memory.total [MiB],power.draw [W]"
    while true; do
      wall_time="$(date --iso-8601=seconds)"
      nvidia-smi \
        --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
        --format=csv,noheader,nounits | sed "s/^/${wall_time},/"
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) > "${PROJ_DIR}/gpu_monitor.csv" 2>&1 &
  GPU_MONITOR_PID="$!"
fi

if [[ "$CPU_MONITOR" == "1" ]] && command -v vmstat >/dev/null 2>&1; then
  vmstat "$CPU_MONITOR_INTERVAL" > "${PROJ_DIR}/vmstat.log" 2>&1 &
  CPU_MONITOR_PID="$!"
fi

TRAIN_ARGS=(
  --load_model "$LOAD_MODEL"
  --wandb ""
  --proj_dir "$PROJ_DIR"
  --my_testing "$MODEL_TYPE"
  --ctx_len "$CTX_LEN"
  --train_stage 0
  --epoch_steps 1000000
  --epoch_count 999999
  --epoch_begin 0
  --sft_one_pass 0
  --sft_eval_tail_ratio "$SFT_EVAL_TAIL_RATIO"
  --sft_eval_tail_docs "$SFT_EVAL_TAIL_DOCS"
  --sft_eval_include_in_train "$SFT_EVAL_INCLUDE_IN_TRAIN"
  --sft_eval_every_n_steps "$SFT_EVAL_EVERY_N_STEPS"
  --sft_eval_steps "$SFT_EVAL_STEPS"
  --sft_train_shuffle "$SFT_TRAIN_SHUFFLE"
  --sft_train_shuffle_seed "$SFT_TRAIN_SHUFFLE_SEED"
  --data_file "$DATA_FILE"
  --sft_masked_ce_chunk "$SFT_MASKED_CE_CHUNK"
  --sft_masked_fused_ce_chunk "$SFT_MASKED_FUSED_CE_CHUNK"
  --num_nodes "$N_NODE"
  --micro_bsz "$MICRO_BSZ"
  --accumulate_grad_batches "$ACCUMULATE_GRAD_BATCHES"
  --n_layer "$N_LAYER"
  --n_embd "$N_EMBD"
  --dim_ffn "$DIM_FFN"
  --kernel "$KERNEL"
  --lr_init "$LR_INIT"
  --lr_final "$LR_FINAL"
  --lr_wsd_decay_iters "$LR_WSD_DECAY_ITERS"
  --lr_wsd_decay_style "$LR_WSD_DECAY_STYLE"
  --warmup_steps "$WARMUP_STEPS"
  --beta1 0.9
  --beta2 0.99
  --adam_eps 1e-18
  --data_type "sft_binidx"
  --vocab_size "$VOCAB_SIZE"
  --weight_decay "$WEIGHT_DECAY"
  --epoch_save 0
  --save_every_n_steps 0
  --save_at_step 0
  --keep_last_n_checkpoints 0
  --head_size "$HEAD_SIZE"
  --head_chunk "$HEAD_CHUNK"
  --accelerator gpu
  --devices "$GPU_PER_NODE"
  --precision bf16
  --strategy "$STRATEGY"
  --grad_cp "$GRAD_CP"
  --enable_progress_bar True
  --ds_bucket_mb "$DS_BUCKET_MB"
  --ds_offload_pin_memory "$DS_OFFLOAD_PIN_MEMORY"
  --ds_stage3_param_persistence_threshold "$DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD"
  --ds_stage3_prefetch_bucket_size "$DS_STAGE3_PREFETCH_BUCKET_SIZE"
  --ds_stage3_max_live_parameters "$DS_STAGE3_MAX_LIVE_PARAMETERS"
  --master_port "$MASTER_PORT"
  --d_decay_lora "$D_DECAY_LORA"
  --d_aaa_lora "$D_AAA_LORA"
  --d_mv_lora "$D_MV_LORA"
  --d_gate_lora "$D_GATE_LORA"
  --max_steps "$PROFILE_STEPS"
)

if [[ "$GPU_PER_NODE" -gt 1 && "$STRATEGY" == *deepspeed* ]]; then
  CMD=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$GPU_PER_NODE" --master_port "$MASTER_PORT" train.py "${TRAIN_ARGS[@]}")
else
  CMD=("$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}")
fi

if [[ "$PROFILE_MODE" == "nsys" ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "PROFILE_MODE=nsys requested, but nsys was not found in PATH." >&2
    exit 1
  fi
  CMD=(nsys profile --force-overwrite=true --trace="$NSYS_TRACE" --sample=none --output "${PROJ_DIR}/nsys_sft_profile" "${CMD[@]}")
elif [[ "$PROFILE_MODE" != "none" ]]; then
  echo "Unknown PROFILE_MODE=${PROFILE_MODE}. Use none or nsys." >&2
  exit 1
fi

{
  printf 'PROJ_DIR=%s\n' "$PROJ_DIR"
  printf 'PROFILE_STEPS=%s\n' "$PROFILE_STEPS"
  printf 'PROFILE_MODE=%s\n' "$PROFILE_MODE"
  printf 'STRATEGY=%s\n' "$STRATEGY"
  printf 'SFT_MASKED_FUSED_CE_CHUNK=%s\n' "$SFT_MASKED_FUSED_CE_CHUNK"
  printf 'DS_BUCKET_MB=%s\n' "$DS_BUCKET_MB"
  printf 'DS_OFFLOAD_PIN_MEMORY=%s\n' "$DS_OFFLOAD_PIN_MEMORY"
  printf 'DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD=%s\n' "$DS_STAGE3_PARAM_PERSISTENCE_THRESHOLD"
  printf 'DS_STAGE3_PREFETCH_BUCKET_SIZE=%s\n' "$DS_STAGE3_PREFETCH_BUCKET_SIZE"
  printf 'DS_STAGE3_MAX_LIVE_PARAMETERS=%s\n' "$DS_STAGE3_MAX_LIVE_PARAMETERS"
  printf 'NCCL_DEBUG=%s\n' "$NCCL_DEBUG"
  printf 'NCCL_DEBUG_FILE=%s\n' "$NCCL_DEBUG_FILE"
  printf 'Command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
} | tee "${PROJ_DIR}/profile_command.txt"

"${CMD[@]}" 2>&1 | tee "${PROJ_DIR}/train.log"

if [[ "$PROFILE_MODE" == "nsys" && "$NSYS_STATS" == "1" && -f "${PROJ_DIR}/nsys_sft_profile.nsys-rep" ]]; then
  nsys stats \
    --force-overwrite=true \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum \
    "${PROJ_DIR}/nsys_sft_profile.nsys-rep" > "${PROJ_DIR}/nsys_stats.txt" 2>&1 || true
fi

echo "Profile artifacts written to: $PROJ_DIR"
