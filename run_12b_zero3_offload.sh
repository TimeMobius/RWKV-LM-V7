#!/bin/bash
#######################################################################################################################
#
# Directly continue training from an existing 12B checkpoint
#
# First run calc_rwkv12b_params.py, then copy VOCAB_SIZE / MY_EXIT_TOKENS / MAGIC_PRIME into this script
#
# !!! If launch gets stuck, clean lock files in your TORCH_EXTENSIONS_DIR !!!
#
#######################################################################################################################
#
MODEL_TYPE="x070" # x070 => rwkv-7.0
#
N_LAYER="61"
N_EMBD="4096"
#
CTX_LEN="86016" # if you change ctx_len, rerun calc_rwkv12b_params.py
PROJ_DIR="/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/outs/12b-zero3-offload"
LOAD_MODEL="/mnt/data/Models/RWKV-7/rwkv7-g1f-13.3b-20260415-ctx8192.pth"
DATA_FILE="/mnt/data/Codes/RWKV/megatron_data_process/data/test_text_document" # set dataset prefix, DO NOT add .bin
#
# !!! by default train.py will load the checkpoint specified by --load_model here !!!
#
#######################################################################################################################
#
# Fill these 3 values from calc_rwkv12b_params.py output
#
VOCAB_SIZE="65536"
MY_EXIT_TOKENS="18972636245"
MAGIC_PRIME="220559"
#
#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Large model => use smaller LR
# Continuing from a full pretrained model => use a small LR first
#
M_BSZ="1"
LR_INIT="2e-5"
LR_FINAL="2e-6"
GRAD_CP=1 # 1 => slower, save VRAM; 0 => faster, more VRAM
HEAD_CHUNK=0 # 0 => faster, more VRAM; 4096 / 65536 => slower, less VRAM
KERNEL="@rwkv3" # usually faster on H100 / H800
EPOCH_SAVE=10
#
#######################################################################################################################
#
N_NODE=1 # number of nodes
GPU_PER_NODE=8 # number of GPUs per node
DS_BUCKET_MB=64
#
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_EXTENSIONS_DIR="/mnt/data/Codes/RWKV/RWKV-LM-V7-12B-train/.torch_extensions"
export MAX_JOBS=16

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo
# export NCCL_IB_DISABLE=1
# export OMP_NUM_THREADS=4
# export MKL_NUM_THREADS=4
# export OPENBLAS_NUM_THREADS=4
# export NUMEXPR_NUM_THREADS=4

#
mkdir -p "$PROJ_DIR"
mkdir -p "$TORCH_EXTENSIONS_DIR"
#
python train.py --load_model "$LOAD_MODEL" --wandb "" --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --train_stage 0 --epoch_count 999999 --epoch_begin 0 \
 --data_file "$DATA_FILE" --my_exit_tokens $MY_EXIT_TOKENS --magic_prime $MAGIC_PRIME \
 --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --kernel $KERNEL \
 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 100 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 --data_type "binidx" --vocab_size $VOCAB_SIZE \
 --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size 64 --head_chunk $HEAD_CHUNK \
 --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_3_offload --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB \
 --d_decay_lora 192 --d_aaa_lora 192 --d_mv_lora 128 --d_gate_lora 384 --save_every_n_steps 20 --keep_last_n_checkpoints 2