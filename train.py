########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import os
import re

from src.sft_split import compute_sft_tail_eval_count

logging.basicConfig(level=logging.INFO)

EPOCH_CKPT_PATTERN = re.compile(r"^rwkv-(init|\d+)\.pth$")
STEP_CKPT_PATTERN = re.compile(r"^rwkv-step-(\d+)\.pth$")


def parse_epoch_checkpoint_name(name: str):
    match = EPOCH_CKPT_PATTERN.match(name)
    if not match:
        return None
    token = match.group(1)
    return -1 if token == "init" else int(token)


def parse_step_checkpoint_name(name: str):
    match = STEP_CKPT_PATTERN.match(name)
    if not match:
        return None
    return int(match.group(1))


def is_deepspeed_strategy(strategy: str) -> bool:
    return "deepspeed" in str(strategy)


def is_deepspeed_checkpoint_dir(path: str) -> bool:
    if not path or not os.path.isdir(path) or not path.endswith(".pth"):
        return False
    try:
        entries = set(os.listdir(path))
    except OSError:
        return False
    if "latest" in entries:
        return True
    return any(
        name.endswith("_model_states.pt") or name.endswith("_optim_states.pt")
        for name in entries
    )


def find_stage2_checkpoint(proj_dir: str) -> tuple[str, int, int | None]:
    """Return the newest epoch checkpoint and its optional fallback slot."""
    slots = []
    for name in os.listdir(proj_dir):
        slot = parse_epoch_checkpoint_name(name)
        if slot is None:
            continue
        path = os.path.join(proj_dir, name)
        if os.path.isfile(path) or os.path.isdir(path):
            slots.append(slot)

    if not slots:
        init_path = os.path.join(proj_dir, "rwkv-init.pth")
        if os.path.isfile(init_path) or os.path.isdir(init_path):
            return init_path, -1, None
        raise FileNotFoundError(
            f"train_stage>=2 found no epoch checkpoints in {proj_dir!r} "
            "and no rwkv-init.pth fallback."
        )

    slots.sort()
    max_slot = slots[-1]
    load_name = "rwkv-init.pth" if max_slot == -1 else f"rwkv-{max_slot}.pth"
    previous_slot = slots[-2] if len(slots) > 1 else None
    return os.path.join(proj_dir, load_name), max_slot, previous_slot


def resolve_resume_checkpoint_path(path: str, strategy: str):
    if not path:
        return None
    if is_deepspeed_checkpoint_dir(path):
        if not is_deepspeed_strategy(strategy):
            raise ValueError(
                f"Checkpoint directory {path} is a DeepSpeed sharded checkpoint. "
                "Please resume it with a deepspeed strategy."
            )
        return path
    return None


def configure_epoch_schedule(args):
    if args.data_type == "binidx":
        args.epoch_count = args.magic_prime // 40320
        args.epoch_steps = 40320 // args.real_bsz
        assert args.epoch_steps * args.real_bsz == 40320
        return
    if args.data_type == "sft_binidx":
        if int(getattr(args, "sft_one_pass", 0) or 0):
            return
        if args.epoch_steps <= 0:
            raise ValueError("epoch_steps must be positive for sft_binidx training.")
        if args.epoch_count <= 0:
            raise ValueError("epoch_count must be positive for sft_binidx training.")
        return
    raise ValueError(f"Unsupported data_type: {args.data_type}")


def configure_training_limits(args):
    if args.data_type == "sft_binidx":
        args.max_epochs = args.epoch_count
    else:
        args.max_epochs = -1


def count_binidx_documents(prefix_path: str) -> int:
    from src.binidx import MMapIndexedDataset, index_file_path

    index = MMapIndexedDataset.Index(index_file_path(prefix_path))
    return len(index)


def normalize_accumulate_grad_batches(args):
    value = getattr(args, "accumulate_grad_batches", 1)
    if value is None:
        value = 1
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("accumulate_grad_batches must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("accumulate_grad_batches must be a positive integer.")
    args.accumulate_grad_batches = value
    return value


def configure_batch_sizes(args):
    accumulation = normalize_accumulate_grad_batches(args)
    if args.data_type == "sft_binidx":
        args.effective_bsz = args.real_bsz * accumulation
    else:
        args.effective_bsz = args.real_bsz


def configure_samples_per_epoch(args):
    accumulation = normalize_accumulate_grad_batches(args)
    if args.data_type == "sft_binidx":
        args.samples_per_epoch = args.epoch_steps * args.real_bsz * accumulation
    else:
        args.samples_per_epoch = args.epoch_steps * args.real_bsz


def configure_sft_one_pass(args):
    enabled = int(getattr(args, "sft_one_pass", 0) or 0)
    args.sft_one_pass = enabled
    args.sft_one_pass_documents = 0
    args.sft_eval_tail_documents = 0
    if not enabled:
        return
    if args.data_type != "sft_binidx":
        raise ValueError("--sft_one_pass only supports data_type=sft_binidx.")
    if args.effective_bsz <= 0:
        raise ValueError("effective_bsz must be positive for --sft_one_pass.")

    total_document_count = count_binidx_documents(args.data_file)
    if total_document_count <= 0:
        raise ValueError("SFT token dataset must contain at least one document for --sft_one_pass.")
    eval_count = compute_sft_tail_eval_count(
        total_document_count,
        float(getattr(args, "sft_eval_tail_ratio", 0.0) or 0.0),
        int(getattr(args, "sft_eval_tail_docs", 0) or 0),
        require_train_docs=not int(getattr(args, "sft_eval_include_in_train", 0) or 0),
    )
    if int(getattr(args, "sft_eval_include_in_train", 0) or 0):
        document_count = total_document_count
    else:
        document_count = total_document_count - eval_count

    args.sft_one_pass_documents = document_count
    args.sft_eval_tail_documents = eval_count
    args.epoch_steps = (document_count + args.effective_bsz - 1) // args.effective_bsz
    args.epoch_count = 1


def validate_sft_eval_settings(args):
    eval_every = int(getattr(args, "sft_eval_every_n_steps", 0) or 0)
    eval_steps = int(getattr(args, "sft_eval_steps", 0) or 0)
    tail_ratio = float(getattr(args, "sft_eval_tail_ratio", 0.0) or 0.0)
    tail_docs = int(getattr(args, "sft_eval_tail_docs", 0) or 0)
    include_in_train = int(getattr(args, "sft_eval_include_in_train", 0) or 0)
    if eval_every < 0:
        raise ValueError("sft_eval_every_n_steps must be non-negative.")
    if eval_steps < 0:
        raise ValueError("sft_eval_steps must be non-negative.")
    if tail_docs < 0:
        raise ValueError("sft_eval_tail_docs must be non-negative.")
    if tail_ratio < 0 or tail_ratio > 1:
        raise ValueError("sft_eval_tail_ratio must be in [0, 1].")
    if include_in_train not in (0, 1):
        raise ValueError("sft_eval_include_in_train must be 0 or 1.")
    if not include_in_train and tail_ratio >= 1:
        raise ValueError("sft_eval_tail_ratio must be < 1 for held-out eval.")
    if eval_every > 0:
        if args.data_type != "sft_binidx":
            raise ValueError("--sft_eval_every_n_steps only supports data_type=sft_binidx.")
        if eval_steps <= 0:
            raise ValueError("sft_eval_steps must be positive when SFT eval is enabled.")
        if tail_ratio <= 0 and tail_docs <= 0:
            raise ValueError("SFT eval requires sft_eval_tail_ratio > 0 or sft_eval_tail_docs > 0.")


def validate_sft_train_shuffle_settings(args):
    enabled = int(getattr(args, "sft_train_shuffle", 0) or 0)
    if enabled not in (0, 1):
        raise ValueError("sft_train_shuffle must be 0 or 1.")
    seed = getattr(args, "sft_train_shuffle_seed", 1234)
    if seed is None:
        seed = 1234
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("sft_train_shuffle_seed must be a non-negative integer.") from exc
    if seed < 0:
        raise ValueError("sft_train_shuffle_seed must be a non-negative integer.")
    args.sft_train_shuffle = enabled
    args.sft_train_shuffle_seed = seed


def validate_sft_loss_settings(args):
    if getattr(args, "sft_masked_ce_chunk", 0) < 0:
        raise ValueError("sft_masked_ce_chunk must be a non-negative integer.")
    if getattr(args, "sft_masked_fused_ce_chunk", 0) < 0:
        raise ValueError("sft_masked_fused_ce_chunk must be a non-negative integer.")
    if getattr(args, "sft_masked_ce_chunk", 0) > 0 and getattr(args, "sft_masked_fused_ce_chunk", 0) > 0:
        raise ValueError("Use either sft_masked_ce_chunk or sft_masked_fused_ce_chunk, not both.")


def _deepspeed_int_arg(args, name: str, default: int = -1) -> int:
    value = getattr(args, name, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def configure_deepspeed_zero3_config(args, strategy_config):
    if not is_deepspeed_strategy(getattr(args, "strategy", "")):
        return

    zero = strategy_config.setdefault("zero_optimization", {})
    ds_bucket_mb = _deepspeed_int_arg(args, "ds_bucket_mb", 0)
    if ds_bucket_mb < 0:
        raise ValueError("ds_bucket_mb must be non-negative.")
    if ds_bucket_mb > 0:
        bucket_bytes = ds_bucket_mb * 1000 * 1000
        zero["allgather_bucket_size"] = bucket_bytes
        zero["reduce_bucket_size"] = bucket_bytes

    pin_memory = _deepspeed_int_arg(args, "ds_offload_pin_memory", -1)
    if pin_memory not in (-1, 0, 1):
        raise ValueError("ds_offload_pin_memory must be -1, 0, or 1.")
    if pin_memory >= 0:
        for offload_key in ("offload_optimizer", "offload_param"):
            offload_config = zero.get(offload_key)
            if isinstance(offload_config, dict):
                offload_config["pin_memory"] = bool(pin_memory)

    stage3_options = {
        "ds_stage3_param_persistence_threshold": "stage3_param_persistence_threshold",
        "ds_stage3_prefetch_bucket_size": "stage3_prefetch_bucket_size",
        "ds_stage3_max_live_parameters": "stage3_max_live_parameters",
    }
    for arg_name, zero_key in stage3_options.items():
        value = _deepspeed_int_arg(args, arg_name, -1)
        if value < -1:
            raise ValueError(f"{arg_name} must be -1 or non-negative.")
        if value >= 0:
            zero[zero_key] = value


if __name__ == "__main__":  # pragma: no cover
    import os
    import subprocess
    import sys
    from argparse import ArgumentParser
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    import pytorch_lightning as pl

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument("--sft_mask_file", default="", type=str)
    parser.add_argument("--sft_pad_token_id", default=65532, type=int)
    parser.add_argument("--sft_eval_tail_ratio", default=0.0, type=float)  # SFT only: reserve tail ratio for held-out eval
    parser.add_argument("--sft_eval_tail_docs", default=0, type=int)  # SFT only: reserve exact tail docs for held-out eval
    parser.add_argument("--sft_eval_include_in_train", default=0, type=int)  # 1 means eval tail is also used by train
    parser.add_argument("--sft_eval_every_n_steps", default=0, type=int)  # SFT only: run tail-split eval every N optimizer steps
    parser.add_argument("--sft_eval_steps", default=0, type=int)  # SFT only: eval micro-batch steps per rank
    parser.add_argument("--sft_train_shuffle", default=0, type=int)  # SFT only: 1 shuffles train docs with an epoch-stable permutation
    parser.add_argument("--sft_train_shuffle_seed", default=1234, type=int)  # SFT only: seed for train doc permutation
    parser.add_argument("--vocab_size", default=0, type=int)  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_count", default=500, type=int)  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--sft_one_pass", default=0, type=int)  # SFT only: auto epoch_steps=ceil(num_docs/effective_bsz), epoch_count=1
    parser.add_argument("--sft_masked_ce_chunk", default=0, type=int)  # SFT only: 0 disables, >0 chunks trainable-token head CE
    parser.add_argument("--sft_masked_fused_ce_chunk", default=0, type=int)  # SFT CUDA only: 0 disables, >0 uses fused masked head CE
    parser.add_argument("--epoch_begin", default=0, type=int)  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--save_every_n_steps", default=0, type=int)  # save every N real steps (0 to disable)
    parser.add_argument("--save_at_step", default=0, type=int)  # save once at this real step (0 to disable)
    parser.add_argument("--keep_last_n_checkpoints", default=0, type=int)  # keep only the most recent N numbered checkpoints (0 to disable)

    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument("--lr_init", default=6e-4, type=float)  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--lr_wsd_decay_iters", default=0, type=int)  # SFT only: decay over the final N optimizer steps
    parser.add_argument("--lr_wsd_decay_style", default="cosine", type=str)  # SFT only: none, linear, cosine
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=0, type=float) # try 0.1
    parser.add_argument("--grad_clip", default=1.0, type=float) # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--train_stage", default=0, type=int)  # my special pile mode
    parser.add_argument("--ds_bucket_mb", default=200, type=int)  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument("--ds_offload_pin_memory", default=-1, type=int)  # -1 keep strategy default, 0 false, 1 true
    parser.add_argument("--ds_stage3_param_persistence_threshold", default=-1, type=int)  # -1 keep strategy default
    parser.add_argument("--ds_stage3_prefetch_bucket_size", default=-1, type=int)  # DeepSpeed param elements, -1 keep default
    parser.add_argument("--ds_stage3_max_live_parameters", default=-1, type=int)  # DeepSpeed param elements, -1 keep default
    parser.add_argument("--dist_timeout_sec", default=1800, type=int)
    parser.add_argument("--master_port", default=29501, type=int)

    parser.add_argument("--head_size", default=64, type=int) # can try larger values for larger models
    parser.add_argument("--head_chunk", default=0, type=int) # 0 = fast, takes more VRAM; 65536 = saves 70% VRAM (when your bsz is large), slower; 4096 = saves 80% VRAM (when your bsz is large), slower
    parser.add_argument("--d_decay_lora", default=0, type=int)
    parser.add_argument("--d_aaa_lora", default=0, type=int)
    parser.add_argument("--d_mv_lora", default=0, type=int)
    parser.add_argument("--d_gate_lora", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--kernel", default="", type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    def _should_auto_torchrun(parsed_args) -> bool:
        if "deepspeed" not in str(parsed_args.strategy):
            return False
        if int(parsed_args.num_nodes) != 1:
            return False
        if int(parsed_args.devices) <= 1:
            return False
        if os.environ.get("LOCAL_RANK") is not None:
            return False
        if os.environ.get("WORLD_SIZE") is not None:
            return False
        return True

    if _should_auto_torchrun(args):
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(args.devices),
            "--master_port",
            str(args.master_port),
            os.path.abspath(__file__),
            *sys.argv[1:],
        ]
        print(f"########## Re-launching with torchrun: {' '.join(cmd)} ##########")
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    ########################################################################################################

    import warnings, math, datetime, time
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # pretrain continues forever unless my_exit_tokens stops it
    args.betas = (args.beta1, args.beta2)
    validate_sft_loss_settings(args)
    validate_sft_eval_settings(args)
    validate_sft_train_shuffle_settings(args)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    configure_batch_sizes(args)
    os.environ["DEEPSPEED_TIMEOUT"] = str(args.dist_timeout_sec)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_KERNEL"] = args.kernel
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = str(args.head_chunk)
    os.environ["RWKV_SFT_MASKED_FUSED_CE_CHUNK"] = str(args.sft_masked_fused_ce_chunk)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    configure_epoch_schedule(args)
    configure_sft_one_pass(args)
    configure_training_limits(args)
    configure_samples_per_epoch(args)

    if args.train_stage >= 2:  # find latest saved model
        args.load_model, max_p, args.my_pile_prev_p = find_stage2_checkpoint(args.proj_dir)
        if max_p >= 0 and args.warmup_steps < 0:
            args.warmup_steps = 10
        if not is_deepspeed_checkpoint_dir(args.load_model):
            args.epoch_begin = max_p + 1

    args.resume_ckpt_path = resolve_resume_checkpoint_path(args.load_model, args.strategy)
    if args.resume_ckpt_path:
        args.epoch_begin = 0

    args.resume_global_step = 0
    args.resume_epoch = args.epoch_begin
    args.resume_step_offset = 0
    step_ckpt = parse_step_checkpoint_name(os.path.basename(args.load_model))
    epoch_ckpt = parse_epoch_checkpoint_name(os.path.basename(args.load_model))
    if args.resume_ckpt_path and step_ckpt is not None:
        args.resume_global_step = step_ckpt
        args.resume_epoch = step_ckpt // args.epoch_steps
        args.resume_step_offset = step_ckpt % args.epoch_steps
    elif args.resume_ckpt_path and epoch_ckpt is not None and epoch_ckpt >= 0:
        args.resume_epoch = epoch_ckpt

    if args.resume_ckpt_path:
        rank_zero_info(
            f"########## Preloading resume position: global_step={args.resume_global_step} "
            f"epoch={args.resume_epoch} step_offset={args.resume_step_offset}/{args.epoch_steps} ##########"
        )

    samples_per_epoch = args.samples_per_epoch
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    sft_one_pass_line = ""
    if getattr(args, "sft_one_pass", 0):
        repeated_samples = samples_per_epoch - args.sft_one_pass_documents
        sft_one_pass_line = (
            f"# SFT one pass = {args.sft_one_pass_documents} documents, "
            f"ceil -> {args.epoch_steps} steps, repeated tail samples {repeated_samples}\n#\n"
        )
    sft_eval_line = ""
    if args.data_type == "sft_binidx" and (
        float(getattr(args, "sft_eval_tail_ratio", 0.0) or 0.0) > 0
        or int(getattr(args, "sft_eval_tail_docs", 0) or 0) > 0
    ):
        total_documents = count_binidx_documents(args.data_file)
        eval_documents = compute_sft_tail_eval_count(
            total_documents,
            float(getattr(args, "sft_eval_tail_ratio", 0.0) or 0.0),
            int(getattr(args, "sft_eval_tail_docs", 0) or 0),
            require_train_docs=not int(getattr(args, "sft_eval_include_in_train", 0) or 0),
        )
        eval_mode = "overlap" if int(getattr(args, "sft_eval_include_in_train", 0) or 0) else "heldout"
        train_documents = total_documents if eval_mode == "overlap" else total_documents - eval_documents
        args.sft_eval_tail_documents = eval_documents
        sft_eval_line = (
            f"# SFT tail eval split = mode {eval_mode}, train {train_documents} documents, "
            f"eval {eval_documents} documents, eval every {args.sft_eval_every_n_steps} steps "
            f"for {args.sft_eval_steps} micro-batches/rank\n#\n"
        )
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, accumulate_grad_batches {args.accumulate_grad_batches}, effective_bsz {args.effective_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
{sft_one_pass_line}\
{sft_eval_line}\
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx", "sft_binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info("\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0" # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    if "deepspeed" in args.strategy and dist.is_available() and not dist.is_initialized():
        local_rank = os.environ.get("LOCAL_RANK")
        world_size = os.environ.get("WORLD_SIZE")
        rank = os.environ.get("RANK")
        if local_rank is not None and world_size is not None and rank is not None:
            backend = "nccl" if args.accelerator == "gpu" else "gloo"
            if backend == "nccl":
                torch.cuda.set_device(int(local_rank))
            dist.init_process_group(
                backend=backend,
                init_method="env://",
            )

    ########################################################################################################

    from src.trainer import train_callback, generate_init_weight
    from src.dataset import MyDataset

    train_data = MyDataset(args, sft_split="train")
    eval_loader = None
    if args.data_type == "sft_binidx" and args.sft_eval_every_n_steps > 0:
        eval_data = MyDataset(args, sft_split="eval")
        eval_loader = DataLoader(
            eval_data,
            shuffle=False,
            pin_memory=True,
            batch_size=args.micro_bsz,
            num_workers=1,
            persistent_workers=False,
            drop_last=True,
        )
    args.vocab_size = train_data.vocab_size

    from src.model import RWKV
    model = RWKV(args)

    if len(args.load_model) == 0 or args.train_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    if args.resume_ckpt_path:
        rank_zero_info(f"########## Resuming trainer state from {args.resume_ckpt_path}... ##########")
        if args.load_partial == 1:
            raise ValueError("load_partial=1 is not supported when resuming from a DeepSpeed checkpoint directory.")
    else:
        rank_zero_info(f"########## Loading {args.load_model}... ##########")
        try:
            load_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
            load_keys = list(load_dict.keys())
            for k in load_keys:
                if k.startswith('_forward_module.'):
                    load_dict[k.replace('_forward_module.','')] = load_dict[k]
                    del load_dict[k]
        except:
            rank_zero_info(f"Bad checkpoint {args.load_model}")
            if args.train_stage >= 2:  # try again using another checkpoint
                previous_slot = getattr(args, "my_pile_prev_p", None)
                if previous_slot is None:
                    fallback_path = os.path.join(args.proj_dir, "rwkv-init.pth")
                    if fallback_path == args.load_model or not (
                        os.path.isfile(fallback_path) or os.path.isdir(fallback_path)
                    ):
                        raise RuntimeError(
                            f"Could not load checkpoint {args.load_model!r}; "
                            "no previous checkpoint or rwkv-init.pth fallback is available."
                        )
                    args.load_model = fallback_path
                    max_p = -1
                elif previous_slot == -1:
                    args.load_model = os.path.join(args.proj_dir, "rwkv-init.pth")
                    max_p = -1
                else:
                    args.load_model = os.path.join(args.proj_dir, f"rwkv-{previous_slot}.pth")
                    max_p = previous_slot
                if not is_deepspeed_checkpoint_dir(args.load_model):
                    args.epoch_begin = max_p + 1
                args.resume_ckpt_path = resolve_resume_checkpoint_path(args.load_model, args.strategy)
                rank_zero_info(f"Trying {args.load_model}")
                if args.resume_ckpt_path:
                    load_dict = None
                else:
                    load_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
            else:
                raise

        if args.resume_ckpt_path:
            rank_zero_info(f"########## Resuming trainer state from {args.resume_ckpt_path}... ##########")
            if args.load_partial == 1:
                raise ValueError("load_partial=1 is not supported when resuming from a DeepSpeed checkpoint directory.")
        else:
            if args.load_partial == 1:
                load_keys = load_dict.keys()
                for k in model.state_dict():
                    if k not in load_keys:
                        load_dict[k] = model.state_dict()[k]
            model.load_state_dict(load_dict)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args, eval_loader=eval_loader)],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in args.strategy:
        configure_deepspeed_zero3_config(args, trainer.strategy.config)

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, shuffle=False, pin_memory=True, batch_size=args.micro_bsz, num_workers=1, persistent_workers=False, drop_last=True)

    if trainer.global_rank == 0:
        print(f'### Preparing for training (loaded {args.load_model}). Please wait...')
    trainer.fit(model, data_loader, ckpt_path=args.resume_ckpt_path)
