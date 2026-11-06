#!/usr/bin/env python3
"""Compute token PPL for a single RWKV-7 checkpoint on JSONL text data.

The evaluator uses the same x070 CUDA forward path as training and the legacy
20230424 tokenizer.  Windows are scored independently, which keeps the metric
well-defined for documents longer than the selected evaluation context.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--vocab-path",
        default=str(REPO_ROOT / "data" / "tokenizer" / "rwkv_vocab_v20230424.txt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means the complete JSONL file")
    parser.add_argument("--max-docs", type=int, default=0, help="0 means all documents")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def infer_dims(state: dict[str, torch.Tensor]) -> dict[str, int]:
    layer_ids = {
        int(name.split(".")[1])
        for name in state
        if name.startswith("blocks.") and name.endswith(".ln1.weight")
    }
    emb = state["emb.weight"]
    return {
        "n_layer": max(layer_ids) + 1,
        "n_embd": int(emb.shape[1]),
        "vocab_size": int(emb.shape[0]),
        "dim_ffn": int(state["blocks.0.ffn.key.weight"].shape[0]),
        "head_size": int(state["blocks.0.att.r_k"].shape[1]),
        "d_decay_lora": int(state["blocks.0.att.w1"].shape[1]),
        "d_aaa_lora": int(state["blocks.0.att.a1"].shape[1]),
        "d_mv_lora": int(state["blocks.0.att.v1"].shape[1]),
        "d_gate_lora": int(state["blocks.0.att.g1"].shape[1]),
    }


def load_model(model_path: Path, device: torch.device):
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_KERNEL", "@rwkv3")
    os.environ.setdefault("RWKV_HEAD_SIZE", "64")
    os.environ.setdefault("RWKV_JIT_ON", "0")
    os.environ.setdefault("RWKV_HEAD_L2WRAP_CE_CHUNK", "0")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")

    from src.model import RWKV

    state = torch.load(model_path, map_location="cpu", weights_only=True, mmap=True)
    dims = infer_dims(state)
    args = SimpleNamespace(**dims, my_testing="x070", grad_cp=0, ctx_len=1 << 30)
    with torch.device("meta"):
        model = RWKV(args)
    model.load_state_dict(state, assign=True)
    del state
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model, dims


def iter_texts(path: Path, max_docs: int):
    with path.open("r", encoding="utf-8") as handle:
        for doc_index, line in enumerate(handle):
            if max_docs and doc_index >= max_docs:
                break
            record = json.loads(line)
            text = record.get("text")
            if isinstance(text, str) and text:
                yield doc_index, text


def evaluate(args: argparse.Namespace) -> dict[str, float | int | str]:
    if args.window_tokens < 16 or args.window_tokens % 16:
        raise ValueError("--window-tokens must be a multiple of 16 and at least 16")
    device = torch.device(args.device)

    from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

    tokenizer = TRIE_TOKENIZER(args.vocab_path, special_first=False)
    model, dims = load_model(Path(args.model_path), device)
    print(
        f"[ppl] model={args.model_path} L={dims['n_layer']} D={dims['n_embd']} "
        f"ffn={dims['dim_ffn']} device={device} window={args.window_tokens}",
        flush=True,
    )

    total_nll = 0.0
    total_tokens = 0
    docs = 0
    windows = 0
    started = time.time()
    with torch.inference_mode():
        for doc_index, text in iter_texts(Path(args.data_path), args.max_docs):
            token_ids = tokenizer.encode(text)
            if len(token_ids) < 2:
                continue
            if args.max_tokens and total_tokens >= args.max_tokens:
                break
            docs += 1
            for start in range(0, len(token_ids) - 1, args.window_tokens):
                if args.max_tokens and total_tokens >= args.max_tokens:
                    break
                # Include one extra token as the final target. Consecutive
                # windows then share their boundary token as input, so every
                # document token after the first is scored exactly once.
                end = min(start + args.window_tokens + 1, len(token_ids))
                source = token_ids[start : end - 1]
                targets = token_ids[start + 1 : end]
                if args.max_tokens:
                    remaining = args.max_tokens - total_tokens
                    if len(targets) > remaining:
                        targets = targets[:remaining]
                        source = source[:remaining]
                if not targets:
                    continue

                actual_len = len(source)
                padded_len = (actual_len + 15) // 16 * 16
                source = source + [0] * (padded_len - actual_len)
                input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0)
                target_ids = torch.tensor(targets, dtype=torch.long, device=device)
                logits = model(input_ids)[0, :actual_len]
                token_nll = F.cross_entropy(
                    logits.float(), target_ids, reduction="sum"
                )
                total_nll += float(token_nll.item())
                total_tokens += len(targets)
                windows += 1

                if windows % args.progress_every == 0:
                    elapsed = time.time() - started
                    ppl = math.exp(total_nll / total_tokens)
                    print(
                        f"[ppl] docs={docs} windows={windows} tokens={total_tokens} "
                        f"ppl={ppl:.6f} tok/s={total_tokens / max(elapsed, 1e-6):.1f}",
                        flush=True,
                    )
            if args.max_tokens and total_tokens >= args.max_tokens:
                break

    if total_tokens == 0:
        raise RuntimeError("No scoreable tokens found in the input JSONL")
    mean_nll = total_nll / total_tokens
    return {
        "model": str(args.model_path),
        "data": str(args.data_path),
        "window_tokens": args.window_tokens,
        "documents": docs,
        "windows": windows,
        "tokens": total_tokens,
        "nll": total_nll,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "elapsed_seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    result = evaluate(args)
    print("[ppl] result=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
