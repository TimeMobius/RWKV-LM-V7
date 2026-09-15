#!/usr/bin/env python3
"""Run prompt inference against a converted RWKV single-file checkpoint."""

from __future__ import annotations

import argparse
import gc
import math
import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import rwkv_v7_demo_runtime as runtime
from src.sft_prompt import resolve_prompt_from_args

DTYPE_MAP = {
    "auto": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to a converted single-file RWKV checkpoint such as rwkv-step-20.bf16.pth")
    parser.add_argument("--vocab-path", default=str(REPO_ROOT / "data" / "tokenizer" / "rwkv_vocab_v20230424.txt"), help="Tokenizer vocab path")
    parser.add_argument("--prompt", default="你好，请用一句话介绍 RWKV。", help="User prompt. By default it is rendered through --chat-template before inference.")
    parser.add_argument("--chat-template", default=str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"), help="Chat template used to render --prompt before inference")
    parser.add_argument("--raw-prompt", action="store_true", help="Use --prompt directly without chat-template rendering")
    parser.add_argument("--system-prompt", default="", help="Optional system message content used when rendering --prompt through the chat template")
    parser.add_argument("--current-date", default="", help="Optional current_date field injected into the rendered system message")
    parser.add_argument("--current-location", default="", help="Optional current_location field injected into the rendered system message")
    parser.add_argument("--add-generation-prompt", action=argparse.BooleanOptionalAction, default=True, help="When rendering chat messages, append the assistant generation prompt")
    parser.add_argument("--enable-thinking", action="store_true", help="When rendering chat messages, open a <think> block in the assistant generation prompt")
    parser.add_argument("--force-thinking", action="store_true", help="Alias of --enable-thinking; force the prompt to end with an open <think> block")
    parser.add_argument("--no-add-thinking", action="store_true", help="When rendering chat messages, do not add an empty <think> block")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="auto", help="Runtime dtype. 'auto' follows the checkpoint dtype.")
    parser.add_argument("--topk", type=int, default=10, help="How many next-token candidates to print")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Generate this many new tokens after the prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --sample is enabled")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p nucleus sampling cutoff when --sample is enabled. 0 disables it.")
    parser.add_argument("--sample", action="store_true", help="Sample tokens instead of greedy argmax when generating")
    parser.add_argument("--no-state-cache", action="store_true", help="Disable recurrent state caching and recompute the full sequence at every generation step")
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> str:
    return resolve_prompt_from_args(args)


def torch_load_state_dict(path: Path) -> "OrderedDict[str, torch.Tensor]":
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a dict-like state_dict in {path}, got {type(state)!r}")
    return OrderedDict((str(name), tensor.detach().cpu()) for name, tensor in state.items())


def infer_runtime_dtype(state_dict: "OrderedDict[str, torch.Tensor]", requested: str) -> torch.dtype:
    if requested != "auto":
        dtype = DTYPE_MAP[requested]
        assert dtype is not None
        return dtype

    for tensor in state_dict.values():
        if torch.is_tensor(tensor) and tensor.is_floating_point():
            return tensor.dtype
    return torch.float32


def infer_model_dims_from_state_dict(state_dict: "OrderedDict[str, torch.Tensor]") -> dict[str, int]:
    n_layer = len({int(name.split(".")[1]) for name in state_dict if name.startswith("blocks.") and ".ln1.weight" in name})
    n_embd = int(state_dict["emb.weight"].shape[1])
    vocab_size = int(state_dict["emb.weight"].shape[0])
    d_decay_lora = int(state_dict["blocks.0.att.w1"].shape[1])
    d_aaa_lora = int(state_dict["blocks.0.att.a1"].shape[1])
    d_mv_lora = int(state_dict["blocks.0.att.v1"].shape[1])
    d_gate_lora = int(state_dict["blocks.0.att.g1"].shape[1])
    head_size = int(state_dict["blocks.0.att.r_k"].shape[1])
    n_head = int(state_dict["blocks.0.att.r_k"].shape[0])
    return {
        "n_layer": n_layer,
        "n_embd": n_embd,
        "vocab_size": vocab_size,
        "d_decay_lora": d_decay_lora,
        "d_aaa_lora": d_aaa_lora,
        "d_mv_lora": d_mv_lora,
        "d_gate_lora": d_gate_lora,
        "head_size": head_size,
        "n_head": n_head,
    }


def build_model(state_dict: "OrderedDict[str, torch.Tensor]", device: str, dtype: torch.dtype) -> tuple[torch.nn.Module, dict[str, int]]:
    dims = infer_model_dims_from_state_dict(state_dict)
    runtime.configure_runtime(
        dtype=dtype,
        head_size=dims["head_size"],
        d_decay_lora=dims["d_decay_lora"],
        d_aaa_lora=dims["d_aaa_lora"],
        d_mv_lora=dims["d_mv_lora"],
        d_gate_lora=dims["d_gate_lora"],
    )

    args = types.SimpleNamespace()
    args.n_layer = dims["n_layer"]
    args.n_embd = dims["n_embd"]
    args.vocab_size = dims["vocab_size"]
    args.head_size_a = dims["head_size"]

    model = runtime.RWKV(args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint/model mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, dims


def build_tokenizer(vocab_path: Path) -> runtime.RWKV_TOKENIZER:
    return runtime.RWKV_TOKENIZER(str(vocab_path))


def safe_decode_token(tokenizer: runtime.RWKV_TOKENIZER, token_id: int) -> str:
    token_bytes = tokenizer.idx2token[token_id]
    try:
        return token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return repr(token_bytes)


def safe_decode_sequence(tokenizer: runtime.RWKV_TOKENIZER, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids)
    except UnicodeDecodeError:
        return tokenizer.decodeBytes(token_ids).decode("utf-8", errors="replace")


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p <= 0.0 or top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove_mask = cumulative > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_indices, sorted_probs.masked_fill(remove_mask, 0.0))
    norm = filtered.sum()
    if norm.item() <= 0:
        return probs
    return filtered / norm


def sample_token(logits: torch.Tensor, *, do_sample: bool, temperature: float, top_p: float) -> int:
    if not do_sample:
        return int(torch.argmax(logits).item())

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when --sample is enabled")

    probs = F.softmax(logits.float() / temperature, dim=-1)
    probs = apply_top_p(probs, top_p)
    return int(torch.multinomial(probs, num_samples=1).item())


def print_next_token_topk(tokenizer: runtime.RWKV_TOKENIZER, logits: torch.Tensor, topk: int) -> None:
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = torch.topk(probs, min(topk, probs.shape[-1]))
    print("\nTop next-token candidates:")
    for rank, (value, token_id) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        token_text = safe_decode_token(tokenizer, token_id).replace("\n", "\\n")
        print(f"{rank:2d}. id={token_id:<6} prob={value:.4%} token={token_text!r}")


def generate(
    model: torch.nn.Module,
    tokenizer: runtime.RWKV_TOKENIZER,
    prompt_tokens: list[int],
    *,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    topk: int,
    use_state_cache: bool = True,
) -> dict[str, Any]:
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")

    tokens = list(prompt_tokens)
    first_logits = None

    with torch.no_grad():
        if use_state_cache:
            # Prefill the prompt once, then feed only the newly sampled token at
            # each subsequent step. This avoids recomputing the entire history.
            state = model.init_state(batch_size=1, device=device)
            input_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).reshape(1, -1)
            logits, state = model(input_tensor, state=state, return_state=True)
            logits = logits[0, -1]
            first_logits = logits.detach().cpu()

            for step in range(max_new_tokens):
                next_token = sample_token(logits, do_sample=do_sample, temperature=temperature, top_p=top_p)
                tokens.append(next_token)
                if step + 1 < max_new_tokens:
                    input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    logits, state = model(input_tensor, state=state, return_state=True)
                    logits = logits[0, -1]
        else:
            for _ in range(max(1, max_new_tokens)):
                input_tensor = torch.tensor(tokens, dtype=torch.long, device=device).reshape(1, -1)
                logits = model(input_tensor)[0, -1]
                if first_logits is None:
                    first_logits = logits.detach().cpu()

                if max_new_tokens <= 0:
                    break

                next_token = sample_token(logits, do_sample=do_sample, temperature=temperature, top_p=top_p)
                tokens.append(next_token)

    assert first_logits is not None
    print_next_token_topk(tokenizer, first_logits, topk)

    generated_tokens = tokens[len(prompt_tokens) :]
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "full_tokens": tokens,
        "generated_text": safe_decode_sequence(tokenizer, generated_tokens) if generated_tokens else "",
        "full_text": safe_decode_sequence(tokenizer, tokens),
    }


def main() -> int:
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    vocab_path = Path(args.vocab_path).expanduser().resolve()
    if not model_path.is_file():
        print(f"[demo] model file not found: {model_path}", file=sys.stderr)
        return 2
    if not vocab_path.is_file():
        print(f"[demo] vocab file not found: {vocab_path}", file=sys.stderr)
        return 2
    try:
        prompt = resolve_prompt(args)
    except Exception as exc:
        print(f"[demo] failed to build prompt: {exc}", file=sys.stderr)
        return 2

    print(f"[demo] model:  {model_path}")
    print(f"[demo] vocab:  {vocab_path}")
    print(f"[demo] device: {args.device}")

    state_dict = torch_load_state_dict(model_path)
    dims = infer_model_dims_from_state_dict(state_dict)
    dtype = infer_runtime_dtype(state_dict, args.dtype)

    print(
        "[demo] dims:   "
        f"L={dims['n_layer']} D={dims['n_embd']} vocab={dims['vocab_size']} "
        f"head={dims['n_head']}x{dims['head_size']} "
        f"lora=({dims['d_decay_lora']},{dims['d_aaa_lora']},{dims['d_mv_lora']},{dims['d_gate_lora']})"
    )
    print(f"[demo] dtype:  {dtype}")

    tokenizer = build_tokenizer(vocab_path)
    model, _ = build_model(state_dict, args.device, dtype)

    del state_dict
    gc.collect()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt_tokens = tokenizer.encode(prompt)
    print(f"\nPrompt:\n{prompt}")
    print(f"\nPrompt tokens ({len(prompt_tokens)}):\n{prompt_tokens}")

    result = generate(
        model,
        tokenizer,
        prompt_tokens,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
        topk=args.topk,
        use_state_cache=not args.no_state_cache,
    )

    if args.max_new_tokens > 0:
        print(f"\nGenerated token ids:\n{result['generated_tokens']}")
        print(f"\nGenerated text:\n{result['generated_text']}")
        print(f"\nFull text:\n{result['full_text']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
