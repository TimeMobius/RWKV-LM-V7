#!/usr/bin/env python3
"""Diagnose RWKV masked SFT loss on the exact documents used by a train step."""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
from scripts.eval_rwkv_ppl import load_model
from src.binidx import MMapIndexedDataset

SPECIAL_START = 65530
SPECIAL_NAMES = {
    65530: "<|im_start|>",
    65531: "<|im_end|>",
    65532: "<|endoftext|>",
    65533: "<think>",
    65534: "<tool_call>",
    65535: "<unused:65535>",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-prefix", required=True)
    parser.add_argument("--mask-prefix", default="")
    parser.add_argument("--doc-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ctx-len", type=int, default=86016)
    parser.add_argument(
        "--sequence-start",
        type=int,
        default=0,
        help="Start evaluation at this absolute token position with a fresh RWKV state.",
    )
    parser.add_argument(
        "--forward-window",
        type=int,
        default=0,
        help="Reset RWKV state at this interval; 0 evaluates one continuous ctx-len sequence.",
    )
    parser.add_argument("--position-bin", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-chunk", type=int, default=512)
    parser.add_argument("--top-losses", type=int, default=40)
    parser.add_argument(
        "--drop-system-prompt",
        action="store_true",
        help="Remove the first chat message, ending immediately before the second <|im_start|>.",
    )
    parser.add_argument("--pad-token-id", type=int, default=65532)
    parser.add_argument(
        "--vocab-path",
        default=str(REPO_ROOT / "rwkv_vocab_v20260603.txt"),
    )
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def token_text(tokenizer: TRIE_TOKENIZER, token_id: int) -> str:
    if token_id in SPECIAL_NAMES:
        return SPECIAL_NAMES[token_id]
    raw = tokenizer.idx2token.get(token_id, b"")
    return raw.decode("utf-8", errors="replace")


def decode_context(tokenizer: TRIE_TOKENIZER, token_ids: torch.Tensor, position: int) -> str:
    start = max(0, position - 12)
    ids = token_ids[start : position + 2].tolist()
    raw = b"".join(tokenizer.idx2token.get(int(token), b"") for token in ids)
    return raw.decode("utf-8", errors="replace")[-240:]


def add_group(group_stats: dict[str, list[float]], name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    group_stats[name][0] += float(values.sum().item())
    group_stats[name][1] += int(values.numel())


def finalize_groups(group_stats: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, (loss_sum, count) in sorted(group_stats.items()):
        result[name] = {
            "tokens": int(count),
            "mean_nll": loss_sum / count,
            "ppl": math.exp(loss_sum / count),
        }
    return result


def main() -> int:
    args = parse_args()
    if args.ctx_len <= 0 or args.ctx_len % 16:
        raise ValueError("--ctx-len must be a positive multiple of 16")
    if args.head_chunk <= 0:
        raise ValueError("--head-chunk must be positive")
    if args.forward_window < 0 or (args.forward_window and args.forward_window % 16):
        raise ValueError("--forward-window must be 0 or a positive multiple of 16")
    if args.position_bin <= 0:
        raise ValueError("--position-bin must be positive")
    if args.sequence_start < 0 or args.sequence_start >= args.ctx_len or args.sequence_start % 16:
        raise ValueError("--sequence-start must be a multiple of 16 in [0, ctx-len)")

    doc_ids = [int(value) for value in args.doc_ids.split(",") if value.strip()]
    data = MMapIndexedDataset(args.data_prefix)
    mask = MMapIndexedDataset(args.mask_prefix or args.data_prefix + ".mask")
    tokenizer = TRIE_TOKENIZER(args.vocab_path, special_first=True)
    device = torch.device(args.device)
    model, dims = load_model(Path(args.model_path), device)
    print(
        f"[sft-loss] model={args.model_path} L={dims['n_layer']} D={dims['n_embd']} "
        f"docs={doc_ids} ctx={args.ctx_len} device={device}",
        flush=True,
    )

    total_full_nll = 0.0
    total_full_tokens = 0
    total_masked_nll = 0.0
    total_masked_tokens = 0
    total_no_special_nll = 0.0
    total_no_special_tokens = 0
    total_special_wins = 0
    total_margin = 0.0
    input_special_counts: Counter[int] = Counter()
    target_special_counts: Counter[int] = Counter()
    prediction_counts: Counter[int] = Counter()
    group_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    position_stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0, 0.0, 0])
    per_doc = []
    removed_system_tokens: dict[int, int] = {}
    top_heap: list[tuple[float, int, dict[str, object]]] = []
    heap_serial = 0
    started = time.time()

    with torch.inference_mode():
        for doc_id in doc_ids:
            raw_tokens = data[doc_id].astype("int64")
            raw_mask = mask[doc_id].astype("int64")
            required = args.ctx_len + 1
            if len(raw_tokens) < required:
                raise ValueError(f"document {doc_id} has {len(raw_tokens)} tokens, expected at least {required}")
            raw_tokens = raw_tokens[:required]
            raw_mask = raw_mask[:required]
            valid_sequence_tokens = required
            if args.drop_system_prompt:
                message_starts = np.flatnonzero(raw_tokens == SPECIAL_START)
                if len(message_starts) < 2:
                    raise ValueError(f"document {doc_id} has fewer than two <|im_start|> tokens")
                removed = int(message_starts[1])
                removed_system_tokens[doc_id] = removed
                raw_tokens = raw_tokens[removed:]
                raw_mask = raw_mask[removed:]
                valid_sequence_tokens = len(raw_tokens)
                padding = required - valid_sequence_tokens
                raw_tokens = np.concatenate(
                    [raw_tokens, np.full(padding, args.pad_token_id, dtype=raw_tokens.dtype)]
                )
                raw_mask = np.concatenate(
                    [raw_mask, np.zeros(padding, dtype=raw_mask.dtype)]
                )
            valid_target_end = valid_sequence_tokens - 1
            x_cpu = torch.from_numpy(raw_tokens[:-1].copy())
            y_cpu = torch.from_numpy(raw_tokens[1:].copy())
            mask_cpu = torch.from_numpy(raw_mask[1:].copy()).bool()
            if args.sequence_start:
                mask_cpu[: args.sequence_start] = False
            active_count = int(mask_cpu.sum().item())
            active_positions = torch.nonzero(mask_cpu, as_tuple=False).flatten()
            evaluated_x = x_cpu[args.sequence_start : valid_target_end]
            for token_id in evaluated_x[evaluated_x >= SPECIAL_START].tolist():
                input_special_counts[int(token_id)] += 1
            for token_id in y_cpu[mask_cpu & (y_cpu >= SPECIAL_START)].tolist():
                target_special_counts[int(token_id)] += 1

            doc_full_nll = 0.0
            doc_masked_nll = 0.0
            doc_masked_tokens = 0
            doc_no_special_nll = 0.0
            doc_no_special_tokens = 0
            doc_special_wins = 0
            doc_margin = 0.0

            forward_window = args.forward_window or args.ctx_len
            for forward_start in range(args.sequence_start, args.ctx_len, forward_window):
                forward_end = min(forward_start + forward_window, args.ctx_len)
                input_ids = x_cpu[forward_start:forward_end].to(
                    device=device, dtype=torch.long
                ).unsqueeze(0)
                hidden = model._forward_features(input_ids)[0]
                del input_ids

                score_end = min(forward_end, valid_target_end)
                for start in range(forward_start, score_end, args.head_chunk):
                    bin_end = ((start // args.position_bin) + 1) * args.position_bin
                    end = min(start + args.head_chunk, score_end, bin_end)
                    logits = F.linear(
                        hidden[start - forward_start : end - forward_start], model.head.weight
                    ).float()
                    targets = y_cpu[start:end].to(device=device, dtype=torch.long)
                    losses = F.cross_entropy(logits, targets, reduction="none")
                    chunk_full_nll = float(losses.sum().item())
                    doc_full_nll += chunk_full_nll
                    position_bucket = (start // args.position_bin) * args.position_bin
                    position_stats[position_bucket][0] += chunk_full_nll
                    position_stats[position_bucket][1] += int(losses.numel())

                    local_mask_cpu = mask_cpu[start:end]
                    if local_mask_cpu.any():
                        local_indices_cpu = torch.nonzero(local_mask_cpu, as_tuple=False).flatten()
                        local_indices = local_indices_cpu.to(device=device)
                        active_logits = logits.index_select(0, local_indices)
                        active_targets = targets.index_select(0, local_indices)
                        active_losses = losses.index_select(0, local_indices)
                        active_predictions = active_logits.argmax(dim=-1)
                        special_max = active_logits[:, SPECIAL_START:].max(dim=-1).values
                        ordinary_max = active_logits[:, :SPECIAL_START].max(dim=-1).values
                        margins = special_max - ordinary_max

                        count = int(active_losses.numel())
                        loss_sum = float(active_losses.sum().item())
                        doc_masked_nll += loss_sum
                        doc_masked_tokens += count
                        doc_special_wins += int((margins > 0).sum().item())
                        doc_margin += float(margins.sum().item())
                        position_stats[position_bucket][2] += loss_sum
                        position_stats[position_bucket][3] += count
                        for prediction in active_predictions.tolist():
                            prediction_counts[int(prediction)] += 1

                        ordinary_target = active_targets < SPECIAL_START
                        if ordinary_target.any():
                            ordinary_logits = active_logits[ordinary_target].clone()
                            ordinary_targets = active_targets[ordinary_target]
                            ordinary_logits[:, SPECIAL_START:] = -torch.inf
                            no_special_losses = F.cross_entropy(
                                ordinary_logits, ordinary_targets, reduction="none"
                            )
                            doc_no_special_nll += float(no_special_losses.sum().item())
                            doc_no_special_tokens += int(no_special_losses.numel())

                        active_targets_cpu = active_targets.cpu()
                        active_losses_cpu = active_losses.cpu()
                        active_predictions_cpu = active_predictions.cpu()
                        for token_id in torch.unique(active_targets_cpu).tolist():
                            selector = active_targets_cpu == token_id
                            group = SPECIAL_NAMES.get(int(token_id), "ordinary")
                            add_group(group_stats, group, active_losses_cpu[selector])

                        keep = min(args.top_losses, count)
                        chunk_values, chunk_offsets = torch.topk(active_losses_cpu, keep)
                        for loss_value, active_offset in zip(chunk_values.tolist(), chunk_offsets.tolist()):
                            local_offset = int(local_indices_cpu[active_offset].item())
                            position = start + local_offset
                            target_id = int(y_cpu[position].item())
                            prediction_id = int(active_predictions_cpu[active_offset].item())
                            detail = {
                                "doc": doc_id,
                                "position": position,
                                "nll": float(loss_value),
                                "target_id": target_id,
                                "target": token_text(tokenizer, target_id),
                                "prediction_id": prediction_id,
                                "prediction": token_text(tokenizer, prediction_id),
                                "context": decode_context(tokenizer, x_cpu, position),
                            }
                            heap_serial += 1
                            entry = (float(loss_value), heap_serial, detail)
                            if len(top_heap) < args.top_losses:
                                heapq.heappush(top_heap, entry)
                            elif loss_value > top_heap[0][0]:
                                heapq.heapreplace(top_heap, entry)

                    del logits, targets, losses
                del hidden

            total_full_nll += doc_full_nll
            evaluated_tokens = valid_target_end - args.sequence_start
            total_full_tokens += evaluated_tokens
            total_masked_nll += doc_masked_nll
            total_masked_tokens += doc_masked_tokens
            total_no_special_nll += doc_no_special_nll
            total_no_special_tokens += doc_no_special_tokens
            total_special_wins += doc_special_wins
            total_margin += doc_margin
            doc_result = {
                "doc": doc_id,
                "removed_system_tokens": removed_system_tokens.get(doc_id, 0),
                "masked_position_min": int(active_positions.min().item()),
                "masked_position_max": int(active_positions.max().item()),
                "masked_tokens": doc_masked_tokens,
                "masked_mean_nll": doc_masked_nll / doc_masked_tokens,
                "masked_ppl": math.exp(doc_masked_nll / doc_masked_tokens),
                "all_mean_nll": doc_full_nll / evaluated_tokens,
                "all_ppl": math.exp(doc_full_nll / evaluated_tokens),
                "ordinary_masked_no_appended_mean_nll": (
                    doc_no_special_nll / doc_no_special_tokens if doc_no_special_tokens else None
                ),
                "special_output_top1_fraction": doc_special_wins / doc_masked_tokens,
                "mean_special_vs_ordinary_logit_margin": doc_margin / doc_masked_tokens,
            }
            per_doc.append(doc_result)
            print("[sft-loss] doc=" + json.dumps(doc_result, sort_keys=True), flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "model": args.model_path,
        "data_prefix": args.data_prefix,
        "doc_ids": doc_ids,
        "ctx_len": args.ctx_len,
        "drop_system_prompt": args.drop_system_prompt,
        "removed_system_tokens": {str(k): v for k, v in sorted(removed_system_tokens.items())},
        "sequence_start": args.sequence_start,
        "forward_window": args.forward_window or args.ctx_len,
        "all_tokens": total_full_tokens,
        "all_mean_nll": total_full_nll / total_full_tokens,
        "all_ppl": math.exp(total_full_nll / total_full_tokens),
        "masked_tokens": total_masked_tokens,
        "masked_mean_nll": total_masked_nll / total_masked_tokens,
        "masked_ppl": math.exp(total_masked_nll / total_masked_tokens),
        "ordinary_masked_tokens": total_no_special_tokens,
        "ordinary_masked_no_appended_mean_nll": total_no_special_nll / total_no_special_tokens,
        "ordinary_masked_no_appended_ppl": math.exp(total_no_special_nll / total_no_special_tokens),
        "special_output_top1_tokens": total_special_wins,
        "special_output_top1_fraction": total_special_wins / total_masked_tokens,
        "mean_special_vs_ordinary_logit_margin": total_margin / total_masked_tokens,
        "input_special_counts": {SPECIAL_NAMES.get(k, str(k)): v for k, v in sorted(input_special_counts.items())},
        "masked_target_special_counts": {
            SPECIAL_NAMES.get(k, str(k)): v for k, v in sorted(target_special_counts.items())
        },
        "masked_loss_by_target_group": finalize_groups(group_stats),
        "loss_by_position": {
            f"{start}:{min(start + args.position_bin, args.ctx_len)}": {
                "all_tokens": int(values[1]),
                "all_mean_nll": values[0] / values[1],
                "masked_tokens": int(values[3]),
                "masked_mean_nll": values[2] / values[3] if values[3] else None,
            }
            for start, values in sorted(position_stats.items())
        },
        "masked_top_prediction_counts": {
            f"{key}:{token_text(tokenizer, key)}": value
            for key, value in prediction_counts.most_common(30)
        },
        "per_doc": per_doc,
        "top_masked_losses": [entry[2] for entry in sorted(top_heap, reverse=True)],
        "elapsed_seconds": time.time() - started,
    }
    print("[sft-loss] result=" + json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
