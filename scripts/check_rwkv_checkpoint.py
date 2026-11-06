#!/usr/bin/env python3
"""Inspect and optionally validate an RWKV state-dict checkpoint.

The training entry point receives model dimensions as command-line arguments,
so a stale launcher can otherwise construct a different model and fail only
when ``load_state_dict`` runs.  This checker reads tensor metadata on the meta
device and validates the dimensions without materializing the 5+ GB weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch


BLOCK_RE = re.compile(r"^blocks\.(\d+)\.ln1\.weight$")
CONTEXT_RE = re.compile(r"(?:ctx|context)[_-]?(\d+)", re.IGNORECASE)
EXPECTED_FIELDS = (
    "n_layer",
    "n_embd",
    "dim_ffn",
    "vocab_size",
    "head_size",
    "n_head",
    "d_decay_lora",
    "d_aaa_lora",
    "d_mv_lora",
    "d_gate_lora",
)


def _load_metadata(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="meta", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        try:
            state = torch.load(path, map_location="meta", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="meta")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state)!r}.")
    return state


def _shape(state: dict[str, torch.Tensor], key: str) -> tuple[int, ...]:
    if key not in state:
        raise ValueError(f"Checkpoint is missing required parameter {key!r}.")
    value = state[key]
    if not hasattr(value, "shape"):
        raise TypeError(f"Checkpoint parameter {key!r} has no tensor shape.")
    return tuple(int(dim) for dim in value.shape)


def infer_dims(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    state = _load_metadata(checkpoint)

    layer_ids = []
    for name in state:
        match = BLOCK_RE.match(str(name))
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        raise ValueError("Could not infer n_layer: no blocks.*.ln1.weight parameters found.")

    emb = _shape(state, "emb.weight")
    head = _shape(state, "head.weight")
    ffn_key = _shape(state, "blocks.0.ffn.key.weight")
    r_k = _shape(state, "blocks.0.att.r_k")
    if len(emb) != 2 or len(head) != 2 or len(ffn_key) != 2 or len(r_k) != 2:
        raise ValueError("Checkpoint has an unexpected parameter rank in its model head.")

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "n_layer": max(layer_ids) + 1,
        "n_embd": emb[1],
        "dim_ffn": ffn_key[0],
        "vocab_size": emb[0],
        "head_size": r_k[1],
        "n_head": r_k[0],
        "d_decay_lora": _shape(state, "blocks.0.att.w1")[1],
        "d_aaa_lora": _shape(state, "blocks.0.att.a1")[1],
        "d_mv_lora": _shape(state, "blocks.0.att.v1")[1],
        "d_gate_lora": _shape(state, "blocks.0.att.g1")[1],
        "parameter_count": sum(int(value.numel()) for value in state.values()),
        "dtype": str(next(value.dtype for value in state.values() if torch.is_tensor(value))),
        "context_hint": None,
        "state_dict_keys": len(state),
    }
    context_match = CONTEXT_RE.search(checkpoint.name)
    if context_match:
        result["context_hint"] = int(context_match.group(1))

    if head != (result["vocab_size"], result["n_embd"]):
        raise ValueError(f"head.weight shape {head} does not match emb.weight shape {emb}.")
    if result["n_embd"] % result["head_size"] != 0:
        raise ValueError("n_embd must be divisible by head_size.")
    if result["n_head"] != result["n_embd"] // result["head_size"]:
        raise ValueError("r_k head count does not match n_embd / head_size.")
    return result


def validate_expected(dims: dict[str, Any], values: list[str]) -> None:
    failures = []
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got {raw!r}.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if key not in EXPECTED_FIELDS:
            raise ValueError(f"Unsupported expectation {key!r}; use {', '.join(EXPECTED_FIELDS)}.")
        try:
            expected = int(value)
        except ValueError as exc:
            raise ValueError(f"Expected integer value for {key!r}, got {value!r}.") from exc
        if dims[key] != expected:
            failures.append(f"{key}: checkpoint={dims[key]} expected={expected}")
    if failures:
        raise ValueError("Checkpoint shape validation failed: " + "; ".join(failures))


def _format_summary(dims: dict[str, Any]) -> str:
    lines = ["RWKV checkpoint architecture summary"]
    for key in EXPECTED_FIELDS + ("parameter_count", "dtype", "context_hint", "state_dict_keys", "checkpoint"):
        lines.append(f"{key}: {dims[key]}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Path to a single-file RWKV .pth checkpoint.")
    parser.add_argument("--expect", action="append", default=[], metavar="KEY=VALUE", help="Validate one inferred dimension.")
    parser.add_argument("--summary-file", default="", help="Write a compact text architecture summary.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of key/value lines.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dims = infer_dims(args.checkpoint)
        validate_expected(dims, args.expect)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"[checkpoint-check] ERROR: {exc}", file=sys.stderr)
        return 2

    if args.summary_file:
        summary_path = Path(args.summary_file).expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_format_summary(dims), encoding="utf-8")
    if args.json:
        print(json.dumps(dims, ensure_ascii=True, sort_keys=True))
    else:
        for key, value in dims.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
