#!/usr/bin/env python3
"""Validate an SFT mask sidecar and optionally require one exact value."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.binidx import MMapIndexedDataset, data_file_path, index_file_path  # noqa: E402


def resolve_mask_prefix(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    text = str(raw)
    if text.endswith(".mask.bin"):
        return Path(text[: -len(".bin")])
    if text.endswith(".mask.idx"):
        return Path(text[: -len(".idx")])
    if text.endswith(".mask"):
        return raw
    return Path(text + ".mask")


def validate_mask(
    mask_path: str | Path,
    *,
    required_value: int | None = None,
    token_prefix: str | Path | None = None,
    chunk_elements: int = 16 * 1024 * 1024,
) -> dict[str, object]:
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")

    mask_prefix = resolve_mask_prefix(mask_path)
    mask_bin = Path(data_file_path(str(mask_prefix)))
    mask_idx = Path(index_file_path(str(mask_prefix)))
    if not mask_bin.is_file():
        raise FileNotFoundError(f"missing mask bin file: {mask_bin}")
    if not mask_idx.is_file():
        raise FileNotFoundError(f"missing mask idx file: {mask_idx}")

    index = MMapIndexedDataset.Index(str(mask_idx))
    dtype = np.dtype(index.dtype)
    expected_elements = int(np.asarray(index.sizes, dtype=np.int64).sum())
    expected_bytes = expected_elements * dtype.itemsize
    actual_bytes = mask_bin.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"mask bin size mismatch: {mask_bin} has {actual_bytes} bytes, "
            f"but idx expects {expected_bytes} bytes"
        )

    values = np.memmap(mask_bin, dtype=dtype, mode="r", shape=(expected_elements,))
    unique_values: set[int | float] = set()
    min_value = None
    max_value = None
    invalid_count = 0
    first_invalid = None
    try:
        for start in range(0, expected_elements, chunk_elements):
            stop = min(start + chunk_elements, expected_elements)
            chunk = np.asarray(values[start:stop])
            if chunk.size:
                chunk_min = chunk.min().item()
                chunk_max = chunk.max().item()
                min_value = chunk_min if min_value is None else min(min_value, chunk_min)
                max_value = chunk_max if max_value is None else max(max_value, chunk_max)
                unique_values.update(np.unique(chunk).tolist())
                if required_value is not None:
                    bad = np.flatnonzero(chunk != required_value)
                    if bad.size:
                        invalid_count += int(bad.size)
                        if first_invalid is None:
                            first_invalid = start + int(bad[0])
    finally:
        del values

    token_document_count = None
    if token_prefix is not None:
        token_prefix = Path(token_prefix).expanduser()
        token_idx = Path(index_file_path(str(token_prefix)))
        if not token_idx.is_file():
            raise FileNotFoundError(f"missing token idx file: {token_idx}")
        token_index = MMapIndexedDataset.Index(str(token_idx))
        token_document_count = len(token_index)
        if token_document_count != len(index):
            raise ValueError(
                f"token/mask document count mismatch: {token_document_count} != {len(index)}"
            )
        if not np.array_equal(token_index.sizes, index.sizes):
            raise ValueError("token/mask document sizes differ")

    if required_value is not None and invalid_count:
        raise ValueError(
            f"mask contains {invalid_count} values other than {required_value}; "
            f"first invalid flat element is {first_invalid}"
        )

    return {
        "mask_prefix": str(mask_prefix),
        "documents": int(len(index)),
        "elements": expected_elements,
        "dtype": str(dtype),
        "min": min_value,
        "max": max_value,
        "unique": sorted(unique_values),
        "required_value": required_value,
        "token_documents": token_document_count,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_path", help="dataset prefix, .mask prefix, or .mask.bin/.mask.idx path")
    parser.add_argument(
        "--require-value",
        type=int,
        default=None,
        help="fail if any mask element differs from this exact integer value",
    )
    parser.add_argument(
        "--token-prefix",
        default=None,
        help="also verify token/mask document counts and sizes against this token prefix",
    )
    parser.add_argument("--chunk-elements", type=int, default=16 * 1024 * 1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_mask(
            args.mask_path,
            required_value=args.require_value,
            token_prefix=args.token_prefix,
            chunk_elements=args.chunk_elements,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[sft-mask-check] ERROR: {exc}", file=sys.stderr)
        return 2

    for key, value in result.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
