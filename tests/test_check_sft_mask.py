import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.sft_binidx import EncodedDocument, write_documents


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_sft_mask", ROOT / "scripts" / "check_sft_mask.py")
assert SPEC is not None and SPEC.loader is not None
check_sft_mask = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_sft_mask)


def test_validate_mask_requires_exact_value_and_matches_token_sizes(tmp_path: Path):
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [
            EncodedDocument(input_ids=[1, 2, 3], loss_mask=[1, 1, 1]),
            EncodedDocument(input_ids=[4, 5], loss_mask=[1, 1]),
        ],
    )

    result = check_sft_mask.validate_mask(
        f"{prefix}.mask",
        required_value=1,
        token_prefix=prefix,
        chunk_elements=2,
    )

    assert result["documents"] == 2
    assert result["elements"] == 5
    assert result["unique"] == [1]
    assert result["token_documents"] == 2


def test_validate_mask_rejects_non_matching_values(tmp_path: Path):
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2, 3], loss_mask=[0, 1, 1])],
    )

    with pytest.raises(ValueError, match="values other than 1"):
        check_sft_mask.validate_mask(f"{prefix}.mask", required_value=1)


def test_validate_mask_rejects_token_size_mismatch(tmp_path: Path):
    prefix = str(tmp_path / "sft")
    write_documents(
        prefix,
        [EncodedDocument(input_ids=[1, 2, 3], loss_mask=[1, 1, 1])],
    )
    mask = np.memmap(f"{prefix}.mask.bin", dtype=np.uint8, mode="r+")
    mask.flush()
    del mask

    other_prefix = str(tmp_path / "other")
    write_documents(
        other_prefix,
        [EncodedDocument(input_ids=[1, 2], loss_mask=[1, 1])],
    )

    with pytest.raises(ValueError, match="document sizes differ"):
        check_sft_mask.validate_mask(f"{prefix}.mask", token_prefix=other_prefix)
