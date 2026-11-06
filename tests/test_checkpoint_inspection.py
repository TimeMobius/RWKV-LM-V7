from __future__ import annotations

import torch

from scripts.check_rwkv_checkpoint import infer_dims, validate_expected


def _write_checkpoint(path):
    state = {
        "emb.weight": torch.zeros(65536, 2560, dtype=torch.bfloat16),
        "head.weight": torch.zeros(65536, 2560, dtype=torch.bfloat16),
        "blocks.0.ln1.weight": torch.zeros(2560, dtype=torch.bfloat16),
        "blocks.1.ln1.weight": torch.zeros(2560, dtype=torch.bfloat16),
        "blocks.0.ffn.key.weight": torch.zeros(10240, 2560, dtype=torch.bfloat16),
        "blocks.0.att.r_k": torch.zeros(40, 64, dtype=torch.bfloat16),
        "blocks.0.att.w1": torch.zeros(2560, 96, dtype=torch.bfloat16),
        "blocks.0.att.a1": torch.zeros(2560, 96, dtype=torch.bfloat16),
        "blocks.0.att.v1": torch.zeros(2560, 64, dtype=torch.bfloat16),
        "blocks.0.att.g1": torch.zeros(2560, 320, dtype=torch.bfloat16),
    }
    torch.save(state, path)


def test_infer_g1j_dimensions(tmp_path):
    checkpoint = tmp_path / "rwkv7-g1j-2.9b-ctx16384.pth"
    _write_checkpoint(checkpoint)

    dims = infer_dims(checkpoint)

    assert dims["n_layer"] == 2
    assert dims["n_embd"] == 2560
    assert dims["dim_ffn"] == 10240
    assert dims["vocab_size"] == 65536
    assert dims["head_size"] == 64
    assert dims["n_head"] == 40
    assert dims["d_decay_lora"] == 96
    assert dims["d_aaa_lora"] == 96
    assert dims["d_mv_lora"] == 64
    assert dims["d_gate_lora"] == 320
    assert dims["context_hint"] == 16384


def test_validate_expected_reports_shape_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pth"
    _write_checkpoint(checkpoint)
    dims = infer_dims(checkpoint)

    validate_expected(dims, ["n_embd=2560", "d_gate_lora=320"])

    try:
        validate_expected(dims, ["n_embd=4096"])
    except ValueError as exc:
        assert "n_embd" in str(exc)
    else:
        raise AssertionError("shape mismatch should fail validation")
