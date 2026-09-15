import importlib.util
import sys
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_converted_rwkv_demo.py"
SPEC = importlib.util.spec_from_file_location("run_converted_rwkv_demo_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
demo_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo_script)


def test_infer_model_dims_from_state_dict():
    state_dict = OrderedDict(
        [
            ("emb.weight", torch.zeros(65536, 4096, dtype=torch.bfloat16)),
            ("blocks.0.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.1.ln1.weight", torch.zeros(4096, dtype=torch.bfloat16)),
            ("blocks.0.att.w1", torch.zeros(4096, 192, dtype=torch.bfloat16)),
            ("blocks.0.att.a1", torch.zeros(4096, 192, dtype=torch.bfloat16)),
            ("blocks.0.att.v1", torch.zeros(4096, 128, dtype=torch.bfloat16)),
            ("blocks.0.att.g1", torch.zeros(4096, 384, dtype=torch.bfloat16)),
            ("blocks.0.att.r_k", torch.zeros(64, 64, dtype=torch.bfloat16)),
        ]
    )

    dims = demo_script.infer_model_dims_from_state_dict(state_dict)

    assert dims == {
        "n_layer": 2,
        "n_embd": 4096,
        "vocab_size": 65536,
        "d_decay_lora": 192,
        "d_aaa_lora": 192,
        "d_mv_lora": 128,
        "d_gate_lora": 384,
        "head_size": 64,
        "n_head": 64,
    }


def test_infer_runtime_dtype_prefers_checkpoint_dtype_in_auto_mode():
    state_dict = OrderedDict([("emb.weight", torch.zeros(2, 2, dtype=torch.bfloat16))])

    assert demo_script.infer_runtime_dtype(state_dict, "auto") == torch.bfloat16
    assert demo_script.infer_runtime_dtype(state_dict, "fp32") == torch.float32


def test_resolve_prompt_renders_user_prompt_with_chat_template():
    args = Namespace(
        raw_prompt=False,
        prompt="你好",
        chat_template=str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja"),
        system_prompt="系统提示",
        current_date="2026-06-09",
        current_location="Shanghai",
        add_generation_prompt=True,
        enable_thinking=False,
        force_thinking=False,
        no_add_thinking=False,
    )

    prompt = demo_script.resolve_prompt(args)

    assert "<|im_start|>System: 系统提示" in prompt
    assert "Current date: 2026-06-09" in prompt
    assert "Current location: Shanghai" in prompt
    assert "<|im_start|>User: 你好<|im_end|>" in prompt
    assert prompt.endswith("<|im_start|>Assistant: <think>\n\n</think>\n\n")


def test_parse_args_uses_plain_prompt_not_message_files(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_converted_rwkv_demo.py",
            "--model-path",
            "model.pth",
            "--prompt",
            "你好",
            "--force-thinking",
        ],
    )

    args = demo_script.parse_args()

    assert args.prompt == "你好"
    assert args.force_thinking is True
    assert args.raw_prompt is False
    assert args.no_state_cache is False
    assert not hasattr(args, "messages_file")
    assert not hasattr(args, "messages_json")


def test_resolve_prompt_can_use_raw_prompt_and_rejects_thinking_conflict():
    args = Namespace(raw_prompt=True, prompt="raw text", enable_thinking=True, force_thinking=False, no_add_thinking=True)
    assert demo_script.resolve_prompt(args) == "raw text"

    args.raw_prompt = False
    args.chat_template = str(REPO_ROOT / "data" / "SFT" / "sample" / "chat_template.jinja")
    args.system_prompt = ""
    args.current_date = ""
    args.current_location = ""
    args.add_generation_prompt = True
    args.force_thinking = False
    with pytest.raises(ValueError, match="mutually exclusive"):
        demo_script.resolve_prompt(args)


def test_runtime_block_only_creates_ln0_for_first_layer():
    args = type("Args", (), {"n_embd": 128, "dim_att": 128, "dim_ffn": 512, "head_size_a": 64, "n_layer": 2})()

    block0 = demo_script.runtime.Block(args, 0)
    block1 = demo_script.runtime.Block(args, 1)

    assert hasattr(block0, "ln0")
    assert not hasattr(block1, "ln0")


@pytest.fixture(params=[torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def tiny_cpu_runtime_model(request):
    previous_runtime = (
        demo_script.runtime.DTYPE,
        demo_script.runtime.HEAD_SIZE,
        demo_script.runtime.D_DECAY_LORA,
        demo_script.runtime.D_AAA_LORA,
        demo_script.runtime.D_MV_LORA,
        demo_script.runtime.D_GATE_LORA,
    )
    dtype = request.param
    demo_script.runtime.configure_runtime(
        dtype=dtype,
        head_size=64,
        d_decay_lora=8,
        d_aaa_lora=8,
        d_mv_lora=4,
        d_gate_lora=8,
    )
    torch.manual_seed(1234)
    args = type("Args", (), {"n_embd": 64, "n_layer": 2, "vocab_size": 32, "head_size_a": 64})()
    model = demo_script.runtime.RWKV(args).to(dtype=dtype).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)
    try:
        yield model
    finally:
        demo_script.runtime.configure_runtime(
            dtype=previous_runtime[0],
            head_size=previous_runtime[1],
            d_decay_lora=previous_runtime[2],
            d_aaa_lora=previous_runtime[3],
            d_mv_lora=previous_runtime[4],
            d_gate_lora=previous_runtime[5],
        )


def test_runtime_stateful_prefill_and_chunked_decode_match_full_forward(tiny_cpu_runtime_model):
    model = tiny_cpu_runtime_model
    token_ids = torch.tensor([[1, 5, 7, 2, 8, 3]], dtype=torch.long)

    with torch.no_grad():
        full_logits = model(token_ids)

        state = model.init_state(batch_size=1, device="cpu")
        prefill_logits, state = model(token_ids, state=state, return_state=True)

        state = model.init_state(batch_size=1, device="cpu")
        first_logits, state = model(token_ids[:, :3], state=state, return_state=True)
        second_logits, state = model(token_ids[:, 3:], state=state, return_state=True)

    torch.testing.assert_close(prefill_logits, full_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.cat((first_logits, second_logits), dim=1), full_logits, rtol=1e-5, atol=1e-6)
    assert len(state) == 2
    assert state[0]["att"]["wkv"].dtype is torch.float32
    assert state[0]["att"]["x"].shape == (1, 64)
    assert state[0]["ffn"].shape == (1, 64)


class _CountingRWKV(demo_script.runtime.RWKV):
    def __init__(self, args):
        super().__init__(args)
        self.forward_lengths: list[int] = []

    def forward(self, idx, **kwargs):
        self.forward_lengths.append(idx.shape[1])
        return super().forward(idx, **kwargs)


class _TinyTokenizer:
    def __init__(self):
        self.idx2token = {index: bytes([ord("a") + index]) for index in range(32)}

    def decode(self, token_ids):
        return b"".join(self.idx2token[token_id] for token_id in token_ids).decode("ascii")

    def decodeBytes(self, token_ids):
        return b"".join(self.idx2token[token_id] for token_id in token_ids)


def test_generate_uses_recurrent_cache_without_changing_greedy_output(tiny_cpu_runtime_model, capsys):
    base_model = tiny_cpu_runtime_model
    args = type("Args", (), {"n_embd": 64, "n_layer": 2, "vocab_size": 32, "head_size_a": 64})()
    model = _CountingRWKV(args).to(dtype=base_model.emb.weight.dtype).eval()
    model.load_state_dict(base_model.state_dict())
    tokenizer = _TinyTokenizer()
    prompt_tokens = [1, 5, 7, 2]

    cached_result = demo_script.generate(
        model,
        tokenizer,
        prompt_tokens,
        device="cpu",
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=0.0,
        topk=1,
        use_state_cache=True,
    )
    cached_lengths = model.forward_lengths

    model.forward_lengths = []
    uncached_result = demo_script.generate(
        model,
        tokenizer,
        prompt_tokens,
        device="cpu",
        max_new_tokens=4,
        do_sample=False,
        temperature=1.0,
        top_p=0.0,
        topk=1,
        use_state_cache=False,
    )

    assert cached_result == uncached_result
    assert cached_lengths == [4, 1, 1, 1]
    assert model.forward_lengths == [4, 5, 6, 7]
    assert "Top next-token candidates:" in capsys.readouterr().out
