from __future__ import annotations

"""Demo-derived RWKV runtime used by checkpoint equivalence tests.

This module intentionally keeps only the tokenizer and model definitions needed
for forward-parity checks. It does not execute the original demo's top-level
inference flow or compile the optional CUDA extension at import time.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

DTYPE = torch.half
HEAD_SIZE = 64
D_DECAY_LORA = 64
D_AAA_LORA = 64
D_MV_LORA = 32
D_GATE_LORA = 128


def configure_runtime(
    *,
    dtype: torch.dtype | None = None,
    head_size: int | None = None,
    d_decay_lora: int | None = None,
    d_aaa_lora: int | None = None,
    d_mv_lora: int | None = None,
    d_gate_lora: int | None = None,
) -> None:
    global DTYPE, HEAD_SIZE, D_DECAY_LORA, D_AAA_LORA, D_MV_LORA, D_GATE_LORA
    if dtype is not None:
        DTYPE = dtype
    if head_size is not None:
        HEAD_SIZE = head_size
    if d_decay_lora is not None:
        D_DECAY_LORA = d_decay_lora
    if d_aaa_lora is not None:
        D_AAA_LORA = d_aaa_lora
    if d_mv_lora is not None:
        D_MV_LORA = d_mv_lora
    if d_gate_lora is not None:
        D_GATE_LORA = d_gate_lora


class RWKV_TOKENIZER:
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]

    def __init__(self, file_name: str):
        self.idx2token: dict[int, bytes] = {}
        sorted_tokens: list[bytes] = []
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for line in lines:
            idx = int(line[: line.index(" ")])
            token = eval(line[line.index(" ") : line.rindex(" ")])
            token = token.encode("utf-8") if isinstance(token, str) else token
            assert isinstance(token, bytes)
            assert len(token) == int(line[line.rindex(" ") :])
            sorted_tokens.append(token)
            self.idx2token[idx] = token

        self.token2idx = {token: int(idx) for idx, token in self.idx2token.items()}

        self.table = [[[] for _ in range(256)] for _ in range(256)]
        self.good = [set() for _ in range(256)]
        self.wlen = [0 for _ in range(256)]

        for token in reversed(sorted_tokens):
            if len(token) < 2:
                continue
            first = int(token[0])
            second = int(token[1])
            self.table[first][second].append(token)
            self.wlen[first] = max(self.wlen[first], len(token))
            self.good[first].add(second)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len = len(src)
        tokens: list[int] = []
        i = 0
        while i < src_len:
            token = src[i : i + 1]
            if i < src_len - 1:
                first = int(src[i])
                second = int(src[i + 1])
                if second in self.good[first]:
                    window = src[i : i + self.wlen[first]]
                    try:
                        token = next(filter(window.startswith, self.table[first][second]))
                    except StopIteration:
                        pass
            tokens.append(self.token2idx[token])
            i += len(token)
        return tokens

    def decodeBytes(self, tokens: list[int]) -> bytes:
        return b"".join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str) -> list[int]:
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return self.decodeBytes(tokens).decode("utf-8")


def RWKV7_OP(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state: torch.Tensor | None = None,
    return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run the recurrent WKV operation, optionally continuing from ``state``.

    The state is kept in float32 because the recurrence itself has always used
    float32 in this reference runtime.  This lets a full prompt prefill and
    later single-token decode calls produce the same result as one full forward.
    """
    batch, steps, channels = r.size()
    n_head = channels // HEAD_SIZE
    head_size = HEAD_SIZE
    r = r.view(batch, steps, n_head, head_size).float()
    k = k.view(batch, steps, n_head, head_size).float()
    v = v.view(batch, steps, n_head, head_size).float()
    a = a.view(batch, steps, n_head, head_size).float()
    b = b.view(batch, steps, n_head, head_size).float()
    w = torch.exp(-torch.exp(w.view(batch, steps, n_head, head_size).float()))
    out = torch.zeros((batch, steps, n_head, head_size), device=r.device, dtype=torch.float)
    if state is None:
        state = torch.zeros((batch, n_head, head_size, head_size), device=r.device, dtype=torch.float)
    else:
        expected_shape = (batch, n_head, head_size, head_size)
        if tuple(state.shape) != expected_shape:
            raise ValueError(f"Invalid WKV state shape: expected {expected_shape}, got {tuple(state.shape)}")
        state = state.to(device=r.device, dtype=torch.float)

    for t in range(steps):
        kk = k[:, t, :].view(batch, n_head, 1, head_size)
        rr = r[:, t, :].view(batch, n_head, head_size, 1)
        vv = v[:, t, :].view(batch, n_head, head_size, 1)
        aa = a[:, t, :].view(batch, n_head, head_size, 1)
        bb = b[:, t, :].view(batch, n_head, 1, head_size)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(batch, n_head, head_size)

    out = out.view(batch, steps, channels).to(dtype=DTYPE)
    if return_state:
        return out, state
    return out


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        n_head = self.n_head
        head_size = self.head_size
        channels = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1, 1, channels))
        self.x_w = nn.Parameter(torch.empty(1, 1, channels))
        self.x_k = nn.Parameter(torch.empty(1, 1, channels))
        self.x_v = nn.Parameter(torch.empty(1, 1, channels))
        self.x_a = nn.Parameter(torch.empty(1, 1, channels))
        self.x_g = nn.Parameter(torch.empty(1, 1, channels))

        self.w0 = nn.Parameter(torch.empty(1, 1, channels))
        self.w1 = nn.Parameter(torch.empty(channels, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, channels))

        self.a0 = nn.Parameter(torch.empty(1, 1, channels))
        self.a1 = nn.Parameter(torch.empty(channels, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, channels))

        self.v0 = nn.Parameter(torch.empty(1, 1, channels))
        self.v1 = nn.Parameter(torch.empty(channels, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, channels))

        self.g1 = nn.Parameter(torch.empty(channels, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, channels))

        self.k_k = nn.Parameter(torch.empty(1, 1, channels))
        self.k_a = nn.Parameter(torch.empty(1, 1, channels))
        self.r_k = nn.Parameter(torch.empty(n_head, head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(n_head, channels, eps=64e-5)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor,
        *,
        state: dict[str, torch.Tensor] | None = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, steps, channels = x.size()
        input_x = x
        n_head = self.n_head
        if state is None:
            previous_x = torch.zeros_like(x[:, :1, :])
            wkv_state = None
        else:
            previous_x = state["x"].to(device=x.device, dtype=x.dtype).unsqueeze(1)
            wkv_state = state["wkv"]
        xx = torch.cat((previous_x, x[:, :-1, :]), dim=1) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(batch, steps, n_head, -1), dim=-1, p=2.0).view(batch, steps, channels)
        k = k * (1 + (a - 1) * self.k_a)

        if return_state:
            x, final_wkv_state = RWKV7_OP(r, w, k, v, -kk, kk * a, state=wkv_state, return_state=True)
        else:
            x = RWKV7_OP(r, w, k, v, -kk, kk * a, state=wkv_state)
        x = self.ln_x(x.view(batch * steps, channels)).view(batch, steps, channels)
        x = x + (
            (r.view(batch, steps, n_head, -1) * k.view(batch, steps, n_head, -1) * self.r_k).sum(dim=-1, keepdim=True)
            * v.view(batch, steps, n_head, -1)
        ).view(batch, steps, channels)
        x = self.output(x * g)
        if return_state:
            return x, v_first, {"x": input_x[:, -1, :], "wkv": final_wkv_state}
        return x, v_first


class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_x = x
        previous_x = torch.zeros_like(x[:, :1, :]) if state is None else state.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        xx = torch.cat((previous_x, x[:, :-1, :]), dim=1) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        x = self.value(k)
        if return_state:
            return x, input_x[:, -1, :]
        return x


class Block(nn.Module):
    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor,
        *,
        state: dict[str, object] | None = None,
        return_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if self.layer_id == 0:
            x = self.ln0(x)

        if return_state:
            xx, v_first, att_state = self.att(self.ln1(x), v_first, state=None if state is None else state["att"], return_state=True)
        else:
            xx, v_first = self.att(self.ln1(x), v_first, state=None if state is None else state["att"])
        x = x + xx
        ffn_input = self.ln2(x)
        if return_state:
            xx, ffn_state = self.ffn(ffn_input, state=None if state is None else state["ffn"], return_state=True)
        else:
            xx = self.ffn(ffn_input, state=None if state is None else state["ffn"])
        x = x + xx
        if return_state:
            return x, v_first, {"att": att_state, "ffn": ffn_state}
        return x, v_first


class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def init_state(self, batch_size: int, *, device: torch.device | str | None = None) -> list[dict[str, object]]:
        """Create the recurrent cache used by stateful prefill and decoding."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        parameter = next(self.parameters())
        device = parameter.device if device is None else device
        dtype = parameter.dtype
        state: list[dict[str, object]] = []
        for block in self.blocks:
            state.append(
                {
                    "att": {
                        "x": torch.zeros((batch_size, self.emb.embedding_dim), device=device, dtype=dtype),
                        "wkv": torch.zeros(
                            (batch_size, block.att.n_head, block.att.head_size, block.att.head_size),
                            device=device,
                            dtype=torch.float32,
                        ),
                    },
                    "ffn": torch.zeros((batch_size, self.emb.embedding_dim), device=device, dtype=dtype),
                }
            )
        return state

    def forward(
        self,
        idx: torch.Tensor,
        *,
        state: list[dict[str, object]] | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, object]]]:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError(f"idx must have shape [batch, non-empty steps], got {tuple(idx.shape)}")
        use_state = state is not None or return_state
        if not use_state:
            x = self.emb(idx)
            v_first = torch.empty_like(x)
            for block in self.blocks:
                x, v_first = block(x, v_first)
            x = self.ln_out(x)
            return self.head(x)

        if state is None:
            state = self.init_state(idx.shape[0], device=idx.device)
        if len(state) != len(self.blocks):
            raise ValueError(f"Invalid layer-state count: expected {len(self.blocks)}, got {len(state)}")

        x = self.emb(idx)
        v_first = torch.empty_like(x)
        final_state: list[dict[str, object]] = []
        for block, block_state in zip(self.blocks, state):
            x, v_first, block_state = block(x, v_first, state=block_state, return_state=True)
            final_state.append(block_state)
        x = self.ln_out(x)
        return self.head(x), final_state
