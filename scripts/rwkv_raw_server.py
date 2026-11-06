#!/usr/bin/env python3
"""Small raw-prompt HTTP service for a single RWKV .pth checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_converted_rwkv_demo import (  # noqa: E402
    build_model,
    build_tokenizer,
    infer_runtime_dtype,
    torch_load_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--vocab-path",
        default=str(REPO_ROOT / "data" / "tokenizer" / "rwkv_vocab_v20260603.txt"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8031)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--default-max-new-tokens", type=int, default=128)
    return parser.parse_args()


def sample_token(logits: torch.Tensor, *, sample: bool, temperature: float, top_p: float) -> int:
    if not sample:
        return int(torch.argmax(logits).item())
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero when sample=true")
    probs = F.softmax(logits.float() / temperature, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices, sorted_probs.masked_fill(remove, 0.0))
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


class RawRWKVService:
    def __init__(self, args: argparse.Namespace):
        model_path = Path(args.model_path).expanduser().resolve()
        vocab_path = Path(args.vocab_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"model not found: {model_path}")
        if not vocab_path.is_file():
            raise FileNotFoundError(f"vocab not found: {vocab_path}")

        state_dict = torch_load_state_dict(model_path)
        dtype = infer_runtime_dtype(state_dict, args.dtype)
        self.model, self.dims = build_model(state_dict, args.device, dtype)
        self.tokenizer = build_tokenizer(vocab_path)
        self.device = args.device
        self.dtype = str(dtype)
        self.model_path = str(model_path)
        self.default_max_new_tokens = args.default_max_new_tokens
        self.lock = threading.Lock()

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("request field 'prompt' must be a non-empty string")

        max_new_tokens = int(request.get("max_new_tokens", self.default_max_new_tokens))
        if max_new_tokens < 0 or max_new_tokens > 4096:
            raise ValueError("max_new_tokens must be between 0 and 4096")
        sample = bool(request.get("sample", False))
        temperature = float(request.get("temperature", 1.0))
        top_p = float(request.get("top_p", 0.0))
        prompt_tokens = self.tokenizer.encode(prompt)
        if not prompt_tokens:
            raise ValueError("prompt encoded to zero tokens")

        with self.lock, torch.inference_mode():
            state = self.model.init_state(batch_size=1, device=self.device)
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
            logits, state = self.model(input_tensor, state=state, return_state=True)
            logits = logits[0, -1]
            generated_tokens: list[int] = []
            for step in range(max_new_tokens):
                next_token = sample_token(
                    logits,
                    sample=sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                generated_tokens.append(next_token)
                if step + 1 == max_new_tokens:
                    break
                input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                logits, state = self.model(input_tensor, state=state, return_state=True)
                logits = logits[0, -1]

        generated_text = self.tokenizer.decode(generated_tokens) if generated_tokens else ""
        return {
            "text": generated_text,
            "full_text": prompt + generated_text,
            "prompt_tokens": len(prompt_tokens),
            "generated_tokens": len(generated_tokens),
            "model": self.model_path,
        }


class RequestHandler(BaseHTTPRequestHandler):
    service: RawRWKVService

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[rwkv-raw] " + (format % args) + "\n")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(404, {"error": "not found"})
            return
        self.send_json(
            200,
            {
                "status": "ok",
                "model": self.service.model_path,
                "device": self.service.device,
                "dtype": self.service.dtype,
                "dims": self.service.dims,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/generate":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("JSON body must be an object")
            self.send_json(200, self.service.generate(request))
        except Exception as exc:  # keep the small service alive after bad requests
            self.send_json(400, {"error": str(exc)})


def main() -> int:
    args = parse_args()
    service = RawRWKVService(args)
    RequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(
        f"[rwkv-raw] listening on http://{args.host}:{args.port} "
        f"model={service.model_path} dims={service.dims}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
