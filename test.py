import argparse
from collections import OrderedDict

import torch


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]

    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a dict-like checkpoint")

    out = OrderedDict()
    for k, v in obj.items():
        if not torch.is_tensor(v):
            continue
        if k.startswith("_forward_module."):
            k = k[len("_forward_module.") :]
        out[k] = v.detach().cpu()
    return out


def compare(a, b, atol=0.0, rtol=0.0, topk=30, show_same=False):
    keys_a = set(a.keys())
    keys_b = set(b.keys())

    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    same_tensors = 0
    changed_tensors = 0
    shape_mismatch = []
    dtype_mismatch = []
    changed = []
    same = []

    checked = 0

    for k in common:
        ta = a[k]
        tb = b[k]
        checked += 1

        if ta.shape != tb.shape:
            shape_mismatch.append((k, tuple(ta.shape), tuple(tb.shape)))
            changed_tensors += 1
            continue

        if ta.dtype != tb.dtype:
            dtype_mismatch.append((k, str(ta.dtype), str(tb.dtype)))

        fa = ta.float()
        fb = tb.float()

        if atol == 0.0 and rtol == 0.0:
            is_same = torch.equal(ta, tb)
        else:
            is_same = torch.allclose(fa, fb, atol=atol, rtol=rtol)

        if is_same:
            same_tensors += 1
            same.append((k, str(ta.dtype), tuple(ta.shape)))
            continue

        diff = (fa - fb).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())

        denom = fb.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item())

        changed_tensors += 1
        changed.append((max_abs, mean_abs, max_rel, k, str(ta.dtype), tuple(ta.shape)))

    changed.sort(reverse=True, key=lambda x: x[0])

    print(f"checked: {checked}")
    print(f"same_tensors: {same_tensors}")
    print(f"changed_tensors: {changed_tensors}")
    print(f"only_in_a: {len(only_a)}")
    print(f"only_in_b: {len(only_b)}")
    print(f"shape_mismatch: {len(shape_mismatch)}")
    print(f"dtype_mismatch: {len(dtype_mismatch)}")

    if only_a:
        print("\nOnly in A:")
        for k in only_a[:topk]:
            print(" ", k)

    if only_b:
        print("\nOnly in B:")
        for k in only_b[:topk]:
            print(" ", k)

    if shape_mismatch:
        print("\nShape mismatch:")
        for k, sa, sb in shape_mismatch[:topk]:
            print(f"  {k}: {sa} vs {sb}")

    if dtype_mismatch:
        print("\nDtype mismatch:")
        for k, da, db in dtype_mismatch[:topk]:
            print(f"  {k}: {da} vs {db}")

    if changed:
        print(f"\nTop {min(topk, len(changed))} changed tensors by max_abs:")
        for max_abs, mean_abs, max_rel, k, dtype, shape in changed[:topk]:
            print(
                f"  {k} | dtype={dtype} shape={shape} "
                f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g} max_rel={max_rel:.8g}"
            )

    if show_same and same:
        print(f"\nSame tensors, showing first {min(topk, len(same))}:")
        for k, dtype, shape in same[:topk]:
            print(f"  {k} | dtype={dtype} shape={shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_a")
    parser.add_argument("model_b")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--show-same", action="store_true")
    args = parser.parse_args()

    a = load_state_dict(args.model_a)
    b = load_state_dict(args.model_b)
    compare(a, b, atol=args.atol, rtol=args.rtol, topk=args.topk, show_same=args.show_same)


if __name__ == "__main__":
    main()