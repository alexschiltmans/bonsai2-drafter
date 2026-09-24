#!/usr/bin/env python3
"""The Bonsai 2 pack loader: what it recognises, what it builds, and (with the pack) what it
answers.

    ~/.venv-dspark/bin/python bench/tests/test_bonsai_loader.py              (no models)
    ~/.venv-dspark/bin/python bench/tests/test_bonsai_loader.py --with-models

Without models: the pack detection on synthetic config files, the Hadamard transform's
round trip (forward then inverse is the identity, up to the sign vector), and that a
PackedLinear is what mlx-dspark's kernels look for: an `nn.QuantizedLinear` carrying
`bits`, `group_size` and `mode`, whose call applies the transform first.

With models (the 8.6 GB pack in the HuggingFace cache): load through the patch, check the
402 packed modules are installed as PackedLinear/PackedEmbedding, that the model runs in
bfloat16, and that greedy generation answers a fixed arithmetic question correctly. That last
check is the whole reason the loader exists: an ordinary loader answers wrong with no error.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from typing import cast

import mlx.core as mx
from mlx import nn

from patches import bonsai_loader as bl


def _real(a: mx.array) -> float:
    """A scalar reduction as a float; mlx types item() as int | float | complex."""
    return cast(float, a.item())


FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ 1. detection
print("== 1. pack detection")
with tempfile.TemporaryDirectory() as d:
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "prism_hadamard_qwen35"}, f)
    check("a Hadamard pack is recognised", bl.is_bonsai_pack(d))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "qwen3_5"}, f)
    check("a stock checkpoint is not", not bl.is_bonsai_pack(d))
    with open(os.path.join(d, "config.json"), "w") as f:
        f.write("{not json")
    check("a broken config is not, and does not raise", not bl.is_bonsai_pack(d))
check("a missing directory is not", not bl.is_bonsai_pack("/nonexistent/path"))

# ------------------------------------------------------------------ 2. the transform
print("== 2. the Hadamard transform")
signs = mx.where(mx.random.uniform(shape=(1024,)) < 0.5, -1.0, 1.0)
x = mx.random.normal((3, 1024)).astype(mx.bfloat16)
y = bl.fwht(x, 1024, signs)
back = bl.fwht(y, 1024, signs, inverse=True)
diff = _real(mx.max(mx.abs(back.astype(mx.float32) - x.astype(mx.float32))))
check("forward then inverse is the identity", diff < 0.05, f"max diff {diff:.4f}")
check("the transform preserves the norm (orthogonal)",
      abs(_real(mx.sum(y.astype(mx.float32) ** 2))
          / _real(mx.sum(x.astype(mx.float32) ** 2)) - 1.0) < 0.02)
try:
    bl.fwht(mx.zeros((2, 1000)), 1024, mx.ones((1000,)))
    check("a width the block does not divide is refused", False)
except ValueError:
    check("a width the block does not divide is refused", True)

# ------------------------------------------------ 3. PackedLinear is a QuantizedLinear
print("== 3. PackedLinear")
w = mx.random.normal((256, 1024))
q, s, b = mx.quantize(w, group_size=128, bits=2)
pl = bl.PackedLinear((q, s, b), 1024, signs)
check("is an nn.QuantizedLinear", isinstance(pl, nn.QuantizedLinear))
check("carries the format the kernels read",
      pl.bits == 2 and pl.group_size == 128 and pl.mode == "affine")
check("is frozen", not pl.trainable_parameters())
xin = mx.random.normal((2, 1024)).astype(mx.bfloat16)
want = mx.quantized_matmul(bl.fwht(xin, 1024, signs), q, s.astype(mx.bfloat16),
                           b.astype(mx.bfloat16), transpose=True, group_size=128, bits=2)
pl.set_dtype(mx.bfloat16)
got = pl(xin)
check("call is transform, then the affine matmul",
      _real(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))) < 0.05)
plain = bl.PackedLinear((q, s, b), 0, None)
check("block 0 skips the transform",
      _real(mx.max(mx.abs(plain(xin).astype(mx.float32)
                          - mx.quantized_matmul(xin, q, s, b, transpose=True, group_size=128,
                                                bits=2).astype(mx.float32)))) < 0.05)
emb = bl.PackedEmbedding((q, s, b), 1024, signs, mx.bfloat16)
out = emb(mx.array([[3, 7], [0, 255]]))
check("the embedding returns [.., H] rows in the requested dtype",
      out.shape == (2, 2, 1024) and out.dtype == mx.bfloat16)

# ------------------------------------------------------------------ 4. with the pack
if "--with-models" in sys.argv[1:]:
    print("== 4. the pack itself (models)")
    from mlx_dspark.load import _resolve

    path = _resolve("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
    check("the cached pack is recognised", bl.is_bonsai_pack(path))
    model, tok = bl.load_bonsai(path)
    packed = [m for _, m in model.named_modules()
              if isinstance(m, bl.PackedLinear | bl.PackedEmbedding)]
    check("402 packed modules installed", len(packed) == 402, str(len(packed)))
    check("the head is packed", isinstance(model.language_model.lm_head, bl.PackedLinear))
    check("the embedding is packed",
          isinstance(model.language_model.model.embed_tokens, bl.PackedEmbedding))
    check("the model runs in bfloat16", model.language_model.model.norm.weight.dtype == mx.bfloat16)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    question = "What is 17*23? Answer with the number only."
    prompt = tok.apply_chat_template([{"role": "user", "content": question}],
                                     add_generation_prompt=True, tokenize=False,
                                     enable_thinking=False)
    text = generate(model, tok, prompt=prompt, max_tokens=8, sampler=make_sampler(0.0),
                    verbose=False)
    check("greedy generation answers 17*23", "391" in text, text[:60])

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("all checks passed")
