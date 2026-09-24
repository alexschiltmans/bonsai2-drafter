#!/usr/bin/env python3
"""Does the 2-bit, group-128 small-M unpack read what MLX packed? No server needed, but a GPU is.

    ~/.venv-dspark/bin/python bench/tests/test_small_m_2bit.py

Three checks, in the order the patch's docstring makes its claims:

1. The source edit. The kernel variant is upstream's source with the group index moved from
   64 to 128 values; the edit must land, and must refuse an upstream that has changed the
   lines it edits rather than silently producing a kernel that reads the wrong scale.
2. The packing. MLX's 2-bit affine layout is asserted to be a dense little-endian bitstream,
   16 values per uint32; unpack a quantized row in numpy and compare against `mx.dequantize`
   for every index.
3. The kernel. Run it against `quantized_matmul` on the shapes the served model has, at every
   verify width it serves (M 6..8), under the tolerance upstream's `measure_shapes` gate uses,
   and confirm `eligible` now admits a 2-bit group-128 QuantizedLinear and still refuses a
   2-bit group-64 one.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_dspark import small_m_qmm as smm

from patches.small_m_2bit import BITS, GROUP, TwoBitG128VerifyKernel, kernel_source

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ 1. the source edit
print("== 1. the source edit")
src = kernel_source(smm._SRC)
check("group index moved to 128 values", "int g = ka >> 7;" in src and "(K / 128)" in src)
check("no group-64 indexing survives", "ka >> 6" not in src and "(K / 64)" not in src)
check("the unpack is spliced in", "__UNPACK__" not in src and "(p >> (2 * t)) & 3u" in src)
try:
    kernel_source(smm._SRC.replace("int g = ka >> 6;", "int g = ka / 64;"))
    check("a changed upstream is refused", False)
except RuntimeError:
    check("a changed upstream is refused", True)

# ------------------------------------------------------------------ 2. the packing
print("== 2. the packing")
w = mx.random.normal((8, 512))
q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
deq = np.asarray(mx.dequantize(q, s, b, group_size=GROUP, bits=BITS).astype(mx.float32))
words = np.asarray(q).astype(np.uint32)
codes = np.stack([(words >> (2 * t)) & 3 for t in range(16)], axis=-1).reshape(8, -1)
sc = np.asarray(s.astype(mx.float32))
bi = np.asarray(b.astype(mx.float32))
manual = codes.reshape(8, -1, GROUP) * sc[..., None] + bi[..., None]
check("16 little-endian 2-bit values per word reproduce mx.dequantize",
      np.allclose(manual.reshape(8, -1), deq, atol=1e-3),
      f"max diff {np.abs(manual.reshape(8, -1) - deq).max():.4f}")

# ------------------------------------------------------------------ 3. the kernel
print("== 3. the kernel")
p = TwoBitG128VerifyKernel()
ok, status = p.install()
check("patch installs", ok, status)
check("registered under bits 2", BITS in smm._KERNELS)

SHAPES = [(17408, 5120), (5120, 17408), (10240, 5120), (248320, 5120)]
worst = 0.0
for N, K in SHAPES:
    wq, ws, wb = mx.quantize(mx.random.normal((N, K)), group_size=GROUP, bits=BITS)
    ws, wb = ws.astype(mx.bfloat16), wb.astype(mx.bfloat16)
    for M in range(smm.M_MIN, smm.M_MAX + 1):
        x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
        x8 = x if M == 8 else mx.concatenate([x, mx.zeros((8 - M, K), dtype=x.dtype)], axis=0)
        ref = mx.quantized_matmul(x, wq, ws, wb, transpose=True, group_size=GROUP,
                                  bits=BITS).astype(mx.float32)
        got = smm._mma(x8, wq, ws, wb, M, N, K, BITS).astype(mx.float32)
        d = float(mx.max(mx.abs(ref - got)).item())  # type: ignore[arg-type]
        scale = float(mx.max(mx.abs(ref)).item())  # type: ignore[arg-type]
        worst = max(worst, d / max(scale, 1.0))
    mx.clear_cache()
check(f"numerics within upstream's {smm._REL_TOL} tolerance on {len(SHAPES)} shapes, M 6-8",
      worst <= smm._REL_TOL,
      f"worst {worst:.4f}")

ql = nn.QuantizedLinear(5120, 17408, bias=False, group_size=GROUP, bits=BITS)
check("a 2-bit group-128 projection is eligible", smm.eligible(ql))
ql64 = nn.QuantizedLinear(5120, 17408, bias=False, group_size=64, bits=BITS)
check("a 2-bit group-64 projection is not", not smm.eligible(ql64))
ql4 = nn.QuantizedLinear(5120, 17408, bias=False, group_size=64, bits=4)
check("upstream's 4-bit group-64 rule is untouched", smm.eligible(ql4))
small = nn.QuantizedLinear(5120, 1024, bias=False, group_size=GROUP, bits=BITS)
check("N below the grid minimum is refused", not smm.eligible(small))

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("all checks passed")
