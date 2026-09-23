"""Does the 5-bit small-M unpack read what MLX packed? No server needed, but a GPU is.

Two checks, in the order the patch's docstring makes its claims:

1. The packing. MLX's 5-bit affine layout is asserted to be a dense little-endian
   bitstream. Unpack a quantized row bit by bit in numpy and compare against
   ``mx.dequantize`` for every index, so the claim rests on a probe rather than the docs.
2. The kernel. Run the patched kernel against ``quantized_matmul`` on the shapes the
   served model has, at every verify width the kernel serves (M 6..8), under the same
   tolerance upstream's ``measure_shapes`` gate uses. The 6-bit unpack runs through the
   same harness as the control, so a failure here is the unpack and not the test.

Run with the dspark interpreter:  ~/.venv-dspark/bin/python bench/tests/test_small_m_5bit.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import mlx.core as mx
import numpy as np
from mlx_dspark import small_m_qmm as smm

import patches

GROUP = 64
# The served Qwen3.8-27B shapes the kernel is eligible for (N >= 4096, K % 512 == 0).
SHAPES = [(4096, 5120), (5120, 4096), (13824, 5120)]


def bitstream_unpack(words: np.ndarray, bits: int) -> np.ndarray:
    """Values of one packed row, reading bit i of the row as bit (i & 31) of word i >> 5."""
    words = words.astype(np.uint64)
    n = words.size * 32 // bits
    idx = np.arange(n, dtype=np.uint64)[:, None] * bits + np.arange(bits, dtype=np.uint64)
    bit = (words[idx >> 5] >> (idx & 31)) & 1
    return (bit << np.arange(bits, dtype=np.uint64)).sum(axis=1)


def check_packing(bits: int) -> None:
    K = 512
    w = mx.random.normal((8, K))
    wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits)
    ref = np.array(mx.dequantize(wq, sc, bi, group_size=GROUP, bits=bits).astype(mx.float32))
    wq_np, sc_np, bi_np = (np.array(a.astype(mx.float32) if a.dtype != mx.uint32 else a)
                           for a in (wq, sc, bi))
    for r in range(8):
        v = bitstream_unpack(wq_np[r], bits).astype(np.float32)
        g = np.arange(K) // GROUP
        got = v * sc_np[r][g] + bi_np[r][g]
        assert np.allclose(got, ref[r], atol=1e-3, rtol=1e-3), f"{bits}-bit packing, row {r}"
    print(f"  packing  {bits}-bit: dense LE bitstream confirmed for all {K} indices x 8 rows")


def check_kernel(bits: int) -> None:
    for N, K in SHAPES:
        w = mx.random.normal((N, K)) * 0.05
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits)
        worst = 0.0
        for M in range(smm.M_MIN, smm.M_MAX + 1):
            x = (mx.random.normal((M, K)) * 0.1).astype(mx.bfloat16)
            x8 = x if M == 8 else mx.concatenate(
                [x, mx.zeros((8 - M, K), dtype=x.dtype)], axis=0)
            ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True,
                                      group_size=GROUP, bits=bits).astype(mx.float32)
            got = smm._mma(x8, wq, sc, bi, M, N, K, bits).astype(mx.float32)
            diff = mx.max(mx.abs(ref - got)).item()
            scale = max(mx.max(mx.abs(ref)).item(), 1.0)
            worst = max(worst, diff / scale)
            assert diff <= smm._REL_TOL * scale, (
                f"{bits}-bit kernel {N}x{K} M={M}: rel err {diff/scale:.2e} > {smm._REL_TOL}")
        print(f"  kernel   {bits}-bit {K}x{N}: worst rel err {worst:.2e} (tol {smm._REL_TOL})")


def main() -> None:
    live = {p.name for p in patches.install_all(stream=sys.stdout)}
    assert "5-bit small-M verify kernel" in live, "5-bit patch did not install"
    for bits in (6, 5):
        check_packing(bits)
        check_kernel(bits)
    print("OK")


if __name__ == "__main__":
    main()
