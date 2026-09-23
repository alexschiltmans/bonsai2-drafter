"""Teach mlx-dspark's small-M verify kernel to read 6-bit affine weights.

Why this file exists
--------------------
mlx-dspark's ``small_m_qmm`` kernel is what makes a wide speculative verify cheap: it
dequantizes each quantized weight group once and reuses it across the 6-8 verify rows,
where stock ``mx.quantized_matmul`` re-pays the weight read per row. Upstream ships two
unpack routines, for 4-bit and 8-bit affine weights, and ``eligible()`` refuses everything
else. A **6-bit** checkpoint such as Qwen3.8-27B-MLX-6bit never reaches the
kernel: the calibration cache records ``smallm|Qwen3.8-27B-MLX-6bit|- -> shapes: []``, and
the measured verify curve is the rising one the kernel exists to flatten (94 ms at width 2,
104 at 4, 195 at 8), which pins the derived cap at 3.

This module adds the missing unpack. Nothing else changes: the split-K walk, the staging
layout, the MMA and the reduction are upstream's, and so are both gates that decide whether
the kernel is used at all (``measure_shapes`` checks numerics against ``quantized_matmul``
per shape and then races it, keeping only shapes that pass both).

The packing
-----------
MLX affine quantization at 6 bits is a dense little-endian bitstream: value ``i`` of a row
occupies bits ``[6i, 6i+6)`` of that row's uint32 array, straddling word boundaries. Probed
rather than assumed, and reproduced for every index of a 128-wide row.

Two alignments make the unpack cheap, and both are guaranteed by the kernel's own
preconditions rather than hoped for:

- 16 values are exactly 96 bits, so one lane's slice is exactly 3 whole uint32 words.
- ``ka`` is always a multiple of 64 (``eligible`` requires ``K % 512 == 0``, so the
  per-simdgroup span ``K/8`` is a multiple of 64, and the loop steps by 64), and ``kq*16``
  is a multiple of 16. So a lane's first value index is a multiple of 16 and its bit offset
  is a multiple of 96, which is word-aligned.

A 64-value quantization group is 12 words, so group boundaries are word-aligned too, and
the one scale/bias pair the surrounding code loads per group stays correct.

Usage
-----
Installed by :func:`patches.install_all`, which `bin/mlx-dspark-patched` calls before any
model loads. That timing is load-bearing: the dispatch table is read when a model is
quantized, so registering afterwards would be a no-op.
"""

from __future__ import annotations

from . import Patch

# 16 six-bit values from 3 uint32 words. ``b``, ``wi`` and ``sh`` are compile-time
# constants once the trip-16 loop is unrolled, so this costs no dynamic indexing.
#
# The spill branch is what the 4-bit and 8-bit unpacks never need: a value whose 6 bits
# cross a word boundary (t = 5 at bit 30, t = 10 at bit 60) takes its low bits from one
# word and its high bits from the next. ``sh > 26`` is exactly that case, and it also keeps
# the shift below 32: ``p[wi] >> 32`` and ``p[wi + 1] << 32`` are both undefined in Metal,
# and at t = 15 (bit 90, sh = 26) the condition is false, so ``p[3]`` is never read.
UNPACK_6BIT = r"""
            const device uint* wr = w + (size_t)n * ((size_t)K * 3 / 16)
                                      + (((size_t)ka * 3) >> 4) + kq * 3;
            uint p[3] = {wr[0], wr[1], wr[2]};
            for (int t = 0; t < 16; ++t) {
                int b  = 6 * t;
                int wi = b >> 5;
                int sh = b & 31;
                uint v = p[wi] >> sh;
                if (sh > 26) v |= p[wi + 1] << (32 - sh);
                v &= 63u;
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)v * s + bb);
            }
"""

BITS = 6


class SixBitVerifyKernel(Patch):
    """Register the 6-bit unpack in mlx-dspark's small-M kernel table.

    Nothing else is touched: the kernel source, the shape eligibility rule and both of
    upstream's gates (numerics against ``quantized_matmul`` per shape, then a speed race)
    are its own, so a wrong unpack here is caught at load and simply leaves the path off.
    """

    name = "6-bit small-M verify kernel"

    def applied(self) -> bool:
        # Also the stand-aside: a future mlx-dspark shipping its own 6-bit unpack wins.
        from mlx_dspark import small_m_qmm as smm

        return BITS in smm._KERNELS

    def apply(self) -> None:
        import mlx.core as mx
        from mlx_dspark import small_m_qmm as smm

        smm._UNPACK[BITS] = UNPACK_6BIT
        smm._KERNELS[BITS] = mx.fast.metal_kernel(
            name=f"dspark_qmm_mma{BITS}",
            input_names=["x", "w", "sc", "bi"],
            output_names=["out"],
            source=smm._SRC.replace("__UNPACK__", UNPACK_6BIT),
        )
