"""Teach mlx-dspark's small-M verify kernel to read 5-bit affine weights.

Why this file exists
--------------------
Investigation 8 predicted that "a 5-bit checkpoint might miss the custom small-M kernel
entirely", and it does: upstream unpacks 4-bit and 8-bit, ``small_m_6bit.py`` adds 6-bit,
and ``eligible()`` refuses every other width. Served without the kernel the target's verify
curve is the rising one the kernel flattens, the derived cap falls to about 3, and a 5-bit
arm would measure the missing kernel rather than the precision. This adds the unpack so a
5-bit target can be compared like for like with the 4, 6 and 8-bit arms.

The packing
-----------
MLX affine quantization at 5 bits is the same dense little-endian bitstream as at 6: value
``i`` of a row occupies bits ``[5i, 5i+5)`` of that row's uint32 array. Probed rather than
assumed (``bench/tests/test_small_m_5bit.py`` reproduces it for every index of a row).

The alignment the 6-bit unpack relies on is gone at 5 bits. Sixteen values are 80 bits,
two and a half words, so a lane's slice starts on a word boundary for even ``kq`` and 16
bits into a word for odd ``kq``. Both cases are handled by the same loop with a different
starting bit; the body is emitted twice, once per start, so the word index and shift stay
compile-time constants after unrolling, as they are in the 4/6/8-bit unpacks. Everything
else still holds: ``ka`` is a multiple of 64 (320 bits, ten words), so the per-lane bit
offset is ``80 * kq`` past a word boundary, and a 64-value quantization group is ten whole
words, so the one scale/bias pair loaded per group stays correct.

Reads stay inside the row. The furthest lane (start bit 16, value 15) ends at bit 96 of its
three words and never touches a fourth; the last lane of a row lands exactly on the row's
last word. The spill condition is ``sh > 27``: five bits starting above bit 27 cross into
the next word, and at the last value of the odd start (bit 91, sh 27) it is false, so
``p[3]`` is never read and no shift reaches 32.

Usage
-----
Installed by :func:`patches.install_all`, like the 6-bit unpack, before any model loads.
"""

from __future__ import annotations

from . import Patch

BITS = 5

# One lane's 16 values from three words, starting ``S`` bits into the first. ``b``, ``wi``
# and ``sh`` are constants once the loop is unrolled, for a constant ``S``.
_LOOP = r"""
                for (int t = 0; t < 16; ++t) {
                    int b  = %(S)d + 5 * t;
                    int wi = b >> 5;
                    int sh = b & 31;
                    uint v = p[wi] >> sh;
                    if (sh > 27) v |= p[wi + 1] << (32 - sh);
                    v &= 31u;
                    bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)v * s + bb);
                }
"""

# Row stride is K*5/32 words (K % 512 == 0 makes it whole). The lane's first bit is
# 5*(ka + 16*kq) = 320*(ka/64) + 80*kq, so its word is that >> 5 and its start bit within
# the word is 0 for even kq and 16 for odd kq.
_UNPACK_TEMPLATE = r"""
            const size_t bit0 = ((size_t)ka + (size_t)kq * 16) * 5;
            const device uint* wr = w + (size_t)n * ((size_t)K * 5 / 32) + (bit0 >> 5);
            uint p[3] = {wr[0], wr[1], wr[2]};
            if ((kq & 1) == 0) {%(even)s            } else {%(odd)s            }
"""
# Metal source is full of braces, so %-substitution is the safe one here.
UNPACK_5BIT = _UNPACK_TEMPLATE % {"even": _LOOP % {"S": 0}, "odd": _LOOP % {"S": 16}}


class FiveBitVerifyKernel(Patch):
    """Register the 5-bit unpack in mlx-dspark's small-M kernel table.

    As with the 6-bit patch, only the unpack is new. The kernel source, the shape rule and
    both of upstream's gates (numerics against ``quantized_matmul`` per shape, then a speed
    race) are its own, so a wrong unpack is caught at load and leaves the path off.
    """

    name = "5-bit small-M verify kernel"

    def applied(self) -> bool:
        from mlx_dspark import small_m_qmm as smm

        return BITS in smm._KERNELS

    def apply(self) -> None:
        import mlx.core as mx
        from mlx_dspark import small_m_qmm as smm

        smm._UNPACK[BITS] = UNPACK_5BIT
        smm._KERNELS[BITS] = mx.fast.metal_kernel(
            name=f"dspark_qmm_mma{BITS}",
            input_names=["x", "w", "sc", "bi"],
            output_names=["out"],
            source=smm._SRC.replace("__UNPACK__", UNPACK_5BIT),
        )
