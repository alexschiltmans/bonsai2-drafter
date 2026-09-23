"""Teach mlx-dspark's small-M verify kernel to read 2-bit, group-128 affine weights.

Why this file exists
--------------------
Bonsai 2's language model (patches/bonsai_loader.py) is MLX affine 2-bit at group size 128.
mlx-dspark's ``small_m_qmm`` kernel, which makes a wide speculative verify cheap by
dequantizing each weight group once and reusing it across the verify rows, ships unpacks
for 4 and 8 bits (this repo added 5 and 6) and hardcodes a 64-value group. So the ternary
target never reaches it: with the DFlash 2 drafter accepting 5.0 tokens a round at cap 7,
the verify round measured 485 ms against 48 ms for one token before this patch, and
the controller pinned the cap at 1. Ten rows for the price of ten, where the 6-bit qwen
target pays about three.

What changes
------------
Two index expressions and one unpack. The kernel walks K in 64-value chunks per simdgroup
and loads one scale/bias pair per chunk at ``sc[n * (K/64) + (ka >> 6)]``; for a 128-value
group the pair lives at ``sc[n * (K/128) + (ka >> 7)]``, and since every chunk start is a
multiple of 64 a chunk never straddles a group. The unpack reads one uint32 per lane
slice: 16 two-bit values, little-endian, value ``t`` at bits ``[2t, 2t+2)``, which is what
``mx.quantize`` packs and what the pack's ``codec.py`` writes.

The kernel is registered under bits 2 and ``eligible`` is widened to accept group 128 at
2 bits. Both of upstream's gates still apply per shape at load: numerics against
``quantized_matmul`` within 2%, then a dependent-chain race it must win by 1.15x at M=8.
A wrong unpack here leaves the path off, not the answers wrong.

Coupling to know about: the kernel keyed under bits 2 is the group-128 one, so the widened
``eligible`` refuses a 2-bit group-64 layer outright (bench/tests/test_small_m_2bit.py checks
that it does); none is in this repo.
"""

from __future__ import annotations

import contextlib
from typing import Any

from . import Patch

BITS = 2
GROUP = 128

UNPACK_2BIT = r"""
            const device uint* wr = w + (size_t)n * (K / 16) + (ka >> 4) + kq;
            uint p = wr[0];
            for (int t = 0; t < 16; ++t)
                bt[(kq * 16 + t) * 8 + j] = (bfloat16_t)((float)((p >> (2 * t)) & 3u) * s + bb);
"""


def kernel_source(src: str) -> str:
    """Upstream's kernel with the group indexing moved from 64 to 128 values.

    Each of the two edits must land: an upstream that changed one of the lines would
    otherwise yield a kernel that reads the scale of the wrong group, and the numerics gate
    is the only thing that would catch it."""
    edits = (("int g = ka >> 6;", f"int g = ka >> {GROUP.bit_length() - 1};"),
             ("(K / 64)", f"(K / {GROUP})"))
    out = src
    for old, new in edits:
        if old not in out:
            raise RuntimeError(f"small_m_qmm kernel source no longer carries {old!r}, which this patch edits")
        out = out.replace(old, new)
    return out.replace("__UNPACK__", UNPACK_2BIT)


class TwoBitG128VerifyKernel(Patch):
    name = "2-bit group-128 small-M verify kernel"

    def applied(self) -> bool:
        from mlx_dspark import small_m_qmm as smm

        return BITS in smm._KERNELS and getattr(smm.eligible, "_bonsai2_g128", False)

    def apply(self) -> None:
        import mlx.core as mx
        from mlx_dspark import small_m_qmm as smm

        if BITS not in smm._KERNELS:
            smm._UNPACK[BITS] = UNPACK_2BIT
            smm._KERNELS[BITS] = mx.fast.metal_kernel(
                name=f"dspark_qmm_mma{BITS}g{GROUP}",
                input_names=["x", "w", "sc", "bi"],
                output_names=["out"],
                source=kernel_source(smm._SRC),
            )

        if not getattr(smm.eligible, "_bonsai2_g128", False):
            from mlx import nn

            orig = smm.eligible

            def eligible(mod: Any) -> bool:
                if isinstance(mod, nn.QuantizedLinear) and int(mod.bits) == BITS:
                    # The kernel registered under bits 2 is the group-128 one. Upstream's own
                    # rule would now admit a 2-bit group-64 layer to it; refuse that here.
                    if getattr(mod, "mode", "affine") != "affine" or "biases" not in mod:
                        return False
                    if int(mod.group_size) != GROUP:
                        return False
                    N = int(mod["weight"].shape[0])
                    return N >= smm.N_MIN and smm._in_features(mod) % 512 == 0
                return bool(orig(mod))

            setattr(eligible, "_bonsai2_g128", True)  # noqa: B010 - a marker mypy would otherwise refuse
            smm.eligible = eligible
            # calibrate imported the name at module load; rebind it there too, if it did.
            with contextlib.suppress(ImportError):
                from mlx_dspark import calibrate

                if getattr(calibrate, "eligible", None) is orig:
                    calibrate.eligible = eligible
