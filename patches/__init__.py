"""Patches this repo applies to the pinned mlx-dspark venv, at import time.

Why they live here and not in `site-packages`: the environment is checked against its
`uv.lock`, and an edited file inside the venv is drift that check cannot see. Every
measurement is attached to a lock, so the venv stays byte-identical to it and
the changes ride in front of the CLI instead. All of them belong upstream at
https://github.com/ARahim3/mlx-dspark.

Each patch is a :class:`Patch`: it knows its own name, whether it is already in effect, and
how to put itself in effect. The base class owns the parts that were otherwise going to be
written once per patch and drift apart: idempotency, the "did it actually take" check after
applying, and the rule that a failed patch degrades to stock rather than killing the server.
That last one matters more than it looks. A stock server is slower and loses the odd turn;
no server at all is a broken machine, and these are optimizations.

Adding one: subclass :class:`Patch`, implement :meth:`Patch.applied` and
:meth:`Patch.apply`, and add it to :data:`REGISTRY`. `bin/mlx-dspark-patched` needs no edit.
"""

from __future__ import annotations

import sys
from typing import TextIO


class Patch:
    """One change against the pinned mlx-dspark venv.

    Subclasses implement two questions and nothing else. :meth:`applied` answers whether the
    change is in effect, which is asked both before applying (to stay idempotent, and to
    stand aside for a future mlx-dspark that ships the fix itself) and after (so a patch
    that silently failed to take is reported rather than assumed). :meth:`apply` does it.
    """

    #: Shown in the launch log. A short noun phrase, not a sentence.
    name = "unnamed patch"

    def applied(self) -> bool:
        """True when this patch is already in effect."""
        raise NotImplementedError

    def apply(self) -> None:
        """Put it in effect. Called only when :meth:`applied` is False."""
        raise NotImplementedError

    def install(self) -> tuple[bool, str]:
        """Apply if needed. Returns (in effect now, one-word status).

        Never raises: an optimization that cannot be applied leaves a server that is slower
        rather than one that is missing, so the exception becomes a status line instead.
        """
        try:
            if self.applied():
                return True, "already present"
            self.apply()
            if self.applied():
                return True, "installed"
            return False, "applied but did not take effect"
        except Exception as exc:  # noqa: BLE001 - see the docstring
            return False, f"unavailable ({type(exc).__name__}: {exc})"


def registry() -> list[Patch]:
    """The patches to apply, in order. Imported lazily so importing this package costs
    nothing until something actually installs them."""
    from .bonsai_loader import BonsaiLoader
    from .dflash_prequantized import DFlashPrequantized
    from .dflash_shrink_guard import DFlashShrinkGuard
    from .kv_group_size import KVGroupSizePatch
    from .small_m_2bit import TwoBitG128VerifyKernel
    from .small_m_5bit import FiveBitVerifyKernel
    from .small_m_6bit import SixBitVerifyKernel

    # KVGroupSizePatch is inert without LLM_KV_GROUP_SIZE in the environment, so it is safe
    # ahead of the others; it only widens what a launch can ask for.
    # BonsaiLoader only answers for one model_type, so it is inert for every other target.
    # DFlashPrequantized only answers for a drafter directory carrying its own marker, so it
    # is inert for every bfloat16 checkpoint, including the shipped ft5 one.
    return [SixBitVerifyKernel(), FiveBitVerifyKernel(), TwoBitG128VerifyKernel(),
            DFlashShrinkGuard(), KVGroupSizePatch(), BonsaiLoader(), DFlashPrequantized()]


def install_all(stream: TextIO = sys.stderr) -> list[Patch]:
    """Install every patch, log one line each, and return the ones now in effect.

    Called by `bin/mlx-dspark-patched` before any model loads, which is the only moment that
    works: the kernel dispatch table is read when a model is quantized, and the drafter's
    restore path is bound when a request arrives.
    """
    live = []
    for patch in registry():
        ok, status = patch.install()
        print(f"[bonsai2-drafter] {patch.name}: {status}", file=stream, flush=True)
        if ok:
            live.append(patch)
    return live
