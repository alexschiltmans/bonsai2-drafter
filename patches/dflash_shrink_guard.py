"""Stop the dflash shrink fault from costing a turn and a cold prefill.

The fault
---------
A request whose prompt is SHORTER than the conversation the drafter is already caching,
while still sharing its prefix, returns HTTP 500 with `ValueError: [full] Negative
dimensions not allowed` and empties the target's prefix cache, so the retry pays a full
cold prefill. Reproduced deterministically by a standalone script, and hit 9 times in 84 requests of the tool-call battery.

The cause
---------
The drafter's per-layer context cache is an mlx-lm `RotatingKVCache(max_size=
sliding_window - 1)`, so 2047 rows on this checkpoint. On a prefix-cache hit,
`dflash_generate` restores the stored context window by presetting each layer cache's
`offset` to the window's absolute start and appending the window's rows:

    for c in dcache:
        c.offset = window.start
    drafter.append_ctx(restored, dcache)

`RotatingKVCache` never expected an offset preset. Its lazy allocation assumes `offset`
counts rows it actually holds, and after this restore it holds `window.rows` rows while
`offset` reads `window.end`. That is harmless as long as the buffer ends up full. It is
not harmless when the window was **trimmed**, which is exactly what a shorter prompt
causes: the prefix cache trims the stored window back to the shared prefix, so the restore
appends fewer than `max_size` rows while `offset` still lands past `max_size`.

The next draft round appends a single row, which routes to `_update_in_place`, and:

    prev = self.offset                                  # 2680, past max_size
    if self.keys is None or (prev >= self.keys.shape[2]  # 2680 >= 2009, true
            and self.keys.shape[2] < self.max_size):     # 2009 <  2047, true
        new_size = min(self.step, self.max_size - prev)  # min(256, -633) = -633
        new_k = mx.zeros((B, H, new_size, D), ...)       # ValueError

So the unsafe state is exactly `rows < max_size and start + rows > max_size`. A full
window (`rows == max_size`) is safe, which is why an ordinary appending turn never trips
this and only a shrink does. A short conversation (`end <= max_size`) is safe too.

The fix
-------
Skip the restore when it would produce that state, and reset the offsets so the drafter
starts context-bare instead. That is not a new behaviour: it is what the code already does
for a window trimmed too deep to cover its rung, and `DFlashCtxWindow`'s own docstring
says what it costs.

    A window trimmed too deep to cover its rung goes EMPTY: drafting then starts
    context-bare and self-heals as rounds append - less acceptance for a while, never a
    correctness issue (the target verifies every token).

So the price is a few rounds of weaker drafting on the turn that would otherwise have
failed outright, against a lost turn plus a cold prefill (measured at 22.2s at chat depth
and around 290s at 33k). The target verifies every token either way, so no output changes
that could not already have come out of a cold start.

`DFlashDraftModel.append_ctx` is the right place to put it: its own docstring says it exists
the prefix-cache restore, so wrapping it catches every restore and nothing else.

Usage
-----
Installed by :func:`patches.install_all`, which `bin/mlx-dspark-patched` calls before any
model loads.
"""

from __future__ import annotations

import sys
from typing import Any

from . import Patch


class DFlashShrinkGuard(Patch):
    """Decline a drafter context restore that a rotating cache cannot represent.

    ``DFlashDraftModel.append_ctx`` is the wrapping point because its own docstring says it
    exists for the prefix-cache restore, so this catches every restore and nothing else.
    """

    name = "dflash shrink guard"

    #: Restores declined so far, for anyone who wants to count them.
    def __init__(self) -> None:
        self.skipped = 0

    def applied(self) -> bool:
        from mlx_dspark import dflash_model

        return getattr(dflash_model.DFlashDraftModel.append_ctx, "_bonsai2_guard", False)

    def apply(self) -> None:
        from mlx_dspark import dflash_model

        original = dflash_model.DFlashDraftModel.append_ctx
        guard = self

        def append_ctx(self: Any, h_ctx: Any, cache: Any) -> Any:
            rows = int(h_ctx.shape[1])
            first = next((c for c in cache if c is not None), None)
            max_size = getattr(first, "max_size", None)
            start = getattr(first, "offset", 0)
            # Unrepresentable for a RotatingKVCache: fewer rows held than max_size, with the
            # offset already past it. Restoring here is what raises two rounds later.
            if max_size is not None and rows < max_size and start + rows > max_size:
                guard.skipped += 1
                for c in cache:
                    if c is not None:
                        c.offset = 0
                print(f"[bonsai2-drafter] dflash restore declined: a trimmed context window "
                      f"({rows} rows ending at {start + rows}, max {max_size}) cannot be "
                      f"restored into a rotating cache; drafting starts context-bare for "
                      f"this turn instead of failing it", file=sys.stderr, flush=True)
                return None
            return original(self, h_ctx, cache)

        setattr(append_ctx, "_bonsai2_guard", True)  # noqa: B010 - a marker mypy would otherwise refuse
        dflash_model.DFlashDraftModel.append_ctx = append_ctx
