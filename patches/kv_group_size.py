"""Make the KV cache's quantization group size reachable, and provable, so it can be measured.

mlx-dspark hard-codes it: ``Target.__init__(..., kv_group_size: int = 64)`` in target.py,
with no CLI flag and no caller that passes anything else. It reaches
``QuantizedKVCache(self.kv_group_size, self.kv_bits)`` and from there the group layout that
``quantized_scaled_dot_product_attention`` reads on every verify round.

**Why it is worth reaching.** Pinned-cap arms established that quantizing the
KV cache costs a flat **~24 ms per round at 22.5k**, independent of verify width — the cache
is dequantized once per round no matter how many query rows read it. A matched-shape kernel
sweep (`bench/kernels/kv_kernel_sweep.py`) then found the group size is the dominant term in it, and
that the shipped 64 is not the good end of the range: group 32 costs about half of group 64
for 6% more KV, everywhere measured.

**This patch changes nothing by default.** Absent ``LLM_KV_GROUP_SIZE`` the default stays 64
and a server launched without it is byte-identical in behaviour to stock. Nothing here is a
recommendation; it makes an arm runnable and, as importantly, *identifiable*.

Three things this has to get right, and the first version got the second one wrong:

**One parser, not two.** The constructor read the variable with ``int(raw)`` while the cache
key matched ``raw in ("32", "128")``. So ``LLM_KV_GROUP_SIZE=" 32"`` — or ``032``, or ``+32``
— built a group-32 cache and stored its curves under the **stock group-64 key**, which is
exactly the cross-contamination the key tag exists to prevent. Both now call :func:`resolved`,
so they cannot disagree by construction.

**Tag the calibration key.** ``calibrate._cache_key`` encodes ``kv_bits`` — the line that adds
it says "quantized KV changes the verify curve" — but not the group size, which changes that
curve by ~13 ms per round at 22.5k. Without a tag, a group-32 run writes curves that a later
stock run loads, and the cap controller prices a kernel it is not running. Same collision
``|smm`` and ``|sdps`` already exist to prevent, so it uses the same shape.

**All of it or none of it.** If the key wrapper failed to install after the constructor
wrapper succeeded, the result would be a changed cache layout under an unpatched key — worse
than not applying at all, and silently so. :meth:`apply` unwinds the constructor if the key
wrap raises.

**Provenance.** The resolved values are printed to the server log at construction and at first
calibration, because neither the command line nor ``/health`` exposes them: an arm whose group
size is asserted only by its label is not a scored arm. ``instrument.kv_cache_identity()``
reads them back.
"""

from __future__ import annotations

import os
from typing import Any

from . import Patch

ENV = "LLM_KV_GROUP_SIZE"
STOCK = 64
VALID = (32, 64, 128)
#: Printed to the server log so a run record can carry what actually ran, not what was asked.
MARK_CACHE = "[bonsai2-drafter] kv cache resolved:"
MARK_KEY = "[bonsai2-drafter] calibration key:"

_warned: set[str] = set()
_logged_keys: set[str] = set()


class InvalidGroupSize(ValueError):
    """The override is set to something unusable. Raised at install, never at request time."""


def resolved(strict: bool = False) -> int | None:
    """The group size this process will use, or ``None`` for stock. The only parser.

    ``int(raw.strip())`` accepts ``032``, ``+32`` and padded forms, and canonicalizes them to
    the same integer every caller sees — the constructor and the cache key included. An
    unusable value returns ``None`` (stock) after warning once, or raises under ``strict``,
    which is how install-time rejects a typo instead of running stock under a wrong label.
    """
    raw = os.environ.get(ENV)
    if raw is None or not raw.strip():
        return None
    try:
        want = int(raw.strip())
    except ValueError:
        want = None
    if want in VALID:
        return want
    msg = f"{ENV}={raw!r} is not one of {'/'.join(map(str, VALID))}"
    if strict:
        raise InvalidGroupSize(msg)
    if raw not in _warned:
        _warned.add(raw)
        print(f"  {msg}; using the stock {STOCK}", flush=True)
    return None


class KVGroupSizePatch(Patch):
    name = "kv group size from LLM_KV_GROUP_SIZE"

    def applied(self) -> bool:
        import importlib

        from mlx_dspark import target
        cal = importlib.import_module("mlx_dspark.calibrate")
        # BOTH halves, always. A half-applied patch is the dangerous state this guards.
        return (getattr(target.Target.__init__, "_bonsai2_kv_group", False)
                and getattr(cal._cache_key, "_bonsai2_kv_group", False))

    def apply(self) -> None:
        import importlib

        from mlx_dspark import target
        cal = importlib.import_module("mlx_dspark.calibrate")

        # Reject a typo here rather than at request time: a server that quietly ran stock
        # under an arm labelled "g32" is how a benchmark lies. install() turns this into a
        # logged "unavailable" line and leaves a stock server, which is the safe failure.
        resolved(strict=True)

        original_init = target.Target.__init__

        def __init__(self: Any, model: Any, tokenizer: Any, *, kv_bits: int | None = None,
                     kv_group_size: int = STOCK, **kw: Any) -> None:
            want = resolved()
            if want is not None:
                kv_group_size = want
            out = original_init(self, model, tokenizer, kv_bits=kv_bits,
                                kv_group_size=kv_group_size, **kw)
            # What actually got built, read back off the instance rather than off the
            # environment, so the log records the resolved cache and not the request for one.
            print(f"{MARK_CACHE} bits={getattr(self, 'kv_bits', None)} "
                  f"group={getattr(self, 'kv_group_size', None)}", flush=True)
            return out

        setattr(__init__, "_bonsai2_kv_group", True)  # noqa: B010 - a marker mypy would otherwise refuse
        __init__.__doc__ = original_init.__doc__
        target.Target.__init__ = __init__

        original_key = cal._cache_key

        def _cache_key(mode: str, target_repo: str, drafter_repo: str | None,
                       ctx_len: int = cal.CTX_LEN, kv_bits: int | None = None) -> str:
            key = original_key(mode, target_repo, drafter_repo, ctx_len=ctx_len,
                               kv_bits=kv_bits)
            want = resolved()
            # Only when the cache is actually quantized AND the group is not stock, so every
            # stock key, and every non-KV mode's key, is byte-identical to upstream and all
            # existing cache entries stay valid.
            if kv_bits and want is not None and want != STOCK:
                key += f"|g{want}"
            # Once per distinct key. _cache_key is called several times per load and the
            # line is provenance, not a trace.
            if key not in _logged_keys:
                _logged_keys.add(key)
                print(f"{MARK_KEY} {key}", flush=True)
            return key

        setattr(_cache_key, "_bonsai2_kv_group", True)  # noqa: B010 - as above
        try:
            setattr(cal, "_cache_key", _cache_key)  # noqa: B010 - rebinding a module global
        except Exception:
            target.Target.__init__ = original_init      # never leave the halves disagreeing
            raise
