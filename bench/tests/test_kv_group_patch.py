#!/usr/bin/env python3
"""Does a run prove which cache format and calibration key it actually used?

    ~/.venv-dspark/bin/python bench/tests/test_kv_group_patch.py

`LLM_KV_GROUP_SIZE` is an environment override, not a launch argument, and neither the command
line nor `/health` exposes what it resolved to. An arm whose group size is asserted only by its
label is not a scored arm -- and the first version of the patch made that concrete: the
constructor parsed with `int(raw)` while the cache key matched the exact strings "32"/"128", so
`032`, `+32` and whitespace-padded `32` all built a group-32 cache and stored its curves under
the **stock group-64 key**.

These run in seconds and load no model weights. They exist because the failure they check for
is silent: a wrong cache layout under a stock key produces plausible numbers.
"""
import contextlib
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import patches
from patches import kv_group_size as kvg

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"   {'ok  ' if ok else 'FAIL'} {name:<58} {got!r}" + ("" if ok else f" != {want!r}"))
    if not ok:
        FAILURES.append(name)


def fresh_install():
    """Reinstall from stock, so ordering and idempotency can both be exercised."""
    for mod, attr in (("mlx_dspark.target", "Target"), ("mlx_dspark.calibrate", "_cache_key")):
        importlib.reload(importlib.import_module(mod))
    return patches.registry()


with open(os.devnull, "w") as _devnull:
    patches.install_all(stream=_devnull)
from mlx_dspark import target

cal = importlib.import_module("mlx_dspark.calibrate")


class Stub:
    pass


def layout(kv_bits=8, explicit=None):
    """The kv_group_size the Target actually ends up holding."""
    s = Stub()
    kw = {} if explicit is None else {"kv_group_size": explicit}
    # fails later on is_vlm; kv_group_size is assigned before that, which is all this reads
    with contextlib.suppress(Exception):
        target.Target.__init__(s, model=None, tokenizer=None, kv_bits=kv_bits, **kw)
    return getattr(s, "kv_group_size", None)


def key(kv_bits=8):
    return cal._cache_key("dflash", "org/Target", "org/Drafter", kv_bits=kv_bits)


def tag(kv_bits=8):
    k = key(kv_bits)
    last = k.rsplit("|", 1)[-1]
    return last if last.startswith("g") else ""


def setenv(v):
    if v is None:
        os.environ.pop(kvg.ENV, None)
    else:
        os.environ[kvg.ENV] = v


print("== 1. stock behaviour is preserved exactly")
setenv(None)
check("absent variable -> stock layout", layout(), 64)
check("absent variable -> no key tag", tag(), "")
setenv("64")
check("explicit stock 64 -> stock layout", layout(), 64)
check("explicit stock 64 -> no key tag (existing entries stay valid)", tag(), "")

print("\n== 2. requested layouts produce distinct keys")
for v, want in (("32", 32), ("128", 128)):
    setenv(v)
    check(f"{v} -> layout", layout(), want)
    check(f"{v} -> key tag", tag(), f"g{want}")

print("\n== 3. native KV gets no group suffix (nothing is quantized to group)")
setenv("32")
check("kv_bits None -> no key tag", tag(kv_bits=None), "")
check("kv_bits 0 -> no key tag", tag(kv_bits=0), "")

print("\n== 4. THE BUG: noncanonical spellings must not split layout from key")
for v in ("032", "+32", " 32", "32 ", "\t32\n"):
    setenv(v)
    g, t = layout(), tag()
    check(f"{v!r} -> layout and key agree", (g, t), (32, "g32"))

print("\n== 5. invalid values fall back to stock, and say so")
for v in ("48", "0", "-32", "", "   ", "banana", "32.0", "3 2"):
    setenv(v)
    g, t = layout(), tag()
    check(f"{v!r} -> stock layout, stock key", (g, t), (64, ""))

print("\n== 6. an invalid value is REJECTED at install, not run under a wrong label")
for v in ("48", "banana", "32.0"):
    setenv(v)
    try:
        kvg.resolved(strict=True)
        check(f"{v!r} raises at install", "no exception", "InvalidGroupSize")
    except kvg.InvalidGroupSize:
        check(f"{v!r} raises at install", "InvalidGroupSize", "InvalidGroupSize")
setenv("32")
try:
    check("valid value does not raise at install", kvg.resolved(strict=True), 32)
except kvg.InvalidGroupSize:
    check("valid value does not raise at install", "raised", 32)

print("\n== 7. an explicit constructor argument still loses to the override, consistently")
setenv("32")
check("explicit 128 with override 32 -> override wins", layout(explicit=128), 32)
check("...and the key describes the RESOLVED cache, not the argument", tag(), "g32")
setenv(None)
check("explicit 128 with no override -> argument wins", layout(explicit=128), 128)
check("...and the key is stock, which is a KNOWN GAP: a caller that passes a non-stock",
      tag(), "")
print("        group directly would go untagged. No such caller exists in mlx-dspark today")
print("        (the only construction site uses the default), so this is recorded rather")
print("        than fixed -- fixing it needs the key to be derived from the built cache.")

print("\n== 8. repeated installation is idempotent")
setenv("32")
before_init, before_key = target.Target.__init__, cal._cache_key
with open(os.devnull, "w") as _devnull:
    patches.install_all(stream=_devnull)
check("Target.__init__ unchanged by a second install", target.Target.__init__ is before_init, True)
check("_cache_key unchanged by a second install", cal._cache_key is before_key, True)
check("applied() is true", kvg.KVGroupSizePatch().applied(), True)

print("\n== 9. a half-applied patch is refused rather than left in place")
saved_init, saved_key = target.Target.__init__, cal._cache_key
target.Target.__init__ = saved_init.__wrapped__ if hasattr(saved_init, "__wrapped__") else saved_init
cal._cache_key = lambda *a, **k: ""
check("applied() detects a missing key wrapper", kvg.KVGroupSizePatch().applied(), False)
cal._cache_key = saved_key
target.Target.__init__ = saved_init
check("both halves restored", kvg.KVGroupSizePatch().applied(), True)

setenv(None)
print()
if FAILURES:
    print(f"!! {len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all checks passed")
