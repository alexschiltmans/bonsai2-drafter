"""What served_accept.py refuses before it writes a report record; standard library only.

served_accept.py loads the serving stack as it starts, so its checks live here, where the
tests that need no GPU can reach them. bench/REPORTS.md is the specification: a record's
`category` is a string, its `thinking` a boolean and its `decode_seconds` a positive number.
The analyser refuses a report that breaks any of these, so a writer that let one through
would only find out after the run.
"""
from __future__ import annotations

import math
from typing import Any


def corpus_strata(row: dict[str, Any]) -> tuple[str, bool]:
    """A corpus row's (category, thinking) as a record carries them, or ValueError.

    A row without a category is "unknown", as the specification says; a row with one must
    give a string. `thinking` must be a boolean, not merely something that compares equal to
    one: the analyser pairs and stratifies on it. ValueError, not TypeError, because this is
    a malformed corpus, not a caller's mistake, and served_accept.py reports it as a usage
    error.
    """
    category = row.get("category", "unknown")
    if not isinstance(category, str):
        raise ValueError(f"a corpus category must be a string, not {category!r}")  # noqa: TRY004
    thinking = row.get("thinking")
    if not isinstance(thinking, bool):
        raise ValueError(f"a corpus row needs a boolean thinking, not {thinking!r}")  # noqa: TRY004
    return category, thinking


def decode_seconds(seconds: float, prefill_seconds: float) -> float:
    """dflash_generate's wall time after prefill, or ValueError when it is not positive.

    mlx-dspark 0.18.0 reads both from `time.time()`, a clock that can be stepped backwards,
    so the difference is positive in practice but not by construction. Its own
    `decode_seconds` property clamps it to a nanosecond, which would write a time nobody
    measured, and a report refuses a zero, so the run stops instead.
    """
    decode = seconds - prefill_seconds
    if not (math.isfinite(decode) and decode > 0):
        raise ValueError(
            f"dflash_generate timed {seconds} s in all and {prefill_seconds} s of prefill, "
            "which leaves no positive decode time")
    return decode
