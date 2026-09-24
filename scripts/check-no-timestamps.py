#!/usr/bin/env python3
"""Fail if a file outside evidence/ carries a timestamp. Standard library only.

    python3 scripts/check-no-timestamps.py

Published files carry no timestamps. The known source is a regenerated `uv.lock`, which writes an
`upload-time` for every package; strip those fields (uv accepts the lock without them). Any ISO
date-time is caught too. `evidence/` is checked by `evidence/tools/verify_sanitized.py scan`.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", ".venv", "evidence", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules"}
# The single-character class keeps this line from matching itself.
PATTERN = re.compile(r"upload-tim[e] = |\b20\d\d-[01]\d-[0-3]\dT[0-2]\d:[0-5]\d")


def main() -> int:
    hits = []
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            path = os.path.join(base, name)
            try:
                with open(path, encoding="utf-8") as f:
                    for n, line in enumerate(f, 1):
                        if PATTERN.search(line):
                            hits.append(f"{os.path.relpath(path, ROOT)}:{n}: {line.strip()[:120]}")
            except (UnicodeDecodeError, OSError):
                continue
    for hit in hits[:20]:
        print(hit)
    if hits:
        print(f"{len(hits)} timestamp(s) outside evidence/")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
