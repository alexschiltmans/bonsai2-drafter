#!/usr/bin/env python3
"""Pool bench5 legs into arm figures and compare two arms. Standard library only.

    python3 bench/throughput/pool.py A1.json A2.json -- B1.json B2.json

Each arm's figure is its total completion tokens over its total seconds, summed over its legs,
never an average of leg rates; the same for the decode pool and the end-to-end pool. It prints
both arms, every leg's rate, whether each leg was clean, the truncation counts, and B over A.
"""
from __future__ import annotations

import json
import sys
from typing import Any


def pool(paths: list[str]) -> dict[str, Any]:
    legs = []
    for path in paths:
        with open(path) as f:
            legs.append(json.load(f))
    d_tok = sum(leg["pooled_decode_tokens"] for leg in legs)
    d_sec = sum(leg["pooled_decode_seconds"] for leg in legs)
    e_tok = sum(leg["pooled_e2e_tokens"] for leg in legs)
    e_sec = sum(leg["pooled_e2e_seconds"] for leg in legs)
    return {
        "labels": [leg["label"] for leg in legs],
        "decode_tps": d_tok / d_sec if d_sec else None,
        "e2e_tps": e_tok / e_sec,
        "legs_decode_tps": [leg["pooled_decode_tps"] for leg in legs],
        "legs_e2e_tps": [leg["pooled_e2e_tps"] for leg in legs],
        "clean": [leg["clean"] for leg in legs],
        "truncated": sum(leg["truncated_requests"] for leg in legs),
        "no_decode_timing_requests": sum(leg["no_decode_timing_requests"] for leg in legs),
    }


def main(argv: list[str]) -> None:
    if "--" not in argv:
        sys.exit(__doc__.split("\n\n")[1])
    cut = argv.index("--")
    a, b = pool(argv[:cut]), pool(argv[cut + 1:])
    out: dict[str, Any] = {"a": a, "b": b, "b_over_a_e2e": b["e2e_tps"] / a["e2e_tps"]}
    if a["decode_tps"] and b["decode_tps"]:
        out["b_over_a_decode"] = b["decode_tps"] / a["decode_tps"]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
