#!/usr/bin/env python3
"""GPU-free checks for bench/throughput/pool.py: arm figures pool tokens over seconds across
legs, never average leg rates, and a leg without decode timing leaves the decode pool empty."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.throughput import pool


def leg(label: str, d_tok: int, d_sec: float, e_tok: int, e_sec: float) -> dict:
    return {"label": label, "pooled_decode_tokens": d_tok, "pooled_decode_seconds": d_sec,
            "pooled_e2e_tokens": e_tok, "pooled_e2e_seconds": e_sec,
            "pooled_decode_tps": d_tok / d_sec if d_sec else 0.0, "pooled_e2e_tps": e_tok / e_sec,
            "clean": True, "truncated_requests": 1, "no_decode_timing_requests": 0}


class PoolTest(unittest.TestCase):
    def write(self, tmp: str, record: dict) -> str:
        path = Path(tmp) / f"{record['label']}.json"
        path.write_text(json.dumps(record))
        return str(path)

    def test_pools_tokens_over_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # 100 tokens in 10 s and 300 in 10 s: pooled 20 tok/s, not the mean of 10 and 30
            a = pool.pool([self.write(tmp, leg("a1", 100, 10.0, 100, 11.0)),
                           self.write(tmp, leg("a2", 300, 10.0, 300, 11.0))])
            self.assertAlmostEqual(a["decode_tps"], 20.0)
            self.assertAlmostEqual(a["e2e_tps"], 400 / 22.0)
            self.assertEqual(a["truncated"], 2)
            self.assertEqual(a["labels"], ["a1", "a2"])

    def test_no_decode_timing_leaves_decode_pool_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            b = pool.pool([self.write(tmp, leg("b1", 0, 0.0, 200, 10.0))])
            self.assertIsNone(b["decode_tps"])
            self.assertAlmostEqual(b["e2e_tps"], 20.0)


if __name__ == "__main__":
    unittest.main(verbosity=0)
