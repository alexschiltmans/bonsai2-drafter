#!/usr/bin/env python3
"""GPU-free checks for the throughput runner, against a mock OpenAI-compatible server.

The mock answers in both flavours the runner reads: mlx-dspark's `x_mlx_dspark` block with a
JSON /metrics, and llama.cpp's `timings` object with /props and a Prometheus /metrics. The
checks hold the parts a wrong number would come from: which field decode seconds is read
from, the pooling arithmetic, the warm-up staying out of the pool, failed requests staying out
of it, and a record that can be published as written.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.throughput import bench5

# sha256 of the five prompts joined by newlines, as the published numbers were taken with.
PROMPTS_SHA256 = "73f6bb02c67534ba3e0a6c373201e6a3d31e0addfff676fdb332359e67d586a7"
DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}")
TIME_KEY = re.compile(r"created|date|time|stamp|when|host|user|path", re.IGNORECASE)


class Mock:
    """What the mock server does; tests set it and read it back."""

    def __init__(self, flavour: str) -> None:
        self.flavour = flavour
        self.bodies: list[dict[str, Any]] = []
        self.requests = 0              # served chat requests, as /metrics reports them
        self.tokens = 0                # generated tokens, as llama.cpp's /metrics reports them
        self.extra_requests = 0        # another client's traffic, added between /metrics reads
        self.fail: dict[int, int] = {}     # chat request index -> HTTP status
        self.no_usage: set[int] = set()    # chat request indices answered without usage
        self.metrics_reads = 0
        self.lock = threading.Lock()

    def completion(self, index: int, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if index in self.fail:
            return self.fail[index], {"error": {"message": "boom"}}
        n = int(body["max_tokens"]) if index == 0 else 100 + 10 * (index % 5)
        d: dict[str, Any] = {"id": "chatcmpl-x", "object": "chat.completion",
                             "created": 1700000000,
                             "choices": [{"index": 0, "finish_reason": "length" if n >= 400
                                          else "stop",
                                          "message": {"role": "assistant", "content": "x"}}]}
        if index not in self.no_usage:
            d["usage"] = {"prompt_tokens": 40, "completion_tokens": n, "total_tokens": 40 + n}
        decode = n / 50                 # 50 tok/s, server-timed
        if self.flavour == "dspark":
            d["x_mlx_dspark"] = {"mode": "dflash", "accept_len": 2.5, "cap": 7,
                                 "decode_seconds": decode, "prefill_seconds": 0.2,
                                 "ttft_seconds": 0.2, "target_forwards": n // 2,
                                 "decode_tokens_per_sec": 999.0,   # must not be used
                                 "created_at": 1700000000}
        elif self.flavour == "llama":
            d["timings"] = {"prompt_n": 40, "prompt_ms": 200.0, "predicted_n": n,
                            "predicted_ms": decode * 1000, "draft_n": n, "draft_n_accepted": n // 2,
                            "cache_n": 0}
        self.tokens += n if index not in self.no_usage else 0
        return 200, d


class Handler(BaseHTTPRequestHandler):
    mock: Mock

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, payload: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        m = self.mock
        if self.path == "/health":
            self._send(200, json.dumps({"status": "ok", "max_draft": 7,
                                        "model": "/secret/home/models/target"}).encode())
        elif self.path == "/props" and m.flavour == "llama":
            self._send(200, json.dumps({
                "model_path": "/secret/home/models/bonsai.gguf", "build_info": "b1-abc",
                "total_slots": 1, "default_generation_settings": {
                    "n_ctx": 4096, "params": {"temperature": 0.8, "top_p": 0.95}}}).encode())
        elif self.path == "/metrics" and m.flavour in ("dspark", "llama"):
            with m.lock:
                m.metrics_reads += 1
                if m.metrics_reads > 1:
                    m.requests += m.extra_requests
                    m.tokens += 50 * m.extra_requests
                if m.flavour == "dspark":
                    self._send(200, json.dumps({"requests": m.requests,
                                                "completion_tokens": m.tokens}).encode())
                else:
                    text = (f"# HELP x\nllamacpp:tokens_predicted_total {m.tokens}\n"
                            f"llamacpp:n_decode_total {m.tokens}\n")
                    self._send(200, text.encode(), "text/plain")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        m = self.mock
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions":
            self._send(404, b"{}")
            return
        with m.lock:
            index = len(m.bodies)
            m.bodies.append(body)
            m.requests += 1
            status, d = m.completion(index, body)
        self._send(status, json.dumps(d).encode())


@contextlib.contextmanager
def serve(mock: Mock) -> Iterator[str]:
    handler = type("BoundHandler", (Handler,), {"mock": mock})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def quiet_run(client: bench5.Client, reps: int, mode: bench5.Mode = None) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return bench5.run_arm(client, "test arm", reps, mode)


def keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from keys(v)


def walk(value: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from walk(v, f"{path}[{i}]")
    else:
        yield path, value


class DerivationTests(unittest.TestCase):
    def test_prompts_are_the_published_ones(self) -> None:
        digest = hashlib.sha256("\n".join(bench5.PROMPTS).encode()).hexdigest()
        self.assertEqual(digest, PROMPTS_SHA256)
        self.assertEqual(bench5.MAX_TOKENS, 400)

    def test_llama_timings_give_predicted_ms_over_1000(self) -> None:
        spec = bench5.spec_from_timings({"predicted_ms": 2500.0, "predicted_n": 120,
                                         "prompt_ms": 300.0, "prompt_n": 40,
                                         "draft_n": 90, "draft_n_accepted": 60})
        t = bench5.Turn(wall=3.0, completion_tokens=120, spec=spec)
        self.assertAlmostEqual(t.decode_seconds, 2.5)
        self.assertAlmostEqual(t.ttft, 0.3)
        self.assertAlmostEqual(t.prefill_seconds, 0.3)
        self.assertIsNone(t.rounds)            # predicted_n counts tokens, not rounds
        self.assertEqual((t.tokens_per_round, t.round_ms), (0.0, 0.0))
        self.assertIsNone(spec["cap"])
        self.assertEqual((spec["draft_n"], spec["draft_n_accepted"]), (90, 60))
        self.assertAlmostEqual(t.decode_tps, 48.0)
        self.assertEqual(bench5.spec_from_timings(None), {})
        self.assertNotIn("draft_n", bench5.spec_from_timings({"predicted_ms": 1.0}))

    def test_dspark_block_is_read_for_decode_seconds_not_its_rate(self) -> None:
        d = {"x_mlx_dspark": {"decode_seconds": 4.0, "decode_tokens_per_sec": 999.0,
                              "accept_len": 2.5, "cap": 7},
             "timings": {"predicted_ms": 1.0}}
        spec = bench5.spec_from_response(d)
        t = bench5.Turn(wall=5.0, completion_tokens=200, spec=spec)
        self.assertAlmostEqual(t.decode_seconds, 4.0)
        self.assertAlmostEqual(t.decode_tps, 50.0)

    def test_no_block_means_no_decode_timing_not_the_wall_clock(self) -> None:
        t = bench5.Turn(wall=5.0, completion_tokens=200, spec=bench5.spec_from_response({}))
        self.assertEqual(t.decode_seconds, 0.0)
        self.assertEqual(t.decode_tps, 0.0)
        self.assertAlmostEqual(t.e2e_tps, 40.0)

    def test_pooling_is_total_tokens_over_total_seconds(self) -> None:
        turns = [bench5.Turn(wall=3.0, completion_tokens=100, spec={"decode_seconds": 2.0}),
                 bench5.Turn(wall=5.0, completion_tokens=300, spec={"decode_seconds": 4.0}),
                 bench5.Turn(wall=2.0, completion_tokens=50, spec={})]
        rate, tok, sec = bench5.pooled(turns)
        self.assertEqual((tok, sec), (400, 6.0))       # the untimed turn is out on both sides
        self.assertAlmostEqual(rate, 400 / 6)
        self.assertNotAlmostEqual(rate, (50 + 75) / 2)  # not the mean of per-request rates
        _, e2e_tok, wall = bench5.pooled(turns, "wall")
        self.assertEqual((e2e_tok, wall), (450, 10.0))
        self.assertEqual(bench5.pooled([]), (0.0, 0, 0.0))

    def test_sampling_modes(self) -> None:
        self.assertEqual(bench5.sampling(None), {"temperature": 0})
        self.assertEqual(bench5.sampling("greedy"), {"temperature": 0})
        self.assertEqual(bench5.sampling("default"), {})
        self.assertEqual(bench5.sampling("1.0"), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})

    def test_server_root_strips_v1(self) -> None:
        self.assertEqual(bench5.server_root("http://h:1/v1/"), "http://h:1")
        self.assertEqual(bench5.server_root("http://h:1"), "http://h:1")


class ServedTests(unittest.TestCase):
    def test_dspark_arm_end_to_end(self) -> None:
        mock = Mock("dspark")
        with serve(mock) as url:
            record = quiet_run(bench5.Client(url, "k"), reps=2)
        # One warm-up, then 2 reps x 5 prompts; the warm-up is in no row and no pool.
        self.assertEqual(len(mock.bodies), 11)
        self.assertEqual(mock.bodies[0]["messages"][0]["content"], "hi")
        self.assertEqual(mock.bodies[0]["max_tokens"], 8)
        self.assertTrue(all(b["max_tokens"] == 400 for b in mock.bodies[1:]))
        self.assertEqual([b["messages"][0]["content"] for b in mock.bodies[1:6]], bench5.PROMPTS)
        rows = record["requests"]
        self.assertEqual([(r["rep"], r["prompt"]) for r in rows],
                         [(rep, p) for rep in (1, 2) for p in range(1, 6)])
        tokens = [100 + 10 * (i % 5) for i in range(1, 11)]
        self.assertEqual([r["completion_tokens"] for r in rows], tokens)
        self.assertEqual(record["pooled_decode_tokens"], sum(tokens))
        self.assertAlmostEqual(record["pooled_decode_seconds"], sum(tokens) / 50)
        self.assertAlmostEqual(record["pooled_decode_tps"], 50.0)
        self.assertEqual(len(record["per_rep"]), 2)
        rep1 = record["per_rep"][0]
        self.assertEqual(rep1["completion_tokens"], sum(tokens[:5]))
        self.assertAlmostEqual(rep1["decode_tps"], 50.0)
        self.assertEqual(rows[0]["spec"]["cap"], 7)
        self.assertEqual(rows[0]["spec"]["accept_len"], 2.5)
        self.assertEqual(record["server"]["model"], "target")      # basename, never the path
        self.assertIs(record["clean"], True)
        self.assertEqual(record["sampling"], {"temperature": 0})

    def test_llama_arm_end_to_end(self) -> None:
        mock = Mock("llama")
        with serve(mock) as url:
            record = quiet_run(bench5.Client(url), reps=1, mode="default")
        self.assertTrue(all("temperature" not in b and "top_p" not in b and "top_k" not in b
                            for b in mock.bodies))
        row = record["requests"][0]
        self.assertAlmostEqual(row["decode_seconds"], row["completion_tokens"] / 50)
        self.assertAlmostEqual(row["ttft_seconds"], 0.2)
        self.assertEqual(row["spec"]["draft_n"], row["completion_tokens"])
        self.assertEqual(row["spec"]["draft_n_accepted"], row["completion_tokens"] // 2)
        self.assertEqual(record["server"]["engine"], "llama.cpp")
        self.assertEqual(record["server"]["model"], "bonsai.gguf")
        self.assertIs(record["clean"], True)            # tokens compared on llama.cpp
        self.assertIn("tokens generated", record["contamination"])

    def test_rounds_are_null_on_llama_cpp_and_counted_on_mlx_dspark(self) -> None:
        # llama.cpp's predicted_n is a token count once a drafter commits several tokens a
        # round, so it must not reach `rounds` or any per-round figure the runner prints.
        records, printed = {}, {}
        for flavour in ("llama", "dspark"):
            out = io.StringIO()
            with serve(Mock(flavour)) as url, contextlib.redirect_stdout(out):
                records[flavour] = bench5.run_arm(bench5.Client(url), "test arm", 1, None)
            printed[flavour] = out.getvalue()
        self.assertEqual([r["rounds"] for r in records["llama"]["requests"]], [None] * 5)
        self.assertEqual([r["rounds"] for r in records["dspark"]["requests"]],
                         [(100 + 10 * (i % 5)) // 2 for i in range(1, 6)])
        self.assertNotIn("tok/round", printed["llama"])
        self.assertNotIn("tokens/round", printed["llama"])
        self.assertIn("tok/round", printed["dspark"])
        self.assertIn("tokens/round 2.00", printed["dspark"])
        written = json.loads(json.dumps(records["llama"]))
        self.assertIsNone(written["requests"][0]["rounds"])     # null in the JSON record

    def test_another_client_is_reported(self) -> None:
        for flavour in ("dspark", "llama"):
            mock = Mock(flavour)
            mock.extra_requests = 3
            with serve(mock) as url:
                record = quiet_run(bench5.Client(url), reps=1)
            self.assertIs(record["clean"], False, flavour)
            self.assertIn("CONTAMINATED", record["contamination"])

    def test_no_metrics_is_unavailable_not_clean(self) -> None:
        mock = Mock("plain")
        with serve(mock) as url:
            record = quiet_run(bench5.Client(url), reps=1)
        self.assertIsNone(record["clean"])
        self.assertEqual(record["no_decode_timing_requests"], 5)
        self.assertIsNone(record["pooled_decode_tps"])
        self.assertGreater(record["pooled_e2e_tps"], 0)

    def test_failed_and_usage_less_requests_stay_out_of_the_pool(self) -> None:
        mock = Mock("dspark")
        mock.fail = {2: 500}
        mock.no_usage = {4}
        with serve(mock) as url:
            record = quiet_run(bench5.Client(url), reps=1)
        self.assertEqual([r["prompt"] for r in record["requests"]], [1, 3, 5])
        errors = {f["prompt"]: f["error"] for f in record["failures"]}
        self.assertTrue(errors[2].startswith("HTTP 500"))
        self.assertIn("usage", errors[4])
        tokens = [110, 130, 100]      # indices 1, 3, 5: 100 + 10 * (index % 5)
        self.assertEqual(record["pooled_decode_tokens"], sum(tokens))

    def test_warmup_failure_is_recorded_and_excluded(self) -> None:
        mock = Mock("dspark")
        mock.fail = {0: 503}
        with serve(mock) as url:
            record = quiet_run(bench5.Client(url), reps=1)
        self.assertTrue(record["warmup"]["error"].startswith("HTTP 503"))
        self.assertEqual(len(record["requests"]), 5)
        self.assertEqual(record["failures"], [])

    def test_main_writes_the_default_name_and_fails_when_nothing_answers(self) -> None:
        mock = Mock("dspark")
        with serve(mock) as url, tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = bench5.main(["--base-url", url, "--label", "Stock A1", "--reps", "1"])
                self.assertEqual(rc, 0)
                self.assertEqual(os.listdir(tmp), ["bench5-stock-a1.json"])
                mock.fail = {i: 500 for i in range(len(mock.bodies), len(mock.bodies) + 6)}
                out = os.path.join(tmp, "all-failed.json")
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    rc = bench5.main(["--base-url", url, "--reps", "1", "--out", out])
                self.assertEqual(rc, 1)
            finally:
                os.chdir(cwd)

    def test_record_has_no_timestamps_hosts_or_paths(self) -> None:
        for flavour in ("dspark", "llama"):
            mock = Mock(flavour)
            with serve(mock) as url:
                record = quiet_run(bench5.Client(url, "secret-key"), reps=1, mode="1.0")
            text = json.dumps(record)
            self.assertNotIn("secret-key", text)
            self.assertNotIn("/secret/", text)
            self.assertNotIn("127.0.0.1", text)
            for where, value in walk(record):
                if isinstance(value, str):
                    self.assertIsNone(DATE_LIKE.search(value), f"{flavour} {where}={value!r}")
                    self.assertFalse(value.startswith("/"), f"{flavour} {where}={value!r}")
            self.assertEqual([k for k in keys(record) if TIME_KEY.search(k)], [], flavour)


if __name__ == "__main__":
    unittest.main()
