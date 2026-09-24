#!/usr/bin/env python3
"""Five-prompt throughput against any OpenAI-compatible server, standard library only.

One prompt is not enough whenever two arms compute different logits: they diverge into
different text, and both round time and acceptance follow the text. This asks five distinct
code-generation prompts at chat depth and pools them, so a single lucky or unlucky
continuation cannot carry the result. `BENCHMARK.md` names this runner for its throughput
measurements; `bench/throughput/README.md` says how to run an ABBA group with it.

    python3 bench/throughput/bench5.py --base-url http://127.0.0.1:8088/v1 --label stock
    python3 bench/throughput/bench5.py --base-url ... --label ft5 --temperature 1.0 --reps 5
    python3 bench/throughput/bench5.py --base-url ... --label srv --temperature default

What it reports, and why there are two of most things:

    decode    completion tokens over the server's own decode-only seconds. This is the number
              an arm is compared on. mlx-dspark reports it in the `x_mlx_dspark` block it
              attaches to every non-streaming completion (`decode_seconds`); llama.cpp in its
              `timings` object (`predicted_ms` / 1000). A server that reports neither gives no
              decode rate: its requests carry tokens but no seconds, and are left out of the
              pool on both sides instead of being timed by the wall clock.
    e2e       completion tokens over the whole request as the client saw it, prefill and HTTP
              included. A request rate, not a decode rate; it moves when prefill moves.
    POOLED    total tokens over total seconds, per arm. The mean of per-rep rates weights a
              short answer the same as a long one; it is printed beside the pool as
              `mean of reps`, never instead of it.

Requests are not streamed, so TTFT is the server's figure (`ttft_seconds`, or llama.cpp's
`prompt_ms`), not a client-side first-byte time.

Deliberately not representative of an agent: short prompts, 400 tokens, no tools, no reasoning
effort. That is what makes it a stable unit across runtimes and drafters.

The record holds no timestamps, host names, user names or absolute paths, so it can be
published as written.
"""
from __future__ import annotations

import argparse
import http.client
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Copied verbatim from the runner the published numbers were taken with. Identical prompts are
# what make two records comparable; change one and it is a different benchmark.
PROMPTS = [
    "Write a Python function that parses an ISO-8601 duration string like P3DT4H5M into a "
    "timedelta. Include a docstring and handle invalid input.",
    "Write a Python function that merges two sorted lists into one sorted list, without "
    "using sorted(). Include a docstring.",
    "Write a Python context manager that times the block it wraps and logs the elapsed "
    "milliseconds. Include a docstring.",
    "Write a Python function that flattens an arbitrarily nested list of integers "
    "iteratively rather than recursively. Include a docstring.",
    "Write a Python function that returns the n most common words in a string, ties broken "
    "alphabetically. Include a docstring.",
]
MAX_TOKENS = 400
WARMUP_MESSAGES = [{"role": "user", "content": "hi"}]
WARMUP_MAX_TOKENS = 8

# Sent with an explicit temperature, so a sampled arm is the target's published sampling
# rather than temperature with everything else left to the server.
TOP_P, TOP_K = 0.95, 20

Mode = str | float | None


# --------------------------------------------------------------------------- sampling

def sampling(mode: Mode) -> dict[str, Any]:
    """The three modes an arm is run in.

    ``None``/``"greedy"``  temperature 0, so two reps of one arm produce the same text
    ``"default"``          no sampling parameters at all, so the server's launch flags apply
    a number               that temperature, with top-p 0.95 and top-k 20
    """
    if mode == "default":
        return {}
    if mode in (None, "", "greedy"):
        return {"temperature": 0}
    assert mode is not None
    return {"temperature": float(mode), "top_p": TOP_P, "top_k": TOP_K}


def mode_label(mode: Mode) -> str:
    if mode == "default":
        return "server default (nothing sent)"
    if mode in (None, "", "greedy"):
        return "greedy"
    return f"temperature {mode}, top_p {TOP_P}, top_k {TOP_K}"


def parse_mode(text: str) -> Mode:
    """``--temperature`` as given: ``greedy``, ``default`` or a number."""
    if text in ("greedy", "default"):
        return text
    try:
        float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected 'greedy', 'default' or a number, got {text!r}") from None
    return text


# --------------------------------------------------------------------------- server timing

# Keys that would carry a wall-clock time or a date if a server added them to its block. The
# record is published as written, so they are dropped rather than trusted to be absent.
_TIMESTAMP_KEY = re.compile(r"(^|_)(created|date|time|when)(_|$)|stamp", re.IGNORECASE)


def _scrub(value: Any) -> Any:
    """A server block, minus timestamp-like keys, with any absolute path cut to its last part."""
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if not _TIMESTAMP_KEY.search(str(k))}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str) and value.startswith(("/", "~")):
        return value.rstrip("/").rsplit("/", 1)[-1]
    return value


def spec_from_timings(t: dict[str, Any] | None) -> dict[str, Any]:
    """llama-server's `timings` object, in the shape of mlx-dspark's `x_mlx_dspark` block, so a
    request's decode and prefill are read the same way from both servers.

    `decode_seconds` is `predicted_ms / 1000`: llama.cpp's own timer around the generation
    phase, which excludes prompt processing. `ttft_seconds` and `prefill_seconds` are both
    `prompt_ms / 1000`. `target_forwards` is `predicted_n` as reported: the forward count
    without a drafter, a token count with one, so it is never read as rounds and a llama.cpp
    request records `rounds` as None. Read `draft_n` and `draft_n_accepted` instead, which are
    copied through when the server reports them. `cap` is None: llama.cpp does not report a
    verify cap."""
    if not t:
        return {}
    spec: dict[str, Any] = {
        "engine": "llama.cpp",
        "decode_seconds": t.get("predicted_ms", 0.0) / 1000,
        "prefill_seconds": t.get("prompt_ms", 0.0) / 1000,
        "ttft_seconds": t.get("prompt_ms", 0.0) / 1000,
        "target_forwards": int(t.get("predicted_n", 0)),
        "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "cap": None}
    for key in ("predicted_n", "draft_n", "draft_n_accepted"):
        if key in t:
            spec[key] = t[key]
    return spec


def spec_from_response(d: dict[str, Any]) -> dict[str, Any]:
    """The server's decode-side block: mlx-dspark's `x_mlx_dspark` as sent, else llama.cpp's
    `timings` translated, else ``{}`` (a server that reports no decode timing)."""
    block = d.get("x_mlx_dspark")
    if isinstance(block, dict) and block:
        return dict(block)
    timings = d.get("timings")
    return spec_from_timings(timings if isinstance(timings, dict) else None)


@dataclass
class Turn:
    """One completion, with the client's view and the server's view kept apart."""
    wall: float                       # what the client waited: prefill + decode + HTTP
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    spec: dict[str, Any] = field(default_factory=dict)   # the server's block, {} if absent
    error: str = ""                   # set instead of the above when the request failed

    # The server's own numbers. All fall back to 0.0 rather than to the wall clock, so a
    # missing block reads as missing instead of quietly becoming an end-to-end number.
    @property
    def decode_seconds(self) -> float:
        return float(self.spec.get("decode_seconds") or 0.0)

    @property
    def prefill_seconds(self) -> float:
        return float(self.spec.get("prefill_seconds") or 0.0)

    @property
    def ttft(self) -> float:
        return float(self.spec.get("ttft_seconds") or 0.0)

    @property
    def rounds(self) -> int | None:
        """Target forwards, from mlx-dspark's block. None on llama.cpp: its only count is
        `predicted_n`, which is tokens, and one round commits several tokens with a drafter."""
        if self.spec.get("engine") == "llama.cpp":
            return None
        return int(self.spec.get("target_forwards") or 0)

    @property
    def decode_tps(self) -> float:
        return self.completion_tokens / self.decode_seconds if self.decode_seconds else 0.0

    @property
    def e2e_tps(self) -> float:
        return self.completion_tokens / self.wall if self.wall else 0.0

    @property
    def tokens_per_round(self) -> float:
        rounds = self.rounds
        return self.completion_tokens / rounds if rounds else 0.0

    @property
    def round_ms(self) -> float:
        rounds = self.rounds
        return self.decode_seconds / rounds * 1000 if rounds else 0.0

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


# --------------------------------------------------------------------------- transport

def server_root(base_url: str) -> str:
    """``http://host:port/v1`` -> ``http://host:port``: where /health, /props and /metrics
    live on both llama-server and mlx-dspark."""
    root = base_url.rstrip("/")
    return root.removesuffix("/v1")


class Client:
    """Counts its own requests and tokens, so the server-wide counters can be checked."""

    def __init__(self, base_url: str, api_key: str = "", model: str = "local",
                 timeout: float = 3600) -> None:
        self.base_url = base_url.rstrip("/")
        self.root = server_root(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.requests = 0
        self.completion_tokens = 0
        self._engine: str | None = None

    def _raw(self, url: str, body: dict[str, Any] | None = None,
             timeout: float | None = None) -> bytes:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(
            url, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            data: bytes = r.read()
            return data

    def get(self, path: str, timeout: float = 10) -> bytes:
        """A GET against the server root (``/health``, ``/props``, ``/metrics``)."""
        return self._raw(self.root + path, timeout=timeout)

    def engine(self) -> str:
        """``"llama"`` if the server answers /props with a model path (only llama-server
        does), else ``"dspark"``. Decided once; only the counter check depends on it."""
        if self._engine is None:
            try:
                self._engine = "llama" if b"model_path" in self.get("/props") else "dspark"
            except Exception:  # noqa: BLE001 - no /props: not llama-server
                self._engine = "dspark"
        return self._engine

    def chat(self, messages: list[dict[str, Any]], max_tokens: int = MAX_TOKENS,
             mode: Mode = None) -> Turn:
        body: dict[str, Any] = {"model": self.model, "messages": messages,
                                "max_tokens": max_tokens}
        body.update(sampling(mode))
        self.requests += 1
        t0 = time.perf_counter()
        try:
            raw = self._raw(self.base_url + "/chat/completions", body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300].replace("\n", " ")
            return Turn(wall=time.perf_counter() - t0, error=f"HTTP {e.code} {detail}")
        except urllib.error.URLError as e:
            return Turn(wall=time.perf_counter() - t0, error=f"URLError {e.reason}")
        except (TimeoutError, OSError, http.client.HTTPException) as e:
            return Turn(wall=time.perf_counter() - t0, error=f"{type(e).__name__} {e}")
        wall = time.perf_counter() - t0
        try:
            d = json.loads(raw)
        except ValueError:
            return Turn(wall=wall, error="response is not JSON")
        if not isinstance(d, dict):
            return Turn(wall=wall, error="response is not a JSON object")
        u = d.get("usage")
        # Without a completion count the request would add seconds and no tokens to the pool.
        if not isinstance(u, dict) or not isinstance(u.get("completion_tokens"), int):
            return Turn(wall=wall, error="response has no usage.completion_tokens")
        ch = (d.get("choices") or [{}])[0]
        self.completion_tokens += u["completion_tokens"]
        return Turn(wall=wall, prompt_tokens=int(u.get("prompt_tokens") or 0),
                    completion_tokens=u["completion_tokens"],
                    finish_reason=str(ch.get("finish_reason") or ""),
                    spec=_scrub(spec_from_response(d)))


def require_server(client: Client) -> None:
    """Exit unless something answers: /health (llama-server, mlx-dspark), else /v1/models."""
    errors = []
    for url in (client.root + "/health", client.base_url + "/models"):
        try:
            client._raw(url, timeout=10)
            return
        except Exception as e:  # noqa: BLE001 - any failure to answer: try the next probe
            errors.append(f"{url.rsplit('/', 1)[-1]}: {e}")
    sys.exit(f"no server answering at {client.base_url} ({'; '.join(errors)})")


# --------------------------------------------------------------------------- pooling

def pooled(turns: list[Turn], attr: str = "decode_tps") -> tuple[float, int, float]:
    """Total tokens over total seconds: the pooled rate.

    For the decode rate, a request without decode timing is dropped from both sides; left in,
    it would add tokens and no seconds and inflate the rate silently."""
    if attr == "decode_tps":
        turns = [t for t in turns if t.decode_seconds > 0]
    tok = sum(t.completion_tokens for t in turns)
    sec = sum((t.decode_seconds if attr == "decode_tps" else t.wall) for t in turns)
    return (tok / sec if sec else 0.0), tok, sec


def spread(values: list[float]) -> tuple[float, float, float]:
    """min, max, and the percentage between them."""
    if not values:
        return 0.0, 0.0, 0.0
    lo, hi = min(values), max(values)
    return lo, hi, (hi - lo) / lo * 100 if lo else 0.0


# --------------------------------------------------------------------------- contamination

def _prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            k, _, v = line.partition(" ")
            try:
                out[k[len("llamacpp:"):]] = float(v)
            except ValueError:
                pass
    return out


class Contamination:
    """Server-wide counters, read at both ends, checked against the client's own count.

    mlx-dspark's JSON /metrics counts requests; llama.cpp's Prometheus /metrics (served only
    with `--metrics`) counts generated tokens, so the check compares tokens there. An arm
    measured while another client was served is not this arm's alone, and a counter that goes
    backwards means the server restarted mid-arm. A server with neither says so."""

    def __init__(self, client: Client) -> None:
        self.client = client
        self.before = self._snap()

    def _snap(self) -> dict[str, Any]:
        try:
            raw = self.client.get("/metrics")
            if self.client.engine() == "llama":
                m = _prometheus(raw.decode(errors="replace"))
                if "tokens_predicted_total" not in m:
                    return {}
                return {"requests": None, "completion_tokens": int(m["tokens_predicted_total"])}
            d = json.loads(raw)
            if not isinstance(d, dict) or not isinstance(d.get("requests"), int):
                return {}
            return {"requests": d["requests"],
                    "completion_tokens": d.get("completion_tokens", 0)}
        except Exception:  # noqa: BLE001 - an unreadable /metrics is reported, not raised
            return {}

    def check(self) -> tuple[bool | None, str]:
        """(clean, message); clean is None when the server has no usable counters."""
        after = self._snap()
        if not self.before or not after:
            return None, "contamination check unavailable: /metrics gave no usable counter"
        if self.before.get("requests") is None:
            served = after["completion_tokens"] - self.before["completion_tokens"]
            mine = self.client.completion_tokens
            unit_s, unit_m = "tokens generated", "received by this arm"
        else:
            served = after["requests"] - self.before["requests"]
            mine = self.client.requests
            unit_s, unit_m = "requests served", "issued by this arm"
        if served > mine:
            return False, (f"CONTAMINATED: {served} {unit_s}, {mine} {unit_m}. Another client "
                           f"was active; these figures are not this arm's alone. Rerun.")
        if served < mine:
            return False, (f"COUNTERS WENT BACKWARDS: {served} {unit_s}, {mine} {unit_m}. The "
                           f"server restarted mid-arm; discard this run.")
        return True, f"clean: {served} {unit_s}, {mine} {unit_m}"


# --------------------------------------------------------------------------- server identity

def server_identity(client: Client) -> dict[str, Any]:
    """What the server says it resolved, from an allowlist of fields that identify the
    configuration and not the machine: the model file's name (never its path), the build,
    the context window, the verify cap. Best effort; missing fields are left out."""
    out: dict[str, Any] = {"engine": "llama.cpp" if client.engine() == "llama" else "unknown"}
    try:
        if client.engine() == "llama":
            props = json.loads(client.get("/props"))
            gs = props.get("default_generation_settings") or {}
            params = gs.get("params") or {}
            out.update({"model": props.get("model_path"), "build": props.get("build_info"),
                        "context_window": gs.get("n_ctx"), "slots": props.get("total_slots"),
                        "sampling": {k: params.get(k)
                                     for k in ("temperature", "top_p", "top_k", "min_p")}})
        else:
            health = json.loads(client.get("/health"))
            if isinstance(health, dict):
                if "max_draft" in health:      # the derived verify cap only mlx-dspark reports
                    out["engine"] = "mlx-dspark"
                out.update({k: health[k] for k in ("model", "max_draft", "kv_bits",
                                                   "context_window", "small_m")
                            if k in health})
    except Exception as e:  # noqa: BLE001 - identity is context, never a reason to abort
        out["error"] = type(e).__name__
    return _scrub(out)


# --------------------------------------------------------------------------- the arm

def request_row(rep: int, prompt: int, t: Turn) -> dict[str, Any]:
    return {"rep": rep, "prompt": prompt, "completion_tokens": t.completion_tokens,
            "decode_seconds": t.decode_seconds, "e2e_seconds": t.wall,
            "ttft_seconds": t.ttft, "prefill_seconds": t.prefill_seconds,
            "rounds": t.rounds, "finish_reason": t.finish_reason, "spec": t.spec}


def _spec_note(t: Turn) -> str:
    s = t.spec
    if "draft_n" in s:
        n, acc = s.get("draft_n") or 0, s.get("draft_n_accepted") or 0
        return f"drafted {n} accepted {acc}" + (f" ({acc / n:.0%})" if n else "")
    note = f"accept {s['accept_len']}  cap {s.get('cap')}  " if "accept_len" in s else ""
    if t.rounds:
        note += f"{t.tokens_per_round:4.2f} tok/round  round {t.round_ms:5.1f} ms"
    return note


def run_arm(client: Client, label: str, reps: int, mode: Mode,
            prompts: list[str] | None = None) -> dict[str, Any]:
    """Warm up once (discarded), then ``reps`` passes over the prompts. Returns the record."""
    prompts = PROMPTS if prompts is None else prompts
    print(f"== {label}  ({mode_label(mode)}, {reps} reps x {len(prompts)} prompts)")
    identity = server_identity(client)
    print("   server " + "  ".join(f"{k} {v}" for k, v in identity.items() if v is not None))
    watch = Contamination(client)

    warm = client.chat(WARMUP_MESSAGES, max_tokens=WARMUP_MAX_TOKENS, mode=mode)
    if warm.error:
        print(f"   warm-up FAILED  {warm.error}")

    all_turns: list[Turn] = []
    per_rep: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        turns: list[Turn] = []
        for i, p in enumerate(prompts, 1):
            t = client.chat([{"role": "user", "content": p}], max_tokens=MAX_TOKENS, mode=mode)
            if t.error:
                print(f"   rep {rep} prompt {i}: FAILED  {t.error}")
                failures.append({"rep": rep, "prompt": i, "error": t.error})
                continue
            turns.append(t)
            rows.append(request_row(rep, i, t))
            print(f"   rep {rep} prompt {i}: {t.completion_tokens:>4} tok  "
                  f"e2e {t.e2e_tps:>5.1f}  decode {t.decode_tps:>5.1f} tok/s  "
                  f"ttft {t.ttft * 1000:>6.0f} ms  {_spec_note(t)}"
                  + ("  TRUNCATED" if t.truncated else "")
                  + ("" if t.decode_seconds else "  NO DECODE TIMING"))
        if not turns:
            continue
        all_turns += turns
        rate, tok, sec = pooled(turns)
        e2e, e2e_tok, wall = pooled(turns, "wall")
        per_rep.append({"rep": rep, "completion_tokens": tok, "decode_seconds": sec,
                        "decode_tps": rate, "e2e_tokens": e2e_tok, "e2e_seconds": wall,
                        "e2e_tps": e2e})
        print(f"   rep {rep}: {tok} tok  decode {sec:6.1f}s -> {rate:5.2f} tok/s   "
              f"wall {wall:6.1f}s -> {e2e:5.2f} tok/s e2e")

    clean, contamination = watch.check()
    record: dict[str, Any] = {
        "runner": "bench5", "label": label, "mode": mode_label(mode),
        "sampling": sampling(mode), "reps": reps, "prompts": len(prompts),
        "max_tokens": MAX_TOKENS, "server": identity,
        "warmup": {"discarded": True, "error": warm.error or None},
        "requests": rows, "failures": failures, "per_rep": per_rep,
        "per_rep_decode_tps": [r["decode_tps"] for r in per_rep],
        "clean": clean, "contamination": contamination,
    }
    if not all_turns:
        record.update({"pooled_decode_tps": None, "pooled_e2e_tps": None})
        return record

    rate, tok, sec = pooled(all_turns)
    e2e, e2e_tok, wall = pooled(all_turns, "wall")
    rep_rates = [r["decode_tps"] for r in per_rep if r["decode_seconds"] > 0]
    lo, hi, pct = spread(rep_rates)
    mean = sum(rep_rates) / len(rep_rates) if rep_rates else 0.0
    untimed = sum(1 for t in all_turns if not t.decode_seconds)
    trunc = sum(t.truncated for t in all_turns)
    record.update({
        "pooled_decode_tps": rate if sec else None, "pooled_decode_tokens": tok,
        "pooled_decode_seconds": sec, "pooled_e2e_tps": e2e, "pooled_e2e_tokens": e2e_tok,
        "pooled_e2e_seconds": wall, "mean_of_reps_decode_tps": mean,
        "no_decode_timing_requests": untimed, "truncated_requests": trunc,
    })
    print(f"   POOLED **{rate:5.2f} tok/s decode**   {e2e:5.2f} tok/s e2e   "
          f"mean of reps {mean:5.2f}   spread {lo:.2f}-{hi:.2f} ({pct:.1f}%)")
    # Rounds are target forwards on mlx-dspark only; llama.cpp reports none (rounds is None).
    # A per-round figure needs a round count for every request in the decode pool.
    timed = [t for t in all_turns if t.decode_seconds > 0]
    counted = [r for r in (t.rounds for t in timed) if r]
    rounds = sum(counted) if counted and len(counted) == len(timed) else 0
    print(f"   {tok} tokens over {sec:.1f}s decode / {wall:.1f}s wall"
          + (f"   tokens/round {tok / rounds:4.2f}   round {sec / rounds * 1000:5.1f} ms"
             if rounds else ""))
    if untimed:
        print(f"   !! {untimed}/{len(all_turns)} responses carried no decode timing "
              f"(no x_mlx_dspark block, no timings object); they are out of the decode pool.")
    # A truncated answer is a timed truncation, not a timed answer.
    if trunc:
        print(f"   !! {trunc}/{len(all_turns)} answers stopped at the token cap (finish_reason "
              f"'length'). Those measured a truncated generation, not a completed one.")
    if failures:
        print(f"   !! {len(failures)} requests failed and are not in the pool.")
    print(("   " if clean else "   !! ") + contamination)
    return record


def default_out(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "arm"
    return f"bench5-{slug}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0] if __doc__ else None)
    ap.add_argument("--base-url", required=True,
                    help="the OpenAI-compatible API root, e.g. http://127.0.0.1:8088/v1")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""),
                    help="bearer key (default: $BENCH_API_KEY; none sent if empty)")
    ap.add_argument("--label", default="arm", help="names the arm in the output and record")
    ap.add_argument("--reps", type=int, default=3, help="passes over the five prompts")
    ap.add_argument("--temperature", type=parse_mode, default="greedy",
                    help="'greedy' (temperature 0, the default), 'default' (send no sampling, "
                         "so the server's flags apply) or a number (with top-p 0.95, top-k 20)")
    ap.add_argument("--model", default="local", help="the model field sent (default: local)")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    ap.add_argument("--out", help="record path (default: ./bench5-<label>.json)")
    args = ap.parse_args(argv)
    if args.reps < 1:
        ap.error("--reps must be at least 1")

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    client = Client(args.base_url, args.api_key, args.model, args.timeout)
    require_server(client)
    record = run_arm(client, args.label, args.reps, args.temperature)
    out = args.out or default_out(args.label)
    with open(out, "w") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    print(f"   record written to {out}")
    if record["pooled_decode_tps"] is None and not record["requests"]:
        print("every request failed; nothing to pool", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
