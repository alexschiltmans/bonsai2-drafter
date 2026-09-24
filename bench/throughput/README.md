# Throughput runner

`bench5.py` is the runner `BENCHMARK.md` names for its throughput measurements. It needs only
the Python standard library and talks to any OpenAI-compatible server by URL:

```
python3 bench/throughput/bench5.py --base-url http://127.0.0.1:8088/v1 --label stock-a1
```

It sends one warm-up request (`hi`, 8 tokens), which is discarded, then `--reps` passes (default
3) over five fixed code-generation prompts at `max_tokens` 400. The prompts must not change:
identical prompts are what make two records comparable, and the test suite pins their hash.

- `--temperature greedy` (the default) sends `temperature: 0`.
- `--temperature 1.0` sends that temperature with `top_p: 0.95` and `top_k: 20`, the target's
  published sampling.
- `--temperature default` sends no sampling parameters, so the server's launch flags apply.
- `--api-key` or `BENCH_API_KEY` sets the bearer key; `--out` the record path (default
  `./bench5-<label>.json`).

## Where decode time comes from

Requests are not streamed. Each rate is completion tokens (`usage.completion_tokens`) over
seconds, and the decode rate uses the server's own decode-only timer, never the wall clock:

| Server | Response field | `decode_seconds` | TTFT and prefill |
|---|---|---|---|
| mlx-dspark | `x_mlx_dspark` block | its `decode_seconds`, as sent | its `ttft_seconds`, `prefill_seconds` |
| llama.cpp | `timings` object | `predicted_ms / 1000` | `prompt_ms / 1000` |
| anything else | none | 0: no decode timing | 0 |

The `x_mlx_dspark` block is recorded whole, so `accept_len`, `cap`, `target_forwards` and the
server's own `decode_tokens_per_sec` sit beside each request, but only `decode_seconds` enters
the pool. From llama.cpp's `timings`, `predicted_n`, `prompt_n`, `cache_n`, and, when a drafter
is loaded, `draft_n` and `draft_n_accepted` are recorded. Acceptance on llama.cpp is
`draft_n_accepted / draft_n`. `target_forwards` there is `predicted_n`, which counts forwards
only when no drafter is loaded, and `cap` is null.

A response without decode timing adds nothing to the decode pool, on either side: its tokens
without its seconds would inflate the rate. The run says how many responses that was. A
failed request (a non-200, a transport error, a body that is not JSON, or no
`usage.completion_tokens`) is listed under `failures` and is in neither pool.

The pooled decode rate is total completion tokens over total decode seconds for the arm. The
per-rep rates and their mean are printed and recorded beside it, never instead of it.

## The record

Per request: rep, prompt, `completion_tokens`, `decode_seconds`, `e2e_seconds`,
`ttft_seconds`, `prefill_seconds`, `rounds`, `finish_reason` and the server block. Per rep:
tokens, decode seconds and rate, and the same end to end. Per arm: `pooled_decode_tps`,
`pooled_decode_tokens`, `pooled_decode_seconds`, the e2e pool, `mean_of_reps_decode_tps`, the
truncation count (answers that hit the 400-token cap), and the server identity (engine, model
file name, build, context window) from `/props` or `/health`.

It carries no timestamps, host names, user names, keys or absolute paths. Timestamp-like keys
in a server block are dropped and a path is cut to its file name, so the record can be
published as written. Date and host belong in the commit that publishes it.

## Counter check

When the server exposes counters, the runner reads them before the warm-up and after the last
request and compares them with its own count. mlx-dspark's JSON `/metrics` counts requests;
llama.cpp's Prometheus `/metrics` (only with `--metrics`) counts generated tokens
(`llamacpp:tokens_predicted_total`), so the check compares tokens there. More than the runner
sent means another client was served during the arm (`CONTAMINATED`), and fewer means the
server restarted mid-arm. Either way the arm is rerun. The result is recorded as `clean`
(`true`, `false`, or `null` when no counters were readable).

## An ABBA group by hand

One server per leg, started fresh, with nothing else talking to it. For a pair A and B:

1. Start server A. Wait until `/health` answers.
2. `python3 bench/throughput/bench5.py --base-url http://127.0.0.1:8088/v1 --label A-1`
3. Stop server A and check that the process has exited.
4. Start server B, run with `--label B-1`, stop it.
5. Start B again, run with `--label B-2`, stop it.
6. Start A again, run with `--label A-2`, stop it.

Each arm's figure pools its two legs: sum `pooled_decode_tokens` and `pooled_decode_seconds`
over the legs and divide. Don't average the leg rates. Check that every leg reads
`"clean": true` and that the truncation counts match between arms. For sampled throughput,
use `--temperature 1.0 --reps 5` and report the per-leg spread; the texts differ between arms,
so the result is not a paired one.

On the PrismML llama.cpp fork, the fork probe's legs differed only in the drafter flags. The
shared part of the launch line, without host, alias and key:

```
llama-server -m Ternary-Bonsai-2-27B-PQ2_0.gguf --port 8088 -ngl 99 -fa on -c 131072 -np 1 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --metrics
```

- No drafter: the line as it stands.
- DFlash drafter: add `-md <drafter.gguf> --spec-type draft-dflash --spec-draft-n-max 7
  -ngld 99`. The draft maximum must equal the drafter's block size, or the server asserts.
- MTP: serve the grafted-MTP GGUF as `-m` and add `--spec-type draft-mtp --spec-draft-n-max 2`.
  No `-md`; the MTP head is inside the model file.

Add `--api-key <key>` to the server and pass the same key with `--api-key` or
`BENCH_API_KEY`. The runner's greedy default sends `temperature: 0`, which overrides `--temp`;
the sampling flags matter only for `--temperature default`.
