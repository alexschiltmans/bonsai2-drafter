# Two serving stacks on one Mac: MTPLX (MTP) against mlx-dspark (DFlash 2)

`BENCHMARK.md` version 2 (tag `benchmark-v2`), registered before these legs ran: which way of
serving Ternary-Bonsai-2-27B is fastest on this Mac. A stack comparison, the protocol's one
exception to "throughput is never compared across runtimes", not a drafter result.

- **MTPLX 2.12.0** (PyPI), MTP with Qwen3.8's own head, serving
  `Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed@03bd60bb`, whose `model.safetensors` is
  byte-identical to PrismML's pack (sha256 `130de592…`). The pack's defaults (turbo profile, MTP
  depth 1), with `--no-auth`, `--no-stats-footer` (otherwise a speed footer is appended to the
  text) and `--ssd-session-cache off` (a fresh server per leg, as version 1 requires).
- **mlx-dspark** exactly as in `throughput-ft5-procreations/`, with ft5, the drafter that had
  the higher pooled greedy decode rate there, as the amendment specified.

`bench/throughput/bench5.py`, five code prompts at 400 tokens, three repetitions greedy and five
at temperature 1.0, top-p 0.95, top-k 20, in ABBA order (MTPLX, mlx-dspark, mlx-dspark, MTPLX).
One Apple M4 Pro with 48 GB, on AC power.

## Files

| path | what it is |
|---|---|
| `records/{mtplx,ft5}-{greedy,sampled}-{a1,a2,b1,b2}.json` | the eight `bench5` records, unchanged |
| `probes/probe-{mtplx,dspark}.json` | one untimed greedy request per stack with `bench5`'s first prompt, 96 tokens: the raw response and whether the answer opened with reasoning |
| `machine-state.tsv` | the checks before and after every leg |

## Claims and how to reproduce them

```sh
R=evidence/mac-stacks-mtplx/records
for m in greedy sampled; do
  python3 bench/throughput/pool.py $R/ft5-$m-b1.json $R/ft5-$m-b2.json -- $R/mtplx-$m-a1.json $R/mtplx-$m-a2.json
done
```

| | mlx-dspark + ft5 | MTPLX | MTPLX / mlx-dspark |
|---|---|---|---|
| greedy, pooled end to end, tok/s (registered metric) | 28.133 (legs 28.172, 28.093) | 38.892 (legs 38.41, 39.386) | **1.3824** |
| sampled, pooled end to end, tok/s | 25.679 (legs 25.453, 25.918) | 37.439 (legs 37.839, 37.049) | **1.4580** |
| greedy, pooled decode, tok/s | 28.871 | 40.951 | 1.4184 |
| sampled, pooled decode, tok/s | 26.249 | 39.291 | 1.4968 |

**Why.** From the records, MTPLX commits 1.80 tokens a round (81.5% of its one-token drafts
accepted, greedy) in 43.9 ms a round, 24 ms a token; mlx-dspark with ft5 commits 3.68 tokens a
round in 127.6 ms, 35 ms a token. MTPLX wins on the cost of a round, not on drafting.

## What this does not show

- **Same text.** Both stacks reason by default (the probes), but they render the prompt
  differently: 46 prompt tokens on MTPLX, 88 on mlx-dspark. The generated texts differ, and so do
  the truncation counts (greedy 30 of 30 for MTPLX, 24 of 30 for mlx-dspark). Read the ratios as
  what each stack does with its own defaults on these prompts.
- **A counter check for MTPLX.** It has no request counter `bench5` can read, so its legs record
  `"clean": null`; nothing else was running (`machine-state.tsv`).
- **Other Macs or longer contexts.** One machine, chat-length prompts.

**A correction to the amendment.** It says MTPLX's responses carry no decode timer. They do: a
llama.cpp-style `timings` object, which `bench5` reads, so the decode pool above exists. The
comparison still uses the registered end-to-end pool; the two agree in direction and size.

## What was stripped

Only the probes were edited: each response's `created` timestamp was removed
(`verify_sanitized.py pairs` passes on both). The records are `bench5`'s output as written.
