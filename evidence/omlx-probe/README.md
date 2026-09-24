# oMLX: a compatibility probe

Can oMLX (GitHub `jundot/omlx` at `5aa6c7f9`, with its forks of dflash-mlx, mlx-lm and mlx-vlm)
serve Ternary-Bonsai-2-27B with a DFlash 2 drafter? A functional check, not a `BENCHMARK.md`
measurement: one greedy request of 200 tokens with `bench5`'s first prompt, target
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b41`, drafter ft5 quantized to 4 bits by oMLX's
DFlash settings, on one Apple M4 Pro with 48 GB.

| run | what happened | total time for 200 tokens |
|---|---|---|
| oMLX as released (`plain-fallback.json`) | serves Bonsai 2 on its VLM engine, but DFlash refuses the model type: "Model type prism_hadamard_qwen35 not supported", and it falls back to plain decoding | 11.73 s (2.27 s of it loading) |
| with `prism_hadamard_qwen35` added to oMLX's list of batched-DFlash model types at run time, no file edited (`dflash-routing-patched.json`) | ft5 loads as a batched DFlash 2 drafter and drafts: 76 rounds, 2.63 tokens a round, 122 of 198 drafts accepted | 39.40 s (2.22 s loading) |

What each run did (the refusal, the drafter loading, its round and acceptance counts) is from
oMLX's server log, which is not in this bundle; the files hold the responses. The two runs'
greedy texts are identical, so the drafter changes nothing but speed there too. But with it,
oMLX is about four times slower than without it: its verify path has no fast multi-row kernel
for the ternary pack, the same failure as PrismML's llama.cpp fork on Metal (`fork-probe/`).
**So oMLX runs Bonsai 2 but not DFlash 2 on it, and a one-line routing change would make DFlash
2 work but lose.** The times are oMLX's own `usage.total_time`, one request each, not a
throughput measurement.

## Files

`plain-fallback.json` and `dflash-routing-patched.json`: the request's wall time, the raw
response (usage, with oMLX's timings) and the generated text. Each response's `created`
timestamp was removed (`verify_sanitized.py pairs` passes on both); nothing else was changed.
