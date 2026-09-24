# Throughput on mlx-dspark: ft5 against ProCreations, r3 and the stock drafter

`BENCHMARK.md` version 1, measurements 2 and 3, for the two fine-tunes that accept the most:
ft5 (`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, weights sha256 `63399215…`) and
`ProCreations/Ternary-Bonsai-2-27B-DFlash2` at `4cfb6ad0` (bf16 weights sha256 `708e141b…`), each
loaded unchanged. One fresh server per leg, started with this repository's
`scripts/serve-bonsai2.sh` (mlx-dspark 0.18.0 with the patches, target
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b41`, 8-bit KV cache, drafter quantized to 4 bits
at load, cap chosen by the controller up to 7), and `bench/throughput/bench5.py`: five code
prompts at 400 tokens, three repetitions greedy and five at the target's published sampling
(temperature 1.0, top-p 0.95, top-k 20), in ABBA order. One Apple M4 Pro with 48 GB, on AC power.

## Files

| path | what it is |
|---|---|
| `records/{ft5,pc}-{greedy,sampled}-{a1,a2,b1,b2}.json` | the eight `bench5` records, unchanged: per request, tokens, decode and end-to-end seconds, rounds and finish reason |
| `machine-state.tsv` | memory pressure, swap, power source and competing inference processes before and after every leg |
| `records-r3/{ft5,r3}-{greedy,sampled}-{a1,a2,b1,b2}.json`, `machine-state-r3.tsv` | a second ABBA group, ft5 against naklitechie's r3 (`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`, its two codebook keys renamed to z-lab's names for mlx-dspark's strict loader; `second-runtime/` shows the renamed copy serves the identical loop) |
| `records-stock/dspark-{stock,ft5}-{greedy,sampled}-{a1,a2,b1,b2}.json`, `machine-state-stock.tsv` | a third ABBA group, the stock drafter (`z-lab/Qwen3.8-27B-DFlash2@50307d4c`, loaded unchanged) against ft5; each file is named after the record's own label |

## Claims and how to reproduce them

```sh
R=evidence/throughput-ft5-procreations/records
for m in greedy sampled; do
  python3 bench/throughput/pool.py $R/ft5-$m-a1.json $R/ft5-$m-a2.json -- $R/pc-$m-b1.json $R/pc-$m-b2.json
done
```

| | ft5 | ProCreations | ProCreations / ft5 |
|---|---|---|---|
| greedy, pooled decode tok/s | 28.7317 (legs 28.535, 28.931) | 27.2771 (legs 27.296, 27.258) | **0.9494** |
| sampled, pooled decode tok/s | 26.5764 (legs 26.509, 26.646) | 26.1006 (legs 26.196, 26.004) | **0.9821** |

Every leg reads `"clean": true`. Greedy truncation is 24 of 30 in both arms; sampled, 37 and 40.
As the protocol says, throughput is reported as ratios with every leg's value, and no
significance is claimed.

**Why ft5 is faster here.** Both drafters cost the same per round (128.2 against 127.6 ms,
greedy, from each request's `rounds` and `decode_seconds`), so the difference is acceptance:
3.683 against 3.480 tokens a round, with ft5 ahead or level on each of the five prompts.
`bench5`'s prompts are all code generation, ft5's training domain. This does not contradict
`procreations-drafter/`: ProCreations accepts more on the general chat suite, and the two are
level on the 40-prompt code suite.

## Against r3

```sh
R=evidence/throughput-ft5-procreations/records-r3
for m in greedy sampled; do
  python3 bench/throughput/pool.py $R/ft5-$m-a1.json $R/ft5-$m-a2.json -- $R/r3-$m-b1.json $R/r3-$m-b2.json
done
```

| | ft5 | r3 | r3 / ft5 |
|---|---|---|---|
| greedy, pooled decode tok/s | 28.889 (legs 28.967, 28.812) | 26.118 (legs 26.139, 26.097) | **0.9041** |
| sampled, pooled decode tok/s | 26.830 (legs 26.866, 26.795) | 25.016 (legs 25.094, 24.940) | **0.9324** |

Every leg reads `"clean": true`; truncation is 24 of 30 in both greedy arms, and 36 and 38 of 50
sampled. ft5's greedy rate in this group is within 0.6% of its rate in the group above.

## Against the stock drafter

```sh
R=evidence/throughput-ft5-procreations/records-stock
for m in greedy sampled; do
  python3 bench/throughput/pool.py $R/dspark-stock-$m-a1.json $R/dspark-stock-$m-a2.json -- $R/dspark-ft5-$m-b1.json $R/dspark-ft5-$m-b2.json
done
```

| | stock | ft5 | ft5 / stock |
|---|---|---|---|
| greedy, pooled decode tok/s | 25.150 (legs 25.140, 25.160) | 28.981 (legs 28.947, 29.015) | **1.1523** |
| sampled, pooled decode tok/s | 24.410 (legs 24.517, 24.303) | 26.052 (legs 26.343, 25.780) | **1.0673** |

Every leg reads `"clean": true`. Greedy truncation is 24 of 30 in both arms; sampled, 42 and 40 of
50. Both drafters cost the same per round (127.1 ms greedy, 130.6 and 130.5 ms sampled), and ft5
commits more tokens a round (3.683 against 3.197 greedy, 3.399 against 3.188 sampled), ahead on
each of the five prompts in both modes. The publication battery's greedy arms (25.0918 and
28.9295, `ft5-publication-battery/`) agree with this group within 0.3%. The sampled gain is smaller
than the +9.3% of `sampled-screen/`, a screen run before the protocol whose per-repetition rates
overlapped; this group is the protocol result. The machine carried about 3.0 GB of swap
throughout.

## Notes

A first launch of this queue stopped before any server started: it was started from an x86_64
shell, and the quickstart's architecture check refused to run. No measurement was taken; the
queue was relaunched unchanged from a native shell. Nothing here was edited: the records are
`bench5`'s output as written, and `bench5` writes no timestamps, host names or paths. The stock
group ran in one queue with `second-runtime-throughput/records-pc/`; `machine-state-stock.tsv` keeps
that queue's rows for this group, unchanged, and the rest are in `machine-state-pc.tsv` there.
