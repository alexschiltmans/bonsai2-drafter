# Throughput on the second runtime: dflash-mlx-bonsai2

`BENCHMARK.md` version 1, measurement 2, on
[dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2) at `223e0f3a`: the stock
drafter (`z-lab/Qwen3.8-27B-DFlash2@50307d4c`) against ft5 (weights sha256 `63399215…`), each served
by the runtime's own `dflash serve` with the Bonsai 2 pack
(`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b41`), `DFLASH_PRISM_VERIFY=v7`, prefill step 512,
and the drafter quantized to 4 bits, group 64 (`--draft-quant w4:gs64`, given explicitly for both).
One fresh server per leg, `bench/throughput/bench5.py` greedy, three repetitions, ABBA (stock,
ft5, ft5, stock). One Apple M4 Pro with 48 GB, on AC power.

Measurement 3 (sampled throughput) was not run: this runtime speculates only on greedy requests,
so a sampled leg would compare two identical plain-decoding arms.

## Files

| path | what it is |
|---|---|
| `records/{stock,ft5}-greedy-{a1,a2,b1,b2}.json` | the four `bench5` records, unchanged |
| `machine-state.tsv` | memory pressure, swap and power before and after every leg |
| `records-r3/{r3,ft5}-greedy-{a1,a2,b1,b2}.json`, `machine-state-r3.tsv` | a second ABBA group, naklitechie's r3 (`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`, loaded unchanged) against ft5, with the runtime's on-disk prefix cache cleared before every leg |

## Claims and how to reproduce them

```sh
R=evidence/second-runtime-throughput/records
python3 bench/throughput/pool.py $R/stock-greedy-a1.json $R/stock-greedy-a2.json -- $R/ft5-greedy-b1.json $R/ft5-greedy-b2.json
```

| | stock | ft5 | ft5 / stock |
|---|---|---|---|
| greedy, pooled end to end, tok/s | 23.131 (legs 23.15, 23.112) | 24.788 (legs 24.787, 24.789) | **1.0716** |

Truncation is 24 of 30 in both arms. The acceptance gain on this runtime is +5.01% on general
chat (`second-runtime/`); these five prompts are code.

Against r3, on r3's own runtime:

```sh
R=evidence/second-runtime-throughput/records-r3
python3 bench/throughput/pool.py $R/r3-greedy-a1.json $R/r3-greedy-a2.json -- $R/ft5-greedy-b1.json $R/ft5-greedy-b2.json
```

| | r3 | ft5 | ft5 / r3 |
|---|---|---|---|
| greedy, pooled end to end, tok/s | 23.484 (legs 23.482, 23.485) | 24.782 (legs 24.788, 24.776) | **1.0553** |

ft5's legs in the two groups agree within 0.1% (24.787 and 24.789; 24.788 and 24.776).

**The metric differs from version 1's wording.** The server, built on `mlx_lm.server`, reports no
decode timer, so every response lands in `bench5`'s end-to-end pool and none in its decode pool.
Both arms pay the same prefill on the same prompts, so the ratio carries the drafter's effect;
the absolute rates include prefill. The server has no request counter either, so the legs record
`"clean": null`; nothing else was running (`machine-state.tsv`). The machine carried about 3.6 GB
of swap throughout, under the protocol's 4 GB limit. The runtime keeps an on-disk prefix cache
that survives a restart, so the later legs could restore prompts the first one wrote. It did not
move the result: the stock drafter's first leg, which ran cold, and its last agree within 0.2%
(23.15 and 23.112), and the prompts are short.

The records are `bench5`'s output as written; nothing in this group was edited.
