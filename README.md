# bonsai2-drafter

An unofficial DFlash 2 drafter fine-tuned against PrismML's **Ternary-Bonsai-2-27B**, with the
trainer that made it, an evaluation kit that runs on any runtime, and the patches that serve it
on mlx-dspark.

| | |
| --- | --- |
| **Drafter** | [`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5), Apache-2.0 |
| **Drop-in for** | `z-lab/Qwen3.8-27B-DFlash2`: the same 81 tensor names, dtypes, shapes and `config.json`. Any runtime that loads the stock drafter loads this one unchanged |
| **Measured on** | one Apple M4 Pro with 48 GB, mlx-dspark 0.18.0 with the patches below, 8-bit KV cache, drafter quantized to 4 bits at load, draft cap 7 |
| **Protocol** | [`BENCHMARK.md`](BENCHMARK.md), registered before the comparisons it governs |

## What it changes

A drafter changes how fast the target answers, not what it answers. Every greedy comparison
below produced the same tokens from both drafters within the output budget.

| measurement | stock | ft5 | gain |
| --- | --- | --- | --- |
| acceptance, general chat, 200 tokens | 2.7768 | 3.0403 tokens a round | **+9.49%**, paired 95% +7.25% to +11.69% |
| acceptance, general chat, 1024 tokens | 2.8149 | 3.0399 | +7.99%, +6.75% to +9.24% |
| acceptance, held-out code, 200 tokens | 3.8166 | 4.2130 | +10.39%, +7.80% to +13.13% (this suite chose the training iteration, so it does not decide) |
| decode, greedy, HTTP, ABBA | 25.09 | 28.93 tok/s | **+15.3%** |
| decode at the target's published sampling (temperature 1.0), ABBA | 24.07 | 26.30 tok/s | +9.3% (a screen: reps overlap) |

The acceptance intervals are a stratified, paired prompt bootstrap over 40 frozen prompts per
suite. Truncated answers are prefixes, not complete responses. Figures come from one machine.

## Use it

### Serve on mlx-dspark

```sh
scripts/serve-bonsai2.sh
```

This builds the pinned environment in `.venv` with uv, fetches the target and the drafter at
pinned revisions, and serves an OpenAI-compatible API at `http://127.0.0.1:8088/v1`, with the
settings the measurements used. Any model name is accepted in requests. It needs macOS on Apple
Silicon, uv and about 13 GB of disk; long contexts want a 48 GB machine.

Environment variables override the defaults. `BONSAI2_DRAFTER` points it at another drafter (a
Hub repo or a local directory) and `BONSAI2_DRAFTER_REVISION` pins that repo's revision.
Without one, the default drafter keeps its pinned revision and any other repo is fetched at
`main`. `BONSAI2_HOST` and `BONSAI2_PORT` move the address, and `BONSAI2_VENV` the environment.

`bin/mlx-dspark-patched` is mlx-dspark with this repository's patches installed first:

- a loader for the pack's rotated 2-bit format (`patches/bonsai_loader.py`);
- a 2-bit, group-128 unpack for mlx-dspark's small-M verify kernel (`patches/small_m_2bit.py`),
  which is what lets the draft cap reach 7 on this target;
- a loader for prequantized DFlash 2 drafters (`patches/dflash_prequantized.py`);
- and the stability and kernel patches the measurements ran with.

Each patch degrades to stock mlx-dspark if it cannot apply.

### Use it elsewhere

Point any DFlash 2 runtime that serves Bonsai 2 at the drafter, as you would the stock drafter.
It has been measured on two runtimes that share no serving code:

| runtime | draft cap | general chat, 200 tokens: stock → ft5 |
| --- | --- | --- |
| mlx-dspark 0.18.0 with the patches here | 7 | 2.7768 → 3.0403 tokens a round, **+9.49%** (+7.25% to +11.69%) |
| [dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2) at `223e0f3` | 4 (its DFlash 2 limit) | 2.6392 → 2.7714, **+5.01%** (+3.35% to +6.83%) |

The second runtime loads the drafter unchanged; its loader maps the codebook names itself. The
gain is smaller there because a block of five commits at most five tokens a round. Per-runtime
figures are not comparable with each other, only within a runtime. If you run the drafter
somewhere else, a report with acceptance against the stock drafter is the most useful thing
you can send.

### Move a drafter between runtimes

The selector's two codebooks circulate under different names: z-lab's (and this drafter's)
bare `…_codebook`, and an embedding-style `…_codebook.weight`. A loader that checks names
strictly refuses the other. `scripts/rename-codebooks.py` converts between them, rewriting only
the header and checking that the tensor bytes are unchanged:

```sh
python3 scripts/rename-codebooks.py SRC_DIR DST_DIR --to zlab   # or --to embedding
```

## Evaluate any drafter

`bench/analysis/analyse_served_accept.py` needs only the Python standard library. It reads two
per-prompt reports (tokens, rounds, finish reason, output token ids, optionally round lengths)
and returns the paired acceptance gain with its interval. It also classifies every unequal pair
under the contract `budgeted-prefix-identity/v2`. A generator can overshoot its budget by one
draft block, and that passes. A difference inside the budget is a finding. `served_accept.py`
writes those reports for mlx-dspark, and `bench/adapters/dflash_mlx_bonsai2.py` writes them for
dflash-mlx-bonsai2; any runtime that can emit the same fields can be compared.
The report format is specified in [`bench/REPORTS.md`](bench/REPORTS.md), with a JSON Schema
and a standard-library validator (`bench/analysis/validate_report.py`), so an adapter for
another runtime can be written without reading this code.
`BENCHMARK.md` fixes how comparisons between drafters and runtimes are run and reported.
`bench/throughput/bench5.py` takes its throughput measurements against any OpenAI-compatible
server by URL; `bench/throughput/README.md` has the ABBA procedure.

## Train a drafter on a Mac

`bench/drafter/` is a DFlash 2 fine-tuning loop in MLX: corpus generation from the target's own
greedy answers, distillation from the target's hidden states, the served-geometry anchors that
make training match what serving presents, export and the served acceptance gate. It trained
ft5 on one 48 GB M4 Pro. The recipe and its pitfalls are in
[`bench/drafter/README.md`](bench/drafter/README.md). The most expensive pitfall: training that
lets the block see the target's row at the anchor looks better on a proxy and serves worse.

## Known limitations

- Only the pinned stack in `envs/dspark/` (mlx-dspark 0.18.0) is validated.
- At an 8-bit KV cache, one prompt in the quality suite (an ISO-8601 parser) exhausts its whole
  budget in reasoning and never answers. It does so with and without any drafter, and passes at
  16-bit KV. It is a property of the cache setting, disclosed on the model card.
- Everything was measured on one machine.

## About this code

The measurements on the model card were taken with this code. The evidence behind them is under
[`evidence/`](evidence/README.md): the raw reports with full token arrays, sanitized of timestamps and
local details, with a verifier and checksums, and the commands that reproduce every published number.

Checks: `bench/check.sh` (lint, types, no-GPU tests), `--gpu` and `--models` for the rest.

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). This project works with Bonsai by
Prism ML and is not affiliated with Prism ML, Qwen, Inco AI, z-lab or mlx-dspark.
