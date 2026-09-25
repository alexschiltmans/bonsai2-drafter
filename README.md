# bonsai2-drafter

An unofficial DFlash 2 drafter fine-tuned for PrismML's **Ternary-Bonsai-2-27B**, with the trainer
that made it, an evaluation kit that works with any runtime, and the patches that serve it on
mlx-dspark.

| | |
| --- | --- |
| **Drafter** | [`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5), Apache-2.0; also as [4-bit MLX](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit) and [Q8_0 GGUF](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-GGUF) |
| **Drop-in for** | `z-lab/Qwen3.8-27B-DFlash2`: the same 81 tensor names, dtypes, shapes and `config.json` |
| **Protocol** | [`BENCHMARK.md`](BENCHMARK.md), registered before the comparisons it governs |
| **Write-up** | [One drafter, several runtimes, two methods](docs/one-drafter-several-runtimes.md): how ft5 was made and how it compares with the other drafters |

## Results

Against the stock drafter on one Apple M4 Pro with 48 GB: mlx-dspark 0.18.0 with the patches
below, 8-bit KV cache, drafter quantized to 4 bits at load, draft cap 7.

| measurement | stock | ft5 | gain |
| --- | --- | --- | --- |
| acceptance, general chat, 200 tokens | 2.7768 | 3.0403 tokens a round | **+9.49%** (paired 95% +7.25% to +11.69%) |
| acceptance, general chat, 1024 tokens | 2.8149 | 3.0399 | +7.99% (+6.75% to +9.24%) |
| acceptance, held-out code, 200 tokens | 3.8166 | 4.2130 | +10.39% (+7.80% to +13.13%); not deciding, it chose the training iteration |
| decode, greedy | 25.15 | 28.98 tok/s | **+15.2%** |
| decode, target's published sampling | 24.41 | 26.05 tok/s | +6.7% |

Greedy output was the same for both drafters within the output budget. The intervals are a
paired prompt bootstrap over 40 frozen prompts per suite; the decode rows are `BENCHMARK.md`'s
measurements 2 and 3. Truncated answers are prefixes, not complete responses.

On [dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2) at `223e0f3`, a
runtime that shares no serving code with mlx-dspark and caps a round at four drafts, general chat
went from 2.6392 to 2.7714 tokens a round, **+5.01%** (+3.35% to +6.83%). Figures compare within
a runtime, not across runtimes.

## Serve on mlx-dspark

```sh
scripts/serve-bonsai2.sh
```

This builds the pinned environment in `.venv` with uv, fetches the target and the drafter at
pinned revisions, and serves an OpenAI-compatible API at `http://127.0.0.1:8088/v1` with the
measured settings. It needs macOS on Apple Silicon, uv and about 13 GB of disk; long contexts
want 48 GB. `BONSAI2_DRAFTER` serves another drafter (a Hub repo, fetched at `main` unless
`BONSAI2_DRAFTER_REVISION` pins it, or a local directory); `BONSAI2_HOST`, `BONSAI2_PORT` and
`BONSAI2_VENV` move the address and the environment.

`bin/mlx-dspark-patched` is mlx-dspark with the patches in `patches/` installed first: a loader
for the pack's rotated 2-bit format, a 2-bit, group-128 path for the small-M verify kernel (which
is what lets the draft cap reach 7 on this target), a loader for prequantized drafters, and the
stability patches the measurements ran with. Each falls back to stock mlx-dspark if it cannot
apply.

## Other runtimes

Point any DFlash 2 runtime that serves Bonsai 2 at the drafter, as you would the stock one. Some
releases name the selector's codebooks `…_codebook.weight` instead of `…_codebook`, and a strict
loader refuses the other form. `scripts/rename-codebooks.py` converts between them, rewriting only
the header and checking that the tensor bytes are unchanged:

```sh
python3 scripts/rename-codebooks.py SRC_DIR DST_DIR --to zlab   # or --to embedding
```

Reports from other runtimes, with acceptance against the stock drafter, are welcome as pull
requests.

## Evaluate a drafter

- `bench/analysis/analyse_served_accept.py` needs only the standard library. It compares two
  per-prompt reports and returns the paired acceptance gain with its interval, classifying every
  unequal output pair under `budgeted-prefix-identity/v2`: overshooting the budget by one draft
  block passes, a difference inside it is a finding.
- `bench/drafter/served_accept.py` writes those reports on mlx-dspark and
  `bench/adapters/dflash_mlx_bonsai2.py` on dflash-mlx-bonsai2. The format is specified in
  [`bench/REPORTS.md`](bench/REPORTS.md), with a JSON Schema and a validator, so an adapter for
  another runtime needs no other code from here.
- `bench/throughput/bench5.py` measures throughput against any OpenAI-compatible server;
  `bench/throughput/README.md` has the ABBA procedure.

## Train a drafter on a Mac

`bench/drafter/` fine-tunes a DFlash 2 drafter in MLX: corpus generation from the target's greedy
answers, distillation from its hidden states, anchors placed where serving places them, export
and the served acceptance gate. It trained ft5 on one 48 GB M4 Pro; the recipe and its pitfalls
are in [`bench/drafter/README.md`](bench/drafter/README.md). The costliest one: letting the block
see the target's row at the anchor looks better on a proxy and serves worse.

## Limitations

- Only the pinned stack in `envs/dspark/` (mlx-dspark 0.18.0) is validated, and everything was
  measured on one machine.
- At an 8-bit KV cache, one quality prompt (an ISO-8601 parser) reasons through its whole budget
  and never answers, with or without any drafter. It passes at 16-bit KV; the model card has the
  details.

## Evidence and checks

[`evidence/`](evidence/README.md) holds the raw records behind the model cards: full token arrays,
sanitized of timestamps and local details, with checksums, a verifier and the commands that
recompute each figure. The model card marks any figure that is not in the bundle.

`bench/check.sh` runs lint, types and the no-GPU tests; `--gpu` and `--models` run the rest. Every
Python file passes `mypy --strict` and the rules in `ruff.toml`, at the versions `check.sh` pins.

## License

Apache-2.0, see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). Created using Bonsai by Prism ML.
This project is not affiliated with Prism ML, Qwen, Inco AI, z-lab or mlx-dspark.

The records under `evidence/` include the target model's outputs. Responses in the code suite often
quote their prompt, which is from
[sahil2801/CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k)
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)); `NOTICE` has the attributions.
