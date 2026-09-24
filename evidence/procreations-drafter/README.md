# ProCreations' drafter on both MLX runtimes

[`ProCreations/Ternary-Bonsai-2-27B-DFlash2`](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2)
at `4cfb6ad03268fed0f60ca96c1a659c0b1c77e50b` (Apache-2.0) is another DFlash 2 fine-tune of
z-lab's stock drafter against PrismML's ternary target. It was loaded exactly as published: its
bf16 `model.safetensors` (sha256 `708e141bf4a33b09620c4f86263f1c7edcec7882f5dd39448e83e0c0defdc230`,
as its `SHA256SUMS` lists), with no conversion. It has the same 81 tensor names, dtypes and shapes
as stock and ft5, so it loaded unchanged on both runtimes.

**Protocol status.** On dflash-mlx-bonsai2 this is `BENCHMARK.md` version 1's fourth DFlash 2
arm, acceptance only: the target alone, stock, ft5 and r3 ran under the protocol in
`second-runtime/`, at the same runtime commit and settings. On mlx-dspark it is **not a protocol
result**. The protocol re-runs every arm, and the mlx-dspark baselines here are earlier reports:
stock and ft5 from before the protocol was registered, r3 from the pre-publication screen. They
are matched comparisons (same harness, settings and prompts; greedy acceptance on it is
deterministic, as the battery's identical repeated arms `general-1-stock` and `general-4-stock`,
`general-2-ft5` and `general-3-ft5` show). Under the protocol only general chat at 200 tokens
decides; general chat at 1024 tokens is reported beside it, and code is flagged because ft5's
training iteration was selected on it.

## Files

| path | what it is |
|---|---|
| `reports/dspark-{general200,code200,general1024}-pc.json` | mlx-dspark 0.18.0 with this repository's patches, cap 7, drafter at 4 bits, 8-bit KV; written by the same `served_accept.py` copy as `derivative-abliterated/` |
| `reports/dflash-{general200,code200,general1024}-pc.json` | dflash-mlx-bonsai2 at `223e0f3a`, cap 4, `w4`, written by `bench/adapters/dflash_mlx_bonsai2.py`; the adapter records the drafter's sha256 and a strict load |
| `analysis/{dspark,dflash}-*-{stock,ft5,r3}-to-pc.json` | the published analyser's `compare` output, unchanged: each baseline drafter first, ProCreations second |
| `analysis/dflash-*-target-vs-pc.json` | the same against the target alone |
| `verify/overlap.json` | the contamination check below |
| `verify/crosscheck.json` | the GGUF cross-check below |

The baselines: on mlx-dspark, stock from `ft5-publication-battery/acceptance/`, ft5 from
`prequantized-4bit/served/*-1-source.json` (the same outputs as the battery's, with
`round_lengths`) and r3 from `sampled-screen/acceptance/`; on dflash-mlx-bonsai2, all from
`second-runtime/reports/`.

## Claims and how to reproduce them

```sh
G=evidence/procreations-drafter/reports; A=evidence/ft5-publication-battery/acceptance
Q=evidence/prequantized-4bit/served; S=evidence/sampled-screen/acceptance; R=evidence/second-runtime/reports
an() { python3 bench/analysis/analyse_served_accept.py "$1" "$2"; }
an $A/general-1-stock.json $G/dspark-general200-pc.json;       an $Q/general-1-source.json $G/dspark-general200-pc.json
an $S/general-r3.json $G/dspark-general200-pc.json
an $A/code-1-stock.json $G/dspark-code200-pc.json;             an $Q/code-1-source.json $G/dspark-code200-pc.json
an $S/code-r3.json $G/dspark-code200-pc.json
an $A/general-long-1-stock.json $G/dspark-general1024-pc.json; an $Q/long-1-source.json $G/dspark-general1024-pc.json
an $S/long-r3.json $G/dspark-general1024-pc.json
for s in general200 code200 general1024; do
  for b in stock ft5 r3 target; do an $R/$s-$b.json $G/dflash-$s-pc.json; done
done
```

Tokens per round, and ProCreations' change against each baseline (paired 95% interval):

| runtime | suite | stock | ProCreations | stock → PC | **ft5 → PC** | r3 → PC |
|---|---|---|---|---|---|---|
| mlx-dspark, cap 7 | general, 200 | 2.7768 | 3.0960 | +11.50% [+9.95, +13.04] | **+1.83% [+0.26, +3.54]** | +6.77% [+5.31, +8.26] |
| | code, 200 | 3.8166 | 4.1773 | +9.45% [+7.20, +11.65] | **−0.85% [−3.14, +1.26]** | +5.01% [+2.76, +7.19] |
| | general, 1024 | 2.8149 | 3.1178 | +10.76% [+9.74, +11.74] | **+2.56% [+1.65, +3.48]** | +6.46% [+5.45, +7.43] |
| dflash-mlx-bonsai2, cap 4 | general, 200 | 2.6392 | 2.8795 | +9.10% [+7.95, +10.26] | **+3.90% [+2.62, +5.14]** | +6.47% [+5.42, +7.43] |
| | code, 200 | 3.1834 | 3.4061 | +7.00% [+5.38, +8.58] | **−0.10% [−1.94, +1.61]** | +5.03% [+3.61, +6.45] |
| | general, 1024 | 2.6452 | 2.8806 | +8.90% [+8.08, +9.67] | **+3.87% [+3.23, +4.54]** | +6.51% [+5.83, +7.17] |

So ProCreations' drafter accepts more than ft5 on general chat on both runtimes, which under
the protocol's deciding rule means it beats ft5 on dflash-mlx-bonsai2. The two are level on
code, the suite ft5 was selected on. Both accept more than stock and r3 on every suite and
runtime. Every pair has zero prefix divergences and zero early-stop mismatches: on mlx-dspark
the outputs differ only above the budget, and on dflash-mlx-bonsai2 they are identical on all
40 prompts. The two code pairs against ft5 print `investigate` and exit 1 because the gain is
not significant; that is the finding. Against the target alone, ProCreations diverges on 6, 4 and 19 prompts, the same
prompts at the same token positions as stock, ft5 and r3: the runtime's floating-point ties
(see `second-runtime/`), not the drafter.

## Contamination check

The 80 evaluation prompts' user texts were searched in the training and selection corpus
ProCreations publishes in its repository at this revision (`prompts.json`,
`training-texts.jsonl` and `generated.jsonl` of its r2 continuation experiment): lower-cased,
whitespace-normalised, exact substring search of each whole text and of 60-character windows
at a 30-character stride. **No match**, whole or window, in either suite
(`verify/overlap.json`).

## The bf16 file is the GGUF's source

ProCreations' primary release is a Q8_0 GGUF for llama.cpp. Requantizing the bf16 tensors with
llama.cpp's reference Q8_0 rule reproduces the GGUF's blocks byte for byte for the three
tensors checked (491,520 blocks), and two norm tensors are exactly equal
(`verify/crosscheck.json`). So the file measured here is the checkpoint that GGUF was made
from. Against stock, 8 of 81 tensors are identical and the rest differ by a median of 0.21%.

## What this does not show

- **Speed.** This is acceptance only; ProCreations was not timed under the ABBA protocol. The
  reports' decode times were taken while another job shared the GPU, so do not read speed from
  them. Tokens per round is unaffected.
- **llama.cpp.** ProCreations' own runtime is a CUDA build of PrismML's llama.cpp fork. On Metal
  that fork's DFlash 2 path is slower than no drafter at all (`fork-probe/`), so it was not run.
- **Why.** The three fine-tunes used different data. Which part of ProCreations' recipe helps
  on general chat is not tested here.

## What was stripped

Paths only: the local drafter directory became `ProCreations/Ternary-Bonsai-2-27B-DFlash2`, and
the dflash reports' `target_path`, `drafter_path` and `resolved_model_ref` were removed.
`verify_sanitized.py pairs` passed on all 8 data files. The analyses were written by the
published analyser from the files in this bundle.
