# Second runtime: dflash-mlx-bonsai2

ft5 against the stock drafter (and naklitechie's r3 fine-tune) on an independent runtime,
[dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2) at
`223e0f3a9cb4806da0cdc5190f9191b545d1f60b` (dflash-mlx 0.1.10, MLX 0.32.2, mlx-lm 0.31.3), same
target pack `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b41`, greedy, one Apple M4 Pro with
48 GB. The reports were written by this repository's `bench/adapters/dflash_mlx_bonsai2.py`.
The runtime clamps DFlash 2 to a block of five, so cap 4; drafters quantized to 4 bits at load
(`w4`, group 64) unless marked bf16; KV cache unquantized (the runtime's default).

## Files

| path | what it is |
|---|---|
| `reports/{general200,code200,general1024}-{target,stock,ft5,r3}.json` | 12 per-prompt reports: target alone (the runtime's own exact autoregressive path), stock, ft5 and r3, on each suite and budget. Per prompt: hash, category, thinking flag, tokens, rounds, `round_lengths`, finish reason, the full `response_ids`, durations, and the runtime's own per-pass counts under `runtime` |
| `reports/general200-r3z.json` | r3 from a copy whose two codebook keys were renamed to z-lab's bare names: the loader check |
| `reports-bf16/general200-{stock,ft5}-bf16.json` | the two drafters left in bf16: a precision sensitivity check |
| `analysis/*-stock-to-ft5.json`, `*-stock-to-r3.json`, `*-ft5-to-r3.json` | paired analyses, `budgeted-prefix-identity/v2` |
| `analysis/*-identity-target-vs-*.json` | each drafter arm against the target alone, with every divergence named |
| `analysis/*-tie-margins.json` | the target's logit margin between its top two tokens at each divergence position |
| `analysis/general200-bf16-stock-to-ft5.json`, `general200-equivalence-r3-native-vs-r3z.json`, `summary.json` | the bf16 pair, the loader equivalence check, and all of the above in one file |

## Claims and how to reproduce them

The README's and card's second-runtime claims: general chat at 200 tokens **2.6392 → 2.7714
tokens a round, +5.01% (+3.35% to +6.83%)**; code +7.10%; general at 1024 tokens +4.84%; and
on the card's related-releases line, r3 **2.41% lower than ft5 (−3.83% to −1.08%) on its own
runtime**. From the repository root:

```sh
R=evidence/second-runtime/reports
for s in general200 code200 general1024; do
  python3 bench/analysis/analyse_served_accept.py $R/$s-stock.json $R/$s-ft5.json
  python3 bench/analysis/analyse_served_accept.py $R/$s-stock.json $R/$s-r3.json
  python3 bench/analysis/analyse_served_accept.py $R/$s-ft5.json   $R/$s-r3.json
done
python3 bench/analysis/analyse_served_accept.py evidence/second-runtime/reports-bf16/general200-stock-bf16.json evidence/second-runtime/reports-bf16/general200-ft5-bf16.json
python3 bench/analysis/analyse_served_accept.py $R/general200-r3.json $R/general200-r3z.json --equivalence
```

Rerun on these sanitized files:

| suite | stock | ft5 | r3 | stock → ft5 | stock → r3 | ft5 → r3 |
|---|---|---|---|---|---|---|
| general, 200 (deciding) | 2.6392 | 2.7714 | 2.7046 | **+5.01% [+3.35, +6.83]** | +2.48% [+1.70, +3.28] | **−2.41% [−3.83, −1.08]** |
| code, 200 | 3.1834 | 3.4096 | 3.2429 | +7.10% [+5.25, +9.03] | +1.87% [+0.71, +3.08] | −4.89% [−6.66, −3.12] |
| general, 1024 | 2.6452 | 2.7732 | 2.7044 | +4.84% [+3.59, +6.03] | +2.24% [+1.74, +2.72] | −2.48% [−3.43, −1.48] |
| general, 200, bf16 drafters | 2.6438 | 2.7878 | – | +5.44% [+3.37, +7.59] | – | – |

Every drafter pair is 40/40 token-identical. The stock → ft5 and stock → r3 pairs print
`"gate": "pass"` and exit 0; the ft5 → r3 pairs print `investigate` and exit 1, because r3's
gain over ft5 is negative, which is the finding. The r3/r3z equivalence check passes, 40/40
equal.

**Output agreement.** Against the target alone, the drafter arms differ at the same prompts and
positions whichever drafter runs: general@200 34 identical and 6 prefix divergences, code@200
36 and 4, general@1024 21 and 19 (`analysis/*-identity-target-vs-*.json`). Every one of those
29 positions is a floating-point tie in the target's own logits, with a top-two margin of 0 or
one fp16 step (0.0156) on the prefill or one-row path and at most 0.0096 on the verify path
(`analysis/*-tie-margins.json`): the verify kernel and the one-token decode kernel break the
tie differently. That is the card's "logit margins ≤ 0.016".

## What was stripped

Local paths to the target snapshot and the drafter directories (`arm.target_path`,
`arm.drafter_path`, `arm.draft_meta.resolved_model_ref`), and, in the tie-margin files, the
directory part of the report paths (they now read `reports/...`). Everything else is as the
adapter wrote it: settings, package versions, the runtime commit, drafter and config hashes,
every per-request measurement. `../tools/verify_sanitized.py` proved every number unchanged.

## What was left out

The run's queue and chain scripts, the per-arm logs, smoke-test reports, an aborted first start
of the general@200 target arm (killed within 40 seconds because it ran before the adapter
recorded the runtime commit, and replaced before any analysis), and the runtime's package
freeze (its versions are in each report's `settings`). The adapter that produced the reports
is `bench/adapters/dflash_mlx_bonsai2.py`; after the runs its docstring was edited, not its
code.
