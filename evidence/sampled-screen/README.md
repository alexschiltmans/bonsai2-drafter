# Pre-publication screen on mlx-dspark

**A screen, not a benchmark result.** One screen ran on mlx-dspark only, to inform the
publication decision. `BENCHMARK.md` was committed before the screen started and describes it
in its section on screens; the benchmark re-runs every arm under the protocol and does not
reuse these numbers. It covered:

1. naklitechie's r3 fine-tune (`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` at
   `3fc0d6ef43e933a20f1e9ee53fac6f56402c0f83`) on the served acceptance loop, all three suites,
   paired against the existing ft5 and stock reports;
2. a greedy HTTP comparison, ft5 against r3, ABBA, three repetitions a leg;
3. a sampled HTTP comparison, stock against ft5, ABBA, five repetitions a leg, at the target's
   published sampling (the server's defaults: temperature 1.0, top-p 0.95, top-k 20).

Same stack as `../ft5-publication-battery/`: mlx-dspark 0.18.0 with this repository's patches,
8-bit KV, drafter quantized to 4 bits at load, one Apple M4 Pro with 48 GB. This is ft5's home
runtime; r3 was trained on CUDA, and `../second-runtime/` measures both on r3's own runtime.

## The r3 codebook rename

mlx-dspark's `load_dflash` checks tensor names strictly and refused r3, which names its two
selector codebooks with an embedding-style `.weight` suffix. r3 therefore ran from a copy with
only those two header keys renamed to z-lab's bare names: a same-length header rewrite, tensor
bytes untouched. `r3-rename.json` records it: the two key renames, `data_section_identical:
true`, and the sha256 before (`eb141d0f…`, which matches the Hub's LFS sha256 in
`r3-source.json`) and after. Reports that ran r3 name the drafter
`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2 (codebook keys renamed)`. The repository's
`scripts/rename-codebooks.py` does the same conversion. On the second runtime, which accepts
both spellings, the renamed and native copies ran the same loop (`../second-runtime/`,
`general200-equivalence-r3-native-vs-r3z.json`).

## Files

| path | what it is |
|---|---|
| `acceptance/{general,code,long}-r3.json` | r3's `served_accept.py --report` records: general and code at 200 tokens, general at 1024 |
| `screen-acceptance.json` | the six pairs (r3 against ft5 and against stock, per suite), `budgeted-prefix-identity/v2` |
| `http/greedy-leg{1-ft5,2-r3,3-r3,4-ft5}.json`, `screen-greedy.json` | greedy HTTP legs and their pooled summary. The legs were written by an unpublished version of the author's harness whose five prompts are those of `bench/throughput/bench5.py` but whose record layout differs (`../README.md`) |
| `http/sampled-leg{1-stock,2-ft5,3-ft5,4-stock}.json`, `screen-sampled.json` | sampled HTTP legs and their pooled summary |
| `r3-rename.json`, `r3-source.json` | the rename record and r3's Hub revision, licence, LFS sha256 and sizes |

## Claims and how to reproduce them

The card: r3 "accepted less on every suite … On general chat at 200 tokens it was … **4.63%
lower (−6.39% to −2.72%) on mlx-dspark**", and "At the target's published sampling (a screen):
**24.07 tok/s stock against 26.30 tok/s ft5, +9.3%**. The legs ran 23.71, 26.43, 26.17 and
24.43, every request at cap 7 … per-rep rates 22.5–27.1 stock, 25.0–27.1 ft5".

The screen paired r3 with ft5's reports from the 4-bit equivalence run (the bf16 "source" arm,
the same drafter as the battery, with identical tokens a round) and with the battery's stock
reports. From the repository root:

```sh
A=evidence/sampled-screen/acceptance
F=evidence/prequantized-4bit/served
S=evidence/ft5-publication-battery/acceptance
python3 bench/analysis/analyse_served_accept.py $F/general-1-source.json $A/general-r3.json
python3 bench/analysis/analyse_served_accept.py $S/general-1-stock.json  $A/general-r3.json
python3 bench/analysis/analyse_served_accept.py $F/code-1-source.json    $A/code-r3.json
python3 bench/analysis/analyse_served_accept.py $S/code-1-stock.json     $A/code-r3.json
python3 bench/analysis/analyse_served_accept.py $F/long-1-source.json    $A/long-r3.json
python3 bench/analysis/analyse_served_accept.py $S/general-long-1-stock.json $A/long-r3.json
```

Rerun on these sanitized files:

| suite | stock | r3 | ft5 | stock → r3 | ft5 → r3 |
|---|---|---|---|---|---|
| general, 200 (the suite `BENCHMARK.md` decides on) | 2.7768 | 2.8996 | 3.0403 | +4.42% [+3.18, +5.69] | **−4.63% [−6.39, −2.72]** |
| code, 200 (ft5 was selected on it) | 3.8166 | 3.9779 | 4.2130 | +4.23% [+2.80, +5.65] | −5.58% [−7.82, −3.40] |
| general, 1024 | 2.8149 | 2.9286 | 3.0399 | +4.04% [+3.52, +4.55] | −3.66% [−4.70, −2.61] |

No prefix divergence and no early-stop mismatch in any of the six pairs; every unequal pair is
a boundary overshoot of at most 7 tokens. The stock → r3 pairs print `"gate": "pass"`; the
ft5 → r3 pairs print `investigate` and exit 1, since the gain is negative.

HTTP, total completion tokens over total decode seconds per arm across its two legs:

```sh
python3 - <<'PY'
import json
H = "evidence/sampled-screen/http/"
def pooled(*legs):
    rows = [r for leg in legs for r in json.load(open(f"{H}{leg}.json"))["requests"]]
    return sum(r["completion_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows)
ft5, r3 = pooled("greedy-leg1-ft5", "greedy-leg4-ft5"), pooled("greedy-leg2-r3", "greedy-leg3-r3")
stock, ft5s = pooled("sampled-leg1-stock", "sampled-leg4-stock"), pooled("sampled-leg2-ft5", "sampled-leg3-ft5")
print(f"greedy   ft5 {ft5:.3f}  r3 {r3:.3f}  r3/ft5 {r3 / ft5:.4f}")
print(f"sampled  stock {stock:.3f}  ft5 {ft5s:.3f}  ft5/stock {ft5s / stock:.4f}")
PY
```

prints greedy ft5 28.921, r3 26.128, ratio 0.9034, and sampled stock 24.067, ft5 26.300, ratio
1.0928 (+9.3%). Sampled legs 23.71, 26.43, 26.17, 24.43; per-repetition rates 22.49–27.11
stock and 25.03–27.09 ft5 (`per_rep_decode_tps`); every request at cap 7. Under sampling the
arms produce different text, so this is a spread, not a paired result: the repetitions overlap.

## What was stripped

As in `../ft5-publication-battery/`: run timestamps, the private repository revision, the host
process list, the server log path, calibration cache keys, dated labels and file names. In
strings: local drafter directories became public names (r3's renamed copy is marked
"codebook keys renamed", including the two directory fields of `r3-rename.json`), launcher
paths became `python bin/mlx-dspark-patched`, the local API key became `REDACTED`. Every number
is unchanged; `../tools/verify_sanitized.py` proved it pair by pair.

## What was left out

The screen's supervisor status file and snapshot manifest (keyed by local paths, mostly
commands and machine-state dumps), its logs, and the record of a first attempt that stopped
within a minute when mlx-dspark refused r3's codebook names, before measuring anything. The
ft5 and stock reports the r3 reports pair with are not duplicated here; they are in
`../prequantized-4bit/served/` and `../ft5-publication-battery/acceptance/`.
