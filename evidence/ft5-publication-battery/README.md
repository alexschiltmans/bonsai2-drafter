# ft5 publication battery

The paired battery behind the ft5 model card's acceptance, throughput, output-agreement and
quality tables: stock `z-lab/Qwen3.8-27B-DFlash2` against
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, on mlx-dspark 0.18.0 with this repository's
patches, an 8-bit KV cache, the drafter quantized to 4 bits at load, cap 7, one Apple M4 Pro
with 48 GB.

## Files

| path | what it is | card claim it supports |
|---|---|---|
| `acceptance/general-{1-stock,2-ft5,3-ft5,4-stock}.json` | raw `served_accept.py --report` records, general suite, 200 tokens, run in the order 1-4 (stock, ft5, ft5, stock) | general chat 2.7768 → 3.0403 tokens a round, **+9.49%**, paired 95% +7.25% to +11.69%, forward (1→2) and reversed (4→3) |
| `acceptance/code-{1-stock,2-ft5}.json` | the same, code suite, 200 tokens | held-out code 3.8166 → 4.2130, +10.39% (+7.80% to +13.13%) |
| `acceptance/general-long-{1-stock,2-ft5}.json` | the same, general suite, 1024 tokens | 2.8149 → 3.0399, +7.99% (+6.75% to +9.24%) |
| `acceptance/acceptance-summary.json`, `long-acceptance-summary.json` | the version 1 analyses, unchanged, `investigate` verdicts included | the history behind the v2 contract |
| `reanalysis-v2/*.json`, `NOTES.md` | the four pairs under `budgeted-prefix-identity/v2`, and the two same-drafter identity controls | "zero differing tokens in 160 paired responses", overshoot at most 7 tokens, controls token-identical on 40/40 |
| `http/greedy-leg{1-stock,2-ft5,3-ft5,4-stock}.json` | the four `bench5.py` legs, ABBA, three repetitions of five prompts at 400 tokens, greedy | **25.0918 against 28.9295 tok/s, +15.29%**; legs 25.087, 28.925, 28.934, 25.097; every request at cap 7 |
| `quality/quality-{stock,ft5}.json` | `quality.py`, 14 tasks, three graders, one executable | 13/14 in both arms; `iso_seconds` fails in both at 16,391 and 16,389 tokens |
| `quality/tools-{stock,ft5}.json` | `tool_battery.py` at depths 2500 and 13000, four repetitions | 56/56 in both arms |
| `quality-controls/quality-nodrafter-kv8.json` | no drafter, 8-bit KV | `iso_seconds` fails, 16,384 tokens, no answer |
| `quality-controls/quality-nodrafter-kv16.json` | no drafter, unquantized (16-bit) KV (`--kv-bits 0`) | passes, 2,232 tokens |
| `quality-controls/quality-ft5-kv16.json` | ft5, unquantized (16-bit) KV | passes, 13,417 tokens, 14/14 |

Each acceptance record holds, per prompt: the prompt's sha256, category, thinking flag, token
and round counts, decode seconds, finish reason and the **full output token array**. The prompts
themselves are identified by hash; the corpora are `general.jsonl`
(`3dcc1327c0d254e2191322503f6cc4fe0cc464389ef0799093778ca55996c50b`) and `code.jsonl`
(`4252b5bc10babf0605a8afb454bc1296befceb87e1e8042d6a65e0478841654d`), see `BENCHMARK.md`.

The stock/16-bit cell of the card's `iso_seconds` grid (passes, 11,995 tokens) is an earlier
run on the same settings; it is not part of this battery and is not included here.

## Reproduce on a CPU

From the repository root, Python 3.12 and nothing else:

```sh
A=evidence/ft5-publication-battery/acceptance
python3 bench/analysis/analyse_served_accept.py $A/general-1-stock.json      $A/general-2-ft5.json
python3 bench/analysis/analyse_served_accept.py $A/general-4-stock.json      $A/general-3-ft5.json
python3 bench/analysis/analyse_served_accept.py $A/code-1-stock.json         $A/code-2-ft5.json
python3 bench/analysis/analyse_served_accept.py $A/general-long-1-stock.json $A/general-long-2-ft5.json
```

Rerun on these sanitized files, each prints `"gate": "pass"` and exits 0:

| pair | stock | ft5 | gain | paired 95% | classes |
|---|---|---|---|---|---|
| general, 200, forward | 2.7768 | 3.0403 | +9.49% | +7.25% to +11.69% | 23 identical, 17 boundary overshoot |
| general, 200, reversed | 2.7768 | 3.0403 | +9.49% | +7.25% to +11.69% | 23 identical, 17 boundary overshoot |
| code, 200 | 3.8166 | 4.2130 | +10.39% | +7.80% to +13.13% | 26 identical, 14 boundary overshoot |
| general, 1024 | 2.8149 | 3.0399 | +7.99% | +6.75% to +9.24% | 34 identical, 6 boundary overshoot |

No prefix divergence and no early-stop mismatch in any pair; maximum overshoot 7 tokens;
truncated 34/34, 34/34, 25/25 and 13/13. The HTTP figure is total completion tokens over total
decode seconds per arm, across its two legs:

```sh
python3 - <<'PY'
import json
H = "evidence/ft5-publication-battery/http/greedy-leg"
def pooled(*legs):
    rows = [r for leg in legs for r in json.load(open(f"{H}{leg}.json"))["requests"]]
    return sum(r["completion_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows)
stock, ft5 = pooled("1-stock", "4-stock"), pooled("2-ft5", "3-ft5")
print(f"stock {stock:.4f}  ft5 {ft5:.4f}  {ft5 / stock - 1:+.2%}")
PY
```

prints `stock 25.0918  ft5 28.9295  +15.29%`. The quality scores are in each quality record's
`scores` and `results`, the tool tallies in `tally`.

## What was stripped

Run timestamps (`provenance.when`, and the clock time in `provenance.machine.uptime`, which
went with its load averages), date-prefixed file names and dated run labels, the private
repository revision and worktree flag (`provenance.git`), the host process list
(`provenance.machine.top_rss`), the server log path (`provenance.kv_cache.source`) and
mlx-dspark's calibration cache keys (`provenance.kv_cache.calibration_keys`, which named a
local drafter directory). In strings: the local drafter directory became
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, the interpreter and launcher paths in
`provenance.launch` became `python bin/mlx-dspark-patched`, and the local API key became
`REDACTED`. Every number is unchanged; `../tools/verify_sanitized.py` proved it pair by pair.
Kept: package versions, the server's health and configuration, memory and swap readings, power
and thermal state, every per-request measurement and the model's own outputs.

## What was left out

- The run's supervisor status file and its manifest. Both are keyed by local paths and consist
  mostly of commands and machine-state dumps; the measurements they point at are the files
  above. The machine state each measurement ran in is kept inside every HTTP and quality
  record (`provenance.server.system`, `provenance.machine`).
- The 110k-token depth record of the same night: the card makes no 110k speed claim.
- The prompt corpora (identified by hash above), the drafter weights (published on the Hub,
  sha256 `63399215d7af9b7aaa73464e7e10835d608ae4d9571abf520040b2b8746455b3`), the server logs,
  an isolated mlx-dspark 0.19.0 compatibility check, and the draft card and licence review
  that preceded publication.
