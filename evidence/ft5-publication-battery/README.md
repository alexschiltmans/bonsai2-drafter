# ft5 publication battery

The paired battery behind the ft5 model card's acceptance, throughput, output-agreement and
quality tables: stock `z-lab/Qwen3.8-27B-DFlash2` against
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, on mlx-dspark 0.18.0 with this repository's
patches, an 8-bit KV cache, the drafter quantized to 4 bits at load, cap 7, one Apple M4 Pro
with 48 GB.

Three subdirectories hold records the card quotes from outside the paired battery, each marked
as such below: `depth/` (the depth table), `quality-history/` (the quality runs behind the
card's `iso_seconds` history, including the stock/16-bit cell of its grid) and
`compatibility/` (the mlx-dspark 0.19.0 slice).

**Which harness wrote these records.** The acceptance reports are `served_accept.py --report`
records, read by this repository's analyser. The HTTP legs, the quality and tool records and
the depth records were written by an unpublished version of the author's measurement harness:
its throughput runner asks the same five prompts as the public `bench/throughput/bench5.py`
but writes a different record layout, and its quality battery (`quality.py`), tool battery
(`tool_battery.py`) and depth runner are not in this repository. `../README.md` has the
details.

## Files

| path | what it is | card claim it supports |
|---|---|---|
| `acceptance/general-{1-stock,2-ft5,3-ft5,4-stock}.json` | raw `served_accept.py --report` records, general suite, 200 tokens, run in the order 1-4 (stock, ft5, ft5, stock) | general chat 2.7768 → 3.0403 tokens a round, **+9.49%**, paired 95% +7.25% to +11.69%, forward (1→2) and reversed (4→3) |
| `acceptance/code-{1-stock,2-ft5}.json` | the same, code suite, 200 tokens | held-out code 3.8166 → 4.2130, +10.39% (+7.80% to +13.13%) |
| `acceptance/general-long-{1-stock,2-ft5}.json` | the same, general suite, 1024 tokens | 2.8149 → 3.0399, +7.99% (+6.75% to +9.24%) |
| `acceptance/acceptance-summary.json`, `long-acceptance-summary.json` | the version 1 analyses, unchanged, `investigate` verdicts included | the history behind the v2 contract |
| `reanalysis-v2/*.json`, `NOTES.md` | the four pairs under `budgeted-prefix-identity/v2`, and the two same-drafter identity controls | "zero differing tokens in 160 paired responses", overshoot at most 7 tokens, controls token-identical on 40/40 |
| `http/greedy-leg{1-stock,2-ft5,3-ft5,4-stock}.json` | the four HTTP throughput legs (the unpublished harness's five-prompt runner), ABBA, three repetitions of five prompts at 400 tokens, greedy | **25.0918 against 28.9295 tok/s, +15.29%**; legs 25.087, 28.925, 28.934, 25.097; every request at cap 7 |
| `quality/quality-{stock,ft5}.json` | the unpublished harness's quality battery, 14 tasks, three graders, one executable | 13/14 in both arms; `iso_seconds` fails in both at 16,391 and 16,389 tokens |
| `quality/tools-{stock,ft5}.json` | the unpublished harness's tool battery at depths 2500 and 13000, four repetitions | 56/56 in both arms |
| `quality-controls/quality-nodrafter-kv8.json` | no drafter, 8-bit KV | `iso_seconds` fails, 16,384 tokens, no answer |
| `quality-controls/quality-nodrafter-kv16.json` | no drafter, unquantized (16-bit) KV (`--kv-bits 0`) | passes, 2,232 tokens |
| `quality-controls/quality-ft5-kv16.json` | ft5, unquantized (16-bit) KV | passes, 13,417 tokens, 14/14 |
| `depth/{stock,ft5}-{35k,65k,110k}.json` | depth records, outside the battery except `ft5-110k.json` | the depth table: 16.3 against 17.4 tok/s at 35k, 12.6 against 13.1 at 65k, 8.9 stock at 110k, and the ft5 110k arm |
| `quality-history/*.json` | nine quality runs outside the battery | the `iso_seconds` history, and the grid's stock/16-bit cell (`bonsai-stock-kv16-greedy.json`) |
| `compatibility/mlx-dspark-0.19.0.txt` | the log of an isolated mlx-dspark 0.19.0 check | the Limitations bullet on 0.19.0 |

Each acceptance record holds, per prompt: the prompt's sha256, category, thinking flag, token
and round counts, decode seconds, finish reason and the **full output token array**. The prompts
themselves are identified by hash; the corpora are `general.jsonl`
(`3dcc1327c0d254e2191322503f6cc4fe0cc464389ef0799093778ca55996c50b`) and `code.jsonl`
(`4252b5bc10babf0605a8afb454bc1296befceb87e1e8042d6a65e0478841654d`), see `BENCHMARK.md`.

The stock/16-bit cell of the card's `iso_seconds` grid (passes, 11,995 tokens) is not part of
this battery. It is a run with the battery's server flags except that it leaves `--kv-bits`
unset (the unquantized default, where the controls pass `--kv-bits 0`), and it is in
`quality-history/bonsai-stock-kv16-greedy.json`.

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

## Depth, mostly outside the battery

The card's depth table. `ft5-110k.json` ran with the battery, from the same launcher; the other
five are arms outside it, run with the same server flags (8-bit KV, the drafter quantized to 4
bits at load, a 131,072-token context window). The ft5 arms ran the published weights (sha256
`63399215…`). Each record is one session: the long prompt is prefilled once, then five requests
of up to 400 tokens are served against it, and the record keeps each request's tokens, decode
seconds, cache state and the server's own `spec` block. The prompt itself (filler text with
four planted facts, the "needles") is not in the records; the needles, the questions' expected
answers and the model's answers are, in `needle_log`.

| file | prompt tokens | warm decode tok/s | all served tok/s | warm requests | miss rate | sheds (anchor pass) | restore declines | peak memory |
|---|---|---|---|---|---|---|---|---|
| `depth/stock-35k.json` | 34,931 | 16.302 | 16.302 | 5/5 | 0.0 | 0 | 8 | 20.89 GB |
| `depth/ft5-35k.json` | 34,931 | 17.441 | 17.441 | 5/5 | 0.0 | 0 | 8 | 20.89 GB |
| `depth/stock-65k.json` | 65,188 | 12.649 | 12.649 | 5/5 | 0.0 | 0 | 8 | 27.40 GB |
| `depth/ft5-65k.json` | 65,188 | 13.134 | 13.134 | 5/5 | 0.0 | 0 | 8 | 27.25 GB |
| `depth/stock-110k.json` | 110,226 | 8.866 | 8.866 | 5/5 | 0.0 | 0 | 8 | 37.11 GB |
| `depth/ft5-110k.json` | 110,226 | 8.919 | 8.711 | 2/5 | 0.6 | 4 | 3 | 37.98 GB |

Every arm served five of five requests with none failed and found all four needles, and every
request ran at cap 1 (`spec.cap`): the server allows 7 (`provenance.health.max_draft`), and the
cap controller narrowed the draft to one token at these depths. That is why the depth figures
are not comparable with the cap-7 acceptance counts. The rates are total completion tokens over
total decode seconds, over the warm requests (`warm_decode_tps`) or over all served requests
(`served_decode_tps`). At 35k ft5's warm rate is 6.99% above stock's (17.441 / 16.302), at 65k
3.83% (13.134 / 12.649). At 110k the two arms differ in more than the drafter: the ft5 arm
missed the prefix cache on three of its five requests after four memory-guard sheds during the
anchor pass (1.20 tok/s end to end, against 8.67 for stock), so its warm subset is not a matched
comparison and the card makes no 110k speed claim.

```sh
python3 - <<'PY'
import json
D = "evidence/ft5-publication-battery/depth/"
for arm in ("stock-35k", "ft5-35k", "stock-65k", "ft5-65k", "stock-110k", "ft5-110k"):
    d = json.load(open(f"{D}{arm}.json"))
    print(f"{arm:11s} warm {d['warm_decode_tps']:.3f}  served {d['served_decode_tps']:.3f}  "
          f"needles {d['needles_found']}/{d['needles_total']}  miss {d['miss_rate']}  "
          f"e2e {d['all_e2e_tps']:.2f}  caps {sorted({r['spec']['cap'] for r in d['requests']})}")
PY
```

## The `iso_seconds` history

These are all the quality runs of this harness that include the task, the records behind the
card's account of the failure: the nine in `quality-history/`, which ran outside this battery
(on other targets too: Qwen3.8-27B in four MLX precisions from `lmstudio-community`), and the
seven elsewhere in this bundle. Every run used a 16,384-token task budget.

| record | target | KV cache | drafter | decoding | `iso_seconds` | tokens | score |
|---|---|---|---|---|---|---|---|
| `quality-history/qwen-mlx6bit-kv16-sampled.json` | Qwen3.8-27B MLX 6-bit | 16-bit | stock | sampled | passes | 12,415 | 14/14 |
| `quality-history/qwen-mlx6bit-kv4-greedy.json` | Qwen3.8-27B MLX 6-bit | 4-bit | stock | greedy | passes | 14,726 | 14/14 |
| `quality-history/bonsai-stock-kv16-greedy.json` | Bonsai 2 | 16-bit | stock | greedy | passes | 11,995 | 14/14 |
| `quality-controls/quality-nodrafter-kv16.json` | Bonsai 2 | 16-bit | none | greedy | passes | 2,232 | 14/14 |
| `quality-controls/quality-ft5-kv16.json` | Bonsai 2 | 16-bit | ft5 | greedy | passes | 13,417 | 14/14 |
| `quality-history/qwen-mlx6bit-kv8-sampled-1.json` | Qwen3.8-27B MLX 6-bit | 8-bit | stock | sampled | fails | 16,387 | 13/14 |
| `quality-history/qwen-mlx6bit-kv8-sampled-2.json` | Qwen3.8-27B MLX 6-bit | 8-bit | stock | sampled | fails | 16,386 | 13/14 |
| `quality-history/qwen-mlx4bit-kv8-sampled.json` | Qwen3.8-27B MLX 4-bit | 8-bit | stock | sampled | fails | 16,391 | 12/14 |
| `quality-history/qwen-mlx8bit-kv8-sampled.json` | Qwen3.8-27B MLX 8-bit | 8-bit | stock | sampled | fails | 16,388 | 13/14 |
| `quality-history/qwen-mlx6bit-kv8-greedy.json` | Qwen3.8-27B MLX 6-bit | 8-bit | stock | greedy | **passes** | 14,777 | 14/14 |
| `quality-history/qwen-mlx5bit-kv8-greedy.json` | Qwen3.8-27B MLX 5-bit | 8-bit | stock | greedy | fails | 16,387 | 13/14 |
| `quality/quality-stock.json` | Bonsai 2 | 8-bit | stock | greedy | fails | 16,391 | 13/14 |
| `quality/quality-ft5.json` | Bonsai 2 | 8-bit | ft5 | greedy | fails | 16,389 | 13/14 |
| `quality-controls/quality-nodrafter-kv8.json` | Bonsai 2 | 8-bit | none | greedy | fails | 16,384 | 13/14 |
| `../prequantized-4bit/quality/quality-source.json` | Bonsai 2 | 8-bit | ft5 | greedy | fails | 16,389 | 13/14 |
| `../prequantized-4bit/quality/quality-artifact.json` | Bonsai 2 | 8-bit | ft5, 4-bit export | greedy | fails | 16,389 | 13/14 |

So: at 8-bit KV it fails in 10 of 11 runs, always by exhausting the budget (`finish_reason`
`length`, 16,384 to 16,391 tokens); at 4- or 16-bit KV it passes in all 5 (2,232 to 14,726
tokens). The runs span five target precisions (Qwen3.8-27B at MLX 4, 5, 6
and 8 bits, and the ternary Bonsai 2 pack) and both sampled and greedy decoding; the one 8-bit
pass is `qwen-mlx6bit-kv8-greedy.json`. Two of the 11 are the 4-bit variant's equivalence
arms, which ran the same ft5 weights; without them it is 8 of 9. Sampled means
temperature 0.7, top-p 0.95, top-k 20 (each record's `mode`).
Bonsai 2 is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`. The quality-history records ran with the
development tree's launcher (`python bin/mlx-dspark-patched`), whose patch set at each run is
recorded only as far as `provenance.health` shows it.

```sh
python3 - <<'PY'
import glob, json
B = "evidence/ft5-publication-battery/"
files = (sorted(glob.glob(B + "quality-history/*.json")) + sorted(glob.glob(B + "quality*/quality-*.json"))
         + sorted(glob.glob("evidence/prequantized-4bit/quality/quality-*.json")))
for f in dict.fromkeys(files):
    d = json.load(open(f))
    r = next(x for x in d["results"] if "iso_seconds" in x["prompt"])
    h = d["provenance"]["health"]
    print(f"{f.split('evidence/')[1]:70s} kv {h['kv_bits']}  {'pass' if r['passed'] else 'FAIL'} "
          f"{r['completion_tokens']:6d} {r['finish_reason']}")
PY
```

## mlx-dspark 0.19.0 slice

`compatibility/mlx-dspark-0.19.0.txt` is the log of one isolated check behind the card's
Limitations bullet: a fresh environment with mlx-dspark 0.19.0 (MLX 0.32.2 and mlx-lm 0.31.3,
as in the pinned stack; the full resolved package list is in the log), in which this
repository's patches installed (all six report `installed`), and then the development copies
of four tests this repository publishes in `bench/tests/` passed: `test_small_m_2bit.py` (the
2-bit verify kernel's source edit, packing, numerics and eligibility rules),
`test_bonsai_loader.py --with-models` (the Hadamard pack checks and the model loader, ending in
a greedy answer), `test_dflash_ft.py --with-models` (including the batched forward against the
served forward) and `test_kv_group_patch.py` (the KV patch). There is no 0.19.0 HTTP, quality
or throughput arm, so this is not serving equivalence.

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
and thermal state, every per-request measurement and the model's own outputs. The records in
`depth/` and `quality-history/` were sanitized the same way; three of them also lost the epoch
time of a memory-guard event (`last_shed.at`). In the 0.19.0 log, only
paths were rewritten, line by line with no line added or removed: the environment's local
path became `venv-019`, the interpreter `.venv-dspark/bin/python` (as `bench/check.sh` names
it), the Hub cache path `org/name@revision`, and the patches' log prefix, which carried the
private repository's name, reads `[bonsai2-drafter]`, as the published patches print it.

## What was left out

- The run's supervisor status file and its manifest. Both are keyed by local paths and consist
  mostly of commands and machine-state dumps; the measurements they point at are the files
  above. The machine state each measurement ran in is kept inside every HTTP and quality
  record (`provenance.server.system`, `provenance.machine`).
- The prompt corpora (identified by hash above), the drafter weights (published on the Hub,
  sha256 `63399215d7af9b7aaa73464e7e10835d608ae4d9571abf520040b2b8746455b3`), the server logs,
  the depth runner's filler text, and the draft card and licence review.
