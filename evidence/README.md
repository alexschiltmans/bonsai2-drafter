# Evidence

The raw records behind the numbers on the two model cards
([`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5)
and its [4-bit variant](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit))
and in this repository's README: per-prompt reports with full output token arrays, HTTP
throughput records, quality records and the analyses. The author of these files trained ft5;
that is why the raw outputs are published rather than summarised, including every result where
ft5 loses.

| group | what it supports |
|---|---|
| [`ft5-publication-battery/`](ft5-publication-battery/) | the card's acceptance table (general chat +9.49%, paired 95% +7.25% to +11.69%; code +10.39%; 1024 tokens +7.99%), output agreement under `budgeted-prefix-identity/v2`, greedy decode +15.29%, quality 13/14 and tools 56/56 in both arms, and the `iso_seconds` drafter-by-KV grid. Five of the grid's six cells are the battery and its controls; the stock/16-bit cell is a run outside the battery, in `quality-history/`. Also, from outside the battery and marked as such: the depth table (`depth/`), the quality runs behind the `iso_seconds` history (`quality-history/`) and the mlx-dspark 0.19.0 slice (`compatibility/`) |
| [`prequantized-4bit/`](prequantized-4bit/) | the 4-bit card's served equivalence (`artifact-equivalence/v1`, 40/40 on every suite), throughput ratio 1.0001, no newly failing quality fixture, 56/56 tools, and the released files in mlx-lm's `load_model` and dflash-mlx-bonsai2, 153 tensors identical in both (`second-loaders/`) |
| [`sampled-screen/`](sampled-screen/) | a pre-registration screen on mlx-dspark: decode at the target's published sampling, 24.07 against 26.30 tok/s (+9.3%), and r3 4.63% below ft5 there |
| [`second-runtime/`](second-runtime/) | dflash-mlx-bonsai2: stock → ft5 +5.01% (+3.35% to +6.83%), r3 2.41% below ft5, and agreement with the target alone up to floating-point ties |
| [`second-runtime-throughput/`](second-runtime-throughput/) | decode speed on dflash-mlx-bonsai2 under `BENCHMARK.md` v1: ft5 / stock 1.0716 greedy, end to end (the server reports no decode timer) |
| [`fork-probe/`](fork-probe/) | PrismML's llama.cpp fork on Metal: no drafter 20.56, MTP 11.70, DFlash 2 7.10 tok/s |
| [`procreations-drafter/`](procreations-drafter/) | ProCreations' fine-tune on both MLX runtimes: it accepts more than ft5 on general chat (+3.90% [+2.62, +5.14] on dflash-mlx-bonsai2 under `BENCHMARK.md`, +1.83% in a matched comparison on mlx-dspark) and is level on code; no overlap between the eval prompts and its training corpus |
| [`derivative-abliterated/`](derivative-abliterated/) | ft5 on BoldingBuilds' abliterated derivative (98 of 851 tensors edited): stock → ft5 +7.34% [+5.89, +8.77] on general chat and +10.09% on code, not significantly below its gain on the base target |
| [`throughput-ft5-procreations/`](throughput-ft5-procreations/) | decode speed on mlx-dspark under `BENCHMARK.md` v1, code prompts: ProCreations / ft5 0.9494 greedy, 0.9821 sampled; ft5 accepts more on these prompts at the same round cost |
| [`mac-stacks-mtplx/`](mac-stacks-mtplx/) | the v2 stack comparison: MTPLX (MTP) against mlx-dspark with ft5 on one Mac, 1.3824 greedy and 1.4580 sampled end to end in MTPLX's favour |
| [`omlx-probe/`](omlx-probe/) | oMLX serves Bonsai 2 but not DFlash 2 on it; with a one-line routing change ft5 drafts losslessly but decodes about four times slower |
| [`gguf-fork/`](gguf-fork/) | ft5 as a Q8_0 GGUF: the same tensor layout as z-lab's stock GGUF, and on PrismML's fork with DFlash 2 patched in it drafts with 0.405 of drafts accepted against the stock GGUF's 0.346 (a diagnostic) |
| [`tools/verify_sanitized.py`](tools/verify_sanitized.py) | the checker described below |

Each group's README lists its files, quotes the claims they support, gives the commands that
reproduce those claims on a CPU with `bench/analysis/analyse_served_accept.py` (standard library
only), and says what was stripped and what was left out. The prompt corpora are identified by
sha256 in `BENCHMARK.md`; they are not part of this bundle, and are published as the dataset
[`Schiltmans/bonsai2-drafter-eval`](https://huggingface.co/datasets/Schiltmans/bonsai2-drafter-eval).

## Which programs wrote these records

- The per-prompt acceptance reports: on mlx-dspark, `served_accept.py --report`, in development
  versions of the script this repository publishes as `bench/drafter/served_accept.py` (the
  publication battery's reports lack the `round_lengths` field it writes); on
  dflash-mlx-bonsai2, `bench/adapters/dflash_mlx_bonsai2.py` (`second-runtime/`). The published
  analyser, `bench/analysis/analyse_served_accept.py`, reads both.
- The HTTP throughput legs, the quality and tool records and the depth records: an unpublished
  version of the author's measurement harness. Its throughput runner asks the same five
  prompts as the public `bench/throughput/bench5.py` (the records number them 1 to 5, in the
  order of that file's `PROMPTS`) at the same 400 tokens, but its record layout differs: it
  writes `mode`, `reps`, `pooled_decode_tps`, `pooled_e2e_tps`, `per_rep_decode_tps`,
  `requests` and `provenance`, where the public runner writes `runner`, `label`, `server`,
  `per_rep` and `clean`. Its quality battery (`quality.py`), tool battery (`tool_battery.py`)
  and depth runner are not in this repository, so those batteries cannot be rerun from it. What
  their records keep is enough to recompute every figure quoted from them: each quality task's
  prompt, the model's answer and reasoning and the grade; each tool request's expected tool,
  the model's text and calls and the verdict; each HTTP or depth request's tokens and timings;
  and the depth runs' needle answers (not their filler text).
- Anything else (the second-runtime analysis extras, the 4-bit loader checks, the fork probe's
  summaries) is described in its group's README.

## How these files were sanitized

These are sanitized copies. The originals are kept privately and unedited. Sanitizing removed
every timestamp (ISO times, epoch seconds, the clock time in an uptime line, dated labels and
date-prefixed file names, which were renamed to descriptive ones such as
`http/greedy-leg1-stock.json`); every local path, which became the public name of what it
pointed at (a drafter directory became `Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, a Hub
cache path became `org/name@revision/file`, a launcher became `python bin/mlx-dspark-patched`);
the private development repository's revision and name, the host process list, a server log
path, calibration cache keys that named a local directory, a local API key and a local build
commit. One file is a text log (`ft5-publication-battery/compatibility/mlx-dspark-0.19.0.txt`);
it was rewritten line by line, paths only, with no line added or removed.
Durations were kept: they are measurements. **No number was changed.** Where a source file was
mostly private machinery (supervisor status files, manifests keyed by local paths, server
logs), it was left out, and the group's README says so.

## Checking a bundle

Every group directory has a `SHA256SUMS` covering every file in it:

```sh
(cd evidence/ft5-publication-battery && shasum -a 256 -c SHA256SUMS)
python3 evidence/tools/verify_sanitized.py scan evidence
```

`scan` rechecks every group's `SHA256SUMS`, fails on any file a `SHA256SUMS` does not list,
and searches the whole tree for what sanitization removes: dates, epoch seconds, home-directory
paths and e-mail addresses. It needs nothing but this directory. Sanitization also removed the
names of private files and directories; those names are not written into the published
checker, because listing them there would publish them. Their keeper runs the same scan with
`--extra-patterns FILE`, a private list of regular expressions, one per line, and the bundle
passes it.

`verify_sanitized.py pairs` is the check that proves no number changed. It walks each original
and its sanitized copy in parallel and fails unless every number, boolean and null is
identical in type and value at the same place, every list keeps its length, every deleted field
is one of the listed timestamp, path or provenance fields holding the kind of value that field
holds, and every rewritten string carries no number the original string did not (digits inside
a public model name excepted). It compares the text log line by line under the same rules. It
needs the originals, so only their keeper can run it; it was run on all 150 sanitized data files
in this bundle (149 JSON files and the text log), and all 150 passed. Some files have no private
original: the analyses in `procreations-drafter/` and `derivative-abliterated/`, which the
published analyser and `derivative-abliterated/cross_target.py` wrote from files in this bundle,
and the `bench5` records in `throughput-ft5-procreations/` and `mac-stacks-mtplx/`, which are
published as `bench5` wrote them. It is published so the rules it
enforces can be read, and so anyone given an original can rerun it.

The Markdown files were written for this bundle. `ft5-publication-battery/reanalysis-v2/NOTES.md`
is a rewrite of the original notes with dates and local paths taken out.
