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
| [`ft5-publication-battery/`](ft5-publication-battery/) | the card's acceptance table (general chat +9.49%, paired 95% +7.25% to +11.69%; code +10.39%; 1024 tokens +7.99%), output agreement under `budgeted-prefix-identity/v2`, greedy decode +15.29%, quality 13/14 and tools 56/56 in both arms, and the `iso_seconds` drafter-by-KV grid |
| [`prequantized-4bit/`](prequantized-4bit/) | the 4-bit card's served equivalence (`artifact-equivalence/v1`, 40/40 on every suite), throughput ratio 1.0001, no newly failing quality fixture, 56/56 tools |
| [`sampled-screen/`](sampled-screen/) | a pre-registration screen on mlx-dspark: decode at the target's published sampling, 24.07 against 26.30 tok/s (+9.3%), and r3 4.63% below ft5 there |
| [`second-runtime/`](second-runtime/) | dflash-mlx-bonsai2: stock → ft5 +5.01% (+3.35% to +6.83%), r3 2.41% below ft5, and agreement with the target alone up to floating-point ties |
| [`fork-probe/`](fork-probe/) | PrismML's llama.cpp fork on Metal: no drafter 20.56, MTP 11.70, DFlash 2 7.10 tok/s |
| [`tools/verify_sanitized.py`](tools/verify_sanitized.py) | the checker described below |

Each group's README lists its files, quotes the claims they support, gives the commands that
reproduce those claims on a CPU with `bench/analysis/analyse_served_accept.py` (standard library
only), and says what was stripped and what was left out. The prompt corpora are identified by
sha256 in `BENCHMARK.md`; they are not part of this bundle.

## How these files were sanitized

These are sanitized copies. The originals are kept privately and unedited. Sanitizing removed
every timestamp (ISO times, epoch seconds, the clock time in an uptime line, dated labels and
date-prefixed file names, which were renamed to descriptive ones such as
`http/greedy-leg1-stock.json`); every local path, which became the public name of what it
pointed at (a drafter directory became `Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5`, a Hub
cache path became `org/name@revision/file`, a launcher became `python bin/mlx-dspark-patched`);
the private development repository's revision, the host process list, a server log path,
calibration cache keys that named a local directory, a local API key and a local build commit.
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
and searches the whole tree for the patterns sanitization removes. It needs nothing but this
directory.

`verify_sanitized.py pairs` is the check that proves no number changed. It walks each original
and its sanitized copy in parallel and fails unless every number, boolean and null is
identical in type and value at the same place, every list keeps its length, every deleted field
is one of the listed timestamp, path or provenance fields holding the kind of value that field
holds, and every rewritten string carries no number the original string did not (digits inside
a public model name excepted). It needs the originals, so only their keeper can run it; it was
run on all 111 JSON files in this bundle, and all 111 passed. It is published so the rules it
enforces can be read, and so anyone given an original can rerun it.

The Markdown files were written for this bundle. `ft5-publication-battery/reanalysis-v2/NOTES.md`
is a rewrite of the original notes with dates and local paths taken out.
