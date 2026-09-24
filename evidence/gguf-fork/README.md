# ft5 as a GGUF, on PrismML's llama.cpp fork

The Q8_0 GGUF of ft5 (`Ternary-Bonsai-2-27B-DFlash2-ft5-Q8_0.gguf`, sha256 `70a7c6be…99c9`),
converted with llama.cpp's `convert_hf_to_gguf.py` carrying upstream's DFlash 2 support
(`--outtype q8_0`, tokenizer from `Qwen/Qwen3.8-27B`), and checked two ways. A diagnostic, not a
`BENCHMARK.md` measurement.

## Files

| path | what it is |
|---|---|
| `header-compare.json` | the GGUF's header against `z-lab/Qwen3.8-27B-DFlash2-GGUF@2d9571f8` and `ProCreations/Ternary-Bonsai-2-27B-DFlash2@4cfb6ad0` Q8_0 files, read over HTTP range requests: metadata differences and tensor-layout equality |
| `tensor-sha256.json` | the sha256, type and shape of each of the 81 tensors' data |
| `bench5-ft5-gguf.json` | the `bench5` record of one leg on the fork, unchanged |
| `machine-state.tsv` | memory pressure and swap before and after that leg |

## Claims

**Layout.** The file has the same 81 tensor names, shapes and types as z-lab's stock Q8_0 GGUF,
and the same metadata apart from the descriptive `general.*` fields; only the tensor order and
those fields differ (`header-compare.json`). Against ProCreations' file the tensors also match;
its metadata leaves out the chat template and one tokenizer flag.

**On the fork.** PrismML's fork at `0324c665` with upstream's DFlash 2 patched in, built as in
`fork-probe/`, with that probe's server flags and `-md` pointing at this file, on one Apple M4 Pro.
The stock figures are the probe's (`fork-probe/results-df2build.json`, arm `dflash2stock`); the
ft5 figures are this group's record (`pooled_decode_tps`, and per request `draft_n` and
`draft_n_accepted`).

| | stock drafter's GGUF (`fork-probe/`, two legs) | ft5's GGUF (one leg) |
|---|---|---|
| tokens per leg | 5,706 | 5,706 |
| drafts accepted | 0.346 | 0.405 (4,200 of 10,359) |
| pooled decode, tok/s | 7.10 | 7.97 |

The same token count is what identical greedy outputs would give; the texts themselves are not
recorded. The server log, not included, shows the file loading as a DFlash 2 drafter
with block size 8, mask token 248070 and five extracted target layers. This leg ran separately
from the probe's, not in ABBA order with it, so the comparison is a diagnostic; and both arms are
below the fork's 20.56 tok/s with no drafter on Metal (`fork-probe/`).

The `bench5` record is published as written; nothing in this group was edited.
