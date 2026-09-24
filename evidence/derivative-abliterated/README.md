# A derivative target: BoldingBuilds' abliterated Bonsai 2

Does ft5, fine-tuned against PrismML's Ternary Bonsai 2 27B, still help on a model derived from
it? This measures stock and ft5 on
[`BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF`](https://huggingface.co/BoldingBuilds/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP-GGUF)
at `25bdc69496884b958428e722e2c3d15bf0e3857d`, file `Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf`
(7,206,168,928 bytes, sha256 `4915a0df1ffa0d73e5c004780ad83acf57e5b134b60e971e0769be4fb3159f9d`,
Apache-2.0). Its card says it flips ternary digits in 98 of the 851 tensors and never touches a
block scale; `transcode/stream.json` confirms that byte by byte.

**Protocol status.** Not a `BENCHMARK.md` arm. It uses the same harness, suites and settings as
the publication battery (mlx-dspark 0.18.0 with this repository's patches, cap 7, 4-bit drafter,
8-bit KV, greedy, 40 prompts per suite at 200 tokens), with only the target changed.

## How the target was run

mlx-dspark reads PrismML's MLX pack, not GGUF, so the derivative was transcoded into that pack
with the converter PrismML ships inside it (`runtime/codec.py` in
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit@3f926b41`, sha256 `7f7fd676…`):

1. **The converter is lossless on this data** (`transcode/lossless.json`, `permutation.json`).
   On PrismML's own PQ2_0 GGUF, 19 of 21 sampled tensors, including every kind the derivative
   edits, convert byte-identical to PrismML's pack. The other two, the linear-attention input
   projections, differ only by an exact reordering of 128-row head blocks, which the
   derivative does not edit.
2. **The derivative edits what its card says** (`transcode/stream.json`). Its header and 753
   tensors are byte-identical to PrismML's GGUF; 98 differ (`ffn_down` 49, `ssm_out` 36,
   `attn_output` 13, blocks 15 to 63), with 0.197% to 0.271% of each one's digits flipped,
   14,094,798 in all, and no block scale changed.
3. **The pack is PrismML's with those 98 tensors replaced** (`transcode/build.json`). Only
   their `.weight` codes changed; every other tensor range and all 22 other files match
   PrismML's pack. The resulting `model.safetensors` has sha256
   `93503ce47022201fa4186451ace9f21463412a55f3e66b93fc5ad60bd10a2d48`.

The pack is not published, and neither are the scripts that built it. The reports call the
target `pack-abliterated`.

## Files

| path | what it is |
|---|---|
| `reports/{general,code}-{stock,ft5}.json` | four per-prompt reports on the derivative, with `round_lengths`, written by the author's development copy of `served_accept.py`, which writes the same fields as `bench/drafter/served_accept.py` without its two input checks |
| `analysis/{general,code}-stock-to-ft5.json` | the published analyser's `compare` output on the derivative, unchanged |
| `analysis/{general,code}-base-vs-derivative.json` | `cross_target.py` output: each drafter from base to derivative, and ft5's gain on the derivative minus its gain on the base |
| `cross_target.py` | the script that wrote those two files, standard library only |
| `transcode/*.json` | the three checks above |

The base-target reports are the publication battery's
`ft5-publication-battery/acceptance/{general-1-stock,general-2-ft5,code-1-stock,code-2-ft5}.json`.

## Claims and how to reproduce them

```sh
G=evidence/derivative-abliterated; A=evidence/ft5-publication-battery/acceptance
for s in general code; do
  python3 bench/analysis/analyse_served_accept.py $G/reports/$s-stock.json $G/reports/$s-ft5.json
done
python3 $G/cross_target.py $A/general-1-stock.json $A/general-2-ft5.json $G/reports/general-stock.json $G/reports/general-ft5.json
python3 $G/cross_target.py $A/code-1-stock.json $A/code-2-ft5.json $G/reports/code-stock.json $G/reports/code-ft5.json
```

| suite | stock, base → derivative | ft5, base → derivative | stock → ft5 on the derivative | on the base | difference |
|---|---|---|---|---|---|
| general | 2.777 → 2.777, +0.01% [−2.84, +2.89] | 3.040 → 2.981, −1.95% [−4.78, +0.96] | **+7.34% [+5.89, +8.77]** | +9.49% | −2.15 points [−4.60, +0.22] |
| code | 3.817 → 3.668, −3.89% [−8.34, +0.74] | 4.213 → 4.039, −4.14% [−8.70, +0.70] | **+10.09% [+5.65, +14.63]** | +10.39% | −0.29 points [−4.54, +4.23] |

Both stock → ft5 pairs pass the analyser's gate: 26 of 40 outputs identical and 14 differing
only above the budget, none diverging inside it. So on this derivative ft5 keeps 77% (general)
and 97% (code) of its base-target gain, and neither loss is significant.

## What this does not show

- **One light derivative.** It flips about 0.05% of the model's ternary digits. A derivative
  that edits more, or is requantized, may move the hidden states the drafters read much
  further, and needs its own measurement.
- **Why acceptance moved.** The derivative's greedy output differs from the base target's on
  all 40 prompts in both suites (the first differing token has a median position of 14 in
  general and 30 in code), so base → derivative changes the text being drafted as well as the
  model. The two cannot be separated here.
- **n = 40.** The general gain's difference has an interval that barely reaches zero; more
  prompts would settle it.

## What was stripped

Paths only: the local pack directory became `pack-abliterated`, the drafter's local directory
its public name, Hub cache paths `org/name@revision/file`, and the byte diffs' local directory
was dropped from their file names (the diffs are not published). No field was removed, and
`verify_sanitized.py pairs` passed on all 8 files. The analyses were written by the
published scripts from the files here.
