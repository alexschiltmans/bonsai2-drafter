# Fine-tuning the DFlash 2 drafter against the ternary target

Four scripts, run from the repository root with the pinned environment in `envs/dspark/`:

```
gen_data.py       the ternary target's own responses to CodeAlpaca prompts, exact token ids,
                  appended one per line; a rerun skips finished prompts. --temperature 0 gives
                  the target's argmax path and records each served round's length, which is
                  where the loop drew its anchors
dflash_ft.py      the library: the batched draft forward (a re-statement of mlx-dspark's, checked
                  against it), the distillation and selector losses, uniform and served anchors,
                  the proxy evaluation, checkpoints and export. Typed; bench/tests/test_dflash_ft.py
train_drafter.py  the loop around it: schedule, --anchor-mode uniform|served|mixed, checkpoints
                  every N sequences and on SIGTERM, resume, --export-only, --eval-only, --self-test
served_accept.py  the served loop itself on a corpus's prompts: mean accepted tokens a round.
                  The gate. The trainer's proxy prints two accept figures: "expected accept"
                  multiplies marginal slot accuracies and understates the loop when slot one is
                  weak (hits are correlated across slots); "prefix accept" is the mean accepted
                  prefix per anchor, and with --eval-mode served is the number to compare here
```

The corpus and the fine-tuned drafters are not committed: the corpus is this machine's
samples and a drafter is 3.8 GB. What is committed is the code. The released drafter is on
Hugging Face as `Ternary-Bonsai-2-27B-DFlash2-ft5`, and its card carries the measurements.

## Reproduce Iteration Five

Run from the repository root, with no server, trainer or other GPU workload running.
Keep the pinned dspark environment; do not upgrade it between arms. Set `RUN` to a
new local run directory and create it before starting. These are the successful recipe's
commands, not the old drivers' fallback generation command (which omitted greedy sampling):

```bash
PY=$HOME/.venv-dspark/bin/python
STOCK=z-lab/Qwen3.8-27B-DFlash2
$PY -u bench/drafter/gen_data.py "$RUN/greedy.jsonl" --n 300 --eval 40 --seed 7 --temperature 0
$PY -u bench/drafter/train_drafter.py "$RUN/greedy.jsonl" "$RUN/ft5" --ckpt-every 10 --lr 1e-4 --epochs 2 --anchor-mode served --eval-mode served
```

The source is `sahil2801/CodeAlpaca-20k` (CC BY 4.0). Generation uses the stock drafter,
KV8, cap 7, thinking on for two of every three prompts, 1024 output tokens with thinking
and 400 without. The last 40 sampled prompts are held out. Preserve the exact corpus
and its hash: regenerating from a mutable dataset/model revision is not an exact replay.
Training starts from stock, not iteration four. Defaults are rank 64, 48 anchors, two
steps per sequence, top-32 distillation, selector projection trained but codebooks frozen,
seed 0, 2048-token truncation, AdamW without weight decay, 4% warmup and cosine decay.
The recorded run performed 1032 steps; do not infer completed steps from scheduled steps.

Training exports fused bfloat16 weights when complete. Resume by repeating the identical
training command. To export an existing checkpoint without more training:

```bash
$PY -u bench/drafter/train_drafter.py "$RUN/greedy.jsonl" "$RUN/ft5" --export-only
$PY -u bench/drafter/served_accept.py "$RUN/greedy.jsonl" "$STOCK" --n 40 --max-new 200 --report "$RUN/code-stock.json"
$PY -u bench/drafter/served_accept.py "$RUN/greedy.jsonl" "$RUN/ft5" --n 40 --max-new 200 --report "$RUN/code-ft5.json"
python3 bench/analysis/analyse_served_accept.py "$RUN/code-stock.json" "$RUN/code-ft5.json"
```

The served gate uses a 4-bit drafter and pinned cap 7. It is not the depth profile's
derived-cap measurement. Keep the full logs, exit status and per-prompt reports, not
just lines filtered for `SERVED`. A success marker belongs after a successful exit.
Run `bench/check.sh --models` separately, never alongside a measurement workload.
The HTTP throughput gate (a fresh server per arm, stock and fine-tuned in ABBA order under
like-for-like machine state) uses a runner that is not in this repository yet; BENCHMARK.md
describes it.

The publication battery's reviewed evidence (the raw acceptance reports with their
output token arrays, the hashes and the versioned reanalysis) will be published in this
repository under `evidence/`. Until then, its tables are on the model card.

## Non-Code Gate

`general_chat.json` is a fixed, locally authored screening set: eight categories with
five prompts each, balanced 20/20 thinking on/off. It is deliberately distinct from
CodeAlpaca, but is not a representative sample of all chat traffic or a quality benchmark.
Freeze it before evaluating either arm; do not tune against its results.

```bash
$PY -u bench/drafter/gen_data.py "$RUN/general.jsonl" --prompts-json bench/drafter/general_chat.json --temperature 0 --n 40 --eval 40
$PY -u bench/drafter/served_accept.py "$RUN/general.jsonl" "$STOCK" --n 40 --max-new 200 --report "$RUN/general-stock.json"
$PY -u bench/drafter/served_accept.py "$RUN/general.jsonl" "$RUN/ft5" --n 40 --max-new 200 --report "$RUN/general-ft5.json"
python3 bench/analysis/analyse_served_accept.py "$RUN/general-stock.json" "$RUN/general-ft5.json"
```

The predeclared gate requires identical greedy output token IDs *within the requested
budget* and a positive lower bound of a 95% paired prompt-bootstrap interval for pooled
tokens-per-round gain. Resampling is stratified by category and thinking; it quantifies
variation within this small suite, not general-domain coverage. The budget qualifier is
contract `budgeted-prefix-identity/v2`, written into every summary the analyser produces:
the generator tests its budget once per round and then commits a whole block, so a
200-token request can return 207 tokens whose first 200 are identical. The analyser
classifies each unequal pair — boundary overshoot above the budget is arithmetic and
passes, divergence inside the budget or a mismatched early stop is a finding and blocks.
A blocking class is a numerical investigation, not automatically a quality regression. Reports retain truncations, output IDs, round
counts and durations. At 200 tokens many thinking responses are truncated: this times
prefixes, not complete answers. Reverse arm order before making a speed claim; inspect
per-category and thinking strata before making a broad acceptance claim. Do not publish
weights on the strength of the code-only gate.

## The Prequantized Variant

`export_quantized.py` writes the 4-bit form of a bfloat16 DFlash 2 drafter once, instead of
`load_dflash` quantizing it on every start. The bits that reach the GPU are the same either
way: this is a distribution-size change, not a fine-tune and not a speed claim. Its
gates are the checks below.

The artifact only loads through this repository's patches. `patches/dflash_prequantized.py`
builds the model, quantizes exactly the modules the pinned loader's rule selects, and then
loads the packed tensors into them; nothing requantizes. An unmarked bfloat16 checkpoint
still goes through the original loader untouched.

```bash
PY=$HOME/.venv-dspark/bin/python
SOURCE=$HOME/models/Ternary-Bonsai-2-27B-DFlash2-ft5  # the bf16 drafter
ARTIFACT=$RUN/ft5-4bit                       # must not exist; the exporter never replaces one

"$PY" bench/drafter/export_quantized.py --source "$SOURCE" --output "$ARTIFACT"
"$PY" bench/tests/test_dflash_prequantized.py --source "$SOURCE" --artifact "$ARTIFACT"
```

The exporter stages the whole artifact in a directory it creates for that one run beside the
destination, reloads it there through the patched loader, requires every tensor to come back
identical, and only then renames it into place. A failure removes that staging directory,
and only that directory, leaving no destination behind: a half-written release is not one of
the outcomes. There is no flag to skip the reload. It is not a redundant comparison —
rebuilding the config, agreeing with the recorded module set and loading the tensors
strictly all happen inside it, so skipping it would publish a directory nothing had ever
loaded.

The equivalence test is three tiers and the first two need no artifact at all:

```bash
python3 bench/tests/test_dflash_prequantized.py            # the format's rules; no mlx, no GPU
"$PY"   bench/tests/test_dflash_prequantized.py --gpu      # a synthetic export and reload
"$PY"   bench/tests/test_dflash_prequantized.py --source "$SOURCE" --artifact "$ARTIFACT"
```

The first two are registered in `bench/check.sh` (no-GPU tier and `--gpu`). The third is
not, because it needs an artifact that does not exist until someone exports one. It compares
the source quantized at load time against the saved artifact loaded through the patch —
parameter names, shapes, dtypes and exact values, module classes, bits/group/mode and the
DFlash configuration — then repeats the load in a fresh process that `sandbox-exec` has
denied all access to the source directory, which is how "the artifact loads independently of
the source" is checked without deleting or moving the canonical checkpoint.

For the served loop, run `served_accept.py` once per artifact and compare the two reports
under the *equivalence* criterion, not the improvement one:

```bash
"$PY" -u bench/drafter/served_accept.py "$CORPUS" "$SOURCE"   --n 40 --bits 4 --max-new 200 --report "$RUN/source.json"
"$PY" -u bench/drafter/served_accept.py "$CORPUS" "$ARTIFACT" --n 40 --bits 4 --max-new 200 --report "$RUN/artifact.json"
python3 bench/analysis/analyse_served_accept.py "$RUN/source.json" "$RUN/artifact.json" --equivalence
```

`--equivalence` requires equal per-prompt token arrays, finish reasons, token counts, round
counts and round-length arrays; decode durations are exempt and no speed claim follows from
them. Zero gain is the expected and passing result here, which is exactly why the default
improvement gate is the wrong instrument: it reads an exactly-equal pair as `investigate`,
because its criterion is a positive lower bound. Do not quote one gate's verdict for the
other's question.
