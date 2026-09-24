# One drafter, several runtimes, two methods

I fine-tuned a DFlash 2 drafter for PrismML's Ternary-Bonsai-2-27B on one Mac. It's called ft5,
and it drops in wherever z-lab's stock drafter does. On the runtime I trained it for, it raises
acceptance on general chat by 9.49% and greedy decode speed by 15.3%. On a second runtime that
shares none of that code, the acceptance gain is 5.01%. An independent fine-tune by ProCreations,
measured the same way, accepts more than ft5 on general chat and matches it on code. On the Metal
path of PrismML's llama.cpp fork, at the head I tested, no DFlash 2 drafter can speed up the
target, and MTP can't either.

## What was built

Bonsai 2 is Qwen3.8-27B with its weights stored at 2 bits in a Hadamard-rotated basis. DFlash 2,
from z-lab and Inco AI, drafts a block of tokens in parallel from the target's hidden states. The
target checks the block in one verify pass and keeps the longest correct prefix plus one token of
its own, so the drafter decides how fast the target answers, never what it answers.

The stock drafter was trained against full-precision Qwen3.8-27B and already works on the ternary
target: served unchanged on one code request, it accepted 5.0 tokens a round (my own measurement,
not in the evidence bundle). The question was whether tuning it to the ternary hidden states
would do better.

The unchanged target answered 300 CodeAlpaca prompts greedily, 260 for training and 40 held out.
Training distils the target's top-32 distribution into a rank-64 LoRA, trains the selector
projection with its codebooks frozen, and runs two epochs, 1032 steps, in MLX on one Apple M4 Pro
with 48 GB. The other public DFlash 2 trainers I found target CUDA.

## Three fine-tunes that served worse

My first three fine-tunes looked better on the trainer's proxy and served worse. The cause was
the shape of the training examples. In serving, the drafter starts where the last round stopped:
the target has just emitted a bonus token it hasn't read yet, the drafter's context holds hidden
states for the positions before the anchor, and the block starts at the anchor with that unread
token at its head. My training mask let the block see one more row, the target's hidden state
after reading the anchor token. That row is the target's own prediction of the next token, which
is the label for slot one. Every fine-tune learned to read it. Iteration three's slot one reached
0.918 accuracy teacher-forced, against the stock drafter's 0.764 (from my training evaluations,
which are not in the evidence bundle), and serving never provides that row. The proxy read the
same leaked row, so it agreed with training.

The proxy's acceptance figure had a second problem. It multiplied per-slot accuracies as though
slots missed independently. They don't: a position that's easy tends to be easy for every slot,
so a weak slot one drags the product down far more than it drags down the loop, which commits one
token plus the run of hits from slot one. The right figure is the mean accepted prefix per
anchor. For the stock drafter at the anchors the loop actually drew, the product read 2.40, the
prefix 3.16, and the loop itself 3.82 tokens a round. The loop's figure matches the publication
battery's stock arm on the held-out code suite (3.8166, in
`evidence/ft5-publication-battery/acceptance/`); the two proxy figures come from my training
evaluations and are not in the evidence bundle.

With the context ending before the anchor, anchors taken from served rounds and the prefix figure,
proxy and loop agreed in rank and roughly in ratio (0.79 against 0.81 for iteration three over
stock). Iteration four was iteration three's recipe on the corrected geometry; ft5 is iteration
four trained for two epochs.

| drafter | old proxy | corrected proxy | served loop, 40 held-out code prompts |
| --- | --- | --- | --- |
| stock | 3.26 | 4.24 | 3.82 |
| iteration 2 | 3.69 | – | 2.74 |
| iteration 3 | 4.13 | 3.36 | 3.08 |
| iteration 4 | – | 4.51 | 4.08 |
| iteration 5 (ft5) | – | 4.60 | 4.21 |

Tokens a round throughout. Only the stock and ft5 cells of the served column are backed by the
evidence bundle: they match the publication battery's code-suite arms, 3.8166 and 4.2130. Every
other cell, and the 0.79 against 0.81 above, comes from my own training and served-loop runs
outside it. The old proxy kept iterations two and three in their served order but put both above
the stock drafter, which served better than either of them. What I took from it: build the proxy
from exactly what the loop hands the drafter, and keep the served loop (`served_accept.py` in the
repository) as the gate.

## How I measured

At temperature 0 the token path belongs to the target, so two drafters have to emit the same
tokens. On mlx-dspark they did: across 160 paired responses on four suites, zero tokens differed
inside the requested budget, and two same-drafter control pairs were token-identical on all 40
prompts. That's agreement on these suites, which is as far as I'd take it.

My first identity gate failed anyway, with 17, 17, 14 and 6 mismatched pairs, all of them the
generator's arithmetic. mlx-dspark checks the output budget once per round and then commits a
whole block. A round that starts at 199 tokens can commit seven drafts plus the target's token, so
a 200-token request returns up to 207 tokens whose first 200 are identical. The analyser
states this as the contract `budgeted-prefix-identity/v2`: compare the first `max_new` tokens,
keep the raw arrays, and classify each unequal pair. Overshoot above the budget passes; a
divergence inside it, or a mismatched early stop, blocks. A control confirmed it from the
other side: with no drafter the loop commits one token per forward pass, and it stopped at exactly
16,384 tokens where the drafter arms stopped at 16,391 and 16,389.

Acceptance counts every committed token over every round, including the round that crosses the
budget. Counting only the part of that round inside the budget (a trimmed numerator) moved the
gain by under 0.4 points in every pair, at most from +10.39% to +9.99% on code. Letting each arm
count that round whole or drop it, independently, gives a looser envelope, as wide as +8.81% to
+11.79% on code. Both arms see the same 40 prompts
per suite, and each interval is a paired prompt bootstrap stratified by category and by thinking
on or off, which measures variation within a small frozen suite and nothing wider. Throughput legs
run in ABBA order (stock, ft5, ft5, stock), each on a fresh server, pooled as total tokens over
total decode seconds.

`BENCHMARK.md` fixed the arms, the corpora by hash, the metrics and the deciding rule before the
comparisons with other drafters ran. Drafter A beats B on a runtime only if the lower bound of the
paired 95% interval on the general suite at 200 tokens is above zero. The code suite chose ft5's
training iteration, so it never decides. No arm gets rerun to change its result.

## Results on two runtimes

mlx-dspark 0.18.0 ran with the repository's patches, an 8-bit KV cache and a 4-bit drafter at
cap 7. naklitechie's dflash-mlx-bonsai2, a fork of bstnxbt/dflash-mlx with its own Bonsai loader
and 2-bit verify kernel, ran at commit `223e0f3` with its default unquantized KV cache and a 4-bit
drafter. It clamps DFlash 2 to a block of five, a cap of 4, and loaded ft5 unchanged. r3 is
naklitechie's independent fine-tune against the same target, trained on CUDA, and PC is
ProCreations', released as a Q8_0 GGUF together with the bf16 file it was made from.

| general chat, 200 tokens unless noted | mlx-dspark, cap 7 | dflash-mlx-bonsai2, cap 4 |
| --- | --- | --- |
| stock → ft5, tokens a round | 2.7768 → 3.0403 | 2.6392 → 2.7714 |
| stock → ft5, gain (paired 95%) | +9.49% (+7.25% to +11.69%) | +5.01% (+3.35% to +6.83%) |
| stock → ft5, held-out code | +10.39% (+7.80% to +13.13%) | +7.10% (+5.25% to +9.03%) |
| stock → ft5, general at 1024 tokens | +7.99% (+6.75% to +9.24%) | +4.84% (+3.59% to +6.03%) |
| stock → r3 | +4.42% (+3.18% to +5.69%) | +2.48% (+1.70% to +3.28%) |
| r3 against ft5 | −4.63% (−6.39% to −2.72%) | −2.41% (−3.83% to −1.08%) |
| stock → PC | +11.50% (+9.95% to +13.04%) | +9.10% (+7.95% to +10.26%) |
| PC against ft5 | **+1.83% (+0.26% to +3.54%)** | **+3.90% (+2.62% to +5.14%)** |
| PC against ft5, held-out code | −0.85% (−3.14% to +1.26%) | −0.10% (−1.94% to +1.61%) |

The general suite is 40 frozen chat prompts, deliberately unlike CodeAlpaca, and ft5 only ever
trained on code. All drafters produced identical greedy output within the budget on both runtimes.
Against the target running alone on dflash-mlx-bonsai2, outputs differed only at floating-point
ties (logit margins ≤ 0.016), at the same positions for every drafter. The mlx-dspark comparison
with r3 was a screen on ft5's home runtime, with r3 loaded from a copy whose codebooks I had
renamed, which is why I repeated it on r3's own runtime.

PC's bf16 file loaded unchanged on both runtimes, and it accepts more than ft5 on general chat on
both. On dflash-mlx-bonsai2, where every arm ran under `BENCHMARK.md`, that means it beats ft5 by
the deciding rule. The mlx-dspark row pairs it with reports from before the protocol, so it is a
matched comparison rather than a protocol result. On code, the suite ft5 was selected on, the two
are level. None of my evaluation prompts appears in the training corpus ProCreations publishes.

Three independent fine-tunes against the ternary target beat the stock drafter on both runtimes,
and mine is not the best of them on general chat. That the gain reproduces across trainers and
runtimes matters more to me than the ranking.

Why is ft5's gain about half at cap 4? A better drafter wins by extending runs of correct drafts,
and under cap 4 any run that would have passed four drafts is cut to four for both drafters, so
part of ft5's advantage has nowhere to land. Depth does the same on mlx-dspark: the cap controller
narrows the draft to about one token at long context, and ft5's decode gain falls from 15% at chat
depth to about 7% at 35k and 4% at 65k (17.44 against 16.30 and 13.13 against 12.65 tok/s, in
`evidence/ft5-publication-battery/depth/`). PC keeps more of its gain over stock at cap 4 (+11.50%
at cap 7, +9.10% at cap 4), which would fit if more of its advantage comes early in the block; I
haven't tested that.

Throughput, on mlx-dspark only (it is never compared across runtimes):

| comparison | legs, ABBA, tok/s | pooled |
| --- | --- | --- |
| greedy, three reps a leg | 25.087, 28.925, 28.934, 25.097 | 25.0918 → 28.9295, +15.29% |
| temperature 1.0, top-p 0.95, top-k 20, five reps a leg (a screen) | 23.71, 26.43, 26.17, 24.43 | 24.07 → 26.30, +9.3% |

Under sampling the two arms produce different text, so that row isn't paired. Its per-rep rates
overlap (22.5–27.1 stock, 25.0–27.1 ft5), which is why I call it a screen. The gain is smaller
than greedy's because sampling lowers acceptance for both drafters.

## A derivative of the target

Does a model derived from the target need its own drafter? BoldingBuilds' abliterated Bonsai 2
flips ternary digits in 98 of the target's 851 tensors and leaves every block scale alone; a byte
comparison of its GGUF with PrismML's confirms both. I converted it into PrismML's MLX pack with
the converter PrismML ships in that pack, after checking on PrismML's own GGUF that the converter
is lossless for every tensor kind the derivative edits, and ran stock and ft5 on it with the
battery's settings. ft5 still gains +7.34% (+5.89% to +8.77%) on general chat and +10.09%
(+5.65% to +14.63%) on code, not significantly less than on the original target. For an edit
this light, the answer is no; a heavier one needs its own measurement. The records, and a small
script for comparing drafters across two targets, are in `evidence/derivative-abliterated/`.

## On a Mac, the kernel matters more than the drafter

Most Bonsai 2 downloads are the GGUF for PrismML's llama.cpp fork, so I tried the fork on Metal. It
has no DFlash 2 support of its own. I added upstream llama.cpp's, from the copy of the patch that
naklitechie's dflash-mlx-bonsai2 repository carries, and it applied cleanly. Upstream, that support
landed as Xuan-Son Nguyen's PR #27816, which re-lands Zihan Zhang's PR #27342, whose first commit is
Jian Chen's. The stock arm used z-lab's Q8_0 GGUF, the MTP arm decent-jawfish's grafted-MTP GGUF,
whose 851 trunk tensors all hash-match Prism's own. ABBA, three repetitions a leg.

| fork, Metal | decode tok/s | share of drafts accepted |
| --- | --- | --- |
| no drafter | 20.56 | – |
| MTP, 2 drafts | 11.70 | 0.585 |
| DFlash 2 stock, 7 drafts | 7.10 | 0.346 |

A single token cost 48.65 ms on the patched build (48.54 ms on the unmodified one). Counting a round
as the drafts it accepted plus the one token the target adds, so that rounds are tokens minus
accepted drafts, an MTP round cost 184.6 ms, 3.8 times a token (counting one round per two drafted
tokens instead gives 186.3 ms), and a DFlash 2 round about 476 ms, 9.8 times. At that price even
perfect acceptance of a block of eight reaches only 16.8 tok/s, against 20.56 with no drafter, and
MTP tops out near 16 tok/s. These figures are computed from the probe's records in
`evidence/fork-probe/`, a feasibility probe outside `BENCHMARK.md`. Whatever a drafter accepts, it
can't win on this path at that fork head. (The logs don't separate draft time from verify time.)
With the gate failed, I didn't convert ft5 to GGUF.

mlx-dspark without the kernel patch below was in the same place. With no drafter it served the
target at 20.9 tok/s, close to the fork's 20.60 (20.56 with the DFlash 2 patch in). With the stock
drafter, a verify round of eight rows cost 485 ms against 48 ms for one token, and the cap
controller pinned the cap at 1. The mlx-dspark figures in this section (20.9 tok/s, 485 ms, 48 ms,
and the 1.59–2.28x and 124 ms below) are my own kernel measurements and are not in the evidence
bundle.

A single-token step reads each weight once and is limited by memory bandwidth. A verify pass
multiplies the same weights by a handful of rows, one per drafted token, and ought to cost little
more. A generic quantized matmul pays the weight read and the unpacking again for every row, so
eight rows cost about eight steps. mlx-dspark's small-M kernel loads each quantized weight group
once, dequantizes it once, and reuses it across all the verify rows. It handled 4- and 8-bit
weights in groups of 64; Bonsai 2 is 2 bits in groups of 128, so the target never reached it.

My patch teaches the kernel that layout: two index expressions and one unpack routine, plus making
the packed projection a module type the kernel routes. Upstream's gates still decide per shape
whether the kernel runs: it has to match the reference matmul within 2% and win a timing race by
1.15x at eight rows, so a wrong unpack just leaves the fast path off. On the model's shapes it was
1.59–2.28x faster at eight rows. The controller then chose cap 7 by itself, and a round took 124 ms.

At chat depth that gives 20.9 tok/s with no drafter, 25.1 with the stock DFlash 2 drafter and 28.9
with ft5 (the last two are the publication battery's greedy arms). Without the kernel, the stock
drafter was slower than no drafter at all; ft5's gain is measured on top of what the kernel made
possible.

## Moving it between runtimes

ft5 has the same 81 tensor names, dtypes and shapes as `z-lab/Qwen3.8-27B-DFlash2`, and the same
`config.json`; only the weight values differ. Anything that loads the stock drafter should load
ft5. Beyond the two runtimes above, vLLM, SGLang and the rest are untested.

The snag is the selector's two codebooks, which circulate under three conventions:

| release | codebook tensors |
| --- | --- |
| z-lab, ft5 | `candidate_selector.{predecessor,successor}_codebook`, bare parameters |
| naklitechie r3 | the same names with `.weight`, as an `nn.Embedding` names them |
| nathansutton's MLX pack | `.weight`, `.scales` and `.biases`: a quantized embedding |

A strict loader refuses the other forms; mlx-dspark refused r3 on exactly this, while
dflash-mlx-bonsai2 maps the names itself. `scripts/rename-codebooks.py` converts between the two
bf16 conventions by rewriting two names in the safetensors header, hashing the data section
before and after to prove nothing else moved.

The 4-bit variant is the same weights quantized once instead of at every load, a download 2.83
times smaller: 1,361,734,159 bytes against 3,848,817,907. No 4-bit convention for DFlash 2 drafters
is in common use, so it proposes mlx-lm's: a `quantization` block (group size 64, 4 bits, affine)
with explicit `false` entries for the 13 matrices kept in bf16, both codebooks among them. On the
patched mlx-dspark it ran the same served loop as the bf16 file, at a throughput ratio of 1.0001.
It also loads in mlx-lm's `load_model` with a DFlash 2 model class and in dflash-mlx-bonsai2, all
153 tensors identical to the patched loader's (`evidence/prequantized-4bit/second-loaders/`). Don't
pass `--draft-quant` to dflash-mlx-bonsai2 with this file, or it will pack the 13 bf16 matrices
too.

## Limits

Everything was measured on one Apple M4 Pro with 48 GB. Only the pinned mlx-dspark 0.18.0 stack is
fully validated; the second runtime has acceptance numbers but no throughput, and nothing ran on
CUDA.

Both drafters scored 13/14 on the quality suite. The failed task asks for an ISO-8601 duration
parser, and at 8-bit KV the target spends the whole 16,384-token budget reasoning and never
answers, with or without a drafter. One deterministic greedy run per cell:

| `iso_seconds` | 8-bit KV | 16-bit KV |
| --- | --- | --- |
| no drafter | fails, 16,384 tokens | passes, 2,232 |
| stock drafter | fails, 16,391 | passes, 11,995 |
| ft5 | fails, 16,389 | passes, 13,417 |

The drafter isn't in the failure path, but the measurements used 8-bit KV, so this is a
limitation of that setup. One run per cell is not a rate, but across all archived runs the pattern
holds: at 8-bit KV it fails in 10 of 11 runs; at 4- or 16-bit KV it passes in all 5.

At 200 tokens, 34/40 general and 25/40 code responses were truncated in each arm, and 13/40 at
1024, so the acceptance figures describe prefixes, not complete answers or response latency. The
code suite chose ft5's training iteration. The general suite is locally authored, eight categories
of five prompts: a screening suite, not a sample of real traffic. I make no speed claim at 110k,
because the ft5 run there carried cache pressure the stock run didn't.

And I benchmarked my own drafter. I trained ft5, wrote the protocol and ran every comparison,
including those against r3 and PC. Against that I can offer a protocol committed before the comparisons,
a deciding suite ft5 wasn't selected on, an analyser that needs only the Python standard library,
and the raw per-prompt reports with full token arrays, which are in the repository under
`evidence/`, including every result where ft5 loses, PC's among them.

## What's published, and how to use it

- The drafter, bf16, Apache-2.0:
  [Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5)
- The 4-bit variant:
  [Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit](https://huggingface.co/Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit)
- The code, [github.com/alexschiltmans/bonsai2-drafter](https://github.com/alexschiltmans/bonsai2-drafter):
  the MLX trainer, the mlx-dspark patches, the rename tool and the protocol, `BENCHMARK.md`.
- The evaluation prompts and the target's greedy responses to them:
  [Schiltmans/bonsai2-drafter-eval](https://huggingface.co/datasets/Schiltmans/bonsai2-drafter-eval)
- The evidence: `evidence/` in the repository, the raw reports behind every figure here that is
  not marked otherwise, with the commands that recompute them.
- The evaluation kit: `bench/analysis/analyse_served_accept.py` takes two per-prompt reports and
  returns the paired gain, its interval and the identity classification. `served_accept.py`
  writes those reports for mlx-dspark and `bench/adapters/dflash_mlx_bonsai2.py` for
  dflash-mlx-bonsai2; any runtime that emits the same fields can be compared.

To serve it on a Mac (Apple Silicon, uv, about 13 GB of disk):

```sh
git clone https://github.com/alexschiltmans/bonsai2-drafter && cd bonsai2-drafter
scripts/serve-bonsai2.sh    # OpenAI-compatible API on http://127.0.0.1:8088/v1
```

It fetches the target and ft5 at pinned revisions and serves with the settings the measurements
used. For the 4-bit file, set
`BONSAI2_DRAFTER=Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit` and
`BONSAI2_DRAFTER_REVISION=c0db7e148bb2fe9782f85ea06c2b438b2a47c326`. Elsewhere, point your DFlash 2
runtime at ft5 as you would the stock drafter, and if you measure acceptance against stock, a pull
request with the numbers is the most useful thing you could send.

## Credits

Bonsai 2 is PrismML's, and the release carries their notice and requested attribution, "Created
using Bonsai by Prism ML". The backbone is Qwen's Qwen3.8-27B. DFlash 2 and the stock drafter come
from z-lab and Inco AI (DFlash 2: Keep Drafting Parallel, 2026), building on DFlash by Jian Chen,
Yesheng Liang and Zhijian Liu (2026). mlx-dspark, small-M verify kernel included, is ARahim3's.
dflash-mlx-bonsai2 and r3 are naklitechie's, PC is ProCreations', and the abliterated derivative is
BoldingBuilds'. Upstream llama.cpp's DFlash 2 support, used in the fork probe through naklitechie's
copy of the patch, is Zihan Zhang's PR #27342, whose first commit is Jian Chen's, landed by
Xuan-Son Nguyen as PR #27816. The grafted MTP GGUF is decent-jawfish's, and the training prompts
are sahil2801's CodeAlpaca-20k. This is an independent project, not affiliated with any of them.