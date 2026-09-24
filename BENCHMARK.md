# Bonsai 2 drafter benchmark: protocol

Version 1, registered before any benchmark arm ran. Results state the protocol
version they ran under. An amendment gets a new version, a date and a reason; it never
rewrites this one.

**Conflict of interest.** The author of this protocol trained one of the drafters it compares
(ft5). That is why the rules below are fixed in advance, why the deciding suite is one ft5 was
not selected on, and why every raw output is published.

## Question

Which public speculative decoders speed up Ternary-Bonsai-2-27B, by how much, and on which
runtimes. Where a runtime supports both, the benchmark also asks how DFlash 2 compares with
multi-token prediction (MTP).

## Arms

- **No drafter**, on every runtime: the baseline.
- **DFlash 2 drafters**, each at a pinned revision identified by sha256:
  - `z-lab/Qwen3.8-27B-DFlash2`, the stock drafter trained on the bf16 base;
  - `Bonsai-2-27B-DFlash2-ft5`, fine-tuned by this author against the ternary target;
  - `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`, fine-tuned against the ternary target;
  - `ProCreations/Ternary-Bonsai-2-27B-DFlash2`, on any runtime that can load its published files.
- **MTP**, on the PrismML llama.cpp fork: a grafted-MTP GGUF. Its trunk tensors are checked
  against Prism's own GGUF by hash, and any difference is disclosed beside the result.

A drafter that a runtime cannot load is reported as such, not skipped silently.

## Runtimes and targets

- mlx-dspark 0.18.0 with this repository's patches, from its pinned lock, serving
  `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` with an 8-bit KV cache and a 4-bit drafter.
- dflash-mlx-bonsai2 at a pinned commit, with the same MLX pack.
- The PrismML llama.cpp fork at a named release, serving `Ternary-Bonsai-2-27B-PQ2_0.gguf`,
  with this repository's DFlash 2 patch where a DFlash 2 arm needs it.

The target revision, runtime revision, package set and every launch flag are recorded with each
result.

## Inputs

The frozen prompt corpora, 40 held-out prompts each, identified by sha256:

    3dcc1327c0d254e2191322503f6cc4fe0cc464389ef0799093778ca55996c50b  general.jsonl
    4252b5bc10babf0605a8afb454bc1296befceb87e1e8042d6a65e0478841654d  code.jsonl

`general.jsonl` comes from `bench/drafter/general_chat.json`: eight categories, five prompts
each, balanced between thinking on and off. `code.jsonl` comes from CodeAlpaca-20k. **ft5's
training iteration was selected on the code suite**, so the code results are reported and
flagged, and they never decide.

## Measurements

1. **Acceptance.** The runtime's served loop, greedy, on each corpus. Budgets: 200 tokens on
   both suites, and 1024 on the general suite. The metric is tokens per round, whole-round:
   every committed token over every round, including the round that crosses the budget.
2. **Throughput.** `bench/throughput/bench5.py`, three repetitions of five prompts at 400
   tokens, on a fresh server per leg. Every compared pair runs in ABBA order. The pooled decode
   rate is total completion tokens over total decode seconds per arm, never an average of rates.
3. **Sampled throughput.** As in 2, at the target's published sampling (temperature 1.0, top-p
   0.95, top-k 20), five repetitions per leg, ABBA. Outputs differ between arms under sampling,
   so this is reported with its per-leg spread and never as a paired result.

## Analysis

`bench/analysis/analyse_served_accept.py` at the tagged revision of this file, under the
contract `budgeted-prefix-identity/v2`. Drafters change speed, not output. So on one runtime,
every greedy arm must produce the same tokens within the budget. A boundary overshoot passes; a
prefix divergence or an early-stop mismatch is reported as a finding against that runtime.

- **Deciding rule.** On a given runtime, drafter A beats drafter B only if the lower bound of
  the 95% paired, stratified prompt-bootstrap interval of A's gain over B, on the **general
  suite at 200 tokens**, is above zero. The 1024-token general result and the code suite are
  reported beside it and do not decide.
- **Throughput** is reported as ratios with every leg's value. No significance is claimed for it.
- **Across runtimes**, acceptance may be compared, since the prompts and metric are the same.
  Throughput is never compared across runtimes.
- Subgroups (category, thinking on or off) are descriptive, not independent tests.

## Machine state

One Apple M4 Pro with 48 GB, on AC power. One workload at a time. Normal memory pressure, under
4 GB of swap, and no competing inference process. These checks are recorded before and after
every stage. Results from one machine are labelled as such.

## Stopping and publication

No arm is rerun to change its result. A failed arm is reported as failed, with its log.
Everything is published: the raw per-prompt reports with full token arrays, the throughput
records, the analyses and the machine-state checks. That includes every result where ft5 loses.

## Screens run before publication

Before publication, a screen ran on mlx-dspark only. It covered naklitechie's drafter's
acceptance on both suites, paired against the existing ft5 and stock reports; a greedy HTTP
comparison, ft5 against naklitechie, in ABBA order; and a sampled HTTP comparison, stock against
ft5, in ABBA order. Its purpose was a publication decision, not a result. **This protocol was
committed before that screen started.** The screen's numbers are reported separately and labelled as a screen. The benchmark
re-runs every arm under this protocol and does not reuse them.

## Amendment: version 2

Registered at the git tag `benchmark-v2`; that commit's time is its date. Version 1 above is
unchanged and still governs every arm it names.

**Reason.** After version 1 was registered, a Mac runtime appeared that serves this target with
MTP instead of a drafter: MTPLX, with its own kernels for the pack. Version 1 has an MTP arm
only on the PrismML fork, so it cannot say which way of serving this target is fastest on a Mac.

**Added arm.** MTP on MTPLX 2.12.0 (PyPI), serving
`Youssofal/Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed` at `03bd60bb82755f2446f5083426dca9708eb8e0fe`, whose
`model.safetensors` is byte-identical to `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`'s (sha256
`130de5925082c168b7866b2e91b52e44abbafc99017e3ca352b77b5b55a269ed`), at the MTP depth and settings
the pack ships. No tuning run changes them.

**Added comparison.** That arm against mlx-dspark as in version 1, with whichever of ft5 and
`ProCreations/Ternary-Bonsai-2-27B-DFlash2` has the higher pooled greedy decode rate in the
version-1 throughput arms run just before it. Measurements 2 and 3 of version 1, in ABBA order,
with one change: MTPLX's responses carry no decode timer, so this comparison pools end to end
(completion tokens over client wall-clock seconds) for both arms, and reports mlx-dspark's
decode pool beside it. Each stack uses its own defaults for the target's chat template; one
untimed request per stack records whether its answer opens with reasoning.

This is a comparison of two complete serving stacks on one machine, the one exception to
"throughput is never compared across runtimes". It is reported as such, not as a drafter
result, and it has no acceptance comparison, since MTP and DFlash 2 count drafts differently.
