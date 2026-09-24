# Served-acceptance reports: format specification

A served-acceptance report records one arm of a measurement: one drafter, or none, on one
runtime, over one fixed prompt corpus at one token budget, decoded greedily. Two reports make a
comparison. `bench/analysis/analyse_served_accept.py` reads a pair and answers one of two
questions: does the second drafter draft better than the first, or do the two arms run the same
loop. `BENCHMARK.md` says which comparisons are run and how they are reported. This file says
what a report must contain so that any runtime can take part, including one this repository has
never run.

Two writers exist:

- `bench/drafter/served_accept.py`, for mlx-dspark;
- `bench/adapters/dflash_mlx_bonsai2.py`, for dflash-mlx-bonsai2.

`bench/report.schema.json` is the JSON Schema (draft 2020-12) for the per-field rules. The rules
that span fields are stated below and checked by `bench/analysis/validate_report.py`, which
imports the analyser's own record check rather than restating it. Where this file and the
analyser disagree, the analyser is what runs, and the disagreement is a bug in this file.

## A report

One JSON object.

| Field      | Type           | Meaning |
|------------|----------------|---------|
| `drafter`  | string or null | Label of the drafter. null only for an arm with no drafter. Not read by the analyser. |
| `settings` | object         | Everything two paired arms must share. See [Settings](#settings). |
| `complete` | boolean        | false until every prompt's record has been written. |
| `requests` | array          | One record per prompt, in corpus order. See [Records](#records). |

Writers save the report after every prompt, with `complete` false, so a crashed run leaves its
partial record behind. The analyser refuses an arm that is not complete, and it refuses one with
no records.

Any other top-level field is allowed and ignored by the analyser. The dflash-mlx-bonsai2 adapter
adds three, all optional and specific to it:

- `mode`: `"speculative"` for a drafter arm, `"target_only"` for the target's own
  autoregressive path;
- `arm`: per-arm details that legitimately differ between paired arms, such as paths, file
  hashes, the codebook naming of the drafter's tensors and load time;
- `summary`: pooled figures written with `complete: true` (`tokens_per_round`,
  `runtime_tokens_per_cycle`, `decode_tokens_per_second`, `wall_seconds`). The analyser
  recomputes everything from `requests` and does not read it.

## Settings

The analyser compares the two arms' `settings` objects for exact equality and refuses the pair on
any difference, including a key present in one and absent in the other. So `settings` holds only
what both arms must share: the target, the corpus, the budget, the block, the decoding and cache
configuration, the runtime and its version. Anything that differs by design between arms, such
as the drafter's path or hash, goes in `drafter`, in `arm`, or in another top-level field.

Three keys are required:

| Key           | Type        | Meaning |
|---------------|-------------|---------|
| `max_new`     | integer ≥ 1 | The token budget, counting the prefill's own token. |
| `cap`         | integer ≥ 1 | The most drafts one block can carry. One **block** is `cap + 1` tokens: the drafts plus the target's own token from the same verify pass. |
| `temperature` | 0           | Both contracts assume greedy decoding. |

The analyser reads `max_new` and `cap` and nothing else from `settings`, apart from the equality
test. It does not read `temperature`. The schema requires it to be 0 anyway, because under
sampling the arms' outputs differ for reasons that have nothing to do with the drafter, and every
identity class below loses its meaning.

The writers record more. mlx-dspark: `target`, `split`, `bits` (the drafter's served precision,
0 for bf16), `kv_bits`. dflash-mlx-bonsai2: `runtime`, `runtime_version`, `runtime_commit`,
`mlx`, `mlx_lm`, `target`, `corpus_sha256`, `split`, `block_tokens_requested`, `block_tokens`,
`verify_mode`, `prism_verify`, `draft_quant`, `kv_bits` (null for an unquantized cache),
`copyspec`, `prefill_step_size`, `draft_sink_size`, `draft_window_size`, `stop_token_ids`.
A new runtime should record whatever would make two of its arms not comparable if it differed.

A consequence of exact equality: an arm without a drafter must record the same `cap` as the
drafter arms it is paired with, and the same values for every drafter-side setting, even though
none of them applies to it. The dflash-mlx-bonsai2 adapter's `--target-only` takes `--cap` for
this reason.

## Records

| Field            | Type                     | Meaning |
|------------------|--------------------------|---------|
| `prompt_sha256`  | string, 64 lowercase hex | The prompt's identity. See below. Unique within a report. |
| `category`       | string                   | The prompt's stratum; `"unknown"` when the corpus row has none. |
| `thinking`       | boolean                  | Whether the templated prompt enables thinking. Also a stratum. |
| `tokens`         | integer ≥ 2              | Emitted tokens, **counting the prefill's own token**. Equal to `len(response_ids)`. |
| `rounds`         | integer ≥ 1              | Speculative rounds. The prefill's token belongs to no round. |
| `round_lengths`  | array of integers ≥ 1, optional | Tokens each round committed. Length equals `rounds`. |
| `finish`         | `"length"` or `"stop"`   | Stopped on the budget, or on a stop token. |
| `response_ids`   | array of integers        | The emitted token ids in order, from the prefill's token to the first stop token inclusive. |
| `decode_seconds` | number > 0               | Wall time after prefill. No contract compares it. |

`prompt_sha256` is the SHA-256 of the UTF-8 bytes of Python's `json.dumps(prompt_ids)` with its
default separators, where `prompt_ids` are the templated prompt's token ids, written as lowercase
hex. For the prompt `[151644, 872, 198]` the hashed text is exactly `[151644, 872, 198]`: a comma
and a space between ids, no space inside the brackets. A writer in another language must
reproduce those bytes, or its hashes will not pair with anyone else's.

All counts (`tokens`, `rounds`, each round length, `max_new`, `cap`) must be JSON integer
literals. The analyser refuses `8.0` and `true` where it expects a count, since arithmetic would
otherwise accept both. The schema's `integer` has this narrower meaning, which is stricter than
JSON Schema's.

`round_lengths` is optional because older mlx-dspark reports predate it. Both writers now record
it, and a new writer should: the equivalence contract refuses records without it, and without it
the within-budget variant of the metric is only an envelope.

Any other field is allowed. The dflash-mlx-bonsai2 adapter adds `prefill_seconds`,
`prompt_tokens` and `runtime`, the runtime's own per-pass record left unmapped (`passes`,
`committed_tokens`, `acceptance`, `block_lens`, `excluded_passes`, `post_stop_tokens`,
`tokens_per_cycle`, `peak_memory_gb`), so the runtime's native bookkeeping can be recomputed
from the report. The analyser reads none of them.

## Token accounting

The reference loop is mlx-dspark 0.18.0's `dflash_generate`:

    out_ids = [pending]              # the prefill's own token, before any round
    accept_lengths = []
    while len(out_ids) < max_new_tokens and pending not in eos_ids and not stopped:
        ...                          # draft a block, verify it, commit the accepted part
        accept_lengths.append(len(committed))
        for tok in committed:
            out_ids.append(tok)
            if tok in eos_ids:
                break

`tokens` is `len(out_ids)`, `rounds` is `len(accept_lengths)` and `round_lengths` is
`accept_lengths`. Two things follow. The prefill's token is emitted before any round, so a run
that stops on the budget emits `1 + sum(round_lengths)` tokens, not `sum(round_lengths)`. And the
budget is tested once, at the top of the loop, after which the round commits its whole block: a
round entered at 199 tokens with a budget of 200 and a cap of 7 can finish at 207. That is **boundary
overshoot**. It comes from the loop's budget arithmetic, not from the model.

With `block = cap + 1` and `budget = max_new`, every record must satisfy:

1. `tokens == len(response_ids)`.
2. `finish == "length"` implies `tokens >= budget`. A run below the budget stopped on a stop
   token.
3. `tokens < budget + block`. The overshoot is at most `cap` tokens, because the last round
   began below the budget.
4. `1 + rounds <= tokens <= 1 + rounds * block`. Every round emits at least one token and
   commits at most one block.

When `round_lengths` is present, also:

5. `len(round_lengths) == rounds`, and each length is between 1 and `block`.
6. The last round began below the budget: `1 + sum(round_lengths[:-1]) < budget`. Only the last
   round can have been cut short, so every earlier round emitted its whole length.
7. `finish == "length"` implies `tokens == 1 + sum(round_lengths)`: no stop token, so every
   committed token was emitted.
8. `finish == "stop"` implies `1 <= tokens - (1 + sum(round_lengths[:-1])) <= round_lengths[-1]`.
   A stop token may truncate the last round's emission, and nothing else. The recorded length is
   still the committed length, so on a stop `1 + sum(round_lengths)` may exceed `tokens`.

These rules exist because totals alone certify histories no loop can produce: eight tokens over
rounds of `[8, 8]` at a budget of eight, or one token after three full blocks. Two matching
impossible histories would pass the equivalence contract. Rules 6 to 8 check chronology, not
just totals.

JSON Schema cannot express rules 1 to 8, the uniqueness of `prompt_sha256` within a report, or
that a complete arm has at least one record; nor the pairing rules below. The validator checks
all of them except pairing, which needs two reports.

A runtime whose loop never overshoots also fits. dflash-mlx-bonsai2 shrinks its last block to
the budget that remains, so its `length` records always emit exactly `max_new` tokens.

## Pairing

The analyser pairs record *i* of one report with record *i* of the other. Under either contract
it refuses the pair unless:

- both reports have `complete: true`;
- their `settings` objects are equal;
- both list the same, non-zero number of records;
- `prompt_sha256` values are unique within the first report;
- records at the same position agree on `prompt_sha256`, `thinking` and `category`;
- every record in both reports passes rules 1 to 8.

A refusal is a message beginning `refusing comparison:` and a non-zero exit, not a verdict.

## The contracts

Every result states the contract it was produced under.

### `budgeted-prefix-identity/v2` (the default)

Asks whether the second arm drafts better than the first. Greedy decoding makes the token path
the target's own, so the two arms must emit the same answer and differ only in how many rounds
it took. The analyser compares the first `max_new` tokens and gives each pair one class:

| Class                 | Meaning | Blocks the gate |
|-----------------------|---------|-----------------|
| `identical`           | The `response_ids` agree everywhere. | no |
| `boundary_overshoot`  | Both arms reached the budget and every difference is above it. | no |
| `prefix_divergence`   | The arms disagree inside the budget. | yes |
| `early_stop_mismatch` | The compared prefix agrees, but at least one arm stopped before the budget and the lengths differ. | yes |

The metric is tokens per round, whole-round: the sum of `tokens` over the sum of `rounds`, every
committed token over every round, including the round that crossed the budget. The gain is the
second arm's value over the first's, less one. Its 95% interval comes from a paired bootstrap
over prompts, stratified by (`category`, `thinking`), with a fixed seed.

The gate is `pass` when the interval's lower bound is above zero and no pair is in a blocking
class, and `investigate` otherwise. Two identical arms therefore read `investigate`: no gain is
not an improvement.

Two sensitivity variants are reported beside the headline and never replace it: a numerator
trimmed to the budget, and the metric with each budget-crossing round dropped from both sides.
The second is exact when `round_lengths` are recorded and a bounded interval otherwise.

### `artifact-equivalence/v1` (`--equivalence`)

Asks whether two arms are the same served loop, which is what a repackaging of one drafter, such
as a prequantized export, has to prove. Each pair must be equal in `response_ids`, `finish`,
`tokens`, `rounds` and `round_lengths`. `decode_seconds` is exempt, and only it. `round_lengths`
is required on every record, because two arms can emit the same tokens in the same number of
rounds and still have committed them in different blocks.

The gate is `pass` only if every pair is equal in all five fields. A `boundary_overshoot` blocks
here: between two packagings of one drafter the budget arithmetic has nothing to explain. The
identity classes are still reported, and the first ten differing prompts are named with the
fields that differ. A pass says nothing about speed.

## Writing an adapter for a new runtime

An adapter runs the runtime's own speculative loop, greedy, in process if it can, over the first
`n` rows of a corpus split in file order, and writes one report per arm. The corpus is JSON
lines with `prompt_ids` (already templated), `split`, `thinking` (a boolean) and optionally
`category` (a string). Both writers refuse a row that breaks either before any model loads,
since the records would carry the bad value into the report.

Record, for each prompt:

- the emitted tokens, starting with the token the prefill produced, cut after the first stop
  token;
- for each round, the tokens it committed: the accepted drafts plus the target's own token from
  the same verify pass;
- whether the run stopped on the budget or on a stop token;
- decode wall time, excluding prefill.

Then check the mapped record against rules 1 to 8 before writing it, or run the validator on
the finished report.

Pitfalls:

- **The seed token.** The prefill's token is emitted and counted in `tokens` but belongs to no
  round. An adapter that forgets it reports `tokens == sum(round_lengths)` and fails rule 7.
- **A loop shifted by one.** A runtime may book the target's token to the next pass instead of
  the current one: pass *i* verifies `[staged, d_1 .. d_k]` and commits the staged token with
  its accepted drafts. The same tokens come out, but a pass is not a round until it is
  remapped. The docstring of `bench/adapters/dflash_mlx_bonsai2.py` documents such a mapping,
  and its `map_speculative` checks the mapping against the loop's arithmetic instead of assuming
  it. `--self-test` exercises the cases.
- **Passes that are not rounds.** A final pass that only commits a token the previous pass
  already produced (a one-token block at the budget, or a pass that starts on a stop token)
  emits nothing new. Counting it adds a round with no tokens and fails rule 4 or rule 6. The
  dflash-mlx adapter lists such passes in `runtime.excluded_passes`.
- **Tokens after a stop.** A runtime may commit tokens past the stop token inside the same
  block. Drop them from `response_ids`, but keep the final round's committed length (rule 8).
- **A shrunken last block.** A runtime that trims its last block to the remaining budget emits
  exactly `max_new` tokens on a `length` finish. The final round's length is then what it
  emitted, not what the untrimmed block would have held.
- **The effective cap.** Record the cap the runtime actually used, not the one requested. A
  runtime may clamp the block for a given drafter, and rules 3 to 5 are checked against
  `cap + 1`.
- **Hashes.** `prompt_sha256` must be computed from the exact bytes described under
  [Records](#records).
- **Settings hygiene.** Put nothing in `settings` that differs between arms that should pair, or
  the analyser will refuse them. Put everything there whose difference would make them not
  comparable.
- **Counts and durations.** Write counts as integers, not floats. Write `decode_seconds` as a
  positive number; a zero is refused. If the runtime's timings leave zero or less, stop
  rather than clamp, as both writers do: a clamped value is a time nobody measured.

## Validating and analysing

Both tools need only the Python standard library.

    python3 bench/analysis/validate_report.py arm.json [more.json ...]

prints `ok` or `FAIL` for each report, with one line per problem, and exits non-zero if any
report fails. It checks one report at a time.

    python3 bench/analysis/analyse_served_accept.py first.json second.json [--out result.json]
    python3 bench/analysis/analyse_served_accept.py first.json second.json --equivalence

The first form applies `budgeted-prefix-identity/v2` with the first report as the baseline, the
second `artifact-equivalence/v1`. Both print the result as JSON and exit 0 only when the gate is
`pass`.
