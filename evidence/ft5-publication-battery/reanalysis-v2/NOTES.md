# Reanalysis v2 of the publication battery's acceptance arms

Contract `budgeted-prefix-identity/v2`. No model was loaded, no GPU was used and no arm was
rerun: every number here comes from the token arrays already saved in `../acceptance/`.

The version 1 summaries are kept unchanged at `../acceptance/acceptance-summary.json` and
`../acceptance/long-acceptance-summary.json`. This directory does not replace them; it explains
them.

## What changed, and what did not

Version 1 compared the two arms' whole `response_ids` arrays and called any inequality an
output mismatch. It reported 17/40 non-code, 14/40 code and 6/40 long-budget mismatches, and
returned `investigate` on all four pairs.

Version 2 changes the identity criterion only: it compares the first `max_new` tokens, the
tokens the request asked for, and classifies each unequal pair instead of counting it. The
headline acceptance metric is untouched (whole committed tokens over whole rounds), so every
rate and interval in the v1 summaries reproduces here to the last digit.

The mechanism behind the v1 mismatches is in the pinned mlx-dspark 0.18.0 generator. Its round
loop is `while len(out_ids) < max_new_tokens and pending not in eos_ids and not st.stopped`,
and once inside, the committed block is appended in full, breaking only for EOS. A round
entered at 199 emitted tokens therefore ends at up to 207: the budget test is per round, not
per token. `_finish_reason` checks EOS before the length test, so such a request can even be
labelled `stop` above its budget. Observed maxima across the arms are 207 against a 200-token
budget and 1031 against 1024, exactly one block (cap 7 + the target's own token) minus one.

## Results

| pair | v1 mismatches | v2 classes | gate | gain (whole rounds) |
|---|---|---|---|---|
| general-forward | 17 | 23 identical, 17 boundary overshoot | pass | +9.4892% |
| general-reverse | 17 | 23 identical, 17 boundary overshoot | pass | +9.4892% |
| code | 14 | 26 identical, 14 boundary overshoot | pass | +10.3860% |
| general-long | 6 | 34 identical, 6 boundary overshoot | pass | +7.9929% |

Zero prefix divergences and zero early-stop mismatches in all four pairs. Maximum overshoot 7
tokens, strictly under one block, in every pair. `identity-controls.json` records the two
same-drafter pairs: 40/40 token-identical each, which is what makes the greedy-path assumption
behind this criterion checkable rather than assumed.

## Numerator and denominator

These reports carry per-request totals, not per-round lengths, so a trimmed-numerator figure
cannot be reconstructed exactly. Rather than choose silently, each summary reports three
variants; the gain stays positive and close under all of them, which is why the contract change
needed no rerun:

| pair | whole rounds (headline) | trimmed numerator | crossing round dropped (envelope) |
|---|---|---|---|
| general-forward | +9.4892% | +9.3876% | +8.1053% to +10.8700% |
| general-reverse | +9.4892% | +9.3876% | +8.1053% to +10.8700% |
| code | +10.3860% | +9.9940% | +8.8051% to +11.7915% |
| general-long | +7.9929% | +7.9754% | +7.8059% to +8.1566% |

The envelope crosses each arm's independent bounds, so it is wider than the difference the
choice actually makes: holding the same assumption on both arms moves the general-forward gain
between +9.4267% and +9.5312%. The reports in `../../prequantized-4bit/served/` and
`../../second-runtime/reports/` carry `round_lengths`, so for those groups this variant is
resolved exactly.

## Reproducing this directory

From the repository root, with nothing loaded and no server running:

```sh
A=evidence/ft5-publication-battery/acceptance
V=evidence/ft5-publication-battery/reanalysis-v2
python3 bench/analysis/analyse_served_accept.py "$A/general-1-stock.json"      "$A/general-2-ft5.json"      --out "$V/general-forward.json"
python3 bench/analysis/analyse_served_accept.py "$A/general-4-stock.json"      "$A/general-3-ft5.json"      --out "$V/general-reverse.json"
python3 bench/analysis/analyse_served_accept.py "$A/code-1-stock.json"         "$A/code-2-ft5.json"         --out "$V/code.json"
python3 bench/analysis/analyse_served_accept.py "$A/general-long-1-stock.json" "$A/general-long-2-ft5.json" --out "$V/general-long.json"
```

Writing with `--out` overwrites the committed summaries; the numbers are identical, the JSON
layout may differ in whitespace, so compare values rather than bytes (or drop `--out`).

`identity-controls.json` was written by a short script over the analyser's `classify` on the
two same-drafter pairs; the acceptance gate does not apply to them, since no gain is expected.

## What this does and does not license

It licenses one sentence: on these four fixed prompt suites, the two drafters produced the same
greedy answer inside the requested budget, and ft5 reached it in fewer rounds. It does not
license a completeness claim (34/40 non-code and 25/40 code responses hit the 200-token budget
in each arm, 13/40 at 1024), and it does not turn the code suite into an untouched test set,
since that suite informed the choice of iteration. The shared `iso_seconds` quality failure is
a separate matter, covered by `../quality/` and `../quality-controls/`.
