#!/usr/bin/env python3
"""Paired prompt bootstrap for served_accept --report; no model or GPU needed.

    python3 bench/analysis/analyse_served_accept.py <stock.json> <tuned.json> [--out summary.json]
    python3 bench/analysis/analyse_served_accept.py <a.json> <b.json> --equivalence

Two questions, two criteria, one set of record checks. The default asks whether the second
arm drafts *better* than the first, which is what a fine-tune has to prove. `--equivalence`
asks whether two arms are the *same* loop, which is what a repackaging has to prove; see
EQUIVALENCE_CONTRACT below. Do not read one as the other: a repackaging that passes the
improvement gate has changed something it should not have, and an improvement that passes
the equivalence gate is not an improvement.

The measurement contract, version CONTRACT below, in two parts.

**Identity.** Greedy decoding at temperature 0 makes the token path the target's own, so the
two arms must emit the same answer and differ only in how many rounds it took. Version 1
compared the whole `response_ids` arrays and, on the publication battery, returned `investigate` on 17 of
40 non-code prompts, 14 of 40 code prompts and 6 of 40 at the 1024-token budget. Direct
comparison of those saved arrays found ZERO differing tokens inside the requested budget.
Every unequal pair was longer than the budget: the pinned mlx-dspark 0.18.0
`dflash_generate` tests its budget at the top of the round loop (`while len(out_ids) <
max_new_tokens`) and then appends the whole committed block, breaking only for EOS, so a
round entered at 199 emitted tokens can finish at 207. Version 1 was reading that boundary
overshoot as numerical divergence.

So this compares the first `max_new` tokens, keeps the raw arrays untouched, and classifies
every unequal pair rather than counting it:

    identical           the arrays agree everywhere
    boundary_overshoot  both arms reached the budget, every difference is above it, and
                        each overshoot is under one block (cap + 1) -- the artifact above
    prefix_divergence   the arms disagree INSIDE the budget. A real finding; blocks the gate
    early_stop_mismatch the budgeted prefix agrees but one arm stopped early and the other
                        did not. Also a real finding; also blocks the gate

A drafter is lossless or it is not, and only the last two classes can tell you. Boundary
overshoot cannot: it is the harness's budget arithmetic, not the model's arithmetic.
`output_mismatches` is still reported, with version 1's meaning, so the two versions can be
read against each other. Do not relabel version 1's `investigate` verdicts as passes; run
this and cite it by contract version.

**Numerator and denominator.** The headline metric is unchanged from version 1 and stays
whole-round: every token committed in a round counts, and every round counts, including the
round that crossed the budget. That is the quantity the serving loop actually pays for, and
it is the only one the publication battery's reports can reproduce exactly -- they carry per-request
totals, not per-round lengths. Trimming the numerator to the budget while keeping whole
rounds in the denominator would be a different metric, so it is reported as a sensitivity
variant and never as the headline. `complete_rounds_within_budget` drops the crossing round
from both sides; it is exact when `round_lengths` are recorded (served_accept writes them
in every report) and otherwise a bounded envelope, since a round that crossed the budget
committed between `overshoot + 1` and `cap + 1` tokens. On the publication battery's arms the three
variants moved the gain by under 0.4 points, which is why no rerun was needed to settle this.
"""
import argparse
import json
import random

CONTRACT = "budgeted-prefix-identity/v2"
BLOCKING = ("prefix_divergence", "early_stop_mismatch")

#: The second criterion. Two arms that differ only in how the same weights were *packaged*
#: must run the same loop: the same tokens, the same stopping point, and the same number of
#: rounds carrying the same committed lengths. Durations are exempt, and only durations --
#: they are the one quantity a repackaging is allowed to move, and the one this file cannot
#: hold to anything anyway.
#:
#: Note what this is stricter about than CONTRACT. `boundary_overshoot` passes the identity
#: contract because the harness's budget arithmetic explains it; it does NOT pass here,
#: because between two packagings of one drafter there is nothing for it to explain. The
#: classes are still reported, so a failure says which kind it was.
EQUIVALENCE_CONTRACT = "artifact-equivalence/v1"
#: Compared field by field. `decode_seconds` is deliberately absent.
EQUIVALENCE_FIELDS = ("response_ids", "finish", "tokens", "rounds", "round_lengths")


def _settings(report):
    settings = report["settings"]
    budget, cap = settings["max_new"], settings["cap"]
    if budget <= 0 or cap <= 0:
        raise ValueError("max_new and cap must be positive")
    return budget, cap + 1     # one block is the cap's drafts plus the target's own token


def _count(value, name):
    """An integer count, refusing the values that arithmetic silently accepts.

    `True` is an `int` in Python and `4.0 == 4`, so a report carrying either would pass every
    comparison below while describing a run that cannot exist. A count is a count.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        # ValueError, not TypeError: this is a malformed *record*, not a caller passing the
        # wrong type, and every refusal in this file reaches `main` as "refusing comparison".
        raise ValueError(f"{name} is not an integer count: {value!r}")  # noqa: TRY004
    return value


def _check_row(row, budget, block):
    """One record, against the loop that produced it. Raises rather than returning a verdict.

    The arithmetic follows mlx-dspark 0.18.0 `generate.dflash_generate`, which is worth
    stating because one term of it is easy to miss:

        out_ids: list[int] = [pending]      # the prefill's own token, before any round
        accept_lengths: list[int] = []
        while len(out_ids) < max_new_tokens and pending not in eos_ids and not st.stopped:
            ...
            accept_lengths.append(len(committed))
            for tok in committed:
                out_ids.append(tok)
                if tok in eos_ids:
                    break

    `num_tokens` is `len(out_ids)` and counts that seed token; `num_rounds` is
    `len(accept_lengths)` and does not. So a clean run emits **1 + sum(round_lengths)**
    tokens, not `sum(round_lengths)`, and a single round of a full block legitimately
    reports one more token than `rounds * block`. The budget is tested at the TOP of the
    loop, so a round exists only if the tokens emitted before it were below it. EOS breaks
    the append but not the recorded length, so only the final round may emit fewer tokens
    than it committed.

    A code review found that the pre-existing checks -- count, bounds, and a sum that
    may not fall below the emitted tokens -- certify histories no loop can produce: eight
    tokens over rounds of `[8, 8]` at a budget of eight, or one token after three full
    blocks. Two matching impossible histories then passed `artifact-equivalence/v1`. The
    rules below close that by checking chronology, not just totals.
    """
    rounds, tokens = _count(row["rounds"], "rounds"), _count(row["tokens"], "tokens")
    if rounds <= 0 or tokens <= 0 or row["decode_seconds"] <= 0:
        raise ValueError("invalid tokens, rounds or decode duration")
    if len(row["response_ids"]) != tokens:
        raise ValueError("token count disagrees with output")
    if row["finish"] not in ("length", "stop"):
        raise ValueError(f"unknown finish reason {row['finish']!r}")
    if row["finish"] == "length" and tokens < budget:
        raise ValueError("finish=length below the requested budget")
    # Above one block of overshoot the loop's own arithmetic no longer explains the length,
    # so the record is not the run this contract describes.
    if tokens >= budget + block:
        raise ValueError(f"output overshoots the budget by a whole block ({tokens} > {budget})")
    # The seed token plus at most one block a round, and at least one token a round.
    if tokens > 1 + rounds * block:
        raise ValueError("more tokens than the rounds could have committed")
    if tokens < 1 + rounds:
        raise ValueError("fewer tokens than the rounds must each have emitted")
    lengths = row.get("round_lengths")
    if lengths is None:
        return                      # an older record; the totals above are all of it
    if len(lengths) != rounds:
        raise ValueError("round_lengths disagrees with the round count")
    for length in lengths:
        if not 1 <= _count(length, "a round length") <= block:
            raise ValueError("a round committed more than one block")
    # Where the loop stood when it entered its last round. Every earlier round appended its
    # whole committed block, because only an EOS round breaks the append and only the last
    # round can be one.
    entered_last = 1 + sum(lengths[:-1])
    if entered_last >= budget:
        raise ValueError(
            f"a round began at {entered_last} tokens, at or past the {budget}-token budget "
            f"the loop tests before starting one")
    if row["finish"] == "length":
        # No EOS and no stop string, so every committed token was emitted.
        if tokens != 1 + sum(lengths):
            raise ValueError("finish=length emitted a different count than the rounds committed")
    elif not 1 <= tokens - entered_last <= lengths[-1]:
        # EOS may truncate the final block and nothing else; it cannot unsay earlier rounds.
        raise ValueError(
            f"the final round would have had to emit {tokens - entered_last} of its "
            f"{lengths[-1]} committed tokens")


def classify(left, right, budget, block):
    """One pair, under the identity half of the contract."""
    before, after = left["response_ids"], right["response_ids"]
    if before == after:
        return "identical"
    compared = min(budget, len(before), len(after))
    if before[:compared] != after[:compared]:
        return "prefix_divergence"
    if len(before) >= budget and len(after) >= budget:
        return "boundary_overshoot"     # every difference sits above the requested budget
    return "early_stop_mismatch"


def acceptance(rows):
    return sum(r["tokens"] for r in rows) / sum(r["rounds"] for r in rows)


def _within_budget(rows, budget, block):
    """((lo, hi), exact) pooled acceptance with the budget-crossing round dropped, or None.

    None when dropping leaves no rounds at all, which a budget under one block can do.
    `exact` is False as soon as one crossing round had to be bounded rather than read: a
    row without round_lengths, or one whose block ended in EOS, so the lengths sum past the
    tokens and the crossing round's true contribution cannot be recovered.

    The `1 +` is the seed token `dflash_generate` puts in `out_ids` before the first round
    (see :func:`_check_row`); without it this test never fires on a clean run and every row
    falls into the bounded branch.
    """
    low = high = denominator = 0
    exact = True
    for row in rows:
        overshoot = row["tokens"] - budget
        lengths = row.get("round_lengths")
        if overshoot <= 0:
            low += row["tokens"]; high += row["tokens"]; denominator += row["rounds"]
        elif lengths is not None and 1 + sum(lengths) == row["tokens"]:
            low += row["tokens"] - lengths[-1]; high += row["tokens"] - lengths[-1]
            denominator += row["rounds"] - 1
        else:
            # The crossing round entered at most budget - 1 tokens in and committed at most
            # one block, so its length is in [overshoot + 1, block].
            low += row["tokens"] - block; high += row["tokens"] - overshoot - 1
            denominator += row["rounds"] - 1
            exact = False
    return ((low / denominator, high / denominator), exact) if denominator > 0 else None


def _variants(left, right, budget, block):
    trimmed = [sum(min(r["tokens"], budget) for r in rows) / sum(r["rounds"] for r in rows)
               for rows in (left, right)]
    stock_within, tuned_within = (_within_budget(rows, budget, block) for rows in (left, right))
    if stock_within is None or tuned_within is None:
        within = {"stock": None, "tuned": None, "relative_gain": None, "exact": False,
                  "note": "undefined: a request crossed the budget in its only round"}
    else:
        (stock_lo, stock_hi), stock_exact = stock_within
        (tuned_lo, tuned_hi), tuned_exact = tuned_within
        exact = stock_exact and tuned_exact
        within = {"stock": [stock_lo, stock_hi], "tuned": [tuned_lo, tuned_hi],
                  "relative_gain": [tuned_lo / stock_hi - 1, tuned_hi / stock_lo - 1],
                  "exact": exact,
                  "note": "crossing round dropped; an envelope unless round_lengths were recorded"}
    return {
        "whole_rounds": {"stock": acceptance(left), "tuned": acceptance(right),
                         "relative_gain": acceptance(right) / acceptance(left) - 1,
                         "note": "the headline metric: every committed token over every round"},
        "budget_trimmed_numerator": {
            "stock": trimmed[0], "tuned": trimmed[1],
            "relative_gain": trimmed[1] / trimmed[0] - 1,
            "note": "trimmed numerator over whole rounds; a mixed metric, sensitivity only"},
        "complete_rounds_within_budget": within,
    }


def aligned(first, second):
    """(left, right, budget, block) once both arms are complete, paired and well formed.

    Shared by both criteria on purpose. Whether the question is "better" or "the same", a
    comparison is only worth making over records that finished, agree on their settings, and
    describe runs this harness's arithmetic can account for; the two gates should never be
    able to disagree about whether a pair of reports is admissible.
    """
    if not first.get("complete") or not second.get("complete"):
        raise ValueError("incomplete arm")
    if first["settings"] != second["settings"]:
        raise ValueError("settings differ")
    budget, block = _settings(first)
    left, right = first["requests"], second["requests"]
    if not left or len(left) != len(right):
        raise ValueError("empty or unequal prompt sets")
    ids = [r["prompt_sha256"] for r in left]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompts")
    for a, b in zip(left, right, strict=True):
        if any(a[k] != b[k] for k in ("prompt_sha256", "thinking", "category")):
            raise ValueError("prompt order or strata differ")
        for row in (a, b):
            _check_row(row, budget, block)
    return left, right, budget, block


def compare(stock, tuned, draws=10000):
    left, right, budget, block = aligned(stock, tuned)

    def gain(indices):
        return acceptance([right[i] for i in indices]) / acceptance([left[i] for i in indices]) - 1

    # Resample prompts as pairs, not rounds as independent observations. Preserve the
    # fixed category/thinking composition so a draw does not change the workload mix.
    strata = {}
    for i, row in enumerate(left):
        strata.setdefault((row["category"], row["thinking"]), []).append(i)
    rng = random.Random(7)
    samples = sorted(gain([rng.choice(group) for group in strata.values() for _ in group])
                     for _ in range(draws))
    lo, hi = samples[int(draws * .025)], samples[min(draws - 1, int(draws * .975))]
    classes = dict.fromkeys(
        ("identical", "boundary_overshoot", "prefix_divergence", "early_stop_mismatch"), 0)
    overshoot = 0
    for a, b in zip(left, right, strict=True):
        classes[classify(a, b, budget, block)] += 1
        overshoot = max(overshoot, a["tokens"] - budget, b["tokens"] - budget)
    blocking = sum(classes[name] for name in BLOCKING)
    breakdown = []
    for field in ("category", "thinking"):
        for value in dict.fromkeys(row[field] for row in left):
            groups = [[row for row in rows if row[field] == value] for rows in (left, right)]
            breakdown.append({"field": field, "value": value, "prompts": len(groups[0]),
                              "relative_gain": acceptance(groups[1]) / acceptance(groups[0]) - 1})
    return {"contract": CONTRACT, "prompts": len(left), "budget": budget, "block": block,
            "stock_tokens_per_round": acceptance(left),
            "tuned_tokens_per_round": acceptance(right), "relative_gain": gain(range(len(left))),
            "paired_95_interval": [lo, hi],
            "output_classes": classes, "max_overshoot_tokens": max(overshoot, 0),
            "output_mismatches": sum(a["response_ids"] != b["response_ids"]
                                     for a, b in zip(left, right, strict=True)),
            "stock_truncated": sum(r["finish"] == "length" for r in left),
            "tuned_truncated": sum(r["finish"] == "length" for r in right),
            "acceptance_variants": _variants(left, right, budget, block),
            "breakdown": breakdown,
            "gate": "pass" if lo > 0 and blocking == 0 else "investigate",
            "scope": "fixed prompt suite, stratified paired bootstrap; not population coverage. "
                     "Truncated responses are not complete answers: read stock_truncated."}


def equivalence(before, after):
    """Are these two arms the same served loop? EQUIVALENCE_CONTRACT, field by field.

    Written for docs/prequantized-ft5-plan.md: the fine-tuned drafter and its prequantized
    repackaging hold the same weights in the same precision, so at temperature 0 the served
    loop should be indistinguishable, and any difference at all is a finding rather than a
    result. That is the opposite of :func:`compare`, whose gate needs a positive lower bound
    and therefore reads an exactly-equal pair as `investigate`.

    `round_lengths` are required rather than optional here. Two arms can emit identical
    tokens over an identical number of rounds and still have committed them in different
    blocks, and without the lengths that difference is invisible; `served_accept.py` has
    not always recorded them, so a record that lacks them is one this criterion
    cannot speak about.
    """
    left, right, budget, block = aligned(before, after)
    missing = [r["prompt_sha256"] for rows in (left, right) for r in rows
               if r.get("round_lengths") is None]
    if missing:
        raise ValueError(
            f"{len(missing)} records carry no round_lengths; artifact equivalence needs them")
    mismatches = {field: 0 for field in EQUIVALENCE_FIELDS}
    differing = []
    classes = dict.fromkeys(
        ("identical", "boundary_overshoot", "prefix_divergence", "early_stop_mismatch"), 0)
    overshoot = 0
    for a, b in zip(left, right, strict=True):
        fields = [field for field in EQUIVALENCE_FIELDS if a[field] != b[field]]
        for field in fields:
            mismatches[field] += 1
        if fields:
            differing.append({"prompt_sha256": a["prompt_sha256"], "category": a["category"],
                              "thinking": a["thinking"], "fields": fields,
                              "identity_class": classify(a, b, budget, block)})
        classes[classify(a, b, budget, block)] += 1
        overshoot = max(overshoot, a["tokens"] - budget, b["tokens"] - budget)
    equal = len(left) - len(differing)
    return {"contract": EQUIVALENCE_CONTRACT, "identity_contract": CONTRACT,
            "prompts": len(left), "budget": budget, "block": block,
            "equal_prompts": equal, "differing_prompts": len(differing),
            "field_mismatches": mismatches,
            # The first few, named. A gate that says only "12 differ" sends the operator
            # back to the raw reports to find out which, which is where the
            # identity investigation started.
            "differences": differing[:10],
            "output_classes": classes, "max_overshoot_tokens": max(overshoot, 0),
            "before_tokens_per_round": acceptance(left),
            "after_tokens_per_round": acceptance(right),
            "gate": "pass" if not differing else "investigate",
            "scope": "exact per-prompt equality of tokens, finish reason, round count and "
                     "round lengths; decode durations are not compared and no speed claim "
                     "follows from this. A pass says the two packagings ran the same loop "
                     "on this fixed prompt suite, not that they are equal in general."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stock")
    ap.add_argument("tuned")
    ap.add_argument("--out", help="write the result here as well as printing it")
    ap.add_argument("--equivalence", action="store_true",
                    help="ask whether the two arms ran the same loop (EQUIVALENCE_CONTRACT) "
                         "instead of whether the second drafts better")
    args = ap.parse_args()
    with open(args.stock) as f:
        stock = json.load(f)
    with open(args.tuned) as f:
        tuned = json.load(f)
    try:
        result = equivalence(stock, tuned) if args.equivalence else compare(stock, tuned)
    except (KeyError, ValueError) as exc:
        ap.error(f"refusing comparison: {exc}")
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0 if result["gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
