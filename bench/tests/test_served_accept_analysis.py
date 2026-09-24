#!/usr/bin/env python3
"""Model-free checks for the paired acceptance gate.

The contract these hold to the wall: a difference inside the requested budget blocks the
gate, a difference only above it does not, and the headline metric stays whole-round.
Version 1 of the gate conflated the first two and rejected four clean arms; see the
analyser's docstring.
"""
import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analysis", Path(__file__).resolve().parents[1] / "analysis/analyse_served_accept.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

BUDGET, CAP = 8, 7      # block = cap + 1 = 8, so one round can commit the whole budget

# `dflash_generate` seeds `out_ids` with the prefill's own token before any round runs, so a
# clean run emits 1 + sum(round_lengths) tokens and a round exists only if the count before
# it was below the budget. Every history here obeys both; a code review found the
# earlier fixtures did not, and a validator cannot be tested against runs that cannot happen.
# See `_check_row`'s docstring for the loop these numbers are read off.


def history(*lengths):
    """(tokens, rounds, round_lengths) for a clean run of these committed blocks."""
    return 1 + sum(lengths), len(lengths), list(lengths)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        tokens, rounds, lengths = history(2, 2, 2, 1)       # 8 tokens over 4 rounds
        self.stock = {"complete": True, "requests": [
            {"prompt_sha256": str(i), "thinking": bool(i % 2), "category": "general",
             "tokens": tokens, "rounds": rounds, "decode_seconds": 1, "finish": "length",
             "response_ids": list(range(tokens)), "round_lengths": lengths} for i in range(4)],
            "settings": {"max_new": BUDGET, "cap": CAP, "temperature": 0}}
        self.tuned = copy.deepcopy(self.stock)
        tokens, rounds, lengths = history(4, 3)             # the same 8 tokens over 2 rounds
        for row in self.tuned["requests"]:
            row["rounds"] = rounds
            row["round_lengths"] = lengths

    def overshoot(self, row, extra):
        """Make a row finish `extra` tokens past the budget, as the round loop does.

        The loop tests the budget at the top of a round, so the crossing round has to be
        ENTERED below it. These lengths take the arm to `BUDGET - 1` emitted tokens and then
        commit one more block of `extra + 1`. The previous version appended a round after the
        budget was already exhausted, which no loop does.
        """
        block = row["round_lengths"][0]
        lengths = []
        while sum(lengths) + block <= BUDGET - 2:
            lengths.append(block)
        if sum(lengths) < BUDGET - 2:
            lengths.append(BUDGET - 2 - sum(lengths))
        lengths.append(extra + 1)
        row["tokens"] = BUDGET + extra
        row["response_ids"] = list(range(BUDGET + extra))
        row["rounds"] = len(lengths)
        row["round_lengths"] = lengths

    def test_gain(self):
        result = analysis.compare(self.stock, self.tuned, draws=100)
        self.assertEqual(result["contract"], analysis.CONTRACT)
        self.assertEqual(result["relative_gain"], 1)
        self.assertEqual(result["paired_95_interval"], [1, 1])
        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["stock_truncated"], 4)
        self.assertEqual(result["output_classes"]["identical"], 4)
        self.assertEqual(result["max_overshoot_tokens"], 0)

    def test_boundary_overshoot_passes_but_is_named(self):
        self.overshoot(self.tuned["requests"][0], 3)
        result = analysis.compare(self.stock, self.tuned, 100)
        self.assertEqual(result["output_classes"],
                         {"identical": 3, "boundary_overshoot": 1,
                          "prefix_divergence": 0, "early_stop_mismatch": 0})
        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["max_overshoot_tokens"], 3)
        self.assertEqual(result["output_mismatches"], 1)   # version 1's count, still reported

    def test_prefix_divergence_blocks_gate(self):
        self.tuned["requests"][0]["response_ids"][0] = 99
        result = analysis.compare(self.stock, self.tuned, 100)
        self.assertEqual(result["output_classes"]["prefix_divergence"], 1)
        self.assertEqual(result["gate"], "investigate")

    def test_divergence_above_the_budget_is_still_caught_inside_it(self):
        # Both arms overshoot, but disagree on a token the budget covers.
        self.overshoot(self.tuned["requests"][0], 3)
        self.tuned["requests"][0]["response_ids"][7] = 99
        self.assertEqual(analysis.compare(self.stock, self.tuned, 100)["output_classes"]
                         ["prefix_divergence"], 1)

    def test_early_stop_mismatch_blocks_gate(self):
        # EOS inside the second block: 1 + 3 emitted when it started, one more before the break.
        row = self.tuned["requests"][0]
        row.update(tokens=5, response_ids=list(range(5)), finish="stop",
                   rounds=2, round_lengths=[3, 1])
        result = analysis.compare(self.stock, self.tuned, 100)
        self.assertEqual(result["output_classes"]["early_stop_mismatch"], 1)
        self.assertEqual(result["gate"], "investigate")

    def test_headline_metric_is_whole_round(self):
        self.overshoot(self.tuned["requests"][0], 3)
        result = analysis.compare(self.stock, self.tuned, 100)
        variants = result["acceptance_variants"]
        tuned_rows = self.tuned["requests"]
        self.assertEqual(result["tuned_tokens_per_round"],
                         sum(r["tokens"] for r in tuned_rows) / sum(r["rounds"] for r in tuned_rows))
        self.assertEqual(variants["whole_rounds"]["tuned"], result["tuned_tokens_per_round"])
        # Trimming the numerator lowers it; whole rounds stay in the denominator either way.
        self.assertLess(variants["budget_trimmed_numerator"]["tuned"],
                        variants["whole_rounds"]["tuned"])
        within = variants["complete_rounds_within_budget"]
        self.assertTrue(within["exact"])
        self.assertEqual(within["tuned"][0], within["tuned"][1])   # exact: a point, not a range

    def test_eos_in_the_crossing_block_is_not_exact(self):
        # A block that crosses the budget and ends in EOS: the loop counts the whole committed
        # block but appends only up to EOS, so the lengths sum past the tokens and the crossing
        # round cannot be read off. The flag must say so even though round_lengths are present.
        row = self.tuned["requests"][0]
        row.update(tokens=10, response_ids=list(range(10)), finish="stop",
                   rounds=3, round_lengths=[4, 2, 5])
        within = analysis.compare(self.stock, self.tuned, 100)["acceptance_variants"][
            "complete_rounds_within_budget"]
        self.assertFalse(within["exact"])
        self.assertLess(within["tuned"][0], within["tuned"][1])

    def test_within_budget_is_an_envelope_without_round_lengths(self):
        self.overshoot(self.tuned["requests"][0], 3)
        for arm in (self.stock, self.tuned):
            for row in arm["requests"]:
                row.pop("round_lengths")
        within = analysis.compare(self.stock, self.tuned, 100)["acceptance_variants"][
            "complete_rounds_within_budget"]
        self.assertFalse(within["exact"])
        low, high = within["tuned"]
        self.assertLess(low, high)
        # The crossing round committed between overshoot + 1 = 4 and one block = 8 tokens,
        # over the eight rounds left once it is dropped.
        self.assertEqual([low, high], [(11 - 8 + 24) / 8, (11 - 4 + 24) / 8])

    def test_refusals(self):
        for key, value in (("complete", False),
                           ("settings", {"max_new": BUDGET, "cap": 3, "temperature": 0}),
                           ("requests", [])):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.tuned)
                bad[key] = value
                with self.assertRaises(ValueError):
                    analysis.compare(self.stock, bad, 100)
        for name, change in (
                ("unknown finish", {"finish": "filtered"}),
                ("length below budget", {"finish": "length", "tokens": 5,
                                         "response_ids": list(range(5)), "round_lengths": [3, 2]}),
                ("overshoot past a block", {"tokens": 16, "response_ids": list(range(16)),
                                            "rounds": 3, "round_lengths": [4, 4, 8]}),
                ("round count disagrees", {"round_lengths": [4, 4, 4]}),
                ("a round over one block", {"round_lengths": [9, -1]}),
                ("round lengths below tokens", {"round_lengths": [2, 2]}),
                ("count disagrees with output", {"tokens": 7}),
                ("zero rounds", {"rounds": 0})):
            with self.subTest(name=name):
                bad = copy.deepcopy(self.tuned)
                bad["requests"][0].update(change)
                with self.assertRaises(ValueError):
                    analysis.compare(self.stock, bad, 100)
        self.tuned["requests"].reverse()
        with self.assertRaises(ValueError):
            analysis.compare(self.stock, self.tuned, 100)

    def test_more_tokens_than_the_rounds_could_commit(self):
        # cap 3 is a four-token block, so eight tokens in two rounds is not a run this
        # contract describes -- most likely two arms compared at different caps.
        for arm in (self.stock, self.tuned):
            arm["settings"] = {"max_new": BUDGET, "cap": 3, "temperature": 0}
            for row in arm["requests"]:
                row.pop("round_lengths")
        self.tuned["requests"][0]["rounds"] = 1
        with self.assertRaises(ValueError):
            analysis.compare(self.stock, self.tuned, 100)

    def test_no_gain(self):
        self.assertEqual(analysis.compare(self.stock, self.stock, 100)["gate"], "investigate")


class EquivalenceTests(unittest.TestCase):
    """The second criterion: two packagings of one drafter ran the same loop, or they did not.

    The case that matters most is the first one. A prequantized repackaging is expected to
    show zero gain, which the improvement gate reports as `investigate`; if that verdict were
    reused as this release's pass/fail, the release would be blocked by its own success.
    """

    def setUp(self):
        tokens, rounds, lengths = history(3, 3, 1)          # 8 tokens over 3 rounds
        self.source = {"complete": True, "requests": [
            {"prompt_sha256": str(i), "thinking": bool(i % 2), "category": "general",
             "tokens": tokens, "rounds": rounds, "decode_seconds": 1.5, "finish": "length",
             "response_ids": list(range(tokens)), "round_lengths": lengths} for i in range(4)],
            "settings": {"max_new": BUDGET, "cap": CAP, "temperature": 0}}
        self.artifact = copy.deepcopy(self.source)

    def test_an_identical_pair_passes_where_the_improvement_gate_investigates(self):
        result = analysis.equivalence(self.source, self.artifact)
        self.assertEqual(result["contract"], analysis.EQUIVALENCE_CONTRACT)
        self.assertEqual(result["gate"], "pass")
        self.assertEqual(result["equal_prompts"], 4)
        self.assertEqual(result["differing_prompts"], 0)
        self.assertEqual(result["output_classes"]["identical"], 4)
        self.assertEqual(set(result["field_mismatches"].values()), {0})
        self.assertEqual(analysis.compare(self.source, self.artifact, 100)["gate"], "investigate")

    def test_durations_may_differ(self):
        for row in self.artifact["requests"]:
            row["decode_seconds"] = 0.9
        self.assertEqual(analysis.equivalence(self.source, self.artifact)["gate"], "pass")

    def test_every_compared_field_blocks_and_is_named(self):
        cases = {
            "response_ids": {"response_ids": [99] + list(range(1, 8))},
            # The same eight tokens, ended by EOS in the final block rather than by the budget.
            "finish": {"finish": "stop"},
            # A tokens-only difference is not constructible: finish=length forces
            # tokens >= budget, and any other count changes the array with it. So this is a
            # longer run, which must name `tokens` among the fields that differ.
            "tokens": {"tokens": 11, "response_ids": list(range(11)),
                       "round_lengths": [3, 3, 4]},
            "rounds": {"rounds": 4, "round_lengths": [3, 2, 1, 1]},
            "round_lengths": {"round_lengths": [2, 4, 1]},
        }
        for field, change in cases.items():
            with self.subTest(field=field):
                spoiled = copy.deepcopy(self.artifact)
                spoiled["requests"][0].update(change)
                result = analysis.equivalence(self.source, spoiled)
                self.assertEqual(result["gate"], "investigate")
                self.assertGreaterEqual(result["field_mismatches"][field], 1)
                self.assertEqual(result["differing_prompts"], 1)
                self.assertIn(field, result["differences"][0]["fields"])
                self.assertEqual(result["differences"][0]["prompt_sha256"], "0")

    def test_the_same_rounds_with_different_block_lengths_blocks(self):
        # Identical tokens over an identical round count, committed differently. Invisible
        # without round_lengths, which is why they are required rather than optional.
        self.artifact["requests"][0]["round_lengths"] = [1, 3, 3]
        result = analysis.equivalence(self.source, self.artifact)
        self.assertEqual(result["gate"], "investigate")
        self.assertEqual(result["field_mismatches"]["round_lengths"], 1)
        self.assertEqual(result["field_mismatches"]["response_ids"], 0)
        self.assertEqual(result["output_classes"]["identical"], 4)   # the identity half agrees

    def test_boundary_overshoot_is_named_but_still_blocks(self):
        row = self.artifact["requests"][0]
        row.update(tokens=11, response_ids=list(range(11)), round_lengths=[3, 3, 4])
        result = analysis.equivalence(self.source, self.artifact)
        self.assertEqual(result["output_classes"]["boundary_overshoot"], 1)
        self.assertEqual(result["max_overshoot_tokens"], 3)
        self.assertEqual(result["gate"], "investigate")
        self.assertEqual(result["differences"][0]["identity_class"], "boundary_overshoot")

    def test_round_lengths_are_required(self):
        self.artifact["requests"][0].pop("round_lengths")
        with self.assertRaises(ValueError):
            analysis.equivalence(self.source, self.artifact)
        self.source["requests"][0].pop("round_lengths")
        with self.assertRaises(ValueError):
            analysis.equivalence(self.source, self.artifact)

    def test_it_refuses_exactly_what_the_improvement_gate_refuses(self):
        for name, spoil in (
                ("incomplete arm", lambda arm: arm.update(complete=False)),
                ("settings differ", lambda arm: arm.update(
                    settings={"max_new": BUDGET, "cap": 3, "temperature": 0})),
                ("no requests", lambda arm: arm.update(requests=[])),
                ("unknown finish", lambda arm: arm["requests"][0].update(finish="filtered")),
                ("zero rounds", lambda arm: arm["requests"][0].update(rounds=0)),
                ("a round over one block",
                 lambda arm: arm["requests"][0].update(round_lengths=[9, -1, 1])),
                ("a round after the budget was reached",
                 lambda arm: arm["requests"][0].update(tokens=9, response_ids=list(range(9)),
                                                       rounds=2, round_lengths=[7, 1])),
                ("prompt order differs", lambda arm: arm["requests"].reverse())):
            with self.subTest(name=name):
                bad = copy.deepcopy(self.artifact)
                spoil(bad)
                with self.assertRaises(ValueError):
                    analysis.equivalence(self.source, bad)
                with self.assertRaises(ValueError):
                    analysis.compare(self.source, bad, 100)


class RoundHistoryTests(unittest.TestCase):
    """Whether a recorded round history is one the pinned loop could have produced.

    Both gates share `_check_row`, so a history it waves through can be certified as
    `artifact-equivalence/v1: pass` simply by appearing in both arms. The two cases a
    code review demonstrated are the first two tests here.
    """

    def row(self, tokens, rounds, lengths, finish="length"):
        return {"complete": True, "requests": [
            {"prompt_sha256": "p", "thinking": False, "category": "general",
             "tokens": tokens, "rounds": rounds, "decode_seconds": 1.0, "finish": finish,
             "response_ids": list(range(tokens)), "round_lengths": lengths}],
            "settings": {"max_new": BUDGET, "cap": CAP, "temperature": 0}}

    def assertRefused(self, report):
        """Refused by BOTH gates: they share the validator, and both promise to use it."""
        with self.assertRaises(ValueError):
            analysis.equivalence(report, copy.deepcopy(report))
        with self.assertRaises(ValueError):
            analysis.compare(report, copy.deepcopy(report), 100)

    def test_a_round_after_the_budget_was_exhausted(self):
        # The review's first case: eight tokens at a budget of eight over rounds of [8, 8].
        # Round one alone reaches the budget, so the loop never enters round two.
        self.assertRefused(self.row(8, 2, [8, 8]))

    def test_tokens_that_earlier_rounds_already_exceeded(self):
        # The review's second case. EOS truncates its own block; it cannot unsay two others.
        self.assertRefused(self.row(1, 3, [8, 8, 8], finish="stop"))

    def test_consistent_totals_do_not_excuse_a_premature_crossing(self):
        # 1 + 7 + 1 = 9 tokens, which the totals agree with exactly -- but the second round
        # began at the eighth token, and the loop tests the budget before starting one.
        self.assertRefused(self.row(9, 2, [7, 1]))

    def test_counts_must_be_integers(self):
        for name, change in (("fractional tokens", {"tokens": 8.0}),
                             ("fractional rounds", {"rounds": 2.0}),
                             ("fractional length", {"round_lengths": [4.0, 3]}),
                             ("a boolean round count", {"rounds": True}),
                             ("a boolean length", {"round_lengths": [True, 3]})):
            with self.subTest(name=name):
                report = self.row(8, 2, [4, 3])
                report["requests"][0].update(change)
                self.assertRefused(report)

    def test_the_seed_token_is_counted(self):
        # One full block emits nine tokens, not eight: `out_ids` already held the prefill's
        # own token. The rule this replaced read that legitimate run as impossible.
        self.assertEqual(analysis.equivalence(self.row(9, 1, [8]),
                                              self.row(9, 1, [8]))["gate"], "pass")
        self.assertRefused(self.row(10, 1, [8]))

    def test_a_valid_final_block_overshoot_passes(self):
        # Entered the last round at seven tokens, committed a whole block: 15 emitted.
        report = self.row(15, 2, [6, 8])
        self.assertEqual(analysis.equivalence(report, copy.deepcopy(report))["gate"], "pass")

    def test_a_valid_eos_truncation_passes(self):
        # Entered the last round at seven, EOS after three of its eight committed tokens.
        report = self.row(10, 2, [6, 8], finish="stop")
        self.assertEqual(analysis.equivalence(report, copy.deepcopy(report))["gate"], "pass")

    def test_legacy_records_without_lengths_still_pass_the_improvement_gate(self):
        report = self.row(8, 2, [4, 3])
        report["requests"][0].pop("round_lengths")
        self.assertEqual(analysis.compare(report, copy.deepcopy(report), 100)["gate"],
                         "investigate")     # zero gain, but admitted rather than refused
        with self.assertRaises(ValueError):
            analysis.equivalence(report, copy.deepcopy(report))


class FieldTypeTests(unittest.TestCase):
    """The per-field rules of bench/report.schema.json that the analyser enforces itself.

    Arithmetic and Python's equality accept a float or a bool where a count, a stratum or a
    token id belongs, so each of those is refused by name instead of being compared.
    """

    def setUp(self):
        self.arm = {"complete": True,
                    "settings": {"max_new": BUDGET, "cap": CAP, "temperature": 0},
                    "requests": [{"prompt_sha256": "p", "thinking": False, "category": "general",
                                  "tokens": 8, "rounds": 2, "decode_seconds": 1.0,
                                  "finish": "length", "response_ids": list(range(8)),
                                  "round_lengths": [4, 3]}]}

    def assertRefused(self, first, message, second=None):
        """Refused by BOTH gates, with a message that says why."""
        second = copy.deepcopy(first) if second is None else second
        with self.assertRaisesRegex(ValueError, message):
            analysis.equivalence(first, second)
        with self.assertRaisesRegex(ValueError, message):
            analysis.compare(first, second, 100)

    def spoiled(self, **fields):
        bad = copy.deepcopy(self.arm)
        bad["requests"][0].update(fields)
        return bad

    def test_the_fixture_is_admitted_at_either_spelling_of_zero(self):
        for temperature in (0, 0.0):        # both writers record 0.0
            self.arm["settings"]["temperature"] = temperature
            self.assertEqual(analysis.equivalence(self.arm, copy.deepcopy(self.arm))["gate"],
                             "pass")

    def test_budget_and_cap_are_integer_counts(self):
        for key, value in (("max_new", 8.0), ("max_new", True), ("cap", 7.0), ("cap", True)):
            with self.subTest(key=key, value=value):
                bad = copy.deepcopy(self.arm)
                bad["settings"][key] = value
                self.assertRefused(bad, f"settings.{key} is not an integer count")

    def test_decoding_must_be_greedy(self):
        for value in (0.7, 1, -1.0, False, None, "0"):
            with self.subTest(value=value):
                bad = copy.deepcopy(self.arm)
                bad["settings"]["temperature"] = value
                self.assertRefused(bad, "greedy decoding at temperature 0")

    def test_the_three_settings_are_required(self):
        for key in ("max_new", "cap", "temperature"):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.arm)
                del bad["settings"][key]
                self.assertRefused(bad, f"settings has no '{key}'")

    def test_strata_are_typed(self):
        for change, message in (({"category": None}, "category is not a string"),
                                ({"category": 3}, "category is not a string"),
                                ({"thinking": 0}, "thinking is not a boolean"),
                                ({"thinking": "false"}, "thinking is not a boolean")):
            with self.subTest(change=change):
                self.assertRefused(self.spoiled(**change), message)

    def test_a_stratum_that_only_compares_equal_does_not_pair(self):
        # `1 == True` in Python, so the pairing test alone would have admitted this pair.
        self.arm["requests"][0]["thinking"] = True
        self.assertRefused(self.arm, "thinking is not a boolean: 1", self.spoiled(thinking=1))

    def test_token_ids_are_integers(self):
        for ids in ([0.0, *range(1, 8)], [False, *range(1, 8)], ["0", *range(1, 8)],
                    "01234567", tuple(range(8))):
            with self.subTest(ids=ids):
                self.assertRefused(self.spoiled(response_ids=ids),
                                   "response_ids is not a list of integer token ids")

    def test_float_ids_that_equal_the_other_arm_do_not_pair(self):
        # `[0.0, 1, ...] == [0, 1, ...]`, so without the check this pair classed as identical.
        self.assertRefused(self.arm, "response_ids",
                           self.spoiled(response_ids=[0.0, *range(1, 8)]))

    def test_decode_seconds_is_a_positive_number(self):
        for value in (0, 0.0, -1.0, float("nan"), True, "1.0", None):
            with self.subTest(value=value):
                self.assertRefused(self.spoiled(decode_seconds=value),
                                   "decode_seconds is not a positive number")


if __name__ == "__main__":
    unittest.main()
