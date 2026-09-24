#!/usr/bin/env python3
"""Model-free checks for the report specification: the schema, its validator, and their
agreement with the analyser and with what served_accept.py refuses before it writes.

bench/REPORTS.md is the specification. These hold four things to it: the schema is valid
JSON that the validator can enforce in full; a minimal hand-built report passes, with or
without the optional blocks; each rule the specification states rejects a report that
breaks it, with a message that names the field; and served_accept.py's own checks refuse
values the report format refuses. Where a rule is the analyser's own, the analyser must
refuse the same report, so the validator cannot drift from what it admits.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate = _load("validate_report", BENCH / "analysis" / "validate_report.py")
analysis = _load("analyse_served_accept", BENCH / "analysis" / "analyse_served_accept.py")
checks = _load("report_checks", BENCH / "drafter" / "report_checks.py")

BUDGET, CAP = 8, 3      # block = cap + 1 = 4


def prompt_sha256(prompt_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()


def minimal() -> dict[str, Any]:
    """Two prompts: one stops on the budget, one on a stop token."""
    return {
        "drafter": "example/drafter",
        "settings": {"max_new": BUDGET, "cap": CAP, "temperature": 0.0},
        "complete": True,
        "requests": [
            # Seed token, then rounds of 4 and 3: 1 + 7 = 8 tokens, exactly the budget.
            {"prompt_sha256": prompt_sha256([1, 2, 3]), "category": "reasoning",
             "thinking": False, "tokens": 8, "rounds": 2, "round_lengths": [4, 3],
             "finish": "length", "response_ids": list(range(8)), "decode_seconds": 0.5},
            # Entered the last round at 1 + 2 = 3 tokens; a stop token after 2 of its 4.
            {"prompt_sha256": prompt_sha256([4, 5]), "category": "reasoning",
             "thinking": True, "tokens": 5, "rounds": 2, "round_lengths": [2, 4],
             "finish": "stop", "response_ids": [9, 8, 7, 6, 5], "decode_seconds": 0.25},
        ],
    }


class SchemaFileTests(unittest.TestCase):
    def test_the_schema_is_json_and_declares_its_draft(self) -> None:
        with (BENCH / "report.schema.json").open() as handle:
            schema = json.load(handle)
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")

    def test_every_keyword_is_one_the_validator_enforces(self) -> None:
        validate.load_schema()      # raises on a keyword it would otherwise skip
        with self.assertRaisesRegex(ValueError, "not supported"):
            validate.check_schema_keywords({"type": "object", "oneOf": []})

    def test_the_schema_requires_what_the_analyser_reads(self) -> None:
        schema = validate.load_schema()
        request = schema["$defs"]["request"]["required"]
        for field in ("prompt_sha256", "category", "thinking", "tokens", "rounds", "finish",
                      "response_ids", "decode_seconds"):
            self.assertIn(field, request)
        self.assertEqual(schema["$defs"]["request"]["properties"]["finish"]["enum"],
                         ["length", "stop"])
        for field in analysis.EQUIVALENCE_FIELDS:
            self.assertIn(field, schema["$defs"]["request"]["properties"])


class ValidReportTests(unittest.TestCase):
    def assertValid(self, report: dict[str, Any]) -> None:
        self.assertEqual(validate.report_errors(report), [])
        analysis.aligned(report, copy.deepcopy(report))     # and the analyser admits it

    def test_a_minimal_report_passes(self) -> None:
        self.assertValid(minimal())

    def test_round_lengths_are_optional(self) -> None:
        report = minimal()
        for row in report["requests"]:
            del row["round_lengths"]
        self.assertValid(report)

    def test_the_adapter_blocks_pass(self) -> None:
        report = minimal()
        report.update({"drafter": None, "mode": "target_only", "arm": {"load_seconds": 1.0},
                       "summary": {"tokens_per_round": 1.0, "runtime_tokens_per_cycle": None,
                                   "decode_tokens_per_second": 20.0, "wall_seconds": 3.0}})
        report["settings"].update({"kv_bits": None, "corpus_sha256": "0" * 64})
        report["requests"][0].update({
            "prefill_seconds": 0.1, "prompt_tokens": 3,
            "runtime": {"passes": 2, "committed_tokens": 8, "acceptance": [3, 3],
                        "block_lens": [4, 4], "excluded_passes": [], "post_stop_tokens": 0,
                        "tokens_per_cycle": 4.0, "peak_memory_gb": 12.5}})
        self.assertValid(report)

    def test_a_boundary_overshoot_under_one_block_passes(self) -> None:
        # Entered the last round at 1 + 4 + 2 = 7, below the budget, and committed a whole
        # block: 11 tokens against a budget of 8, an overshoot of 3 = cap.
        report = minimal()
        report["requests"][0].update({"tokens": 11, "rounds": 3, "round_lengths": [4, 2, 4],
                                      "response_ids": list(range(11))})
        self.assertValid(report)

    def test_the_command_line(self) -> None:
        import contextlib
        import io
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp, "good.json"), Path(tmp, "bad.json")
            good.write_text(json.dumps(minimal()))
            broken = minimal()
            broken["requests"][0]["finish"] = "eos"
            bad.write_text(json.dumps(broken))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(validate.main([str(good)]), 0)
                self.assertEqual(validate.main([str(good), str(bad)]), 1)
            self.assertIn("FAIL", out.getvalue())
            self.assertIn("$.requests[0].finish", out.getvalue())


class BrokenReportTests(unittest.TestCase):
    def assertRejected(self, report: dict[str, Any], message: str,
                       analyser_refuses: bool = True) -> None:
        errors = validate.report_errors(report)
        self.assertTrue(errors, "the validator admitted a broken report")
        self.assertTrue(any(message in line for line in errors),
                        f"no error mentions {message!r}: {errors}")
        if analyser_refuses:
            with self.assertRaises((KeyError, ValueError)):
                analysis.aligned(report, copy.deepcopy(report))

    def broken(self, index: int = 0, **fields: Any) -> dict[str, Any]:
        report = minimal()
        report["requests"][index].update(fields)
        return report

    def test_a_missing_field(self) -> None:
        for field in ("tokens", "rounds", "finish", "response_ids", "decode_seconds",
                      "prompt_sha256", "category", "thinking"):
            report = minimal()
            del report["requests"][0][field]
            self.assertRejected(report, f"missing required field {field!r}")
        for field in ("max_new", "cap", "temperature"):
            report = minimal()
            del report["settings"][field]
            self.assertRejected(report, f"missing required field {field!r}")

    def test_an_unknown_finish_reason(self) -> None:
        self.assertRejected(self.broken(finish="eos"), "$.requests[0].finish")

    def test_counts_must_be_integer_literals(self) -> None:
        self.assertRejected(self.broken(tokens=8.0), "$.requests[0].tokens: expected integer")
        self.assertRejected(self.broken(rounds=True), "$.requests[0].rounds: expected integer")
        self.assertRejected(self.broken(round_lengths=[4.0, 3]), "$.requests[0].round_lengths[0]")

    def test_a_malformed_prompt_hash(self) -> None:
        self.assertRejected(self.broken(prompt_sha256="ABC"), "does not match",
                            analyser_refuses=False)

    def test_a_hash_with_a_trailing_newline(self) -> None:
        # Python's `$` matches before a final newline; a pattern here must not.
        good = minimal()["requests"][0]["prompt_sha256"]
        self.assertRejected(self.broken(prompt_sha256=good + "\n"), "does not match",
                            analyser_refuses=False)
        report = minimal()
        report["settings"]["corpus_sha256"] = "0" * 64 + "\n"
        self.assertRejected(report, "$.settings.corpus_sha256", analyser_refuses=False)
        self.assertEqual(validate.schema_errors("a\n", {"pattern": "^a$"}, {}),
                         ["$: 'a\\n' does not match ^a$"])

    def test_sampled_decoding_is_not_a_report(self) -> None:
        for value in (1.0, 1, False, "0"):
            with self.subTest(value=value):
                report = minimal()
                report["settings"]["temperature"] = value
                self.assertRejected(report, "$.settings.temperature")

    def test_budget_and_cap_are_integer_literals(self) -> None:
        for key, value in (("max_new", 8.0), ("max_new", True), ("cap", 3.0), ("cap", True)):
            with self.subTest(key=key, value=value):
                report = minimal()
                report["settings"][key] = value
                self.assertRejected(report, f"$.settings.{key}: expected integer")

    def test_strata_and_token_ids_are_typed(self) -> None:
        for fields, message in (({"category": None}, "$.requests[0].category: expected string"),
                                ({"thinking": 1}, "$.requests[0].thinking: expected boolean"),
                                ({"response_ids": [0.0, *range(1, 8)]},
                                 "$.requests[0].response_ids[0]: expected integer"),
                                ({"response_ids": [True, *range(1, 8)]},
                                 "$.requests[0].response_ids[0]: expected integer")):
            with self.subTest(fields=fields):
                self.assertRejected(self.broken(**fields), message)

    def test_decode_seconds_is_a_positive_number(self) -> None:
        self.assertRejected(self.broken(decode_seconds=True), "expected number")
        self.assertRejected(self.broken(decode_seconds=0), "must be greater than 0")
        # NaN passes the schema's exclusiveMinimum, which compares false, but not the analyser.
        self.assertRejected(self.broken(decode_seconds=math.nan), "not a positive number")

    def test_the_analysers_settings_check_backs_the_schema(self) -> None:
        # With the schema's rule removed, the analyser's own check still rejects the report.
        schema = validate.load_schema()
        schema["$defs"]["settings"]["properties"]["temperature"] = {}
        report = minimal()
        report["settings"]["temperature"] = 0.5
        errors = validate.report_errors(report, schema)
        self.assertTrue(any(line.startswith("$.settings: settings.temperature is 0.5")
                            for line in errors), errors)

    def test_a_non_positive_budget_or_cap(self) -> None:
        report = minimal()
        report["settings"]["cap"] = 0
        self.assertRejected(report, "$.settings.cap")

    def test_an_incomplete_or_empty_arm(self) -> None:
        report = minimal()
        report["complete"] = False
        self.assertRejected(report, "incomplete")
        report = minimal()
        report["requests"] = []
        self.assertRejected(report, "empty")

    def test_duplicate_prompts(self) -> None:
        report = minimal()
        report["requests"][1]["prompt_sha256"] = report["requests"][0]["prompt_sha256"]
        self.assertRejected(report, "duplicates requests[0]")

    def test_tokens_must_match_the_output(self) -> None:
        self.assertRejected(self.broken(response_ids=list(range(7))), "disagrees with output")

    def test_finish_length_below_the_budget(self) -> None:
        self.assertRejected(self.broken(1, finish="length"), "below the requested budget")

    def test_an_overshoot_of_a_whole_block(self) -> None:
        self.assertRejected(self.broken(tokens=12, rounds=3, round_lengths=[4, 3, 4],
                                        response_ids=list(range(12))), "whole block")

    def test_the_seed_token_is_counted(self) -> None:
        # sum(round_lengths) == tokens is the off-by-one an adapter makes when it forgets
        # that the prefill's token belongs to no round.
        self.assertRejected(self.broken(round_lengths=[4, 4]), "different count")

    def test_round_lengths_must_match_the_round_count(self) -> None:
        self.assertRejected(self.broken(round_lengths=[4, 2, 1]), "round count")

    def test_a_round_longer_than_one_block(self) -> None:
        self.assertRejected(self.broken(round_lengths=[5, 2]), "more than one block")

    def test_a_round_that_began_at_the_budget(self) -> None:
        # Rounds of 4 and 3 fill the budget of 8; a third round cannot begin.
        self.assertRejected(self.broken(tokens=9, rounds=3, round_lengths=[4, 3, 1],
                                        response_ids=list(range(9))), "at or past")

    def test_a_stop_cannot_unsay_earlier_rounds(self) -> None:
        # Rounds of 4 and 4 put the last round's start at 5, but only 4 tokens were emitted.
        self.assertRejected(self.broken(1, tokens=4, round_lengths=[4, 4],
                                        response_ids=[9, 8, 7, 6]), "final round")


class WriterCheckTests(unittest.TestCase):
    """served_accept.py refuses, before it writes, values the report format refuses.

    The dflash-mlx adapter's equivalents are exercised by its --self-test.
    """

    def test_corpus_strata(self) -> None:
        self.assertEqual(checks.corpus_strata({"thinking": True}), ("unknown", True))
        self.assertEqual(checks.corpus_strata({"thinking": False, "category": "code"}),
                         ("code", False))
        for row, message in (({"thinking": 1}, "boolean thinking"),
                             ({"thinking": "true"}, "boolean thinking"),
                             ({}, "boolean thinking"),
                             ({"thinking": True, "category": None}, "must be a string"),
                             ({"thinking": True, "category": 3}, "must be a string")):
            with self.subTest(row=row):
                with self.assertRaisesRegex(ValueError, message):
                    checks.corpus_strata(row)
                if "thinking" in row:       # and the same value in a record is not a report
                    report = minimal()
                    report["requests"][0].update(row)
                    self.assertTrue(validate.report_errors(report))

    def test_decode_seconds(self) -> None:
        self.assertEqual(checks.decode_seconds(3.5, 1.0), 2.5)
        for seconds, prefill in ((1.0, 1.0), (1.0, 2.0), (math.nan, 0.0), (math.inf, 0.0)):
            with (self.subTest(seconds=seconds, prefill=prefill),
                  self.assertRaisesRegex(ValueError, "no positive decode time")):
                checks.decode_seconds(seconds, prefill)
        for decode in (0.0, -1.0):         # the finite ones, as a record would carry them
            report = minimal()
            report["requests"][0]["decode_seconds"] = decode
            self.assertTrue(validate.report_errors(report))


if __name__ == "__main__":
    unittest.main()
