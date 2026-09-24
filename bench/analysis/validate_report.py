#!/usr/bin/env python3
"""Check served-acceptance reports against bench/REPORTS.md; standard library only.

    python3 bench/analysis/validate_report.py <report.json> [<report.json> ...]

Two passes per report. The first checks `bench/report.schema.json` with a small validator
for the subset of JSON Schema that file uses; a keyword outside that subset is an error in
the schema, not something to skip. The second checks what a schema cannot say: a complete
arm with at least one record, prompts unique by `prompt_sha256`, and every record against
the token accounting of the loop that produced it. The settings and record checks there are
the analyser's own `_settings` and `_check_row`, imported rather than restated, so this file
and the analyser cannot disagree about which reports are admissible. Some of their rules
(integer counts, a temperature of 0, the types of the strata and the token ids) are in the
schema as well, and the first pass reports them; the second backs it up.

It checks one report at a time. Whether two reports can be paired (equal `settings`, the same
prompts in the same order) is the analyser's to decide; see bench/REPORTS.md.

Exit status 0 when every report passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

BENCH = Path(__file__).resolve().parents[1]
SCHEMA_PATH = BENCH / "report.schema.json"
ANALYSER_PATH = BENCH / "analysis" / "analyse_served_accept.py"

#: Keywords this validator enforces, and the annotations it reads past. A schema that uses
#: anything else is refused, so a constraint can never be written down and silently skipped.
ENFORCED = frozenset({
    "type", "enum", "const", "required", "properties", "additionalProperties", "items",
    "minItems", "minimum", "exclusiveMinimum", "maximum", "pattern", "$ref",
})
ANNOTATIONS = frozenset({"$schema", "$id", "$defs", "$comment", "title", "description"})


def _is_type(value: Any, name: str) -> bool:
    # A count is an integer literal: the analyser refuses `True` and `4.0` where it expects
    # one, so this is stricter than JSON Schema's "any integral number".
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    kinds: dict[str, type | tuple[type, ...]] = {
        "string": str, "boolean": bool, "array": list, "object": dict, "null": type(None)}
    if name not in kinds:
        raise ValueError(f"schema uses an unknown type {name!r}")
    return isinstance(value, kinds[name])


def _equal(a: Any, b: Any) -> bool:
    """JSON equality: 0 == 0.0, but false is not 0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return bool(a == b)


def check_schema_keywords(schema: Any, where: str = "#") -> None:
    """Raise if the schema uses a keyword outside ENFORCED and ANNOTATIONS."""
    if not isinstance(schema, dict):
        # ValueError like every other refusal here: a malformed schema file, not a caller bug.
        raise ValueError(f"{where}: a schema must be an object")  # noqa: TRY004
    for key, value in schema.items():
        if key not in ENFORCED and key not in ANNOTATIONS:
            raise ValueError(f"{where}: keyword {key!r} is not supported by this validator")
        if key in ("properties", "$defs"):
            for name, sub in value.items():
                check_schema_keywords(sub, f"{where}/{key}/{name}")
        elif key == "items" or (key == "additionalProperties" and isinstance(value, dict)):
            check_schema_keywords(value, f"{where}/{key}")


def schema_errors(value: Any, schema: dict[str, Any], root: dict[str, Any],
                  path: str = "$") -> list[str]:
    """Every violation of `schema` by `value`, as 'path: message' strings."""
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError(f"only local references are supported: {ref}")
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        return schema_errors(value, target, root, path)
    errors: list[str] = []
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, name) for name in names):
            return [f"{path}: expected {' or '.join(names)}, got {json.dumps(value)[:60]}"]
    if "enum" in schema and not any(_equal(value, option) for option in schema["enum"]):
        errors.append(f"{path}: {json.dumps(value)[:60]} is not one of {schema['enum']}")
    if "const" in schema and not _equal(value, schema["const"]):
        errors.append(f"{path}: must be {json.dumps(schema['const'])}, got {json.dumps(value)[:60]}")
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} must be greater than {schema['exclusiveMinimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    # fullmatch, not search: Python's `$` also matches before a final newline, and JSON
    # Schema's does not. The schema's patterns are anchored at both ends, so nothing else moves.
    if (isinstance(value, str) and "pattern" in schema
            and not re.fullmatch(schema["pattern"], value)):
        errors.append(f"{path}: {value[:60]!r} does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items, has {len(value)}")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema["items"], root, f"{path}[{index}]"))
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field {name!r}")
        properties = schema.get("properties", {})
        for name, item in value.items():
            if name in properties:
                errors.extend(schema_errors(item, properties[name], root, f"{path}.{name}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field {name!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors.extend(schema_errors(item, schema["additionalProperties"], root,
                                            f"{path}.{name}"))
    return errors


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    with path.open() as handle:
        schema = json.load(handle)
    check_schema_keywords(schema)
    return dict(schema)


def _load_analyser() -> Any:
    spec = importlib.util.spec_from_file_location("analyse_served_accept", ANALYSER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {ANALYSER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_analyser = _load_analyser()
#: (budget, block) from a report's settings, or ValueError.
_settings: Callable[[dict[str, Any]], tuple[int, int]] = _analyser._settings
#: One record against the loop that produced it, or ValueError.
_check_row: Callable[[dict[str, Any], int, int], None] = _analyser._check_row


def report_errors(report: Any, schema: dict[str, Any] | None = None) -> list[str]:
    """Every problem with one report; empty when the analyser would admit it as an arm."""
    schema = load_schema() if schema is None else schema
    errors = schema_errors(report, schema, schema)
    if errors:
        return errors       # the rules below read fields the schema guarantees
    if not report["complete"]:
        errors.append("$.complete: false; the analyser refuses an incomplete arm")
    requests = report["requests"]
    if not requests:
        errors.append("$.requests: empty; the analyser refuses an empty prompt set")
    seen: dict[str, int] = {}
    for index, row in enumerate(requests):
        first = seen.setdefault(row["prompt_sha256"], index)
        if first != index:
            errors.append(f"$.requests[{index}].prompt_sha256: duplicates requests[{first}]")
    try:
        budget, block = _settings(report)
    except ValueError as exc:
        return [*errors, f"$.settings: {exc}"]      # without them no record can be checked
    for index, row in enumerate(requests):
        try:
            _check_row(row, budget, block)
        except ValueError as exc:
            errors.append(f"$.requests[{index}]: {exc} (budget {budget}, block {block})")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = ap.parse_args(argv)
    schema = load_schema(args.schema)
    failed = 0
    for path in args.reports:
        try:
            with path.open() as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors = [f"cannot read: {exc}"]
        else:
            errors = report_errors(report, schema)
        if errors:
            failed += 1
            print(f"FAIL {path}")
            for line in errors[:20]:
                print(f"     {line}")
            if len(errors) > 20:
                print(f"     ... and {len(errors) - 20} more")
        else:
            print(f"ok   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
