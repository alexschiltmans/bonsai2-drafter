#!/usr/bin/env python3
"""Check a sanitized evidence bundle. Standard library only.

    python3 evidence/tools/verify_sanitized.py scan evidence
    python3 evidence/tools/verify_sanitized.py pair ORIGINAL.json SANITIZED.json
    python3 evidence/tools/verify_sanitized.py pairs PAIRS.json

`scan` needs only the bundle. For every directory holding a SHA256SUMS it checks that each
listed file hashes as listed and that no file is unlisted, then searches every file for the
patterns sanitization removes (dates, epoch seconds, home-directory paths). Exit 0 means clean.

`pair` and `pairs` need the unedited originals, which are kept privately. They walk an
original and its sanitized copy in parallel and prove the copy changed nothing it should not:

- every number, boolean and null in the copy is identical, in type and value, to the leaf at
  the same place in the original, and every list has the same length;
- a key may be missing from the copy only if its name is in REMOVABLE (timestamps, paths,
  private provenance, private machinery); no key may be added;
- a string may differ only if every number written inside the new string also appears in the
  original string or in one of the public names in PUBLIC_NAMES. So a path can become a public
  model name, but no figure inside a string can be changed or invented.

`PAIRS.json` is a list of [original, sanitized] paths.
"""
import argparse
import hashlib
import json
import os
import re
import sys

#: Key names that sanitization may delete, wherever they occur. Each is a timestamp, a path,
#: private provenance or machinery that measures nothing; nothing here holds a measurement.
REMOVABLE = {
    # timestamps
    "when", "started", "finished", "updated", "created", "time", "timestamp", "started_at",
    "finished_at", "date", "uptime", "generated", "mtime", "at",
    # private provenance: repository revision and worktree state, host process table
    "git", "top_rss", "revision", "dirty", "worktree", "source_commit",
    # paths and private machinery
    "source", "calibration_keys", "log", "logs", "path", "paths", "cwd", "command", "argv",
    "script", "pid", "target_path", "drafter_path", "resolved_model_ref", "report", "reports",
    "server_log", "supervisor", "runtime_dirty", "files", "record", "records", "arms_files",
}

#: Public names a path may be replaced by. Numbers written in these may appear in a rewritten
#: string even if the original path spelled them differently.
PUBLIC_NAMES = (
    "Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5",
    "Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit",
    "z-lab/Qwen3.8-27B-DFlash2",
    "z-lab/Qwen3.8-27B-DFlash2-GGUF",
    "naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2",
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    "prism-ml/Ternary-Bonsai-2-27B-gguf",
    "decent-jawfish/bonsai-2-27b-mtp",
    "general.jsonl", "code.jsonl",
)

NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: What sanitization removes. The single-character classes such as `[-]` match exactly what
#: the bare character would; they keep this file from matching its own scan.
FORBIDDEN = re.compile(
    r"20\d\d-\d\d-\d\d|\b1[67]\d{8}\b|/User[s]/|/hom[e]/[a-z]|\$HO[M]E|~[/]|experiment[-]state|"
    r"\.cla[u]de|docs/[a-z0-9_-]+\.md|llm[-_]ctl", re.IGNORECASE)
ALLOWED_NUMBERS = {n for name in PUBLIC_NAMES for n in NUMBER.findall(name)}


class Mismatch(Exception):
    pass


def _numbers(text):
    return NUMBER.findall(text)


def compare(original, sanitized, where="$"):
    """Raise Mismatch at the first place the copy departs from the rules above."""
    if isinstance(sanitized, dict):
        if not isinstance(original, dict):
            raise Mismatch(f"{where}: object where the original has {type(original).__name__}")
        added = set(sanitized) - set(original)
        if added:
            raise Mismatch(f"{where}: keys added: {sorted(added)}")
        removed = set(original) - set(sanitized)
        illegal = sorted(k for k in removed if k not in REMOVABLE)
        if illegal:
            raise Mismatch(f"{where}: keys removed that are not removable: {illegal}")
        return len(removed) + sum(compare(original[key], sanitized[key], f"{where}.{key}")
                                  for key in sanitized)
    if isinstance(sanitized, list):
        if not isinstance(original, list) or len(original) != len(sanitized):
            raise Mismatch(f"{where}: list length or type differs")
        return sum(compare(a, b, f"{where}[{i}]")
                   for i, (a, b) in enumerate(zip(original, sanitized, strict=True)))
    if isinstance(sanitized, str):
        if not isinstance(original, str):
            raise Mismatch(f"{where}: string where the original has {type(original).__name__}")
        if sanitized != original:
            before = _numbers(original)
            for n in _numbers(sanitized):
                if n in before:
                    before.remove(n)
                elif n not in ALLOWED_NUMBERS:
                    raise Mismatch(f"{where}: rewritten string carries a number ({n}) "
                                   f"the original string did not")
        return 0
    # numbers, booleans, null: identical type and value
    if type(sanitized) is not type(original) or sanitized != original:
        raise Mismatch(f"{where}: {original!r} became {sanitized!r}")
    return 0


def _leaves(value):
    if isinstance(value, dict):
        return sum(_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(_leaves(v) for v in value)
    return 1


def check_pair(original_path, sanitized_path):
    with open(original_path) as f:
        original = json.load(f)
    with open(sanitized_path) as f:
        sanitized = json.load(f)
    removed = compare(original, sanitized)
    return _leaves(sanitized), removed


def _sums(directory):
    listed = {}
    with open(os.path.join(directory, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.rstrip("\n").split(None, 1)
            listed[name.lstrip("*")] = digest
    return listed


def scan(root):
    problems = []
    covered = set()
    for directory, _, files in os.walk(root):
        if "SHA256SUMS" not in files:
            continue
        for name, digest in _sums(directory).items():
            path = os.path.join(directory, name)
            covered.add(os.path.normpath(path))
            if not os.path.exists(path):
                problems.append(f"{path}: listed in SHA256SUMS but missing")
                continue
            with open(path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != digest:
                    problems.append(f"{path}: sha256 differs from SHA256SUMS")
    for directory, _, files in os.walk(root):
        for name in files:
            path = os.path.normpath(os.path.join(directory, name))
            if name.endswith(".pyc"):
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for m in FORBIDDEN.finditer(text):
                problems.append(f"{path}: forbidden pattern {m.group(0)!r}")
            in_group = os.path.exists(os.path.join(directory, "SHA256SUMS"))
            if in_group and name != "SHA256SUMS" and path not in covered:
                problems.append(f"{path}: not listed in its SHA256SUMS")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root")
    p = sub.add_parser("pair")
    p.add_argument("original")
    p.add_argument("sanitized")
    ps = sub.add_parser("pairs")
    ps.add_argument("spec")
    args = ap.parse_args()
    if args.cmd == "scan":
        problems = scan(args.root)
        for line in problems:
            print(line)
        print(f"scan: {'clean' if not problems else f'{len(problems)} problem(s)'}")
        return 1 if problems else 0
    if args.cmd == "pair":
        pairs = [(args.original, args.sanitized)]
    else:
        with open(args.spec) as f:
            pairs = json.load(f)
    failed = 0
    for original, sanitized in pairs:
        try:
            leaves, removed = check_pair(original, sanitized)
            print(f"ok    {sanitized}  ({leaves} leaves kept, {removed} fields removed)")
        except (Mismatch, OSError, ValueError) as exc:
            failed += 1
            print(f"FAIL  {sanitized}: {exc}")
    print(f"pairs: {len(pairs) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
