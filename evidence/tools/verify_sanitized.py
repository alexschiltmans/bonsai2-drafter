#!/usr/bin/env python3
"""Check a sanitized evidence bundle. Standard library only.

    python3 evidence/tools/verify_sanitized.py scan evidence [--extra-patterns FILE]
    python3 evidence/tools/verify_sanitized.py pair ORIGINAL SANITIZED
    python3 evidence/tools/verify_sanitized.py pairs PAIRS.json

`scan` needs only the bundle. For every directory holding a SHA256SUMS it checks that each
listed file hashes as listed and that no file is unlisted, then searches every file for what
sanitization removes: dates, epoch seconds, home-directory paths and e-mail addresses. In a
JSON file it reads each value in place: a number between 1e9 and 2e9 counts as epoch seconds
unless its key names a byte count or a byte range, and a date or ten-digit number inside the model's own words
(`answer`, `reasoning`) is content the model wrote, not a record of when it ran. Exit 0 means
clean.

`--extra-patterns FILE` adds regular expressions, one per line (blank lines and lines starting
with `#` are skipped), matched case-insensitively everywhere. The private names sanitization
also removes are not written into this file, since publishing them here would publish them;
their keeper passes them this way.

`pair` and `pairs` need the unedited originals, which are kept privately. They walk an
original and its sanitized copy in parallel and prove the copy changed nothing it should not:

- every number, boolean and null in the copy is identical, in type and value, to the leaf at
  the same place in the original, and every list has the same length;
- a key may be missing from the copy only if its name is in REMOVABLE and the original value
  is of the kind listed there (timestamps, paths, private provenance, private machinery); no
  key may be added;
- a string may differ only if every number written inside the new string, outside the public
  names in PUBLIC_NAMES, also appears (as often) in the original string. So a path can become a
  public model name, but no figure inside a string can be changed or invented.

A file that is not JSON (a text log) is compared as its list of lines under the same rules: no
line added or removed, and no number in a rewritten line that its original line did not have.

`PAIRS.json` is a list of [original, sanitized] paths.
"""
import argparse
import hashlib
import json
import os
import re
import sys


def _text(value):
    return isinstance(value, str)


def _epoch(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 1e9 < value < 2e9


def _texts(value):
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _revision(value):
    return isinstance(value, dict) and all(isinstance(v, (str, bool)) for v in value.values())


#: The fields sanitization may delete, by key name, each with the only kind of value it may
#: hold. A key of the same name holding anything else (a measurement block, say) may not go.
REMOVABLE = {
    # timestamps: an ISO time, an epoch second, and the clock time in `uptime`'s output line
    "when": _text,
    "at": _epoch,
    "uptime": _text,
    # private provenance: the private repository's revision and worktree flag
    "git": _revision,
    # machinery: the host process list, a server log path, calibration cache keys that name a
    # local directory, local paths to a target and a drafter
    "top_rss": _text,
    "source": _text,
    "calibration_keys": _texts,
    "target_path": _text,
    "drafter_path": _text,
    "resolved_model_ref": _text,
}

#: Public names a path may be replaced by. The digits inside one of these names may appear in a
#: rewritten string even if the original path spelled them differently.
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
    # this repository, whose name the patches print in their log lines
    "bonsai2-drafter",
)

NUMBER = re.compile(r"\d+(?:\.\d+)?")
#: What sanitization removes, as the scan finds it. The single-character classes such as `[s]`
#: match exactly what the bare character would; they keep this file from matching its own scan.
TIMESTAMP = re.compile(r"20\d\d-\d\d-\d\d|\b1[67]\d{8}\b")
LOCAL = re.compile(r"/User[s]/|/hom[e]/[a-z]|\$HO[M]E|~[/]|"
                   r"[A-Za-z0-9._%+-]+[@][A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.IGNORECASE)
#: Keys holding the model's own words. TIMESTAMP is not applied to them.
MODEL_TEXT = {"answer", "reasoning"}


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
        illegal = sorted(k for k in removed
                         if k not in REMOVABLE or not REMOVABLE[k](original[k]))
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
            # A public name written into the copy may bring its own digits; nothing else may.
            # Longest names first, so a name that contains another is not counted twice.
            rest = sanitized
            for name in sorted(PUBLIC_NAMES, key=len, reverse=True):
                rest = rest.replace(name, " ")
            before = _numbers(original)
            for n in _numbers(rest):
                if n not in before:
                    raise Mismatch(f"{where}: rewritten string carries a number ({n}) "
                                   f"the original string did not")
                before.remove(n)
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


def _load(path):
    """A JSON file as its value; any other file as its list of lines."""
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return f.read().split("\n")


def check_pair(original_path, sanitized_path):
    original, sanitized = _load(original_path), _load(sanitized_path)
    removed = compare(original, sanitized)
    return _leaves(sanitized), removed


def _sums(directory):
    listed = {}
    with open(os.path.join(directory, "SHA256SUMS")) as f:
        for line in f:
            digest, name = line.rstrip("\n").split(None, 1)
            listed[name.lstrip("*")] = digest
    return listed


def _byte_count(key):
    # `range` is a tensor's [start, end) byte offsets inside a model file
    return key is not None and ("bytes" in key or key.startswith("allocator") or key == "range")


def _json_hits(value, extra, key=None):
    """Forbidden matches in a parsed JSON value, read leaf by leaf."""
    if isinstance(value, dict):
        hits = []
        for k, v in value.items():
            hits += [m.group(0) for rx in (LOCAL, *extra) for m in rx.finditer(k)]
            hits += _json_hits(v, extra, k)
        return hits
    if isinstance(value, list):
        return [hit for v in value for hit in _json_hits(v, extra, key)]
    if isinstance(value, str):
        patterns = (LOCAL, *extra) if key in MODEL_TEXT else (TIMESTAMP, LOCAL, *extra)
        return [m.group(0) for rx in patterns for m in rx.finditer(value)]
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and 1e9 < value < 2e9 and not _byte_count(key)):
        return [f"{key}: {value!r} (epoch seconds?)"]
    return []


def _hits(path, text, extra):
    if path.endswith(".json"):
        try:
            return _json_hits(json.loads(text), extra)
        except ValueError:
            pass
    return [m.group(0) for rx in (TIMESTAMP, LOCAL, *extra) for m in rx.finditer(text)]


def load_patterns(path):
    """One regular expression per line; blank lines and `#` comments are skipped."""
    with open(path) as f:
        lines = [line.strip() for line in f]
    return [re.compile(line, re.IGNORECASE) for line in lines if line and not line.startswith("#")]


def scan(root, extra=()):
    problems = []
    covered = set()
    groups = []
    for directory, _, files in os.walk(root):
        if "SHA256SUMS" not in files:
            continue
        groups.append(os.path.normpath(directory))
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
            for hit in _hits(path, text, extra):
                problems.append(f"{path}: forbidden pattern {hit!r}")
            # Anything under a directory that has a SHA256SUMS, at any depth, must be listed.
            in_group = any(path.startswith(group + os.sep) for group in groups)
            if in_group and name != "SHA256SUMS" and path not in covered:
                problems.append(f"{path}: not listed in its SHA256SUMS")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("root")
    s.add_argument("--extra-patterns", metavar="FILE",
                   help="more regular expressions to refuse, one per line")
    p = sub.add_parser("pair")
    p.add_argument("original")
    p.add_argument("sanitized")
    ps = sub.add_parser("pairs")
    ps.add_argument("spec")
    args = ap.parse_args()
    if args.cmd == "scan":
        extra = load_patterns(args.extra_patterns) if args.extra_patterns else ()
        problems = scan(args.root, extra)
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
    for entry in pairs:
        original, sanitized = entry[0], entry[1]
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
