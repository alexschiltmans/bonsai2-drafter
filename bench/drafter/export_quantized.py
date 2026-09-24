#!/usr/bin/env python3
"""Write the 4-bit form of a bfloat16 DFlash 2 drafter, once, instead of at every load.

    "$HOME/.venv-dspark/bin/python" bench/drafter/export_quantized.py \
      --source "$SOURCE" --output "$ARTIFACT"

The ft5 drafter is served at 4 bits but ships as 3.85 GB of bfloat16, which
`load_dflash` quantizes on every start. The weights that reach the GPU are the same either
way, so this is a distribution-size change and nothing else: not a fine-tune, not a speed
claim, and not a checkpoint that can be served without this repository's patches. See
`patches/dflash_prequantized.py` for the format.

How it stays honest
-------------------
The quantization is not reimplemented here. The pinned `mlx_dspark.load.load_dflash` is
called at 4 bits and group 64 exactly as the server calls it, and what this script saves is
the *result*: the evaluated parameters of the model that call returned. Everything the
artifact records about its own format — the bit width, the group size, the mode, the set of
quantized modules, the `DFlashConfig` — is read back off that model rather than asserted, so
a future mlx-dspark that selects different modules or a different mode produces an artifact
that says so, and `patches/dflash_prequantized.py` refuses to load it under the old rule.

This script installs no patches: the source is an ordinary bfloat16 checkpoint, so the
original loader is the one that should answer it. When something else has already installed
them — the equivalence test does, because it also loads the Bonsai target —
:func:`load_source_quantized` asks the wrapper for the original it closed over instead of
trusting it to delegate. The wrapper does delegate, but "the pinned loader's own
quantization path" should be checkable rather than a claim about a wrapper's control flow.

Publishing
----------
Everything is written into a staging directory created for this one run beside the
destination, and validated there — metadata, dtypes, hashes, and a full reload through the
patched loader whose parameters must match what was written, hash for hash. Only then is the
directory renamed into place, by a rename that refuses a taken name rather than replacing
it (:func:`publish`). A failure removes the staging directory, and nothing else, and
leaves the destination absent: a half-written release directory that looks complete is the
one failure mode worth engineering against, and destroying somebody else's directory to make
room is the other.

Importing this module runs nothing.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from patches import dflash_prequantized as pq

#: Recorded so an artifact says which stack produced it. Read off the interpreter that runs
#: the conversion rather than off the lockfile: the lock says what was pinned, this says what
#: actually ran, and for a released artifact the second is the one that matters.
RECORDED_PACKAGES = ("mlx", "mlx-lm", "mlx-dspark", "numpy")


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for name in RECORDED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


def _git_revision() -> str:
    """The revision of this repository, for lineage. Best effort: an export from a snapshot
    without `.git` records "unknown" rather than failing the conversion."""
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    try:
        proc = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20, check=False)
    except OSError:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_hashes(path: str) -> dict[str, str]:
    """SHA256 per file, keyed by name within the directory.

    Names, not paths: a manifest that travels with the weights should not carry the home
    directory of the machine that made them (a redaction rule, applied
    before the fact rather than after).
    """
    return {name: sha256_file(os.path.join(path, name))
            for name in sorted(os.listdir(path))
            if os.path.isfile(os.path.join(path, name))}


def load_source_quantized(source: str) -> tuple[Any, Any]:
    """(drafter, DFlashConfig) from the pinned loader's own 4-bit path, patched or not.

    Also the arm the equivalence check in `bench/tests/test_dflash_prequantized.py` compares
    the artifact against, which is why it is a function here rather than a line inside
    :func:`export_prequantized`: both must call the same thing, with the same arguments the
    server uses (`mlx_dspark/server.py` `_load_models`).

    If `patches.install_all` has already run — which it has in any process that also loads
    the Bonsai target — the wrapper is asked for the original it closed over rather than
    being trusted to delegate. It does delegate for an unmarked checkpoint, but "the pinned
    loader's own path" should not be a claim about a wrapper's control flow.
    """
    from mlx_dspark import load as dload

    original = getattr(dload.load_dflash, "_bonsai2_original", dload.load_dflash)
    return original(source, quantize=True, bits=pq.SUPPORTED_BITS,
                    group_size=pq.SUPPORTED_GROUP_SIZE)


def describe_quantization(drafter: Any) -> dict[str, Any]:
    """What the pinned loader actually did, read back off the model it returned.

    The selection *rule* cannot be re-run here — the `nn.Linear` modules it chose no longer
    exist, having been replaced — so it is checked the other way round: nothing quantized may
    carry an excluded fragment, and nothing still `nn.Linear` may lack one. Between them
    those two say the result is exactly what :data:`pq.SELECTION_RULE` describes.
    """
    from mlx import nn

    quantized = sorted(p for p, m in drafter.named_modules() if isinstance(m, nn.QuantizedLinear))
    if not quantized:
        raise pq.PrequantizedFormatError(
            "the pinned loader quantized nothing; refusing to export a bfloat16 checkpoint "
            "under a prequantized marker")
    formats = {(m.bits, m.group_size, m.mode) for _, m in drafter.named_modules()
               if isinstance(m, nn.QuantizedLinear)}
    if len(formats) != 1:
        raise pq.PrequantizedFormatError(
            f"the loader produced more than one quantization format: {sorted(formats)}")
    bits, group_size, mode = formats.pop()
    if bits != pq.SUPPORTED_BITS or group_size != pq.SUPPORTED_GROUP_SIZE:
        raise pq.PrequantizedFormatError(
            f"the loader produced {bits} bits / group {group_size}, not "
            f"{pq.SUPPORTED_BITS} / {pq.SUPPORTED_GROUP_SIZE}")
    if mode not in pq.SUPPORTED_MODES:
        raise pq.PrequantizedFormatError(
            f"the loader produced quantization mode {mode!r}, which "
            f"patches/dflash_prequantized.py cannot reconstruct")
    wrongly_included = [p for p in quantized
                        if any(frag in p for frag in pq.EXCLUDED_PATH_FRAGMENTS)]
    wrongly_excluded = [p for p, m in drafter.named_modules()
                        if isinstance(m, nn.Linear)
                        and not any(frag in p for frag in pq.EXCLUDED_PATH_FRAGMENTS)]
    if wrongly_included or wrongly_excluded:
        raise pq.PrequantizedFormatError(
            f"the loader's selection is not {pq.SELECTION_RULE}."
            + (f" Quantized despite an excluded path: {wrongly_included[:5]}."
               if wrongly_included else "")
            + (f" Left unquantized with no excluded path: {wrongly_excluded[:5]}."
               if wrongly_excluded else ""))
    return {"bits": bits, "group_size": group_size, "mode": mode,
            "selection_rule": pq.SELECTION_RULE, "quantized_modules": quantized}


def artifact_tensors(drafter: Any, quantized: list[str]) -> dict[str, Any]:
    """The tensors to save: every evaluated parameter, checked against the format's dtypes.

    A DFlash drafter binds the target's embedding and head at generation time and ships
    neither, so neither is in `parameters()` on a model the loader has not bound. It is
    asserted rather than assumed because `bench/drafter/dflash_ft.py:export` filters them
    explicitly, and the difference between "absent" and "filtered" is a 2.5 GB tensor.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(drafter.parameters()))
    bound = sorted(k for k in weights if k.startswith(("embed_tokens.", "lm_head.")))
    if bound:
        raise pq.PrequantizedFormatError(
            f"the model carries target-bound tensors {bound[:5]}; it was bound to a target "
            f"before export, and those weights are the target's, not the drafter's")
    packed = {f"{m}.weight" for m in quantized}
    wrong = [f"{k}: {v.dtype}" for k, v in sorted(weights.items())
             if str(v.dtype) != f"mlx.core.{pq.PACKED_WEIGHT_DTYPE if k in packed else pq.UNPACKED_DTYPE}"]
    if wrong:
        raise pq.PrequantizedFormatError(
            f"{len(wrong)} tensors are not the dtype this format saves: {wrong[:5]}")
    mx.eval(list(weights.values()))
    return weights


def tensor_table(weights: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(array.shape), "dtype": str(array.dtype).split(".")[-1],
             "bytes": array.nbytes} for name, array in sorted(weights.items())]


def parameter_digest(weights: dict[str, Any]) -> dict[str, list[Any]]:
    """dtype, shape and a content hash per tensor: the thing compared across a reload."""
    return {name: [str(array.dtype), list(array.shape),
                   hashlib.sha256(bytes(memoryview(array))).hexdigest()]
            for name, array in weights.items()}


def resolve_paths(source: str, output: str) -> tuple[str, str]:
    """(source, output) as real paths, or raise if this pair is not one we may export.

    Every rule here exists because a code review found a way to destroy the input. The
    first version derived a *predictable* staging directory, `.export-<output basename>`
    beside the output, and removed it if it already existed. Pointing `--source` at
    `run/.export-artifact` and `--output` at `run/artifact` made the staging path equal the
    source, so the exporter deleted its own checkpoint before reading it, and the failure
    handler deleted what was left. :func:`export_prequantized` now stages in a directory
    `tempfile.mkdtemp` created for this invocation and removes nothing else, which makes that
    collision unrepresentable rather than merely checked for; these rules are the rest of it.

    `realpath`, not `abspath`: `abspath` normalizes `..` textually and cannot see a symlinked
    parent, so an output whose directory is a link into the source would have passed the
    containment rule while landing inside the checkpoint. And containment is then decided by
    :func:`_within`, by identity, because `realpath` resolves symlinks and nothing else.
    """
    if not os.path.isdir(source):
        raise pq.PrequantizedFormatError(f"{source}: not a directory")
    real_source = os.path.realpath(source)
    parent = os.path.dirname(os.path.abspath(output))
    if not os.path.isdir(parent):
        raise pq.PrequantizedFormatError(
            f"{parent}: the output's parent directory does not exist. It is not created here, "
            f"so a mistyped run directory is a refusal rather than a new tree in an "
            f"unexpected place.")
    real_output = os.path.join(os.path.realpath(parent), os.path.basename(os.path.abspath(output)))
    # lexists, not exists: a dangling symlink is something already claiming this name, and
    # `exists` follows the link and reports False.
    if os.path.lexists(real_output):
        raise pq.PrequantizedFormatError(
            f"{real_output}: already exists. This tool never replaces a destination; remove "
            f"it deliberately or choose another path.")
    # The parent, because the output does not exist yet: it lands inside the source exactly
    # when its parent is the source or lies beneath it. The source under another name was
    # refused just above, since it exists.
    if _within(os.path.dirname(real_output), real_source):
        raise pq.PrequantizedFormatError(
            f"{real_output}: inside the source directory. The exporter must not write into "
            f"the checkpoint it is reading.")
    return real_source, real_output


def _within(directory: str, ancestor: str) -> bool:
    """Whether `directory` is `ancestor` or lies beneath it, judged by identity, not spelling.

    Comparing strings, even `realpath`ed ones, misses every alias that is not a symlink. On
    the default macOS volume `Run/ckpt` and `run/CKPT` are one directory and `realpath`
    returns whichever spelling it was given; a firmlink or a bind mount reaches the same
    directory by an unrelated path. A device and inode pair has no spelling, so each
    directory on the physical path up from `directory` is compared with `ancestor` that way.
    """
    target = os.stat(ancestor)
    current = directory
    while True:
        if os.path.samestat(os.stat(current), target):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


#: The kernel's own "rename, but never over an existing name": `renamex_np(2)` with
#: RENAME_EXCL on macOS, `renameat2(2)` with RENAME_NOREPLACE on Linux. `os` exposes neither.
_RENAME_EXCL = 0x00000004
_RENAME_NOREPLACE = 0x1
_AT_FDCWD = -100


def _kernel_rename_noreplace(src: str, dst: str) -> bool:
    """Rename `src` to `dst` atomically unless `dst` exists. False if the system cannot.

    True once renamed. Raises the kernel's `OSError` when it has the primitive and refused,
    EEXIST for a taken name among them. False, having touched nothing, when the platform
    lacks the call or the filesystem lacks the flag, so the caller can fall back.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        call = getattr(libc, "renamex_np", None)
        argtypes: list[Any] = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args: tuple[Any, ...] = (os.fsencode(src), os.fsencode(dst), _RENAME_EXCL)
        unsupported = {errno.ENOTSUP, errno.EOPNOTSUPP}
    elif sys.platform.startswith("linux"):
        call = getattr(libc, "renameat2", None)
        argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (_AT_FDCWD, os.fsencode(src), _AT_FDCWD, os.fsencode(dst), _RENAME_NOREPLACE)
        unsupported = {errno.EINVAL, errno.ENOSYS}
    else:
        return False
    if call is None:
        return False
    call.argtypes, call.restype = argtypes, ctypes.c_int
    if call(*args) == 0:
        return True
    err = ctypes.get_errno()
    if err in unsupported:
        return False
    raise OSError(err, os.strerror(err), dst)


def _claimed_rename(src: str, dst: str) -> None:
    """The fallback: claim `dst` with `mkdir(2)`, exclusive everywhere, then rename over it.

    A plain rename replaces an empty directory without a word, which is the hazard, so the
    claim is what makes the only directory it can replace this call's own. Anything put
    inside the claim meanwhile turns the rename into ENOTEMPTY rather than a loss, and the
    claim is withdrawn only while it is still empty.
    """
    os.mkdir(dst)
    try:
        os.rename(src, dst)
    except OSError:
        with contextlib.suppress(OSError):
            os.rmdir(dst)
        raise


def publish(staging: str, output: str) -> None:
    """Move the validated staging directory to `output`, refusing to replace anything there.

    :func:`resolve_paths` found the name free, but a conversion takes minutes and the name
    only matters at the moment of the rename. `os.rename` would still silently replace an
    *empty* directory created in between, and a check immediately before it only narrows
    that window, which is what the first fix after that review did. The kernel's
    exclusive rename closes it; the mkdir claim stands in where the kernel has none. Either
    way a taken name is contention and is said out loud.
    """
    try:
        if not _kernel_rename_noreplace(staging, output):
            _claimed_rename(staging, output)
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            raise pq.PrequantizedFormatError(
                f"{output}: appeared while this export was running. Refusing to replace it; "
                f"the staged artifact is discarded and nothing was published.") from exc
        raise pq.PrequantizedFormatError(
            f"{output}: could not be published ({exc}). Nothing was replaced.") from exc


def export_prequantized(source: str, output: str) -> dict[str, Any]:
    """Convert `source` into a published prequantized artifact at `output`. Returns the manifest.

    The source is only ever read. The destination is produced by renaming a staging directory
    this call created with `tempfile.mkdtemp`, and that directory is the only thing any
    failure path removes — see :func:`resolve_paths` for what went wrong when the staging
    path was predictable and shared.

    Validation before publication is unconditional and has no off switch. The reload is not
    a redundant comparison: config reconstruction, agreement with the recorded module set and
    strict tensor loading happen only inside `pq.load_prequantized`, so without it the
    exporter would publish a directory that nothing had ever loaded. A `--no-verify-reload`
    flag existed until the same review pointed out that it published a normal-looking,
    unverified artifact; there is no legitimate use for that, and a test that needs the
    reload to fail mocks it.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten

    source, output = resolve_paths(source, output)
    if not pq.shard_paths(source):
        raise pq.PrequantizedFormatError(f"{source}: no *.safetensors")
    source_config = pq.read_config(source)
    if pq.marker(source_config) is not None:
        raise pq.PrequantizedFormatError(
            f"{source}: already carries a {pq.FORMAT_KEY} block. The source of an export is "
            f"the bfloat16 checkpoint, not another artifact.")
    source_header = pq.checkpoint_header(source)
    if pq.carries_packed_tensors(source_header):
        raise pq.PrequantizedFormatError(
            f"{source}: carries packed tensors. This exporter quantizes a bfloat16 "
            f"checkpoint; it does not requantize.")

    drafter, cfg = load_source_quantized(source)
    quantization = describe_quantization(drafter)
    weights = artifact_tensors(drafter, quantization["quantized_modules"])
    written_digest = parameter_digest(weights)

    metadata = {
        "format": pq.FORMAT_NAME,
        "format_version": pq.SUPPORTED_VERSIONS[0],
        "bits": quantization["bits"],
        "group_size": quantization["group_size"],
        "mode": quantization["mode"],
        "selection_rule": quantization["selection_rule"],
        "quantized_modules": quantization["quantized_modules"],
        "dflash_config": pq.config_fields(cfg),
        "note": "A repository-specific format read by patches/dflash_prequantized.py. No "
                "other runtime interprets this block, and no other runtime can load these "
                "weights.",
    }
    pq.validate_metadata(metadata, where=f"{output} (about to be written)")
    artifact_config = dict(source_config)
    artifact_config[pq.FORMAT_KEY] = metadata

    source_hashes = directory_hashes(source)
    manifest: dict[str, Any] = {
        "format": pq.FORMAT_NAME,
        "format_version": pq.SUPPORTED_VERSIONS[0],
        "tool": "bench/drafter/export_quantized.py",
        "bonsai2_revision": _git_revision(),
        "environment": _versions(),
        "source": {"name": os.path.basename(source.rstrip(os.sep)),
                   "sha256": source_hashes,
                   "tensors": len(source_header)},
        "quantization": quantization,
        "tensors": tensor_table(weights),
    }

    # Exclusive and ours: mkdtemp creates a directory that did not exist a moment ago, so it
    # can never be the source, another export's staging area, or anything else worth keeping.
    # Nothing below removes a directory this call did not create.
    staging = tempfile.mkdtemp(dir=os.path.dirname(output), prefix=".export-")
    # mkdtemp creates 0700, and the rename below would publish that mode: an artifact
    # meant for distribution that `cp -rp` or `tar` then hands on unreadable to anyone
    # else. Give it the mode an ordinary `mkdir` would have. The umask is read the only
    # way POSIX allows, by setting it; this process is single-threaded, so the brief swap
    # is not observable.
    mask = os.umask(0)
    os.umask(mask)
    os.chmod(staging, 0o777 & ~mask)
    try:
        mx.save_safetensors(os.path.join(staging, "model.safetensors"), weights,
                            metadata={"format": "mlx"})
        with open(os.path.join(staging, "config.json"), "w") as f:
            json.dump(artifact_config, f, indent=2)
            f.write("\n")
        artifact_bytes = sum(os.path.getsize(p) for p in pq.shard_paths(staging))
        source_bytes = sum(os.path.getsize(p) for p in pq.shard_paths(source))
        manifest["byte_counts"] = {
            "source_total": source_bytes, "artifact_total": artifact_bytes,
            "ratio": round(source_bytes / max(artifact_bytes, 1), 4),
            "note": "measured on the written files, not derived from a compression ratio"}
        # Written last, so its hashes cover every file it names. A manifest cannot hash
        # itself; `shasum -a 256 -c` over the release directory is what covers this one.
        manifest["artifact"] = {"sha256": directory_hashes(staging)}
        with open(os.path.join(staging, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        # Validation, before the destination exists. The header and dtype checks are the
        # format's own rules read back off the file; the reload is the one that matters,
        # because it is the patched loader rebuilding the module structure from config.json
        # alone and having to arrive at the same tensors.
        staged_config, staged_meta = pq.inspect(staging)
        assert staged_meta is not None
        pq.check_requested(staged_meta, quantize=True, bits=pq.SUPPORTED_BITS,
                           group_size=pq.SUPPORTED_GROUP_SIZE, where=staging)
        pq.check_tensor_dtypes(pq.checkpoint_header(staging), staged_meta["quantized_modules"],
                               where=staging)
        if artifact_bytes >= source_bytes:
            raise pq.PrequantizedFormatError(
                f"{staging}: the artifact is {artifact_bytes} bytes against the source's "
                f"{source_bytes}; a prequantized export that does not shrink is a failure")
        del weights, drafter
        mx.clear_cache()
        reloaded, _ = pq.load_prequantized(staging, staged_config, staged_meta)
        reloaded_digest = parameter_digest(dict(tree_flatten(reloaded.parameters())))
        if reloaded_digest != written_digest:
            differing = sorted(k for k in set(reloaded_digest) | set(written_digest)
                               if reloaded_digest.get(k) != written_digest.get(k))
            raise pq.PrequantizedFormatError(
                f"{staging}: reloading what was just written does not reproduce it. "
                f"{len(differing)} tensors differ: {differing[:5]}")
        del reloaded
        mx.clear_cache()
        # The destination was free when this call started. If something took the name while
        # the conversion ran, that is contention, refused rather than resolved by overwriting.
        publish(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="an existing local bfloat16 DFlash 2 drafter")
    ap.add_argument("--output", required=True, help="the artifact directory to create")
    args = ap.parse_args(argv)
    try:
        manifest = export_prequantized(args.source, args.output)
    except pq.PrequantizedFormatError as exc:
        print(f"refusing to export: {exc}", file=sys.stderr)
        return 1
    counts = manifest["byte_counts"]
    print(json.dumps({k: manifest[k] for k in ("format", "format_version", "byte_counts")},
                     indent=2))
    print(f"wrote {args.output}: {len(manifest['tensors'])} tensors, "
          f"{len(manifest['quantization']['quantized_modules'])} quantized modules, "
          f"{counts['artifact_total']} bytes against the source's {counts['source_total']} "
          f"({counts['ratio']}x smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
