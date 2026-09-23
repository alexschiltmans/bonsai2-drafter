#!/usr/bin/env python3
"""The prequantized DFlash 2 artifact: its format, its loader, and what it refuses.

    python3 bench/tests/test_dflash_prequantized.py                    (no mlx, no GPU)
    ~/.venv-dspark/bin/python bench/tests/test_dflash_prequantized.py --gpu
    ~/.venv-dspark/bin/python bench/tests/test_dflash_prequantized.py \
        --source <bf16 ft5 dir> --artifact <exported dir>

Three tiers, and the first one is the reason this file is laid out the way it is.

1. **The format, with no mlx at all.** The safetensors header reader, the metadata
   validator, the config-shape checks and the load-option checks run under system python3,
   so `bench/check.sh`'s no-GPU tier covers every rule that decides whether a file is this
   format. These are the checks that stand between a mislabelled artifact and a drafter
   that loads clean and drafts nonsense, so they are the ones that should be cheapest to
   run. Every rule gets a negative case; a validator with only happy-path tests is a
   validator nobody has tested.
2. **`--gpu`: a synthetic round trip.** A two-layer DFlash 2 drafter, ~11 MB, exported and
   reloaded for real through `bench/drafter/export_quantized.py` and the patch. Tensor
   names, shapes, dtypes and exact values; module classes and their quantization
   parameters; the selector and convolution tensors left alone; missing, extra and
   malformed inputs; the exporter's refusals; source immutability; patch idempotence; and
   an ordinary bfloat16 checkpoint still loading the way it always did.
3. **`--source`/`--artifact`: the real pair.** Phase 3 of docs/prequantized-ft5-plan.md:
   the shipped ft5 checkpoint quantized at load time against the saved artifact loaded
   through the patch, materialized on both sides and compared exactly, in this process and
   again in a fresh one that cannot read the source directory at all.
"""

from __future__ import annotations

import copy
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from patches import dflash_prequantized as pq

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def refuses(name: str, fn: Any, *args: Any, **kwargs: Any) -> None:
    """The check every rule in this format needs: it says no, and it says why."""
    try:
        fn(*args, **kwargs)
    except pq.PrequantizedFormatError as exc:
        check(name, bool(str(exc).strip()), "refused with an empty message")
        return
    except Exception as exc:  # noqa: BLE001 - a different exception is a different answer
        check(name, False, f"raised {type(exc).__name__} rather than PrequantizedFormatError: {exc}")
        return
    check(name, False, "accepted")


# --------------------------------------------------------------------------- fixtures

def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int]]],
                      metadata: dict[str, str] | None = None) -> None:
    """A safetensors file with the given dtypes and shapes and zeroed data.

    Written by hand rather than with mlx so tier 1 has real files to read. `mx.save_safetensors`
    writes the same layout; tier 2 checks the reader against one of those.
    """
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        length = count * sizes[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    if metadata is not None:
        header["__metadata__"] = metadata
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\0" * offset)


def tiny_config() -> dict[str, Any]:
    """A DFlash 2 config in the ft5 layout, sized for a test rather than for a target.

    Every dimension a quantized `nn.Linear` reads along is a multiple of 64, as the real
    checkpoint's are, so `nn.quantize` has a whole number of groups to work with.
    """
    return {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "dflash_config": {
            "block_size": 4, "conv_group_size": 16, "conv_kernel_size": 2,
            "mask_token_id": 3, "selector_rank": 64, "selector_top_k": 4,
            "target_layer_ids": [0, 1],
        },
        "dtype": "bfloat16", "eos_token_id": 2, "head_dim": 32, "hidden_act": "silu",
        "hidden_size": 128, "intermediate_size": 256,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "max_position_embeddings": 4096, "model_type": "qwen3",
        "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
        "num_target_layers": 4, "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "sliding_window": 64, "tie_word_embeddings": False, "vocab_size": 256,
    }


TINY_MODULES = sorted(
    ["fc"]
    + [f"layers.{i}.self_attn.{p}_proj" for i in range(2) for p in ("q", "k", "v", "o")]
    + [f"layers.{i}.mlp.{p}_proj" for i in range(2) for p in ("gate", "up", "down")])


def good_meta(modules: list[str] | None = None,
              config: dict[str, Any] | None = None) -> dict[str, Any]:
    """A metadata block every rule accepts, for the negative cases to spoil one field at a
    time. Built from the module's own constants so a change to the format shows up here."""
    return {
        "format": pq.FORMAT_NAME,
        "format_version": pq.SUPPORTED_VERSIONS[0],
        "bits": pq.SUPPORTED_BITS,
        "group_size": pq.SUPPORTED_GROUP_SIZE,
        "mode": pq.SUPPORTED_MODES[0],
        "selection_rule": pq.SELECTION_RULE,
        "quantized_modules": list(modules if modules is not None else TINY_MODULES),
        "dflash_config": dict(config if config is not None else {"hidden_size": 128,
                                                                 "block_size": 4}),
    }


# --------------------------------------------------------------------------- 1. the file

print("== 1. the safetensors header")
with tempfile.TemporaryDirectory() as tmp:
    plain = os.path.join(tmp, "model.safetensors")
    write_safetensors(plain, {"norm.weight": ("BF16", [8]), "fc.weight": ("BF16", [4, 8])},
                      {"format": "mlx"})
    header = pq.safetensors_header(plain)
    check("names, dtypes and shapes come back", set(header) == {"norm.weight", "fc.weight"}
          and header["fc.weight"]["shape"] == [4, 8] and header["fc.weight"]["dtype"] == "BF16",
          str(header))
    check("__metadata__ is not a tensor", "__metadata__" not in header)
    check("a bfloat16 checkpoint is not packed", not pq.carries_packed_tensors(header))

    packed = os.path.join(tmp, "packed.safetensors")
    write_safetensors(packed, {"fc.weight": ("U32", [4, 1]), "fc.scales": ("BF16", [4, 2]),
                               "fc.biases": ("BF16", [4, 2]), "norm.weight": ("BF16", [8])})
    check("scales, biases and integer weights read as packed",
          pq.carries_packed_tensors(pq.safetensors_header(packed)))
    indexed = os.path.join(tmp, "indexed.safetensors")
    write_safetensors(indexed, {"d2t": ("U32", [16]), "fc.weight": ("BF16", [4, 8])})
    check("an integer index table beside bfloat16 weights does not",
          not pq.carries_packed_tensors(pq.safetensors_header(indexed)))

    for name, blob in (("an empty file", b""),
                       ("a truncated header", struct.pack("<Q", 4096) + b"{}"),
                       ("a header that is not JSON", struct.pack("<Q", 3) + b"{,}"),
                       ("a header that is not an object", struct.pack("<Q", 2) + b"[]"),
                       ("an implausible header length", struct.pack("<Q", 1 << 40))):
        broken = os.path.join(tmp, "broken.safetensors")
        with open(broken, "wb") as binary:
            binary.write(blob)
        refuses(f"{name} is refused", pq.safetensors_header, broken)
    bad_entry = os.path.join(tmp, "entry.safetensors")
    body = json.dumps({"fc.weight": {"shape": [1]}}).encode()
    with open(bad_entry, "wb") as binary:
        binary.write(struct.pack("<Q", len(body)) + body)
    refuses("an entry with no dtype is refused", pq.safetensors_header, bad_entry)

with tempfile.TemporaryDirectory() as tmp:
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        write_safetensors(os.path.join(tmp, shard), {"fc.weight": ("BF16", [4, 8])})
    refuses("one tensor name in two shards is refused", pq.checkpoint_header, tmp)

# --------------------------------------------------------------------------- 2. the metadata

print("== 2. the metadata validator")
check("a well-formed block is accepted", pq.validate_metadata(good_meta()) == good_meta())

for field, value, why in (
        ("format", "mlx/quantized", "a foreign format name"),
        ("format", None, "no format name"),
        ("format_version", 2, "a version this build does not implement"),
        ("format_version", 0, "version zero"),
        ("format_version", "1", "a version that is a string"),
        ("format_version", True, "a version that is a bool"),
        ("bits", 8, "8 bits"),
        ("bits", 2, "2 bits"),
        ("bits", "4", "bits as a string"),
        ("group_size", 128, "group 128"),
        ("group_size", 32, "group 32"),
        ("mode", "mxfp4", "a mode this build cannot reconstruct"),
        ("mode", "", "an empty mode"),
        ("selection_rule", "all-linear/v1", "a different selection rule"),
        ("quantized_modules", [], "an empty module list"),
        ("quantized_modules", "fc", "a module list that is a string"),
        ("quantized_modules", ["fc", "fc"], "a repeated module path"),
        ("quantized_modules", ["fc", "layers.0..mlp"], "an empty path segment"),
        ("quantized_modules", ["fc", "../../etc/passwd"], "a filesystem path"),
        ("quantized_modules", ["fc", " layers.0.mlp.up_proj"], "a path with whitespace"),
        ("quantized_modules", ["fc", ""], "an empty path"),
        ("quantized_modules", ["fc", 7], "a path that is not a string"),
        ("quantized_modules", ["fc", "layers.0.attention_conv.kernel_projection"],
         "a module the rule excludes (_conv)"),
        ("quantized_modules", ["fc", "candidate_selector.hidden_projection"],
         "a module the rule excludes (candidate_selector)"),
        ("dflash_config", {}, "an empty dflash config"),
        ("dflash_config", "hidden_size=128", "a dflash config that is a string"),
):
    bad = good_meta()
    bad[field] = value
    refuses(f"{why} is refused", pq.validate_metadata, bad)

for field in ("format", "format_version", "bits", "group_size", "mode", "selection_rule",
              "quantized_modules", "dflash_config"):
    bad = good_meta()
    del bad[field]
    refuses(f"a block with no {field} is refused", pq.validate_metadata, bad)

for value in (None, [], "marked", 4):
    refuses(f"a block that is {value!r} is refused", pq.validate_metadata, value)

print("== 2b. the load options a caller may ask for")
accepted = good_meta()
pq.check_requested(accepted, quantize=True, bits=4, group_size=64)
check("4-bit group-64 quantized loading is accepted", True)
refuses("quantize=False (bfloat16) is refused", pq.check_requested, accepted,
        quantize=False, bits=4, group_size=64)
refuses("a different bit width is refused", pq.check_requested, accepted,
        quantize=True, bits=8, group_size=64)
refuses("a different group size is refused", pq.check_requested, accepted,
        quantize=True, bits=4, group_size=128)

print("== 2c. the dtype rule")
good_header = {f"{m}.weight": {"dtype": "U32", "shape": [1]} for m in TINY_MODULES}
good_header |= {f"{m}.{s}": {"dtype": "BF16", "shape": [1]} for m in TINY_MODULES
                for s in ("scales", "biases")}
good_header |= {"norm.weight": {"dtype": "BF16", "shape": [1]},
                "candidate_selector.predecessor_codebook": {"dtype": "BF16", "shape": [1]},
                "layers.0.attention_conv.base_kernel": {"dtype": "BF16", "shape": [1]}}
pq.check_tensor_dtypes(good_header, TINY_MODULES)
check("uint32 packed weights beside bfloat16 everything else is accepted", True)
for why, name, dtype in (
        ("float16 scales", "fc.scales", "F16"),
        ("float32 scales", "fc.scales", "F32"),
        ("a packed weight left in bfloat16", "fc.weight", "BF16"),
        ("an unquantized weight stored as uint32", "norm.weight", "U32"),
        ("a codebook at float32", "candidate_selector.predecessor_codebook", "F32")):
    damaged = copy.deepcopy(good_header)
    damaged[name]["dtype"] = dtype
    refuses(f"{why} is refused", pq.check_tensor_dtypes, damaged, TINY_MODULES)

# --------------------------------------------------------------------------- 3. the config shape

print("== 3. the config shape this format allows")
args = pq.config_arguments(tiny_config())
check("the ft5 layout yields the DFlash 2 fields",
      args["block_size"] == 4 and args["selector_rank"] == 64 and args["selector_top_k"] == 4
      and args["conv_kernel_size"] == 2 and args["conv_group_size"] == 16
      and args["target_layer_ids"] == (0, 1) and args["layer_types"] == (
          "sliding_attention", "sliding_attention"), str(args))
check("rope_theta is read from rope_parameters", args["rope_theta"] == 10000000)
check("mask_token_id is read from dflash_config", args["mask_token_id"] == 3)
check("the fields this format pins are pinned",
      args["rope_scaling"] is None and args["final_logit_softcapping"] is None
      and args["output_multiplier"] == 1.0)


def spoiled(**changes: Any) -> dict[str, Any]:
    cfg = tiny_config()
    for key, value in changes.items():
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    return cfg


def spoiled_dfc(**changes: Any) -> dict[str, Any]:
    cfg = tiny_config()
    for key, value in changes.items():
        if value is None:
            cfg["dflash_config"].pop(key, None)
        else:
            cfg["dflash_config"][key] = value
    return cfg


for why, cfg in (
        ("a DFlash 1 head", spoiled(architectures=["DFlashDraftModel"])),
        ("no architectures", spoiled(architectures=None)),
        ("a Markov head", spoiled(markov_rank=8)),
        ("rope scaling", spoiled(rope_scaling={"rope_type": "yarn", "factor": 2.0})),
        ("the flat DFlash 1 layout", spoiled(dflash_config=None)),
        ("a dflash_config that is a list", spoiled(dflash_config=[])),
        ("no rope_parameters", spoiled(rope_parameters=None)),
        ("rope_parameters without a theta", spoiled(rope_parameters={"rope_type": "default"})),
        ("rope_theta in two places", spoiled(rope_theta=1e6)),
        ("no selector rank", spoiled_dfc(selector_rank=0)),
        ("no selector top-k", spoiled_dfc(selector_top_k=None)),
        ("no dynamic convolutions", spoiled_dfc(conv_kernel_size=0)),
        ("a logit softcap", spoiled_dfc(final_logit_softcapping=30.0)),
        ("an output multiplier", spoiled_dfc(output_multiplier=2.0)),
        ("no layer types", spoiled(layer_types=None)),
        ("no hidden size", spoiled(hidden_size=None)),
        ("no vocab size", spoiled(vocab_size=None)),
        ("no block size", spoiled_dfc(block_size=None)),
        ("no target layer ids", spoiled_dfc(target_layer_ids=None)),
):
    refuses(f"{why} is refused", pq.config_arguments, cfg)

# --------------------------------------------------------------------------- 4. what gets routed

print("== 4. which loader a directory gets")
with tempfile.TemporaryDirectory() as tmp:
    plain_dir = os.path.join(tmp, "bf16")
    os.makedirs(plain_dir)
    with open(os.path.join(plain_dir, "config.json"), "w") as f:
        json.dump(tiny_config(), f)
    write_safetensors(os.path.join(plain_dir, "model.safetensors"),
                      {"fc.weight": ("BF16", [128, 256])}, {"format": "mlx"})
    config, meta = pq.inspect(plain_dir)
    check("an unmarked bfloat16 checkpoint carries no metadata", meta is None)
    check("its config comes back untouched", config == tiny_config())

    marked_dir = os.path.join(tmp, "marked")
    os.makedirs(marked_dir)
    marked = tiny_config() | {pq.FORMAT_KEY: good_meta()}
    with open(os.path.join(marked_dir, "config.json"), "w") as f:
        json.dump(marked, f)
    write_safetensors(os.path.join(marked_dir, "model.safetensors"),
                      {"fc.weight": ("U32", [128, 32])}, {"format": "mlx"})
    _, meta = pq.inspect(marked_dir)
    check("a marked artifact comes back validated", meta is not None
          and meta["selection_rule"] == pq.SELECTION_RULE)

    stray_dir = os.path.join(tmp, "stray")
    os.makedirs(stray_dir)
    with open(os.path.join(stray_dir, "config.json"), "w") as f:
        json.dump(tiny_config(), f)
    write_safetensors(os.path.join(stray_dir, "model.safetensors"),
                      {"fc.weight": ("U32", [128, 32]), "fc.scales": ("BF16", [128, 4])})
    refuses("packed tensors with no marker are refused, not loaded as bfloat16",
            pq.inspect, stray_dir)

    unsupported_dir = os.path.join(tmp, "v99")
    os.makedirs(unsupported_dir)
    future = tiny_config() | {pq.FORMAT_KEY: good_meta() | {"format_version": 99}}
    with open(os.path.join(unsupported_dir, "config.json"), "w") as f:
        json.dump(future, f)
    write_safetensors(os.path.join(unsupported_dir, "model.safetensors"),
                      {"fc.weight": ("U32", [128, 32])})
    refuses("an unsupported marked artifact fails rather than falling back",
            pq.inspect, unsupported_dir)

    broken_dir = os.path.join(tmp, "broken-config")
    os.makedirs(broken_dir)
    with open(os.path.join(broken_dir, "config.json"), "w") as f:
        f.write("{not json")
    try:
        pq.inspect(broken_dir)
        check("a config that is not JSON raises", False, "accepted")
    except ValueError:
        check("a config that is not JSON raises", True)

    notdict_dir = os.path.join(tmp, "notdict")
    os.makedirs(notdict_dir)
    with open(os.path.join(notdict_dir, "config.json"), "w") as f:
        json.dump([1, 2], f)
    refuses("a config that is not an object is refused", pq.inspect, notdict_dir)


print("== 4b. where the exporter may write, and how it publishes")
# The exporter's path and publication rules touch only the filesystem, so they run here
# rather than behind --gpu; importing it loads the standard library and the format module.
from bench.drafter import export_quantized as ex

with tempfile.TemporaryDirectory() as tmp_link:
    tmp = os.path.realpath(tmp_link)
    ckpt = os.path.join(tmp, "Run", "ckpt")
    os.makedirs(ckpt)
    # Aliases `realpath` leaves alone. The case alias exists on a case-insensitive volume,
    # the macOS default; the firmlink wherever the system volume is split from the data one.
    case_alias = os.path.join(tmp, "run", "CKPT")
    if os.path.isdir(case_alias):
        refuses("an output inside the source under another spelling is refused",
                ex.resolve_paths, ckpt, os.path.join(case_alias, "out"))
        refuses("and so is the source itself under another spelling",
                ex.resolve_paths, ckpt, case_alias)
    else:
        print("   skip the case alias: this volume is case-sensitive")
    firmlink = "/System/Volumes/Data" + ckpt
    if os.path.isdir(firmlink) and os.path.samefile(firmlink, ckpt):
        refuses("an output reaching the source through a firmlink is refused",
                ex.resolve_paths, ckpt, os.path.join(firmlink, "out"))
    else:
        print("   skip the firmlink: no split data volume here")
    sibling = os.path.join(tmp, "Run", "artifact")
    check("a sibling of the source is still accepted",
          ex.resolve_paths(ckpt, sibling) == (ckpt, sibling))

    def staged(name: str) -> str:
        """A directory standing in for a validated staging area, with content to hash."""
        path = os.path.join(tmp, name)
        os.makedirs(path)
        with open(os.path.join(path, "model.safetensors"), "w") as f:
            f.write(f"{name}\n")
        return path

    kernel_rename = ex._kernel_rename_noreplace
    if sys.platform == "darwin":
        check("the kernel's exclusive rename is reachable here",
              kernel_rename(staged("probe"), os.path.join(tmp, "probe-published")))

    def no_kernel_rename(src: str, dst: str) -> bool:
        return False

    # Both mechanisms, because the fallback only runs where the kernel lacks the primitive
    # and would otherwise never run here at all. A plain `os.rename` fails the empty-directory
    # case: it replaces the directory, which is exactly what a check-then-rename left open.
    for how, mechanism in (("the kernel rename", kernel_rename),
                           ("the mkdir claim", no_kernel_rename)):
        ex._kernel_rename_noreplace = mechanism
        try:
            src = staged(f"{how}-staged")
            staged_hashes = ex.directory_hashes(src)
            free = os.path.join(tmp, f"{how}-free")
            ex.publish(src, free)
            check(f"{how}: a free name is published",
                  ex.directory_hashes(free) == staged_hashes and not os.path.exists(src))

            src = staged(f"{how}-staged-again")
            staged_hashes = ex.directory_hashes(src)
            empty = os.path.join(tmp, f"{how}-empty")
            os.mkdir(empty)
            refuses(f"{how}: an empty directory in the way is refused, not replaced",
                    ex.publish, src, empty)
            check(f"{how}: and that directory is still there, still empty",
                  os.path.isdir(empty) and not os.listdir(empty))

            full = os.path.join(tmp, f"{how}-full")
            os.mkdir(full)
            with open(os.path.join(full, "SENTINEL"), "w") as f:
                f.write("another run's work\n")
            full_hashes = ex.directory_hashes(full)
            refuses(f"{how}: a populated directory in the way is refused",
                    ex.publish, src, full)
            check(f"{how}: and its contents are unchanged",
                  ex.directory_hashes(full) == full_hashes)

            dangling = os.path.join(tmp, f"{how}-dangling")
            os.symlink(os.path.join(tmp, "nowhere"), dangling)
            refuses(f"{how}: a dangling symlink in the way is refused", ex.publish, src, dangling)
            check(f"{how}: and the link is untouched",
                  os.readlink(dangling) == os.path.join(tmp, "nowhere"))

            plain_file = os.path.join(tmp, f"{how}-file")
            with open(plain_file, "w") as f:
                f.write("a file\n")
            refuses(f"{how}: a file in the way is refused", ex.publish, src, plain_file)
            check(f"{how}: and every refusal left the staged directory whole",
                  ex.directory_hashes(src) == staged_hashes)
        finally:
            ex._kernel_rename_noreplace = kernel_rename


# --------------------------------------------------------------------------- 5. synthetic models

def run_gpu_tier() -> None:
    """Export and reload a real, tiny DFlash 2 drafter. Needs mlx and the dspark venv."""
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten

    import patches
    from bench.drafter import export_quantized as ex

    def digest(model: Any) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
        return {k: (str(v.dtype), tuple(v.shape), bytes(memoryview(v)))
                for k, v in tree_flatten(model.parameters())}

    def write_source(path: str, config: dict[str, Any],
                     written: dict[str, Any] | None = None) -> Any:
        """A plain bfloat16 DFlash 2 checkpoint: exactly what the pinned loader expects.

        `written` writes a different `config.json` beside the same tensors, for the one case
        that needs a config the *upstream* loader accepts and this format does not. It must
        describe the same architecture, or the checkpoint would not load at all and the test
        would be measuring the wrong refusal.
        """
        from mlx_dspark.dflash_model import DFlashDraftModel

        cfg = pq.build_config(config)
        model = DFlashDraftModel(cfg)
        model.set_dtype(mx.bfloat16)
        mx.eval(model.parameters())
        os.makedirs(path, exist_ok=True)
        mx.save_safetensors(os.path.join(path, "model.safetensors"),
                            dict(tree_flatten(model.parameters())), metadata={"format": "mlx"})
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(written if written is not None else config, f, indent=2)
        return model

    print("== 5. a synthetic export and reload")
    patches.install_all(stream=io.StringIO())
    check("the patch reports itself applied", pq.DFlashPrequantized().applied())
    patches.install_all(stream=io.StringIO())
    check("installing it twice is harmless", pq.DFlashPrequantized().applied())
    from mlx_dspark import load as dload
    from mlx_dspark import server as dserver

    check("the server's own reference is the patched one",
          getattr(dserver.load_dflash, "_bonsai2_prequantized", False))
    check("the package re-export is the patched one",
          getattr(__import__("mlx_dspark").load_dflash, "_bonsai2_prequantized", False))

    tmp = tempfile.mkdtemp(prefix="prequantized-test-")
    try:
        source = os.path.join(tmp, "source")
        write_source(source, tiny_config())
        source_before = {name: os.path.getsize(os.path.join(source, name))
                         for name in sorted(os.listdir(source))}
        source_hashes_before = ex.directory_hashes(source)
        source_header = pq.checkpoint_header(source)
        check("mx.save_safetensors writes a header this reader understands",
              len(source_header) == 36 and all(e["dtype"] == "BF16" for e in source_header.values()),
              f"{len(source_header)} tensors")

        artifact = os.path.join(tmp, "artifact")
        result = ex.export_prequantized(source, artifact)
        check("the exporter published the destination", os.path.isdir(artifact))
        umask = os.umask(0)
        os.umask(umask)
        mode = os.stat(artifact).st_mode & 0o777
        check("with an ordinary directory's mode, not mkdtemp's 0700",
              mode == 0o777 & ~umask, f"{oct(mode)} against {oct(0o777 & ~umask)}")
        check("no temporary sibling is left behind",
              not [n for n in os.listdir(tmp) if n.startswith(".export-")], str(os.listdir(tmp)))
        check("the artifact is smaller than the source",
              result["byte_counts"]["artifact_total"] < result["byte_counts"]["source_total"],
              json.dumps(result["byte_counts"]))
        check("the recorded selection is the 15 backbone projections",
              result["quantization"]["quantized_modules"] == TINY_MODULES,
              str(result["quantization"]["quantized_modules"]))
        check("the mode was read off the quantized modules, not assumed",
              result["quantization"]["mode"] == "affine")
        check("the manifest carries no absolute path",
              not [line for line in json.dumps(result).split('"') if line.startswith("/")],
              "an absolute path reached the manifest")
        check("the source was not mutated",
              {name: os.path.getsize(os.path.join(source, name))
               for name in sorted(os.listdir(source))} == source_before)

        reloaded, reloaded_cfg = dload.load_dflash(artifact)
        runtime, runtime_cfg = ex.load_source_quantized(source)
        left, right = digest(runtime), digest(reloaded)
        check("the same tensor names", set(left) == set(right),
              str(sorted(set(left) ^ set(right))[:5]))
        shared = sorted(set(left) & set(right))
        check("the same shapes and dtypes",
              all(left[k][:2] == right[k][:2] for k in shared),
              str([k for k in shared if left[k][:2] != right[k][:2]][:5]))
        check("the same bytes, exactly", all(left[k] == right[k] for k in shared),
              str([k for k in shared if left[k] != right[k]][:5]))
        check("the DFlash config round-trips",
              pq.config_fields(reloaded_cfg) == pq.config_fields(runtime_cfg))

        quantized = sorted(p for p, m in reloaded.named_modules()
                           if isinstance(m, nn.QuantizedLinear))
        check("the same modules are quantized", quantized == TINY_MODULES, str(quantized))
        check("each one carries 4 bits, group 64, affine",
              all(m.bits == 4 and m.group_size == 64 and m.mode == "affine"
                  for _, m in reloaded.named_modules()
                  if isinstance(m, nn.QuantizedLinear)))
        conv_and_selector = sorted(p for p, m in reloaded.named_modules()
                                   if isinstance(m, nn.Linear))
        check("the convolution and selector projections stayed nn.Linear",
              conv_and_selector == sorted(
                  [f"layers.{i}.{c}.kernel_projection" for i in range(2)
                   for c in ("attention_conv", "mlp_conv")]
                  + ["candidate_selector.hidden_projection"]), str(conv_and_selector))
        check("the selector codebooks are untouched bfloat16",
              all(right[f"candidate_selector.{k}_codebook"][0] == "mlx.core.bfloat16"
                  for k in ("predecessor", "successor")))
        check("the conv base kernels are untouched bfloat16",
              all(right[f"layers.{i}.{c}.base_kernel"][0] == "mlx.core.bfloat16"
                  for i in range(2) for c in ("attention_conv", "mlp_conv")))
        check("no target embedding or head was written",
              not [k for k in right if k.startswith(("embed_tokens", "lm_head"))])

        print("== 5b. the loader's refusals")
        refuses("bfloat16 is refused from a 4-bit artifact", dload.load_dflash, artifact,
                quantize=False)
        refuses("8 bits is refused", dload.load_dflash, artifact, bits=8)
        refuses("group 128 is refused", dload.load_dflash, artifact, group_size=128)

        def variant(mutate: Any) -> str:
            path = tempfile.mkdtemp(dir=tmp, prefix="variant-")
            for name in os.listdir(artifact):
                shutil.copy2(os.path.join(artifact, name), os.path.join(path, name))
            with open(os.path.join(path, "config.json")) as f:
                cfg = json.load(f)
            mutate(cfg, path)
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(cfg, f)
            return path

        def rewrite(path: str, edit: Any) -> None:
            """Edit the shard through a replacement file.

            `mx.load` memory-maps, so saving over the file it is still reading fails with a
            bare "[read] Unable to read from file". Write beside it and rename. The
            replacement keeps the .safetensors extension, which `mx.save_safetensors`
            appends for itself otherwise.
            """
            shard = os.path.join(path, "model.safetensors")
            replacement = os.path.join(path, "rewritten.safetensors")
            weights = mx.load(shard)
            assert isinstance(weights, dict)
            edit(weights)
            mx.eval(list(weights.values()))
            mx.save_safetensors(replacement, weights, metadata={"format": "mlx"})
            os.replace(replacement, shard)

        def drop_tensor(_: Any, path: str) -> None:
            rewrite(path, lambda w: w.pop("fc.scales"))

        def add_tensor(_: Any, path: str) -> None:
            rewrite(path, lambda w: w.update({"fc.surprise": mx.zeros((4,), dtype=mx.bfloat16)}))

        def float16_scales(_: Any, path: str) -> None:
            rewrite(path, lambda w: w.update({"fc.scales": w["fc.scales"].astype(mx.float16)}))

        for why, mutate in (
                ("a missing tensor", drop_tensor),
                ("an extra tensor", add_tensor),
                ("scales saved at float16", float16_scales),
                ("a module list that misses one",
                 lambda cfg, _: cfg[pq.FORMAT_KEY]["quantized_modules"].remove("fc")),
                ("a module list with one too many",
                 lambda cfg, _: cfg[pq.FORMAT_KEY]["quantized_modules"].append(
                     "layers.0.self_attn.q_norm")),
                ("a recorded DFlash config that disagrees with config.json",
                 lambda cfg, _: cfg[pq.FORMAT_KEY]["dflash_config"].update(block_size=7)),
                ("a config.json the recorded DFlash config disagrees with",
                 lambda cfg, _: cfg["dflash_config"].update(block_size=7)),
                ("an unsupported format version",
                 lambda cfg, _: cfg[pq.FORMAT_KEY].update(format_version=99)),
                ("a marker that is not an object",
                 lambda cfg, _: cfg.update({pq.FORMAT_KEY: "yes"})),
        ):
            refuses(f"{why} is refused", dload.load_dflash, variant(mutate))

        print("== 5c. the exporter's refusals")
        refuses("an existing output is refused", ex.export_prequantized, source, artifact)
        refuses("an already quantized source is refused", ex.export_prequantized, artifact,
                os.path.join(tmp, "twice"))
        refuses("a source that is not a directory is refused", ex.export_prequantized,
                os.path.join(tmp, "nowhere"), os.path.join(tmp, "out1"))
        refuses("an output inside the source is refused", ex.export_prequantized, source,
                os.path.join(source, "out"))
        refuses("an output whose parent does not exist is refused", ex.export_prequantized,
                source, os.path.join(tmp, "typo", "out2"))
        check("no refusal left a directory behind",
              not any(os.path.exists(os.path.join(tmp, n))
                      for n in ("twice", "out1", "typo")))

        print("== 5e. nothing but this run's own staging directory is ever removed")
        # A code review's blocking finding, as a test. The staging path used to be a
        # predictable `.export-<basename>` sibling that was removed if it already existed,
        # so a source of `run/.export-artifact` with an output of `run/artifact` made the
        # exporter delete its own input. Hashes, not sizes: a truncated file has a size too.
        collide = tempfile.mkdtemp(dir=tmp, prefix="collide-")
        collide_source = os.path.join(collide, ".export-artifact")
        shutil.copytree(source, collide_source)
        with open(os.path.join(collide_source, "SENTINEL"), "w") as sentinel:
            sentinel.write("this file must survive\n")
        before = ex.directory_hashes(collide_source)
        try:
            ex.export_prequantized(collide_source, os.path.join(collide, "artifact"))
            published = True
        except pq.PrequantizedFormatError:
            published = False
        check("a source named like the old staging path is not destroyed",
              os.path.isdir(collide_source) and ex.directory_hashes(collide_source) == before,
              "the source was modified or removed")
        check("and the export either succeeded or refused, never half-published",
              published == os.path.isdir(os.path.join(collide, "artifact")))

        # The review's reproduction exactly: that collision with a failure injected at the
        # save, which used to delete the source before saving and what was left after.
        original_save = mx.save_safetensors

        def refuse_to_save(*_: Any, **__: Any) -> None:
            raise OSError("injected: the disk filled up")

        mx.save_safetensors = refuse_to_save
        try:
            ex.export_prequantized(collide_source, os.path.join(collide, "failed"))
            check("an injected save failure fails the export", False, "it published")
        except OSError:
            check("an injected save failure fails the export", True)
        finally:
            mx.save_safetensors = original_save
        check("and the colliding source survives it, hash for hash",
              os.path.isdir(collide_source) and ex.directory_hashes(collide_source) == before)
        check("and nothing was published", not os.path.lexists(os.path.join(collide, "failed")))
        check("and the failed run's own staging directory is gone",
              not [n for n in os.listdir(collide)
                   if n.startswith(".export-") and n != ".export-artifact"],
              str(os.listdir(collide)))

        # A directory that merely looks like a staging area, owned by nobody in particular.
        squatter = tempfile.mkdtemp(dir=tmp, prefix="squat-")
        stale = os.path.join(squatter, ".export-artifact")
        os.makedirs(stale)
        with open(os.path.join(stale, "SENTINEL"), "w") as sentinel:
            sentinel.write("another run's work\n")
        stale_before = ex.directory_hashes(stale)
        ex.export_prequantized(source, os.path.join(squatter, "artifact"))
        check("a pre-existing .export-* directory is left alone",
              ex.directory_hashes(stale) == stale_before, "the squatter was modified")
        check("and the export still published its own destination",
              os.path.isdir(os.path.join(squatter, "artifact")))

        # Two exports choosing one output: the second must refuse, not replace.
        refuses("a second export to the same output is refused", ex.export_prequantized,
                source, os.path.join(squatter, "artifact"))
        check("the first export's artifact is intact after the refusal",
              pq.inspect(os.path.join(squatter, "artifact"))[1] is not None)

        # Two exports that both find the output free. The rival runs to completion inside
        # this export's reload: after this one checked the name, before it publishes.
        contended = os.path.join(squatter, "contended")
        original_load = pq.load_prequantized
        rival_hashes: dict[str, str] = {}

        def rival_publishes_first(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            pq.load_prequantized = original_load        # the rival's own reload is the real one
            ex.export_prequantized(source, contended)
            rival_hashes.update(ex.directory_hashes(contended))
            return original_load(*args, **kwargs)

        pq.load_prequantized = rival_publishes_first
        try:
            refuses("the export that finishes second refuses to replace the first",
                    ex.export_prequantized, source, contended)
        finally:
            pq.load_prequantized = original_load
        check("and the rival's artifact is untouched, hash for hash",
              bool(rival_hashes) and ex.directory_hashes(contended) == rival_hashes)
        check("and the refused run left no staging directory",
              not [n for n in os.listdir(squatter)
                   if n.startswith(".export-") and n != ".export-artifact"],
              str(os.listdir(squatter)))

        # An output whose parent is a symlink into the source. `abspath` cannot see this.
        linked = tempfile.mkdtemp(dir=tmp, prefix="linked-")
        os.symlink(source, os.path.join(linked, "into-source"))
        refuses("an output reaching the source through a symlinked parent is refused",
                ex.export_prequantized, source, os.path.join(linked, "into-source", "out"))
        check("the symlinked source is unharmed",
              ex.directory_hashes(source) == source_hashes_before)
        dangling = os.path.join(linked, "dangling")
        os.symlink(os.path.join(tmp, "does-not-exist"), dangling)
        refuses("an output that is a dangling symlink is refused", ex.export_prequantized,
                source, dangling)

        print("== 5f. validation before publication has no off switch")
        try:
            ex.main(["--source", source, "--output", os.path.join(tmp, "flagged"),
                     "--no-verify-reload"])
            check("--no-verify-reload is gone from the CLI", False, "argparse accepted it")
        except SystemExit as exit_code:
            check("--no-verify-reload is gone from the CLI", exit_code.code == 2,
                  f"exited {exit_code.code}")
        check("and it published nothing", not os.path.exists(os.path.join(tmp, "flagged")))

        # The ordinary path reloads, once, from its own staging directory, while the
        # destination does not yet exist.
        reloads: list[tuple[str, bool]] = []
        spied = os.path.join(tmp, "spied")

        def spy_on_reload(path: str, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            reloads.append((path, os.path.lexists(spied)))
            return original_load(path, *args, **kwargs)

        pq.load_prequantized = spy_on_reload
        try:
            ex.export_prequantized(source, spied)
        finally:
            pq.load_prequantized = original_load
        check("a successful export reloads exactly once", len(reloads) == 1, str(reloads))
        check("from a staging directory of its own beside the destination",
              len(reloads) == 1 and os.path.dirname(reloads[0][0]) == os.path.realpath(tmp)
              and os.path.basename(reloads[0][0]).startswith(".export-"), str(reloads))
        check("before the destination existed", len(reloads) == 1 and not reloads[0][1])
        check("and only then published it", pq.inspect(spied)[1] is not None)

        # A reload failure must leave no destination. Mocked, because the point of the
        # unconditional reload is that no successful public path omits it.
        def refuse_to_reload(*_: Any, **__: Any) -> tuple[Any, Any]:
            raise pq.PrequantizedFormatError("injected: the shipped loader refused this")

        pq.load_prequantized = refuse_to_reload
        try:
            refuses("a failed reload refuses the export", ex.export_prequantized, source,
                    os.path.join(tmp, "unverified"))
        finally:
            pq.load_prequantized = original_load
        check("a failed reload leaves no destination",
              not os.path.exists(os.path.join(tmp, "unverified")))
        check("and no staging directory",
              not [n for n in os.listdir(tmp) if n.startswith(".export-")], str(os.listdir(tmp)))
        check("and does not touch the source",
              ex.directory_hashes(source) == source_hashes_before)

        # A config the pinned loader accepts and this format does not: upstream reads a
        # top-level `rope_theta`, `config_arguments` refuses one. It must not finalize.
        upstream_only = os.path.join(tmp, "upstream-only")
        # Same architecture, same rope base, stated where upstream also looks for it. The
        # pinned loader reads it and quantizes normally; `config_arguments` refuses a
        # top-level `rope_theta` outright, so the export cannot finalize.
        write_source(upstream_only, tiny_config(), tiny_config() | {"rope_theta": 10000000})
        check("the pinned loader does accept that config",
              ex.load_source_quantized(upstream_only)[1].rope_theta == 10000000)
        refuses("a source only the upstream loader supports does not finalize",
                ex.export_prequantized, upstream_only, os.path.join(tmp, "upstream-artifact"))
        check("and leaves no artifact",
              not os.path.exists(os.path.join(tmp, "upstream-artifact")))

        print("== 5d. an ordinary bfloat16 checkpoint still loads the old way")
        plain, plain_cfg = dload.load_dflash(source, quantize=True, bits=4, group_size=64)
        plain_paths = sorted(p for p, m in plain.named_modules()
                             if isinstance(m, nn.QuantizedLinear))
        check("the unmarked source still quantizes at load time", plain_paths == TINY_MODULES)
        check("and gives the same config", pq.config_fields(plain_cfg)
              == pq.config_fields(reloaded_cfg))
        bf16, _ = dload.load_dflash(source, quantize=False)
        check("and still loads at bfloat16 when asked",
              not [m for _, m in bf16.named_modules() if isinstance(m, nn.QuantizedLinear)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- 6. the real pair

def fresh_process_digest(artifact: str, deny: str | None = None) -> dict[str, Any]:
    """Load the artifact in a new interpreter and report a digest of every parameter.

    `deny`, when given, is a directory the child must not be able to read. The point is the
    claim in docs/prequantized-ft5-plan.md that the artifact loads independently of the
    checkpoint it came from: a child that cannot open the source at all and still loads is
    the only form of that claim worth writing down. `sandbox-exec` does the denying, so
    nothing is deleted or moved, and the deny is probed before it is trusted.
    """
    script = (
        "import hashlib, json, os, sys\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "import patches\n"
        "patches.install_all(stream=open(os.devnull, 'w'))\n"
        "import mlx.core as mx\n"
        "from mlx.utils import tree_flatten\n"
        "from mlx_dspark.load import load_dflash\n"
        "drafter, cfg = load_dflash(sys.argv[1])\n"
        "out = {}\n"
        "for k, v in tree_flatten(drafter.parameters()):\n"
        "    out[k] = [str(v.dtype), list(v.shape),\n"
        "              hashlib.sha256(bytes(memoryview(v))).hexdigest()]\n"
        "from dataclasses import asdict\n"
        "print('DIGEST ' + json.dumps({'params': out, 'config': asdict(cfg)}, sort_keys=True,\n"
        "                             default=list))\n")
    argv = [sys.executable, "-c", script, artifact]
    if deny is not None:
        real = os.path.realpath(deny)
        profile = f'(version 1)(allow default)(deny file-read* (subpath "{real}"))'
        probe = subprocess.run(["sandbox-exec", "-p", profile, "/bin/ls", real],
                               capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            raise RuntimeError(f"sandbox-exec did not deny reads of {real}; not trusting it")
        argv = ["sandbox-exec", "-p", profile, *argv]
    proc = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=REPO)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("DIGEST ")), None)
    if line is None:
        raise RuntimeError(f"fresh process did not load the artifact:\n{proc.stdout[-2000:]}\n"
                           f"{proc.stderr[-2000:]}")
    return json.loads(line[len("DIGEST "):])


def run_model_tier(source: str, artifact: str) -> None:
    """Phase 3: the shipped checkpoint against the saved artifact, exactly."""
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten

    import patches
    from bench.drafter import export_quantized as ex

    print("== 6. the saved artifact against the source quantized at load time")
    patches.install_all(stream=io.StringIO())
    from mlx_dspark import load as dload

    source_hashes = ex.directory_hashes(source)

    # dtype, shape and a content hash per tensor, not the tensors themselves: both arms are
    # around 1.3 GB and both have to be in hand at once, and a hash compares exactly as well
    # as the bytes do. `parameter_digest` is the exporter's own, so this compares what the
    # exporter compares when it verifies its own reload.
    def materialized(model: Any) -> dict[str, list[Any]]:
        params = dict(tree_flatten(model.parameters()))
        mx.eval(list(params.values()))
        return ex.parameter_digest(params)

    runtime, runtime_cfg = ex.load_source_quantized(source)
    runtime_classes = {p: type(m).__name__ for p, m in runtime.named_modules()
                       if isinstance(m, nn.Linear | nn.QuantizedLinear)}
    runtime_quant = {p: (m.bits, m.group_size, m.mode) for p, m in runtime.named_modules()
                     if isinstance(m, nn.QuantizedLinear)}
    runtime_digest = materialized(runtime)
    del runtime
    mx.clear_cache()

    saved, saved_cfg = dload.load_dflash(artifact)
    saved_classes = {p: type(m).__name__ for p, m in saved.named_modules()
                     if isinstance(m, nn.Linear | nn.QuantizedLinear)}
    saved_quant = {p: (m.bits, m.group_size, m.mode) for p, m in saved.named_modules()
                   if isinstance(m, nn.QuantizedLinear)}
    saved_digest = materialized(saved)
    mx.clear_cache()

    only_runtime = sorted(set(runtime_digest) - set(saved_digest))
    only_saved = sorted(set(saved_digest) - set(runtime_digest))
    check("identical parameter key sets", not only_runtime and not only_saved,
          f"source-only {only_runtime[:5]} artifact-only {only_saved[:5]}")
    shared = sorted(set(runtime_digest) & set(saved_digest))
    dtype_mismatch = [k for k in shared if runtime_digest[k][0] != saved_digest[k][0]]
    shape_mismatch = [k for k in shared if runtime_digest[k][1] != saved_digest[k][1]]
    value_mismatch = [k for k in shared if runtime_digest[k][2] != saved_digest[k][2]]
    check("identical tensor shapes", not shape_mismatch, str(shape_mismatch[:8]))
    check("identical tensor dtypes", not dtype_mismatch, str(dtype_mismatch[:8]))
    check("identical tensor values", not value_mismatch, str(value_mismatch[:8]))
    class_mismatch = sorted(p for p in set(runtime_classes) | set(saved_classes)
                            if runtime_classes.get(p) != saved_classes.get(p))
    check("identical module classes", not class_mismatch, str(class_mismatch[:8]))
    quant_mismatch = sorted(p for p in set(runtime_quant) | set(saved_quant)
                            if runtime_quant.get(p) != saved_quant.get(p))
    check("identical bits, group size and mode", not quant_mismatch, str(quant_mismatch[:8]))
    check("identical DFlash configuration",
          pq.config_fields(runtime_cfg) == pq.config_fields(saved_cfg))

    selector = sorted(p for p in saved_classes if "candidate_selector" in p)
    convs = sorted(p for p in saved_classes if "_conv" in p)
    check("the selector projection is still an nn.Linear",
          bool(selector) and all(saved_classes[p] == "Linear" for p in selector), str(selector))
    check("every convolution projection is still an nn.Linear",
          bool(convs) and all(saved_classes[p] == "Linear" for p in convs), str(convs))
    check("the selector codebooks are bfloat16 in both arms",
          all(saved_digest[f"candidate_selector.{k}_codebook"][0].endswith("bfloat16")
              and runtime_digest[f"candidate_selector.{k}_codebook"][0].endswith("bfloat16")
              for k in ("predecessor", "successor")))
    check("no target embedding or head is present",
          not [k for k in saved_digest if k.startswith(("embed_tokens", "lm_head"))])

    print("== 6b. bytes on disk")
    source_bytes = sum(os.path.getsize(p) for p in pq.shard_paths(source))
    artifact_bytes = sum(os.path.getsize(p) for p in pq.shard_paths(artifact))
    check("the saved artifact is smaller than the bfloat16 checkpoint",
          artifact_bytes < source_bytes,
          f"{artifact_bytes} >= {source_bytes}")
    print(f"      source {source_bytes} bytes, artifact {artifact_bytes} bytes, "
          f"{source_bytes / max(artifact_bytes, 1):.2f}x")

    print("== 6c. a fresh process, with the source out of reach")
    del saved
    mx.clear_cache()
    # The deny, observed rather than assumed: point it at the artifact and the same child
    # must fail. Without this, a macOS that stopped enforcing the profile would turn the
    # independence check below into a child reading whatever it liked and passing.
    try:
        fresh_process_digest(artifact, deny=artifact)
        check("the deny is real: denying the artifact stops the load", False, "loaded anyway")
    except RuntimeError:
        check("the deny is real: denying the artifact stops the load", True)
    fresh = fresh_process_digest(artifact, deny=source)
    fresh_params = fresh["params"]
    want = json.loads(json.dumps(saved_digest))   # tuples through JSON, as the child's came
    check("a fresh process reads the same tensors", set(fresh_params) == set(want),
          str(sorted(set(fresh_params) ^ set(want))[:5]))
    differing = sorted(k for k in set(fresh_params) & set(want) if fresh_params[k] != want[k])
    check("with the same dtypes, shapes and values", not differing, str(differing[:5]))
    check("and the same DFlash configuration",
          json.loads(json.dumps(fresh["config"])) == pq.config_fields(saved_cfg))
    print(f"      the child could not read {os.path.realpath(source)} and loaded anyway")

    print("== 6d. the canonical bfloat16 checkpoint is untouched and still loads")
    check("not one byte of the source changed", ex.directory_hashes(source) == source_hashes,
          str(sorted(k for k, v in ex.directory_hashes(source).items()
                     if source_hashes.get(k) != v)))
    plain = fresh_process_digest(source)
    check("a fresh process still loads it through the patched entry point",
          set(plain["params"]) == set(runtime_digest),
          str(sorted(set(plain["params"]) ^ set(runtime_digest))[:5]))
    plain_differing = sorted(k for k in set(plain["params"]) & set(runtime_digest)
                             if plain["params"][k] != json.loads(json.dumps(runtime_digest[k])))
    check("and quantizes it at load time to the same tensors as before",
          not plain_differing, str(plain_differing[:5]))


def main() -> int:
    argv = sys.argv[1:]
    if "--gpu" in argv or "--source" in argv:
        run_gpu_tier()
    if "--source" in argv:
        source = argv[argv.index("--source") + 1]
        artifact = argv[argv.index("--artifact") + 1]
        run_model_tier(os.path.abspath(source), os.path.abspath(artifact))
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
