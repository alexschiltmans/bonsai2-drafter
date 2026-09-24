"""Load a DFlash 2 drafter whose weights are already 4-bit on disk.

What this is for
----------------
The fine-tuned ft5 drafter is served at 4 bits, but ships as 3.85 GB of
bfloat16 and is quantized at load time, every time. The bits that reach the GPU are the
same either way, so the bfloat16 file is a distribution cost and nothing else. This patch
is the load side of removing that cost: a checkpoint that is *already* packed, marked as
such, and loaded without a second quantization pass.

It is not a speed change and not a new drafter. The Bonsai target, `bonsai_loader.py` and
the 2-bit verify kernel are all still required, and the bfloat16 export stays the canonical
training artifact.

The format
----------
Repository-specific and explicitly versioned, under the ``bonsai2_prequantized`` key of the
drafter's own `config.json`. It is deliberately NOT mlx's standard ``quantization`` key: no
other runtime reads this, and borrowing the standard key would advertise a compatibility
that does not exist. Everything else in the config is the source's, untouched, because
`load_dflash` reads that config to build the model.

The key carries the format version, the bit width, the group size, the quantization mode
*as the pinned implementation produced it* (not as anyone assumed), the module-selection
rule, the exact list of modules the rule selected, and the `DFlashConfig` the pinned loader
derived from this config. `bench/drafter/export_quantized.py` writes it.

Why the loader re-derives what the metadata already states
----------------------------------------------------------
:func:`build_config` rebuilds `DFlashConfig` from `config.json` rather than replaying the
recorded one, and :func:`quantize_backbone` re-applies the selection *rule* rather than
replaying the recorded paths. Both results are then checked against what the exporter
recorded. That is the point: the exporter ran the pinned upstream loader, so the recorded
values are upstream's own, and any drift between this file's re-derivation and upstream's
is caught on every load instead of becoming a drafter that loads clean and drafts wrong.

Upstream's config construction is inseparable from its weight loading (it builds the model
before it can fail), so it cannot be called for the config alone. This one is narrower on
purpose: it accepts the DFlash 2 shape this format was validated against and refuses the
rest, rather than reimplementing upstream's tolerance for the gemma4 and flat layouts.

What must not change
--------------------
An ordinary, unmarked bfloat16 checkpoint goes through the original loader untouched — same
signature, same return, same errors. A marked artifact whose metadata this build does not
support fails; it never falls back to loading the packed tensors as if they were weights.
An unmarked checkpoint that nevertheless carries packed tensors is refused by name, because
upstream's own failure for that case is a tensor-name mismatch that reads like a packaging
mistake.

Usage
-----
Installed by :func:`patches.install_all`, which `bin/mlx-dspark-patched` calls before any
model loads. Applying it twice is a no-op.
"""

from __future__ import annotations

import json
import os
import struct
from typing import TYPE_CHECKING, Any

from . import Patch

# mlx is not imported at module scope, and must not be: the validator half of this file runs
# under plain python3 in `bench/check.sh`'s no-GPU tier.
if TYPE_CHECKING:
    from mlx_dspark.dflash_model import DFlashConfig

#: Where the marker lives inside the drafter's `config.json`.
FORMAT_KEY = "bonsai2_prequantized"
#: The format's own name. A mismatch is a different format, not a different version.
FORMAT_NAME = "bonsai2-drafter/dflash2-prequantized"
#: Versions this build can load. A new version is a new entry, never a widened check.
SUPPORTED_VERSIONS = (1,)
#: The one configuration this format describes. Not a range: see the module docstring.
SUPPORTED_BITS = 4
SUPPORTED_GROUP_SIZE = 64
#: Modes the pinned mlx can round-trip through `nn.quantize` for these modules. The exporter
#: reads the mode off the quantized modules rather than assuming; this is what it may report.
SUPPORTED_MODES = ("affine",)
#: mlx-dspark 0.18.0 `load_dflash`: every `nn.Linear` outside the DFlash 2 dynamic
#: convolutions and the candidate selector. Named so a future rule is a new name, not a
#: silent change of meaning.
SELECTION_RULE = "dflash2-backbone-linear/v1"
#: Path fragments the rule excludes, in the order upstream writes them.
EXCLUDED_PATH_FRAGMENTS = ("_conv", "candidate_selector")

#: Tensor suffixes that only exist once a module is packed.
PACKED_SUFFIXES = (".scales", ".biases")
#: safetensors dtype names that cannot be a bfloat16 checkpoint's.
PACKED_DTYPES = ("U32", "U16", "U8", "I8", "I32")
#: What a tensor's dtype must be in a published artifact.
PACKED_WEIGHT_DTYPE = "uint32"
UNPACKED_DTYPE = "bfloat16"


class PrequantizedFormatError(ValueError):
    """A prequantized artifact this build will not load.

    A ValueError so it reaches the same handlers as upstream's own load refusals: the CLI
    prints it, the server turns it into a failed start, and nothing treats it as a reason to
    fall back to an unquantized load.
    """


# --------------------------------------------------------------------------- the file itself

def safetensors_header(path: str) -> dict[str, dict[str, Any]]:
    """The tensor table of a safetensors file, without reading a byte of tensor data.

    The format is an 8-byte little-endian header length followed by that much JSON. Read
    directly rather than through mlx so the checks that use it — is this checkpoint already
    packed, do the saved dtypes match what the format promises — run in the CPU test tier,
    on an interpreter with no mlx at all.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise PrequantizedFormatError(f"{path}: not a safetensors file (truncated header)")
        (length,) = struct.unpack("<Q", raw)
        if not 0 < length <= 512 * 1024 * 1024:
            raise PrequantizedFormatError(f"{path}: implausible safetensors header length {length}")
        body = f.read(length)
    if len(body) != length:
        raise PrequantizedFormatError(f"{path}: safetensors header is truncated")
    try:
        header = json.loads(body)
    except ValueError as exc:
        raise PrequantizedFormatError(f"{path}: safetensors header is not JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise PrequantizedFormatError(f"{path}: safetensors header is not an object")
    header.pop("__metadata__", None)
    for name, entry in header.items():
        if not isinstance(entry, dict) or "dtype" not in entry or "shape" not in entry:
            raise PrequantizedFormatError(f"{path}: malformed safetensors entry for {name!r}")
    return header


def shard_paths(path: str) -> list[str]:
    """The checkpoint's safetensors files, in the order `load_dflash` merges them."""
    return sorted(os.path.join(path, n) for n in os.listdir(path) if n.endswith(".safetensors"))


def checkpoint_header(path: str) -> dict[str, dict[str, Any]]:
    """The merged tensor table of every shard, refusing a name that appears twice."""
    merged: dict[str, dict[str, Any]] = {}
    for shard in shard_paths(path):
        for name, entry in safetensors_header(shard).items():
            if name in merged:
                raise PrequantizedFormatError(
                    f"{path}: tensor {name!r} appears in more than one shard")
            merged[name] = entry
    return merged


def carries_packed_tensors(header: dict[str, dict[str, Any]]) -> bool:
    """True when a tensor table can only have come from a quantized model.

    Two signals, both unambiguous: a `.scales`/`.biases` companion, or a `.weight` stored as
    integers. Deliberately not "any integer tensor anywhere" — a reduced-vocabulary DSpark
    head ships an integer `d2t` index table, and a checkpoint handed to the wrong loader
    should fail with upstream's tensor-name mismatch rather than with a claim that it is
    quantized.
    """
    return any(name.endswith(PACKED_SUFFIXES)
               or (name.endswith(".weight") and entry["dtype"] in PACKED_DTYPES)
               for name, entry in header.items())


def read_config(path: str) -> dict[str, Any]:
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise PrequantizedFormatError(f"{path}: config.json is not an object")
    return config


def marker(config: dict[str, Any]) -> Any:
    """The raw metadata block, or None for an ordinary checkpoint. Not yet validated."""
    return config.get(FORMAT_KEY)


# --------------------------------------------------------------------------- the metadata

def _require(meta: dict[str, Any], key: str, where: str) -> Any:
    if key not in meta:
        raise PrequantizedFormatError(f"{where}: {FORMAT_KEY} has no {key!r}")
    return meta[key]


def _valid_module_path(value: Any) -> bool:
    """A dotted module path as `nn.quantize` hands one to its predicate.

    Segments are attribute names or list indices, so `layers.0.mlp.down_proj` is the shape.
    Anything that is not — an empty segment, whitespace, a filesystem path, a traversal —
    would silently fail to match a real module and leave the comparison against the recorded
    selection looking like a loader bug rather than a malformed artifact.
    """
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    return all(part.isidentifier() or part.isdigit() for part in value.split("."))


def validate_metadata(meta: Any, where: str = "artifact") -> dict[str, Any]:
    """Check a metadata block against what this build supports, or raise.

    Every rejection here is deliberate. The alternative to refusing an artifact whose bits,
    group, mode or rule this build does not implement is loading packed integers as if they
    meant something else, which produces a drafter that runs, accepts almost nothing, and
    reports no error at all.
    """
    if not isinstance(meta, dict):
        raise PrequantizedFormatError(f"{where}: {FORMAT_KEY} is not an object")
    name = _require(meta, "format", where)
    if name != FORMAT_NAME:
        raise PrequantizedFormatError(
            f"{where}: format {name!r} is not {FORMAT_NAME!r} — a different format, not a "
            f"different version of this one")
    version = _require(meta, "format_version", where)
    if not isinstance(version, int) or isinstance(version, bool) or version not in SUPPORTED_VERSIONS:
        raise PrequantizedFormatError(
            f"{where}: format version {version!r} is not supported by this build "
            f"(supported: {list(SUPPORTED_VERSIONS)})")
    bits = _require(meta, "bits", where)
    if bits != SUPPORTED_BITS:
        raise PrequantizedFormatError(
            f"{where}: {bits!r} bits; this format describes {SUPPORTED_BITS}-bit weights only")
    group = _require(meta, "group_size", where)
    if group != SUPPORTED_GROUP_SIZE:
        raise PrequantizedFormatError(
            f"{where}: group size {group!r}; this format describes "
            f"{SUPPORTED_GROUP_SIZE} only")
    mode = _require(meta, "mode", where)
    if mode not in SUPPORTED_MODES:
        raise PrequantizedFormatError(
            f"{where}: quantization mode {mode!r} is not one this build can reconstruct "
            f"(supported: {list(SUPPORTED_MODES)})")
    rule = _require(meta, "selection_rule", where)
    if rule != SELECTION_RULE:
        raise PrequantizedFormatError(
            f"{where}: module selection rule {rule!r} is not {SELECTION_RULE!r}")
    modules = _require(meta, "quantized_modules", where)
    if not isinstance(modules, list) or not modules:
        raise PrequantizedFormatError(f"{where}: quantized_modules is empty or not a list")
    bad = [m for m in modules if not _valid_module_path(m)]
    if bad:
        raise PrequantizedFormatError(
            f"{where}: malformed module paths in quantized_modules: {bad[:5]}")
    if len(set(modules)) != len(modules):
        raise PrequantizedFormatError(f"{where}: quantized_modules repeats a path")
    offenders = [m for m in modules if any(frag in m for frag in EXCLUDED_PATH_FRAGMENTS)]
    if offenders:
        raise PrequantizedFormatError(
            f"{where}: {SELECTION_RULE} excludes {EXCLUDED_PATH_FRAGMENTS}, but "
            f"quantized_modules names {offenders[:5]}")
    recorded = _require(meta, "dflash_config", where)
    if not isinstance(recorded, dict) or not recorded:
        raise PrequantizedFormatError(f"{where}: dflash_config is empty or not an object")
    return meta


def check_requested(meta: dict[str, Any], *, quantize: bool, bits: int, group_size: int,
                    where: str = "artifact") -> None:
    """Refuse load options this artifact cannot honour.

    The weights on disk are already 4-bit, group 64. There is no bfloat16 in the file to
    return for ``quantize=False`` and no way to serve 8 bits from 4, so a caller asking for
    either is asking for a checkpoint this is not. Upstream answers those questions by
    quantizing what it loaded; here the only honest answer is an error, because the
    alternative — ignoring the argument — makes `--drafter-bits` silently inert.
    """
    if not quantize:
        raise PrequantizedFormatError(
            f"{where}: quantize=False asks for bfloat16 weights, but this is a "
            f"{meta['bits']}-bit prequantized artifact and carries none. Load the bfloat16 "
            f"checkpoint this was exported from instead.")
    if bits != meta["bits"]:
        raise PrequantizedFormatError(
            f"{where}: loaded at {bits} bits, but the artifact is {meta['bits']}-bit. "
            f"Requantizing packed weights is not something this format supports.")
    if group_size != meta["group_size"]:
        raise PrequantizedFormatError(
            f"{where}: loaded at group size {group_size}, but the artifact is "
            f"group {meta['group_size']}.")


def check_tensor_dtypes(header: dict[str, dict[str, Any]], modules: list[str],
                        where: str = "artifact") -> None:
    """Every packed weight is uint32 and everything else is bfloat16, or raise.

    `Module.load_weights` checks names and shapes and says nothing about dtype, so without
    this a shard whose scales were saved at float16 would load, run, and be wrong by a
    factor nobody could see. The rule is stated here rather than read out of the manifest so
    a distributed artifact needs only its weights and its config to be checked.
    """
    packed = {f"{m}.weight" for m in modules}
    wrong = []
    for name, entry in sorted(header.items()):
        want = PACKED_WEIGHT_DTYPE if name in packed else UNPACKED_DTYPE
        got = str(entry["dtype"]).lower().replace("bf16", "bfloat16").replace("u32", "uint32")
        if got != want:
            wrong.append(f"{name}: {entry['dtype']} (expected {want})")
    if wrong:
        raise PrequantizedFormatError(
            f"{where}: {len(wrong)} tensors have the wrong dtype for this format: {wrong[:5]}")


# --------------------------------------------------------------------------- the model

def config_arguments(config: dict[str, Any], where: str = "artifact") -> dict[str, Any]:
    """The `DFlashConfig` keyword arguments for the one config shape this format allows.

    Narrow on purpose. mlx-dspark 0.18.0's `load_dflash` tolerates three checkpoint layouts
    (DFlash 1 flat, DFlash 2 nested, gemma4 rope) because it has to load whatever z-lab and
    Inco published; this format describes one exported artifact, so anything but the nested
    DFlash 2 layout is refused rather than guessed at. :func:`load_prequantized` then checks
    the result field by field against the `DFlashConfig` the pinned loader itself derived
    at export time, so a divergence between this and upstream surfaces as a load error.

    Separate from :func:`build_config` so every refusal below is reachable without mlx, in
    the CPU test tier — these are the checks that decide whether a file is this format at
    all, and they should not need a GPU to exercise.
    """
    architectures = config.get("architectures") or []
    if "DFlash2DraftModel" not in architectures:
        raise PrequantizedFormatError(
            f"{where}: architectures {architectures} is not a DFlash 2 drafter")
    if config.get("markov_rank"):
        raise PrequantizedFormatError(f"{where}: a Markov head is not part of this format")
    if config.get("rope_scaling") is not None:
        raise PrequantizedFormatError(f"{where}: rope_scaling is not part of this format")
    dfc = config.get("dflash_config")
    if not isinstance(dfc, dict):
        raise PrequantizedFormatError(
            f"{where}: no dflash_config object; the flat DFlash 1 layout is not part of "
            f"this format")
    rope = config.get("rope_parameters")
    if not isinstance(rope, dict) or "rope_theta" not in rope:
        raise PrequantizedFormatError(f"{where}: no rope_parameters.rope_theta")
    if config.get("rope_theta") is not None:
        raise PrequantizedFormatError(
            f"{where}: rope_theta is at the top level as well as under rope_parameters")
    selector_rank = int(dfc.get("selector_rank") or 0)
    selector_top_k = int(dfc.get("selector_top_k") or 0)
    if not (selector_rank and selector_top_k):
        raise PrequantizedFormatError(
            f"{where}: no candidate selector; refusing to run a DFlash 2 head as DFlash 1")
    if not dfc.get("conv_kernel_size"):
        raise PrequantizedFormatError(f"{where}: no dynamic convolutions (conv_kernel_size)")
    for unsupported in ("final_logit_softcapping", "output_multiplier"):
        if dfc.get(unsupported) is not None or config.get(unsupported) is not None:
            raise PrequantizedFormatError(f"{where}: {unsupported} is not part of this format")
    # Required rather than defaulted, and that is load-bearing beyond the config shape:
    # `DFlashDraftModel.__init__` fills an empty `layer_types` in on the config object it is
    # handed, so an artifact exported with one empty would record the filled-in value and
    # never match what :func:`load_prequantized` rebuilds before constructing the model.
    layer_types = config.get("layer_types")
    if not layer_types:
        raise PrequantizedFormatError(f"{where}: no layer_types")
    try:
        return {
            "hidden_size": config["hidden_size"],
            "num_hidden_layers": config["num_hidden_layers"],
            "num_attention_heads": config["num_attention_heads"],
            "num_key_value_heads": config["num_key_value_heads"],
            "head_dim": config["head_dim"],
            "intermediate_size": config["intermediate_size"],
            "vocab_size": config["vocab_size"],
            "rms_norm_eps": config["rms_norm_eps"],
            "rope_theta": rope["rope_theta"],
            "max_position_embeddings": config["max_position_embeddings"],
            "block_size": int(dfc["block_size"]),
            "target_layer_ids": tuple(dfc["target_layer_ids"]),
            "num_target_layers": config["num_target_layers"],
            "mask_token_id": dfc.get("mask_token_id", 0),
            "rope_scaling": None,
            "layer_types": tuple(layer_types),
            "sliding_window": config.get("sliding_window"),
            "final_logit_softcapping": None,
            "selector_rank": selector_rank,
            "selector_top_k": selector_top_k,
            "conv_kernel_size": int(dfc["conv_kernel_size"]),
            "conv_group_size": int(dfc.get("conv_group_size") or 16),
            "output_multiplier": 1.0,
        }
    except KeyError as exc:
        raise PrequantizedFormatError(f"{where}: config has no {exc.args[0]!r}") from exc


def build_config(config: dict[str, Any], where: str = "artifact") -> DFlashConfig:
    """The `DFlashConfig` :func:`config_arguments` describes."""
    from mlx_dspark.dflash_model import DFlashConfig

    return DFlashConfig(**config_arguments(config, where=where))


def config_fields(config: DFlashConfig) -> dict[str, Any]:
    """A `DFlashConfig` as JSON-comparable values, so a recorded one and a rebuilt one can
    be compared without a tuple/list difference reading as a mismatch."""
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(config)))


def quantize_backbone(drafter: Any, *, bits: int, group_size: int, mode: str) -> list[str]:
    """Quantize exactly the modules :data:`SELECTION_RULE` names, and report which.

    The predicate is mlx-dspark 0.18.0 `load_dflash`'s, restated: every `nn.Linear` whose
    path carries neither ``_conv`` nor ``candidate_selector``. Running it — rather than
    quantizing the paths the artifact lists — is what makes the comparison in
    :func:`load_prequantized` worth anything: it checks the rule still selects what it
    selected at export time, on a model this build constructed.
    """
    from mlx import nn

    selected: list[str] = []

    def predicate(path: str, module: Any) -> bool:
        chosen = (isinstance(module, nn.Linear)
                  and not any(frag in path for frag in EXCLUDED_PATH_FRAGMENTS))
        if chosen:
            selected.append(path)
        return chosen

    nn.quantize(drafter, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate)
    return sorted(selected)


def load_prequantized(path: str, config: dict[str, Any], meta: dict[str, Any]) -> tuple[Any, Any]:
    """(drafter, DFlashConfig) for a validated prequantized artifact directory.

    Build, quantize, then load — the reverse of upstream's order, and the whole point. The
    packed tensors go into modules that are already `nn.QuantizedLinear`, so nothing
    quantizes a second time and nothing has to be dequantized first.
    """
    import mlx.core as mx
    from mlx import nn
    from mlx_dspark.dflash_model import DFlashDraftModel

    cfg = build_config(config, where=path)
    recorded, rebuilt = meta["dflash_config"], config_fields(cfg)
    drift = sorted(k for k in set(recorded) | set(rebuilt) if recorded.get(k) != rebuilt.get(k))
    if drift:
        raise PrequantizedFormatError(
            f"{path}: this build derives a different DFlash config from the artifact's own "
            f"config.json than the exporter's loader did. Fields: "
            f"{[(k, recorded.get(k), rebuilt.get(k)) for k in drift[:5]]}")

    drafter = DFlashDraftModel(cfg)
    selected = quantize_backbone(drafter, bits=meta["bits"], group_size=meta["group_size"],
                                 mode=meta["mode"])
    expected = sorted(meta["quantized_modules"])
    if selected != expected:
        missing, extra = sorted(set(expected) - set(selected)), sorted(set(selected) - set(expected))
        raise PrequantizedFormatError(
            f"{path}: {SELECTION_RULE} selects a different module set than the artifact "
            f"records."
            + (f"\n  recorded but not selected ({len(missing)}): {missing[:8]}" if missing else "")
            + (f"\n  selected but not recorded ({len(extra)}): {extra[:8]}" if extra else ""))

    header = checkpoint_header(path)
    check_tensor_dtypes(header, expected, where=path)
    weights: dict[str, Any] = {}
    for shard in shard_paths(path):
        loaded = mx.load(shard)
        assert isinstance(loaded, dict)
        weights.update(loaded)
    # The same diagnosis upstream makes before loading, for the same reason: a partially
    # loaded drafter runs and accepts nothing, which is worse than an error.
    from mlx.utils import tree_flatten

    model_keys = {k for k, _ in tree_flatten(drafter.parameters())}
    ckpt_keys = set(weights)
    if model_keys != ckpt_keys:
        missing, extra = sorted(model_keys - ckpt_keys), sorted(ckpt_keys - model_keys)
        raise PrequantizedFormatError(
            f"{path}: tensor names do not match the quantized DFlash 2 drafter this config "
            f"builds."
            + (f"\n  missing in artifact ({len(missing)}): {missing[:8]}" if missing else "")
            + (f"\n  unexpected in artifact ({len(extra)}): {extra[:8]}" if extra else ""))
    drafter.load_weights(list(weights.items()))

    for name in expected:
        module = drafter
        for part in name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        if not isinstance(module, nn.QuantizedLinear):
            raise PrequantizedFormatError(f"{path}: {name} did not become an nn.QuantizedLinear")
        if (module.bits, module.group_size, module.mode) != (meta["bits"], meta["group_size"],
                                                             meta["mode"]):
            raise PrequantizedFormatError(
                f"{path}: {name} carries {module.bits} bits / group {module.group_size} / "
                f"{module.mode}, not the artifact's "
                f"{meta['bits']} / {meta['group_size']} / {meta['mode']}")
    mx.eval(drafter.parameters())
    return drafter, cfg


def inspect(path: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(config, validated metadata or None) for a resolved checkpoint directory.

    The one place that decides which loader a checkpoint gets. An unmarked directory holding
    packed tensors is refused here rather than handed on, because upstream's answer to it is
    a tensor-name mismatch listing 45 unexpected keys, which reads like a broken export
    instead of a marker someone forgot to write.
    """
    config = read_config(path)
    meta = marker(config)
    if meta is None:
        if carries_packed_tensors(checkpoint_header(path)):
            raise PrequantizedFormatError(
                f"{path}: the checkpoint carries packed tensors (scales, biases or integer "
                f"weights) but no {FORMAT_KEY} block in config.json, so nothing states what "
                f"they mean. Export it with bench/drafter/export_quantized.py, or load the "
                f"bfloat16 checkpoint.")
        return config, None
    return config, validate_metadata(meta, where=path)


# --------------------------------------------------------------------------- the patch

class DFlashPrequantized(Patch):
    """Route a marked 4-bit DFlash 2 artifact past upstream's quantize-after-load path."""

    name = "prequantized DFlash 2 drafter loader"

    def applied(self) -> bool:
        from mlx_dspark import load

        return getattr(load.load_dflash, "_bonsai2_prequantized", False)

    def apply(self) -> None:
        from mlx_dspark import load

        orig = load.load_dflash

        def load_dflash(repo_or_path: str, *, quantize: bool = True, bits: int = 4,
                        group_size: int = 64) -> tuple[Any, Any]:
            path = load._resolve(repo_or_path)
            config, meta = inspect(path)
            if meta is None:
                return orig(repo_or_path, quantize=quantize, bits=bits, group_size=group_size)
            check_requested(meta, quantize=quantize, bits=bits, group_size=group_size, where=path)
            return load_prequantized(path, config, meta)

        setattr(load_dflash, "_bonsai2_prequantized", True)  # noqa: B010 - a marker mypy would otherwise refuse
        # The original, kept reachable on purpose: `bench/drafter/export_quantized.py` has to
        # read its source through the pinned loader's own quantization path, and it must not
        # depend on being imported before the patches are installed to get it.
        setattr(load_dflash, "_bonsai2_original", orig)  # noqa: B010 - as above
        load.load_dflash = load_dflash
        # `mlx_dspark.server` imports the name at module load and `mlx_dspark/__init__.py`
        # re-exports it, so both hold their own reference to the original. `mlx_dspark.cli`
        # imports it inside its functions and therefore needs nothing, but is listed so a
        # future module-level import there is covered rather than silently unpatched.
        for modname in ("mlx_dspark", "mlx_dspark.server", "mlx_dspark.cli"):
            try:
                mod = __import__(modname, fromlist=["load_dflash"])
            except ImportError:
                continue
            if hasattr(mod, "load_dflash"):
                setattr(mod, "load_dflash", load_dflash)  # noqa: B010 - rebinding a module global
