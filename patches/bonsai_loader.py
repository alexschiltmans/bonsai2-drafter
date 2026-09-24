"""Load PrismML's Bonsai 2 MLX pack as an mlx-dspark target.

The pack
--------
`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` declares `model_type: prism_hadamard_qwen35`. Its
language model is Qwen3.8-27B with every projection, the embedding and the LM head stored
as MLX affine 2-bit weights, group 128, in a *rotated* basis: each matrix was transformed
blockwise by a Walsh-Hadamard rotation with fixed signs before the ternary assignment, and
the matching transform has to be applied to the activations at runtime (and its inverse to
the embedding output). An ordinary loader builds the right architecture, loads the right
tensors, and answers wrong with no error, which is why the pack ships `runtime/` and why
mlx-dspark, which routes by `model_type`, cannot load it as it stands.

The patch
---------
Wrap `mlx_dspark.load.load_target`. For this one model type, build mlx-lm's `qwen3_5`
Model from the pack's `text_config`, replace the 402 packed modules with :class:`PackedLinear` and
:class:`PackedEmbedding` (the pack's own runtime module, split in two: transform, then
`mx.quantized_matmul` at 2 bits),
load the language-model weights, and hand the result to mlx-dspark's `Target` exactly as
`load_target` would. Every other target goes through the stock path untouched.

The vision tower in the pack is dropped, as mlx-lm's own `qwen3_5` module drops it for
the stock checkpoints; this repo serves text.

The verify kernel
-----------------
The first version kept the pack's plain module. It served at the llama.cpp fork's speed
(20.9 against 21.2 tok/s) and the DFlash 2 drafter accepted 5.0 tokens a round against it,
but the verify round cost 485 ms for 8 rows against 48 ms for one, because
mlx-dspark's small-M kernel routes `nn.QuantizedLinear` instances only. `PackedLinear` is
one now, and patches/small_m_2bit.py gives the kernel the 2-bit, group-128 unpack.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
from mlx import nn

from . import Patch

MODEL_TYPE = "prism_hadamard_qwen35"
DTYPE = mx.bfloat16


def fwht(x: mx.array, block: int, signs: mx.array | None, inverse: bool = False) -> mx.array:
    """Blockwise Walsh-Hadamard transform with fixed signs, as the pack's runtime does it."""
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse and signs is not None:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(shape)
    if inverse and signs is not None:
        x = x * signs
    return x.astype(dtype)


class PackedEmbedding(nn.Module):
    """The rotated 2-bit embedding: dequantise the rows, then the inverse transform."""

    def __init__(self, arrays: Sequence[mx.array], block: int, signs: mx.array | None,
                 dtype: mx.Dtype) -> None:
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        if signs is not None:
            self.signs = signs
        self.block, self.dtype = block, dtype

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        indices = x.reshape(-1)
        out = (mx.dequantize(self.weight[indices], self.scales[indices], self.biases[indices],
                             group_size=128, bits=2).reshape(*shape, -1).astype(self.dtype))
        if not self.block:
            return out
        return fwht(out, self.block, getattr(self, "signs", None), inverse=True)


class PackedLinear(nn.QuantizedLinear):
    """A rotated 2-bit projection: the transform, then an ordinary affine quantised matmul.

    A subclass of ``nn.QuantizedLinear`` rather than the pack's plain module, and that is
    load-bearing: mlx-dspark's small-M verify kernel, its wide-GEMM prefill split and the
    calibration that derives the verify cap all recognise a projection by that class and
    read ``bits``, ``group_size`` and ``mode`` off it. ``__init__`` is bypassed because the
    stock one allocates and quantises a random weight; the arrays are set directly.
    The parent's ``__call__`` is looked up at call time, so mlx-dspark's class-level
    kernel routing applies to the matmul after the transform exactly as it does to any
    other quantised layer.
    """

    def __init__(self, arrays: Sequence[mx.array], block: int, signs: mx.array | None) -> None:
        nn.Module.__init__(self)
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        if signs is not None:
            self.signs = signs
        self.block = block
        self.group_size, self.bits, self.mode = 128, 2, "affine"
        self.freeze()

    def __call__(self, x: mx.array) -> mx.array:
        if self.block:
            x = fwht(x, self.block, getattr(self, "signs", None))
        return nn.QuantizedLinear.__call__(self, x)


def is_bonsai_pack(path: str) -> bool:
    try:
        with open(os.path.join(path, "config.json")) as f:
            return json.load(f).get("model_type") == MODEL_TYPE
    except (OSError, ValueError):
        return False


def load_bonsai(path: str) -> tuple[Any, Any]:
    """(model, tokenizer) for a Bonsai 2 pack directory, on mlx-lm's qwen3_5 module."""
    from mlx_lm.models.qwen3_5 import Model, ModelArgs
    from mlx_lm.utils import load_tokenizer

    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    if config.get("schema_version") not in (1, 2):
        raise ValueError(f"unsupported Bonsai pack schema {config.get('schema_version')}")
    base = config.get("base_model_type", "qwen3_5")
    # model_type is what mlx-dspark's Target routes its hidden-state tap on, so the model
    # carries the base family's name, not the pack's.
    model = Model(ModelArgs(model_type=base, text_config=config["text_config"]))
    weights = mx.load(os.path.join(path, "model.safetensors"))
    if not isinstance(weights, dict):
        raise TypeError(f"{path}/model.safetensors did not load as a tensor dictionary")
    lm = model.language_model
    prefix = "language_model." if any(k.startswith("language_model.") for k in weights) else ""
    seen = set()
    for record in config["modules"]:
        rel = record["path"]
        if rel in seen:
            raise ValueError(f"duplicate packed module {rel}")
        seen.add(rel)
        parts = rel.split(".")
        parent = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = prefix + rel
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        if record["dtype"] != "float16":
            raise ValueError("unsupported packed activation dtype")
        block = record["block"]
        if block and block not in (512, 1024, 2048, 4096):
            raise ValueError(f"unsupported Hadamard block {block}")
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError(f"missing sign vector for {rel}")
        module = (PackedEmbedding(arrays, block, signs, DTYPE) if record["embedding"]
                  else PackedLinear(arrays, block, signs))
        setattr(parent, parts[-1], module)
    # Language-model tensors only; the vision tower is not built.
    lm_weights = [(k, v) for k, v in weights.items()
                  if k.startswith(prefix) and not k.startswith("vision_tower")]
    model.load_weights(lm_weights, strict=True)
    # bfloat16 throughout, where the pack runs float16: mlx-dspark's small-M verify kernel
    # takes bfloat16 activations and nothing else, and mlx-lm's stock targets run bfloat16.
    # Scales, biases and signs are cast with the rest; the ternary levels survive the
    # 3 mantissa bits that costs, and the quality gate is what says so.
    model.set_dtype(DTYPE)
    model.eval()
    mx.eval(model.parameters())
    from pathlib import Path
    tokenizer = load_tokenizer(Path(path), tokenizer_config_extra={"trust_remote_code": False})
    return model, tokenizer


class BonsaiLoader(Patch):
    name = "Bonsai 2 Hadamard pack loader"

    def applied(self) -> bool:
        from mlx_dspark import load
        return getattr(load.load_target, "_bonsai2_bonsai", False)

    def apply(self) -> None:
        from mlx_dspark import load
        from mlx_dspark.target import Target

        orig = load.load_target

        def load_target(repo_or_path: str = load.DEFAULT_TARGET, *, require_tap: bool = False,
                        kv_bits: int | None = None, kv_group_size: int = 64) -> tuple[Any, Any]:
            path = load._resolve(repo_or_path)
            if not is_bonsai_pack(path):
                return orig(repo_or_path, require_tap=require_tap, kv_bits=kv_bits,
                            kv_group_size=kv_group_size)
            model, tokenizer = load_bonsai(path)
            target = Target(model, tokenizer, kv_bits=kv_bits, kv_group_size=kv_group_size)
            if require_tap:
                target.verify_tap()
            return target, tokenizer

        setattr(load_target, "_bonsai2_bonsai", True)  # noqa: B010 - a marker mypy would otherwise refuse
        load.load_target = load_target
        # The CLI and server imported the name at module load, so rebind it there too.
        for modname in ("mlx_dspark.cli", "mlx_dspark.server"):
            try:
                mod = __import__(modname, fromlist=["load_target"])
            except ImportError:
                continue
            if hasattr(mod, "load_target"):
                setattr(mod, "load_target", load_target)  # noqa: B010 - rebinding a module global
