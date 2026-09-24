#!/usr/bin/env python3
"""Served-acceptance reports from dflash-mlx-bonsai2's own speculative loop.

    python -u dflash_mlx_bonsai2_adapter.py --corpus CORPUS.jsonl --target PACK_DIR \\
        --drafter DRAFTER_DIR --max-new 200 --n 40 --report arm.json
    python -u dflash_mlx_bonsai2_adapter.py --corpus CORPUS.jsonl --target PACK_DIR \\
        --target-only --cap 4 --max-new 200 --n 40 --report target.json
    python dflash_mlx_bonsai2_adapter.py --self-test

Runs greedy decoding of a fixed prompt corpus through the runtime at
https://github.com/NakliTechie/dflash-mlx-bonsai2 (a fork of bstnxbt/dflash-mlx with a
loader for PrismML's Hadamard-rotated Bonsai 2 MLX pack and a 2-bit verify kernel), in
process, and writes one JSON report per arm. With --drafter the loop is the runtime's
`stream_dflash_generate` with fixed blocks (`verify_mode="dflash"`); with --target-only it
is the runtime's exact autoregressive path (`stream_baseline_generate`) on the same loaded
pack, which is the reference for output identity. Run it with the runtime's own Python
environment (the runtime installed with `pip install -e .`); it has no other dependencies.
Written against runtime commit 223e0f3 (dflash-mlx 0.1.10, mlx 0.32.2, mlx-lm 0.31.3).

Settings the adapter fixes, so that every arm of a comparison shares them: temperature 0,
`verify_mode="dflash"` (fixed blocks; the runtime's default is adaptive), copy speculation
off, an unquantized KV cache, and the verify kernel from --prism-verify, which is written
to DFLASH_PRISM_VERIFY before the pack loads (a conflicting exported value is refused).
The drafter is quantized at load per --draft-quant, default `w4` (4-bit, group 64): the
runtime's registry default for this target, which its registry applies only when the target
and drafter are given as repository ids, not as local directories.

Corpus: JSON lines with `prompt_ids` (token ids, already templated), `split`, `thinking`
and optionally `category`. The first --n rows of --split are used, in file order.

Report schema (one object):

    drafter    label of the drafter, or null for the target-only arm
    mode       "speculative" or "target_only"
    settings   everything two paired arms must share; paired analysis refuses reports
               whose settings differ, so per-arm details live under `arm`, not here
    arm        per-arm details (paths as given, drafter file hash, codebook naming, ...)
    complete   false until every prompt has been written
    requests   one record per prompt, in corpus order:
               prompt_sha256  sha256 of json.dumps(prompt_ids)
               category, thinking
               tokens         len(response_ids), counting the prefill's own token
               rounds         len(round_lengths)
               round_lengths  see "Rounds" below
               finish         "length" or "stop"
               response_ids   emitted token ids, cut after the first stop token
               decode_seconds wall time after prefill, as the runtime measures it
               prefill_seconds, prompt_tokens
               runtime        the loop's own per-pass record, unmapped

Rounds. The schema counts the token the prefill produces as a seed that belongs to no
round, and a round's length is the drafts it accepted plus the target's own token from
the same verify pass, so a run that stops on the budget emits `1 + sum(round_lengths)`
tokens. The runtime's loop is shifted by one against that: verify pass i checks the block
`[staged, d_1 .. d_k]`, accepts `a_i` drafts, commits `[staged, d_1 .. d_a_i]` and stages
the target's next token for pass i + 1 (the first staged token is the prefill's). The same
tokens come out; only the bookkeeping differs. The mapping, which `map_speculative`
checks against the loop's own arithmetic rather than assuming:

  * pass i is round i, with length `a_i + 1`;
  * the runtime shrinks the last block to the budget that remains and never emits the
    token the last pass stages, so when the run ends on the budget the final round's
    length is the part it emitted, `a_i` (the loop guarantees this fills the budget
    exactly, so nothing is dropped from `tokens`);
  * a final pass that only commits a token the previous pass already produced is not a
    round: a one-token block at the budget (no drafts), or a pass that starts on a stop
    token the previous pass produced. Such a pass is listed in `runtime.excluded_passes`;
  * on a stop, the final round keeps its full length even when the stop token cuts its
    emission short, and tokens the runtime committed after the stop token are dropped
    from `response_ids` (counted in `runtime.post_stop_tokens`).

`runtime.acceptance` is the loop's own acceptance history and `runtime.passes` its own
cycle count, so the runtime's native tokens-per-cycle (committed tokens over passes) can
always be recomputed from the report. The target-only arm records one round of length 1
per decode step after the seed.

`settings.cap` is the most drafts one block can carry, i.e. the effective block size minus
one as the runtime resolves it for the drafter, which can be smaller than --block-tokens:
the runtime clamps DFlash 2 drafters to its `max_block_tokens` capability. The target-only
arm has no drafter to resolve it from, so --cap must be given to pair it with drafter arms.

Identity. Greedy speculative decoding reproduces the target's own greedy output only up to
ties: the multi-row verify kernel and the one-row decode kernel accumulate in different
orders, so where the top two logits are equal at the model's output precision the two
paths can pick different tokens and continue differently from there. Compare arms of
this adapter against its --target-only arm with that in mind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRISM_VERIFY_MODES = ("v7", "v4b", "fp16", "off")


class MappingError(ValueError):
    """The runtime's record does not fit the loop arithmetic the mapping relies on."""


@dataclass(frozen=True)
class SchemaRun:
    response_ids: list[int]
    finish: str
    round_lengths: list[int]
    excluded_passes: list[int]
    block_lens: list[int]
    post_stop_tokens: int


def _first_stop(tokens: Sequence[int], stop_ids: Iterable[int]) -> int | None:
    stops = {int(t) for t in stop_ids}
    for index, token in enumerate(tokens):
        if int(token) in stops:
            return index
    return None


def map_speculative(
    generated: Sequence[int],
    acceptance: Sequence[int],
    block_tokens: int,
    budget: int,
    stop_ids: Iterable[int],
) -> SchemaRun:
    """Map one run of the runtime's fixed-block loop onto the report's rounds.

    `generated` is the loop's committed token list (its summary's `generated_token_ids`),
    `acceptance` its per-pass accepted-draft counts, `block_tokens` the effective block.
    """
    if block_tokens < 1 or budget < 1:
        raise MappingError("block_tokens and budget must be positive")
    if not acceptance:
        raise MappingError("the run has no verify passes")
    if sum(1 + int(a) for a in acceptance) != len(generated):
        raise MappingError(
            f"passes committed {sum(1 + int(a) for a in acceptance)} tokens but the run "
            f"reports {len(generated)}")
    block_lens: list[int] = []
    committed = 0
    for index, accepted in enumerate(acceptance):
        remaining = budget - committed
        if remaining <= 0:
            raise MappingError(f"pass {index} began at or past the budget")
        block_len = min(block_tokens, remaining)
        if not 0 <= int(accepted) < block_len:
            raise MappingError(
                f"pass {index} accepted {accepted} drafts from a block of {block_len}")
        block_lens.append(block_len)
        committed += 1 + int(accepted)

    stop_at = _first_stop(generated, stop_ids)
    if stop_at is None:
        if committed != budget:
            raise MappingError(
                f"the run ended at {committed} tokens, below the {budget}-token budget, "
                "without a stop token")
        response, finish = list(generated), "length"
    else:
        last_start = committed - (1 + int(acceptance[-1]))
        if not last_start <= stop_at < committed:
            raise MappingError("the first stop token was not committed by the final pass")
        response, finish = list(generated[: stop_at + 1]), "stop"

    emitted = len(response)
    lengths: list[int] = []
    excluded: list[int] = []
    start = 1  # 1-based position of the first token pass i commits
    for index, accepted in enumerate(acceptance):
        if start < emitted:
            produced = int(accepted) + 1
            if finish == "length":
                produced = min(produced, emitted - start)
            lengths.append(produced)
        else:
            excluded.append(index)
        start += 1 + int(accepted)

    if excluded:
        if excluded != [len(acceptance) - 1]:
            raise MappingError(f"passes {excluded} emit nothing but are not the final pass")
        if finish == "length" and block_lens[excluded[0]] != 1:
            raise MappingError("a drafting pass at the budget emitted nothing")
    if not lengths:
        raise MappingError("no rounds: the prefill's own token ended the run")
    if finish == "length" and 1 + sum(lengths) != emitted:
        raise MappingError("rounds do not account for the emitted tokens")
    if finish == "stop" and not 1 <= emitted - (1 + sum(lengths[:-1])) <= lengths[-1]:
        raise MappingError("the final round cannot have emitted the tokens after its start")
    return SchemaRun(
        response_ids=[int(t) for t in response],
        finish=finish,
        round_lengths=lengths,
        excluded_passes=excluded,
        block_lens=block_lens,
        post_stop_tokens=len(generated) - emitted,
    )


def map_autoregressive(
    generated: Sequence[int], budget: int, stop_ids: Iterable[int]
) -> SchemaRun:
    """Map one run of the runtime's exact autoregressive path: one token per step."""
    stop_at = _first_stop(generated, stop_ids)
    if stop_at is None:
        if len(generated) != budget:
            raise MappingError(
                f"{len(generated)} tokens without a stop token, budget {budget}")
        finish = "length"
    else:
        if stop_at != len(generated) - 1:
            raise MappingError("the autoregressive path continued past a stop token")
        finish = "stop"
    if len(generated) < 2:
        raise MappingError("no rounds: the run emitted only the prefill's own token")
    return SchemaRun(
        response_ids=[int(t) for t in generated],
        finish=finish,
        round_lengths=[1] * (len(generated) - 1),
        excluded_passes=[],
        block_lens=[],
        post_stop_tokens=0,
    )


def _self_test() -> None:
    eos = [99]
    # Budget 10, block 5: two full passes fill it; the last pass's staged token is not emitted.
    run = map_speculative(list(range(10)), [4, 4], 5, 10, eos)
    assert run.round_lengths == [5, 4] and run.finish == "length" and not run.excluded_passes
    # One token short of the budget: the runtime spends a one-token pass to commit it.
    run = map_speculative(list(range(10)), [4, 3, 0], 5, 10, eos)
    assert run.round_lengths == [5, 4] and run.excluded_passes == [2]
    assert run.block_lens == [5, 5, 1]
    # A stop token among the accepted drafts of the final pass.
    run = map_speculative([1, 2, 3, 4, 99, 6, 7], [2, 3], 5, 20, eos)
    assert run.response_ids == [1, 2, 3, 4, 99] and run.round_lengths == [3, 4]
    assert run.finish == "stop" and run.post_stop_tokens == 2
    # A stop token produced by the target: the next pass starts on it and is not a round.
    run = map_speculative([1, 2, 3, 99, 5], [2, 1], 5, 20, eos)
    assert run.response_ids == [1, 2, 3, 99] and run.round_lengths == [3]
    assert run.excluded_passes == [1]
    # The shrunken final block: 3 tokens left, the block is 3 and fills them.
    run = map_speculative(list(range(8)), [4, 2], 5, 8, eos)
    assert run.block_lens == [5, 3] and run.round_lengths == [5, 2]
    for bad in (([4, 4], 11), ([5], 6)):
        try:
            map_speculative(list(range(sum(1 + a for a in bad[0]))), bad[0], 5, bad[1], eos)
        except MappingError:
            continue
        raise AssertionError(f"accepted an impossible run {bad}")
    run = map_autoregressive([1, 2, 3], 3, eos)
    assert run.round_lengths == [1, 1] and run.finish == "length"
    run = map_autoregressive([1, 2, 99], 10, eos)
    assert run.finish == "stop" and run.round_lengths == [1, 1]
    print("self-test passed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_cache_label(path: Path) -> str:
    """`org/name@rev` for a Hugging Face cache snapshot directory, else the directory name."""
    parts = path.resolve().parts
    if len(parts) >= 3 and parts[-2] == "snapshots" and parts[-3].startswith("models--"):
        repo = parts[-3][len("models--"):].replace("--", "/", 1)
        return f"{repo}@{parts[-1]}"
    return path.name


def _safetensors_names(model_dir: Path) -> list[str]:
    names: list[str] = []
    for file in sorted(model_dir.glob("*.safetensors")):
        with file.open("rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(size))
        names.extend(name for name in header if name != "__metadata__")
    return sorted(names)


def _codebook_naming(names: Sequence[str]) -> str:
    bare = {f"candidate_selector.{n}_codebook" for n in ("predecessor", "successor")}
    suffixed = {f"{name}.weight" for name in bare}
    present = set(names)
    if bare <= present and not (suffixed & present):
        return "bare (candidate_selector.*_codebook)"
    if suffixed <= present and not (bare & present):
        return "embedding (candidate_selector.*_codebook.weight)"
    return "mixed or absent"


def _installed_verify_mode(model: Any) -> str | None:
    """The verify kernel the pack loader patched into its packed modules, if any."""
    for _, module in model.named_modules():
        mode = getattr(type(module), "_dflash_verify_mode", None)
        if mode is not None:
            return str(mode)
    return None


def _git_head(directory: Path) -> str | None:
    """The checked-out commit, read from the repository files (no git process, so the
    value cannot depend on the environment); None when the runtime is not a checkout."""
    git = directory / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[len("ref: "):]
        loose = git / ref
        if loose.is_file():
            return loose.read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split(" ", 1)[0]
    except OSError:
        return None
    return None


def _git_dirty(directory: Path) -> bool | None:
    """Best effort: tracked files modified in the runtime checkout (None if unknown)."""
    try:
        out = subprocess.run(["git", "-C", str(directory), "status", "--porcelain",
                              "--untracked-files=no"],
                             capture_output=True, text=True, check=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip())


def _load_rows(corpus: Path, split: str, n: int) -> list[dict[str, Any]]:
    with corpus.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("split") == split][:n]
    if len(rows) != n:
        raise SystemExit(f"requested {n} prompts but found {len(rows)} in split {split!r}")
    for row in rows:
        ids = row.get("prompt_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(t, int) for t in ids):
            raise SystemExit("every corpus row needs a non-empty integer `prompt_ids` list")
        if not isinstance(row.get("thinking"), bool):
            raise SystemExit("every corpus row needs a boolean `thinking`")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="check the round mapping on synthetic runs and exit (no model)")
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--split", default="eval")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--target", type=Path, help="local directory of the PrismML MLX pack")
    ap.add_argument("--target-label", help="label recorded in settings (default: derived)")
    arm = ap.add_mutually_exclusive_group()
    arm.add_argument("--drafter", type=Path, help="local DFlash 2 drafter directory")
    arm.add_argument("--target-only", action="store_true",
                     help="the runtime's exact autoregressive path, no drafter")
    ap.add_argument("--drafter-label", help="label recorded as `drafter` (default: derived)")
    ap.add_argument("--draft-quant", default="w4",
                    help="runtime draft quantization spec, e.g. w4 or w4:gs64; 'none' = bf16")
    ap.add_argument("--block-tokens", type=int, default=8,
                    help="requested block; the runtime may clamp it (see settings.block_tokens)")
    ap.add_argument("--cap", type=int,
                    help="drafts per block; required with --target-only, checked otherwise")
    ap.add_argument("--prism-verify", choices=PRISM_VERIFY_MODES, default="v7",
                    help="verify-path kernel for the pack (DFLASH_PRISM_VERIFY)")
    ap.add_argument("--prefill-step-size", type=int, default=None,
                    help="runtime prefill chunk (default: the runtime's offline default)")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args(argv)
    if args.self_test:
        return args
    missing = [flag for flag, value in (("--corpus", args.corpus), ("--target", args.target),
                                        ("--report", args.report)) if value is None]
    if missing:
        ap.error("required: " + ", ".join(missing))
    if args.drafter is None and not args.target_only:
        ap.error("give --drafter DIR or --target-only")
    if args.target_only and args.cap is None:
        ap.error("--target-only needs --cap, the drafter arms' cap, to pair with them")
    if args.n <= 0 or args.max_new <= 0 or args.block_tokens <= 0:
        ap.error("--n, --max-new and --block-tokens must be positive")
    if not (args.target / "config.json").is_file():
        ap.error(f"--target must be a local pack directory with config.json: {args.target}")
    if args.drafter is not None and not (args.drafter / "config.json").is_file():
        ap.error(f"--drafter must be a local directory with config.json: {args.drafter}")
    env_verify = os.environ.get("DFLASH_PRISM_VERIFY")
    if env_verify is not None and env_verify.lower() != args.prism_verify:
        ap.error(f"DFLASH_PRISM_VERIFY={env_verify} conflicts with --prism-verify "
                 f"{args.prism_verify}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    # The pack loader reads the verify kernel choice from the environment at load time.
    os.environ["DFLASH_PRISM_VERIFY"] = args.prism_verify
    rows = _load_rows(args.corpus, args.split, args.n)

    from importlib import metadata

    import dflash_mlx
    import mlx.core as mx
    from dflash_mlx.engine.config import resolve_speculative_cycle_config
    from dflash_mlx.engine.events import PrefillCompleteEvent, SummaryEvent, TokenEvent
    from dflash_mlx.engine.fallback import stream_baseline_generate
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context
    from dflash_mlx.runtime.loading import load_target_bundle

    context = build_offline_runtime_context(
        quantize_kv_cache=False,
        prefill_step_size=args.prefill_step_size,
        verify_mode="dflash",
        copyspec_mode="off",
    )
    runtime = context.runtime
    draft_quant = None if args.draft_quant.strip().lower() == "none" else args.draft_quant.strip()

    arm: dict[str, Any] = {"target_path": str(args.target),
                           "target_config_sha256": _sha256_file(args.target / "config.json")}
    started = time.time()
    if args.target_only:
        target_bundle = load_target_bundle(args.target, lazy=True, quantize_kv_cache=False,
                                           verify_config=context.verify)
        target_model, tokenizer = target_bundle.model, target_bundle.tokenizer
        target_ops, target_meta = target_bundle.target_ops, target_bundle.meta
        draft_model = draft_backend = None
        block_tokens = args.cap + 1
        drafter_label = None
    else:
        names = _safetensors_names(args.drafter)
        arm.update({
            "drafter_path": str(args.drafter),
            "drafter_tensors": len(names),
            "drafter_codebook_naming": _codebook_naming(names),
            "drafter_safetensors_sha256": {
                f.name: _sha256_file(f) for f in sorted(args.drafter.glob("*.safetensors"))},
        })
        bundle = load_runtime_bundle(model_ref=args.target, draft_ref=str(args.drafter),
                                     draft_quant=draft_quant or "none",
                                     verify_config=context.verify, quantize_kv_cache=False)
        target_model, tokenizer = bundle.target_model, bundle.tokenizer
        target_ops, target_meta = bundle.target_ops, bundle.target_meta
        draft_model, draft_backend = bundle.draft_model, bundle.draft_backend
        cycle = resolve_speculative_cycle_config(runtime, draft_model, args.block_tokens)
        if cycle.verify_len_cap != cycle.effective_block_tokens:
            raise SystemExit("a verify length cap below the block is not supported")
        block_tokens = cycle.effective_block_tokens
        if args.cap is not None and args.cap != block_tokens - 1:
            raise SystemExit(f"the runtime resolved cap {block_tokens - 1}, not --cap {args.cap}")
        arm.update({
            "drafter_loaded": "strict load succeeded",
            "draft_block_size": int(cycle.draft_block_size),
            "effective_draft_quant": bundle.effective_draft_quant,
            "draft_meta": {k: v for k, v in bundle.draft_meta.items() if k != "config"},
        })
        drafter_label = args.drafter_label or _hf_cache_label(args.drafter)
    runtime_dir = Path(dflash_mlx.__file__).resolve().parent.parent
    arm.update({
        "runtime_dirty": _git_dirty(runtime_dir),
        "load_seconds": time.time() - started,
        "prism_verify_installed": _installed_verify_mode(target_model),
        "verify_linear_enabled": target_meta.get("verify_linear_enabled"),
        "target_family": target_meta.get("target_family"),
    })
    stop_ids = [int(t) for t in get_stop_token_ids(tokenizer)]

    settings: dict[str, Any] = {
        "runtime": "dflash-mlx-bonsai2",
        "runtime_version": metadata.version("dflash-mlx"),
        "runtime_commit": _git_head(runtime_dir),
        "mlx": metadata.version("mlx"),
        "mlx_lm": metadata.version("mlx-lm"),
        "target": args.target_label or _hf_cache_label(args.target),
        "corpus_sha256": _sha256_file(args.corpus),
        "split": args.split,
        "max_new": args.max_new,
        "cap": block_tokens - 1,
        "block_tokens_requested": args.block_tokens,
        "block_tokens": block_tokens,
        "verify_mode": runtime.verify_mode,
        "prism_verify": args.prism_verify,
        "draft_quant": draft_quant or "none",
        "kv_bits": None,
        "copyspec": runtime.copyspec_mode,
        "prefill_step_size": int(runtime.prefill_step_size),
        "draft_sink_size": int(runtime.draft_sink_size),
        "draft_window_size": int(runtime.draft_window_size),
        "stop_token_ids": stop_ids,
        "temperature": 0.0,
    }
    report: dict[str, Any] = {
        "drafter": drafter_label,
        "mode": "target_only" if args.target_only else "speculative",
        "settings": settings,
        "arm": arm,
        "complete": False,
        "requests": [],
    }
    _write_json(args.report, report)
    print(f"loaded in {arm['load_seconds']:.1f}s; block {block_tokens} "
          f"(requested {args.block_tokens}); stop ids {stop_ids}", flush=True)

    totals = {"tokens": 0, "rounds": 0, "committed": 0, "passes": 0, "decode": 0.0}
    for index, row in enumerate(rows):
        prompt_ids = [int(t) for t in row["prompt_ids"]]
        if args.target_only:
            stream = stream_baseline_generate(
                target_model=target_model, target_ops=target_ops, tokenizer=tokenizer,
                prompt="", max_new_tokens=args.max_new, use_chat_template=False,
                stop_token_ids=stop_ids, prompt_tokens_override=prompt_ids,
                quantize_kv_cache=False)
        else:
            stream = stream_dflash_generate(
                target_model=target_model, target_ops=target_ops, tokenizer=tokenizer,
                draft_model=draft_model, draft_backend=draft_backend, prompt="",
                max_new_tokens=args.max_new, use_chat_template=False,
                block_tokens=args.block_tokens, stop_token_ids=stop_ids,
                prompt_tokens_override=prompt_ids, quantize_kv_cache=False,
                publish_generation_snapshot=False, runtime_context=context)
        streamed: list[int] = []
        summary: SummaryEvent | None = None
        prefill: PrefillCompleteEvent | None = None
        try:
            for event in stream:
                if isinstance(event, TokenEvent):
                    streamed.append(int(event.token_id))
                elif isinstance(event, SummaryEvent):
                    summary = event
                elif isinstance(event, PrefillCompleteEvent):
                    prefill = event
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        if summary is None or prefill is None:
            raise SystemExit(f"prompt {index}: the runtime produced no summary")
        generated = [int(t) for t in summary.generated_token_ids]
        if streamed != generated:
            raise SystemExit(f"prompt {index}: streamed tokens disagree with the summary")
        if bool(summary.fallback_ar) != bool(args.target_only):
            raise SystemExit(f"prompt {index}: unexpected fallback_ar={summary.fallback_ar} "
                             f"({summary.fallback_reason})")
        acceptance = [int(a) for a in summary.acceptance_history]
        if args.target_only:
            mapped = map_autoregressive(generated, args.max_new, stop_ids)
        else:
            if int(summary.block_tokens or 0) != block_tokens:
                raise SystemExit(f"prompt {index}: block {summary.block_tokens} != {block_tokens}")
            mapped = map_speculative(generated, acceptance, block_tokens, args.max_new, stop_ids)
        prefill_us = float(summary.phase_timings_us.get("prefill", prefill.prefill_us))
        decode_seconds = max(0.0, float(summary.elapsed_us) - prefill_us) / 1e6
        record = {
            "prompt_sha256": hashlib.sha256(json.dumps(row["prompt_ids"]).encode()).hexdigest(),
            "category": row.get("category", "unknown"),
            "thinking": row["thinking"],
            "tokens": len(mapped.response_ids),
            "rounds": len(mapped.round_lengths),
            "round_lengths": mapped.round_lengths,
            "finish": mapped.finish,
            "response_ids": mapped.response_ids,
            "decode_seconds": decode_seconds,
            "prefill_seconds": prefill_us / 1e6,
            "prompt_tokens": len(prompt_ids),
            "runtime": {
                "passes": int(summary.cycles_completed),
                "committed_tokens": len(generated),
                "acceptance": acceptance,
                "block_lens": mapped.block_lens,
                "excluded_passes": mapped.excluded_passes,
                "post_stop_tokens": mapped.post_stop_tokens,
                "tokens_per_cycle": float(summary.tokens_per_cycle),
                "peak_memory_gb": summary.peak_memory_gb,
            },
        }
        report["requests"].append(record)
        _write_json(args.report, report)
        totals["tokens"] += record["tokens"]
        totals["rounds"] += record["rounds"]
        totals["committed"] += len(generated)
        totals["passes"] += int(summary.cycles_completed)
        totals["decode"] += decode_seconds
        print(f"[{index + 1}/{len(rows)}] think={int(row['thinking'])} {record['tokens']} tok "
              f"{record['rounds']} rounds {record['tokens'] / max(1, record['rounds']):.3f} "
              f"tok/round ({mapped.finish}; {summary.cycles_completed} passes) "
              f"{decode_seconds:.1f}s", flush=True)
        mx.clear_cache()

    report["complete"] = True
    report["summary"] = {
        "tokens_per_round": totals["tokens"] / max(1, totals["rounds"]),
        "runtime_tokens_per_cycle": (totals["committed"] / totals["passes"]
                                     if totals["passes"] else None),
        "decode_tokens_per_second": totals["tokens"] / max(totals["decode"], 1e-9),
        "wall_seconds": time.time() - started,
    }
    _write_json(args.report, report)
    print(f"DONE {report['drafter'] or 'target-only'}: {totals['tokens']} tok over "
          f"{totals['rounds']} rounds = {report['summary']['tokens_per_round']:.4f} tok/round; "
          f"runtime {report['summary']['runtime_tokens_per_cycle']} tok/cycle", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
