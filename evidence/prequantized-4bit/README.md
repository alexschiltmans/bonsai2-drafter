# Prequantized 4-bit variant: equivalence

The hardware validation behind the 4-bit card's "Served equivalence" paragraph: the bf16 ft5
drafter quantized at load ("source") against its prequantized export ("artifact"), on
mlx-dspark 0.18.0 with this repository's patches, 8-bit KV, cap 7, one Apple M4 Pro with 48 GB.
The artifact tested here has the same `model.safetensors` bytes as the published
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit` under a different metadata layout; the
release changed only `config.json`, as the card states. In these files it is named
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit (pre-release export)`. The released
files themselves were loaded by two further loaders; `second-loaders/` has those checks.

## Files

| path | what it is |
|---|---|
| `served/{code,general,long}-1-source.json`, `-2-artifact.json` | six `served_accept.py --report` records: code and general at 200 tokens, general at 1024, each with `round_lengths` and full `response_ids` |
| `gates/acceptance-equivalence.json`, `gates/long-equivalence.json` | `artifact-equivalence/v1` on the 200-token pairs and the 1024-token pair, with the improvement analysis beside it for reference only (it is expected to say `investigate`: zero gain is the right answer for a repackaging) |
| `gates/throughput-gate.json` | pooled decode rate per arm over its two ABBA legs, artifact/source ratio, threshold 0.95 |
| `gates/quality-gate.json` | no newly failing fixture, tool tallies in both arms |
| `http/greedy-leg{1-source,2-artifact,3-artifact,4-source}.json` | the four HTTP throughput legs, ABBA, three repetitions of five prompts at 400 tokens (the gate names them `http-1-source` … `http-4-source`). Written by an unpublished version of the author's harness: its five prompts are those of `bench/throughput/bench5.py`, its record layout is not (`../README.md`) |
| `http/discarded-first-attempt-leg1-source.json` | **discarded, not used by any gate.** The first attempt's leg 1, 26.79 tok/s pooled. The run stopped after it on a false configuration-check failure (the supervisor compared an anonymised drafter path against an absolute one); all four legs were then re-run, decided before any artifact leg was measured, so the ABBA order has no gap. Kept so the discarded number is visible, not hidden |
| `quality/quality-{source,artifact}.json`, `quality/tools-{source,artifact}.json` | the unpublished harness's quality battery and tool battery (depths 2500 and 13000, four repetitions) in each arm. Neither battery (`quality.py`, `tool_battery.py`) is in this repository |
| `second-loaders/mlx-lm-load-model.json`, `second-loaders/dflash-mlx-bonsai2.json`, `second-loaders/reference-digest.json` | the released export loaded by mlx-lm's `load_model` and by dflash-mlx-bonsai2, each compared tensor by tensor with this repository's loader; see below |

## Claims and how to reproduce them

The card: "the bf16 and 4-bit packagings produced the same served loop: the same tokens, finish
reasons, token counts, rounds and round lengths (contract `artifact-equivalence/v1`). HTTP decode
ran ABBA at a throughput ratio of 1.0001, with no newly failing quality fixture and 56/56 tool
calls in both arms." From the repository root:

```sh
S=evidence/prequantized-4bit/served
python3 bench/analysis/analyse_served_accept.py $S/code-1-source.json    $S/code-2-artifact.json    --equivalence
python3 bench/analysis/analyse_served_accept.py $S/general-1-source.json $S/general-2-artifact.json --equivalence
python3 bench/analysis/analyse_served_accept.py $S/long-1-source.json    $S/long-2-artifact.json    --equivalence
```

Rerun on these sanitized files, each prints `"gate": "pass"` with 40/40 prompts equal in every
field and exits 0. Tokens a round are identical in both arms: code 4.2130, general 3.0403,
general at 1024 tokens 3.0399, the same values as ft5's arms in `../ft5-publication-battery/`.

Throughput, total completion tokens over total decode seconds per arm across its two legs:

```sh
python3 - <<'PY'
import json
H = "evidence/prequantized-4bit/http/greedy-leg"
def pooled(*legs):
    rows = [r for leg in legs for r in json.load(open(f"{H}{leg}.json"))["requests"]]
    return sum(r["completion_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows)
source, artifact = pooled("1-source", "4-source"), pooled("2-artifact", "3-artifact")
print(f"source {source:.4f}  artifact {artifact:.4f}  ratio {artifact / source:.4f}")
PY
```

prints `source 28.8301  artifact 28.8341  ratio 1.0001` (legs 28.870, 28.826, 28.843, 28.790;
11,460 tokens per arm; every request at cap 7), matching `gates/throughput-gate.json`. Quality:
13/14 in both arms with `iso_seconds` failing in both (the disclosed 8-bit KV non-termination),
none newly failing, and 56/56 tool calls in both (`gates/quality-gate.json`).

## The released files in two other loaders

The 4-bit card: "mlx-lm's `load_model`, with a DFlash 2 model class: **loads**, all 153 tensors
identical to the patched loader, the same 36 modules at 4 bits" and "dflash-mlx-bonsai2
(`load_draft_bundle(path, draft_quant=None)`): **loads**, all 153 tensors identical". The
export these checks loaded is named
`Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5-mlx-4bit (release export)`: its `config.json` and
`model.safetensors` have the same sha256 as the files at the Hub revision
`c0db7e148bb2fe9782f85ea06c2b438b2a47c326` (`e2ccea1eefc7add1ed8de4db9f7d2cc64b0da11c938c1c4c8a034b2a32e9b99b`
and `3cfd72ea84f09d3392e5486b8a4aec083a82709b070c60a8ef0c55b38ee2c574`).

- `reference-digest.json`: the artifact as this repository's loader (the patched mlx-dspark
  load path) builds it: dtype, shape and sha256 of each of the 153 parameters, and the class
  and quantization of each module.
- `mlx-lm-load-model.json`: `A_project_loader` is that loader (153 tensors, 36 quantized
  modules). `B_mlx_lm_load_model` loads the same files with mlx-lm's `load_model` and
  mlx-dspark's DFlash 2 model class: 153 tensors, no tensor or module mismatch,
  `identical: true`. `C_explicit_entries_only` repeats B deciding quantization from the
  `quantization` block's explicit entries alone: also identical. `D_mlx_lm_on_old_artifact`
  loads the pre-release export the same way and fails as expected (72 parameters not in the
  model), since that export has no `quantization` block.
- `dflash-mlx-bonsai2.json`: naklitechie's runtime at `223e0f3a9cb4806da0cdc5190f9191b545d1f60b`,
  `load_draft_bundle(path, draft_quant=None)`, compared with `reference-digest.json` after
  mapping the codebooks' `.weight` names back to the bare names: `DFlash2DraftModel`, 153
  tensors, no mismatch, the quantized modules equal at `[4, 64, "affine"]`, and the 13 matrices
  the artifact keeps in bf16 left in bf16. With `w4` quantization on top
  (`new_artifact_with_w4_on_top`) the runtime also packs those 13, codebooks included, which is
  why the card says not to pass `--draft-quant`. The pre-release export fails here too.

These three files were written by two short check scripts that are not in this repository; the
records name the loaders they called. The mlx-lm comparison is also part of the model tier of
`bench/tests/test_dflash_prequantized.py`, which reruns it against local copies of the drafters.

To read the headline fields, from the repository root:

```sh
python3 - <<'PY'
import json
L = "evidence/prequantized-4bit/second-loaders/"
m, d = json.load(open(L + "mlx-lm-load-model.json")), json.load(open(L + "dflash-mlx-bonsai2.json"))
print("reference", len(json.load(open(L + "reference-digest.json"))["digest"]), "tensors")
print("project  ", m["A_project_loader"])
for k in ("B_mlx_lm_load_model", "C_explicit_entries_only"):
    print(k, {f: m[k][f] for f in ("tensors", "n_tensor_mismatches", "n_module_mismatches", "identical")})
a = d["new_artifact"]
print("dflash-mlx", {f: a[f] for f in ("model_class", "tensors", "n_tensor_mismatches",
                                       "quantized_modules_equal", "quantized_params", "identical")},
      len(a["bf16_quantizable_modules"]), "bf16 modules")
PY
```

## What was stripped

As in `../ft5-publication-battery/`: run timestamps, including one epoch time of a memory-guard
event (`last_shed.at`) inside a tool record; the private repository revision; the host process
list; the server log path; calibration cache keys; dated labels and file names. In strings, the
local drafter directories became the public names above, launcher paths became
`python bin/mlx-dspark-patched`, and the local API key became `REDACTED`. Every number is
unchanged; `../tools/verify_sanitized.py` proved it pair by pair. In `second-loaders/`, the
local paths of the two exports became the public name with "(release export)" or
"(pre-release export)", and the runtime file path became its path inside naklitechie's
repository (`dflash-mlx-bonsai2/dflash_mlx/__init__.py`); nothing was removed.

## What was left out

The artifact itself (published on the Hub), the export and tensor-equality logs, the run's
supervisor status files and snapshot manifest (keyed by local paths; mostly commands and
machine-state dumps), the server and stage logs. The card's tensor-equality result is a test
log, not a measurement record, and is not here; `bench/tests/test_dflash_prequantized.py`
with `--source` and `--artifact` reruns that check against local copies of both drafters.
