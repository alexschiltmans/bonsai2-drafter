# Fork probe: DFlash 2 and MTP on PrismML's llama.cpp fork (Metal)

A feasibility probe, run once, behind the card's compatibility row "PrismML llama.cpp fork,
Metal: no speedup possible today". Three arms on one Apple M4 Pro with 48 GB, each an ABBA pair
of `bench5.py` legs (three repetitions of five prompts at 400 tokens, fresh server per leg), in
two contiguous groups:

1. **Unmodified fork**: no drafter against MTP.
2. **Fork with upstream DFlash 2 patched in**: no drafter against the stock DFlash 2 drafter.

The predeclared gate: DFlash 2 must beat both no drafter and MTP by more than `bench5.py`'s
repetition spread. It failed, so no ft5 GGUF was made and none is published.

## What ran

| | |
| --- | --- |
| **Fork** | [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp), branch `prism`, head `0324c66521960d67aa7da8687fb1453a79a6565c`; the binary reports build `b10731-0324c665` |
| **DFlash 2 patch** | `lab/leg9/patches/0001-spec-add-DFlash2-support-local-convolution-candidate.patch` from [NakliTechie/dflash-mlx-bonsai2](https://github.com/NakliTechie/dflash-mlx-bonsai2) at `223e0f3a9cb4806da0cdc5190f9191b545d1f60b` (sha256 `b111b88959b33a27ce38480275519dce5b431c5fc5913df1ab6deac607e99a79`). It is upstream llama.cpp commit `8f29f159cb09e94f9ddd08881535c1c88e4422e1` by Xuan-Son Nguyen (MIT), and applied cleanly to the fork head with `git apply`. The patched binary reports build `b10732-patched` (the local commit that held the applied patch was never published, so its hash is omitted). That repository's other `0001` patch (Hadamard transforms) was not stacked: the fork already carries its own version |
| **Build flags** (both builds) | `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_USE_BF16=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF`, targets `llama-server` and `llama-cli`; the macOS arm64 defines of the fork's release workflow |
| **Target** | `prism-ml/Ternary-Bonsai-2-27B-gguf` at `6ed5e12bf84b7a63069882c91dd9e9218647d17b`, `Ternary-Bonsai-2-27B-PQ2_0.gguf` |
| **MTP** | `decent-jawfish/bonsai-2-27b-mtp` at `5edf5f552d45e40b81f0255a8bb443af35850722`, `Bonsai-2-27B-PQ2_0-MTP.gguf` (Apache-2.0), `--spec-type draft-mtp --spec-draft-n-max 2` |
| **DFlash 2 drafter** | `z-lab/Qwen3.8-27B-DFlash2-GGUF` at `2d9571f8ce46e151f61c6499c99dee6079e1d610`, `Qwen3.8-27B-DFlash2-Q8_0.gguf` (Apache-2.0), `-md … --spec-type draft-dflash --spec-draft-n-max 7 -ngld 99` |
| **Server flags** (every arm) | `-ngl 99 -fa on -c 131072 -np 1 --jinja --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --metrics`, text only; each record's `provenance.launch` has the exact line. `bench5.py` requests greedy decoding, so both arms of a pair emit the same 11,412 tokens |

## Files

| path | what it is |
|---|---|
| `bench5/head-nodrafter-a{1,2}.json`, `bench5/head-mtp-b{1,2}.json` | group 1, run in the order A1, B1, B2, A2 |
| `bench5/df2build-nodrafter-a{1,2}.json`, `bench5/df2build-dflash2stock-b{1,2}.json` | group 2, same order |
| `results-head.json`, `results-df2build.json` | per arm: pooled decode rate, tokens, decode seconds, repetition spread, and the draft acceptance counts parsed from each leg's server log (warm-up request excluded) |
| `trunk-hash.json` | the MTP GGUF's trunk against Prism's GGUF, tensor by tensor: type, shape and sha256 of the raw bytes |

## Claims and how to reproduce them

The card: "with upstream's patched in, the stock DFlash 2 drafter decoded at **7.10 tok/s
against 20.56 with no drafter (and MTP at 11.70)**. A DFlash 2 round cost about ten
single-token steps there." From the repository root:

```sh
python3 - <<'PY'
import json
B = "evidence/fork-probe/bench5/"
def pooled(*legs):
    rows = [r for leg in legs for r in json.load(open(f"{B}{leg}.json"))["requests"]]
    return sum(r["completion_tokens"] for r in rows) / sum(r["decode_seconds"] for r in rows)
for arm, legs in [("no drafter, unmodified fork", ("head-nodrafter-a1", "head-nodrafter-a2")),
                  ("MTP, unmodified fork", ("head-mtp-b1", "head-mtp-b2")),
                  ("no drafter, patched fork", ("df2build-nodrafter-a1", "df2build-nodrafter-a2")),
                  ("DFlash 2 stock, patched fork", ("df2build-dflash2stock-b1", "df2build-dflash2stock-b2"))]:
    print(f"{arm:30s} {pooled(*legs):6.2f} tok/s")
PY
```

prints 20.60, 11.70, 20.56 and 7.10 tok/s, matching `results-*.json`. Legs: no drafter 20.70
and 20.51 (spread 1.17%), MTP 11.70 and 11.69 (0.15%); patched no drafter 20.63 and 20.48
(0.89%), DFlash 2 7.10 and 7.10 (0.08%). The patch does not slow the plain path (20.56 against
20.60). DFlash 2 acceptance was 8040 of 23232 drafts (0.346), MTP's 6126 of 10476 (0.585).

The round cost: a DFlash 2 leg emitted 5706 tokens with 4020 accepted drafts, so 1686 rounds,
in 803.5 seconds of decode: about 476 ms a round, against 48.65 ms a token with no drafter, or
9.8 single-token steps. Even a round that accepted all seven drafts (eight tokens) would then
run at 8 / 0.476 = 16.8 tok/s, below no drafter; that is why the card says no drafter can win
on that path yet. The trunk check: all 851 tensors of Prism's GGUF are present in the MTP GGUF
with identical type, shape and bytes (aggregate sha256 identical); the MTP file adds 15
`blk.64.*` tensors, and its metadata differs only in `qwen35.block_count` (64 → 65) and
`qwen35.nextn_predict_layers` (absent → 1). So the MTP arm runs Prism's trunk.

`analyse_served_accept.py` does not apply here: `bench5.py` records per-request totals over
HTTP, not per-prompt token arrays, and no acceptance claim is made from this probe beyond the
counts above.

## Reading the records

`provenance.server.mode` reads "llama.cpp, no drafter" in every leg, because the harness cannot
see llama.cpp's speculative flags; the arm is given by `provenance.launch` and
`provenance.props`. `provenance.versions` lists the harness's own pinned Python packages and its
default llama.cpp pin, not the binary under test; the binary is identified by
`provenance.health.build` and the launch path (`build-head/` or `build-df2/`).

## What was stripped

Run timestamps, including the clock time in the machine's uptime line; the private repository
revision; the host process list; dated labels and date-prefixed file names (records are named
by group, arm and leg); the local path parts of the server binary, model and drafter paths
(model files now read `org/name@revision/file`, matching the Hub); the local API key; and the
local commit hash of the patched build. Every number is unchanged;
`../tools/verify_sanitized.py` proved it pair by pair.

## What was left out

The server logs the acceptance counts were parsed from (machine logs, with the local model
paths throughout), the smoke tests, the build logs and scripts, and the probe's own analysis
script. The patch itself is not redistributed here; it is public at the pinned commit above.
