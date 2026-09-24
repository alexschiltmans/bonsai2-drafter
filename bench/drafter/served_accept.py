#!/usr/bin/env python3
"""Served acceptance on a corpus's prompts: the draft loop itself, greedy, in-process.

    ~/.venv-dspark/bin/python bench/drafter/served_accept.py <corpus.jsonl> <drafter> [--split eval]
                                                             [--max-new 200] [--n 40]

The teacher-forced proxy in train_drafter.py disagreed with the served battery
(iteration 2: proxy +13%, served -26%), so this measures the quantity that matters with the
loop that serves it: mlx-dspark's dflash_generate at cap 7, greedy, on the corpus's own held-out
prompts. Mean accepted tokens a round and decode rate, per prompt and pooled.

--bits is the drafter's served precision (4, as the measurements serve it; 0 = bf16, the precision
the trainer's proxy sees). --record writes the eval rows back out in corpus format with THIS
drafter's round lengths, so train_drafter.py --eval-mode served can score the proxy at the
anchors this drafter actually drew rather than the corpus generator's. Greedy decoding means
the token path is the target's own, so the rows differ only in their round lengths.
"""
import argparse
import hashlib
import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import patches
from bench.drafter.report_checks import corpus_strata, decode_seconds

patches.install_all()
import mlx.core as mx
from mlx_dspark.generate import dflash_generate
from mlx_dspark.load import load_dflash, load_target

ap = argparse.ArgumentParser()
ap.add_argument("corpus"); ap.add_argument("drafter")
ap.add_argument("--target", default="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
ap.add_argument("--split", default="eval"); ap.add_argument("--max-new", type=int, default=200)
ap.add_argument("--n", type=int, default=40); ap.add_argument("--bits", type=int, default=4)
ap.add_argument("--record", default=None, help="write the rows with this drafter's round lengths here")
ap.add_argument("--report", help="write settings and per-prompt measurements as JSON for paired analysis")
args = ap.parse_args()
if args.n <= 0 or args.max_new <= 0:
    ap.error("n and max-new must be positive")
with open(args.corpus) as f:
    rows = [json.loads(line) for line in f if line.strip()]
rows = [r for r in rows if r["split"] == args.split][:args.n]
if len(rows) != args.n:
    ap.error(f"requested {args.n} prompts but found {len(rows)} in split {args.split}")
try:        # before any model loads: the analyser would refuse the report these produce
    strata = [corpus_strata(r) for r in rows]
except ValueError as exc:
    ap.error(str(exc))
report: dict[str, Any] = {
    "settings": {"target": args.target, "split": args.split, "max_new": args.max_new,
                 "bits": args.bits, "kv_bits": 8, "cap": 7, "temperature": 0.0},
    "drafter": args.drafter, "complete": False, "requests": [],
}
measurements = report["requests"]


# Keep the partial record if the process fails, but analysis must refuse it.
def save_report() -> None:
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

save_report()
target, tok = load_target(args.target, require_tap=True, kv_bits=8)
drafter, cfg = load_dflash(args.drafter, quantize=args.bits > 0, bits=max(args.bits, 1))
drafter.bind(target.model)
acc_all, tok_all, sec_all, rounds_all = [], 0, 0.0, 0
if args.record:
    open(args.record, "w").close()  # rows are appended as each prompt completes
for i, r in enumerate(rows):
    res = dflash_generate(target, tok, drafter, prompt_ids=r["prompt_ids"], apply_chat_template=False,
                          max_new_tokens=args.max_new, max_draft_tokens=7, temperature=0.0)
    try:
        decode = decode_seconds(res.seconds, res.prefill_seconds)
    except ValueError as exc:
        raise SystemExit(f"prompt {i}: {exc}") from None     # the saved report stays incomplete
    a = res.num_tokens / max(1, res.num_rounds)
    acc_all.append(a); tok_all += res.num_tokens; sec_all += decode; rounds_all += res.num_rounds
    print(f"[{i+1}/{len(rows)}] think={int(r['thinking'])} {res.num_tokens} tok  {a:.2f} tok/round", flush=True)
    measurements.append({
        "prompt_sha256": hashlib.sha256(json.dumps(r["prompt_ids"]).encode()).hexdigest(),
        "category": strata[i][0], "thinking": strata[i][1],
        "tokens": res.num_tokens, "rounds": res.num_rounds,
        "decode_seconds": decode,
        "finish": res.finish_reason, "response_ids": [int(t) for t in res.token_ids],
        # Per-round lengths belong in the report, not only in --record: the round that
        # crosses max_new is the one the analysis has to account for, and without these it
        # can only bound it. Note the loop counts a round's committed length before the
        # append, so an EOS round can make this sum exceed tokens.
        "round_lengths": [int(x) for x in res.accept_lengths],
    })
    save_report()
    if args.record:
        out = dict(r)
        out["response_ids"] = [int(t) for t in res.token_ids]
        out["round_lengths"] = [int(x) for x in res.accept_lengths]
        out["finish"] = res.finish_reason
        with open(args.record, "a") as rec:
            rec.write(json.dumps(out) + "\n")
    mx.clear_cache()
print(f"SERVED {args.drafter}: {tok_all} tok over {rounds_all} rounds = {tok_all/max(1,rounds_all):.3f} tok/round pooled, "
      f"mean of prompts {sum(acc_all)/len(acc_all):.3f}, decode {tok_all/max(sec_all,1e-9):.1f} tok/s "
       f"(drafter {'bf16' if args.bits <= 0 else f'{args.bits}-bit'})", flush=True)
report["complete"] = True
save_report()
