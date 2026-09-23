#!/usr/bin/env python3
"""Fine-tune the DFlash 2 drafter against the ternary target (direction D).

    ~/.venv-dspark/bin/python bench/drafter/train_drafter.py <corpus.jsonl> <out_dir>
        [--rank 64] [--lr 1e-4] [--anchors 48] [--anchor-mode uniform|served|mixed]
        [--steps-per-seq 2] [--epochs 1] [--gamma 4] [--distill-topk 32] [--selector 1.0]
        [--selector-codebooks] [--eval-every 40] [--ckpt-every 10] [--limit N]
        [--self-test | --eval-only | --export-only]

The numerics live in bench/drafter/dflash_ft.py; this is the loop around them: the schedule,
the checkpoints, the signals, the logging.

Checkpointing. Every --ckpt-every sequences, and on SIGINT/SIGTERM after the step in flight,
the adapters, the optimizer state, the data position and the RNG go to <out_dir>/ckpt/. A
rerun with the same arguments resumes from there. --export-only fuses the checkpoint into a
servable drafter without training, so a paused run still yields something to measure.

Evaluation. --eval-only reports the teacher-forced proxy on the corpus's eval split in the
chosen anchor mode and exits; --self-test compares the batched training forward against the
served forward on a real sequence. Neither is the served loop: bench/drafter/served_accept.py
is, and it is the gate.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import signal
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import patches

patches.install_all()

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx_dspark.load import _resolve, load_dflash, load_target

from bench.drafter import dflash_ft as ft


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--target", default="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit")
    ap.add_argument("--drafter", default="z-lab/Qwen3.8-27B-DFlash2")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchors", type=int, default=48)
    ap.add_argument("--anchor-mode", choices=("uniform", "served", "mixed"), default="uniform")
    ap.add_argument("--steps-per-seq", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=4.0)
    ap.add_argument("--distill-topk", type=int, default=32,
                    help="soft labels from the target's top-K; 0 = hard labels from the corpus tokens")
    ap.add_argument("--selector", type=float, default=1.0, help="weight of the selector lattice loss")
    ap.add_argument("--selector-codebooks", action="store_true",
                    help="also train the two [vocab, rank] codebooks (iteration 2 did, and diverged)")
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--eval-mode", choices=("uniform", "served", "mixed"), default=None,
                    help="anchor mode for evaluation (default: the training mode)")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    mx.random.seed(args.seed)
    rng = random.Random(args.seed)
    eval_mode: str = args.eval_mode or args.anchor_mode

    rows = ft.load_corpus(args.corpus)
    train = [r for r in rows if r.split == "train"]
    evals = [r for r in rows if r.split == "eval"]
    if args.limit:
        train = train[: args.limit]
    print(f"corpus: {len(train)} train, {len(evals)} eval sequences", flush=True)

    target, _tok = load_target(args.target, require_tap=True, kv_bits=8)
    drafter, cfg = load_dflash(args.drafter, quantize=False)
    drafter.bind(target.model)
    tr = ft.Trainer(target, drafter, cfg, gamma=args.gamma, distill_topk=args.distill_topk,
                    selector_weight=args.selector, max_len=args.max_len)
    print(f"drafter: block {tr.block}, taps {tr.taps}, window {tr.window}, layers {len(drafter.layers)}",
          flush=True)

    if args.self_test:
        st = tr.self_test((evals or train)[0])
        print(f"self-test: top token agreed {st.agree}/{st.total} ({st.agree_clear}/{st.total_clear} where the "
              f"served margin is >= {st.margin}), worst relative diff {st.worst:.4f}", flush=True)
        print("SELF-TEST", "OK" if st.ok else "FAILED", flush=True)
        return 0 if st.ok else 1

    n_train = ft.attach_lora(drafter, args.rank, selector_projection=args.selector > 0,
                             selector_codebooks=args.selector_codebooks and args.selector > 0)
    print(f"lora attached, {n_train / 1e6:.1f}M trainable parameters", flush=True)

    if args.eval_only:
        r = tr.evaluate(evals, args.anchors, eval_mode, rng)
        print(f"eval-only ({len(evals)} seqs, {r.anchors} {eval_mode} anchors): loss {r.loss:.4f}  "
              f"slot acc {[round(x, 3) for x in r.slot_accuracy]}  expected accept {r.expected_accept:.2f}  "
              f"prefix accept {r.prefix_accept:.2f}", flush=True)
        return 0

    # ---- schedule, optimizer, log
    total_steps = args.epochs * len(train) * args.steps_per_seq
    warm = max(1, int(0.04 * total_steps))
    sched = optim.join_schedules(
        [optim.linear_schedule(0.0, args.lr, warm), optim.cosine_decay(args.lr, max(1, total_steps - warm))],
        [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    grad_fn = nn.value_and_grad(drafter, tr.loss)
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "ckpt")
    log = open(os.path.join(args.out, "train.log"), "a")  # noqa: SIM115 - held open for the run

    def say(s: str) -> None:
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    def eval_line(tag: str, rows_: list[ft.Row]) -> ft.EvalResult:
        r = tr.evaluate(rows_, args.anchors, eval_mode, rng)
        say(f"eval {tag} ({len(rows_)} seqs, {r.anchors} {eval_mode} anchors): loss {r.loss:.4f}  "
            f"slot acc {[round(x, 3) for x in r.slot_accuracy]}  expected accept {r.expected_accept:.2f}  "
            f"prefix accept {r.prefix_accept:.2f}")
        return r

    stop = {"now": False}

    def on_signal(signum: int, _frame: Any) -> None:
        stop["now"] = True
        say(f"signal {signum}: finishing the step in flight, then checkpointing")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    resumed = ft.load_checkpoint(drafter, opt, ckpt)
    src = _resolve(args.drafter)
    if args.export_only:
        if resumed is None:
            say(f"no checkpoint to export in {ckpt}")
            return 1
        n, missing, extra = ft.export(drafter, src, args.out)
        say(f"export-only from step {resumed['step']}: {n} tensors"
            + (f"; key mismatch missing {missing[:5]} extra {extra[:5]}" if missing or extra else ""))
        return 0

    if resumed:
        step, ep0, order, pos, ema, ea0 = (resumed["step"], resumed["epoch"], resumed["order"],
                                           resumed["pos"], resumed["ema"], resumed["ea0"])
        say(f"resumed from step {step} (epoch {ep0}, position {pos}/{len(order)})")
    else:
        ea0 = eval_line("before", evals[:8]).prefix_accept
        step, ep0, order, pos, ema = 0, 0, None, 0, None

    # ---- the loop
    t0 = time.time()
    t_steps = 0
    for ep in range(ep0, args.epochs):
        if order is None or ep != ep0:
            order = list(range(len(train)))
            random.shuffle(order)
            pos = 0
        while pos < len(order):
            row = train[order[pos]]
            anchors = tr.anchors_for(row, args.anchors, args.anchor_mode, rng)
            if anchors:
                ids, _, _, _ = tr.batch(row, anchors)
                fused, tlogits = tr.teacher(ids)
                for _ in range(args.steps_per_seq):
                    anchors = tr.anchors_for(row, args.anchors, args.anchor_mode, rng)
                    _, blk, anc, lab = tr.batch(row, anchors)
                    loss, grads = grad_fn(blk, fused, tlogits, anc, lab)
                    grads, gnorm = optim.clip_grad_norm(grads, 1.0)
                    mx.eval(loss, gnorm)
                    lv, gv = loss.item(), gnorm.item()
                    step += 1
                    t_steps += 1
                    if not (math.isfinite(lv) and math.isfinite(gv)):
                        say(f"step {step}: non-finite loss {lv} / gnorm {gv} on len {len(ids)}; step skipped")
                        continue
                    opt.update(drafter, grads)
                    mx.eval(drafter.trainable_parameters(), opt.state)
                    ema = lv if ema is None else 0.95 * ema + 0.05 * lv
                    if step % 5 == 0:
                        say(f"step {step}/{total_steps}  loss {lv:.4f}  ema {ema:.4f}  gnorm {gv:.2f}  "
                            f"lr {sched(step):.2e}  len {len(ids)}  {(time.time() - t0) / t_steps:.1f} s/step")
                    if args.eval_every and step % args.eval_every == 0:
                        eval_line(f"@ {step}", evals[:8])
            pos += 1
            mx.clear_cache()
            if stop["now"] or (args.ckpt_every and pos % args.ckpt_every == 0):
                ft.save_checkpoint(drafter, opt, {"step": step, "epoch": ep, "order": order, "pos": pos,
                                                  "ema": ema, "ea0": ea0}, ckpt)
                say(f"checkpoint at step {step} (sequence {pos}/{len(order)} of epoch {ep})")
                if stop["now"]:
                    say("PAUSED")
                    return 0
        order = None

    ft.save_checkpoint(drafter, opt, {"step": step, "epoch": args.epochs, "order": [], "pos": 0,
                                      "ema": ema, "ea0": ea0}, ckpt)
    r = eval_line("after", evals)
    n, missing, extra = ft.export(drafter, src, args.out)
    say(f"exported {n} tensors to {args.out}"
        + (f"; key mismatch missing {missing[:5]} extra {extra[:5]}" if missing or extra else ""))
    say(f"before/after prefix accept {ea0:.2f} (first 8 seqs) -> {r.prefix_accept:.2f} (all {len(evals)} seqs); "
        f"compare like for like with --eval-only on the stock drafter")
    say("TRAIN DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
