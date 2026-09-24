"""Fine-tuning a DFlash 2 drafter against a served target: the library behind train_drafter.py.

What is here and why it is a module
-----------------------------------
`train_drafter.py` started as one script and grew three iterations of findings in a day.
The parts that make claims about numerics live here, typed and testable without a server:

* :func:`rope_at`, :func:`build_mask`, :class:`Trainer.drafter_hidden`: a re-statement of
  mlx-dspark's draft forward for a batch of anchors. Block rows are the batch axis, so the
  within-block convolution and the causal-within-block, windowed-over-context attention
  are what a served draft round sees. `Trainer.self_test` checks it against the served
  forward on real anchors.
* :meth:`Trainer.slot_targets`: the target's own next-token distribution at each mask slot,
  from the same forward that produced the hidden states. Serving verifies each drafted
  token against the target's argmax, so that is the label; the corpus tokens are the
  conditioning path only.
* :func:`served_anchors`: where the served loop actually invoked the drafter, recovered
  from the round lengths a corpus row carries. Uniformly sampled anchors over-represent the
  positions a drafter finds easy; three early fine-tunes raised a uniform-anchored
  proxy and lowered served acceptance, and this is the instrument that tests that reading.
* checkpoints (adapters, optimizer, data position, RNG) and the export back into the
  checkpoint's own tensor layout.

Numerics are bfloat16 with float32 accumulation where the served path does the same.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear

LORA_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------------------- corpus

@dataclass
class Row:
    """One corpus line: the target's own response to a prompt, as exact token ids."""

    prompt_ids: list[int]
    response_ids: list[int]
    thinking: bool
    split: str
    temperature: float = 1.0
    round_lengths: list[int] = field(default_factory=list)
    prompt: str = ""

    @property
    def ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    @staticmethod
    def from_json(d: dict[str, Any]) -> Row:
        return Row(
            prompt_ids=[int(t) for t in d["prompt_ids"]],
            response_ids=[int(t) for t in d["response_ids"]],
            thinking=bool(d.get("thinking", True)),
            split=str(d.get("split", "train")),
            temperature=float(d.get("temperature", 1.0)),
            round_lengths=[int(r) for r in d.get("round_lengths", [])],
            prompt=str(d.get("prompt", "")),
        )


def load_corpus(path: str | os.PathLike[str], min_response: int = 16) -> list[Row]:
    rows: list[Row] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = Row.from_json(json.loads(line))
            if len(row.response_ids) >= min_response:
                rows.append(row)
    return rows


def served_anchors(row: Row, max_len: int | None = None) -> list[int]:
    """The anchor positions the served loop drew on this row, from its round lengths.

    A round's block head is the last committed token: the first generated token, at
    position P, for the first round, then P plus the cumulative committed count. Only
    anchors with at least one label position after them count. Empty when the row carries
    no round lengths."""
    if not row.round_lengths:
        return []
    L = len(row.ids) if max_len is None else min(len(row.ids), max_len)
    P = len(row.prompt_ids)
    out: list[int] = []
    pos = P
    for r in row.round_lengths:
        if pos > L - 2:
            break
        out.append(pos)
        pos += r
    return out


def uniform_anchors(row: Row, n: int, rng: random.Random, max_len: int | None = None) -> list[int]:
    L = len(row.ids) if max_len is None else min(len(row.ids), max_len)
    lo, hi = len(row.prompt_ids), L - 2   # the loop never anchors inside the prompt
    if hi < lo:
        return []
    return sorted(rng.sample(range(lo, hi + 1), min(n, hi - lo + 1)))


# --------------------------------------------------------------------------- numerics

def rope_at(x: mx.array, positions: mx.array, base: float) -> mx.array:
    """Non-traditional RoPE (half-split rotation) at arbitrary positions.

    ``x`` [n, heads, L, D]; ``positions`` [n, L] int. Equals ``mx.fast.rope`` with
    ``traditional=False, scale=1.0`` at those offsets, to bfloat16 precision
    (bench/tests/test_dflash_ft.py checks it)."""
    D = x.shape[-1]
    inv = mx.power(base, -mx.arange(0, D, 2, dtype=mx.float32) / D)
    theta = positions.astype(mx.float32)[:, None, :, None] * inv
    cos, sin = mx.cos(theta), mx.sin(theta)
    x1, x2 = x[..., : D // 2].astype(mx.float32), x[..., D // 2 :].astype(mx.float32)
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def build_mask(anchors: mx.array, L: int, block: int, window: int) -> mx.array:
    """Additive attention mask [n, 1, block, L + block] for a batch of anchors.

    Context column p is visible to block row j of anchor a iff ``a + j - window < p < a``
    (no window when ``window`` is 0); block column j' iff ``j' <= j``. That is causal within
    the block, as mlx-dspark's sliding layers serve it, and windowed over the context.

    The context stops one row short of the anchor. The served block head is the token the
    target has just emitted and not yet read, so its fused row does not exist at draft time;
    the drafter's cache holds rows for positions ``< a`` and the block ropes from ``a``.
    An earlier version of this mask admitted row ``a`` itself, the target's state after reading
    the head, which carries its prediction of slot one's label: a leak the fine-tunes learned
    to read and serving never provides."""
    n = anchors.shape[0]
    j = mx.arange(block)[None, :, None]
    p = mx.arange(L)[None, None, :]
    a = anchors[:, None, None]
    ok_ctx = (p < a) & (p > a + j - window) if window else mx.broadcast_to(p < a, (n, block, L))
    jj = mx.arange(block)[None, None, :]
    ok_blk = mx.broadcast_to(jj <= j, (n, block, block))
    ok = mx.concatenate([ok_ctx, ok_blk], axis=-1)
    return mx.where(ok, 0.0, -1e9).astype(mx.bfloat16)[:, None]


def accept_figures(hits: list[float]) -> float:
    """The product-of-marginals accept: 1 + p1 + p1*p2 + ..., the round's expected commits if
    slots missed independently. Kept for older logs; see :func:`prefix_lengths`."""
    expected, running = 1.0, 1.0
    for p in hits:
        running *= p
        expected += running
    return expected


def prefix_lengths(hit: mx.array) -> mx.array:
    """Per anchor, what the greedy loop would commit: 1 (the bonus token) plus the run of
    hits from slot one up to the first miss. ``hit`` is [anchors, slots] of 0/1."""
    return 1 + mx.cumprod(hit.astype(mx.int32), axis=1).sum(axis=1)


@dataclass
class SelfTest:
    """The batched training forward against the served forward at a few anchors."""

    worst: float          # worst relative max-abs hidden-state difference over the anchors
    agree: int            # slots whose top token matched
    total: int            # slots compared
    agree_clear: int      # matches among the slots whose reference top-2 margin >= margin
    total_clear: int
    margin: float

    @property
    def ok(self) -> bool:
        """Every decisive slot agrees and the hidden states sit within bf16 noise. A slot whose
        best two logits are within ``margin`` can flip between two runs of the SAME forward
        (a 0.09 margin once measured 4.9% then 1.1%), so it counts only as measured."""
        return self.agree_clear == self.total_clear and self.worst < 0.06


@dataclass
class EvalResult:
    loss: float
    slot_accuracy: list[float]
    expected_accept: float
    prefix_accept: float
    anchors: int


# --------------------------------------------------------------------------- trainer

class Trainer:
    """The forward, the losses and the evaluations, bound to a loaded target and drafter."""

    def __init__(self, target: Any, drafter: Any, cfg: Any, *, gamma: float = 4.0,
                 distill_topk: int = 32, selector_weight: float = 1.0, max_len: int = 2048) -> None:
        self.target = target
        self.drafter = drafter
        self.taps = [int(t) for t in cfg.target_layer_ids]
        self.block = int(cfg.block_size)
        self.mask_id = int(cfg.mask_token_id)
        self.window = int(cfg.sliding_window or 0)
        self.head = int(cfg.head_dim)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv = int(cfg.num_key_value_heads)
        self.rope_theta = float(cfg.rope_theta)
        self.vocab = int(cfg.vocab_size)
        self.gamma = gamma
        self.distill_topk = distill_topk
        self.selector_weight = selector_weight
        self.max_len = max_len
        self.selector = getattr(drafter, "candidate_selector", None)
        self.w_pos = mx.array([math.exp(-k / gamma) for k in range(self.block - 1)])

    # -- forward
    def block_attention(self, attn: Any, a: mx.array, ctx: mx.array, anchors: mx.array,
                        mask: mx.array) -> mx.array:
        n, L = a.shape[0], ctx.shape[1]
        B, H, KV, D = self.block, self.n_heads, self.n_kv, self.head
        kc = attn.k_norm(attn.k_proj(ctx).reshape(1, L, KV, D)).transpose(0, 2, 1, 3)
        vc = attn.v_proj(ctx).reshape(1, L, KV, D).transpose(0, 2, 1, 3)
        kc = rope_at(kc, mx.arange(L)[None, :], self.rope_theta)
        q = attn.q_norm(attn.q_proj(a).reshape(n, B, H, D)).transpose(0, 2, 1, 3)
        kb = attn.k_norm(attn.k_proj(a).reshape(n, B, KV, D)).transpose(0, 2, 1, 3)
        vb = attn.v_proj(a).reshape(n, B, KV, D).transpose(0, 2, 1, 3)
        bpos = anchors[:, None] + mx.arange(B)[None, :]      # the head sits at the anchor
        q, kb = rope_at(q, bpos, self.rope_theta), rope_at(kb, bpos, self.rope_theta)
        keys = mx.concatenate([mx.broadcast_to(kc, (n, *kc.shape[1:])), kb], axis=2)
        values = mx.concatenate([mx.broadcast_to(vc, (n, *vc.shape[1:])), vb], axis=2)
        out = mx.fast.scaled_dot_product_attention(q, keys, values, scale=attn.scale, mask=mask)
        return attn.o_proj(out.transpose(0, 2, 1, 3).reshape(n, B, -1))

    def drafter_hidden(self, ids_blk: mx.array, fused: mx.array, anchors: mx.array) -> mx.array:
        """[n, block] block ids, [1, L, taps*H] fused target hidden -> post-norm hidden of the
        mask rows [n, block-1, H]."""
        d = self.drafter
        L = fused.shape[1]
        ctx = d.project_ctx(fused)
        h = d.embed_tokens(ids_blk) * d.embed_scale
        mask = build_mask(anchors, L, self.block, self.window)
        for layer in d.layers:
            a = layer.input_layernorm(h)
            if layer.attention_conv is not None:
                a, oc = layer.attention_conv.prepare(a)
            att = self.block_attention(layer.self_attn, a, ctx, anchors, mask)
            if layer.attention_conv is not None:
                att = layer.attention_conv.finish(att, oc)
            h = h + att
            m = layer.post_attention_layernorm(h)
            if layer.mlp_conv is not None:
                m, oc = layer.mlp_conv.prepare(m)
            mm = layer.mlp(m)
            if layer.mlp_conv is not None:
                mm = layer.mlp_conv.finish(mm, oc)
            h = h + mm
        return d.norm(h)[:, 1:]

    def teacher(self, ids: list[int]) -> tuple[mx.array, mx.array]:
        """Fused hidden states and next-token logits of the target for one sequence."""
        logits, fused = self.target.run(mx.array([ids]), self.target.make_cache(), self.taps)
        logits = logits.astype(mx.bfloat16)
        mx.eval(fused, logits)
        return mx.stop_gradient(fused), mx.stop_gradient(logits)

    # -- labels
    def slot_targets(self, tlogits: mx.array, anchors: mx.array,
                     valid: mx.array) -> tuple[mx.array | None, mx.array | None, mx.array]:
        """For anchor a and slot j the target predicts token a+1+j from position a+j. Returns
        (top-K ids, renormalised top-K probabilities, argmax) with -1 where invalid."""
        pos = anchors[:, None] + mx.arange(self.block - 1)[None, :]
        # past the end there is no row; the weight is 0
        pos = mx.minimum(pos, tlogits.shape[1] - 1)
        tl = tlogits[0][pos].astype(mx.float32)
        hard = mx.where(valid, mx.argmax(tl, axis=-1).astype(mx.int32), -1)
        k = self.distill_topk
        if k <= 0:
            return None, None, hard
        ids = mx.argpartition(tl, kth=-k, axis=-1)[..., -k:]
        probs = mx.softmax(mx.take_along_axis(tl, ids, axis=-1), axis=-1)
        return mx.stop_gradient(ids), mx.stop_gradient(probs), hard

    def selector_scores(self, hid: mx.array, logits: mx.array, ids_blk: mx.array,
                        hard: mx.array) -> tuple[mx.array, mx.array]:
        """Lattice scores over each slot's top-K candidates given the true predecessor
        (the anchor for slot 0, the target's argmax at slot s-1 otherwise), as
        ``CandidateSelector.lattice`` builds them for one predecessor. Returns (scores, ids)."""
        sel = self.selector
        if sel is None:
            raise RuntimeError("this drafter has no candidate selector")
        k = int(sel.top_k)
        lg = logits[..., : self.vocab]
        cand = mx.stop_gradient(mx.argpartition(lg, kth=-k, axis=-1)[..., -k:]).astype(mx.int32)
        unary = self.drafter._transform_unary(mx.take_along_axis(lg, cand, axis=-1))
        pred = mx.concatenate([ids_blk[:, :1], mx.maximum(hard[:, :-1], 0)], axis=1)
        pre = sel.predecessor_codebook[pred] * sel.hidden_projection(hid)
        suc = sel.successor_codebook[cand]
        scores = unary + (pre[:, :, None, :] * suc).sum(-1).astype(mx.float32)
        return scores, cand

    def selector_loss(self, hid: mx.array, logits: mx.array, ids_blk: mx.array,
                      hard: mx.array, valid: mx.array) -> mx.array:
        scores, cand = self.selector_scores(hid, logits, ids_blk, hard)
        hit = cand == hard[..., None]
        present = hit.any(axis=-1) & valid
        tgt = mx.argmax(hit, axis=-1)
        ce = nn.losses.cross_entropy(scores, tgt, reduction="none")
        w = self.w_pos[None, :] * present
        return (ce * w).sum() / mx.maximum(w.sum(), 1.0)

    def loss(self, ids_blk: mx.array, fused: mx.array, tlogits: mx.array, anchors: mx.array,
             labels: mx.array) -> mx.array:
        hid = self.drafter_hidden(ids_blk, fused, anchors)
        logits = self.drafter.lm_head(hid).astype(mx.float32)
        valid = labels >= 0
        tids, tprobs, hard = self.slot_targets(tlogits, anchors, valid)
        w = self.w_pos[None, :] * valid
        if tids is None or tprobs is None:
            ce = nn.losses.cross_entropy(logits, mx.maximum(labels, 0), reduction="none")
        else:
            logq = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            ce = -(tprobs * mx.take_along_axis(logq, tids, axis=-1)).sum(-1)
        ce = mx.where(valid, ce, 0.0)
        total = (ce * w).sum() / mx.maximum(w.sum(), 1.0)
        if self.selector is not None and self.selector_weight > 0:
            total = total + self.selector_weight * self.selector_loss(hid, logits, ids_blk, hard,
                                                                      valid)
        return total

    # -- batches
    def batch(self, row: Row, anchors: list[int]) -> tuple[list[int], mx.array, mx.array, mx.array]:
        """(ids, block ids [n, block], anchors [n], labels [n, block-1]) for the given anchors.
        Labels are the corpus tokens, -1 past the end; the loss uses them only for validity
        when distilling."""
        ids = row.ids[: self.max_len]
        L = len(ids)
        B = self.block
        blk = [[ids[a]] + [self.mask_id] * (B - 1) for a in anchors]
        lab = [[ids[a + 1 + j] if a + 1 + j < L else -1 for j in range(B - 1)] for a in anchors]
        return ids, mx.array(blk), mx.array(anchors), mx.array(lab)

    def anchors_for(self, row: Row, n: int, mode: str, rng: random.Random) -> list[int]:
        """``uniform``: n positions drawn over the response. ``served``: the positions the loop
        drew on this row (up to n of them, evenly thinned), falling back to uniform when the
        row carries none. ``mixed``: half and half."""
        if mode == "uniform":
            return uniform_anchors(row, n, rng, self.max_len)
        served = served_anchors(row, self.max_len)
        if mode == "served":
            if not served:
                return uniform_anchors(row, n, rng, self.max_len)
            if len(served) > n:
                step = len(served) / n
                served = [served[int(i * step)] for i in range(n)]
            return served
        if mode == "mixed":
            half = max(1, n // 2)
            s = served[:: max(1, len(served) // half)][:half] if served else []
            u = uniform_anchors(row, n - len(s), rng, self.max_len)
            return sorted(set(s) | set(u))
        raise ValueError(f"unknown anchor mode {mode!r}")

    # -- evaluation
    def evaluate(self, rows: list[Row], n: int, mode: str, rng: random.Random) -> EvalResult:
        """Teacher-forced accuracy at each mask slot against the target's argmax, using the
        selector's walk from the true predecessor when the drafter has one.

        Two accept figures come back. ``expected_accept`` multiplies the marginal slot
        accuracies as if slots were independent; ``prefix_accept`` is the mean over anchors
        of 1 + the run of hits from slot one, which is what the greedy loop commits a round.
        Hits are strongly correlated across slots (an easy position is easy for every slot),
        so the product understates the loop whenever slot one is weak: in one measurement
        the stock drafter read 2.40 by the product at served anchors while the loop committed
        3.82 on the same prompts. Compare ``prefix_accept`` with served_accept.py; the
        product is kept so older logs still line up."""
        B = self.block
        hits = mx.zeros((B - 1,))
        tot = mx.zeros((B - 1,))
        prefix_sum = mx.zeros(())
        losses: list[float] = []
        n_anchors = 0
        for row in rows:
            anchors = self.anchors_for(row, n, mode, rng)
            if not anchors:
                continue
            ids, blk, anc, lab0 = self.batch(row, anchors)
            n_anchors += len(anchors)
            fused, tlogits = self.teacher(ids)
            hid = self.drafter_hidden(blk, fused, anc)
            logits = self.drafter.lm_head(hid).astype(mx.float32)
            valid = lab0 >= 0
            _, _, hard = self.slot_targets(tlogits, anc, valid)
            if self.selector is not None:
                scores, cand = self.selector_scores(hid, logits, blk, hard)
                best = mx.argmax(scores, axis=-1)[..., None]
                pred = mx.take_along_axis(cand, best, axis=-1)[..., 0]
            else:
                pred = mx.argmax(logits[..., : self.vocab], axis=-1).astype(mx.int32)
            hit = ((pred == hard) & valid).astype(mx.int32)
            hits = hits + hit.sum(axis=0)
            tot = tot + valid.sum(axis=0)
            prefix_sum = prefix_sum + (prefix_lengths(hit) - 1).sum()
            losses.append(float(self.loss(blk, fused, tlogits, anc, lab0).item()))  # type: ignore[arg-type]
            mx.eval(hits, tot, prefix_sum)
        if not losses:
            return EvalResult(float("nan"), [], float("nan"), float("nan"), 0)
        acc_list = (hits / mx.maximum(tot, 1)).tolist()
        if not isinstance(acc_list, list):
            raise TypeError("per-slot accuracy did not come back as a list")
        acc = [float(x) for x in acc_list]  # type: ignore[arg-type]
        prefix = 1.0 + float(prefix_sum.item()) / n_anchors  # type: ignore[arg-type]
        return EvalResult(sum(losses) / len(losses), acc, accept_figures(acc), prefix, n_anchors)

    def self_test(self, row: Row, anchors: list[int] | None = None,
                  margin: float = 0.5) -> SelfTest:
        """The batched forward against the served one (`forward_hidden` with a fresh cache
        and the context rows before the anchor, as the loop hands them over)."""
        ids = row.ids[: self.max_len]
        fused, _ = self.teacher(ids)
        P, L, B = len(row.prompt_ids), len(ids), self.block
        anchors = anchors or [P, P + 5, (P + L) // 2, L - B - 1]
        worst, agree, total, agree_clear, total_clear = 0.0, 0, 0, 0, 0
        for a in anchors:
            blk = mx.array([[ids[a]] + [self.mask_id] * (B - 1)])
            ref = self.drafter.forward_hidden(blk, fused[:, :a], self.drafter.make_cache(),
                                              logits_start=1)
            got = self.drafter_hidden(blk, fused, mx.array([a]))
            d = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())  # type: ignore[arg-type]
            m = float(mx.max(mx.abs(ref.astype(mx.float32))).item())  # type: ignore[arg-type]
            lr = self.drafter.lm_head(ref)[0].astype(mx.float32)
            top_ref = cast(list[int], mx.argmax(lr, axis=-1).tolist())
            top_got = cast(list[int], mx.argmax(self.drafter.lm_head(got)[0], axis=-1).tolist())
            srt = mx.sort(lr, axis=-1)
            margins = cast(list[float], (srt[:, -1] - srt[:, -2]).tolist())
            worst = max(worst, d / max(m, 1e-9))
            for x, y, mg in zip(top_ref, top_got, margins, strict=True):
                agree += x == y
                total += 1
                if mg >= margin:
                    agree_clear += x == y
                    total_clear += 1
        return SelfTest(worst, agree, total, agree_clear, total_clear, margin)


# ------------------------------------------------------------------ lora, checkpoints, export

def attach_lora(drafter: Any, rank: int, *, scale: float = 20.0, selector_projection: bool = True,
                selector_codebooks: bool = False) -> int:
    """Freeze the drafter, wrap the backbone projections and the fusion in LoRA, and unfreeze
    what the selector is allowed to move. Returns the number of trainable parameters."""
    drafter.freeze()
    for layer in drafter.layers:
        mods = [(k, LoRALinear.from_base(m, r=rank, scale=scale))
                for k, m in layer.named_modules()
                if isinstance(m, nn.Linear) and k.split(".")[-1] in LORA_KEYS]
        layer.update_modules(tree_unflatten(mods))
    drafter.fc = LoRALinear.from_base(drafter.fc, r=rank, scale=scale)
    sel = getattr(drafter, "candidate_selector", None)
    if sel is not None:
        if selector_codebooks:
            sel.unfreeze()
        elif selector_projection:
            sel.hidden_projection.unfreeze()
    return int(sum(v.size for _, v in tree_flatten(drafter.trainable_parameters())))


def save_checkpoint(drafter: Any, opt: optim.Optimizer, state: dict[str, Any], ckpt: str) -> None:
    """Adapters, optimizer state, data position and RNG. The previous checkpoint is replaced
    only after the new one is complete, so a kill mid-save leaves it intact."""
    tmp = ckpt + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    mx.save_safetensors(os.path.join(tmp, "adapters.safetensors"),
                        dict(tree_flatten(drafter.trainable_parameters())))
    mx.save_safetensors(os.path.join(tmp, "optimizer.safetensors"),
                        {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)})
    with open(os.path.join(tmp, "state.json"), "w") as f:
        json.dump({**state, "random": random.getstate()}, f)
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
    os.rename(tmp, ckpt)


def load_checkpoint(drafter: Any, opt: optim.Optimizer, ckpt: str) -> dict[str, Any] | None:
    st = os.path.join(ckpt, "state.json")
    if not os.path.exists(st):
        return None
    with open(st) as f:
        state: dict[str, Any] = json.load(f)
    adapters = mx.load(os.path.join(ckpt, "adapters.safetensors"))
    if not isinstance(adapters, dict):
        raise TypeError(f"{ckpt}/adapters.safetensors did not load as a tensor dictionary")
    drafter.load_weights(list(adapters.items()), strict=False)
    ostate = mx.load(os.path.join(ckpt, "optimizer.safetensors"))
    if not isinstance(ostate, dict):
        raise TypeError(f"{ckpt}/optimizer.safetensors did not load as a tensor dictionary")
    opt.state = tree_unflatten(list(ostate.items()))
    rs = state.pop("random")
    random.setstate((rs[0], tuple(rs[1]), rs[2]))
    mx.eval(drafter.trainable_parameters())
    return state


def export(drafter: Any, src_dir: str, out_dir: str) -> tuple[int, list[str], list[str]]:
    """Fuse the adapters into plain weights and write them in the source checkpoint's own
    layout (the bound embed/lm_head are the target's and are not written). Returns
    (tensors written, keys missing against the source, keys extra against the source)."""
    for layer in drafter.layers:
        fused = [(k, m.fuse()) for k, m in layer.named_modules() if isinstance(m, LoRALinear)]
        layer.update_modules(tree_unflatten(fused))
    if isinstance(drafter.fc, LoRALinear):
        drafter.fc = drafter.fc.fuse()
    weights: dict[str, mx.array] = {
        k: v for k, v in tree_flatten(drafter.parameters())
        if not k.startswith(("embed_tokens", "lm_head"))}
    ref: dict[str, mx.array] = {}
    for st in Path(src_dir).glob("*.safetensors"):
        loaded = mx.load(str(st))
        if not isinstance(loaded, dict):
            raise TypeError(f"{st} did not load as a tensor dictionary")
        ref.update(loaded)
    missing = sorted(set(ref) - set(weights))
    extra = sorted(set(weights) - set(ref))
    weights = {k: (v.astype(ref[k].dtype) if k in ref else v) for k, v in weights.items()}
    os.makedirs(out_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), weights,
                        metadata={"format": "mlx"})
    shutil.copyfile(os.path.join(src_dir, "config.json"), os.path.join(out_dir, "config.json"))
    return len(weights), missing, extra
