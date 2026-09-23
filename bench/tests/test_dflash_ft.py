#!/usr/bin/env python3
"""The drafter fine-tuning library, checked without a server.

    ~/.venv-dspark/bin/python bench/tests/test_dflash_ft.py            (GPU, no models)
    ~/.venv-dspark/bin/python bench/tests/test_dflash_ft.py --with-models

Four groups, in the order the library's claims are made:

1. Anchors. Served anchors from round lengths land where the loop restarted (P, the first
   generated token, then the cumulative commits), stop before the last label position, and
   are empty without round lengths; uniform anchors stay inside the response.
2. The mask. Every entry of `build_mask` follows from the rule in its docstring: causal
   within the block, windowed over the context, and the context ends before the anchor's
   own row (the leak the first three fine-tunes learned to read).
3. RoPE. `rope_at` equals mx.fast.rope at three offsets, to bfloat16 precision. Then the
   two accept figures on a handcrafted hit matrix, where they differ.
4. With --with-models: the batched training forward against the served forward on a real
   sequence from the corpus if one is on disk, else a synthetic one. This is the check that
   makes the training numerics the served numerics, and it needs the 8.6 GB pack and the
   3.9 GB drafter in the HuggingFace cache.

Also a round trip of checkpoint save/load on a tiny stand-in module, since the resume path
is what a paused overnight run depends on.
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
from typing import cast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten

from bench.drafter import dflash_ft as ft

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   {'ok  ' if ok else 'FAIL'} {name}" + (f"  {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


# ------------------------------------------------------------------ 1. anchors
print("== 1. anchors")
row = ft.Row(prompt_ids=list(range(10)), response_ids=list(range(100, 120)), thinking=False,
             split="eval", round_lengths=[4, 3, 8, 5])
served = ft.served_anchors(row)
check("served anchors start at P and follow the commits", served == [10, 14, 17, 25], str(served))
row_end = ft.Row(prompt_ids=list(range(10)), response_ids=list(range(100, 106)), thinking=False,
                 split="eval", round_lengths=[4, 3, 8])
check("an anchor past the last label position is dropped", ft.served_anchors(row_end) == [10, 14],
      str(ft.served_anchors(row_end)))
check("no round lengths, no served anchors", ft.served_anchors(ft.Row([1, 2], [3, 4, 5], True, "train")) == [])
check("max_len bounds the served anchors", ft.served_anchors(row, max_len=15) == [10],
      str(ft.served_anchors(row, max_len=15)))
rng = random.Random(0)
u = ft.uniform_anchors(row, 8, rng)
check("uniform anchors lie in [P, L-2]", all(10 <= a <= 28 for a in u) and len(u) == 8, str(u))
check("uniform anchors are capped by the response length", len(ft.uniform_anchors(row_end, 50, rng)) == 5)

# ------------------------------------------------------------------ 2. the mask
print("== 2. the mask")
B, L, W = 4, 12, 6
anchors = mx.array([3, 9])
m = ft.build_mask(anchors, L, B, W)
check("mask shape [n, 1, block, L + block]", m.shape == (2, 1, B, L + B), str(m.shape))
vis = cast(list[list[list[bool]]], (m[:, 0] == 0).tolist())   # [n, B, L+B]


def expected(a: int, j: int, col: int) -> bool:
    if col < L:                     # context column
        return col < a and col > a + j - W
    jj = col - L                    # block column
    return jj <= j


bad = [(i, j, c) for i, a in enumerate([3, 9]) for j in range(B) for c in range(L + B)
       if vis[i][j][c] != expected(a, j, c)]
check("every mask entry follows the rule", not bad, str(bad[:5]))
m0 = ft.build_mask(anchors, L, B, 0)
vis0 = cast(list[list[list[bool]]], (m0[:, 0] == 0).tolist())
check("window 0 means every context row before the anchor", all(vis0[1][j][c] for j in range(B) for c in range(9))
      and not any(vis0[1][j][c] for j in range(B) for c in range(9, L)))
check("the anchor's own row is never visible", not any(vis[i][j][a] for i, a in enumerate([3, 9]) for j in range(B)))

# ------------------------------------------------------------------ 3. rope
print("== 3. rope")
x = mx.random.normal((1, 4, 8, 128)).astype(mx.bfloat16)
worst = 0.0
for off in (0, 137, 5000):
    ref = mx.fast.rope(x, 128, traditional=False, base=1e7, scale=1.0, offset=off)
    got = ft.rope_at(x, off + mx.arange(8)[None, :], 1e7)
    diff = float(mx.max(mx.abs(ref.astype(mx.float32) - got.astype(mx.float32))).item())  # type: ignore[arg-type]
    worst = max(worst, diff / float(mx.max(mx.abs(ref.astype(mx.float32))).item()))  # type: ignore[arg-type]
# one bfloat16 ulp is 2^-8 = 0.0039 of the value; offset 5000 lands on it
check("rope_at matches mx.fast.rope within a bfloat16 ulp at three offsets", worst < 5e-3, f"{worst:.4f}")

# ------------------------------------------------------------------ checkpoints
# ------------------------------------------------------------------ 3b. accept figures
print("== 3b. the two accept figures")
# three anchors, four slots: prefixes 3, 1, 5 (a miss at slot one ends the run at once)
hit = mx.array([[1, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
pl = cast(list[int], ft.prefix_lengths(hit).tolist())
check("prefix length is 1 + the run of hits from slot one", pl == [3, 1, 5], str(pl))
check("a miss at slot one commits only the bonus token", pl[1] == 1)
marg = cast(list[float], (hit.sum(axis=0) / 3).tolist())
ea = ft.accept_figures(marg)
check("product of marginals: 1 + 2/3 + 2/3 + 4/9 + 4/9", abs(ea - (1 + 2 / 3 + 2 / 3 + 4 / 9 + 4 / 9)) < 1e-6, f"{ea}")  # float32 marginals
check("the two figures differ on correlated hits (mean prefix 3.0)", abs(sum(pl) / 3 - 3.0) < 1e-9 and abs(ea - 3.0) > 0.1)
check("an empty accuracy list is the bonus token alone", ft.accept_figures([]) == 1.0)

print("== checkpoints")


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(4, 4)]
        self.fc = nn.Linear(4, 4)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc(self.layers[0](x))


tiny = Tiny()
opt = optim.AdamW(learning_rate=1e-3)
x = mx.random.normal((2, 4))
loss_fn = lambda model, x_: (model(x_) ** 2).mean()
_, grads = nn.value_and_grad(tiny, loss_fn)(tiny, x)
opt.update(tiny, grads)
mx.eval(tiny.parameters(), opt.state)
before = dict(tree_flatten(tiny.parameters()))
with tempfile.TemporaryDirectory() as tmp:
    ck = os.path.join(tmp, "ckpt")
    random.seed(123)
    r1 = random.random()
    random.seed(123)
    ft.save_checkpoint(tiny, opt, {"step": 7, "epoch": 0, "order": [2, 0, 1], "pos": 1, "ema": 0.5, "ea0": 3.0}, ck)
    tiny2 = Tiny()
    opt2 = optim.AdamW(learning_rate=1e-3)
    state = ft.load_checkpoint(tiny2, opt2, ck)
    after = dict(tree_flatten(tiny2.parameters()))
    same = all(mx.array_equal(before[k], after[k]).item() for k in before)
    check("adapters round-trip through a checkpoint", same)
    check("state round-trips", state == {"step": 7, "epoch": 0, "order": [2, 0, 1], "pos": 1, "ema": 0.5, "ea0": 3.0},
          str(state))
    check("the RNG stream resumes where it left off", random.random() == r1)
    check("optimizer step count round-trips", int(opt2.state["step"].item()) == int(opt.state["step"].item()))
    check("nothing to load from an empty directory", ft.load_checkpoint(Tiny(), optim.AdamW(1e-3), os.path.join(tmp, "none")) is None)

# ------------------------------------------------------------------ 4. with models
if "--with-models" in sys.argv[1:]:
    print("== 4. the batched forward against the served forward (models)")
    import patches

    patches.install_all()
    from mlx_dspark.load import load_dflash, load_target

    target, tok = load_target("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", require_tap=True, kv_bits=8)
    drafter, cfg = load_dflash("z-lab/Qwen3.8-27B-DFlash2", quantize=False)
    drafter.bind(target.model)
    tr = ft.Trainer(target, drafter, cfg)
    # A corpus written by gen_data.py, if one is at hand; otherwise a synthetic row.
    corpus = os.environ.get("BONSAI2_DRAFTER_CORPUS", "")
    if corpus and os.path.exists(corpus):
        row = next(r for r in ft.load_corpus(corpus) if r.split == "eval")
    else:
        ids = tok.encode("Write a Python function that merges two sorted lists. " * 12)
        row = ft.Row(prompt_ids=ids[:40], response_ids=ids[40:], thinking=False, split="eval")
    st = tr.self_test(row)
    check("top token agrees at every decisive slot of every anchor", st.agree_clear == st.total_clear,
          f"{st.agree_clear}/{st.total_clear} (all slots {st.agree}/{st.total})")
    check("hidden states within bf16 noise of the served forward", st.worst < 0.06, f"{st.worst:.4f}")
    check("the served forward is decisive at most slots", st.total_clear >= st.total // 2, f"{st.total_clear}/{st.total}")
    r = tr.evaluate([row], 16, "uniform", random.Random(0))
    check("evaluate reports every slot", len(r.slot_accuracy) == tr.block - 1 and r.anchors == 16)
    check("prefix accept lies between the bonus token and the block",
          1.0 <= r.prefix_accept <= tr.block and 1.0 <= r.expected_accept <= tr.block)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)}: {FAILURES}")
    sys.exit(1)
print("all checks passed")
