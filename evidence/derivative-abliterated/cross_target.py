#!/usr/bin/env python3
"""Two drafters on a base target and on a derivative of it. Standard library only.

    python3 cross_target.py BASE_STOCK BASE_TUNED DERIVATIVE_STOCK DERIVATIVE_TUNED [--out FILE]

`bench/analysis/analyse_served_accept.py` refuses to pair reports whose settings differ, and
the target is a setting, so it cannot answer whether a drafter still helps on a derivative.
This reads the same four reports and answers two questions with the analyser's own
resampling (tokens per round = all tokens over all rounds; a paired prompt bootstrap
stratified by category x thinking, seed 7, 10,000 draws, 2.5th and 97.5th percentiles):

- per drafter, the change in tokens per round from base to derivative, with how many prompts
  produced the same greedy output within the budget on both targets and where the rest first
  differed;
- the tuned drafter's gain over stock on the derivative minus the same gain on the base, from
  one joint bootstrap over all four reports.

All four reports must have the same prompts, in the same order, and the same settings apart
from `target`; each pair of reports must have the same drafter.
"""
import argparse
import json
import random
import sys

DRAWS = 10000


def load(path):
    with open(path) as f:
        return json.load(f)


def acceptance(rows):
    return sum(r["tokens"] for r in rows) / sum(r["rounds"] for r in rows)


def strata(rows):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault((row["category"], row["thinking"]), []).append(i)
    return groups


def interval(statistic, groups):
    rng = random.Random(7)
    samples = sorted(statistic([rng.choice(g) for g in groups.values() for _ in g]) for _ in range(DRAWS))
    return [samples[int(DRAWS * .025)], samples[min(DRAWS - 1, int(DRAWS * .975))]]


def first_difference(a, b):
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return k
    return min(len(a), len(b)) if len(a) != len(b) else None


def change(base, derivative):
    left, right = base["requests"], derivative["requests"]
    budget = base["settings"]["max_new"]
    same = [i for i, (x, y) in enumerate(zip(left, right))
            if x["response_ids"][:budget] == y["response_ids"][:budget]
            and (x["finish"] == "length") == (y["finish"] == "length")]
    firsts = sorted(k for k in (first_difference(x["response_ids"][:budget], y["response_ids"][:budget])
                                for x, y in zip(left, right)) if k is not None)
    return {
        "drafter": base["drafter"],
        "base_tokens_per_round": acceptance(left),
        "derivative_tokens_per_round": acceptance(right),
        "relative_change": acceptance(right) / acceptance(left) - 1,
        "paired_95_interval": interval(
            lambda idx: acceptance([right[i] for i in idx]) / acceptance([left[i] for i in idx]) - 1,
            strata(left)),
        "base_truncated": sum(r["finish"] == "length" for r in left),
        "derivative_truncated": sum(r["finish"] == "length" for r in right),
        "prompts": len(left),
        "prompts_same_output_within_budget": len(same),
        "first_difference_token": ({"min": firsts[0], "median": firsts[len(firsts) // 2], "max": firsts[-1]}
                                   if firsts else None),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in ("base_stock", "base_tuned", "derivative_stock", "derivative_tuned"):
        ap.add_argument(name)
    ap.add_argument("--out")
    args = ap.parse_args()
    bs, bt, ds, dt = (load(p) for p in (args.base_stock, args.base_tuned,
                                        args.derivative_stock, args.derivative_tuned))
    for report in (bs, bt, ds, dt):
        if not report.get("complete"):
            sys.exit("every report must be complete")
    settings = [{k: v for k, v in r["settings"].items() if k != "target"} for r in (bs, bt, ds, dt)]
    if any(s != settings[0] for s in settings):
        sys.exit("settings other than the target differ")
    keys = [[(q["prompt_sha256"], q["category"], q["thinking"]) for q in r["requests"]] for r in (bs, bt, ds, dt)]
    if any(k != keys[0] for k in keys):
        sys.exit("prompts, order or strata differ")
    if bs["drafter"] != ds["drafter"] or bt["drafter"] != dt["drafter"]:
        sys.exit("a drafter differs between the two targets")

    def gain(stock, tuned, idx):
        return acceptance([tuned[i] for i in idx]) / acceptance([stock[i] for i in idx]) - 1

    B, T, D, E = (r["requests"] for r in (bs, bt, ds, dt))
    every = range(len(B))
    gain_base, gain_derivative = gain(B, T, every), gain(D, E, every)
    out = {
        "base_target": bs["settings"]["target"],
        "derivative_target": ds["settings"]["target"],
        "settings": settings[0],
        "stock": change(bs, ds),
        "tuned": change(bt, dt),
        "gain_difference": {
            "gain_base": gain_base,
            "gain_derivative": gain_derivative,
            "difference": gain_derivative - gain_base,
            "difference_95_interval": interval(lambda idx: gain(D, E, idx) - gain(B, T, idx), strata(B)),
            "share_of_base_gain_kept": gain_derivative / gain_base,
        },
    }
    text = json.dumps(out, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
