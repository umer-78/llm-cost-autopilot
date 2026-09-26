"""Every measurement in the README, on HELM questions split in half by a hash of their id:
the router learns from one half and is measured on the other.

Two registries: the five models from US providers (GPT-4o, GPT-4o-mini, Llama 3.1 70B and
8B, Gemini 1.5 Flash), and the same plus DeepSeek-V3, for teams whose rules allow it.
Everything is compared with sending every request to GPT-4o.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import data
from .router import Registry, Router, out_of_fold, pick

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REGISTRIES = {"US providers": ["gpt-4o", "gpt-4o-mini", "llama-3.1-70b", "gemini-1.5-flash", "llama-3.1-8b"],
              "any provider": None}
PARITY = 0.005          # tune for accuracy no more than half a point under the frontier's, on training data
GRID = np.round(np.arange(0.40, 0.981, 0.02), 2)


def keys_and_split(table):
    keys = sorted(table)
    half = np.array([int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) % 2 for k in keys])
    return keys, half == 0, half == 1


def features(table, keys):
    """Embedding of the prompt, the task that sent it, and its length in words."""
    path = data.cache_dir() / "features.npy"
    if path.exists() and np.load(path).shape[0] == len(keys):
        emb = np.load(path)
    else:
        from .embed import OnnxEmbedder
        emb = OnnxEmbedder("bge-small").encode([table[k]["question"][:2000] for k in keys])
        np.save(path, emb)
    tasks = sorted({table[k]["task"] for k in keys})
    onehot = np.array([[table[k]["task"] == t for t in tasks] for k in keys], float)
    length = np.log1p([[len(table[k]["question"].split())] for k in keys]) / 8
    return np.hstack([emb, onehot, length]), tasks


def paired_ci(a, b, draws=2000, seed=0):
    """95% bootstrap interval for mean(a) - mean(b) over the same questions."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(draws)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def outcome(chosen, rows, right, cost):
    idx = np.arange(len(chosen))
    acc = np.array([right[m][r] for m, r in zip(chosen, rows)])
    spend = np.array([cost[m][r] for m, r in zip(chosen, rows)])
    return acc, spend, idx


def run_registry(table, keys, x, train, test, only):
    config = data.registry()
    reg = Registry.from_config(config, only)
    f = reg.frontier
    right = {n: np.array([table[k]["models"][n]["correct"] for k in keys], float) for n in reg.names}
    agree = {n: np.array([table[k]["models"][n]["answer"] == table[k]["models"][f]["answer"] for k in keys], float) for n in reg.names}
    cost = {n: np.array([reg.cost(n, table[k]["models"][n]["prompt_tokens"], table[k]["models"][n]["output_tokens"]) for k in keys]) for n in reg.names}
    te, tr = np.where(test)[0], np.where(train)[0]
    per1k = lambda spend: round(1000 * float(np.mean(spend)), 3)
    frontier_acc = right[f][te]
    out = {"models": reg.names, "frontier": f, "test_questions": int(len(te)),
           "single": {n: {"accuracy": round(float(right[n][te].mean()), 4), "per_1000": per1k(cost[n][te])} for n in reg.names}}

    # the oracle: the cheapest model that got it right (the frontier when none did)
    chosen = [next((n for n in reg.names if right[n][i]), f) for i in te]
    out["oracle"] = {"accuracy": round(float(np.mean([right[m][i] for m, i in zip(chosen, te)])), 4),
                     "per_1000": per1k([cost[m][i] for m, i in zip(chosen, te)])}

    # a fixed map from task to model, the spec's routing table, chosen on training questions
    tasks = sorted({table[k]["task"] for k in keys})
    task_of = np.array([table[k]["task"] for k in keys])
    mapping = {}
    for t in tasks:
        rows = np.where(train & (task_of == t))[0]
        target = right[f][rows].mean() - PARITY
        mapping[t] = next((n for n in reg.names if right[n][rows].mean() >= target), f)
    chosen = [mapping[task_of[i]] for i in te]
    acc = [right[m][i] for m, i in zip(chosen, te)]
    out["task_map"] = {"map": mapping, "accuracy": round(float(np.mean(acc)), 4), "per_1000": per1k([cost[m][i] for m, i in zip(chosen, te)])}

    # the learned router: confidence chosen on out-of-fold chances over the training half
    oof = out_of_fold(reg, x[tr], {n: right[n][tr] for n in reg.names})
    target = right[f][tr].mean() - PARITY
    curve_train = []
    for c in GRID:
        m = pick(reg, oof, c)
        curve_train.append((float(c), float(np.mean([right[n][i] for n, i in zip(m, tr)]))))
    confidence = min((c for c, a in curve_train if a >= target), default=float(GRID[-1]))
    router = Router(reg, confidence).fit(x[tr], {n: right[n][tr] for n in reg.names})
    chances = router.chances(x[te])
    curve = []
    for c in GRID:
        m = pick(reg, chances, c)
        a = [right[n][i] for n, i in zip(m, te)]
        curve.append({"confidence": float(c), "accuracy": round(float(np.mean(a)), 4), "per_1000": per1k([cost[n][i] for n, i in zip(m, te)])})
    m = pick(reg, chances, confidence)
    acc = np.array([right[n][i] for n, i in zip(m, te)])
    spend = np.array([cost[n][i] for n, i in zip(m, te)])
    share = {n: round(float(np.mean(m == n)), 4) for n in reg.names}
    by_task = {t: {"router": round(float(acc[task_of[te] == t].mean()), 4), "frontier": round(float(frontier_acc[task_of[te] == t].mean()), 4),
                   "per_1000": per1k(spend[task_of[te] == t]), "frontier_per_1000": per1k(cost[f][te][task_of[te] == t])} for t in tasks}
    out["router"] = {"confidence": confidence, "accuracy": round(float(acc.mean()), 4), "per_1000": per1k(spend),
                     "saving": round(1 - float(spend.sum() / cost[f][te].sum()), 4), "vs_frontier_ci": paired_ci(acc, frontier_acc),
                     "share": share, "by_task": by_task, "curve": curve, "agrees_with_frontier": round(float(np.mean([agree[n][i] for n, i in zip(m, te)])), 4)}

    # a cascade with a verifier: two cheaper models answer; if they agree that answer stands,
    # otherwise the frontier answers too and its answer is used. The pair is chosen on the
    # training half: the cheapest that keeps accuracy at parity, else the most accurate.
    def cascade(rows, a, b):
        same = np.array([table[keys[i]]["models"][a]["answer"] == table[keys[i]]["models"][b]["answer"] for i in rows])
        return same, np.where(same, right[a][rows], right[f][rows]), cost[a][rows] + cost[b][rows] + np.where(same, 0, cost[f][rows])
    pairs = [(a, b) for i, a in enumerate(reg.names) for b in reg.names[i + 1:] if f not in (a, b)]
    scored = [(p, *cascade(tr, *p)[1:]) for p in pairs]
    ok = [(p, s.mean()) for p, acc_, s in scored if acc_.mean() >= right[f][tr].mean() - PARITY]
    a, b = min(ok, key=lambda o: o[1])[0] if ok else max(scored, key=lambda o: o[1].mean())[0]
    same = cascade(te, a, b)[0]
    acc = np.where(same, right[a][te], right[f][te])
    spend = cost[a][te] + cost[b][te] + np.where(same, 0, cost[f][te])
    out["cascade"] = {"pair": [a, b], "accuracy": round(float(acc.mean()), 4), "per_1000": per1k(spend), "escalated": round(float(1 - same.mean()), 4),
                      "vs_frontier_ci": paired_ci(acc, frontier_acc)}
    return out


def run():
    t0 = time.time()
    table = data.table()
    keys, train, test = keys_and_split(table)
    x, tasks = features(table, keys)
    out = {"questions": len(keys), "tasks": {t: sum(table[k]["task"] == t for k in keys) for t in tasks},
           "prices": {n: {"input": m["input_per_million"], "output": m["output_per_million"]} for n, m in data.registry()["models"].items()},
           "registries": {name: run_registry(table, keys, x, train, test, only) for name, only in REGISTRIES.items()}}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def pct(v):
    return f"{100 * v:.1f}%"


def report(r):
    lines = []
    for name, g in r["registries"].items():
        f = g["frontier"]
        base = g["single"][f]
        lines += [f"### {name} ({', '.join(g['models'])})", "",
                  "| Strategy | Accuracy | $ per 1,000 requests | Saving vs " + f + " |", "|---|---|---|---|",
                  f"| everything to {f} | {pct(base['accuracy'])} | {base['per_1000']:.3f} | - |"]
        rows = [("learned router (confidence " + str(g["router"]["confidence"]) + ")", g["router"]),
                ("cascade: " + " + ".join(g["cascade"]["pair"]) + ", escalate when they disagree", g["cascade"]),
                ("fixed task-to-model map", g["task_map"]), ("oracle (cheapest model that was right)", g["oracle"])]
        rows += [(f"everything to {n}", g["single"][n]) for n in g["models"] if n != f]
        for label, v in rows:
            lines.append(f"| {label} | {pct(v['accuracy'])} | {v['per_1000']:.3f} | {pct(1 - v['per_1000'] / base['per_1000'])} |")
        rt = g["router"]
        lines += ["", f"Router against {f} on the same questions: {100 * (rt['accuracy'] - base['accuracy']):+.1f} points "
                      f"(95% interval {100 * rt['vs_frontier_ci'][0]:+.1f} to {100 * rt['vs_frontier_ci'][1]:+.1f}); "
                      f"traffic share {', '.join(f'{n} {pct(s)}' for n, s in rt['share'].items() if s)}.", ""]
    return "\n".join(lines)


def save(r, path=RESULTS / "bench.json"):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(r, indent=1, default=float))
