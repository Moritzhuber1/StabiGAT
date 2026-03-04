#!/usr/bin/env python3
from pathlib import Path
import json
import math

RUNS_DIR = Path("runs")

def get(d, path, default=float("nan")):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

rows = []
for s in RUNS_DIR.glob("**/summary.json"):
    try:
        summary = json.loads(s.read_text())
        rows.append({
            "path": str(s.parent),
            "method": s.parent.parent.name,   # sss/skf/gkf
            "run": s.parent.name,
            "micro_f1": float(get(summary, "f1")),
            "acc": float(get(summary, "accuracy")),
            "mean_f1": float(get(summary, "folds.mean_f1")),
            "std_f1": float(get(summary, "folds.std_f1")),
            "n": int(get(summary, "n_samples", 0)),
        })
    except Exception:
        pass

# sort: primär mean_f1, sekundär micro_f1
rows.sort(key=lambda r: (r["mean_f1"] if not math.isnan(r["mean_f1"]) else -1e9,
                         r["micro_f1"] if not math.isnan(r["micro_f1"]) else -1e9),
          reverse=True)

print("\nTOP 20 (über alle Methoden):")
for r in rows[:20]:
    print(f"{r['mean_f1']:.4f} ± {r['std_f1']:.4f} | micro_f1={r['micro_f1']:.4f} | acc={r['acc']:.4f} | {r['method']} | {r['run']}")

print("\nBEST PRO METHODE:")
for m in ["sss","skf","gkf"]:
    cand = [r for r in rows if r["method"] == m]
    if not cand:
        print(m, "— keine runs")
        continue
    best = cand[0]
    print(f"{m}: mean_f1={best['mean_f1']:.4f} ± {best['std_f1']:.4f} | micro_f1={best['micro_f1']:.4f} | {best['run']}")