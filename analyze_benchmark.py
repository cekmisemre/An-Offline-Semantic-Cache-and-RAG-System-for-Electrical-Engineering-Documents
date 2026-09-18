"""Aggregates benchmark_results.csv into a per-configuration comparison table.

Why this exists: benchmark_results.csv accumulates every row from every run, so
comparing "was raising top_k worth it?" by hand means scrolling through hundreds of
rows. This groups by the run label (and optionally by run_id) and prints the numbers
side by side, which is also the shape the final report needs.

Usage (from the folder containing benchmark_results.csv):
    python analyze_benchmark.py                  # group by label
    python analyze_benchmark.py --by-run         # group by label + run_id
    python analyze_benchmark.py --csv other.csv  # a different file
"""

import csv
import sys
from collections import defaultdict


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_true(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    hits = [r for r in rows if is_true(r.get("cache_hit"))]
    misses = [r for r in rows if not is_true(r.get("cache_hit"))]
    graded = [r for r in rows if r.get("verdict")]
    correct = [r for r in graded if r["verdict"] == "correct"]
    bad_hits = [r for r in graded if is_true(r.get("cache_hit")) and r["verdict"] == "wrong"]

    hit_mean = mean([to_float(r.get("total_s")) for r in hits])
    miss_mean = mean([to_float(r.get("total_s")) for r in misses])

    return {
        "queries": len(rows),
        "repeats": len({r.get("repeat", "1") for r in rows}),
        "hit_rate": len(hits) / len(rows) * 100 if rows else 0.0,
        "hit_mean": hit_mean,
        "miss_mean": miss_mean,
        "speedup": miss_mean / hit_mean if hit_mean else 0.0,
        "llm": mean([to_float(r.get("llm_inference_s")) for r in misses]),
        "retrieval": mean([to_float(r.get("retrieval_s")) for r in misses]),
        "canonicalize": mean([to_float(r.get("canonicalize_s")) for r in misses]),
        "search": mean([to_float(r.get("cache_search_s")) for r in hits]),
        "accuracy": len(correct) / len(graded) * 100 if graded else 0.0,
        "graded": len(graded),
        "correct": len(correct),
        "bad_hits": len(bad_hits),
    }


def print_table(groups):
    header = f"{'configuration':<26} {'n':>4} {'hit%':>6} {'hit s':>7} {'miss s':>8} {'LLM s':>7} {'speedup':>8} {'acc%':>6} {'bad':>4}"
    print(header)
    print("-" * len(header))
    for name, rows in groups:
        s = summarize(rows)
        print(f"{name:<26} {s['queries']:>4} {s['hit_rate']:>6.1f} {s['hit_mean']:>7.3f} "
              f"{s['miss_mean']:>8.3f} {s['llm']:>7.3f} {s['speedup']:>7.0f}x "
              f"{s['accuracy']:>5.1f} {s['bad_hits']:>4}")

    print("\nColumns: n = rows, hit% = cache hit rate, hit s / miss s = mean response time,")
    print("LLM s = mean LLM inference on a miss, acc% = graded accuracy, bad = wrong cache hits.")


def print_detail(groups):
    for name, rows in groups:
        s = summarize(rows)
        print(f"\n=== {name} ===")
        print(f"  rows / repeats     : {s['queries']} / {s['repeats']}")
        print(f"  cache hit rate     : {s['hit_rate']:.1f}%")
        print(f"  mean response (hit): {s['hit_mean']:.3f}s   (semantic search {s['search']:.4f}s)")
        print(f"  mean response (miss): {s['miss_mean']:.3f}s")
        print(f"    - canonicalize   : {s['canonicalize']:.3f}s")
        print(f"    - RAG retrieval  : {s['retrieval']:.4f}s")
        print(f"    - LLM inference  : {s['llm']:.3f}s  ({s['llm'] / s['miss_mean'] * 100 if s['miss_mean'] else 0:.0f}% of a miss)")
        print(f"  hit vs miss        : {s['speedup']:.0f}x")
        print(f"  accuracy           : {s['accuracy']:.1f}%  ({s['correct']}/{s['graded']})")
        print(f"  wrong cache hits   : {s['bad_hits']}")


def print_per_query(rows, label):
    """Per-question accuracy within one configuration - shows WHICH questions fail,
    not just how many, so a change that fixes one question and breaks another is
    visible instead of averaging out."""
    by_q = defaultdict(list)
    for r in rows:
        if r.get("verdict"):
            by_q[int(r["query_no"])].append(r)
    if not by_q:
        return
    print(f"\n--- per-question accuracy: {label} ---")
    for q_no in sorted(by_q):
        group = by_q[q_no]
        correct = sum(1 for r in group if r["verdict"] == "correct")
        flag = ""
        if correct == 0:
            flag = "  <-- always wrong"
        elif correct < len(group):
            flag = "  <-- inconsistent"
        print(f"  Q{q_no:<3} {correct}/{len(group)}   {group[0]['query'][:52]}...{flag}")


def main():
    csv_path = "benchmark_results.csv"
    by_run = "--by-run" in sys.argv
    if "--csv" in sys.argv:
        csv_path = sys.argv[sys.argv.index("--csv") + 1]

    try:
        rows = load_rows(csv_path)
    except FileNotFoundError:
        print(f"Could not find {csv_path} - run the pipeline first, or pass --csv <path>.")
        return
    if not rows:
        print(f"{csv_path} is empty.")
        return

    buckets = defaultdict(list)
    for r in rows:
        key = f"{r.get('label') or 'unlabeled'}" + (f" @ {r.get('run_id')}" if by_run else "")
        buckets[key].append(r)

    # Sort by the first run_id seen, so configurations appear in the order they were run
    groups = sorted(buckets.items(), key=lambda kv: min(r.get("run_id", "") for r in kv[1]))

    print(f"Loaded {len(rows)} rows from {csv_path}\n")
    print_table(groups)
    print_detail(groups)
    for name, group_rows in groups:
        print_per_query(group_rows, name)


if __name__ == "__main__":
    main()
