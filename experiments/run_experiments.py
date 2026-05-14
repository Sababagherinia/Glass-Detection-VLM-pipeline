#!/usr/bin/env python3
"""Experiment runner for Glass-Detection-VLM-pipeline.

Run predefined pipeline variants and collect runtime/memory summaries.

This script intentionally does not modify the core pipeline; it launches
existing entrypoints and collects stdout/stderr plus wall-clock and
optional memory-peak (requires `psutil`). Results are written to
`<out>/<experiment_name>/` and a CSV summary `<out>/experiments_summary.csv`.

Usage (dry run):
  python experiments/run_experiments.py --out output/experiments --dry-run

Run for real:
  python experiments/run_experiments.py --out output/experiments

Adjust the EXPERIMENTS list below to customize commands.
"""
import argparse
import csv
import os
import shlex
import subprocess
import sys
import time

try:
    import psutil
except Exception:
    psutil = None

BASE_PY = sys.executable

# Simple experiment definitions. Update commands if your CLI flags differ.
EXPERIMENTS = [
    {
        "name": "combined_map_full",
        "desc": "Full unified pipeline (combined occupancy + semantics)",
        "cmd": f"{BASE_PY} unified_pipeline.py --dataset data/rgbd_dataset_freiburg1_360 --output output/test_experiment --frame-step 5 --quiet",
    },
    {
        "name": "geometric_only",
        "desc": "Geometric-only mapping (no semantics)",
        "cmd": f"{BASE_PY} unified_pipeline.py --dataset data/rgbd_dataset_freiburg1_360 --output output/test_experiment --no-semantics --frame-step 5 --quiet",
    },
    {
        "name": "semantic_only",
        "desc": "Semantic mapping only",
        "cmd": f"{BASE_PY} unified_pipeline.py --dataset data/rgbd_dataset_freiburg1_360 --output output/test_experiment --semantics-only --frame-step 5 --quiet",
    },
]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def run_and_measure(cmd, timeout=None):
    start = time.perf_counter()
    proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    mem_peak = None
    try:
        if psutil:
            p = psutil.Process(proc.pid)
            mem_peak = 0
            while proc.poll() is None:
                try:
                    m = p.memory_info().rss
                    mem_peak = max(mem_peak, m)
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.05)
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
    end = time.perf_counter()
    return {
        "returncode": proc.returncode,
        "elapsed_s": end - start,
        "mem_peak_bytes": mem_peak,
        "stdout": (stdout.decode(errors="ignore") if stdout else ""),
        "stderr": (stderr.decode(errors="ignore") if stderr else ""),
    }


def summarize(results, csv_path):
    ensure_dir(os.path.dirname(csv_path))
    keys = ["name", "desc", "cmd", "returncode", "elapsed_s", "mem_peak_bytes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in keys})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="output/experiments", help="output folder to write logs and summary")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--timeout", type=int, default=0, help="per-run timeout in seconds (0 = none)")
    args = p.parse_args()

    ensure_dir(args.out)
    results = []
    for e in EXPERIMENTS:
        print(f"=== Experiment: {e['name']} — {e['desc']}")
        print(f"  cmd: {e['cmd']}")
        rec = {"name": e["name"], "desc": e.get("desc", ""), "cmd": e["cmd"]}
        if args.dry_run:
            rec.update({"returncode": None, "elapsed_s": None, "mem_peak_bytes": None})
            results.append(rec)
            continue
        res = run_and_measure(e["cmd"], timeout=(args.timeout or None))
        rec.update(res)
        results.append(rec)

        base = os.path.join(args.out, e["name"])
        ensure_dir(base)
        with open(os.path.join(base, "stdout.log"), "w") as f:
            f.write(res.get("stdout", ""))
        with open(os.path.join(base, "stderr.log"), "w") as f:
            f.write(res.get("stderr", ""))

    csv_path = os.path.join(args.out, "experiments_summary.csv")
    summarize(results, csv_path)
    print("Summary written to:", csv_path)


if __name__ == "__main__":
    main()
