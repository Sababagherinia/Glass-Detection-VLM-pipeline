"""Simple analysis helpers for experiment outputs.

This module provides small utilities to aggregate experiment outputs into
compact CSV summaries that are easy to include in a thesis. It does not
depend on the core pipeline internals — it reads files and the
`experiments_summary.csv` produced by `run_experiments.py`.
"""
import csv
import os
from pathlib import Path


def collect_file_stats(root_dirs, out_csv):
    """Collect basic stats (file size, existence) for common output files.

    root_dirs: iterable of experiment output directories
    out_csv: path to write aggregated CSV
    """
    rows = []
    for d in root_dirs:
        p = Path(d)
        row = {"experiment": p.name}
        combined_ot = p / "combined_map.ot"
        combined_bt = p / "combined_map.bt"
        combined = combined_ot if combined_ot.exists() else combined_bt
        geom = p / "geometric_map.bt"
        sem = p / "semantic_only_map.bt"
        for label, fp in [("combined", combined), ("geometric", geom), ("semantic", sem)]:
            row[f"{label}_exists"] = fp.exists()
            row[f"{label}_size_bytes"] = fp.stat().st_size if fp.exists() else None
        rows.append(row)

    keys = ["experiment", "combined_exists", "combined_size_bytes", "geometric_exists", "geometric_size_bytes", "semantic_exists", "semantic_size_bytes"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_experiments_summary(summary_csv, out_csv):
    """Read experiments_summary.csv and produce a compact CSV used for reporting.

    Keeps name, returncode, elapsed_s, mem_peak_bytes.
    """
    out_rows = []
    with open(summary_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out_rows.append({
                "name": row.get("name"),
                "returncode": row.get("returncode"),
                "elapsed_s": row.get("elapsed_s"),
                "mem_peak_bytes": row.get("mem_peak_bytes"),
            })
    keys = ["name", "returncode", "elapsed_s", "mem_peak_bytes"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)


if __name__ == "__main__":
    print("Quick helpers for experiment outputs. Use from CLI or import in notebooks.")
