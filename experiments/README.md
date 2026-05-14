# Experiments (quick guide)

This folder contains lightweight utilities to run evaluation variants and
collect compact summaries suitable for thesis reporting without touching
the core pipeline.

Files:

- `run_experiments.py` — Launches predefined pipeline variants, collects
  stdout/stderr and records wall-clock and optional memory-peak (requires `psutil`).
- `analysis_tools.py` — Small helpers to aggregate output files and the
  generated `experiments_summary.csv` into compact CSVs for reporting.

Quick steps

1. Dry-run to inspect commands:

```bash
python experiments/run_experiments.py --out output/experiments --dry-run
```

2. Run the experiments (may take time):

```bash
python experiments/run_experiments.py --out output/experiments
```

3. Produce a compact experiments table for inclusion in your report:

```bash
python -c "from experiments import analysis_tools; analysis_tools.summarize_experiments_summary('output/experiments/experiments_summary.csv','output/experiments/experiments_report.csv')"
```

4. Optionally gather file-size stats for produced maps:

```bash
python -c "from experiments import analysis_tools; import glob; dirs=glob.glob('output/experiments/*/'); analysis_tools.collect_file_stats(dirs,'output/experiments/file_stats.csv')"
```

Notes

- If you want per-frame timings or voxel-level metrics, export necessary
  logs or map files from the pipeline; the `analysis_tools` can be
  extended to parse those artifacts. The current helpers are intentionally
  minimal so they work with whatever outputs your pipeline produces.
