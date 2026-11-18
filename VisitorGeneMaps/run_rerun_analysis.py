#!/usr/bin/env python3
"""Run the full rerun analysis: aggregate the sweep, then draw the overview maps.

The script does three things.
It finds the folder that holds the condition folders of the sweep.
It aggregates the finished runs into one summary table.
It draws the overview maps from that table and reports the microscopy correlation.

The first N finished runs of each condition are used, 60 by default.
Interrupted runs are skipped and replaced by later runs of the same condition.
Every condition then contributes six groups of ten runs, which gives 1,980 rows.

Usage:
    python run_rerun_analysis.py
    python run_rerun_analysis.py --n-runs 60
    python run_rerun_analysis.py --box-dir "/Volumes/Extreme Pro/.../box11"
    python run_rerun_analysis.py --skip-aggregation      # redraw the maps only
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DRIVE = Path("/Volumes/Extreme Pro/Analysis_EnhancerShoebox")
SUMMARY_NAME = "summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt"
AGGREGATE_SCRIPT = SCRIPT_DIR / "dist_genestages_grouped_5percentile_10xaveraging.py"
MAPS_SCRIPT = SCRIPT_DIR / "shoebox_summary_maps.py"
N_CONDITIONS = 330
GROUP_SIZE = 10


def check_packages():
    """Stop early when a required package is missing."""
    missing = []
    for name in ["numpy", "pandas", "matplotlib", "scipy", "sklearn"]:
        try:
            __import__(name)
        except ImportError:
            missing.append("scikit-learn" if name == "sklearn" else name)
    if missing:
        print("Missing Python packages: " + ", ".join(missing))
        print("Install them with:")
        print(f"  {sys.executable} -m pip install " + " ".join(missing))
        sys.exit(1)


def find_box_dir(drive):
    """Folders under the drive that contain condition folders, with their counts."""
    candidates = {}
    for root, dirs, _files in os.walk(drive):
        depth = len(Path(root).relative_to(drive).parts)
        if depth > 3:
            dirs[:] = []
            continue
        n = sum(1 for d in dirs if d.startswith("Control_Promoter"))
        if n:
            candidates[Path(root)] = n
            dirs[:] = [d for d in dirs if not d.startswith("Control_Promoter")]
    return candidates


def count_runs(box_dir):
    """Number of completion markers per condition, without reading the trajectories.

    This is a quick overview before the aggregation. The aggregation itself applies
    the stricter criterion of the marker plus 2,000 lines in geneTrack.txt.
    """
    counts = {}
    for cond in sorted(box_dir.glob("Control_Promoter*")):
        marker_dir = cond / "parallel_counter"
        if not marker_dir.is_dir():
            counts[cond.name] = 0
            continue
        numbers = [int(m.group(1)) for m in
                   (re.fullmatch(r"run(\d+)\.txt", f.name) for f in marker_dir.iterdir()) if m]
        counts[cond.name] = len(numbers)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive", default=str(DEFAULT_DRIVE),
                        help="folder that holds the sweep output (default: the external drive)")
    parser.add_argument("--box-dir", default=None,
                        help="folder with the condition folders, if the automatic search is wrong")
    parser.add_argument("--n-runs", type=int, default=60,
                        help="number of finished runs used per condition (default: 60)")
    parser.add_argument("--out-dir", default=None,
                        help="folder for the summary table and the figures "
                             "(default: <drive>/rerun_analysis_<n-runs>runs)")
    parser.add_argument("--skip-aggregation", action="store_true",
                        help="keep the existing summary table and only redraw the maps")
    args = parser.parse_args()

    check_packages()

    # --------------------------
    # 1. Locate the sweep output
    # --------------------------
    if args.box_dir:
        box_dir = Path(args.box_dir).resolve()
        if not box_dir.is_dir():
            sys.exit(f"Not a folder: {box_dir}")
    else:
        drive = Path(args.drive).resolve()
        if not drive.is_dir():
            sys.exit(f"Not a folder: {drive}\nIs the external drive connected?")
        candidates = find_box_dir(drive)
        if not candidates:
            sys.exit(f"No folder with Control_Promoter* condition folders found under {drive}")
        print("Folders with condition folders:")
        for path, n in sorted(candidates.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {path}")
        # The old partial sweep also sits on the drive. Prefer anything else.
        preferred = {p: n for p, n in candidates.items() if "new_simulations" not in p.parts}
        pool = preferred or candidates
        box_dir = max(pool, key=lambda p: pool[p])
        print(f"\nUsing: {box_dir}")
        if len(pool) > 1:
            print("More than one candidate. Pass --box-dir if this is the wrong one.")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else \
        Path(args.drive).resolve() / f"rerun_analysis_{args.n_runs}runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / SUMMARY_NAME

    # --------------------------
    # 2. Check how far the sweep got
    # --------------------------
    counts = count_runs(box_dir)
    if counts:
        values = list(counts.values())
        complete = sum(1 for v in values if v >= args.n_runs)
        print(f"\nCondition folders found: {len(counts)} of {N_CONDITIONS}")
        print(f"Finished runs per condition (completion markers): min {min(values)}, max {max(values)}")
        print(f"Conditions with at least {args.n_runs} finished runs: {complete} of {len(counts)}")
        usable_groups = min(min(values), args.n_runs) // GROUP_SIZE
        print(f"Groups of {GROUP_SIZE} available in every condition: {usable_groups}")
        if complete < len(counts):
            short = sorted((v, k) for k, v in counts.items() if v < args.n_runs)[:5]
            print("Conditions with the fewest runs:")
            for v, name in short:
                print(f"  {v:4d}  {name}")
            print("These conditions contribute fewer groups of ten.")

    # --------------------------
    # 3. Aggregate
    # --------------------------
    if args.skip_aggregation:
        if not summary_path.is_file():
            sys.exit(f"No summary table to reuse at {summary_path}")
        print(f"\nStep 1 skipped, reusing {summary_path}")
    else:
        print(f"\nStep 1: aggregating the first {args.n_runs} finished runs per condition into {summary_path.name}")
        print("This reads every trajectory once and takes a few minutes.")
        cmd = [sys.executable, str(AGGREGATE_SCRIPT),
               "--box-dir", str(box_dir),
               "--n-runs", str(args.n_runs),
               "--output", str(summary_path)]
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        if subprocess.run(cmd, cwd=str(SCRIPT_DIR)).returncode != 0:
            sys.exit("Aggregation failed.")

    # --------------------------
    # 4. Draw the maps
    # --------------------------
    print(f"\nStep 2: drawing the overview maps in {out_dir}")
    cmd = [sys.executable, str(MAPS_SCRIPT), "--out-dir", str(out_dir)]
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = dict(os.environ, MPLBACKEND=os.environ.get("MPLBACKEND", "Agg"))
    if subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env).returncode != 0:
        sys.exit("Map production failed.")

    print("\nFinished. Output files:")
    for name in ["shoebox_colormaps.svg", "shoebox_colormaps_preview.png",
                 "shoebox_scatter_maps.svg", "shoebox_scatter_maps_preview.png",
                 "mutual_information_knn_k3.csv", SUMMARY_NAME]:
        f = out_dir / name
        if f.is_file():
            print(f"  {f}  ({f.stat().st_size/1e3:.0f} kB)")


if __name__ == "__main__":
    main()
