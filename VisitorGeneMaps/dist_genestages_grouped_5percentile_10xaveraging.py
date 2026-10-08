# Generate summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt
# Averaging every 10 runs of a condition, in run-number order (run1-10, run11-20, ...)
#
# Only finished runs are used. A run is finished when parallel_counter/run<r>.txt
# exists and run<r>/geneTrack.txt has 2000 lines (same criterion as run_sweep.py).
# Unfinished runs, e.g. runs still in progress or interrupted, are ignored.
# Only complete groups of 10 runs enter the table. Leftover runs (fewer than 10)
# are skipped and reported.
#
# Usage:
#   python dist_genestages_grouped_5percentile_10xaveraging.py
#   python dist_genestages_grouped_5percentile_10xaveraging.py --max-repeats 50
#       (use only runs 1-50 of each condition, e.g. while the sweep is still running)
#   python dist_genestages_grouped_5percentile_10xaveraging.py --legacy
#       (time windows of the original analysis, for comparison only)
#
# All readouts (Ser5P, Ser2P, contact, activation distance, 5th-percentile
# distances) are evaluated over the same time window, from Python step 150
# (gene induction) to the end of the trajectory. --from-step changes the start.
# The original analysis averaged Ser5P and contact over all 2000 steps and all
# other readouts from step 150 on; --legacy reproduces that.

# %%
import argparse
import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# --------------------------
# Define parameters
# --------------------------
contact_dist = 250
n_steps = 2000          # Python steps per trajectory
t_induction = 150       # first Python step after induction
group_size = 10

promoters = [1, 2, 3]
activations = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100]
thresholds = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]

parser = argparse.ArgumentParser()
parser.add_argument('--box-dir', default='box11', help='folder with the condition folders (default: box11)')
parser.add_argument('--max-repeats', type=int, default=None,
                    help='use only runs with run number <= this value (default: all)')
parser.add_argument('--from-step', type=int, default=t_induction,
                    help='first Python step included in all readouts (default: 150, i.e. gene induction)')
parser.add_argument('--legacy', action='store_true',
                    help='use the time windows of the original analysis (Ser5P and contact over all steps, '
                         'other readouts from step 150); for comparison only')
parser.add_argument('--output', default='summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt')
args = parser.parse_args()


def finished_runs(root_folder):
    """Run numbers of finished runs in a condition folder, sorted by number."""
    numbers = []
    for name in os.listdir(root_folder):
        match = re.fullmatch(r'run(\d+)', name)
        if not match:
            continue
        r = int(match.group(1))
        if args.max_repeats is not None and r > args.max_repeats:
            continue
        marker = os.path.join(root_folder, 'parallel_counter', f'run{r}.txt')
        track = os.path.join(root_folder, name, 'geneTrack.txt')
        if not (os.path.isfile(marker) and os.path.isfile(track)):
            continue
        with open(track) as f:
            if sum(1 for _ in f) != n_steps:
                continue
        numbers.append(r)
    return sorted(numbers)


def run_statistics(track_file):
    data = np.loadtxt(track_file, delimiter=',')
    d_rp = data[:, 4]          # distance between promoter and enhancer
    d_rg = data[:, 5]          # distance between gene and enhancer
    s5p_promoter = data[:, 6]
    gene_state = data[:, 8].astype(int)

    if args.legacy:
        # Original analysis: Ser5P and contact over all steps, other readouts from step 150
        I = [x for x in range(1, len(gene_state)) if gene_state[x] == 2 and gene_state[x - 1] != 2]
        J = [x for x in range(1, len(gene_state)) if gene_state[x] == 2]
        return dict(
            S5PInt=np.mean(s5p_promoter),
            S2PInt=100 * len(J) / (len(gene_state) - t_induction),
            Contact=100 * np.sum(d_rp < contact_dist) / len(d_rp),
            DistActivation=np.mean(d_rp[I]) if len(I) > 0 else np.nan,
            RG5=np.percentile(d_rg[t_induction:], 5),
            RP5=np.percentile(d_rp[t_induction:], 5),
        )

    # One common time window for all readouts: steps t0 ... end of trajectory
    t0 = args.from_step
    window = range(max(t0, 1), len(gene_state))
    I = [x for x in window if gene_state[x] == 2 and gene_state[x - 1] != 2]   # activation events
    n_active = np.sum(gene_state[t0:] == 2)                                     # active steps
    n_window = len(gene_state) - t0

    return dict(
        S5PInt=np.mean(s5p_promoter[t0:]),
        S2PInt=100 * n_active / n_window,
        Contact=100 * np.sum(d_rp[t0:] < contact_dist) / n_window,
        DistActivation=np.mean(d_rp[I]) if len(I) > 0 else np.nan,
        RG5=np.percentile(d_rg[t0:], 5),
        RP5=np.percentile(d_rp[t0:], 5),
    )


# --------------------------
# Main processing loop
# --------------------------
rows = []
runs_used = {}
leftover = {}
for promoter in promoters:
    print(f"Promoter: {promoter}")
    for activation in activations:
        for threshold in thresholds:
            root_folder = os.path.join(args.box_dir, f'Control_Promoter{promoter}_Threshold{threshold}_Act{activation}')
            if not os.path.isdir(root_folder):
                continue
            numbers = finished_runs(root_folder)
            n_groups = len(numbers) // group_size
            runs_used[(promoter, threshold, activation)] = n_groups * group_size
            leftover[(promoter, threshold, activation)] = len(numbers) - n_groups * group_size

            for g in range(n_groups):
                group = numbers[g * group_size:(g + 1) * group_size]
                stats = pd.DataFrame([run_statistics(os.path.join(root_folder, f'run{r}', 'geneTrack.txt'))
                                      for r in group])
                means = stats.mean(skipna=True)
                rows.append([promoter, threshold, activation, f"{group[0]}-{group[-1]}",
                             means.S5PInt, means.S2PInt, means.Contact, means.DistActivation,
                             means.RG5, means.RP5])

df_summary = pd.DataFrame(rows, columns=[
    'Promoter', 'Threshold', 'Activation', 'Runs',
    'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
    '5percentRG', '5percentRP'
])

# --------------------------
# Report
# --------------------------
n_conditions = len(promoters) * len(activations) * len(thresholds)
counts = list(runs_used.values())
print(f"\nConditions with output folder: {len(runs_used)} of {n_conditions}")
if counts:
    print(f"Runs used per condition: min {min(counts)}, max {max(counts)}")
    if min(counts) != max(counts) or len(runs_used) < n_conditions:
        print("WARNING: conditions differ in the number of runs used, or conditions are missing.")
n_left = sum(leftover.values())
if n_left:
    print(f"Skipped {n_left} finished runs that did not complete a group of {group_size}.")
if args.legacy:
    print("Legacy time windows: Ser5P and contact over all steps, other readouts from step 150.")
else:
    print(f"All readouts evaluated from Python step {args.from_step} to the end of the trajectory.")

# --------------------------
# Save averaged summary
# --------------------------
df_summary.to_csv(args.output, index=False)
print(f"Averaged summary saved as: {args.output} ({len(df_summary)} rows)")
