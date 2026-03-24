#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# ------------------------------------------------------------
# ARGUMENT PARSING
# ------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--begin", required=True, help="start run (e.g. run12)")
parser.add_argument("-e", "--end", required=True, help="end run (e.g. run35)")
args = parser.parse_args()

run_start = int(args.begin.replace("run", ""))
run_end   = int(args.end.replace("run", ""))


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

WINDOW_SMALL = 10
WINDOW_LARGE = 500
EVENT_WINDOW = 200   # +/- frames around activation
MAX_LAG = 200


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

all_event_traces = []
all_event_times  = []
all_event_runs   = []

all_psi = []
all_act = []


# ------------------------------------------------------------
# LOOP OVER RUNS
# ------------------------------------------------------------

for run_nr in range(run_start, run_end + 1):

    run_path = f"./run{run_nr}"

    if not os.path.isdir(run_path):
        print(f"Skipping run{run_nr}: folder not found")
        continue

    print(f"Processing run{run_nr}")

    # files
    sph_file = os.path.join(run_path, "sphericity.txt")
    active_file   = os.path.join(run_path, f"dist_active_run{run_nr}.txt")
    induced_file  = os.path.join(run_path, f"dist_induced_run{run_nr}.txt")
    approach_file = os.path.join(run_path, f"dist_approaching_run{run_nr}.txt")
    recede_file   = os.path.join(run_path, f"dist_receding_run{run_nr}.txt")

    required = [sph_file, active_file, induced_file, approach_file, recede_file]

    if not all(os.path.exists(f) for f in required):
        print(f"Skipping run{run_nr}: missing required files")
        continue

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    sph_data = np.loadtxt(sph_file, delimiter=",", skiprows=1)

    timesteps = sph_data[:,0].astype(int)
    psi = sph_data[:,1]

    mask = timesteps >= 151
    timesteps = timesteps[mask]
    psi = psi[mask]

    # gene states
    active_frames   = set(np.loadtxt(active_file)[:,0].astype(int))
    induced_frames  = set(np.loadtxt(induced_file)[:,0].astype(int))
    approach_frames = set(np.loadtxt(approach_file)[:,0].astype(int))
    recede_frames   = set(np.loadtxt(recede_file)[:,0].astype(int))

    # ----------------------------------------------------------------------------------------------------
    # BUILD ACTIVATION SIGNAL (weights: 1 for active, 0 for induced and 0.5 for approaching and receding)
    # ---------------------------------------------------------------------------------------------------

    activation = []

    for t in timesteps:

        if t in active_frames:
            activation.append(1.0)

        elif t in approach_frames:
            activation.append(0.5)

        elif t in recede_frames:
            activation.append(0.5)

        elif t in induced_frames:
            activation.append(0.0)

        else:
            activation.append(0.0)

    activation = np.array(activation)

    # store full traces for ensemble correlation
    all_psi.append(psi)
    all_act.append(activation)

    # ------------------------------------------------------------------------
    # BASELINE CORRECTION: remove slow trends and keep only fast fluctuations
    # -----------------------------------------------------------------------

    psi_series = pd.Series(psi)

    psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
    psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()

    psi_corr = (psi_fast - psi_slow).fillna(0).to_numpy()

    # --------------------------------------------------------
    # FIND ACTIVATION EVENTS (induced → active)
    # --------------------------------------------------------

    sorted_active = sorted(active_frames)

    activation_starts = []

    for i in range(1, len(sorted_active)):
        if sorted_active[i] == sorted_active[i-1] + 1:
            activation_starts.append(sorted_active[i-1])

    if len(activation_starts) == 0 and len(sorted_active) > 0:
        activation_starts.append(sorted_active[0])

    # --------------------------------------------------------
    # EXTRACT EVENT-ALIGNED WINDOWS
    # --------------------------------------------------------

    for event in activation_starts:

        rel_t = []
        psi_evt = []

        for t, p in zip(timesteps, psi_corr):

            dt = t - event

            if -EVENT_WINDOW <= dt <= EVENT_WINDOW:
                rel_t.append(dt)
                psi_evt.append(p)

        if len(rel_t) > 0:
            all_event_traces.append(np.array(psi_evt))
            all_event_times.append(np.array(rel_t))
            all_event_runs.append(run_nr)


# ------------------------------------------------------------
# CHECK DATA
# ------------------------------------------------------------

if len(all_event_traces) == 0:
    print("No activation events found across runs.")
    sys.exit(1)


# ------------------------------------------------------------
# ALIGN EVENTS TO COMMON GRID
# ------------------------------------------------------------
#aligning psi around each activation: extract +/-200 frames around each activation event

common_time = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)

aligned_traces = []

for t_arr, psi_arr in zip(all_event_times, all_event_traces):

    interp = np.interp(common_time, t_arr, psi_arr)
    aligned_traces.append(interp)

aligned_traces = np.array(aligned_traces)


# ------------------------------------------------------------
# PLOT 1: RUN-AVERAGED EVENT TRAJECTORIES 
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

run_traces = []

for run_nr in range(run_start, run_end + 1):

    # collect events for this run
    indices = [i for i, r in enumerate(all_event_runs) if r == run_nr]

    if len(indices) == 0:
        continue

    run_events = aligned_traces[indices]

    # mean trajectory for this run
    run_mean = np.mean(run_events, axis=0)

    run_traces.append(run_mean)

    # plot one gray line per run
    plt.plot(
        common_time,
        run_mean,
        color="gray",
        alpha=0.8,
        linewidth=1
    )

run_traces = np.array(run_traces)

# global mean across runs
mean_trace = np.mean(run_traces, axis=0)

plt.plot(common_time, mean_trace, color="red", linewidth=3)

plt.axvline(0, color="black", linestyle="--")

plt.xlabel("Time relative to activation")
plt.ylabel(r'$\psi$ (baseline corrected)')

plt.title("Run-averaged sphericity")

plt.tight_layout()
plt.savefig("ensemble_event_overlay.svg")
plt.close()

# ------------------------------------------------------------
# PLOT 2: ENSEMBLE MEAN + STD
# ------------------------------------------------------------

std_trace = np.std(aligned_traces, axis=0)

plt.figure(figsize=(6,6))

plt.plot(common_time, mean_trace, color="black", linewidth=2)
plt.fill_between(common_time,
                 mean_trace - std_trace,
                 mean_trace + std_trace,
                 alpha=0.3)

plt.axvline(0, color="red", linestyle="--")

plt.xlabel("Time relative to activation")
plt.ylabel(r'$\psi$')

plt.title("Ensemble-averaged sphericity")

plt.tight_layout()
plt.savefig("ensemble_mean_band.svg")
plt.close()


# ------------------------------------------------------------
# PLOT 3: ENSEMBLE CROSS-CORRELATION
# ------------------------------------------------------------

# concatenate all runs (ensemble pooling)
psi_all = np.concatenate(all_psi)
act_all = np.concatenate(all_act)

# baseline correction on pooled signal
psi_series = pd.Series(psi_all)

psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()

psi_corr = (psi_fast - psi_slow).fillna(0).to_numpy()

# normalization (ensemble mean/variance)
psi_norm = psi_corr - np.mean(psi_corr)
act_norm = act_all - np.mean(act_all)

corr = np.correlate(psi_norm, act_norm, mode="full")

lags = np.arange(-len(psi_norm)+1, len(psi_norm))

corr = corr / (np.std(psi_norm) * np.std(act_norm) * len(psi_norm))

# restrict lag
mask = (lags >= -MAX_LAG) & (lags <= MAX_LAG)

lags = lags[mask]
corr = corr[mask]

# plot
plt.figure(figsize=(6,6))

plt.plot(lags, corr, color="black", linewidth=2)
plt.axvline(0, color="red", linestyle="--")

plt.xlabel("Time lag")
plt.ylabel("Cross-correlation")

plt.title("Ensemble-normalized cross-correlation")

plt.tight_layout()
plt.savefig("ensemble_cross_correlation.svg")
plt.close()

# ------------------------------------------------------------
# PLOT 4: ALL RUNS, BASELINE-CORRECTED PER-EVENT TRAJECTORIES
# ------------------------------------------------------------

plt.figure(figsize=(6,6))
# store interpolated event traces for global mean
all_event_traces_plot4 = []

for run_nr in range(run_start, run_end + 1):

    run_path = f"./run{run_nr}"

    if not os.path.isdir(run_path):
        continue

    sph_file = os.path.join(run_path, "sphericity.txt")
    active_file = os.path.join(run_path, f"dist_active_run{run_nr}.txt")

    if not os.path.exists(sph_file) or not os.path.exists(active_file):
        continue

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    sph_data = np.loadtxt(sph_file, delimiter=",", skiprows=1)

    timesteps = sph_data[:,0].astype(int)
    psi = sph_data[:,1]

    mask = timesteps >= 151
    timesteps = timesteps[mask]
    psi = psi[mask]

    # --------------------------------------------------------
    # BASELINE CORRECTION 
    # --------------------------------------------------------

    psi_series = pd.Series(psi)

    psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
    psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()

    psi_corr = (psi_fast - psi_slow).to_numpy()

    # --------------------------------------------------------
    # LOAD ACTIVE FRAMES
    # --------------------------------------------------------

    active_all = np.loadtxt(active_file)[:,0].astype(int)

    # --------------------------------------------------------
    # FIND TRUE EVENT STARTS
    # --------------------------------------------------------

    event_starts = []

    if len(active_all) > 0:
        event_starts.append(active_all[0])

        for i in range(1, len(active_all)):
            if active_all[i] != active_all[i-1] + 1:
                event_starts.append(active_all[i])

    # --------------------------------------------------------
    # EXTRACT AND PLOT EACH EVENT
    # --------------------------------------------------------

    for event in event_starts:

        t_min = event - EVENT_WINDOW
        t_max = event + EVENT_WINDOW

        mask = (timesteps >= t_min) & (timesteps <= t_max)

        t_rel = timesteps[mask] - event
        psi_rel = psi_corr[mask]

        if len(t_rel) == 0:
            continue

        plt.plot(
            t_rel,
            psi_rel,
            color="gray",
            linewidth=0.8,
            alpha=0.6
        )
        
        # clean + sort before interpolation
        valid = np.isfinite(t_rel) & np.isfinite(psi_rel)

        t_clean = t_rel[valid]
        psi_clean = psi_rel[valid]

        if len(t_clean) < 2:
            continue

        # ensure strictly increasing time
        order = np.argsort(t_clean)
        t_clean = t_clean[order]
        psi_clean = psi_clean[order]

        # remove duplicate time points
        t_unique, unique_idx = np.unique(t_clean, return_index=True)
        psi_unique = psi_clean[unique_idx]

        # interpolate
        interp = np.interp(common_time, t_unique, psi_unique)

        all_event_traces_plot4.append(interp)

# ------------------------------------------------------------

# compute and plot global mean trajectory (red line)
if len(all_event_traces_plot4) > 0:
    mean_trace_plot4 = np.mean(np.array(all_event_traces_plot4), axis=0)

    plt.plot(
        common_time,
        mean_trace_plot4,
        color="red",
        linewidth=3
    )

plt.axvline(0, color="black", linestyle="--")

plt.xlabel("Time relative to activation")
plt.ylabel(r'$\psi$ (baseline corrected)')
plt.title("All runs: per-event sphericity trajectories")

plt.tight_layout()
plt.savefig("ensemble_event_overlay_per_event.svg")
plt.close()

print("Ensemble analysis completed.")
