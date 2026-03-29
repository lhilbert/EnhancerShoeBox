#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Treatment of event: this script defines one activation event as everything between the end of 'approaching' and the beginning of 'receding' phase; one activation event can last for several (up till ca. 100) time frames.
# The first 150 time frames (equilibration part) are excluded from data analysis and plotting.
# Binary treatment of activation: active = 1.0, not active (induced, receding, approaching) = 0.0

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
EVENT_WINDOW = 200
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

    mask = timesteps >= 151 #remove the first 150 frames (equilibration phase)
    timesteps = timesteps[mask]
    psi = psi[mask]

    active_frames   = set(np.loadtxt(active_file)[:,0].astype(int))
    induced_frames  = set(np.loadtxt(induced_file)[:,0].astype(int))
    approach_frames = set(np.loadtxt(approach_file)[:,0].astype(int))
    recede_frames   = set(np.loadtxt(recede_file)[:,0].astype(int))

    # --------------------------------------------------------
    # BUILD ACTIVATION SIGNAL
    # --------------------------------------------------------

    activation = np.array([1.0 if t in active_frames else 0.0 for t in timesteps])

    all_psi.append(psi)
    all_act.append(activation)

    # --------------------------------------------------------
    # BASELINE CORRECTION
    # --------------------------------------------------------

    psi_series = pd.Series(psi)

    psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
    psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()

    psi_corr = (psi_fast - psi_slow).fillna(0).to_numpy()

    # --------------------------------------------------------
    # FIND ACTIVATION EVENTS
    # --------------------------------------------------------

    sorted_active = sorted(active_frames)
    activation_events = []

    if len(sorted_active) > 0:

        start = sorted_active[0]
        prev  = sorted_active[0]

        for i in range(1, len(sorted_active)):

            if sorted_active[i] == prev + 1:
                prev = sorted_active[i]
            else:
                end = prev

                # activation event start at the first time frame when the gene is active
                activation_events.append(start)

                start = sorted_active[i]
                prev  = sorted_active[i]

        # last block
        end = prev
        activation_events.append(start)

    # --------------------------------------------------------
    # PRINT PER-RUN EVENTS
    # --------------------------------------------------------

    print(f"run{run_nr}, number of events per run: {len(activation_events)}")

    # --------------------------------------------------------
    # EXTRACT EVENT WINDOWS
    # --------------------------------------------------------

    for event_onset in activation_events:

        rel_t = []
        psi_evt = []

        for t, p in zip(timesteps, psi_corr):

            dt = t - event_onset   # alignment at activation onset

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
# ALIGN EVENTS
# ------------------------------------------------------------

common_time = np.arange(-EVENT_WINDOW, EVENT_WINDOW + 1)

aligned_traces = []

for t_arr, psi_arr in zip(all_event_times, all_event_traces):
    interp = np.interp(common_time, t_arr, psi_arr)
    aligned_traces.append(interp)

aligned_traces = np.array(aligned_traces)


# ------------------------------------------------------------
# PLOT 1: events averaged peer run, plotted averaged sphericity for each run
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

run_traces = []
plotted_runs = []

all_event_runs_arr = np.array(all_event_runs)
unique_runs = sorted(set(all_event_runs_arr))

print("\n--- DEBUG: EVENTS PER RUN ---")

for run_nr in unique_runs:

    indices = np.where(all_event_runs_arr == run_nr)[0]
    n_events = len(indices)

    print(f"run{run_nr}: {n_events} detected events")

    if n_events == 0:
        continue

    run_events = aligned_traces[indices]

    run_mean = np.mean(run_events, axis=0)

    run_traces.append(run_mean)
    plotted_runs.append(run_nr)

    plt.plot(common_time, run_mean, color="gray", alpha=0.8, linewidth=1)

print(f"\n--- DEBUG: NUMBER OF RUNS PLOTTED = {len(plotted_runs)} ---\n")

if len(run_traces) > 0:
    run_traces = np.array(run_traces)

    mean_trace = np.mean(aligned_traces, axis=0)

    plt.plot(common_time, mean_trace, color="red", linewidth=3)

plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Time relative to activation onset")
plt.ylabel(r'$\psi$ (baseline corrected)')
plt.title("Run-averaged sphericity")

plt.tight_layout()
plt.savefig("ensemble_event_overlay_corrected.svg")
plt.close()

# ------------------------------------------------------------
# PLOT 2: mean trajectory and standard deviation
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
plt.ylabel(r'$\psi$ (baseline corrected)')

plt.title("Ensemble-averaged sphericity")

plt.tight_layout()
plt.savefig("ensemble_mean_band_corrected.svg")
plt.close()

# ------------------------------------------------------------
# PLOT 3: ENSEMBLE CROSS-CORRELATION (CALCULATED PER RUN and AVERAGED AFTERWARDS)
# ------------------------------------------------------------

all_corrs = []

for psi, act in zip(all_psi, all_act):

    # baseline correction
    psi_series = pd.Series(psi)

    psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
    psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()

    psi_corr = (psi_fast - psi_slow).fillna(0).to_numpy()

    # normalization
    psi_norm = psi_corr - np.mean(psi_corr)
    act_norm = act - np.mean(act)

    # correlation
    corr = np.correlate(psi_norm, act_norm, mode="full")

    lags = np.arange(-len(psi_norm)+1, len(psi_norm))

    corr = corr / (np.std(psi_norm) * np.std(act_norm) * len(psi_norm))

    # restrict lag
    mask = (lags >= -MAX_LAG) & (lags <= MAX_LAG)

    all_corrs.append(corr[mask])

# convert to array
all_corrs = np.array(all_corrs)

# average over runs
mean_corr = np.mean(all_corrs, axis=0)

lags = np.arange(-MAX_LAG, MAX_LAG + 1)

# plot
plt.figure(figsize=(6,6))

plt.plot(lags, mean_corr, color="black", linewidth=2)
plt.axvline(0, color="red", linestyle="--")

plt.xlabel("Time lag")
plt.ylabel("Cross-correlation")

plt.title("Ensemble-normalized cross-correlation (per run avg)")

plt.tight_layout()
plt.savefig("ensemble_cross_correlation_corrected.svg")
plt.close()

# ------------------------------------------------------------
# PLOT 4: ALL EVENTS (no per run averaging)
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

all_event_traces_plot4 = []

for run_nr in range(run_start, run_end + 1):

    run_path = f"./run{run_nr}"

    if not os.path.isdir(run_path):
        continue

    sph_file = os.path.join(run_path, "sphericity.txt")
    active_file = os.path.join(run_path, f"dist_active_run{run_nr}.txt")

    if not os.path.exists(sph_file) or not os.path.exists(active_file):
        continue

    # load data
    sph_data = np.loadtxt(sph_file, delimiter=",", skiprows=1)
    timesteps = sph_data[:,0].astype(int)
    psi = sph_data[:,1]

    mask = timesteps >= 151
    timesteps = timesteps[mask]
    psi = psi[mask]

    # baseline correction
    psi_series = pd.Series(psi)
    psi_fast = psi_series.rolling(window=WINDOW_SMALL, center=True).mean()
    psi_slow = psi_series.rolling(window=WINDOW_LARGE, center=True).mean()
    psi_corr = (psi_fast - psi_slow).fillna(0).to_numpy()

    # load active frames
    active_all = np.loadtxt(active_file)[:,0].astype(int)
    active_all = np.sort(active_all)

    # detect blocks
    events = []
    if len(active_all) > 0:
        start = active_all[0]
        prev = active_all[0]

        for i in range(1, len(active_all)):
            if active_all[i] == prev + 1:
                prev = active_all[i]
            else:
                end = prev
                center = (start + end) // 2
                #events.append(center)
                events.append(start)

                start = active_all[i]
                prev = active_all[i]

        # last event
        end = prev
        center = (start + end) // 2
        #events.append(center)
        events.append(start)

    # extract and plot
    #for center in events:
    for onset in events:

        #t_rel = timesteps - center
        t_rel = timesteps - onset
        mask_evt = (t_rel >= -EVENT_WINDOW) & (t_rel <= EVENT_WINDOW)

        t_evt = t_rel[mask_evt]
        psi_evt = psi_corr[mask_evt]

        if len(t_evt) < 2:
            continue

        plt.plot(t_evt, psi_evt, color="gray", alpha=0.4)

        interp = np.interp(common_time, t_evt, psi_evt)
        all_event_traces_plot4.append(interp)

# global mean
if len(all_event_traces_plot4) > 0:
    mean_evt = np.mean(all_event_traces_plot4, axis=0)
    plt.plot(common_time, mean_evt, color="red", linewidth=3)

plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Time relative to activation onset")
plt.ylabel(r'$\psi$ (baseline corrected)')
plt.title("All events (corrected)")

plt.tight_layout()
plt.savefig("ensemble_event_overlay_per_event_corrected.svg")
plt.close()


# ------------------------------------------------------------
# PLOT 5 (mean sphercity + standard error of mean)
# ------------------------------------------------------------

n_events = aligned_traces.shape[0]
sem_trace = np.std(aligned_traces, axis=0) / np.sqrt(n_events)

plt.figure(figsize=(6,6))

plt.plot(common_time, mean_trace, linewidth=2)
plt.fill_between(common_time,
                 mean_trace - sem_trace,
                 mean_trace + sem_trace,
                 alpha=0.3)

plt.axvline(0, linestyle="--")

plt.xlabel("Time relative to activation")
plt.ylabel(r'$\psi$ (baseline corrected)')

plt.title("Ensemble-averaged sphericity (SEM)")

plt.tight_layout()
plt.savefig("ensemble_mean_stderror_corrected.svg")
plt.close()


# ------------------------------------------------------------
# FINAL SUMMARY PRINT
# ------------------------------------------------------------

n_analyzed_runs = len(set(all_event_runs))
total_events = len(all_event_traces)

print(f"Number of analyzed runs: {n_analyzed_runs}")
print(f"Total number of events: {total_events}")

print("Ensemble analysis completed.")
