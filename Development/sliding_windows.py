#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CHECK INPUT ARGUMENT
# ------------------------------------------------------------

if len(sys.argv) != 2:
    print("Syntax error. Syntax: python3 sliding_windows.py /path/to/run_directory")
    sys.exit(1)

run_path = sys.argv[1]

if not os.path.isdir(run_path):
    print(f"Error: directory {run_path} does not exist")
    sys.exit(1)


# ------------------------------------------------------------
# DETERMINE RUN NUMBER
# ------------------------------------------------------------

run_name = os.path.basename(os.path.normpath(run_path))

if not run_name.startswith("run"):
    print("Error: directory name must be run[nr]")
    sys.exit(1)

run_nr = run_name.replace("run","")


# ------------------------------------------------------------
# DEFINE FILE PATHS
# ------------------------------------------------------------

sphericity_file = os.path.join(run_path,"sphericity.txt")

active_file   = os.path.join(run_path,f"dist_active_run{run_nr}.txt")
induced_file  = os.path.join(run_path,f"dist_induced_run{run_nr}.txt")
approach_file = os.path.join(run_path,f"dist_approaching_run{run_nr}.txt")
recede_file   = os.path.join(run_path,f"dist_receding_run{run_nr}.txt")


# ------------------------------------------------------------
# CHECK FILES
# ------------------------------------------------------------

required = [sphericity_file,active_file,induced_file,approach_file,recede_file]

for f in required:
    if not os.path.exists(f):
        print(f"Error: file {os.path.basename(f)} does not exist")
        sys.exit(1)


# ------------------------------------------------------------
# LOAD SPHERICITY DATA
# ------------------------------------------------------------

sph_data = np.loadtxt(sphericity_file,delimiter=",",skiprows=1)

timesteps = sph_data[:,0].astype(int)
psi = sph_data[:,1]

# remove equilibration frames
mask = timesteps >= 151
timesteps = timesteps[mask]
psi = psi[mask]


# ------------------------------------------------------------
# LOAD GENE STATE FILES
# ------------------------------------------------------------

active_frames   = set(np.loadtxt(active_file)[:,0].astype(int))
induced_frames  = set(np.loadtxt(induced_file)[:,0].astype(int))
approach_frames = set(np.loadtxt(approach_file)[:,0].astype(int))
recede_frames   = set(np.loadtxt(recede_file)[:,0].astype(int))


# ------------------------------------------------------------
# ASSIGN COLORS
# ------------------------------------------------------------

colors = []

for t in timesteps:

    if t in active_frames:
        colors.append("gray")

    elif t in approach_frames:
        colors.append("darkblue")

    elif t in recede_frames:
        colors.append("lightskyblue")

    elif t in induced_frames:
        colors.append("red")

    else:
        colors.append("black")

# ------------------------------------------------------------
# LEGEND ELEMENTS FOR GENE STATES
# ------------------------------------------------------------

legend_elements = [
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='red',label='induced',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',label='active',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='darkblue',label='approaching',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='lightskyblue',label='receding',markersize=8)
]

# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame({
    "timestep": timesteps,
    "psi": psi
})

window = 50


# ------------------------------------------------------------
# 1) SLIDING WINDOW MEAN
# ------------------------------------------------------------
#reveals long-term trend of condensate shape

df["psi_mean"] = df["psi"].rolling(window=window,center=True).mean()         #center=True ensures the smoothing is centered on the time window instead of shifted

plt.figure(figsize=(6,6))

plt.scatter(timesteps,psi,c=colors,s=6, alpha=0.6)

plt.plot(df["timestep"],df["psi_mean"],
         color="black",linewidth=2,label="sliding mean")

plt.xlabel("Time step")
plt.ylabel(r'$\psi$')

plt.title("Sliding window mean")

plt.legend(handles=legend_elements, loc="lower right")
#plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(run_path,f"sphericity_slidingwindow_run{run_nr}.svg"))

plt.close()


# ------------------------------------------------------------
# 2) ROLLING MEDIAN
# ------------------------------------------------------------
#robust to noisy spikes and to outliers

df["psi_median"] = df["psi"].rolling(window=window,center=True).median()

plt.figure(figsize=(6,6))

plt.scatter(timesteps,psi,c=colors,s=6, alpha=0.6)

plt.plot(df["timestep"],df["psi_median"],
         color="black",linewidth=2,label="rolling median")

plt.xlabel("Time step")
plt.ylabel(r'$\psi$')

plt.title("Rolling median")

#plt.legend()
plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()

plt.savefig(os.path.join(run_path,f"sphericity_rollingmedian_run{run_nr}.svg"))

plt.close()


# ------------------------------------------------------------
# 3) ROLLING CONFIDENCE BAND
# ------------------------------------------------------------
#shows variability of sphericity during the simulation; information about the stability of condensate shape

df["psi_mean"] = df["psi"].rolling(window=window,center=True).mean()
df["psi_std"]  = df["psi"].rolling(window=window,center=True).std()

plt.figure(figsize=(6,6))

plt.scatter(timesteps,psi,c=colors,s=6, alpha=0.6) # alpha makes scatter points slightly transparent

plt.plot(df["timestep"],df["psi_mean"],
         color="black",linewidth=2,label="mean")

plt.fill_between(
    df["timestep"],
    df["psi_mean"]-df["psi_std"],
    df["psi_mean"]+df["psi_std"],
    color="black",
    alpha=0.2,
    label="confidence band"
)

plt.xlabel("Time step")
plt.ylabel(r'$\psi$')

plt.title("Rolling confidence band")

#plt.legend()
plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()

plt.savefig(os.path.join(run_path,f"sphericity_rollingband_run{run_nr}.svg"))

plt.close()


print("Sliding window analysis completed.")
