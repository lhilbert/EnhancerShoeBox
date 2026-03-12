#!/usr/bin/env python3

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CHECK INPUT ARGUMENT
# ------------------------------------------------------------

if len(sys.argv) != 2:
    print("Syntax error. Syntax: python3 sphericity.py /path/to/file/directory")
    sys.exit(1)

run_path = sys.argv[1]

if not os.path.isdir(run_path):
    print(f"Error: directory {run_path} does not exist")
    sys.exit(1)


# ------------------------------------------------------------
# DETERMINE RUN NUMBER FROM DIRECTORY NAME
# ------------------------------------------------------------

run_name = os.path.basename(os.path.normpath(run_path))

if not run_name.startswith("run"):
    print("Error: directory name must be run[nr]")
    sys.exit(1)

run_nr = run_name.replace("run", "")


# ------------------------------------------------------------
# DEFINE FILE PATHS
# ------------------------------------------------------------

sphericity_file = os.path.join(run_path, "sphericity.txt")

active_file = os.path.join(run_path, f"dist_active_run{run_nr}.txt")
induced_file = os.path.join(run_path, f"dist_induced_run{run_nr}.txt")
approach_file = os.path.join(run_path, f"dist_approaching_run{run_nr}.txt")
recede_file = os.path.join(run_path, f"dist_receding_run{run_nr}.txt")

ddist_induced_file = os.path.join(run_path, f"ddist_induced_run{run_nr}.txt")
ddist_active_file = os.path.join(run_path, f"ddist_active_run{run_nr}.txt")


# ------------------------------------------------------------
# CHECK REQUIRED FILES
# ------------------------------------------------------------

required_files = [
    sphericity_file,
    active_file,
    induced_file,
    approach_file,
    recede_file,
]

for f in required_files:
    if not os.path.exists(f):
        print(f"Error: file {os.path.basename(f)} does not exist")
        sys.exit(1)


# ------------------------------------------------------------
# LOAD SPHERICITY DATA
# ------------------------------------------------------------

sph_data = np.loadtxt(sphericity_file, delimiter=",", skiprows=1)

timesteps = sph_data[:,0].astype(int)
psi = sph_data[:,1]

# remove first 150 frames (equilibration part)
mask = timesteps >= 151
timesteps = timesteps[mask]
psi = psi[mask]


# ------------------------------------------------------------
# LOAD STATE FILES
# ------------------------------------------------------------

active_frames = set(np.loadtxt(active_file)[:,0].astype(int))
induced_frames = set(np.loadtxt(induced_file)[:,0].astype(int))
approach_frames = set(np.loadtxt(approach_file)[:,0].astype(int))
recede_frames = set(np.loadtxt(recede_file)[:,0].astype(int))


# ------------------------------------------------------------
# COLOR ASSIGNMENT
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
# PLOT 1: SPHERICITY VS TIMESTEP
# ------------------------------------------------------------

plt.figure(figsize=(6,6))

plt.scatter(timesteps, psi, c=colors, s=6)

plt.xlabel("Time step")
plt.ylabel(r'$\psi$')

# legend
legend_elements = [
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='red',label='induced',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',label='active',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='darkblue',label='approaching',markersize=8),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='lightskyblue',label='receding',markersize=8)
]

plt.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()

plt.savefig(os.path.join(run_path, f"sphericity_run{run_nr}.svg"))

plt.close()


# ------------------------------------------------------------
# FUNCTION TO CREATE DDIST PLOTS
# ------------------------------------------------------------

def plot_ddist(file_path, color, output_name):

    if not os.path.exists(file_path):
        print(f"Error: file {os.path.basename(file_path)} does not exist")
        return

    data = np.loadtxt(file_path)

    steps = data[:,0].astype(int)
    distances = data[:,1]

    psi_values = []

    for s in steps:

        idx = np.where(timesteps == s)[0]

        if len(idx) > 0:
            psi_values.append(psi[idx[0]])

    psi_values = np.array(psi_values)

    plt.figure(figsize=(6,6))

    plt.scatter(psi_values, distances[:len(psi_values)], c=color, s=6)

    plt.xlabel(r'$\psi$')
    plt.ylabel(r'$\Delta d_{SE-G}$ [nm]')

    plt.tight_layout()

    plt.savefig(os.path.join(run_path, output_name))

    plt.close()


# ------------------------------------------------------------
# PLOT 2: INDUCED
# ------------------------------------------------------------

plot_ddist(
    ddist_induced_file,
    "red",
    f"sphericity_ddist_induced_run{run_nr}.svg"
)


# ------------------------------------------------------------
# PLOT 3: ACTIVE
# ------------------------------------------------------------

plot_ddist(
    ddist_active_file,
    "gray",
    f"sphericity_ddist_active_run{run_nr}.svg"
)

# ------------------------------------------------------------
# PLOT 4: COMBINED ACTIVE + INDUCED
# ------------------------------------------------------------

if not os.path.exists(ddist_induced_file):
    print(f"Error: file {os.path.basename(ddist_induced_file)} does not exist")
else:
    data_induced = np.loadtxt(ddist_induced_file)
    steps_induced = data_induced[:,0].astype(int)
    dist_induced = data_induced[:,1]

    psi_induced = []

    for s in steps_induced:
        idx = np.where(timesteps == s)[0]
        if len(idx) > 0:
            psi_induced.append(psi[idx[0]])

    psi_induced = np.array(psi_induced)


if not os.path.exists(ddist_active_file):
    print(f"Error: file {os.path.basename(ddist_active_file)} does not exist")
else:
    data_active = np.loadtxt(ddist_active_file)
    steps_active = data_active[:,0].astype(int)
    dist_active = data_active[:,1]

    psi_active = []

    for s in steps_active:
        idx = np.where(timesteps == s)[0]
        if len(idx) > 0:
            psi_active.append(psi[idx[0]])

    psi_active = np.array(psi_active)


# ---- create combined scatter plot ----

plt.figure(figsize=(6,6))

# induced first (red)
plt.scatter(psi_induced, dist_induced[:len(psi_induced)], c="red", s=6, label="induced")

# active second (gray) so it appears on top
plt.scatter(psi_active, dist_active[:len(psi_active)], c="gray", s=6, label="active")

plt.xlabel(r'$\psi$')
plt.ylabel(r'$\Delta d_{SE-G}$ [nm]')

plt.legend(loc="upper right")

plt.tight_layout()

plt.savefig(os.path.join(run_path, f"sphericity_ddist_run{run_nr}.svg"))

plt.close()

print("Plots generated successfully.")
