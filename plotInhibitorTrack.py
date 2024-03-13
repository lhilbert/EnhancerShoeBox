# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sys, getopt
import seaborn as sns

cm = 1/2.54
S5P_thr = 70
dist_thr = 250
NRuns = 2000
time = []
d_rp = []
S5P_pr = []
state = []
# geneTrack_JQ1_VAc_thr70_run19.txt
# geneTrack_FP_VAc_thr70_run22.txt
with open("geneTrack_FP_VAc_thr70_run22.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        data = [float(x) for x in line.strip().split(",")]
        time.append(data[0]/600)
        d_rp.append(data[4])
        S5P_pr.append(data[6])
        state.append(int(data[8]))
    
    
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8*cm, 5*cm))
ax[0].plot(time, d_rp, color="k", linewidth=0.25)
# ax[0].plot([time[x] for x in range(len(d_rp)) if state[x]==2], [d_rp[x] for x in range(len(d_rp)) if state[x]==2], 'ko', markersize=3, mfc="none", mew=0.5)
ax[1].plot(time, S5P_pr, color="k", linewidth=0.25)
# ax[1].plot([time[x] for x in range(len(d_rp)) if state[x]==2], [S5P_pr[x] for x in range(len(d_rp)) if state[x]==2], 'ko', markersize=3, mfc="none", mew=0.5)
ax[0].axhline(dist_thr, linestyle='-.', color=(0.5,0.5,0.5), linewidth=0.5)
ax[1].axhline(S5P_thr, linestyle='-.', color=(0.5,0.5,0.5), linewidth=0.5)
prev_state = 0
time_array = []
d_rp_array = []
S5P_pr_array = []
for i in range(len(d_rp)):
    if state[i]==2:
        time_array.append(time[i])
        d_rp_array.append(d_rp[i])
        S5P_pr_array.append(S5P_pr[i])
        """ if prev_state!=2:
            ax[0].plot(time[i], d_rp[i], 'o', color=(0.5,0.5,0.5), markersize=3, mfc="none", mew=0.5)
            ax[1].plot(time[i], S5P_pr[i], 'o', color=(0.5,0.5,0.5), markersize=3, mfc="none", mew=0.5) """
    if state[i]!=2 and prev_state==2:   
        ax[0].plot(time_array, d_rp_array, color=(0.5,0.5,0.5), linewidth=0.75)
        ax[1].plot(time_array, S5P_pr_array, color=(0.5,0.5,0.5), linewidth=0.75)
        time_array = []
        d_rp_array = []
        S5P_pr_array = []
    prev_state = state[i]
# FP: #ef8a62
# JQ-1: #67a9cf
ax[0].axvline(time[NRuns-1], linestyle='-', color="#ef8a62", linewidth=1)
ax[1].axvline(time[NRuns-1], linestyle='-', color="#ef8a62", linewidth=1)
ax[1].set_xlabel('Time (min)', fontsize=8)
ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[1].tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0.5)
    ax[1].spines[axis].set_linewidth(0.5)
ax[0].tick_params(length=2, width=0.5)
ax[1].tick_params(length=2, width=0.5)
fig.savefig("inhibitor_track.pdf", bbox_inches="tight")