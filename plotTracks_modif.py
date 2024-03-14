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
S5P_thr = 40
dist_thr = 200
r = 4
time = []
d_rp = []
S5P_pr = []
state = []
#with open("gene_track/set"+str(r)+"/geneTrack.txt") as f:
with open("geneTrack.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        data = [float(x) for x in line.strip().split(",")]
        time.append(data[0]/60)
        d_rp.append(data[4])
        S5P_pr.append(data[6])
        state.append(int(data[8]))
    
if r==3:
    t1 = 234000
    t2 = 253000
    t3 = 267000
    t4 = 289000
elif r==4:
    t1 = 86000
    t2 = 93000
    t3 = 110000
    t4 = 118000
    
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8*cm, 5*cm))
for i in range(2):
    ax[i].axvline(t1*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
    ax[i].axvline(t2*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
    ax[i].axvline(t3*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
    ax[i].axvline(t4*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
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
ax[1].set_xlabel('Time (min)', fontsize=6)
ax[0].set_ylabel('Reg-promoter distance [nm]', fontsize=4)
ax[1].set_ylabel('Nr of Ser5P around promoter', fontsize=4)
ax[0].tick_params(axis='both', which='major', labelsize=6)
ax[1].tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax[0].spines[axis].set_linewidth(0.5)
    ax[1].spines[axis].set_linewidth(0.5)
ax[0].tick_params(length=2, width=0.5)
ax[1].tick_params(length=2, width=0.5)
#fig.savefig("gene_track/set"+str(r)+"/gene_track.pdf", bbox_inches="tight")
fig.savefig("gene_track.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(2*cm, 1*cm))
I = [x for x in range(len(time)) if time[x]>=t1/2000 and time[x]<=(t4/2000+2)]
ax.axvline(t1*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
ax.axvline(t2*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
ax.axvline(t3*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
ax.axvline(t4*0.03/60, linestyle='-', color="#d95f02", linewidth=0.25)
ax.plot([time[x] for x in I], [S5P_pr[x] for x in I], color="k", linewidth=0.25)
ax.axhline(S5P_thr, linestyle='-.', color=(0.5,0.5,0.5), linewidth=0.5)
prev_state = 0
time_array = []
d_rp_array = []
S5P_pr_array = []
for i in range(len(d_rp)):
    if i not in I:
        continue
    if state[i]==2:
        time_array.append(time[i])
        d_rp_array.append(d_rp[i])
        S5P_pr_array.append(S5P_pr[i])
    if state[i]!=2 and prev_state==2:
        ax.plot(time_array, S5P_pr_array, color=(0.5,0.5,0.5), linewidth=0.5)
        time_array = []
        d_rp_array = []
        S5P_pr_array = []
    prev_state = state[i]
ax.plot(time_array, S5P_pr_array, color=(0.5,0.5,0.5), linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.25)
ax.tick_params(length=2, width=0.25)
ax.set_xticks([])
ax.set_yticks([])
#fig.savefig("gene_track/set"+str(r)+"/gene_track_inset.pdf", bbox_inches="tight")
fig.savefig("gene_track_inset.pdf", bbox_inches="tight")

#print the times t1, t2, t3 and t4 corresponding to the bottom graph in gene_track.pdf file
print(t1*0.03/60)
print(t2*0.03/60)
print(t3*0.03/60)
print(t4*0.03/60)
