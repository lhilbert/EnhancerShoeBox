#!/usr/bin/env python
# coding: utf-8

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import pandas as pd
from shapely.geometry import Point, Polygon
import statistics as st
import scipy
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
# --------------------------
# Define Parula colormap manually
# --------------------------
parula_cm_data = np.array([
    [0.2081, 0.1663, 0.5292],
    [0.1976, 0.2675, 0.7060],
    [0.0712, 0.3960, 0.7040],
    [0.0329, 0.5656, 0.6195],
    [0.1809, 0.7490, 0.4920],
    [0.4393, 0.8671, 0.2546],
    [0.7008, 0.9024, 0.1600],
    [0.9469, 0.8871, 0.1462],
    [0.9932, 0.9062, 0.1439],
    [0.9932, 0.9500, 0.2500],
    [0.9994, 0.9994, 0.1620]
])
x_old = np.linspace(0, 1, parula_cm_data.shape[0])
x_new = np.linspace(0, 1, 256)
parula_interp = np.zeros((256, 3))
for i in range(3):
    f = interp1d(x_old, parula_cm_data[:, i], kind="cubic", bounds_error=False, fill_value="extrapolate")
    parula_interp[:, i] = f(x_new)
parula_cmap = ListedColormap(parula_interp)

# ----------------------------------------------------------------

cm = 1/2.54

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['fliers'], color=color)

def y3(x1, y1, x2, y2, x3):
    m = (y2-y1)/(x2-x1)
    c = y2 - m*x2
    y3 = m*x3 + c
    return y3

def x5y5(x1, y1, x2, y2, x3, y3, x4, y4):
    m1 = (y2-y1)/(x2-x1)
    c1 = y2 - m1*x2
    m2 = (y4-y3)/(x4-x3)
    c2 = y4 - m2*x4
    x = (c1-c2)/(m2-m1)
    y = m1*x + c1
    return (x,y)

bins=20
box_colors=['#1b9e77', '#d95f02', '#7570b3']
contact_dist=250

# Main parameter space
promoters = [1,2,3]
activations=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,75,100]
act_times = [600/x for x in activations]
thresholds=[70,75,80]
activations=[60*x/600 for x in activations]

add_S2P_noise = 1 

P1 = (3,0)
P2 = (25,0)
P3 = (110, y3(P1[0], P1[1], P2[0], P2[1], 110))
P4 = (2.9,1.5)
P5 = (20,2)
P6 = (85, y3(P4[0], P4[1], P5[0], P5[1], 85))
P7 = (2.8, y3(P1[0], P1[1], P4[0], P4[1], 2.8))
P8 = (12, y3(P2[0], P2[1], P5[0], P5[1], 12))
P9 = x5y5(P3[0], P3[1], P6[0], P6[1], P7[0], P7[1], P8[0], P8[1])

nv_in = Polygon([P1, P2, P5, P4])
nv_ac = Polygon([P4, P5, P8, P7])
v_in = Polygon([P2, P3, P6, P5])
v_ac = Polygon([P5, P6, P9, P8])
tot_reg = Polygon([P1, P3, P9, P7])

uniq_promoters=list(set(promoters))
markers = ['o','s','D','v']
i_condition=0

all_s5p = []
all_s2p = []
all_contact_percent = []
all_regions = []
all_thresholds=[]
all_activations=[]
all_activation_distances=[]
all_promoters=[]
all_markers = []
all_include = []

df_summary = pd.read_csv('summary_contact_grouped_Thresholds10-100.txt')

for i in range(len(df_summary)):
    df_summary.loc[i, "Activation"] = 60*df_summary.loc[i, "Activation"]/600
    y = 0
    if add_S2P_noise:
        y = 2*0.1*(random.random()-0.5)
        if df_summary.loc[i, "S2PInt"]+y<0:
            y=0
    df_summary.loc[i, "S2PInt"] = df_summary.loc[i, "S2PInt"] + y
        
    if df_summary.loc[i, "Promoter"] not in promoters or df_summary.loc[i, "Activation"] not in activations or df_summary.loc[i, "Threshold"] not in thresholds:
        all_include.append(0)
        all_regions.append(-2)
    else:
        all_include.append(1)
        ourPoint = Point(df_summary.loc[i, "S5PInt"], df_summary.loc[i, "S2PInt"])
        if nv_in.contains(ourPoint):
            all_regions.append(1)
        elif nv_ac.contains(ourPoint):
            all_regions.append(2)
        elif v_in.contains(ourPoint):
            all_regions.append(3)
        elif v_ac.contains(ourPoint):
            all_regions.append(4)
        else:
            all_regions.append(-1)
            
        all_promoters.append(df_summary.loc[i, "Promoter"])
        all_markers.append(markers[uniq_promoters.index(df_summary.loc[i, "Promoter"])])
        all_s5p.append(df_summary.loc[i, "S5PInt"])
        all_s2p.append(df_summary.loc[i, "S2PInt"]+y)
        all_contact_percent.append(df_summary.loc[i, "Contact"])
        all_activations.append(df_summary.loc[i, "Activation"])
        all_thresholds.append(df_summary.loc[i, "Threshold"])
        all_activation_distances.append(df_summary.loc[i, "DistActivation"])
    
df_summary["Region"] = all_regions
df_summary["Include"] = all_include
minContact = np.min(df_summary.loc[df_summary["Include"]>0, "Contact"])
maxContact = np.max(df_summary.loc[df_summary["Include"]>0, "Contact"])

# ---------------------- FIGURE 1 ----------------------
fig, ax = plt.subplots(figsize=(7*cm, 5*cm))
df_filtered = df_summary.loc[df_summary["Region"]>=1,:]
minContact = np.min(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])
maxContact = np.percentile(df_filtered.loc[df_filtered["Include"]>=1, "Contact"].tolist(), 99)
plt.scatter(df_summary.loc[df_summary["Include"]>=0, "S5PInt"],
            df_summary.loc[df_summary["Include"]>=0, "S2PInt"],
            c=df_summary.loc[df_summary["Include"]>=0, "Contact"],
            marker='o', s=1, edgecolors="none", alpha=1, cmap=parula_cmap)
cbar = plt.colorbar()
cbar.set_label("Contact %")
cbar.ax.tick_params(axis='both', which='major', labelsize=6)

ax.set_xlabel("Mean S5P Int.")
ax.set_ylabel("Mean S2P Int.")
ax.set_xlim(left=-10)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
    cbar.ax.spines[axis].set_linewidth(0.5)
fig.savefig('contact_maps_parula/contact_all.pdf', bbox_inches='tight')

# ---------------------- FIGURE 2 ----------------------

fig, ax = plt.subplots(figsize=(7*cm, 5*cm))
df_filtered = df_summary.loc[df_summary["Region"]>=1,:]
minContact = np.min(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])
maxContact = np.percentile(df_filtered.loc[df_filtered["Include"]>=1, "Contact"].tolist(), 99)

plt.scatter(df_summary.loc[df_summary["Include"]==1, "S5PInt"],
            df_summary.loc[df_summary["Include"]==1, "S2PInt"],
            color=[0.75,0.75,0.75], marker='o', s=1, edgecolors="none", alpha=1)

plt.scatter(df_filtered.loc[df_filtered["Include"]==1, "S5PInt"],
            df_filtered.loc[df_filtered["Include"]==1, "S2PInt"],
            c=df_filtered.loc[df_filtered["Include"]==1, "Contact"],
            marker='o', s=1, cmap=parula_cmap,  vmin=minContact, vmax=maxContact, edgecolors="none", alpha=1)

for poly in [nv_in, nv_ac, v_in, v_ac]:
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)

# Add labeled colorbar
cbar = plt.colorbar()
cbar.set_label("Contact %", fontsize=8)
cbar.ax.tick_params(axis='both', which='major', labelsize=6)

# Add axis labels
ax.set_xlabel("Mean S5P Int.", fontsize=8)
ax.set_ylabel("Mean S2P Int.", fontsize=8)

ax.set_xlim(left=-10)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
    cbar.ax.spines[axis].set_linewidth(0.5)

fig.savefig('contact_maps_parula/contactRegionFocused.pdf', bbox_inches='tight')


# ---------------------- FIGURE 3 (mean plots by region) ----------------------
PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}
flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='k')
var_order = ["Promoter", "Activation", "DistActivation", "Threshold", "Contact", "S2PInt"]
uniq_regions = [1,2,3,4]
fig, ax = plt.subplots(2, 3, sharex=True, figsize=(10*cm, 7*cm))
k=0
df = df_summary[df_summary["Include"]==1]
for i in range(2):
    for j in range(3):
        for r in range(len(uniq_regions)):
            df_filtered = df[df["Region"]==uniq_regions[r]]
            ax[i,j].plot(r, np.mean(df_filtered[var_order[k]]),'ko')
        k=k+1
        if i==0:
            ax[i,j].set(xlabel=None)
        else:
            ax[i,j].set_xlabel("Region",fontsize=8)
        ax[i,j].tick_params(axis='both', which='major', labelsize=6)
        ax[i,j].set_xticks(range(len(uniq_regions)))
        ax[i,j].set_xticklabels(['NV-in', 'NV-ac', 'V-in', 'V-ac'], rotation = 90)
        for axis in ['top','bottom','left','right']:
            ax[i,j].spines[axis].set_linewidth(0.5)
ax[0,0].set_ylabel("Promoter",fontsize=8)
ax[0,1].set_ylabel("Activation",fontsize=8)
ax[0,2].set_ylabel("d(Act)",fontsize=8)
ax[1,0].set_ylabel("Threshold",fontsize=8)
ax[1,1].set_ylabel("Contact",fontsize=8)
ax[1,2].set_ylabel("S2P int.",fontsize=8)
fig.tight_layout()
fig.savefig('contact_maps_parula/byRegion.pdf', bbox_inches='tight')

# ---------------------- FIGURE 4 (summary means) ----------------------
var_order = ["Promoter", "Activation", "Contact"]
uniq_regions = [1,2,3,4]
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(10*cm, 3*cm))
k=0
df = df_summary[df_summary["Include"]==1]
for i in range(3):
    for r in range(len(uniq_regions)):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]),
                       yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])),
                       fmt='_', markersize=2, color='none', mfc="none",
                       ecolor=[0.5,0.5,0.5], elinewidth=1, capsize=0)
        ax[i].plot([r-0.3,r+0.3], [np.mean(df_filtered[var_order[k]])]*2, "k-", lw=1)
    k=k+1
    ax[i].set(xlabel=None)
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].set_xticks(range(len(uniq_regions)))
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(0.5)
ax[0].set_yticks([1,2,3])
ax[0].set_ylim([0.9,3.1])
ax[1].set_yticks([0,5,10])
ax[2].set_yticks([0,10,20])
ax[2].set_ylim([-1.1,23.1])
ax[0].set_title("Promoter length",fontsize=6)
ax[1].set_title("Activation rate",fontsize=6)
ax[2].set_title("Contact %",fontsize=6)
fig.tight_layout()
fig.savefig('contact_maps_parula/byRegion_figure.pdf', bbox_inches='tight')

# ---------------------- FIGURE 5 (surface map with interpolation) ----------------------
def nonuniform_imshow(x, y, z, aspect=6, cmap=plt.cm.viridis):
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xi, yi = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
    zi = gaussian_filter(zi, sigma=[10,100])
    for i in range(len(zi)):
        for j in range(len(zi[0])):
            if not tot_reg.contains(Point(xi[i,j], yi[i,j])):
                zi[i,j] = np.nan
    fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
    hm = ax.imshow(zi, interpolation='none', cmap=cmap,
                   vmin=minContact, vmax=maxContact,
                   extent=[x.min(), x.max(), y.max(), y.min()]) 
    ax.set_aspect(aspect)
    return fig, ax, hm

fig, ax, heatmap = nonuniform_imshow(df_summary.loc[df_summary["Include"]==1, "S5PInt"],
                                     df_summary.loc[df_summary["Include"]==1, "S2PInt"],
                                     df_summary.loc[df_summary["Include"]==1, "Contact"],
                                     cmap=parula_cmap)
ax.invert_yaxis()
for poly in [nv_in, nv_ac, v_in, v_ac]:
    xs, ys = poly.exterior.xy
    ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
ax.set_xlim(-1,112)
ax.set_ylim(-0.5,12)
ax.set_xticks([0,100])
ax.set_yticks([0,12.5])
fig.savefig('contact_maps_parula/surface.pdf', bbox_inches='tight')

# ---------------------- FIGURE 6: byRegion_new.pdf ----------------------

def compute_fractions(lst):
    """Compute fractions of elements equal to 1, 2, or 3 in a list."""
    return [lst.count(i) / len(lst) if len(lst) > 0 else 0 for i in [1, 2, 3]]

# Figure and plot setup
cm = 1/2.54  # centimeters to inches conversion
barWidth = 0.19

PROPS = {
    'boxprops': {'facecolor': 'none', 'edgecolor': 'k'},
    'medianprops': {'color': 'k'},
    'whiskerprops': {'color': 'k'},
    'capprops': {'color': 'k'}
}
flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                  linestyle='none', markeredgecolor='k')

var_order = ["Promoter", "Activation", "Contact"]
uniq_regions = [1, 2, 3, 4]

fig, ax = plt.subplots(1, 3, sharex=True, figsize=(10*cm, 3*cm))
df = df_summary[df_summary["Include"] == 1]

for i, var in enumerate(var_order):
    for r in range(len(uniq_regions)):
        df_filtered = df[df["Region"] == uniq_regions[r]]

        if i < 1:
            # Promoter variable — use bar plot showing fractions
            fractions = compute_fractions(df_filtered[var].to_list())
            rx = [r - 0.25, r, r + 0.25]
            ax[i].bar(rx, fractions, width=barWidth,
                      color=["#eeeeee", "#bbbbbb", "#888888"],
                      edgecolor=["k"] * 3, linewidth=0.25)
        else:
            # Boxplot for Activation and Contact
            ax[i].boxplot(df_filtered[var].to_list(), positions=[r],
                          showfliers=False, widths=0.5, showmeans=True,
                          boxprops=dict(linewidth=0.5),
                          whiskerprops=dict(linewidth=0.5),
                          capprops=dict(linewidth=0.5),
                          medianprops=dict(linewidth=0),
                          meanline=True,
                          meanprops=dict(linestyle="-", linewidth=1.5,
                                         color='firebrick'))

    # Axis formatting
    ax[i].set(xlabel=None)
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].set_xticks(range(len(uniq_regions)))
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[i].spines[axis].set_linewidth(0.5)

# Y-axis and title setup
ax[0].set_ylim([0, 0.53])
ax[0].set_yticks([0, 0.25, 0.5])
ax[1].set_yticks([0, 5, 10])
ax[2].set_yticks([0, 10, 20, 30, 40, 50])
ax[2].set_ylim([-1.1, 53.1])

ax[0].set_title("Promoter length", fontsize=6)
ax[1].set_title("Activation rate", fontsize=6)
ax[2].set_title("Contact %", fontsize=6)

fig.tight_layout()
fig.savefig('contact_maps_parula/byRegion_new.pdf', bbox_inches='tight')
plt.close(fig)


