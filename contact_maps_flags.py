#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import sys
import pandas as pd
from shapely.geometry import Point, Polygon
import statistics as st
import scipy
import scipy.interpolate
from scipy.ndimage import gaussian_filter
#from matplotlib.patches import Polygon
# from geomdl import fitting
# from geomdl import construct
# from geomdl.visualization import VisMPL as vis

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=18)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
# plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"
cm = 1/2.54
#get_ipython().run_line_magic('matplotlib', 'notebook')
try:
    # Check if running in a Jupyter environment
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'notebook')
except:
    # If not in a Jupyter environment, use a non-interactive backend for Matplotlib
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['fliers'], color=color)
    # plt.setp(bp['medians'], color=color)
    
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


# In[3]:


# Main parameter space
promoters = [1,2,3]
activations=[1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,75,100]
act_times = [600/x for x in activations]
thresholds=[70,75,80]
# thresholds=[30,35,40]
activations=[60*x/600 for x in activations]


# In[4]:
# In original script, P7 was declared twice; it is uncertain which declaration is correct

add_S2P_noise = 1 

P1 = (3,0)
P2 = (25,0) # 30
P3 = (110, y3(P1[0], P1[1], P2[0], P2[1], 110))
# P4 = (2,1.5)
P4 = (2.9,1.5)
P5 = (20,2)
P6 = (85, y3(P4[0], P4[1], P5[0], P5[1], 85))
#P7 = (0, y3(P1[0], P1[1], P4[0], P4[1], 0))
P7 = (2.8, y3(P1[0], P1[1], P4[0], P4[1], 2.8))
P8 = (12, y3(P2[0], P2[1], P5[0], P5[1], 12)) # 5
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

if len(sys.argv) == 2 and (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
        print("Usage: python3 contact_maps_flags.py -f /path/to/file/summary_contact_grouped.txt")
        sys.exit(0)

if len(sys.argv) != 3 or sys.argv[1] != '-f':
    print("Usage: python3 contact_maps_flags.py -f /path/to/file/summary_contact_grouped.txt")
    sys.exit(1)

file_path = sys.argv[2]

if not os.path.exists(file_path):
    print(f"ERROR: Input file 'summary_contact_grouped.txt' does not exist")
    sys.exit(1)

try:
    df_summary = pd.read_csv(file_path)
        
except Exception as e:
    print(f"ERROR: An error occurred while reading the file: {e}")
    sys.exit(1)

# Check if the input file exists
#if not os.path.exists('summary_contact_grouped.txt'):
#    print(f"ERROR: Input file 'summary_contact_grouped.txt' does not exist")
#    sys.exit(1)

#df_summary = pd.read_csv('summary_contact_grouped.txt')

# df_summary = pd.read_csv('summary_contact_all.txt')

# Check if the DataFrame is empty or contains only non-numeric values
if df_summary.empty or not df_summary.applymap(np.isreal).all().all():
    print("ERROR: Input file is empty or does not contain numerical values")
    sys.exit(1)

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


# In[5]:

# Check if the directory 'contact_maps' exists; if not, create it
if not os.path.exists('contact_maps'):
    os.makedirs('contact_maps')

fig, ax = plt.subplots(figsize=(7*cm, 5*cm))
df_filtered = df_summary.loc[df_summary["Region"]>=1,:]
minContact = np.min(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])
maxContact = np.max(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])

if not df_filtered.empty:
    # Calculate percentile only if df_filtered is not empty
    maxContact = np.percentile(df_filtered.loc[df_filtered["Include"] >= 1, "Contact"].tolist(), 99)
else:
    # Handle case when df_filtered is empty (e.g., set maxContact to a default value)
    maxContact = None  # or any other default value you want to use

#maxContact = np.percentile(df_filtered.loc[df_filtered["Include"]>=1, "Contact"].tolist(), 99)
# if filt_name!="all":
plt.scatter(df_summary.loc[df_summary["Include"]>=0, "S5PInt"], df_summary.loc[df_summary["Include"]>=0, "S2PInt"], c=df_summary.loc[df_summary["Include"]>=0, "Contact"], marker='o', s=3, edgecolors="none", alpha=1)
cbar = plt.colorbar()
# cbar.ax.set_ylabel('Contact %', rotation=90, fontsize=8)
cbar.ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('S5P levels', fontsize=8)
# ax.set_ylabel('S2P levels', fontsize=8)
# ax.set_title('Contact map of genes', fontsize=8)
ax.set_xlim(left=-10)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
    cbar.ax.spines[axis].set_linewidth(0.5)
fig.savefig('contact_maps/contact_all.pdf', bbox_inches='tight')


# In[6]:


fig, ax = plt.subplots(figsize=(7*cm, 5*cm))
df_filtered = df_summary.loc[df_summary["Region"]>=1,:]
minContact = np.min(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])
maxContact = np.max(df_filtered.loc[df_filtered["Include"]>=1, "Contact"])

if not df_filtered.empty:
    # Calculate percentile only if df_filtered is not empty
    maxContact = np.percentile(df_filtered.loc[df_filtered["Include"] >= 1, "Contact"].tolist(), 99)
else:
    # Handle case when df_filtered is empty (e.g., set maxContact to a default value)
    maxContact = None  # or any other default value you want to use

#maxContact = np.percentile(df_filtered.loc[df_filtered["Include"]>=1, "Contact"].tolist(), 99)
# if filt_name!="all":
#     plt.scatter(df_summary.loc[df_summary["Include"]>=0, "S5PInt"], df_summary.loc[df_summary["Include"]>=0, "S2PInt"], color=[0.75,0.75,0.75], marker='o', s=3, edgecolors="none", alpha=1)
plt.scatter(df_summary.loc[df_summary["Include"]==1, "S5PInt"], df_summary.loc[df_summary["Include"]==1, "S2PInt"],  color=[0.75,0.75,0.75], marker='o', s=1, edgecolors="none", alpha=1)
plt.scatter(df_filtered.loc[df_filtered["Include"]==1, "S5PInt"], df_filtered.loc[df_filtered["Include"]==1, "S2PInt"], c=df_filtered.loc[df_filtered["Include"]==1, "Contact"], marker='o', s=1, cmap='viridis',  vmin=minContact, vmax=maxContact, edgecolors="none", alpha=1)
xs, ys = nv_in.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = nv_ac.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = v_in.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = v_ac.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
cbar = plt.colorbar()
# cbar.ax.set_ylabel('Contact %', rotation=90, fontsize=8)
cbar.ax.tick_params(axis='both', which='major', labelsize=6)
# ax.set_xlabel('S5P levels', fontsize=8)
# ax.set_ylabel('S2P levels', fontsize=8)
# ax.set_title('Contact map of genes', fontsize=8)
ax.set_xlim(left=-10)
ax.tick_params(axis='both', which='major', labelsize=6)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
    cbar.ax.spines[axis].set_linewidth(0.5)
fig.savefig('contact_maps/contactRegionFocused.pdf', bbox_inches='tight')


# In[7]:


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
        # sns.boxplot(x="Region", y=var_order[k], data=df, ax=ax[i,j], linewidth=0.75, flierprops=flierprops, **PROPS)
        for r in range(len(uniq_regions)):
            df_filtered = df[df["Region"]==uniq_regions[r]]
            ax[i,j].plot(r, np.mean(df_filtered[var_order[k]]),'ko')
            # ax[i,j].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='o', markersize=3, color='none', mfc="none", ecolor='lightgray', elinewidth=2, capsize=0)
            # ax[i,j].errorbar(r, st.mode(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='o', markersize=3, color='black', mfc="none", ecolor='lightgray', elinewidth=2, capsize=0)
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
fig.savefig('contact_maps/byRegion.pdf', bbox_inches='tight')


# In[8]:


PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='k')

var_order = ["Promoter", "Activation", "Contact"]
uniq_regions = [1,2,3,4]
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(10*cm, 3*cm))
k=0
df = df_summary[df_summary["Include"]==1]

for i in range(3):
    # sns.boxplot(x="Region", y=var_order[k], data=df, ax=ax[i,j], linewidth=0.75, flierprops=flierprops, **PROPS)
    for r in range(len(uniq_regions)):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        
        # Calculate standard error only if the length of data is greater than zero
        data_length = len(df_filtered[var_order[k]])
        if data_length > 0:
            standard_error = np.std(df_filtered[var_order[k]]) / math.sqrt(data_length)
        else:
            standard_error = None  # Handle case when data length is zero
        
        # Plot the data using errorbar
        ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=standard_error, fmt='_', markersize=2, color='none', mfc="none", ecolor=[0.5,0.5,0.5], elinewidth=1, capsize=0)
        ax[i].plot([r-0.3,r+0.3], [np.mean(df_filtered[var_order[k]])]*2, "k-", lw=1)
    
    k=k+1
    ax[i].set(xlabel=None)
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].set_xticks(range(len(uniq_regions)))
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(0.5)

'''
for i in range(3):
        # sns.boxplot(x="Region", y=var_order[k], data=df, ax=ax[i,j], linewidth=0.75, flierprops=flierprops, **PROPS)
        for r in range(len(uniq_regions)):
            df_filtered = df[df["Region"]==uniq_regions[r]]
            # ax[i,j].plot(r, np.mean(df_filtered[var_order[k]]),'ko')
            ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='_', markersize=2, color='none', mfc="none", ecolor=[0.5,0.5,0.5], elinewidth=1, capsize=0)
            ax[i].plot([r-0.3,r+0.3], [np.mean(df_filtered[var_order[k]])]*2, "k-", lw=1)
            # ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='_', markersize=2, color='none', mfc="none", ecolor=[0.5,0.5,0.5], elinewidth=0.5, capsize=0.5)
            # ax[i,j].errorbar(r, st.mode(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='o', markersize=3, color='black', mfc="none", ecolor='lightgray', elinewidth=2, capsize=0)
        k=k+1
        ax[i].set(xlabel=None)
        ax[i].tick_params(axis='both', which='major', labelsize=6)
        ax[i].set_xticks(range(len(uniq_regions)))
        # ax[i].set_xticklabels(['NV-in', 'NV-ac', 'V-in', 'V-ac'], rotation = 90)
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(0.5)
'''

ax[0].set_yticks([1,2,3])
ax[0].set_ylim([0.9,3.1])
ax[1].set_yticks([0,5,10])
ax[2].set_yticks([0,10,20])
ax[2].set_ylim([-1.1,23.1])
ax[0].set_title("Promoter length",fontsize=6)
ax[1].set_title("Activation rate",fontsize=6)
# ax[1].set_yticks([0,25,50,75,100])
ax[2].set_title("Contact %",fontsize=6)
fig.tight_layout()
fig.savefig('contact_maps/byRegion_figure.pdf', bbox_inches='tight')


# In[81]:


# Parameter ditributions by regions - new figure

def compute_fractions(lst):
    if len(lst) == 0:
        return [0, 0, 0]
    return [lst.count(i) / len(lst) for i in [1, 2, 3]]

#def compute_fractions(lst):
#    return [lst.count(i)/len(lst) for i in [1, 2, 3]]
barWidth = 0.19

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='k')

var_order = ["Promoter", "Activation", "Contact"]
uniq_regions = [1,2,3,4]
fig, ax = plt.subplots(1, 3, sharex=True, figsize=(10*cm, 3*cm))
k=0
df = df_summary[df_summary["Include"]==1]
for i in range(3):
    # sns.boxplot(x="Region", y=var_order[k], data=df, ax=ax[i,j], linewidth=0.75, flierprops=flierprops, **PROPS)
    for r in range(len(uniq_regions)):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        # ax[i,j].plot(r, np.mean(df_filtered[var_order[k]]),'ko')
        if i<1:
            # ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='_', markersize=2, color='none', mfc="none", ecolor=[0.5,0.5,0.5], elinewidth=1, capsize=0)
            # ax[i].plot([r-0.3,r+0.3], [np.mean(df_filtered[var_order[k]])]*2, "k-", lw=1)

            fractions = compute_fractions(df_filtered[var_order[k]].to_list())
            rx = [r-0.25, r, r+0.25]
            # ax[i].bar(rx, fractions, width=barWidth, color="none", edgecolor="k", linewidth=0.5)
            ax[i].bar(rx, fractions, width=barWidth, color=["#eeeeee", "#bbbbbb", "#888888"], edgecolor=["k"]*3, linewidth=0.25)
        else:
            ax[i].boxplot(df_filtered[var_order[k]].to_list(), positions=[r], showfliers=False, widths=0.5, showmeans=True, boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5), medianprops=dict(linewidth=0), meanline=True, meanprops=dict(linestyle="-", linewidth=1.5, color='firebrick'))
        # ax[i].errorbar(r, np.mean(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='_', markersize=2, color='none', mfc="none", ecolor=[0.5,0.5,0.5], elinewidth=0.5, capsize=0.5)
        # ax[i,j].errorbar(r, st.mode(df_filtered[var_order[k]]), yerr=np.std(df_filtered[var_order[k]])/math.sqrt(len(df_filtered[var_order[k]])), fmt='o', markersize=3, color='black', mfc="none", ecolor='lightgray', elinewidth=2, capsize=0)
    k=k+1
    ax[i].set(xlabel=None)
    ax[i].tick_params(axis='both', which='major', labelsize=6)
    ax[i].set_xticks(range(len(uniq_regions)))
    # ax[i].set_xticklabels(['NV-in', 'NV-ac', 'V-in', 'V-ac'], rotation = 90)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(0.5)

# ax[0].set_yticks([1,2,3])
ax[0].set_ylim([0,0.53])
ax[0].set_yticks([0,0.25, 0.5])
ax[1].set_yticks([0,5,10])
ax[2].set_yticks([0,10,20,30,40,50])
ax[2].set_ylim([-1.1,53.1])
ax[0].set_title("Promoter length",fontsize=6)
ax[1].set_title("Activation rate",fontsize=6)
# ax[1].set_yticks([0,25,50,75,100])
ax[2].set_title("Contact %",fontsize=6)
fig.tight_layout()
fig.savefig('contact_maps/byRegion_new.pdf', bbox_inches='tight')


# In[61]:


np.arange(len(fractions))


# In[46]:


from collections import Counter
Counter(df_filtered[var_order[0]].to_list())
df_filtered


# In[19]:


region_names = ['NV-in', 'NV-ac', 'V-in', 'V-ac']
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10*cm, 7*cm))
r=0
for i in range(2):
    for j in range(2):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        A = df_filtered.groupby(["Promoter", "Activation"])
        B = A["Contact"].count().reset_index().pivot(index="Promoter", columns="Activation", values="Contact")
        B.fillna(0, inplace=True)
        B = B.reindex(promoters, axis=0, fill_value=0)
        B = B.reindex(activations, axis=1, fill_value=0)
        sns.heatmap(B, ax=ax[i,j])
        ax[i,j].set_title(region_names[r])
        r=r+1
fig.tight_layout()
fig.savefig('contact_maps/heatmap_pr.pdf', bbox_inches='tight') 


# In[20]:


region_names = ['NV-in', 'NV-ac', 'V-in', 'V-ac']
fig, ax = plt.subplots(1, 4, sharex=True, figsize=(10*cm, 2*cm))
r=0
for i in range(4):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        A = df_filtered.groupby(["Promoter", "Activation"])
        B = A["Contact"].count().reset_index().pivot(index="Promoter", columns="Activation", values="Contact")
        B.fillna(0, inplace=True)
        B = B.reindex(promoters, axis=0, fill_value=0)
        B = B.reindex(activations, axis=1, fill_value=0)
        sns.heatmap(B/B.max().max(), ax=ax[i], cbar=False)
        ax[i].set(xlabel=None, ylabel=None)
        ax[i].set_xticks([0.5,5.5,9.5,13.5])
        ax[i].set_xticklabels([])
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        ax[i].invert_yaxis()
        for axis in ['top','bottom','left','right']:
            ax[i].spines[axis].set_linewidth(0)
        r=r+1
fig.tight_layout()
fig.savefig('contact_maps/heatmap_pr_figure.pdf', bbox_inches='tight') 


# In[21]:


region_names = ['NV-in', 'NV-ac', 'V-in', 'V-ac']
fig, ax = plt.subplots(2, 2, sharex=True, figsize=(10*cm, 7*cm))
r=0
for i in range(2):
    for j in range(2):
        df_filtered = df[df["Region"]==uniq_regions[r]]
        A = df_filtered.groupby(["Threshold", "Activation"])
        B = A["Contact"].count().reset_index().pivot(index="Threshold", columns="Activation", values="Contact")
        B.fillna(0, inplace=True)
        B = B.reindex(thresholds, axis=0, fill_value=0)
        B = B.reindex(activations, axis=1, fill_value=0)
        sns.heatmap(B, ax=ax[i,j])
        ax[i,j].set_title(region_names[r])
        r=r+1
fig.tight_layout()
fig.savefig('contact_maps/heatmap_thr.pdf', bbox_inches='tight') 


# In[22]:

def nonuniform_imshow(x, y, z, aspect=6, cmap=plt.cm.viridis):
  x = pd.to_numeric(x, errors='coerce')
  y = pd.to_numeric(y, errors='coerce')
  z = pd.to_numeric(z, errors='coerce')
   
  # Create regular grid
  # xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
  xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000) # 1000,1000
  xi, yi = np.meshgrid(xi, yi)

  # tot_reg.contains(Point(x, y))
  # Interpolate missing data
  rbf = scipy.interpolate.Rbf(x, y, z, function='gaussian', epsilon=0.2)
  rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
  # rbf = scipy.interpolate.RBFInterpolator(list(zip(x, y)), z, kernel='linear')
  zi = rbf(xi, yi)
  
  zi = gaussian_filter(zi, sigma=[10,100]) # [10,10] or [10,100]
  for i in range(len(zi)):
    for j in range(len(zi[0])):
      if not tot_reg.contains(Point(xi[i,j], yi[i,j])):
        zi[i,j] = np.nan
      
  fig, ax = plt.subplots(figsize=(6*cm, 6*cm))
  # cmap=plt.cm.rainbow
  hm = ax.imshow(zi, interpolation='none', cmap=cmap, vmin=minContact, vmax=maxContact,
                 extent=[x.min(), x.max(), y.max(), y.min()]) 
  # ax.scatter(x, y)
  ax.set_aspect(aspect)
  return fig, ax, hm

# fig, ax, heatmap = nonuniform_imshow(df_filtered.loc[df_filtered["Include"]==1, "S5PInt"], df_filtered.loc[df_filtered["Include"]==1, "S2PInt"], df_filtered.loc[df_filtered["Include"]==1, "Contact"])
fig, ax, heatmap = nonuniform_imshow(df_summary.loc[df_summary["Include"]==1, "S5PInt"], df_summary.loc[df_summary["Include"]==1, "S2PInt"], df_summary.loc[df_summary["Include"]==1, "Contact"])
ax.invert_yaxis()

# plt.colorbar(heatmap)
# plt.show()
xs, ys = nv_in.exterior.xy

#ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = nv_ac.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = v_in.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
xs, ys = v_ac.exterior.xy
ax.fill(xs, ys, alpha=1, fc='none', ec='k', linewidth=0.75)
# ax.axis('off')
ax.set_xlim(-1,112)
ax.set_ylim(-0.5,12)
ax.set_xticks([0,100])
ax.set_yticks([0,12.5])
fig.savefig('contact_maps/surface.pdf', bbox_inches='tight') 



# In[23]:


df_filtered = df[df["Region"]==uniq_regions[3]]
np.median(df_filtered[var_order[0]])
st.mode(df_filtered[var_order[0]])

