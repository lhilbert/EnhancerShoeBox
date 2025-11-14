# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import pandas as pd
from shapely.geometry import Point, Polygon
import warnings
warnings.filterwarnings('ignore')

bins=20
contact_dist=250
conditions=["box11"]

promoters=[1,2,3]
activations=[1,5,10,15,20,25,30,50,75,100]
thresholds=[10,20,30,40,50,60,70,75,80,90,100]

contact_percent = np.zeros((len(activations), len(promoters), len(thresholds)))
active_percent = np.zeros((len(activations), len(promoters), len(thresholds)))
active_percent_stats = np.zeros((len(activations), len(promoters), len(thresholds)))
s5p_levels = np.zeros((len(activations), len(promoters), len(thresholds)))
activation_distance = np.zeros((len(activations), len(promoters), len(thresholds)))
all_s5p = []
all_active_percent = []
all_contact_percent = []
all_regions = []
all_thresholds=[]
all_activations=[]
all_activation_distances=[]
all_promoters=[]
i_condition=0

# Add S5Pmodif at the end
df_summary = pd.DataFrame(
    columns=['Promoter', 'Threshold', 'Activation',
             'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
             'S5Pmodif']
)

for condition in conditions:
    i_condition = i_condition + 1
    for j in range(len(promoters)):
        promoter = promoters[j]
        print('Promoter: '+str(promoter))
        print('-----------')
        for i in range(len(activations)):
            activation = activations[i]
            print('\nActivation: '+str(activation/1000))
            for k in range(len(thresholds)):
                threshold = thresholds[k]
                rootFolder = 'box11/Control_Promoter'+str(promoter)+'_Threshold'+str(threshold)+'_Act'+str(activation)

                if not os.path.exists(rootFolder+'/dist_active.txt'):
                    continue

                # read dist_active file (kept as in original)
                data=[]
                myFile = open(rootFolder+'/dist_active.txt', 'r')
                for line in myFile:
                    data.append(float(line.strip()))
                all_dist_active = np.array(data)

                # collect all run folders
                run_folders = os.listdir(rootFolder)
                run_folders = [x for x in run_folders if 'run' in x]

                i_r = 0
                s5p_group = []                 # collects timepoint S5P across runs (lists)
                active_percent_group = []      # collects per-run S2PInt scalars
                contact_percent_group = []     # collects per-run contact percent scalars
                activation_distance_group = [] # collects activation distances across runs (lists)
                s5pmodif_group = []            # collects per-run S5Pmodif scalars

                for run in run_folders:
                    i_r = i_r + 1
                    data=[]
                    myFile = open(rootFolder+'/'+run+'/geneTrack.txt', 'r')
                    for line in myFile:
                        data.append([float(x) for x in line.strip().split(',')])

                    d_rp = [x[4] for x in data]
                    # d_rg = [x[5] for x in data]  # unused in original
                    gene_state = [x[8] for x in data]
                    s5p_promoter = [x[6] for x in data]

                    I = [x for x in range(1, len(gene_state))
                         if int(gene_state[x])==2 and int(gene_state[x-1])!=2]
                    J = [x for x in range(1, len(gene_state))
                         if int(gene_state[x])==2]

                    # Accumulate values until 20 runs are collected
                    # exactly as original: extend timepoint lists and append per-run scalars
                    s5p_group.extend(s5p_promoter)
                    # per-run S2PInt (scalar)
                    per_run_S2PInt = 100 * len(J) / (len(gene_state) - 150)
                    active_percent_group.append(per_run_S2PInt)
                    contact_percent_group.append(100 * sum(np.array(d_rp) < contact_dist) / len(d_rp))
                    activation_distance_group.extend([d_rp[x] for x in I])

                    # NEW: compute per-run S5Pmodif (no averaging across runs yet)
                    # per-run S5PInt is mean over this run's s5p_promoter timepoints
                    if len(s5p_promoter) > 0:
                        per_run_S5PInt = np.mean(s5p_promoter)
                    else:
                        per_run_S5PInt = np.nan  # preserve behavior if empty
                    per_run_S5Pmodif = per_run_S5PInt + 0.01 * per_run_S2PInt
                    s5pmodif_group.append(per_run_S5Pmodif)

                    # On every 20th run: compute averages and write one row
                    if i_r % 20 == 0:

                        S5PInt_val = np.nanmean(s5p_group) if len(s5p_group)>0 else np.nan
                        S2PInt_val = np.nanmean(active_percent_group) if len(active_percent_group)>0 else np.nan
                        Contact_val = np.nanmean(contact_percent_group) if len(contact_percent_group)>0 else np.nan
                        DistAct_val = np.nanmean(activation_distance_group) if len(activation_distance_group)>0 else np.nan

                        # AVERAGED S5Pmodif across the 20 runs (since we collected per-run S5Pmodif)
                        S5Pmodif_val = np.nanmean(s5pmodif_group) if len(s5pmodif_group)>0 else np.nan

                        point_row = pd.DataFrame(
                            data=np.array([[promoter, threshold, activation,
                                            S5PInt_val, S2PInt_val, Contact_val,
                                            DistAct_val, S5Pmodif_val]], dtype=object),
                            columns=['Promoter', 'Threshold', 'Activation',
                                     'S5PInt', 'S2PInt', 'Contact',
                                     'DistActivation', 'S5Pmodif']
                        )

                        df_summary = pd.concat([df_summary, point_row], ignore_index=True)

                        # Reset groups for the next block of 20 runs
                        s5p_group = []
                        active_percent_group = []
                        contact_percent_group = []
                        activation_distance_group = []
                        s5pmodif_group = []

# Save output file with modified name
df_summary.to_csv('summary_contact_grouped_Thresholds10-100_20xaveraged_modifSer5P.txt', index=False)

