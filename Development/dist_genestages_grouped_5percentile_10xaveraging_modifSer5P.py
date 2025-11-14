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

bins = 20
contact_dist = 250
conditions = ["box11"]

promoters = [1, 2, 3]
activations = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100]
thresholds = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]

# Create dataframe including the new column S5Pmodif
df_summary = pd.DataFrame(columns=[
    'Promoter', 'Threshold', 'Activation',
    'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
    '5percentRG', '5percentRP', 'S5Pmodif'
])

for condition in conditions:
    for promoter in promoters:
        print("Promoter:", promoter)
        print("-----------")

        for activation in activations:
            print("\nActivation:", activation / 1000)

            for threshold in thresholds:

                rootFolder = f'box11/Control_Promoter{promoter}_Threshold{threshold}_Act{activation}'
                if not os.path.exists(rootFolder + '/dist_active.txt'):
                    continue

                # List only run folders
                run_folders = [x for x in os.listdir(rootFolder) if 'run' in x]

                # Groups for 20× averaging
                i_r = 0
                s5p_group = []
                active_percent_group = []
                contact_percent_group = []
                activation_distance_group = []
                RG_group = []
                RP_group = []

                for run in run_folders:
                    i_r += 1

                    # Load geneTrack
                    data = []
                    with open(rootFolder + '/' + run + '/geneTrack.txt', 'r') as myFile:
                        for line in myFile:
                            data.append([float(x) for x in line.strip().split(',')])

                    d_rp = [x[4] for x in data]
                    d_rg = [x[5] for x in data]
                    gene_state = [x[8] for x in data]
                    s5p_promoter = [x[6] for x in data]

                    # Active frames
                    I = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2 and int(gene_state[x - 1]) != 2]
                    J = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2]

                    # Fill averaging buffers
                    s5p_group.extend(s5p_promoter)
                    active_percent_group.append(100 * len(J) / (len(gene_state) - 150))
                    contact_percent_group.append(100 * sum(np.array(d_rp) < contact_dist) / len(d_rp))
                    activation_distance_group.extend([d_rp[x] for x in I])
                    RG_group.append(np.percentile(d_rg, 5))
                    RP_group.append(np.percentile(d_rp, 5))

                    # Every 20 runs → compute means
                    if i_r % 20 == 0:
                        S5PInt = np.mean(s5p_group)
                        S2PInt = np.mean(active_percent_group)
                        Contact = np.mean(contact_percent_group)
                        DistAct = np.mean(activation_distance_group)
                        RG5 = np.mean(RG_group)
                        RP5 = np.mean(RP_group)

                        # NEW COLUMN:
                        S5Pmodif = S5PInt + 0.01 * S2PInt

                        # Add row
                        df_summary.loc[len(df_summary)] = [
                            promoter,
                            threshold,
                            activation,
                            S5PInt,
                            S2PInt,
                            Contact,
                            DistAct,
                            RG5,
                            RP5,
                            S5Pmodif
                        ]

                        # Reset groups
                        s5p_group = []
                        active_percent_group = []
                        contact_percent_group = []
                        activation_distance_group = []
                        RG_group = []
                        RP_group = []

# Save output with new suffix
df_summary.to_csv('summary_contact_grouped_Thresholds10-100_5percentile_20xaveraged_modifSer5P.txt', index=False)

