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

# Define parameters
bins = 20
contact_dist = 250
conditions = ["box11"]

promoters = [1, 2, 3]
activations = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100]
thresholds = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]

# Prepare data containers
contact_percent = np.zeros((len(activations), len(promoters), len(thresholds)))
active_percent = np.zeros((len(activations), len(promoters), len(thresholds)))
active_percent_stats = np.zeros((len(activations), len(promoters), len(thresholds)))
s5p_levels = np.zeros((len(activations), len(promoters), len(thresholds)))
activation_distance = np.zeros((len(activations), len(promoters), len(thresholds)))

# Prepare empty lists for collecting all results
all_s5p = []
all_active_percent = []
all_contact_percent = []
all_regions = []
all_thresholds = []
all_activations = []
all_activation_distances = []
all_promoters = []
i_condition = 0

# Create an empty DataFrame with the new extra columns
df_summary = pd.DataFrame(columns=[
    'Promoter', 'Threshold', 'Activation', 'Runs',
    'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
    '5percentRG', '5percentRP'  # <-- new columns added
])

# Loop through all experimental conditions
for condition in conditions:
    i_condition = i_condition + 1
    for j in range(len(promoters)):
        promoter = promoters[j]
        print('Promoter: ' + str(promoter))
        print('-----------')
        for i in range(len(activations)):
            activation = activations[i]
            print('\nActivation: ' + str(activation / 1000))
            for k in range(len(thresholds)):
                threshold = thresholds[k]

                # Define folder path for current parameter combination
                rootFolder = 'box11/Control_Promoter' + str(promoter) + '_Threshold' + str(threshold) + '_Act' + str(activation)

                # Skip if the expected file does not exist
                if not os.path.exists(rootFolder + '/dist_active.txt'):
                    continue

                # Read dist_active.txt file
                data = []
                myFile = open(rootFolder + '/dist_active.txt', 'r')
                for line in myFile:
                    data.append(float(line.strip()))
                myFile.close()

                # Identify all run folders in the current directory
                run_folders = os.listdir(rootFolder)
                run_folders = [x for x in run_folders if 'run' in x]

                # Loop through each run folder
                for run in run_folders:
                    data = []
                    myFile = open(rootFolder + '/' + run + '/geneTrack.txt', 'r')
                    for line in myFile:
                        # Each line contains several numeric values separated by commas
                        data.append([float(x) for x in line.strip().split(',')])
                    myFile.close()

                    # Extract relevant columns from geneTrack.txt data
                    d_rp = [x[4] for x in data]  # distance between promoter and enhancer
                    d_rg = [x[5] for x in data]  # distance between gene and enhancer
                    # Exclude the first 150 frames from distance data
                    #d_rp = d_rp[150:]
                    #d_rg = d_rg[150:]
                    
                    gene_state = [x[8] for x in data]  # gene activity states
                    s5p_promoter = [x[6] for x in data]  # S5P intensity at promoter

                    # Identify times when gene switches to active (state 2)
                    I = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2 and int(gene_state[x - 1]) != 2]
                    J = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2]

                    # Calculate summary statistics for this run
                    mean_s5p = np.mean(s5p_promoter)
                    percent_active = 100 * len(J) / (len(gene_state) - 150)
                    percent_contact = 100 * sum(np.array(d_rp) < contact_dist) / len(d_rp)
                    mean_activation_dist = np.mean([d_rp[x] for x in I])

                    # Calculate 5th percentile distances
                    # np.percentile() returns the value below which 5% of the data falls
                    #five_percent_RG = np.percentile(d_rg, 5)
                    #five_percent_RP = np.percentile(d_rp, 5)
                    # Calculate 5th percentile distances, excluding the first 150 frames
                    five_percent_RG = np.percentile(d_rg[150:], 5)
                    five_percent_RP = np.percentile(d_rp[150:], 5)

                    #If geneTrack.txt has fewer than 150 frames for equilibration:
                    #five_percent_RG = np.percentile(d_rg[150:] if len(d_rg) > 150 else d_rg, 5)
                    #five_percent_RP = np.percentile(d_rp[150:] if len(d_rp) > 150 else d_rp, 5)


                    # Save results in memory lists
                    all_s5p.append(mean_s5p)
                    all_active_percent.append(percent_active)
                    all_contact_percent.append(percent_contact)
                    all_activation_distances.append(mean_activation_dist)

                    # Create one row of summary results for this run
                    point_row = pd.DataFrame(data=np.array([[
                        promoter,
                        threshold,
                        activation,
                        run,
                        mean_s5p,
                        percent_active,
                        percent_contact,
                        mean_activation_dist,
                        five_percent_RG,  # <-- new column
                        five_percent_RP   # <-- new column
                    ]], dtype=object), columns=[
                        'Promoter', 'Threshold', 'Activation', 'Runs',
                        'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
                        '5percentRG', '5percentRP'
                    ])

                    # Append this row to the summary dataframe
                    df_summary = pd.concat([df_summary, point_row], ignore_index=True)

# Save final dataframe to a file with a new name
df_summary.to_csv('summary_contact_all_Thresholds10-100_5percentile.txt', index=False)

# %%

