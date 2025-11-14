# Generate summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt
# Averaging every 10 consecutive runs 

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

# --------------------------
# Define parameters
# --------------------------
bins = 20
contact_dist = 250
conditions = ["box11"]

promoters = [1, 2, 3]
activations = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100]
thresholds = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]

# --------------------------
# Prepare empty DataFrame for summary results
# --------------------------
df_summary = pd.DataFrame(columns=[
    'Promoter', 'Threshold', 'Activation', 'Runs',
    'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
    '5percentRG', '5percentRP'
])

# --------------------------
# Main processing loop
# --------------------------
for condition in conditions:
    for promoter in promoters:
        print(f"Promoter: {promoter}")
        print("-----------")
        for activation in activations:
            print(f"\nActivation: {activation/1000}")
            for threshold in thresholds:
                rootFolder = f'box11/Control_Promoter{promoter}_Threshold{threshold}_Act{activation}'
                if not os.path.exists(rootFolder + '/dist_active.txt'):
                    continue

                run_folders = os.listdir(rootFolder)
                run_folders = [x for x in run_folders if 'run' in x]
                if len(run_folders) == 0:
                    continue

                # --- Initialize group accumulators ---
                s5p_group = []
                active_percent_group = []
                contact_percent_group = []
                activation_distance_group = []
                five_percent_RG_group = []
                five_percent_RP_group = []

                run_counter = 0

                for run in run_folders:
                    run_counter += 1
                    data = []
                    with open(f"{rootFolder}/{run}/geneTrack.txt", 'r') as myFile:
                        for line in myFile:
                            data.append([float(x) for x in line.strip().split(',')])

                    d_rp = [x[4] for x in data]  # distance between promoter and enhancer
                    d_rg = [x[5] for x in data]  # distance between gene and enhancer
                    gene_state = [x[8] for x in data]
                    s5p_promoter = [x[6] for x in data]

                    # Find transitions to active state (state 2)
                    I = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2 and int(gene_state[x - 1]) != 2]
                    J = [x for x in range(1, len(gene_state)) if int(gene_state[x]) == 2]

                    # Compute statistics for this run
                    mean_s5p = np.mean(s5p_promoter)
                    percent_active = 100 * len(J) / (len(gene_state) - 150)
                    percent_contact = 100 * sum(np.array(d_rp) < contact_dist) / len(d_rp)
                    mean_activation_dist = np.mean([d_rp[x] for x in I]) if len(I) > 0 else np.nan
                    five_percent_RG = np.percentile(d_rg[150:], 5)  # exclude first 150 frames
                    five_percent_RP = np.percentile(d_rp[150:], 5)

                    # Append to current averaging group
                    s5p_group.append(mean_s5p)
                    active_percent_group.append(percent_active)
                    contact_percent_group.append(percent_contact)
                    activation_distance_group.append(mean_activation_dist)
                    five_percent_RG_group.append(five_percent_RG)
                    five_percent_RP_group.append(five_percent_RP)

                    # --- When we reach 10 runs, take averages and save one summary row ---
                    if run_counter % 20 == 0:
                        point_row = pd.DataFrame(data=np.array([[
                            promoter,
                            threshold,
                            activation,
                            f"{run_counter - 9}-{run_counter}",  # record run range
                            np.nanmean(s5p_group),
                            np.nanmean(active_percent_group),
                            np.nanmean(contact_percent_group),
                            np.nanmean(activation_distance_group),
                            np.nanmean(five_percent_RG_group),
                            np.nanmean(five_percent_RP_group)
                        ]], dtype=object), columns=[
                            'Promoter', 'Threshold', 'Activation', 'Runs',
                            'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
                            '5percentRG', '5percentRP'
                        ])

                        df_summary = pd.concat([df_summary, point_row], ignore_index=True)

                        # Reset for next batch of 10 runs
                        s5p_group = []
                        active_percent_group = []
                        contact_percent_group = []
                        activation_distance_group = []
                        five_percent_RG_group = []
                        five_percent_RP_group = []

                # --- If remaining runs (<10) left at end, average them as well ---
                if len(s5p_group) > 0:
                    point_row = pd.DataFrame(data=np.array([[
                        promoter,
                        threshold,
                        activation,
                        f"{run_counter - (len(s5p_group)-1)}-{run_counter}",
                        np.nanmean(s5p_group),
                        np.nanmean(active_percent_group),
                        np.nanmean(contact_percent_group),
                        np.nanmean(activation_distance_group),
                        np.nanmean(five_percent_RG_group),
                        np.nanmean(five_percent_RP_group)
                    ]], dtype=object), columns=[
                        'Promoter', 'Threshold', 'Activation', 'Runs',
                        'S5PInt', 'S2PInt', 'Contact', 'DistActivation',
                        '5percentRG', '5percentRP'
                    ])
                    df_summary = pd.concat([df_summary, point_row], ignore_index=True)

# --------------------------
# Save averaged summary
# --------------------------
output_file = 'summary_contact_grouped_Thresholds10-100_5percentile_20xaveraged.txt'
df_summary.to_csv(output_file, index=False)
print(f"\nAveraged summary saved as: {output_file}")

