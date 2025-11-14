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
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=18)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
# plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"

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

df_summary = pd.DataFrame(columns = ['Promoter', 'Threshold', 'Activation', 'S5PInt', 'S2PInt', 'Contact', 'DistActivation'])
for condition in conditions:
    i_condition = i_condition + 1
    for j in range(len(promoters)):
        promoter=promoters[j]
        print('Promoter: '+str(promoter))
        print('-----------')
        for i in range(len(activations)):
            activation=activations[i]
            print('\nActivation: '+str(activation/1000))
            for k in range(len(thresholds)):
                threshold=thresholds[k]
                rootFolder='box11/Control_Promoter'+str(promoter)+'_Threshold'+str(threshold)+'_Act'+str(activation)
                if not os.path.exists(rootFolder+'/dist_active.txt'):
                    continue
                # rootFolder=condition
                data=[]
                myFile = open(rootFolder+'/dist_active.txt', 'r')
                for line in myFile:
                    data.append(float(line.strip()))
                all_dist_active = np.array(data)
                run_folders = os.listdir(rootFolder)
                run_folders = [x for x in run_folders if 'run' in x]
                i_r = 0
                s5p_group = []
                active_percent_group = []
                contact_percent_group = []
                activation_distance_group = []
                for run in run_folders:
                    i_r = i_r + 1
                    data=[]
                    myFile = open(rootFolder+'/'+run+'/geneTrack.txt', 'r')
                    for line in myFile:
                        # print([float(x) for x in line.strip().split(',')])
                        data.append([float(x) for x in line.strip().split(',')])
                    d_rp = [x[4] for x in data]
                    d_rg = [x[5] for x in data]
                    gene_state = [x[8] for x in data]
                    s5p_promoter = [x[6] for x in data]
                    I = [x for x in range(1, len(gene_state)) if int(gene_state[x])==2 and int(gene_state[x-1])!=2]
                    J = [x for x in range(1, len(gene_state)) if int(gene_state[x])==2]
         
                    #J = [int(x) for x in range(1, len(gene_state)) if int(gene_state[x]) == 2]
                    #Averaging over 10 consecutive runs
                    if i_r%20!=0:   #For every run except each 10th one, append values to the current group.When you reach the 10th run, calculate the averages of the last 10 runs, append them to the output, and reset the temporary lists.
                        s5p_group.extend(s5p_promoter)
                        active_percent_group.append(100*len(J)/(len(gene_state)-150))
                        #active_percent_group.append(0.054054*len(J))
                        #active_percent_group.append(100 * int(len(J)) / int(len(gene_state) - 150))
                        contact_percent_group.append(100*sum(np.array(d_rp)<contact_dist)/len(d_rp)) # Leave out the initial part of the time trace
                        activation_distance_group.extend([d_rp[x] for x in I])
                    else:
                        s5p_group.extend(s5p_promoter)
                        active_percent_group.append(100*len(J)/(len(gene_state)-150))
                        #active_percent_group.append(0.054054*len(J))
                        #active_percent_group.append(100 * int(len(J)) / int(len(gene_state) - 150))
                        contact_percent_group.append(100*sum(np.array(d_rp)<contact_dist)/len(d_rp))
                        activation_distance_group.extend([d_rp[x] for x in I])
                        #print(active_percent_group)
                        #print(gene_state)
                        point_row = pd.DataFrame(data=np.array([[promoter, threshold, activation, np.mean(s5p_group), np.mean(active_percent_group), np.mean(contact_percent_group), np.mean(activation_distance_group)]], dtype=object), columns = ['Promoter', 'Threshold', 'Activation', 'S5PInt', 'S2PInt', 'Contact', 'DistActivation'])
                        df_summary = pd.concat([df_summary, point_row], ignore_index=True)
                        s5p_group = []
                        active_percent_group = []
                        contact_percent_group = []
                        activation_distance_group = []
                        
                        
df_summary.to_csv('summary_contact_grouped_Thresholds10-100_20xaveraged.txt')
        

# %%
