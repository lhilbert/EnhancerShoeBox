# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from lammps import lammps, IPyLammps
import math
import random
import numpy as np
import os
import subprocess
import time
import shutil
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import Range1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.mlab as mlab
# from mayavi import mlab
from scipy.spatial import KDTree
from scipy import stats
from scipy.ndimage import gaussian_filter
import pickle
# import joypy
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from scipy.spatial.distance import cdist
import sys, getopt
import matplotlib as mpl
from PIL import Image
import seaborn as sns
import glob
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
# output_notebook()

# For the nuclear actin + Ser5P simulation in the nucleus, we need the following ingredients:
#     o Chromatin monomers: native, regulatory regions, inactive genes, transcribed genes (4 types)
#     o Actin monomers: ADP and ATP (2 types)
#     o Pol II: unphosphorylated and Ser5P (2 types)
#     o mRNA (1 type)

# Further, we need the following interactions:
#     o Regulatory regions ATTRACTS Pol II 
#     o Pol II ATTRACTS Pol II
#     o Pol II REPELS Transcribed genes
#     o Pol II REPELS Chromatin
#     o Chromatin REPELS Transcribed regions
#     o Chromatin REPELS Actin
#     o Actin ATTRACTS Pol II (?)

# We can have different types of genomic elements:
#     o Standard gene: short regulatory region beside a gene
#     o Visitor gene: long regulatory region beside a gene
#     o Amphiphile enhancer: very long regulatory region beside a gene
#     o Lonely enhancer: only very long regulatory region
    
# Pol II phosphorylation:
#     o Unphosphorylated Pol II binds to a genomic element gets phosphorylated at Ser5
#     o Ser5P Pol II can either dissociate or undergo pause release to transcribe the region
    
# Actin polymerization and Pol II transport:
#     o Actin is polymerized at the + end where ATP-actin concentration is high
#     o This is typically at the Ser5P cluster 
#     o So actin is polymerized inside the cluster and extends outwards
#     o Pol II is transported into the cluster along actin 
#     o Actin filaments can bundle up

# To do: 
# ---------
# ✅ Change GTP actin position to not blow up energies and forces - [fix nvt/limit xmax]
# ✅ Nucleus
# ✅ Nucleation/seeding rules: GTPm-GTPm, FilPlus-GTPm
# ✅ Explicitly specify plus and minus ends: randomly assign plus and minus at seeding
# ✅ And hence, get rid of GDP vs GTP-actin.
# ✅ Split FilamentEnds group into FilamentEndsPlus and FilamentEndsMinus
# ✅ Conc. of GTP-actin and pol/depol rate ratio: snapshots, lengths
# ✅ Preventing bond crossing: FENE bonds (Kremer 1990, Cutcosky 2013 thesis)
# ✅ Start with compacted chromosomes like at the end of mitosis: chromosome territories in the beginning
# ✅ Fix the atoms crossing wall issue: smaller timestep
# PREVIOUS MEETING:
# ✅ Try smaller actin monomers
# ✅ Color each chromosome in a different color and investigate chromosome territories and chromosome mixing for different bonds - Harmonic, FENE long, FENE short
#       - FENE long: more uniform bond length, good chromosome separation, homogeneous
#       - Timescales: 5 min for chromatin to get uniformly distributed before txn starts, sets the simulation time
# ✅ Spatial distributions: contrast, PCF g(r) [Other options: the Morisita index, Ripley’s K-function and Rényi’s generalized entropy]
# How realistic do we want the chromosomes to be wrt zebrafish?
# THOUGHTS: chromatin gets uniformly distributed in the presence of actin too:
#   - start actin polymerization late, after chromatin expands
#   - introduce chromatin-actin repulsion?
#   - introduce actin filament bundling to form a mesh?
#   - RBP -> RNP 

# In a regime where filaments form, without filaments -> no compartmentalization, but filaments induce compartmentalization 
# Increasing actin conc. from low (uniform chrimatin distr.) -> high (filaments forming): see how filaments interact with chromatin regions
# RBP -> RNP and their affect on where filaments form
# WRITING: Visual overview of the model
# WRITING: Visual overview of configurations
# WRITING: Description of the system

# 1-5 um long filaments. 1-2 um persistence length?
# average chromatin density: 0.015 bp/nm^3
# chromEMT: 15-30 % volume occupied by chromatin

# Actin polymerization:
# ---------------------

# ATP hydrolysis rates assumed?: ATP -> ADP-Pi = 0.3/s per subunit, ADP-Pi -> ADP = 0.002/s per subunit
# Observed rates: 0.7/s per subunit, 0.026/s per subunit (Brooks and Carlsson 2008)
# Implies: each subunit has 70% chance of getting hydrolyzed, each subunit has 2.6% chance of releasing the phosphate per second

# Both ATP-actin and ADP-actin can attach and fall off at both plus and minus ends, but their rates are vastly different.
# ATP+ k_on = 12/uM/s, k_off = 1.4/s, Cc = 0.12 uM
# ADP+ k_on = 4/uM/s, k_off = 8/s, Cc = 2 uM
# ATP- k_on = 1.3/uM/s, k_off = 0.8/s, Cc = 0.6 uM
# ADP- k_on = 0.3/um/s, k_off = 0.16/s, Cc = 2 uM 
# Meaning of parameters: 
#   - for each filament, in 1 second, 1 uM of free G-actin -> 12 added at + end, 0.3 fall of at - end, for instance
# Parameters in our model: concentration of G-actin, Cc+ and Cc- (k_off/k_on)

# Our assumptions:
# ----------------

# ATP hydrolysis + phosphate release is very slow. So, contribution of ADP-actin is not so large.
# Hence, there is only one type of actin present, and it has different on and off rates at the plus and minus ends. 

class Group(object):
    """This is a class to define atom groups (based on atom types as of now), the computes and fixes dependant on it, etc.
    It can be used to refresh groups after atom type resetting.
    :param name: name of group used in LAMMPS
    :type  name: string
    """

    L = None  # L is the common main instance of (I)PyLammps, set to none, but will be initialized in main so that it is common
    lmp = None # lmp is the common main instance of lammps, set to none, but will be initialized in main so that it is common
    name = None # name is the name of the group, default will be set by constructor
    all_groups = [] # all_groups is a list of all groups defined: starts as an empty list, will be appended in constructor
    atom_types = [] # atom_types is a list of included atom types in the group, set in add_types()
    computes = None # computes is a list of computes (dictionaries) defined on the group, initialized in constructor, changed in add_compute
    dumps = None # dumps is a list of dumps (dictionaries) defined on the group, initialized in constructor, changed in add_dump
    def __init__(self, name, atom_types=None, atom_ids=None):
        self.name = name
        self.computes =  [] # list of computes 
        self.dumps =  [] # list of computes 
        Group.all_groups.append(self)
        if atom_types is None:
            self.atom_types = []
            self.atom_ids = atom_ids
            self.group_type = "id"
            atom_ids_string=" ".join(str(item) for item in atom_ids)
            Group.L.group(self.name+' id '+atom_ids_string)
        else:
            self.atom_ids = []
            self.atom_types = atom_types # extend not needed as this is the first initialization
            self.group_type = "type"
            atom_types_string=" ".join(str(item) for item in atom_types)
            Group.L.group(self.name+' type '+atom_types_string)
              
    def add_types(self, atom_types):
        self.atom_types.extend(atom_types)
        atom_types_string=" ".join(str(item) for item in atom_types)
        Group.L.group(self.name+' type '+atom_types_string)
        
    def add_compute(self, compute_name, right_side_command):
        total_command=" ".join([compute_name,self.name,right_side_command])
        self.computes.append({'compute_name':compute_name, 'command':total_command})
        Group.L.compute(total_command)
        
    def add_dump(self, dump_name, right_side_command, dump_modify_command=None):
        total_command=" ".join([dump_name,self.name,right_side_command])
        Group.L.dump(total_command)
        if dump_modify_command is not None:
            total_modify=" ".join([dump_name,dump_modify_command])
            Group.L.dump_modify(total_modify)
        else:
            total_modify=None
        self.dumps.append({'dump_name':dump_name, 'command':total_command, 'modify':total_modify})
            
    def count_atoms(self):
        Group.L.variable('tempCount equal count('+self.name+')')
        a=Group.L.variables['tempCount'].value
        Group.L.variable('tempCount delete')
        return a
    
    def respawn(self):
        for compute_name in [sub['compute_name'] for sub in self.computes]: Group.L.uncompute(compute_name)
        for dump_name in [sub['dump_name'] for sub in self.dumps]: Group.L.undump(dump_name)
        Group.L.group(self.name,'delete')
        if self.group_type=="id":
            atom_ids_string=" ".join(str(item) for item in self.atom_ids)
            Group.L.group(self.name+' id '+atom_ids_string)
        if self.group_type=="type":
            atom_types_string=" ".join(str(item) for item in self.atom_types)
            Group.L.group(self.name+' type '+atom_types_string)
        for compute_command in [sub['command'] for sub in self.computes]: Group.L.compute(compute_command)
        for dump_command in [sub['command'] for sub in self.dumps]: Group.L.dump(dump_command)
        for modify_command in [sub['modify'] for sub in self.dumps]: 
            if modify_command is not None:
                Group.L.dump_modify(modify_command)
    
    def clear(self):
        for compute_name in [sub['compute_name'] for sub in self.computes]: Group.L.uncompute(compute_name)
        for dump_name in [sub['dump_name'] for sub in self.dumps]: Group.L.undump(dump_name)
        Group.L.group(self.name,'clear')
            
def getAtomProperties(LammpsObject, atomVector, n_columns):
    # atomVector should be an output of Group.lmp.extract_compute with number of rows as number of atoms in the LAMMPS system, first column should be ID of atom
    if atomVector: # Pointer exists
        atomVector_np=np.ctypeslib.as_array(atomVector.contents,shape=((LammpsObject.system.natoms,n_columns)))
        atomVector_np=atomVector_np[~np.isnan(atomVector_np[:,0]),:]
        atomVector_np=atomVector_np[np.nonzero(atomVector_np[:,0]),:]
        return atomVector_np[0]
    else:
        return np.empty((0,n_columns))
    
def getAtomPairs(LammpsObject, atomPairVector, thisGroup, n_columns):
    # atomPairVector should be an output of Group.lmp.extract_compute with number of rows as pairwise interactions between atoms in the group corresponding to the compute
    # only n_columns=2 columns allowed. somehow more columns messes this up
    if atomPairVector:
        # count number of atoms in the group corresponding to the compute
        Group.L.variable('atomCount equal count('+thisGroup+')')
        a = Group.L.variables['atomCount'].value
        maxIPs=a*(a-1)/2
        atomPairVector_np=np.ctypeslib.as_array(atomPairVector.contents,shape=((int(maxIPs),n_columns)))
        atomPairVector_np=atomPairVector_np[~np.isnan(atomPairVector_np[:,0]),:]
        atomPairVector_np=atomPairVector_np[atomPairVector_np[:,0]>0.1,:]
        atomPairVector_np=atomPairVector_np[np.nonzero(atomPairVector_np[:,0]),:]
        return atomPairVector_np[0]
    else:
        return np.empty((0,n_columns))

def makeFakeMicroscopyImages(atomPositions, atomTypes, sigma, time, out_folder, run_number):
    
    natoms = len(atomTypes)
    ser5p_atoms=np.array([j+1 for j in range(natoms) if atomTypes[j]==8])
    ser5p_positions = atomPositions[[xx-1 for xx in ser5p_atoms],:]
    H, edges = np.histogramdd(ser5p_positions, bins = (np.linspace(-11, 11, 22), np.linspace(-9, 9, 18), np.linspace(-7, 7, 14)))
    Ser5PImage = gaussian_filter(H, sigma=sigma)

    fig = plt.figure()
    ax = fig.subplots()
    f = ax.imshow(np.amax(Ser5PImage, 2).transpose(), cmap='gray')
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(out_folder+'/run'+str(run_number)+"/microscopy_files/Ser5PImage_"+str(time).zfill(7)+".png", bbox_inches="tight", pad_inches=0)

    ser2p_atoms=np.array([j+1 for j in range(natoms) if atomTypes[j] in [7,11]])
    ser2p_positions = atomPositions[[xx-1 for xx in ser2p_atoms],:]
    H, edges = np.histogramdd(ser2p_positions, bins = (np.linspace(-11, 11, 22), np.linspace(-9, 9, 18), np.linspace(-7, 7, 14)))
    Ser2PImage = gaussian_filter(H, sigma=sigma)

    fig = plt.figure()
    ax = fig.subplots()
    f = ax.imshow(np.amax(Ser2PImage, 2).transpose(), cmap='gray')
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(out_folder+'/run'+str(run_number)+"/microscopy_files/Ser2PImage_"+str(time).zfill(7)+".png", bbox_inches="tight", pad_inches=0)
    
    chromatin_atoms=np.array([j+1 for j in range(natoms) if atomTypes[j] in [1,2,6,7,9,10,11]])
    chromatin_positions = atomPositions[[xx-1 for xx in chromatin_atoms],:]
    H, edges = np.histogramdd(chromatin_positions, bins = (np.linspace(-11, 11, 22), np.linspace(-9, 9, 18), np.linspace(-7, 7, 14)))
    ChromatinImage = gaussian_filter(H, sigma=sigma)

    fig = plt.figure()
    ax = fig.subplots()
    f = ax.imshow(np.amax(ChromatinImage, 2).transpose(), cmap='gray')
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(out_folder+'/run'+str(run_number)+"/microscopy_files/ChromatinImage_"+str(time).zfill(7)+".png", bbox_inches="tight", pad_inches=0)

    gene_atoms=np.array([j+1 for j in range(natoms) if atomTypes[j] in [2,6,7,10,11]])
    gene_positions = atomPositions[[xx-1 for xx in gene_atoms],:]
    H, edges = np.histogramdd(gene_positions, bins = (np.linspace(-11, 11, 22), np.linspace(-9, 9, 18), np.linspace(-7, 7, 14)))
    GeneImage = gaussian_filter(H, sigma=sigma)

    fig = plt.figure()
    ax = fig.subplots()
    f = ax.imshow(np.amax(GeneImage, 2).transpose(), cmap='gray')
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(out_folder+'/run'+str(run_number)+"/microscopy_files/GeneImage_"+str(time).zfill(7)+".png", bbox_inches="tight", pad_inches=0)

    actin_atoms = np.array([j+1 for j in range(natoms) if atomTypes[j] in [3]])
    actin_positions = atomPositions[[xx-1 for xx in actin_atoms],:]
    H, edges = np.histogramdd(actin_positions, bins = (np.linspace(-11, 11, 22), np.linspace(-9, 9, 18), np.linspace(-7, 7, 14)))
    ActinImage = gaussian_filter(H, sigma=sigma)

    fig = plt.figure()
    ax = fig.subplots()
    f = ax.imshow(np.amax(ActinImage, 2).transpose(), cmap='gray')
    ax.axis('off')
    ax.invert_yaxis()
    plt.savefig(out_folder+'/run'+str(run_number)+"/microscopy_files/ActinImage_"+str(time).zfill(7)+".png", bbox_inches="tight", pad_inches=0)
    
def progress_bar(current, total, name, bar_length = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print(name+': [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    
def progress_bars(data, total, bar_length=20):
    
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    for i in range(len(data)):
        percent = float(data[i,1]) * 100 / total
        arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
        spaces  = ' ' * (bar_length - len(arrow))

        print('Run '+str(int(data[i,0]))+': [%s%s] %d %%' % (arrow, spaces, percent), end='\n')

try:
    opts, args = getopt.getopt(sys.argv[1:],"hb:r:t:o:m:c:p:a:x:",["box=","repeat=","total=","outfolder=","make_images=","condition=","promoter=","activation=","threshold="])
except getopt.GetoptError:
    print('run_single_ACTIN_SMALL_BOX.py -b <box_size> -r <repeat> -t <total_runs> -o <out_folder> -m <make_images> -c <condition> -p <promoter_length> -a <activation_rate> -x <threshold>')
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print('run_single_ACTIN_SMALL_BOX.py -b <box_size> -r <repeat> -t <total_runs> -o <out_folder> -m <make_images> -c <condition> -p <promoter_length> -a <activation_rate> -x <threshold>')
        sys.exit()
    elif opt in ("-b", "--box"):
        box = int(arg)
    elif opt in ("-r", "--repeat"):
        run_number = int(arg)
    elif opt in ("-t", "--total"):
        total_runs = int(arg)
    elif opt in ("-o", "--outfolder"):
        out_folder = arg
    elif opt in ("-m", "--make_images"):
        make_snapshots = int(arg)
    elif opt in ("-c", "--condition"):
        condition = arg
    elif opt in ("-p", "--promoter"):
        promoter_length = int(arg)
    elif opt in ("-a", "--activation"):
        p_gene_activation = float(arg)/1000
    elif opt in ("-x", "--threshold"):
        ser5p_to_activate = int(arg)

# if run_number<=100:
#     make_snapshots = 1
     
folders_to_make = ["figures", "image_files", "summaries"]
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
if not os.path.exists(out_folder+"/parallel_counter"):
    os.makedirs(out_folder+"/parallel_counter")
if not os.path.exists(out_folder+'/run'+str(run_number)):
    os.makedirs(out_folder+'/run'+str(run_number))
    for i in range(len(folders_to_make)):
        if not os.path.exists(out_folder+'/run'+str(run_number)+"/"+folders_to_make[i]):
            os.makedirs(out_folder+'/run'+str(run_number)+"/"+folders_to_make[i])
    
if os.path.exists('../log.lammps'):
    os.remove('../log.lammps')
    
Group.lmp = lammps()
Group.L = IPyLammps(ptr=Group.lmp)
Group.L.log(out_folder+'/run'+str(run_number)+'/debug_RSQ.lammps')

sig_chromatin=60 # nm (Chromatin vol fraction 0.06)
    
if box==9:
    Ntot = 300 # 10 Mb
    number_of_Ser5P = 300
    number_of_RBP = 450    
elif box==10:
    Ntot = 336 # 10 Mb
    number_of_Ser5P = 336
    number_of_RBP = 500
elif box==11:
    Ntot = 370 # 10 Mb
    number_of_Ser5P = 370
    number_of_RBP = 550
elif box==12:
    Ntot = 400 # 10 Mb
    number_of_Ser5P = 400
    number_of_RBP = 600
elif box==13:
    Ntot = 433 # 10 Mb
    number_of_Ser5P = 433
    number_of_RBP = 650
elif box==15:
    Ntot = 500 # 10 Mb
    number_of_Ser5P = 500
    number_of_RBP = 750
elif box==18:
    Ntot = 600
    number_of_Ser5P = 600
    number_of_RBP = 900
elif box==21:
    Ntot = 700
    number_of_Ser5P = 700
    number_of_RBP = 1050
elif box==24:
    Ntot = 800
    number_of_Ser5P = 800
    number_of_RBP = 1200
        
# number_of_RBP = number_of_RBP*2

length_gene=5
NChromosomes=25
make_plots=1
make_microscopy=0
print_verbose=0
move_added_atoms=1
move_removed_atoms=1
move_rnp_atoms=1
dump_positions=0

gRE=1
fRE=1
aE_rnp_rnp=0.5 #1.0 
aE_rnp_ag=1.0
aE_ser5p_ser5p=0.5 # 1.0
aE_ser5p_reg=1.5
aE_ser5p_prom=1
filEndEffectRadius=3 #3
monomerEffectRadius=2 #2
pol2release_radius=3
cluster_r = np.linspace(0.65, 2.5, 50)

# RBP -> RNP conversion
transcription_rnp_radius=2.5
sig_actin=0.3
w_sa=0.2
sig_chr=1.0 
sig_rbp=0.3
sig_rnp=1.0
sig_ser5p=0.3
d_ca=1.5 # repulsion distance between actin and inactive chromatin

# sigmas
s_crb=(sig_chr+sig_rbp)/2
s_crn=(sig_chr+sig_rnp)/2
s_cs=(sig_chr+sig_ser5p)/2
s_rbrn=(sig_rbp+sig_rnp)/2
s_rbs=(sig_rbp+sig_ser5p)/2
s_rns=(sig_rnp+sig_ser5p)/2
# raw cutoffs
c_cc=2**(1/6)
c_crb=((sig_chr+sig_rbp)/2)*(2**(1/6))
c_crn=((sig_chr+sig_rnp)/2)*(2**(1/6))
c_cs=((sig_chr+sig_ser5p)/2)*(2**(1/6))
c_rbrn=((sig_rbp+sig_rnp)/2)*(2**(1/6))
c_rbs=((sig_rbp+sig_ser5p)/2)*(2**(1/6))
c_rns=((sig_rnp+sig_ser5p)/2)*(2**(1/6))
 
    
fileBase_input='ser5P'+str(number_of_Ser5P)+'_RBP'+str(number_of_RBP)+'_0RNP_promoter'+str(promoter_length)+'_bs'+str(box)
filePath='input_files/IC_'+fileBase_input+'.data' 

""" if promoter_length==1:
    ser5p_to_activate=15 #10
elif promoter_length==2:
    ser5p_to_activate=30 #20
elif promoter_length==3:
    ser5p_to_activate=50 #40
elif promoter_length==4:
    ser5p_to_activate=70 #60 """ 
    
# Timescales of various processes
NRuns=2000
tRun=200
dt=0.005 # 0.03s
delt=0.03*tRun # 6s

gene_switching_off=0

n_soft=10 # 60s
n_eq=10 # 60 more s
t_ser5p_on = 15 # polymerization can't begin before Ser5P on
t_induction_on=150 # 3 min = 180s
t_activation_on=t_induction_on 

treatment_duration = 0
if condition=="Hexanediol":
    treatment_duration = int(3*60/0.6)
elif condition=="JQ-1":
    treatment_duration = int(30*60/0.6)
elif condition=="Flavopiridol":
    treatment_duration = int(30*60/0.6)
    
t_transcription_on=100*NRuns # 5 min = 300s
t_induction_off=100*NRuns # 5 min = 300s
t_activation_off=100*NRuns # 10 min = 600s
t_transcription_off=100*NRuns # 10 min = 600s

tImageDump=10*tRun

if gene_switching_off:
    t_transcription_on=100*NRuns # 5 min = 300s
    t_induction_on=100*NRuns # 2 min = 120s
    t_activation_on=100*NRuns # 5 min = 300s
# each run is 6s
# 30 min = 1800 s = 300 runs
# 1 hour = 3600 s = 600 runs

t_on_plus=1
t_on_minus=40 # (k_on_plus/k_on_minus) 10
t_off_plus=100*NRuns # (k_on_plus/k_off_plus) 8
t_off_minus=5 # (k_on_plus/k_off_minus)
t_rbprnp=1 # conversion every 48 seconds, a burst
p_rbprnp=1/8
# RNP only degradation ~ 20 min:
t_rnprbp=10
p_rnprbp=1/20 # every 10 steps
# RNP export ~ 6-10 min **PREFERRED MODEL, but where do we convert? everywhere randomly or near the nuclear periphery
t_rnprbp=10 # 2
p_rnprbp=1/30
loc_rnprbp=1 # 1: everywhere randomly, 2: near the nuclear periphery
t_gene_induction=1 # every 30s
t_gene_activation=1 # every 60s
# p_gene_activation=0.05 # 0.33 # with this probability, activate an induced visitor gene (keeping activation stochastic)
t_active_duration=50 # duration for which a gene is active: 50 -> 5 min
t_gene_inactivation=1 # 5; every 60s 
fraction_induce=1.0 # if run every step (6s), this gives 1 per 3 min rate
fraction_activate=1/20 # if run every step, results in 1 per 2 min rate, irrelevant for visitor genes
fraction_inactivate=1/20 # 1/10; run every 15 steps, results in 1 per 15 min rate
activation_delay=2

f_rnp_final = 0.9
t_rnp_final = 2000
slope_rnp = f_rnp_final/(t_rnp_final-t_induction_on)
c_rnp = -slope_rnp*t_induction_on

# Setting up the simulation
while 1:
    # READ IN DATA
    Group.L.atom_style('angle')
    Group.L.boundary('f','f','f')
    Group.L.read_data(filePath, 'extra/bond/per/atom 100 extra/angle/per/atom 100 extra/special/per/atom 100')
    Group.L.neighbor(0.3,'multi')
    # was delay 1
    Group.L.neigh_modify('every',1,'delay',10,'check','yes','one',10000,'page',100000)
    # Group.L.neigh_modify('every',1,'delay',1,'check','yes')
    Group.L.comm_modify('cutoff 6')

    nucleusRadius=abs(Group.L.system.xlo)
    
    atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)),((Group.L.system.natoms,1)))
        
    # DEFINE GROUPS
    NativeChromatin = Group('NativeChromatin',[1])
    InactiveGenes = Group('InactiveGenes',[2])
    InactiveChromatin = Group('InactiveChromatin',[1,2]) 
    InducedGenes = Group('InducedGenes', atom_types=[6])
    ActiveGenes = Group('ActiveGenes', atom_types=[7])
    RegulatoryChromatin = Group('RegulatoryChromatin', atom_types=[9])
    Promoters = Group('Promoters', atom_types=[10])
    ActivePromoters = Group('ActivePromoters', atom_types=[11])
    RBP = Group('RBP', atom_types=[4])
    RNP = Group('RNP', atom_types=[5])
    RBPRNP = Group('RBPRNP', atom_types=[4,5])
    Ser5P = Group('Ser5P', atom_types=[8])
    ImageGroup = Group('ImageGroup',[1,2,6,7,8,9,10,11])
    
    Group.L.angle_style('cosine')
    Group.L.angle_coeff(1, 100/sig_chromatin) # Chromatin # was 1: 100 nm before, now 2: 2*70 = 140 nm
    Group.L.angle_coeff(2, 33.0) # Actin filament, was 33
    # K\sigma is the persistence length

    # Pair interaction between non-bonded atoms
    Group.L.pair_style('soft',2**(1/6))
    Group.L.pair_coeff('*','*',0,2**(1/6)) # default for chromatin hardcore-repulsion
    # Chromatin and inactive genes
    Group.L.pair_coeff('1*2','4',0,c_crb) # RBP
    Group.L.pair_coeff('1*2','5',0,1.5*c_crn) # RNP
    Group.L.pair_coeff('1*2','6',0,c_cc) # induced 
    Group.L.pair_coeff('1*2','7',0,1.5*c_cc) # active genes
    Group.L.pair_coeff('1*2','8',0,1.0*c_cs) # Ser5P 2.0
    # RBP
    Group.L.pair_coeff('4','4',0,sig_rbp*c_cc)
    Group.L.pair_coeff('4','5',0,c_rbrn)
    Group.L.pair_coeff('4','6*7',0,c_crb)
    Group.L.pair_coeff('4','8',0,c_rbs)
    Group.L.pair_coeff('4','9*11',0,c_crb)
    # RNP
    Group.L.pair_coeff('5','5',0,sig_rnp*c_cc)
    Group.L.pair_coeff('5','6',0,1.5*c_crn) # RNP-induced gene repulsion
    Group.L.pair_coeff('5','7',0,c_crn)
    Group.L.pair_coeff('5','8',0,c_rns)
    Group.L.pair_coeff('5','9*11',0,1.5*c_crn) # RNP-regulatory element repulsion
    # Induced genes
    Group.L.pair_coeff('6','7',0,1.5*c_cc) # active
    Group.L.pair_coeff('6','8',0,c_cs) # ser5p
    Group.L.pair_coeff('6','9*11',0,c_cc) # regulatory regions
    # Active genes
    Group.L.pair_coeff('7','7',0,1.5*c_cc) # active
    Group.L.pair_coeff('7','8',0,1.0*c_cs) # ser5p 2.0
    Group.L.pair_coeff('7','9*11',0,1.5*c_cc) # regulatory regions
    # Ser5P
    Group.L.pair_coeff('8','8',0,sig_ser5p*c_cc) # ser5p
    Group.L.pair_coeff('8','9*11',0,c_cs) # regulatory regions

    Group.L.variable('prefactor equal ramp(0,60)')
    Group.L.fix('s1 all adapt 1 pair soft a * * v_prefactor')

    # Pair interaction between bonded atoms
    Group.L.bond_style('harmonic')
    # Group.L.bond_coeff(1,'fene',30.0,1.5,0.001,1.0) # eq. distance around 0.6
    Group.L.bond_coeff(1,'90.0 1') 
    Group.L.bond_coeff(2,'90.0 0.3')
    # Group.L.special_bonds('fene')
    
    Group.L.variable('chromatinr0 equal ramp(0.033,1.0)')
    Group.L.fix('s2 all adapt 10 bond harmonic r0 1 v_chromatinr0')
    
    # FIXES
    Group.L.variable('seed equal',random.randint(1, 100000)) 
    Group.L.fix('1 all nve') # 2.0
    Group.L.fix('2 all langevin',1,1,0.5,random.randint(1, 100000))
    Group.L.fix('wallhi all wall/harmonic xlo EDGE 100 0.0 3.0 xhi EDGE 100 0.0 3.0 ylo EDGE 100 0.0 3.0 yhi EDGE 100 0.0 3.0 zlo EDGE 100 0.0 3.0 zhi EDGE 100 0.0 3.0')
    
    Group.L.region('centerSlice block',-50,50,-50,50,-3,3)
    Group.L.group('atomsCS dynamic all region centerSlice')
    Group.L.group('ser5pCS dynamic Ser5P region centerSlice')
    
    if Ntot==300:
        Anchors = Group('Anchors', atom_ids=[1,100,101,300])
    elif Ntot==336:
        Anchors = Group('Anchors', atom_ids=[1,112,113,336])
    elif Ntot==370:
        Anchors = Group('Anchors', atom_ids=[1,123,124,370])
    elif Ntot==400:
        Anchors = Group('Anchors', atom_ids=[1,133,134,400])
    elif Ntot==433:
        Anchors = Group('Anchors', atom_ids=[1,144,145,433])
    elif Ntot==500:
        Anchors = Group('Anchors', atom_ids=[1,166,167,500])
    elif Ntot==600:
        Anchors = Group('Anchors', atom_ids=[1,200,201,600])
    elif Ntot==700:
        Anchors = Group('Anchors', atom_ids=[1,233,234,700])
    elif Ntot==800:
        Anchors = Group('Anchors', atom_ids=[1,266,267,800])
        
    Group.L.fix('freezeAnchors Anchors setforce 0.0 0.0 0.0')
    
    if make_snapshots:
        # new color scheme
        image_dump_string = 'imageDump pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 color reblue 0.13 0.33 0.62 color s5p 0.69 0.33 0.33 color indgene 0.47 0.03 0.03 acolor 1 lightgrey acolor 2 lightgrey acolor 6 indgene acolor 7 verydarkgrey acolor 8 s5p acolor 9 reblue acolor 10 reblue acolor 11 reblue'
        Group.L.dump('imageDump ImageGroup image',tImageDump, out_folder+'/run'+str(run_number)+'/image_files/BEFORE_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
        Group.L.dump_modify(image_dump_string)
    
    Group.L.timestep(dt)
    break
    
inactive_gene_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==2])
inactive_gene_lociStart=[]
induced_gene_lociStart=[]
active_gene_lociStart=[]
for i in range(len(inactive_gene_atoms)):
    if atomTypes[inactive_gene_atoms[i]-2]==2:
        continue
    a = inactive_gene_atoms[i]
    inactive_gene_lociStart.append(a)
    
gene_start_atoms = inactive_gene_lociStart.copy()
all_actin=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==3])  
chromatin_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [1,2,6,7,9,10,11]])
rbp_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==4])
rnp_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==5])
rbprnp_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [4,5]])
ser5p_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==8])
regulatory_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [9]])
all_promoter_atoms=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [10]])
promoter_atoms=all_promoter_atoms[list(range(0, len(all_promoter_atoms), promoter_length))] # use the length of promoter here!

n_rnp=[len(rnp_atoms)]
ser5p_xprofilematrix = []
actin_xprofilematrix = []
reg_position = []
reg_end1_position = []
reg_end2_position = []
gene_position = []
induced_runs = [] 
active_runs = []
for j in range(len(gene_start_atoms)):
    induced_runs.append([])
    active_runs.append([])
ser5p_around_cluster = []
actin_around_cluster = []
d_rg = []
d_rp = []
ser5p_around_promoter = []
ser5p_around_gene = []
total_active_steps = 0
gene_active_duration = [0] * len(inactive_gene_lociStart)
marked_for_activation=[]
delay_activation=[]

myGeneFile = open(out_folder+"/run"+str(run_number)+"/geneTrack.txt", "w")
ser5pAroundClusterFile = open(out_folder+"/run"+str(run_number)+"/ser5pAroundCluster.txt", "w")
ser5pAroundClusterFile.write("t,")
for j in range(len(cluster_r[1:])):
    if j<(len(cluster_r[1:])-1):
        ser5pAroundClusterFile.write(str(cluster_r[j+1])+",")
    else:
        ser5pAroundClusterFile.write(str(cluster_r[j+1])+"\n")
                
# RUNS
for i in range(NRuns+treatment_duration):
    
    myFile = open(out_folder+"/parallel_counter/progress_run"+str(run_number)+".txt", "w")
    myFile.write(str(run_number)+','+str(i+1))
    myFile.close()

    data=[]
    for file in os.listdir(out_folder+"/parallel_counter"):
        if file.startswith("progress_run"):
            myFile = open(out_folder+"/parallel_counter/"+file, 'r')
            for line in myFile:
                array = line.strip().split(',')
                data.append([float(array[0]), float(array[1])])
            myFile.close()
    data = np.array(data)
        
    # progress_bar(i, NRuns-1, "Runs done "+str(initial_count)+"/"+str(total_runs))
    subprocess.run(["clear"])
    print("Condition: "+condition+", Promoter: "+str(promoter_length)+", Total runs: "+str(total_runs)+"\n")
    progress_bars(data, NRuns)
    
    if print_verbose:
        print('-----------------------------------------------------------------------')
        print('Run '+str(i))
    ppmFiles = glob.glob(out_folder+'/run'+str(run_number)+'/image_files/*.ppm')
    for i_file in range(len(ppmFiles)):
        image = Image.open(ppmFiles[i_file])
        image.save(ppmFiles[i_file].split(".")[0]+'.png', quality=100)
        os.remove(ppmFiles[i_file])
    
    if (i+1)==n_soft:
        Group.L.unfix('s1')
        Group.L.unfix('s2')
        if print_verbose:
            print("Soft done.")
        Group.L.pair_style('table lookup 30000')
        # LJ potentials - default interaction between particles 
        Group.L.pair_coeff('*','*', 'LJsoft_RSQ.table CHROMATIN_CHROMATIN', 1.2) # 1.12 default for chromatin hardcore-repulsion
        # Chromatin and inactive genes
        Group.L.pair_coeff('1*2','3','LJsoft_RSQ.table ACTIN_CHROMATIN', 1.2) # 1.09 actin
        Group.L.pair_coeff('1*2','4','LJsoft_RSQ.table RBP_CHROMATIN', 0.75) # 0.73 RBP
        Group.L.pair_coeff('1*2','5','LJsoft_RSQ.table RNP_CHROMATIN', 1.75) # 1.68
        # Group.L.pair_coeff('1*2','5','LJsoft_RSQ.table RNP_ACTIVE', 1.2) # RNP 1.5*c_crn
        Group.L.pair_coeff('1*2','6','LJsoft_RSQ.table INDUCED_CHROMATIN', 1.2) # 1.12 induced genes
        Group.L.pair_coeff('1*2','7','LJsoft_RSQ.table ACTIVE_CHROMATIN', 1.75) # 1.68 active genes
        # RBP
        Group.L.pair_coeff('4','4','LJsoft_RSQ.table RBP_RBP', 0.4) # 0.34
        Group.L.pair_coeff('4','5','LJsoft_RSQ.table RNP_RBP', 0.75) # 0.73 c_rbrn
        Group.L.pair_coeff('4','6','LJsoft_RSQ.table RBP_INDUCED', 0.75) # 0.73
        Group.L.pair_coeff('4','7','LJsoft_RSQ.table RBP_ACTIVE', 0.75) # 0.73
        Group.L.pair_coeff('4','8','LJsoft_RSQ.table RBP_SER5P', 0.4) # 0.34
        Group.L.pair_coeff('4','9*11','LJsoft_RSQ.table RBP_REGULATORY', 0.75) # 0.73
        # RNP
        # Group.L.pair_coeff('5','5','LJsoft_RSQ.table RNP_RNP_HCREPULSIVE', 1.2) # 1.12
        Group.L.pair_coeff('5','5','LJsoft_RSQ.table RNP_RNP_ATTRACTION', 2.9) # 2.8 
        Group.L.pair_coeff('5','6','LJsoft_RSQ.table RNP_INDUCED', 1.75) # 1.68
        Group.L.pair_coeff('5','7','LJsoft_RSQ.table RNP_ACTIVE', 2.9) # 2.8
        Group.L.pair_coeff('5','8','LJsoft_RSQ.table RNP_SER5P', 0.75) # 0.73
        Group.L.pair_coeff('5','9','LJsoft_RSQ.table RNP_REGULATORY', 1.75) # 1.68
        Group.L.pair_coeff('5','10*11','LJsoft_RSQ.table RNP_PROMOTER', 1.75) # 1.68
        # Induced genes
        Group.L.pair_coeff('6','7','LJsoft_RSQ.table INDUCED_ACTIVE', 1.75) # 1.68
        # Active genes
        Group.L.pair_coeff('7','7','LJsoft_RSQ.table ACTIVE_ACTIVE', 1.75) # 1.68
        Group.L.pair_coeff('7','9','LJsoft_RSQ.table ACTIVE_REGULATORY', 1.75) # 1.68
        # Ser5P behaves like RBP till it is activated for all interactions
        Group.L.pair_coeff('1*2','8','LJsoft_RSQ.table RBP_CHROMATIN', 0.75) # 0.73
        Group.L.pair_coeff('3','8','LJsoft_RSQ.table ACTIN_RBP', 0.4) # 0.34
        Group.L.pair_coeff('6','8','LJsoft_RSQ.table RBP_INDUCED', 0.75) # 0.73
        Group.L.pair_coeff('7','8','LJsoft_RSQ.table RBP_ACTIVE', 0.75) # 0.73
        Group.L.pair_coeff('8','8','LJsoft_RSQ.table RBP_SER5P', 0.4) # 0.34
        Group.L.pair_coeff('8','9*11','LJsoft_RSQ.table RBP_REGULATORY', 0.75) # 0.73
        
    if (i+1)==t_ser5p_on:
        # Ser5P
        if print_verbose:
            print("Switching on Ser5P interactions.")
        Group.L.pair_coeff('8','8','LJsoft_RSQ.table SER5P_SER5P', 0.9) # 0.84
        Group.L.pair_coeff('6','8','LJsoft_RSQ.table INDUCED_SER5P', 0.75) # 0.73
        Group.L.pair_coeff('7','8','LJsoft_RSQ.table ACTIVE_SER5P', 1.2) # 1.12
        Group.L.pair_coeff('1*2','8','LJsoft_RSQ.table CHROMATIN_SER5P', 1.5) # 1.46
        Group.L.pair_coeff('8','9','LJsoft_RSQ.table SER5P_REGULATORY_WEAK', 1.9) # 1.8
        Group.L.pair_coeff('8','10','LJsoft_RSQ.table PROMOTER_SER5P_STRONG', 1.9) # 1.8
        Group.L.pair_coeff('8','11','LJsoft_RSQ.table ACTIVEPROMOTER_SER5P', 1.2) # 1.09
        
    if i==NRuns:
        if condition=="Hexanediol":
            Group.L.pair_coeff('8','8','LJsoft_RSQ.table RBP_RBP', 0.4) # ser5p
            Group.L.pair_coeff('8','9','LJsoft_RSQ.table SER5P_REGULATORY_JQ1', 2.0) # ser5p
            tConditionImageDump = 10*tRun
        elif condition=="JQ-1":
            # Group.L.pair_coeff('8','9','LJsoft_RSQ.table SER5P_REGULATORY_JQ1', 2.0) # ser5p
            Group.L.pair_coeff('8','9','LJsoft_RSQ.table RBP_REGULATORY', 0.75) # ser5p
            tConditionImageDump = 25*tRun
        elif condition=="Flavopiridol":
            t_activation_off = NRuns
            tConditionImageDump = 25*tRun
        if make_snapshots:
            Group.L.undump('imageDump')
            image_dump_string = 'imageDump pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 color reblue 0.13 0.33 0.62 color s5p 0.69 0.33 0.33 color indgene 0.47 0.03 0.03 acolor 1 lightgrey acolor 2 lightgrey acolor 6 indgene acolor 7 verydarkgrey acolor 8 s5p acolor 9 reblue acolor 10 reblue acolor 11 reblue'
            Group.L.dump('imageDump ImageGroup image',tConditionImageDump, out_folder+'/run'+str(run_number)+'/image_files/AFTER_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
            Group.L.dump_modify(image_dump_string)
            
    if (i+1)==(n_soft+n_eq):
        if print_verbose:
            print("Equilibration done.")
        
    # COMPUTES and dumps dependent on them
    # Group.L.compute('AtomProperties all property/atom id type x y z')
    if (i+1)<n_soft:
        Group.L.run(tRun, 'start 0 stop', n_soft*tRun)
    else:
        Group.L.run(tRun)
            
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
    atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)),((Group.L.system.natoms,1)))
    
    ser5p_xpositions=atomPositions[[x-1 for x in ser5p_atoms],0]
    ser5p_xp = np.histogram(ser5p_xpositions, bins=2*box, range=(-box, box))[0]
    ser5p_xprofilematrix.append(list(ser5p_xp))
    
    regulatory_xpositions=atomPositions[[x-1 for x in regulatory_atoms],0]
    regulatory_xp = np.mean(regulatory_xpositions)
    reg_position.append(regulatory_xp)
    if len(regulatory_xpositions)>0:
        reg_end1_position.append(np.min(regulatory_xpositions))
        reg_end2_position.append(np.max(regulatory_xpositions))
    
    RRPosition = np.mean(atomPositions[[x-1 for x in regulatory_atoms]], axis=0)
    GenesPosition = atomPositions[[x-1 for x in gene_start_atoms]]
    d_rg.append([np.sqrt(np.sum((RRPosition - GenesPosition[k,:]) ** 2)) for k in range(len(GenesPosition))])
        
    gene_xpositions=atomPositions[[x-1 for x in promoter_atoms],0] # should we track genes instead?
    gene_xp = np.mean(gene_xpositions)
    gene_position.append(list(gene_xpositions))
    
    ser5p_positions=atomPositions[[x-1 for x in ser5p_atoms],:]
    PromoterPositions = atomPositions[[x-promoter_length+1 for x in gene_start_atoms],:]
    # x-promoter_length+2: promoter starting atom
    # x-promoter_length+1: promoter -1 atom
    # d_rp.append([np.sqrt(np.sum((RRPosition - PromoterPositions[k,:]) ** 2)) for k in range(len(PromoterPositions))])
    
    d_rp.append([np.sqrt(np.sum((RRPosition - np.mean(atomPositions[[y-1 for y in [gene_start_atoms[0]-1-x for x in range(promoter_length)]],:], axis=0)) ** 2))])
    
    ds = cdist(PromoterPositions, ser5p_positions, 'euclidean')
    IPs = ds<pol2release_radius
    ser5p_around_promoter.append([sum(IPs[x]) for x in range(len(IPs))])
    
    ds = cdist(GenesPosition, ser5p_positions, 'euclidean')
    IPs = ds<pol2release_radius
    ser5p_around_gene.append([sum(IPs[x]) for x in range(len(IPs))])
        
    # Update gene active duration:
    for j in range(len(gene_start_atoms)):
        goi = gene_start_atoms[j]
        if goi in active_gene_lociStart:
            gene_active_duration[j] = gene_active_duration[j] + 1
            total_active_steps = total_active_steps + 1
            
    # ---------------------------------------
    # GENE INDUCTION
    # ---------------------------------------

    if i%t_gene_induction==0 and i>=t_induction_on and i<t_induction_off:
        if print_verbose:
            print('Inducing '+str(round(fraction_induce*100))+' percent of '+str(len(inactive_gene_lociStart))+' inactive genes.')
        loci_to_induce = random.sample(inactive_gene_lociStart, math.ceil(fraction_induce*len(inactive_gene_lociStart)))  #randomly select a subset of inactive genes (loci) to be induced during this induction step
        for j in loci_to_induce:
            for x in range(j, j+length_gene):
                Group.L.set('atom', x, 'type', 6)
            inactive_gene_lociStart.remove(j)
            induced_gene_lociStart.append(j)
            
        InactiveChromatin.respawn()
        InactiveGenes.respawn()
        InducedGenes.respawn()
        
    # ---------------------------------------
    # GENE ACTIVATION
    # ---------------------------------------
    
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
    ser5p_positions=atomPositions[[x-1 for x in ser5p_atoms],:]
    induced_gene_positions = atomPositions[[x-promoter_length+1 for x in induced_gene_lociStart],:] 
    # x-promoter_length+2: promoter starting atom
    # x-promoter_length+1: promoter -1 atom
    
    if i%t_gene_activation==0 and i>=t_activation_on and i<t_activation_off:
        if print_verbose:
            print("Gene activation ...") 
        ser5p_to_relocate=[]
        ds = cdist(induced_gene_positions, ser5p_positions, 'euclidean')
        IPs = ds<pol2release_radius
        Iips = [x for x in range(len(IPs)) if sum(IPs[x])>ser5p_to_activate]
        induced_gene_lociStart_copy = induced_gene_lociStart.copy()
        if print_verbose:
            print('Activating '+str(len(Iips))+' visitor genes from '+str(len(induced_gene_lociStart))+' induced visitor genes.')
        for j in range(len(Iips)):
            if random.random()>p_gene_activation:
                continue
            # Convert induced to active genes
            tbc = Iips[j]
            y = induced_gene_lociStart[tbc]
            marked_for_activation.append(y)
            delay_activation.append(0)
            induced_gene_lociStart_copy.remove(y)
            active_gene_lociStart.append(y)
            if make_snapshots:
                image_dump_string = 'imageDumpAct pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 acolor 1 lightgrey acolor 2 verydarkgrey acolor 3 chartreuse acolor 4 yellow acolor 5 aqua acolor 6 maroon acolor 7 darkgreen acolor 8 indianred acolor 9 blue acolor 10 lightblue'
                Group.L.dump('imageDumpAct ImageGroup image',tRun*5, out_folder+'/run'+str(run_number)+'/image_files/BEFORE_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
                Group.L.dump_modify(image_dump_string)
            
        induced_gene_lociStart = induced_gene_lociStart_copy.copy()
                
    marked_for_activation_copy = marked_for_activation.copy()
    delay_activation_copy = delay_activation.copy()
    for j in range(len(marked_for_activation_copy)):
        y = marked_for_activation_copy[j]
        if delay_activation_copy[j]<activation_delay:
            delay_activation_copy[j] = delay_activation_copy[j] + 1
            continue
        marked_for_activation.remove(y)
        del delay_activation_copy[j]
        for x in range(y, y+length_gene):
            Group.L.set('atom', x, 'type', 7)
        for x in range(1,promoter_length+1):
            Group.L.set('atom', y-x, 'type', 11)
    
    delay_activation = delay_activation_copy.copy()
    InducedGenes.respawn()
    ActiveGenes.respawn()
    Promoters.respawn()
    ActivePromoters.respawn()
         
    # ---------------------------------------
    # GENE INACTIVATION
    # ---------------------------------------

    if i%t_gene_inactivation==0 and i>=t_activation_on:
        for j in range(len(gene_start_atoms)):
            goi = gene_start_atoms[j]
            if goi in active_gene_lociStart:
                duration_active = gene_active_duration[j]
                if duration_active>=t_active_duration:
                    gene_active_duration[j] = 0
                    for x in range(goi, goi+length_gene):
                        Group.L.set('atom', x, 'type', 2)
                    for x in range(1,promoter_length+1):
                        Group.L.set('atom', goi-x, 'type', 10)
                    active_gene_lociStart.remove(goi)
                    inactive_gene_lociStart.append(goi) 
                    if make_snapshots:
                        Group.L.undump('imageDumpAct')
            
        """ print('Inactivating '+str(round(fraction_inactivate*100))+' percent of '+str(len(active_gene_lociStart))+' active genes.')
        loci_to_inactivate = random.sample(active_gene_lociStart, math.ceil(fraction_inactivate*len(active_gene_lociStart)))
        for j in loci_to_inactivate:
            for x in range(j, j+length_gene):
                Group.L.set('atom', x, 'type', 2)
            Group.L.set('atom', j-1, 'type', 10)
            Group.L.set('atom', j-2, 'type', 10)
            Group.L.set('atom', j-3, 'type', 10)
            Group.L.set('atom', j-4, 'type', 10)
            active_gene_lociStart.remove(j)
            inactive_gene_lociStart.append(j)  """
            
        InactiveChromatin.respawn()
        InactiveGenes.respawn()
        ActiveGenes.respawn()
        Promoters.respawn()
        ActivePromoters.respawn()
        
    # ---------------------------------------
    # RBP <-> RNP conversions
    # ---------------------------------------
    
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
    rbp_positions = atomPositions[[x-1 for x in rbp_atoms],:]
    
    rnp_atoms_list = list(rnp_atoms)
    rbp_atoms_list = list(rbp_atoms)
    
    #The script gathers atom positions and uses positions of RBP / RNP particles to compute distances, clusters, and to decide which RBPs will convert to RNPs:
    
    if i>=t_transcription_on and i<t_transcription_off and i%t_rbprnp==0:
        active_genes_transcribing = random.sample(active_gene_lociStart, math.ceil(p_rbprnp*len(active_gene_lociStart)))
        if print_verbose:
            print('RBP -> RNP conversion by '+str(len(active_genes_transcribing))+' active genes.') 
        active_gene_atoms = []
        for j in active_genes_transcribing:
            for x in range(j, j+length_gene):
                active_gene_atoms = active_gene_atoms + [x]
        active_gene_positions = atomPositions[[x-1 for x in active_gene_atoms],:]
        ds = cdist(rbp_positions, active_gene_positions, 'euclidean')
        IPs = ds<transcription_rnp_radius
        I = [x for x in range(len(IPs)) if IPs[x].any()]
        #The code changes atom types to convert particles between RBP and RNP states. Here: when RBP particles are recruited to active genes they are converted to type 5:
        for tbc in I:
            to_be_converted=rbp_atoms[tbc]
            rnp_atoms_list.append(to_be_converted)
            rbp_atoms_list.remove(to_be_converted)
            Group.L.set('atom', to_be_converted, 'type', 5)
        rbp_atoms=np.array(rbp_atoms_list)
        rbp_positions=atomPositions[[x-1 for x in rbp_atoms],:] 
        rnp_atoms = np.array(rnp_atoms_list)   
    
    if (i+1)>=t_transcription_on and i%t_rnprbp==0:
        rnp_atoms_copy = list(rnp_atoms).copy()
        rbp_atoms_copy = list(rbp_atoms).copy()
        if loc_rnprbp==1: # randomly convert RNPs from all over the nucleus into RBPs
            rnp_to_switch = random.sample(list(rnp_atoms), math.ceil(p_rnprbp*len(rnp_atoms)))
            for j in rnp_to_switch:
                rnp_atoms_copy.remove(j)
                rbp_atoms_copy.append(j)
                Group.L.set('atom', j, 'type', 4)  #convert RNP → RBP: actively change particle identity to model assembly/disassembly
                #maintaining lists of which atom indices are RBPs or RNPs to use them to compute positions and do conversions:
            rnp_atoms = np.array(rnp_atoms_copy)   
            rbp_atoms = np.array(rbp_atoms_copy) 
        # RBP.respawn()
        # RNP.respawn()
    
    # ---------------------------------------------
    # Alternative: RBP -> RNP gradual increase
    # ---------------------------------------------
    
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
    rbp_positions = atomPositions[[x-1 for x in rbp_atoms],:]
    
    rnp_atoms_list = list(rnp_atoms)
    rbp_atoms_list = list(rbp_atoms)
    
    if (i+1)>=t_induction_on:
        f_rnp = (i+1)*slope_rnp + c_rnp
        expected_rnp = int(f_rnp*len(rbprnp_atoms))
        actual_rnp = len(rnp_atoms_list)
        if expected_rnp>actual_rnp:
            rbp_to_convert = random.sample(rbp_atoms_list, expected_rnp - actual_rnp)
        else:
            rbp_to_convert = []
        
        for rbp_atom in rbp_to_convert:
            rnp_atoms_list.append(rbp_atom)
            rbp_atoms_list.remove(rbp_atom)
            Group.L.set('atom', rbp_atom, 'type', 5)
        rbp_atoms = np.array(rbp_atoms_list)
        rnp_atoms = np.array(rnp_atoms_list)   
        RBP.respawn()
        RBP.respawn()
        
    # ----------------------------------------------------------------------------------------------------------------

    for j in range(len(gene_start_atoms)):
        gene = gene_start_atoms[j]
        if gene in induced_gene_lociStart:
            induced_runs[j].append(i)
        if gene in active_gene_lociStart:
            active_runs[j].append(i)
            
    if i in induced_runs[0]:
        x=1
    elif i in active_runs[0]:
        x=2
    else:
        x=0
    myGeneFile.write(str((i+1)*delt)+","+str(PromoterPositions[0,0]*sig_chromatin)+","+str(PromoterPositions[0,1]*sig_chromatin)+","+str(PromoterPositions[0,2]*sig_chromatin)+","+str(d_rp[-1][0]*sig_chromatin)+","+str(d_rg[-1][0]*sig_chromatin)+","+str(ser5p_around_promoter[-1][0])+","+str(ser5p_around_gene[-1][0])+","+str(x)+"\n")   #write to the geneTrack.txt file

    # CLUSTER AROUND REGULATORY REGION
    ser5p_around_cluster.append([])
    ser5p_positions = atomPositions[[x-1 for x in ser5p_atoms],:]
    reg_positions = atomPositions[[x-1 for x in regulatory_atoms],:]
    ds = cdist(ser5p_positions, reg_positions, 'euclidean')
    s5p_nums = []
    for cr in cluster_r:
        IPs = ds<cr
        I = [x for x in range(len(IPs)) if IPs[x].any()]
        s5p_nums.append(len(I))
    
    diff_list = []
    for x, y in zip(s5p_nums[0::], s5p_nums[1::]):
        diff_list.append(y-x)
    
    ser5p_around_cluster[i] = diff_list        
    n_rnp.append(len(rnp_atoms))   #number of RNP atoms each timestep

if make_snapshots:
    ppmFiles = glob.glob(out_folder+'/run'+str(run_number)+'/image_files/*.ppm')
    for i_file in range(len(ppmFiles)):
        image = Image.open(ppmFiles[i_file])
        image.save(ppmFiles[i_file].split(".")[0]+'.png', quality=100)
        os.remove(ppmFiles[i_file])
    
n_rnp = np.array(n_rnp)

print("Genes have been active for "+str(round(100*total_active_steps/(NRuns-t_activation_on),2))+" percent of time after gene activation has been turned on.")
 
# if total_active_steps==0:
#     os.rmdir(out_folder+'/run'+str(run_number)+'/image_files')



    
for i in range(len(ser5p_around_cluster)):
    ser5pAroundClusterFile.write(str((i+1)*delt)+",")
    for j in range(len(cluster_r[1:])):
        if j<(len(cluster_r[1:])-1):
            ser5pAroundClusterFile.write(str(ser5p_around_cluster[i][j])+",")
        else:
            ser5pAroundClusterFile.write(str(ser5p_around_cluster[i][j])+"\n")

ser5pAroundClusterFile.close()

# Make event log from active_runs
# 0: induced but non-engaging
# 1: before activation; window-size w1
# 2: activation
# 3: post-activation; window-size w2
w1 = 20
w2 = 20

event_log = [0] * len(d_rp)
for i in range(len(event_log)-1):
    if i in active_runs[0]: # one gene
        event_log[i+1] = 2

for i in range(len(event_log)):
    if (i-w2)>=0:
        if event_log[i-w2]==2:
            if event_log[i]!=2:
                event_log[i]=3 # after-activation
    if (i+w1)<=(len(event_log)-1):
        if event_log[i+w1]==2:
            if event_log[i]!=2:
                event_log[i]=1 # before-activation
           
myFile = open(out_folder+"/parallel_counter/run"+str(run_number)+".txt", "w")
myFile.write("Run "+str(run_number)+" done!\n")
myFile.close()
os.remove(out_folder+"/parallel_counter/progress_run"+str(run_number)+".txt")
os.remove(out_folder+'/run'+str(run_number)+'/debug_RSQ.lammps')
    
myFile = open(out_folder+"/active_duration.txt", "a")
myFile.write(f"{run_number},{total_active_steps},{NRuns-t_activation_on}\n")
myFile.close()
    
myFile = open(out_folder+"/gene_stats.txt", "a")
for j in range(len(d_rg)):
    if j<t_induction_on:
        continue
    d = d_rg[j][0]
    in_contact = 0
    if d<pol2release_radius:
        in_contact = 1
    s5p_around = ser5p_around_gene[j][0]
    s2p_around = 0
    if event_log[j]==2:
        s2p_around = 1
    myFile.write(str(in_contact)+","+str(d*sig_chromatin)+","+str(s5p_around)+","+str(s2p_around)+"\n")
myFile.close()
    
for j in range(len(gene_start_atoms)): 
    dist_reg_gene = [d_rg[x][j]*sig_chromatin for x in range(len(d_rg))]
    d_dist_reg_gene = []
    for k in range(len(dist_reg_gene)-1):
        dd = dist_reg_gene[k+1] - dist_reg_gene[k]
        d_dist_reg_gene.append(dd)
       
event_types = ['induced', 'approaching', 'active', 'receding']
for var_i in range(2):
    if var_i==0:
        array_oi = dist_reg_gene.copy()
        var_name = "dist"
    elif var_i==1:
        array_oi = d_dist_reg_gene.copy()
        var_name = "ddist"
    for j in range(len(event_types)):
        event_array = np.array([array_oi[i] for i in range(len(array_oi)) if event_log[i]==j])
        myFile = open(out_folder+"/"+var_name+"_"+event_types[j]+".txt", "a")
        for x in event_array:
            myFile.write(str(x)+"\n")
        myFile.close()
        
# SAVE NUMBER OF RNPs TO AN OUTPUT FILE
#myFile = open(out_folder + "/run"+str(run_number)+"/rnp_counts.txt", "w")
#for tcount in n_rnp:
#    myFile.write(f"{tcount}\n")
#myFile.close()
              
if make_plots:
        
    fig, ax = plt.subplots()
    sns.heatmap(np.array(ser5p_around_cluster), ax=ax, xticklabels=np.round(cluster_r[1:],2))
    fig.savefig(out_folder+'/run'+str(run_number)+"/figures/ser5p_cluster.pdf", bbox_inches="tight")
    
    gray_color = np.linspace(0.5, 1, len(gene_start_atoms))
    
    fig, ax = plt.subplots()
    sns.heatmap(np.array(ser5p_xprofilematrix), ax=ax, xticklabels=range(-box+1,box))
    ax.plot([(x/box + 1)*box for x in reg_position], range(len(reg_position)), 'w')
    ax.plot([(x/box + 1)*box for x in reg_end1_position], range(len(reg_position)), 'k')
    ax.plot([(x/box + 1)*box for x in reg_end2_position], range(len(reg_position)), 'k')
    for j in range(len(gene_start_atoms)):
        gc = gray_color[j]
        ax.plot([(x[j]/box + 1)*box for x in gene_position], range(len(gene_position)), color=(gc, gc, gc), linewidth=1)
        # ax.plot([(x/15 + 1)*15 for x in [gene_position[y][j] for y in induced_runs[j]]], induced_runs[j], 'ro', markersize=1)
        ax.plot([(x/box + 1)*box for x in [gene_position[y][j] for y in active_runs[j]]], active_runs[j], 'go', markersize=2)
    fig.savefig(out_folder+'/run'+str(run_number)+"/figures/ser5p.pdf", bbox_inches="tight")
    
    for j in range(len(gene_start_atoms)):
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot([delt*x/600 for x in list(range(len(d_rp)))], [d_rp[x][j]*sig_chromatin for x in range(len(d_rp))], color="k", linewidth=0.5)
        if len(active_runs[j])>0:
            x_plot = [delt*active_runs[j][0]/600]
            y_plot = [d_rp[active_runs[j][0]][j]*sig_chromatin]
            for i in range(1, len(active_runs[j])):
                if (active_runs[j][i]-active_runs[j][i-1])==1:
                    x_plot.append(delt*active_runs[j][i]/600)
                    y_plot.append(d_rp[active_runs[j][i]][j]*sig_chromatin)
                else:
                    ax[0].plot(x_plot, y_plot, '-', color=(0.5,0.5,0.5), linewidth=1)
                    x_plot = [delt*active_runs[j][i]/600]
                    y_plot = [d_rp[active_runs[j][i]][j]*sig_chromatin]
            ax[0].plot(x_plot, y_plot, '-', color=(0.5,0.5,0.5), linewidth=1)
            
        # ax[0].plot([delt*x/600 for x in active_runs[j]], [d_rp[x][j]*sig_chromatin for x in active_runs[j]], '-', color=(0.5,0.5,0.5), linewidth=1)
        ax[0].axhline(200, linestyle='-.', color=(0.5,0.5,0.5), linewidth=0.5)
        if condition!="Control":
            ax[0].axvline(delt*NRuns/600, linestyle='-', color="#d95f02", linewidth=2)
        ax[1].plot([delt*x/600 for x in list(range(len(d_rp)))], [ser5p_around_promoter[x][j] for x in range(len(d_rp))], color="k", linewidth=0.5)
        
        if len(active_runs[j])>0:
            x_plot = [delt*active_runs[j][0]/600]
            y_plot = [ser5p_around_promoter[active_runs[j][0]][j]]
            for i in range(1, len(active_runs[j])):
                if (active_runs[j][i]-active_runs[j][i-1])==1:
                    x_plot.append(delt*active_runs[j][i]/600)
                    y_plot.append(ser5p_around_promoter[active_runs[j][i]][j])
                else:
                    ax[1].plot(x_plot, y_plot, '-', color=(0.5,0.5,0.5), linewidth=1)
                    x_plot = [delt*active_runs[j][i]/600]
                    y_plot = [ser5p_around_promoter[active_runs[j][i]][j]]
            ax[1].plot(x_plot, y_plot, '-', color=(0.5,0.5,0.5), linewidth=1)
        # ax[1].plot([delt*x/600 for x in active_runs[j]], [ser5p_around_promoter[x][j] for x in active_runs[j]], '-', color=(0.5,0.5,0.5), linewidth=1)
        ax[1].axhline(ser5p_to_activate, linestyle='-.', color=(0.5,0.5,0.5), linewidth=0.5)
        if condition!="Control":
            ax[1].axvline(delt*NRuns/600, linestyle='-', color="#d95f02", linewidth=2)
        ax[1].set_xlabel('Time (min)')
        ax[0].set_ylabel('Reg-Promoter distance (nm)')
        ax[1].set_ylabel('No. of Ser5P around promoter')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_rp_time.pdf", bbox_inches="tight")
        
# %%
