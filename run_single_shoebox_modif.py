# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Necessary module imports for running the simulation and subsequent data processing.
from lammps import lammps, IPyLammps
import math
import random
import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.spatial.distance import cdist
import sys, getopt
from PIL import Image
import seaborn as sns
import glob

# Set the OMP_NUM_THREADS environment variable
#os.environ["OMP_NUM_THREADS"] = "4"  # Adjust the number of threads as needed

# Atom types:
# -----------

# 1: native chromatin
# 2: inactive gene body
# 3: actin
# 4: RBP
# 5: RNP
# 6: induced gene body
# 7: active gene body
# 8: Pol II Ser5P
# 9: regulatory chromatin (super-enhancer)
# 10: promoter of the gene
# 11: "active" promoter (if you want special interactions for the promoter of an active gene)

# average chromatin density: 0.015 bp/nm^3
# chromEMT: 15-30 % volume occupied by chromatin

# Some notes from earlier about actin polymerization:
# ---------------------------------------------------

# 1-5 um long actin filaments. 1-2 um persistence length?

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
    """
    A class to define atom groups based on atom types.
    Used for defining computes, dumps, and redefining groups after atom type resetting.
    """
    
    # Class variables
    L = None  # Main instance of IPyLammps
    lmp = None # Main instance of lammps
    name = None # Group name
    all_groups = [] # List of all defined groups
    atom_types = [] # List of atom types included in the group
    computes = None # Computes defined on the group
    dumps = None # Dumps defined on the group
    
    def __init__(self, name, atom_types=None, atom_ids=None):
        """Constructor for Group class."""
        self.name = name
        self.computes =  [] # Initialize empty computes list
        self.dumps =  [] # Initialize empty dumps list
        Group.all_groups.append(self)
        
        # If atom types are not specified, initialize based on atom IDs
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
        """Add new atom types to the group."""
        self.atom_types.extend(atom_types)
        atom_types_string=" ".join(str(item) for item in atom_types)
        Group.L.group(self.name+' type '+atom_types_string)
        
    def add_compute(self, compute_name, right_side_command):
        """Define a new compute for the group."""
        total_command=" ".join([compute_name,self.name,right_side_command])
        self.computes.append({'compute_name':compute_name, 'command':total_command})
        Group.L.compute(total_command)
        
    def add_dump(self, dump_name, right_side_command, dump_modify_command=None):
        """Define a new dump for the group, with optional modification."""
        total_command=" ".join([dump_name,self.name,right_side_command])
        Group.L.dump(total_command)
        if dump_modify_command is not None:
            total_modify=" ".join([dump_name,dump_modify_command])
            Group.L.dump_modify(total_modify)
        else:
            total_modify=None
        self.dumps.append({'dump_name':dump_name, 'command':total_command, 'modify':total_modify})
            
    def count_atoms(self):
        """Count the number of atoms in the group."""
        Group.L.variable('tempCount equal count('+self.name+')')
        a=Group.L.variables['tempCount'].value
        Group.L.variable('tempCount delete')
        return a
    
    def respawn(self):
        """Recreate the group, its computes, and dumps after a change."""
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
        """Clear all computes and dumps associated with the group."""
        for compute_name in [sub['compute_name'] for sub in self.computes]: Group.L.uncompute(compute_name)
        for dump_name in [sub['dump_name'] for sub in self.dumps]: Group.L.undump(dump_name)
        Group.L.group(self.name,'clear')

# Function to generate synthetic microscopy images based on atom positions and types.
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
    
# Function to display a progress bar in the terminal.
def progress_bar(current, total, name, bar_length = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
    spaces  = ' ' * (bar_length - len(arrow))

    print(name+': [%s%s] %d %%' % (arrow, spaces, percent), end='\r')
    
# Function to display multiple progress bars, useful for parallel runs.
def progress_bars(data, total, bar_length=20):
    
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    for i in range(len(data)):
        percent = float(data[i,1]) * 100 / total
        arrow   = '-' * int(percent/100 * bar_length - 1) + '>'
        spaces  = ' ' * (bar_length - len(arrow))

        print('Run '+str(int(data[i,0]))+': [%s%s] %d %%' % (arrow, spaces, percent), end='\n')

# Parse arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "hb:r:t:o:m:c:x:p:a:x:z:n:q:", ["box=","repeat=","total=","outfolder=","make_images=","condition=","promoter=","activation=","threshold=","actin=","nucleation=","threads="])
except getopt.GetoptError:
    print('run_single_shoebox.py -b <box_size> -r <repeat> -t <total_runs> -o <out_folder> -m <make_images> -c <condition> -p <promoter_length> -a <activation_rate> -x <threshold> -z <actin> -n <p_nucleation> -q <num_threads>')
    sys.exit(2)

# Loop over arguments and assign parameter values
for opt, arg in opts:
    # Help
    if opt == '-h':
        print('run_single_shoebox.py -b <box_size> -r <repeat> -t <total_runs> -o <out_folder> -m <make_images> -c <condition> -p <promoter_length> -a <activation_rate> -x <threshold> -z <actin> -n <p_nucleation> -q <num_threads>')
        sys.exit()
        
    # Length of the shoe-box in monomer units
    elif opt in ("-b", "--box"):
        box = int(arg)
        
    # Repeat/run number (number of the current run)
    elif opt in ("-r", "--repeat"):
        run_number = int(arg)
        
    # Total number of runs in the batch (used for displaying output in the terminal)
    elif opt in ("-t", "--total"):
        total_runs = int(arg)
        
    # Folder to store the outputs of each run in
    elif opt in ("-o", "--outfolder"):
        out_folder = arg
        
    # Do you want to make and store snapshot files?
    elif opt in ("-m", "--make_images"):
        make_snapshots = int(arg)
        
    # Which condition does the simulation correspond to? 
    # Control, LatB etc. and transcription inhibitors. Complete list to be placed in user's manual
    # Later in the code, we'll change potentials and other things depending on the condition
    elif opt in ("-c", "--condition"):
        condition = arg
        
    # Length of the promoter in monomer units
    # This actually models both promoter + proximal enhancer 
    elif opt in ("-p", "--promoter"):
        promoter_length = int(arg)
        
    # Activation rate
    elif opt in ("-a", "--activation"):
        p_gene_activation = float(arg)/1000
        
    # Pol II Ser5P threshold for gene activation (thershold number of PolII molecules to initiate activation)
    elif opt in ("-x", "--threshold"):
        ser5p_to_activate = int(arg)
        
    # No. of actin "monomers" initially free (non-filamentous)
    elif opt in ("-z", "--actin"):
        init_free_actin = int(arg)
        
    # Prob. of nucleating a new actin filament from two adjacent monomers
    # In every Python timestep, and for every free actin atom, this sets the probability of nucleating a new actin filament
    # 0.1, 0.01. 0.001, 0.0001, 0.00001
    # Higher the value (lower no. of decimal points for the reference values above), lower the actin length
    # They are referenced as "pnuc{x}d" in the output folder names, where x is the number of decimal points
    # 0.1: pnuc1d, 0.01: pnuc2d, 0.001: pnuc3d, 0.0001: pnuc4d, 0.00001: pnuc5d
    # So higher the x, longer the resulting actin filaments on average (less nucleation of new filaments)
    # Python time step - user defined with reference to the fastest reaction rate
    elif opt in ("-n", "--nucleation"):
        p_nucleation = float(arg)
        
    # Set the OMP_NUM_THREADS environment variable
    # Adjust the number of threads as needed    
    elif opt in ("-q", "--threads"):
    	number_threads = int(arg)
    	os.environ["OMP_NUM_THREADS"] = str(arg) 
    	print(f"OMP_NUM_THREADS = {arg}")
     
if init_free_actin==0:
    is_actin_simulation = False
else:
    is_actin_simulation = True
    
# Create various sub-directories you might need for each run
#len() returns length of an object, eg. number of objects on a list
folders_to_make = ["figures", "image_files", "microscopy_files"]
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
    
    
# Create a new Group object and initialize a LAMMPS model inside it
# Group.L assigns LAMMPS instance to L variable
Group.lmp = lammps()
Group.L = IPyLammps(ptr=Group.lmp)
Group.L.log(out_folder+'/run'+str(run_number)+'/debug_RSQ.lammps')

# Set the diameter of a chromatin monomer in nm, depends on the value of chromatin volume fraction
sig_chromatin = 60 # in nm (for chromatin volume fraction 0.06)
  
# Good numbers of chromatin monomers, actin, Ser5P and RBP for different box sizes
# You can play around with these but keep these numbers on hand, refer README.md for these values
# init_free_actin is missing here as compared to make_input_file.py
# We pass init_free_actin as a parameter to run_sigle_shoebox.py
#RBP-RNA binding proteins, they are regulating gene expression
if box==9:
    Ntot = 300
    number_of_Ser5P = 300
    number_of_RBP = 450    
elif box==10:
    Ntot = 336
    number_of_Ser5P = 336
    number_of_RBP = 500
elif box==11:
    Ntot = 370
    number_of_Ser5P = 370
    number_of_RBP = 550
elif box==12:
    Ntot = 400
    number_of_Ser5P = 400
    number_of_RBP = 600
elif box==13:
    Ntot = 433
    number_of_Ser5P = 433
    number_of_RBP = 650
elif box==15:
    Ntot = 500
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

# Timescales of various processes
NRuns = 2000 # Total number (for duration of Control / before treatment) of LAMMPS runs (Python timesteps) in this Python script
tRun = 200 # No. of LAMMPS timesteps in each LAMMPS run
dt = 0.005 # Length of each LAMMPS timestep. This should correspond to 0.003 s
delt = 0.03*tRun # Duration of a Python timestep = 0.6 s
# This would mean that the whole simulation is 2000*0.6 seconds = 1200 seconds = 20 min 

# Define treatment duration depending on the condition
treatment_duration = 0
if condition=="Hexanediol":
    treatment_duration = int(3*60/0.6)
elif condition=="JQ-1":
    treatment_duration = int(30*60/0.6)
elif condition=="Flavopiridol":
    treatment_duration = int(30*60/0.6)
    
# If you want to make snapshots for a specific set of runs, you can overwrite `make_snapshots` according to some condition     
if run_number <= 100:
    make_snapshots = 1

# No. of LAMMPS timesteps for which snapshots should be made
tImageDump = 10*tRun

# Gene length in monomer units   
length_gene = 5

# If actin is present, apart from non-filamentous actin, we start with an initial filament of length 3.
# The input configuration file should also contain this filament.
# This is just to make sure the code works. Remove here and from make_input_file.py if an alternative solution is found.
filament_length = 3
filaments = 1

# Do you want to make plots at the end of the run? 
# Ex.: Gene track, Ser5P distribution along the length of the box
#1 - yes, 0 - no
make_plots = 1

# Do you want to make and store synthetic microscopy images?
# 1 - yes, 0 - no
make_microscopy = 1

# Do you want a more verbose output of the simulation's events?
# 1 - yes, 0 - no
print_verbose = 1

# Move actin atoms immediately performing nucleation, polymerization and depolymerization?
# This ensures that the potentials and forces don't blow up
move_added_atoms = 1
move_removed_atoms = 1

# Radii of balls (in monomer units) within which interactions are checked/counted for various processes
filEndEffectRadius = 3 # For actin polymerization
monomerEffectRadius = 2 # For actin nucleation
pol2release_radius = 3 # For Ser5P around promoter (gene activation)
transcription_rnp_radius = 2.5 # For RBP -> RNP conversion around active genes (if implemented)

# Different radii to calculate radial spatial distribution
cluster_r = np.linspace(0.65, 2.5, 50)

# Relative diameter of various components (compared to chromatin monomer = 1)
sig_actin = 0.3
w_sa = 0.2
sig_chr = 1.0 # chromatin monomer in chromatin monomer units
sig_rbp = 0.3
sig_rnp = 1.0
sig_ser5p = 0.3

# Effective raw cutoffs for initial potentials (soft phase)
c_cc = 2**(1/6)
c_crb = ((sig_chr+sig_rbp)/2)*(2**(1/6))
c_crn = ((sig_chr+sig_rnp)/2)*(2**(1/6))
c_cs = ((sig_chr+sig_ser5p)/2)*(2**(1/6))
c_rbrn = ((sig_rbp+sig_rnp)/2)*(2**(1/6))
c_rbs = ((sig_rbp+sig_ser5p)/2)*(2**(1/6))
c_rns = ((sig_rnp+sig_ser5p)/2)*(2**(1/6))
 
# Change initial setup depending on the given condition (Control or some treament)
if condition=="noactin":
    init_free_actin = 0
    
# Filename of input configuration file. Please ensure this is consistent with make_input_file.py 
if is_actin_simulation:
    fileBase_input = f'ser5P{number_of_Ser5P}_actin{init_free_actin}_RBP{number_of_RBP}_0RNP_promoter{promoter_length}_bs{box}'
else:
    fileBase_input = f'ser5P{number_of_Ser5P}_RBP{number_of_RBP}_0RNP_promoter{promoter_length}_bs{box}'
filePath = f'input_files/IC_{fileBase_input}.data' 

# Length of the soft phase in Python timesteps
t_soft = 10 

# inhibition of actin polymerization
if condition=="LatB" or condition=="SwinA_nopol":
    t_polymerization_on = 2*NRuns
else:
    # When to switch on actin polymerization
    t_polymerization_on = 40 
    
# Change this to an earlier timepoint (from 2*NRuns) if you want to stop actin polymerization in the middle of the simulation
t_polymerization_off = 2*NRuns

# When to switch on Pol II Ser5P interactions
t_ser5p_on = 15 

# When to switch on gene induction (allowing for initial equilibration of the system)
t_induction_on = 150 

# When to switch on gene activation (after gene induction, and allowing for initial equilibration of the system)
t_activation_on = t_induction_on 
    
# We don't use active genes to convert RBP to RNP, hence this is set to 2*NRuns
t_transcription_on = 2*NRuns

# Change these to earlier timepoints (from 2*NRuns) if you want to stop any of these processes earlier
t_induction_off = 2*NRuns
t_activation_off = 2*NRuns
t_transcription_off = 2*NRuns

# Actin rates as obtained from literature for reference here. 
# The rates in our simulation were previously set to roughly match these. 
# However, we now consider a different simulation -> realtime mapping so these don't match anymore.
# ATP+ k_on = 12/uM/s, k_off = 1.4/s, Cc = 0.12 uM
# ADP+ k_on = 4/uM/s, k_off = 8/s, Cc = 2 uM
# ATP- k_on = 1.3/uM/s, k_off = 0.8/s, Cc = 0.6 uM
# ADP- k_on = 0.3/um/s, k_off = 0.16/s, Cc = 2 uM 

# Actin polymerization / depolymerization rates (inverse of below parameters)
t_on_plus = 1
t_on_minus = 40 # (k_on_plus/k_on_minus)
t_off_plus = 2*NRuns # (k_on_plus/k_off_plus)
t_off_minus = 5 # (k_on_plus/k_off_minus)

# Only these fraction of actin polymerization cases go through to success (it reduces the effective rate further)
pon = 0.3

# Parameters concerning RBP <-> RNP conversions (we don't use them anymore in the shoebox simulations)
t_rbprnp = 1 
p_rbprnp = 1/8
t_rnprbp = 10
p_rnprbp = 1/30
loc_rnprbp = 1 # Where should RNP -> RBP occur? 1: everywhere randomly, 2: near the nuclear periphery
# Near the nuclear periphery / edge of the box not implemented yet

# How often (Python timesteps) should gene induction, activation and inactivation be attempted? 
# Now, these are every Python timestep
t_gene_induction = 1
t_gene_activation = 1 
t_gene_inactivation = 1

# After marking a gene for activation, delay in Python timesteps before actually activating
activation_delay = 2

# No. of Python timesteps for which an active gene stays active before becoming induced
t_active_duration = 50 

# RNP ramp up parameters
f_rnp_final = 0.9
t_rnp_final = NRuns+treatment_duration 
slope_rnp = f_rnp_final/(t_rnp_final-t_induction_on)
c_rnp = -slope_rnp*t_induction_on

# Setting up the simulation
Group.L.atom_style('angle')
Group.L.boundary('f', 'f', 'f')
Group.L.read_data(filePath, 'extra/bond/per/atom 100 extra/angle/per/atom 100 extra/special/per/atom 100')
Group.L.neighbor(0.3, 'multi')
Group.L.neigh_modify('every', 1, 'delay', 10, 'check', 'yes', 'one', 10000, 'page', 100000)
Group.L.comm_modify('cutoff 6')

if is_actin_simulation:
    filamentous_actin = list(range(Ntot+1, Ntot+filament_length*filaments+1))
    free_actin = list(range(Ntot+filament_length*filaments+1, Ntot+filament_length*filaments+1+init_free_actin))
    
    filament_details = []
    n_start = Ntot+1
    for i in range(filaments):
        filament_details.append(list(range(n_start, n_start+filament_length)))
        n_start = n_start + filament_length

    filament_ends = []
    filament_ends_plus = []
    filament_ends_minus = []
    for i in range(filaments):
        filament_ends.append(filament_details[i][0])
        filament_ends_plus.append(filament_details[i][0])
        filament_ends.append(filament_details[i][-1])
        filament_ends_minus.append(filament_details[i][-1])
    
atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)), ((Group.L.system.natoms,1)))

# Define various Groups
NativeChromatin = Group('NativeChromatin',[1])
InactiveGenes = Group('InactiveGenes',[2])
InactiveChromatin = Group('InactiveChromatin',[1,2]) 
InducedGenes = Group('InducedGenes', atom_types=[6])
ActiveGenes = Group('ActiveGenes', atom_types=[7])
RegulatoryChromatin = Group('RegulatoryChromatin', atom_types=[9])
Promoters = Group('Promoters', atom_types=[10])
ActivePromoters = Group('ActivePromoters', atom_types=[11])
if is_actin_simulation:
    Actin = Group('Actin', atom_types=[3])
RBP = Group('RBP', atom_types=[4])
RNP = Group('RNP', atom_types=[5])
RBPRNP = Group('RBPRNP', atom_types=[4,5])
Ser5P = Group('Ser5P', atom_types=[8])
if is_actin_simulation:
    if len(filamentous_actin)>0:
        FilamentEnds = Group('FilamentEnds', atom_ids=filament_ends)
        FilamentEndsPlus = Group('FilamentEndsPlus', atom_ids=filament_ends_plus)
        FilamentEndsMinus = Group('FilamentEndsMinus', atom_ids=filament_ends_minus)
        FilamentousActin = Group('FilamentousActin', atom_ids=filamentous_actin)
    if len(free_actin)>0:
        FreeActin = Group('FreeActin', atom_ids=free_actin)
ImageGroup = Group('ImageGroup',[1,2,3,5,6,7,8,9,10,11])

Group.L.angle_style('cosine')
Group.L.angle_coeff(1, 100/sig_chromatin) # Chromatin 
Group.L.angle_coeff(2, 33.0) # Actin filament
# K*sigma is the persistence length

# Soft-phase pair interaction between non-bonded atoms
# NOTE: These will be overwritten by the custom soft LJ potentials after the soft phase is finished
Group.L.pair_style('soft', 2**(1/6))
Group.L.pair_coeff('*', '*', 0, 2**(1/6)) # default for chromatin hardcore-repulsion

# Chromatin and inactive genes
Group.L.pair_coeff('1*2', '4', 0, c_crb) # RBP
Group.L.pair_coeff('1*2', '5', 0, 1.5*c_crn) # RNP
Group.L.pair_coeff('1*2', '6', 0, c_cc) # induced 
Group.L.pair_coeff('1*2', '7', 0, 1.5*c_cc) # active genes
Group.L.pair_coeff('1*2', '8', 0, 1.0*c_cs) # Ser5P 2.0

# RBP
Group.L.pair_coeff('4', '4', 0, sig_rbp*c_cc)
Group.L.pair_coeff('4', '5', 0, c_rbrn)
Group.L.pair_coeff('4', '6*7', 0, c_crb)
Group.L.pair_coeff('4', '8', 0, c_rbs)
Group.L.pair_coeff('4', '9*11', 0, c_crb)

# RNP
Group.L.pair_coeff('5', '5', 0, sig_rnp*c_cc)
Group.L.pair_coeff('5', '6', 0, 1.5*c_crn) # RNP-induced gene repulsion
Group.L.pair_coeff('5', '7', 0, c_crn)
Group.L.pair_coeff('5', '8', 0, c_rns)
Group.L.pair_coeff('5', '9*11', 0, 1.5*c_crn) # RNP-regulatory element repulsion

# Induced genes
Group.L.pair_coeff('6', '7', 0, 1.5*c_cc) # active
Group.L.pair_coeff('6', '8', 0, c_cs) # ser5p
Group.L.pair_coeff('6', '9*11', 0, c_cc) # regulatory regions

# Active genes
Group.L.pair_coeff('7', '7', 0, 1.5*c_cc) # active
Group.L.pair_coeff('7', '8', 0, 1.0*c_cs) # ser5p 2.0
Group.L.pair_coeff('7', '9*11', 0, 1.5*c_cc) # regulatory regions

# Ser5P
Group.L.pair_coeff('8', '8', 0, sig_ser5p*c_cc) # ser5p
Group.L.pair_coeff('8', '9*11', 0, c_cs) # regulatory regions

Group.L.variable('prefactor equal ramp(0,60)')
Group.L.fix('s1 all adapt 1 pair soft a * * v_prefactor')

# Pair interaction between bonded atoms
Group.L.bond_style('harmonic')
Group.L.bond_coeff(1, '90.0 1') 
Group.L.bond_coeff(2, '90.0 0.3')

Group.L.variable('chromatinr0 equal ramp(0.033,1.0)')
Group.L.fix('s2 all adapt 10 bond harmonic r0 1 v_chromatinr0')

Group.L.variable('actinr0 equal ramp(0.033,'+str(sig_actin)+')')
Group.L.fix('s3 all adapt 10 bond harmonic r0 2 v_actinr0')

# Important fixes
Group.L.variable('seed equal', random.randint(1, 100000)) 
Group.L.fix('1 all nve')
Group.L.fix('2 all langevin', 1, 1, 0.5, random.randint(1, 100000))

# Defining interactions with edge of box to keep everything inside
Group.L.fix('wallhi all wall/harmonic xlo EDGE 100 0.0 3.0 xhi EDGE 100 0.0 3.0 ylo EDGE 100 0.0 3.0 yhi EDGE 100 0.0 3.0 zlo EDGE 100 0.0 3.0 zhi EDGE 100 0.0 3.0')

# Anchors for different polymer lengths, refer README.md for these values
# These will be fixed to the left and right edges of the box
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
    
# Setting force on anchors as zero
Group.L.fix('freezeAnchors Anchors setforce 0.0 0.0 0.0')

# Defining a dump for snapshots
# Description of the output images needed: red - inactive gene body, blue - promoter, dark grey - active gene etc.
if make_snapshots:
    image_dump_string = 'imageDump pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 color reblue 0.13 0.33 0.62 color s5p 0.69 0.33 0.33 color indgene 0.47 0.03 0.03 acolor 1 lightgrey acolor 2 lightgrey acolor 3 chartreuse acolor 4 yellow acolor 5 aqua acolor 6 indgene acolor 7 verydarkgrey acolor 8 s5p acolor 9 reblue acolor 10 reblue acolor 11 reblue'
    Group.L.dump('imageDump ImageGroup image', tImageDump, out_folder+'/run'+str(run_number)+'/image_files/BEFORE_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
    Group.L.dump_modify(image_dump_string)
    
# Setting the LAMMPS timestep
Group.L.timestep(dt)
    
# Defining starting loci of inactive, induced and active genes
inactive_gene_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==2])
inactive_gene_lociStart = []
induced_gene_lociStart = []
active_gene_lociStart = []
for i in range(len(inactive_gene_atoms)):
    if atomTypes[inactive_gene_atoms[i]-2]==2:
        continue
    a = inactive_gene_atoms[i]
    inactive_gene_lociStart.append(a)
    
# Noting down atom lists by various types
gene_start_atoms = inactive_gene_lociStart.copy()
if is_actin_simulation:
    all_actin = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==3])  
chromatin_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [1,2,6,7,9,10,11]])
rbp_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==4])
rnp_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==5])
rbprnp_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [4,5]])
ser5p_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==8])
regulatory_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [9]])
all_promoter_atoms = np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j] in [10]])
promoter_atoms = all_promoter_atoms[list(range(0, len(all_promoter_atoms), promoter_length))] 

if is_actin_simulation:
    n_filaments = [len(filament_details)]
    mean_len_filaments = [np.mean(np.array([len(x) for x in filament_details]))]
    sd_len_filaments = [np.std(np.array([len(x) for x in filament_details]))]
    n_free_actin = [init_free_actin]
    min_len_filaments = [np.min(np.array([len(x) for x in filament_details]))]
    max_len_filaments = [np.max(np.array([len(x) for x in filament_details]))]
    actin_xprofilematrix = []
    actin_around_cluster = []
    
n_rnp = [len(rnp_atoms)]
ser5p_xprofilematrix = []
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
d_rg = []
d_rp = []
ser5p_around_promoter = []
ser5p_around_gene = []
total_active_steps = 0
gene_active_duration = [0] * len(inactive_gene_lociStart)
marked_for_activation=[]
delay_activation=[]

if is_actin_simulation:
    actinFile = open(out_folder+"/run"+str(run_number)+"/actinTrack.txt", "w")
    actinAroundClusterFile = open(out_folder+"/run"+str(run_number)+"/actinAroundCluster.txt", "w")
myGeneFile = open(out_folder+"/run"+str(run_number)+"/geneTrack.txt", "w")
ser5pAroundClusterFile = open(out_folder+"/run"+str(run_number)+"/ser5pAroundCluster.txt", "w")

ser5pAroundClusterFile.write("t,")
for j in range(len(cluster_r[1:])):
    if j<(len(cluster_r[1:])-1):
        ser5pAroundClusterFile.write(str(cluster_r[j+1])+",")
    else:
        ser5pAroundClusterFile.write(str(cluster_r[j+1])+"\n")
        
if is_actin_simulation:
    actinAroundClusterFile.write("t,")
    for j in range(len(cluster_r[1:])):
        if j<(len(cluster_r[1:])-1):
            actinAroundClusterFile.write(str(cluster_r[j+1])+",")
        else:
            actinAroundClusterFile.write(str(cluster_r[j+1])+"\n")
                
# Python timesteps for loop
for i in range(NRuns+treatment_duration):
    
    myFile = open(out_folder+"/parallel_counter/progress_run"+str(run_number)+".txt", "w")
    myFile.write(str(run_number)+','+str(i+1))
    myFile.close()

    data = []
    for file in os.listdir(out_folder+"/parallel_counter"):
        if file.startswith("progress_run"):
            myFile = open(out_folder+"/parallel_counter/"+file, 'r')
            for line in myFile:
                array = line.strip().split(',')
                data.append([float(array[0]), float(array[1])])
            myFile.close()
    data = np.array(data)
        
    # Clear terminal, print details and progress of all the parallely running simulations
    subprocess.run(["clear"])
    print("Condition: "+condition+", Promoter: "+str(promoter_length)+", Total runs: "+str(total_runs)+", Number of OMP threads: "+str(number_threads)+"\n")
    progress_bars(data, NRuns)
    
    if print_verbose:
        print('-----------------------------------------------------------------------')
        print('Run '+str(i))
        
    # We write snapshots as ppm files and then convert to png files
    ppmFiles = glob.glob(out_folder+'/run'+str(run_number)+'/image_files/*.ppm')
    for i_file in range(len(ppmFiles)):
        image = Image.open(ppmFiles[i_file])
        image.save(ppmFiles[i_file].split(".")[0]+'.png', quality=100)
        os.remove(ppmFiles[i_file])
    
    # If soft phase is over, switch to custom soft LJ potentials from the file
    # You can define more interactions / modify these in the setup_ljsoft_potentials.py file and use them here
    if (i+1)==t_soft:
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
        
        # Actin
        if condition=="SwinA_old" or condition=="SwinA_nopol":
            Group.L.pair_coeff('3','3','LJsoft_RSQ.table ACTIN_ACTIN_STRONGATTRACTION', 0.9) # 0.84
        else:
            Group.L.pair_coeff('3','3','LJsoft_RSQ.table ACTIN_ACTIN', 0.4) # 0.34
        Group.L.pair_coeff('3','4','LJsoft_RSQ.table ACTIN_RBP', 0.4) # 0.34
        Group.L.pair_coeff('3','5','LJsoft_RSQ.table RNP_ACTIN', 1.9) # 0.72 c_arn
        Group.L.pair_coeff('3','6','LJsoft_RSQ.table ACTIN_INDUCED', 1.2) # 1.09
        Group.L.pair_coeff('3','7','LJsoft_RSQ.table ACTIN_ACTIVE', 1.2) # 1.09
        Group.L.pair_coeff('3','9','LJsoft_RSQ.table ACTIN_REGULATORY', 0.75) # 0.73
        Group.L.pair_coeff('3','10*11','LJsoft_RSQ.table ACTIN_PROMOTER', 0.75) # 0.73
        
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
        
    # When it's time to switch on Ser5P interactions, do so
    if (i+1)==t_ser5p_on:
        # Ser5P
        if print_verbose:
            print("Switching on Ser5P interactions.")
        Group.L.pair_coeff('8','8','LJsoft_RSQ.table SER5P_SER5P', 0.9) # 0.84
        if condition=="noActinSer5P":
            Group.L.pair_coeff('3','8','LJsoft_RSQ.table ACTIN_SER5P_NOATTRACTION', 0.4) # 0.34
        else:
            # Group.L.pair_coeff('3','8','LJsoft_RSQ.table ACTIN_SER5P', 0.9) # was previously just repulsion
            # Group.L.pair_coeff('3','8','LJsoft_RSQ.table ACTIN_SER5P_LONGRANGE', 0.9) # 0.84
            Group.L.pair_coeff('3','8','LJsoft_RSQ.table ACTIN_SER5P', 0.9) # 0.84
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
            image_dump_string = 'imageDump pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 color reblue 0.13 0.33 0.62 color s5p 0.69 0.33 0.33 color indgene 0.47 0.03 0.03 acolor 1 lightgrey acolor 2 lightgrey acolor 3 chartreuse acolor 4 yellow acolor 5 aqua acolor 6 indgene acolor 7 verydarkgrey acolor 8 s5p acolor 9 reblue acolor 10 reblue acolor 11 reblue'
            Group.L.dump('imageDump ImageGroup image',tConditionImageDump, out_folder+'/run'+str(run_number)+'/image_files/AFTER_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
            Group.L.dump_modify(image_dump_string)
            
    # Soft-phase run
    if (i+1)<t_soft:
        Group.L.run(tRun, 'start 0 stop', t_soft*tRun)
    # Main run (Python timestep)
    else:
        Group.L.run(tRun)
        
    if (i+1)<t_polymerization_on and is_actin_simulation:
        n_filaments.append(len(filament_details))
        mean_len_filaments.append(np.mean(np.array([len(x) for x in filament_details])))
        sd_len_filaments.append(np.std(np.array([len(x) for x in filament_details])))
        n_free_actin.append(len(free_actin))
        min_len_filaments.append(np.min(np.array([len(x) for x in filament_details])))
        max_len_filaments.append(np.max(np.array([len(x) for x in filament_details])))
    
        if print_verbose:
            print('Switching polymerization on.')
            
    # Update atom positions and atom types for each Python timestep
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)), ((Group.L.system.natoms,3)))
    atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)), ((Group.L.system.natoms,1)))
    
    ser5p_xpositions = atomPositions[[x-1 for x in ser5p_atoms],0]
    ser5p_xp = np.histogram(ser5p_xpositions, bins=2*box, range=(-box, box))[0]
    ser5p_xprofilematrix.append(list(ser5p_xp))
    
    if is_actin_simulation:
        actin_xpositions = atomPositions[[x-1 for x in all_actin],0]
        actin_xp = np.histogram(actin_xpositions, bins=2*box, range=(-box, box))[0]
        actin_xprofilematrix.append(list(actin_xp))
    
    regulatory_xpositions = atomPositions[[x-1 for x in regulatory_atoms],0]
    regulatory_xp = np.mean(regulatory_xpositions)
    reg_position.append(regulatory_xp)
    if len(regulatory_xpositions)>0:
        reg_end1_position.append(np.min(regulatory_xpositions))
        reg_end2_position.append(np.max(regulatory_xpositions))
    
    RRPosition = np.mean(atomPositions[[x-1 for x in regulatory_atoms]], axis=0)
    GenesPosition = atomPositions[[x-1 for x in gene_start_atoms]]
    d_rg.append([np.sqrt(np.sum((RRPosition - GenesPosition[k,:]) ** 2)) for k in range(len(GenesPosition))])
        
    gene_xpositions = atomPositions[[x-1 for x in promoter_atoms], 0] # should we track genes instead?
    gene_xp = np.mean(gene_xpositions)
    gene_position.append(list(gene_xpositions))
    
    ser5p_positions = atomPositions[[x-1 for x in ser5p_atoms],:]
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
        if is_actin_simulation:
            filamentous_actin = list(set(all_actin) & set(filamentous_actin))
    
    # --------------------------------------------
    # ACTIN dynamics:
    # --------------------------------------------
        
    if (i+1)>=t_polymerization_on and (i+1)<t_polymerization_off and is_actin_simulation:
        # --------------------------------------------
        # ACTIN NUCLEATION (monomer-monomer bonding):
        # --------------------------------------------
        
        free_actin_positions=atomPositions[[x-1 for x in free_actin],:]
        filament_end_positions=atomPositions[[x-1 for x in FilamentEnds.atom_ids],:]
        
        # Find monomer pairs to form new actin filaments
        pairs_to_be_formed=[]
        for j in range(len(free_actin)):
            # Monomer-monomer
            if random.random()<=p_nucleation and i%t_on_plus==0:
                curr_atom=free_actin[j]
                other_free_actin=list.copy(free_actin)
                other_free_actin.remove(curr_atom)
                pds=[np.sqrt(np.sum((atomPositions[curr_atom-1] - free_actin_positions[k,:]) ** 2)) for k in range(len(free_actin))]
                pds.remove(np.min(np.array(pds)))
                pds=np.array(pds)
                if len(pds)>0 and np.min(pds)<monomerEffectRadius:
                    to_be_added=other_free_actin[np.argmin(pds)]
                    if any(curr_atom in sl for sl in pairs_to_be_formed) or any(to_be_added in sl for sl in pairs_to_be_formed):
                        continue
                    pairs_to_be_formed.append([curr_atom, to_be_added])
                    
                    # Move atoms if necessary
                    if move_added_atoms:
                        n_tries=1
                        while 1:
                            n_tries=n_tries+1
                            new_loc = np.random.uniform(low=atomPositions[curr_atom-1]-(sig_actin+w_sa), high=atomPositions[curr_atom-1]+(sig_actin+w_sa), size=(1,3))
                            d = np.sqrt(np.sum((atomPositions[curr_atom-1] - new_loc) ** 2))
                            if d>(sig_actin-w_sa) and d<(sig_actin+w_sa):
                                break
                            if n_tries>2000:
                                break
                        if n_tries>2000:
                            continue
                        if print_verbose:
                            print("Nucleation: "+str(n_tries)+" tries")
                        Group.L.set('atom', to_be_added, 'x', new_loc[0,0], 'y', new_loc[0,1], 'z', new_loc[0,2])
                        atomPositions[to_be_added-1]=new_loc[0]
        if print_verbose:
            print("\n")
        
        for j in range(len(pairs_to_be_formed)):
            curr_atom=pairs_to_be_formed[j][0]
            to_be_added=pairs_to_be_formed[j][1]
            Group.L.create_bonds('single/bond 2', curr_atom, to_be_added)
            filamentous_actin.append(curr_atom)
            filamentous_actin.append(to_be_added)
            free_actin.remove(curr_atom)
            free_actin.remove(to_be_added)
            filament_details.append([curr_atom, to_be_added])
            # Current atom becomes plus end, and to_be_added becomes minus end
        
        if i%t_on_plus==0:
            filament_ends=[]
            filament_ends_plus=[]
            filament_ends_minus=[]
            for j in range(len(filament_details)):
                filament_ends.append(filament_details[j][0])
                filament_ends_plus.append(filament_details[j][0])
                filament_ends.append(filament_details[j][-1])
                filament_ends_minus.append(filament_details[j][-1])
            
            FilamentEnds.atom_ids = filament_ends
            FilamentEndsPlus.atom_ids = filament_ends_plus
            FilamentEndsMinus.atom_ids = filament_ends_minus
            if len(filament_details)>0:
                FilamentEnds.respawn()
                FilamentEndsPlus.respawn()
                FilamentEndsMinus.respawn()
            else:
                FilamentEnds.clear()
                FilamentEndsPlus.clear()
                FilamentEndsMinus.clear()
                
            atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
            atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)),((Group.L.system.natoms,1)))
            
            all_actin=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==3])
            filamentous_actin = list(set(all_actin) & set(filamentous_actin))

        # ---------------------------------------
        # ACTIN POLYMERIZATION
        # ---------------------------------------
        
        free_actin_positions=atomPositions[[x-1 for x in free_actin],:]
        filament_end_positions=atomPositions[[x-1 for x in FilamentEnds.atom_ids],:]
        filament_end_plus_positions=atomPositions[[x-1 for x in FilamentEndsPlus.atom_ids],:]
        filament_end_minus_positions=atomPositions[[x-1 for x in FilamentEndsMinus.atom_ids],:]
        
        for j in range(len(filament_details)):
            if i%t_on_plus==0 and random.random()<=pon:
                # Plus end (has to be faster)
                free_actin_copy=free_actin.copy()
                curr_atom=filament_details[j][0]
                adj_atom=filament_details[j][1]
                pds=np.array([np.sqrt(np.sum((atomPositions[curr_atom-1] - free_actin_positions[k,:]) ** 2)) for k in range(free_actin_positions.shape[0])])
                if len(pds)>0:
                    I = list(np.where(pds<filEndEffectRadius))[0]
                    for tba in I:
                        to_be_added=free_actin_copy[tba]
                        if move_added_atoms:
                            n_tries=0
                            while 1:
                                n_tries=n_tries+1
                                new_loc = np.random.uniform(low=atomPositions[curr_atom-1]-(sig_actin+w_sa), high=atomPositions[curr_atom-1]+(sig_actin+w_sa), size=(1,3))
                                d = np.sqrt(np.sum((atomPositions[curr_atom-1] - new_loc) ** 2))
                                da = np.sqrt(np.sum((atomPositions[adj_atom-1] - new_loc) ** 2))
                                if d>(sig_actin-w_sa) and d<(sig_actin+w_sa) and da>((2**(1/2))*sig_actin):       
                                    break
                                if n_tries>2000:
                                    break
                            if n_tries>2000:
                                continue
                            if print_verbose:
                                print("Polymerization + end: "+str(n_tries)+" tries")
                            Group.L.set('atom',to_be_added,'x',new_loc[0,0],'y',new_loc[0,1],'z',new_loc[0,2])
                            atomPositions[to_be_added-1]=new_loc[0]
                        Group.L.create_bonds('single/bond 2', curr_atom, to_be_added)
                        Group.L.create_bonds('single/angle 2', to_be_added, curr_atom, filament_details[j][1])
                        filamentous_actin.append(to_be_added)
                        free_actin.remove(to_be_added)
                        filament_details[j].insert(0,to_be_added)
                        free_actin_positions=atomPositions[[x-1 for x in free_actin],:] 
                        curr_atom=filament_details[j][0]
                        adj_atom=filament_details[j][1]

            if i%(t_on_plus*t_on_minus)==0 and random.random()<=pon:
                # Minus end (has to be slower)
                free_actin_copy=free_actin.copy()
                curr_atom=filament_details[j][-1]
                adj_atom=filament_details[j][-2]
                pds=np.array([np.sqrt(np.sum((atomPositions[curr_atom-1] - free_actin_positions[k,:]) ** 2)) for k in range(free_actin_positions.shape[0])])
                if len(pds)>0:
                    I = list(np.where(pds<filEndEffectRadius))[0]
                    for tba in I:
                        to_be_added=free_actin_copy[tba]
                        if move_added_atoms:
                            n_tries=0
                            while 1:
                                n_tries=n_tries+1
                                new_loc = np.random.uniform(low=atomPositions[curr_atom-1]-(sig_actin+w_sa), high=atomPositions[curr_atom-1]+(sig_actin+w_sa), size=(1,3))
                                d = np.sqrt(np.sum((atomPositions[curr_atom-1] - new_loc) ** 2))
                                da = np.sqrt(np.sum((atomPositions[adj_atom-1] - new_loc) ** 2))
                                if d>(sig_actin-w_sa) and d<(sig_actin+w_sa) and da>((2**(1/2))*sig_actin):
                                    break
                                if n_tries>2000:
                                    break
                            if n_tries>2000:
                                continue
                            if print_verbose:
                                print("Polymerization - end: "+str(n_tries)+" tries")
                            Group.L.set('atom',to_be_added,'x',new_loc[0,0],'y',new_loc[0,1],'z',new_loc[0,2])
                            atomPositions[to_be_added-1]=new_loc[0]
                        Group.L.create_bonds('single/bond 2', curr_atom, to_be_added)
                        Group.L.create_bonds('single/angle 2', filament_details[j][-2], curr_atom, to_be_added)
                        filamentous_actin.append(to_be_added)
                        free_actin.remove(to_be_added)
                        filament_details[j].append(to_be_added)   
                        free_actin_positions=atomPositions[[x-1 for x in free_actin],:] 
                        curr_atom=filament_details[j][-1]
                        adj_atom=filament_details[j][-2]
        if print_verbose:
            print("\n")
            
        if i%t_on_plus==0:
            filament_ends=[]
            filament_ends_plus=[]
            filament_ends_minus=[]
            for j in range(len(filament_details)):
                filament_ends.append(filament_details[j][0])
                filament_ends_plus.append(filament_details[j][0])
                filament_ends.append(filament_details[j][-1])
                filament_ends_minus.append(filament_details[j][-1])
            
            FilamentEnds.atom_ids = filament_ends
            FilamentEndsPlus.atom_ids = filament_ends_plus
            FilamentEndsMinus.atom_ids = filament_ends_minus
            if len(filament_details)>0:
                FilamentEnds.respawn()
                FilamentEndsPlus.respawn()
                FilamentEndsMinus.respawn()
            else:
                FilamentEnds.clear()
                FilamentEndsPlus.clear()
                FilamentEndsMinus.clear()
                
            atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
            atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)),((Group.L.system.natoms,1)))
            
            all_actin=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==3])
            filamentous_actin = list(set(all_actin) & set(filamentous_actin))
            
        # ------------------------------------------
        # ACTIN DEPOLYMERIZATION
        # ------------------------------------------
        
        if i%t_off_minus==0:
            Group.L.delete_bonds('FilamentEndsMinus atom', 3, 'any remove special')
            Group.L.delete_bonds('FilamentEndsMinus angle', 2, 'any remove special')
            for j in range(len(filament_details)):
                # progress_bar(j, len(filament_details)-1, "Actin depolymerization")
                # Minus end
                curr_atom=filament_details[j][-1]
                next_atom=filament_details[j][-2]
                filamentous_actin.remove(curr_atom)
                free_actin.append(curr_atom)
                filament_details[j].remove(curr_atom)
                if move_removed_atoms:
                    n_tries=0
                    while 1:
                        n_tries=n_tries+1
                        new_loc = np.random.uniform(low=atomPositions[curr_atom-1]-1, high=atomPositions[curr_atom-1]+1, size=(1,3))
                        d = np.sqrt(np.sum((atomPositions[next_atom-1] - new_loc) ** 2))
                        if d>sig_actin:
                            break
                        if n_tries>2000:
                            break
                    if n_tries>2000:
                        continue    
                    if print_verbose:
                        print("Depolymerization - end: "+str(n_tries)+" tries")
                    Group.L.set('atom',curr_atom,'x',new_loc[0,0],'y',new_loc[0,1],'z',new_loc[0,2])
                    atomPositions[curr_atom-1]=new_loc[0]
                if len(filament_details[j])<2:
                    filamentous_actin.remove(filament_details[j][0])
                    free_actin.append(filament_details[j][0])
                    continue
            if print_verbose:
                print("\n")
                
        # Remove filaments of length < 2 as they are not filaments anymore!
        to_remove=[]
        for j in range(len(filament_details)):
            if len(filament_details[j])<2:
                to_remove.append(j)
        for j in sorted(to_remove, reverse=True):
            del filament_details[j]
                
        filament_ends=[]
        filament_ends_plus=[]
        filament_ends_minus=[]
        for j in range(len(filament_details)):
            filament_ends.append(filament_details[j][0])
            filament_ends_plus.append(filament_details[j][0])
            filament_ends.append(filament_details[j][-1])
            filament_ends_minus.append(filament_details[j][-1])
            
        if i%t_off_plus==0:
            Group.L.delete_bonds('FilamentEndsPlus atom', 3, 'any remove special')
            Group.L.delete_bonds('FilamentEndsPlus angle', 2, 'any remove special')
            for j in range(len(filament_details)):            
                # Plus end
                curr_atom=filament_details[j][0]
                next_atom=filament_details[j][1]
                filamentous_actin.remove(curr_atom)
                free_actin.append(curr_atom)
                filament_details[j].remove(curr_atom)   
                if move_removed_atoms:
                    n_tries=0
                    while 1:
                        n_tries=n_tries+1
                        new_loc = np.random.uniform(low=atomPositions[curr_atom-1]-1, high=atomPositions[curr_atom-1]+1, size=(1,3))
                        d = np.sqrt(np.sum((atomPositions[next_atom-1] - new_loc) ** 2))
                        if d>sig_actin:
                            break
                        if n_tries>2000:
                            break
                    if n_tries>2000:
                        continue   
                    if print_verbose:
                        print("Depolymerization + end: "+str(n_tries)+" tries")
                    Group.L.set('atom',curr_atom,'x',new_loc[0,0],'y',new_loc[0,1],'z',new_loc[0,2])
                    atomPositions[curr_atom-1]=new_loc[0]         
                if len(filament_details[j])<2:
                    filamentous_actin.remove(filament_details[j][-1])
                    free_actin.append(filament_details[j][-1])
                    continue
                
        # Remove filaments of length < 2 as they are not filaments anymore!
        to_remove=[]
        for j in range(len(filament_details)):
            if len(filament_details[j])<2:
                to_remove.append(j)
        for j in sorted(to_remove, reverse=True):
            del filament_details[j]
                
        filament_ends=[]
        filament_ends_plus=[]
        filament_ends_minus=[]
        for j in range(len(filament_details)):
            filament_ends.append(filament_details[j][0])
            filament_ends_plus.append(filament_details[j][0])
            filament_ends.append(filament_details[j][-1])
            filament_ends_minus.append(filament_details[j][-1])
        
        FilamentEnds.atom_ids = filament_ends
        FilamentEndsPlus.atom_ids = filament_ends_plus
        FilamentEndsMinus.atom_ids = filament_ends_minus
        if len(filament_details)>0:
            FilamentEnds.respawn()
            FilamentEndsPlus.respawn()
            FilamentEndsMinus.respawn()
        else:
            FilamentEnds.clear()
            FilamentEndsPlus.clear()
            FilamentEndsMinus.clear()
        
        atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
        atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)),((Group.L.system.natoms,1)))
        
        all_actin=np.array([j+1 for j in range(Group.L.system.natoms) if atomTypes[j]==3])
        filamentous_actin = list(set(all_actin) & set(filamentous_actin))

        if len(free_actin)>0:
            FreeActin.atom_ids = free_actin
            FreeActin.respawn()
        if len(filamentous_actin)>0:
            FilamentousActin.atom_ids = filamentous_actin            
            FilamentousActin.respawn()
                
        n_filaments.append(len(filament_details))
        mean_len_filaments.append(np.mean(np.array([len(x) for x in filament_details])))
        sd_len_filaments.append(np.std(np.array([len(x) for x in filament_details])))
        n_free_actin.append(len(free_actin))
        min_len_filaments.append(np.min(np.array([len(x) for x in filament_details])))
        max_len_filaments.append(np.max(np.array([len(x) for x in filament_details])))

    # ---------------------------------------
    # GENE INDUCTION
    # ---------------------------------------

    if i%t_gene_induction==0 and i>=t_induction_on and i<t_induction_off:
        loci_to_induce = inactive_gene_lociStart.copy()
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
                image_dump_string = 'imageDumpAct pad 7 backcolor white adiam 1*2 0.5 adiam 3*5 0.4 adiam 6*7 0.5 adiam 8 0.5 adiam 9*11 0.5 bdiam 1 0.05 bdiam 2 0.2 color verydarkgrey 0.2 0.2 0.2 color reblue 0.13 0.33 0.62 color s5p 0.69 0.33 0.33 color indgene 0.47 0.03 0.03 acolor 1 lightgrey acolor 2 lightgrey acolor 3 chartreuse acolor 4 yellow acolor 5 aqua acolor 6 indgene acolor 7 verydarkgrey acolor 8 s5p acolor 9 reblue acolor 10 reblue acolor 11 reblue'
                if i<NRuns:
                    Group.L.dump('imageDumpAct ImageGroup image',tRun*5, out_folder+'/run'+str(run_number)+'/image_files/BEFORE_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
                else:
                    Group.L.dump('imageDumpAct ImageGroup image',tRun*5, out_folder+'/run'+str(run_number)+'/image_files/AFTER_*.ppm type type zoom',3,'size',1700,1000,'view',0,-90,'shiny 0.5 box no 1')
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
            
        InactiveChromatin.respawn()
        InactiveGenes.respawn()
        ActiveGenes.respawn()
        Promoters.respawn()
        ActivePromoters.respawn()
        
    # ------------------------------------------------
    # NOT USED FOR SHOEBOX: RBP <-> RNP conversions
    # ------------------------------------------------
    
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)),((Group.L.system.natoms,3)))
    rbp_positions = atomPositions[[x-1 for x in rbp_atoms],:]
    
    rnp_atoms_list = list(rnp_atoms)
    rbp_atoms_list = list(rbp_atoms)
    
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
                Group.L.set('atom', j, 'type', 4)
            rnp_atoms = np.array(rnp_atoms_copy)   
            rbp_atoms = np.array(rbp_atoms_copy) 
        # RBP.respawn()
        # RNP.respawn()
    
    # ---------------------------------------------
    # CURRENT MODEL: RBP -> RNP gradual increase
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
            
    # gene_state, as tracked in the gene track file next
    # 0: inactive, 1: induced, 2: active
    if i in induced_runs[0]:
        x=1
    elif i in active_runs[0]:
        x=2
    else:
        x=0
        
    # Gene track file - change columns as needed. 
    # Current columns:
    # time, promoter_position_x, promoter_position_y, promoter_position_z, d_reg_promoter, d_reg_gene, Ser5P_around_promoter, Ser5P_around_gene, gene_state 
    myGeneFile.write(str((i+1)*delt)+","+str(PromoterPositions[0,0]*sig_chromatin)+","+str(PromoterPositions[0,1]*sig_chromatin)+","+str(PromoterPositions[0,2]*sig_chromatin)+","+str(d_rp[-1][0]*sig_chromatin)+","+str(d_rg[-1][0]*sig_chromatin)+","+str(ser5p_around_promoter[-1][0])+","+str(ser5p_around_gene[-1][0])+","+str(x)+"\n")
    
    # Synthetic microscopy images
    atomPositions = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('x',1,3)), ((Group.L.system.natoms,3)))
    atomTypes = np.reshape(np.ctypeslib.as_array(Group.lmp.gather_atoms('type',0,1)), ((Group.L.system.natoms,1)))    
    if (i+1)%10==0 and make_microscopy:
        makeFakeMicroscopyImages(atomPositions, atomTypes, 1, (i+1)*tRun, out_folder, run_number)
    
    # Ser5P cluster around regulatory region
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
    
    # Actin around regulatory region
    if is_actin_simulation:
        actin_around_cluster.append([])
        actin_positions = atomPositions[[x-1 for x in all_actin],:]
        ds = cdist(actin_positions, reg_positions, 'euclidean')
        actin_nums = []
        for cr in cluster_r:
            IPs = ds<cr
            I = [x for x in range(len(IPs)) if IPs[x].any()]
            actin_nums.append(len(I))
        
        diff_list = []
        for x, y in zip(actin_nums[0::], actin_nums[1::]):
            diff_list.append(y-x)
        
        actin_around_cluster[i] = diff_list
    
        if print_verbose:
            for j in range(len(filament_details)):
                print("Filament #"+ str(j+1)+": "+str(filament_details[j]))
                
            print(str(n_filaments[-1])+" filaments, average filament length of "+str(round(mean_len_filaments[-1],2)))
            
        actinFile.write(str((i+1)*delt)+"\t"+str(n_filaments[-1])+"\t"+str(round(mean_len_filaments[-1],2))+"\t"+str(round(sd_len_filaments[-1],2))+"\t"+str(n_free_actin[-1])+"\n")
        
    n_rnp.append(len(rnp_atoms))
 
    if i==(NRuns-1) and is_actin_simulation and make_plots:
        num_bins = 20
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(sig_actin*sig_chromatin*np.array([len(x) for x in filament_details]), num_bins)
        plt.xlabel('Filament length (nm)')
        plt.ylabel('Probability')
        plt.title('$\mu=$'+str(round(sig_actin*sig_chromatin*mean_len_filaments[-1],2))+', $\sigma=$'+str(round(sig_actin*sig_chromatin*sd_len_filaments[-1],2))+', range=['+str(sig_actin*sig_chromatin*min_len_filaments[-1])+', '+str(sig_actin*sig_chromatin*max_len_filaments[-1])+']')
        fig.savefig(out_folder+"/run"+str(run_number)+"/figures/filamentLengthHistogram_"+fileBase_input+".pdf", bbox_inches="tight")

    # End of for loop for Python timesteps
    
# Simulation is done by here!

# Convert ppm to png files
if make_snapshots:
    ppmFiles = glob.glob(out_folder+'/run'+str(run_number)+'/image_files/*.ppm')
    for i_file in range(len(ppmFiles)):
        image = Image.open(ppmFiles[i_file])
        image.save(ppmFiles[i_file].split(".")[0]+'.png', quality=100)
        os.remove(ppmFiles[i_file])
    
if is_actin_simulation:
    n_filaments = np.array(n_filaments)
    mean_len_filaments = np.array(mean_len_filaments)
    sd_len_filaments = np.array(sd_len_filaments)
    n_free_actin = np.array(n_free_actin)
n_rnp = np.array(n_rnp)

print("Genes have been active for "+str(round(100*total_active_steps/(NRuns-t_activation_on),2))+" percent of time after gene activation has been turned on.")
    
for i in range(len(ser5p_around_cluster)):
    ser5pAroundClusterFile.write(str((i+1)*delt)+",")
    if is_actin_simulation:
        actinAroundClusterFile.write(str((i+1)*delt)+",")
    for j in range(len(cluster_r[1:])):
        if j<(len(cluster_r[1:])-1):
            ser5pAroundClusterFile.write(str(ser5p_around_cluster[i][j])+",")
            if is_actin_simulation:
                actinAroundClusterFile.write(str(actin_around_cluster[i][j])+",")
        else:
            ser5pAroundClusterFile.write(str(ser5p_around_cluster[i][j])+"\n")
            if is_actin_simulation:
                actinAroundClusterFile.write(str(actin_around_cluster[i][j])+"\n")

ser5pAroundClusterFile.close()
if is_actin_simulation:
    actinAroundClusterFile.close()

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
              
if make_plots:
    
    fig, ax = plt.subplots()
    plt.plot([delt*x/600 for x in list(range(len(n_rnp)))], n_rnp)
    ax.set_title('\# RNPs')
    ax.set_xlabel('Time (min)')
    fig.savefig(out_folder+'/run'+str(run_number)+"/figures/n_rnp.pdf", bbox_inches="tight")
        
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
    
    fig, ax = plt.subplots()
    A = np.log10(np.array(ser5p_xprofilematrix))
    A[A == -np.inf] = np.NaN
    sns.heatmap(A, ax=ax, xticklabels=range(-box+1,box))
    ax.plot([(x/box + 1)*box for x in reg_position], range(len(reg_position)), 'w')
    ax.plot([(x/box + 1)*box for x in reg_end1_position], range(len(reg_position)), 'k')
    ax.plot([(x/box + 1)*box for x in reg_end2_position], range(len(reg_position)), 'k')
    for j in range(len(gene_start_atoms)):
        gc = gray_color[j]
        ax.plot([(x[j]/box + 1)*box for x in gene_position], range(len(gene_position)), color=(gc, gc, gc), linewidth=1)
        # ax.plot([(x/15 + 1)*15 for x in [gene_position[y][j] for y in induced_runs[j]]], induced_runs[j], 'ro', markersize=1)
        ax.plot([(x/box + 1)*box for x in [gene_position[y][j] for y in active_runs[j]]], active_runs[j], 'go', markersize=2)
    fig.savefig(out_folder+'/run'+str(run_number)+"/figures/ser5p_log.pdf", bbox_inches="tight")
    
    for j in range(len(gene_start_atoms)):
        fig, ax = plt.subplots()
        # ax.plot([d_rg[x][j]*sig_chromatin for x in range(len(d_rg))], [ser5p_around_promoter[x][j] for x in range(len(d_rg))], color=(0.5,0.5,0.5), linewidth=0.5)
        ax.plot([d_rg[x][j]*sig_chromatin for x in active_runs[j]], [ser5p_around_gene[x][j] for x in active_runs[j]], 'go', markersize=2)
        ax.plot([d_rg[x][j]*sig_chromatin for x in induced_runs[j]], [ser5p_around_gene[x][j] for x in induced_runs[j]], 'ro', markersize=0.5)
        ax.plot([d_rg[x][j]*sig_chromatin for x in range(len(d_rg))], [ser5p_to_activate for x in range(len(d_rg))], color=(0.5,0.5,0.5), linewidth=0.5)
        ax.set_xlabel('Reg-TSS distance (nm)')
        ax.set_ylabel('No. of Ser5P around gene TSS')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_rg.pdf", bbox_inches="tight")
    
    for j in range(len(gene_start_atoms)):
        fig, ax = plt.subplots()
        # ax.plot([d_rp[x][j]*sig_chromatin for x in range(len(d_rg))], [ser5p_around_promoter[x][j] for x in range(len(d_rg))], color=(0.5,0.5,0.5), linewidth=0.5)
        ax.plot([d_rp[x][j]*sig_chromatin for x in active_runs[j]], [ser5p_around_promoter[x][j] for x in active_runs[j]], 'go', markersize=2)
        ax.plot([d_rp[x][j]*sig_chromatin for x in induced_runs[j]], [ser5p_around_promoter[x][j] for x in induced_runs[j]], 'ro', markersize=0.5)
        ax.plot([d_rp[x][j]*sig_chromatin for x in range(len(d_rp))], [ser5p_to_activate for x in range(len(d_rp))], color=(0.5,0.5,0.5), linewidth=0.5)
        ax.set_xlabel('Reg-Promoter distance (nm)')
        ax.set_ylabel('No. of Ser5P around promoter')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_rp.pdf", bbox_inches="tight")
    
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
         
    for j in range(len(gene_start_atoms)): 
        dist_reg_gene = [d_rg[x][j]*sig_chromatin for x in range(len(d_rp))]
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(np.array(dist_reg_gene), 30)
        plt.xlabel('Reg-RSS distance (nm)')
        plt.ylabel('Probability')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_drg_distr.pdf", bbox_inches="tight")
    
    for j in range(len(gene_start_atoms)): 
        dist_reg_gene = [d_rg[x][j]*sig_chromatin for x in range(len(d_rp))]
        d_dist_reg_gene = []
        for k in range(len(dist_reg_gene)-1):
            dd = dist_reg_gene[k+1] - dist_reg_gene[k]
            d_dist_reg_gene.append(dd)
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(np.array(d_dist_reg_gene), 30)
        plt.xlabel('dReg-RSS distance (nm)')
        plt.ylabel('Probability')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_ddrg_distr.pdf", bbox_inches="tight")
        
    for j in range(len(gene_start_atoms)): 
        dist_reg_gene = [d_rg[x][j]*sig_chromatin for x in range(len(d_rp))]
        d_dist_reg_gene = []
        for k in range(len(dist_reg_gene)-1):
            dd = dist_reg_gene[k+1] - dist_reg_gene[k]
            d_dist_reg_gene.append(dd)
        fig, ax = plt.subplots()
        plt.hist(np.array([d_dist_reg_gene[i] for i in range(len(d_dist_reg_gene)) if event_log[i]==2]), 30, alpha=0.5, label='active', density=True)
        plt.hist(np.array([d_dist_reg_gene[i] for i in range(len(d_dist_reg_gene)) if event_log[i]==0]), 30, alpha=0.5, label='induced', density=True)
        plt.hist(np.array([d_dist_reg_gene[i] for i in range(len(d_dist_reg_gene)) if event_log[i]==1]), 30, alpha=0.5, label='approaching', density=True)
        plt.hist(np.array([d_dist_reg_gene[i] for i in range(len(d_dist_reg_gene)) if event_log[i]==3]), 30, alpha=0.5, label='receding', density=True)
        plt.legend()
        plt.xlabel('dReg-RSS distance (nm)')
        plt.ylabel('Probability')
        fig.savefig(out_folder+'/run'+str(run_number)+"/figures/gene_"+str(j+1)+"_ddrg_events.pdf", bbox_inches="tight")

# %%
