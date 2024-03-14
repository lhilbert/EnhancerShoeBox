# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Parameters of the output files can be modified in the last section

import random
import numpy as np
import os

def draw_next_location(box_size, current_location, allowed_range):
    if current_location.size==0:
        return np.random.uniform(low=-box_size, high=box_size, size=3)
    while True:
        x = np.random.uniform(low=current_location[0]-allowed_range[1], high=current_location[0]+allowed_range[1], size=1)[0]
        y = np.random.uniform(low=current_location[1]-allowed_range[1], high=current_location[1]+allowed_range[1], size=1)[0]
        z = np.random.uniform(low=current_location[2]-allowed_range[1], high=current_location[2]+allowed_range[1], size=1)[0]
        next_location = np.array([x, y, z])
        dist = np.sqrt(np.sum((next_location - current_location)**2))
        if (dist>=allowed_range[0]) and (dist<=allowed_range[1]):
            return next_location

def draw_filament(filament_length, box_size, allowed_range):
    monomer_coordinates = np.zeros((filament_length,3))
    origin_point = np.array([0,0,0])
    current_location = np.empty((0,3))
    for i in range(filament_length):
        next_location = draw_next_location(box_size,origin_point, allowed_range)
        monomer_coordinates[i,:]=next_location
        current_location = next_location
    return monomer_coordinates
 
def put_stuff_on(polymer_length, element_type, number_elements, length_element):
    sub_polymer_length = polymer_length - number_elements*length_element
    type_vec = [1]*sub_polymer_length
    min_gap = 1 # elements cannot be placed within min_gap of each other
    while True:
        if number_elements>1:
            element_positions = random.sample(range(sub_polymer_length), number_elements) # pick spots without replacement
        else:
            element_positions = [int(sub_polymer_length/2)]
        element_positions = sorted(element_positions,reverse=True)
        gaps=[] 
        for i in range(1, len(element_positions)): 
            gaps.append(element_positions[i-1]-element_positions[i]) 
        
        if all(g>min_gap for g in gaps):
            break
    for i in range(number_elements):
        type_vec = type_vec[:(element_positions[i]+1)]+[element_type]*length_element+type_vec[(element_positions[i]+1):]
    return type_vec
 
def make_input_file_general(filament_length=3, number_of_filaments=1, box_size=11, is_actin_simulation=True, promoter_length=3):
    
    # Atom types:
    # -----------

    # 1: native chromatin
    # 2: inactive gene body
    # 3: actin
    # 4: RBP - RNA Binding Protein
    # 5: RNP - RNA Protein Complex
    # 6: induced gene body
    # 7: active gene body
    # 8: Pol II Ser5P
    # 9: regulatory chromatin (super-enhancer)
    # 10: promoter of the gene
    # 11: "active" promoter (if you want special interactions for the promoter of an active gene)
    
    # Two "chromosomes" or rather polymers, first one fixed to the left edge, and the second one fixed to the right edge
    number_of_chromosomes = 2 
    
    # Good numbers of chromatin monomers, actin, Ser5P and RBP for different box sizes
    # You can play around with these but keep these numbers on hand
    if box_size==9:
        polymer_length = 300
        number_of_actin_monomers = 240
        number_of_Ser5P = 300
        number_of_RBP = 450
    elif box_size==10:
        polymer_length = 336
        number_of_actin_monomers = 270
        number_of_Ser5P = 336
        number_of_RBP = 500
    elif box_size==11:
        polymer_length = 370
        number_of_actin_monomers = 300
        number_of_Ser5P = 370
        number_of_RBP = 550
    elif box_size==12:
        polymer_length = 400
        number_of_actin_monomers = 300
        number_of_Ser5P = 400
        number_of_RBP = 600
    elif box_size==13:
        polymer_length = 433
        number_of_actin_monomers = 325
        number_of_Ser5P = 433
        number_of_RBP = 650
    elif box_size==15:
        polymer_length = 500
        number_of_actin_monomers = 400
        number_of_Ser5P = 500
        number_of_RBP = 750
    elif box_size==18:
        polymer_length = 600
        number_of_actin_monomers = 500
        number_of_Ser5P = 600
        number_of_RBP = 900
    elif box_size==21:
        polymer_length = 700
        number_of_actin_monomers = 560
        number_of_Ser5P = 700
        number_of_RBP = 1050
    elif box_size==24:
        polymer_length = 800
        number_of_actin_monomers = 650
        number_of_Ser5P = 800
        number_of_RBP = 1200
            
    if not is_actin_simulation:
        number_of_actin_monomers = 0
        number_of_filaments = 0
        
    # 1 Regulatory Element (super-enhancer) of length 50 monomer units on the left polymer
    number_RR = 1
    length_RR = 50
    
    # 1 gene of length 5 monomer units on the left polymer
    number_genes = 1
    length_gene = 5
    
    # Left polymer of of length 1/3 rd of the total no. of chromatin monomers in the system
    region_RR = int(1*polymer_length/3)
    
    # 2/3 rds is the right polymer containing the gene
    region_genes = polymer_length - region_RR - number_genes*promoter_length # promoter_length*number_genes
    
    gene_type_vec = put_stuff_on(region_genes, 2, number_genes, length_gene)
    reg_type_vec = put_stuff_on(region_RR, 9, number_RR, length_RR)
    dna_type_vec = []
    dna_type_vec.append(reg_type_vec)
    dna_type_vec.append(gene_type_vec)

    # Adding a promoter to each gene
    for i in range(len(dna_type_vec)):
        chromosome_type = dna_type_vec[i].copy()
        prev_type = 0
        added_promoters = 0
        for j in range(len(chromosome_type)):
            curr_type = chromosome_type[j]
            if curr_type==2 and prev_type!=2:
                for pr_i in range(promoter_length):
                    dna_type_vec[i].insert(j+added_promoters, 10)
                    added_promoters = added_promoters + 1
            prev_type = curr_type 
    
    # Initially we don't have any RNP as in the current model, we ramp up the RNP gradually over the simulation run
    f_rnp = 0
    if is_actin_simulation:
        target_file = f'input_files/IC_ser5P{number_of_Ser5P}_actin{number_of_actin_monomers}_RBP{number_of_RBP}_{f_rnp}RNP_promoter{promoter_length}_bs{box_size}.data'
    else:
        target_file = f'input_files/IC_ser5P{number_of_Ser5P}_RBP{number_of_RBP}_{f_rnp}RNP_promoter{promoter_length}_bs{box_size}.data'
        
    number_of_RNP = int(f_rnp*number_of_RBP)
    number_of_RBP = number_of_RBP - number_of_RNP
    
    n_atoms_total = polymer_length + number_of_filaments*filament_length + number_of_actin_monomers + number_of_RBP + number_of_RNP + number_of_Ser5P
    n_bonds = number_of_filaments*filament_length - number_of_filaments + polymer_length - number_of_chromosomes
    n_angles = number_of_filaments*filament_length - 2*number_of_filaments + polymer_length- 2*number_of_chromosomes
    
    n_atom_types = 11
    
    # Chromatin and actin can form bonds and angles
    n_bond_types = 2
    n_angle_types = 2
    
    # All atoms have same mass
    mass_of_types = [1]*n_atom_types
    actin_size = 0.3
    actin_allowed_range = [actin_size-0.1, actin_size+0.1]

    box_x = box_size
    box_y = 7.5
    box_z = 5
    w = 2.0
    
    f=open(target_file,'w+')
    f.write('LAMMPS input file for a SHOEBOX simulation to test induced gene movement along the Ser5P-actin gradient.\n')
    f.write('\n')
    f.write(str(n_atoms_total)+' atoms\n')
    f.write(str(n_bonds)+' bonds\n')
    f.write(str(n_angles)+' angles\n')
    f.write('\n')
    f.write(str(n_atom_types)+' atom types\n')
    f.write(str(n_bond_types)+' bond types\n')
    f.write(str(n_angle_types)+' angle types\n')
    f.write('\n')
    f.write(str(-box_x-w)+' '+str(box_x+w)+' xlo xhi\n')
    f.write(str(-box_y-w)+' '+str(box_y+w)+' ylo yhi\n')
    f.write(str(-box_z-w)+' '+str(box_z+w)+' zlo zhi\n')
    f.write('\n')
    f.write('Masses\n')
    f.write('\n')
    for i in range(n_atom_types):
        f.write(str(i+1)+' '+str(mass_of_types[i])+'\n')
    f.write('\n')
    f.write('Atoms\n')
    f.write('\n')
    
    # Chromatin
    atomID = 1
    
    # Regulatory chromosome
    moleculeID = 1
    reg_start = -box_y/2
    reg_end = box_y/2
    regulatory_y = np.linspace(reg_start, reg_end, region_RR)
    
    for j in range(region_RR):
        f.write(str(atomID)+' '+str(moleculeID)+' '+str(dna_type_vec[moleculeID-1][j])+' '+str(-box_x)+' '+str(regulatory_y[j])+' '+str(0)+' 0 0 0\n')
        atomID += 1
    
    moleculeID += 1
    
    # Genes chromosome
    i=1
    genes_start = -box_y/2
    genes_end = box_y/2
    genes_y = np.linspace(genes_start, genes_end, polymer_length - region_RR)
    for j in range(polymer_length - region_RR):
        if dna_type_vec[i][j] not in [2, 10]:
            f.write(str(atomID)+' '+str(moleculeID)+' '+str(dna_type_vec[moleculeID-1][j])+' '+str(box_x)+' '+str(genes_y[j])+' '+str(0)+' 0 0 0\n')
        else:
            f.write(str(atomID)+' '+str(moleculeID)+' '+str(dna_type_vec[moleculeID-1][j])+' '+str(0)+' '+str(genes_y[j])+' '+str(0)+' 0 0 0\n')
        atomID += 1
            
    moleculeID += 1
        
    # Filamentous actin
    for i in range(number_of_filaments):
        filament_coordinates = draw_filament(filament_length, box_size, actin_allowed_range)
        for j in range(filament_coordinates.shape[0]):
            f.write(str(atomID)+' '+str(moleculeID)+' 3 '+str(round(filament_coordinates[j,0],6))+' '+str(round(filament_coordinates[j,1],6))+' '+str(round(filament_coordinates[j,2],6))+' 0 0 0\n')
            atomID=atomID+1
        moleculeID += 1
    
    # Free G-actin
    for i in range(number_of_actin_monomers):
        a = np.random.uniform(low=[-box_x, -box_y, -box_z], high=[0, box_y, box_z], size=3)
        f.write(str(atomID)+' '+str(moleculeID)+' 3 '+str(round(a[0],6))+' '+str(round(a[1],6))+' '+str(round(a[2],6))+' 0 0 0\n')
        moleculeID += 1
        atomID += 1
        
    # RBP
    for i in range(number_of_RBP):
        a = np.random.uniform(low=[-box_x, -box_y, -box_z], high=[box_x, box_y, box_z], size=3)
        f.write(str(atomID)+' '+str(moleculeID)+' 4 '+str(round(a[0],6))+' '+str(round(a[1],6))+' '+str(round(a[2],6))+' 0 0 0\n')
        moleculeID += 1
        atomID += 1
        
    # RNP
    for i in range(number_of_RNP):
        a = np.random.uniform(low=[-box_x, -box_y, -box_z], high=[0, box_y, box_z], size=3)
        f.write(str(atomID)+' '+str(moleculeID)+' 5 '+str(round(a[0],6))+' '+str(round(a[1],6))+' '+str(round(a[2],6))+' 0 0 0\n')
        moleculeID += 1
        atomID += 1
        
    # Pol II Ser5P
    for i in range(number_of_Ser5P):
        a = np.random.uniform(low=[-box_x, -box_y, -box_z], high=[0, box_y, box_z], size=3)
        f.write(str(atomID)+' '+str(moleculeID)+' 8 '+str(round(a[0],6))+' '+str(round(a[1],6))+' '+str(round(a[2],6))+' 0 0 0\n')
        moleculeID += 1
        atomID += 1
        
    f.write('\n')
    
    # Velocities, initialized to 0
    f.write('Velocities\n')
    f.write('\n')
    for i in range(n_atoms_total):
        f.write(str(i+1)+' 0 0 0\n')
    f.write('\n')
    
    # Bonds
    f.write('Bonds\n')
    f.write('\n')
    bondID, atomID = 1, 1
    
    # Chromatin bonds
    for j in range(polymer_length-1):
        if j==(region_RR-1):
            atomID += 1
            continue
        f.write(str(bondID)+' 1 '+str(atomID)+' '+str(atomID+1)+'\n')
        bondID += 1
        atomID += 1
    atomID += 1
    
    # Filamentous actin bonds
    for i in range(number_of_filaments):
        for j in range(filament_length-1):
            f.write(str(bondID)+' 2 '+str(atomID)+' '+str(atomID+1)+'\n')
            bondID += 1
            atomID += 1
        atomID += 1
    f.write('\n')
    
    # Angles
    f.write('Angles\n')
    f.write('\n')
    angleID, atomID = 1, 1
    
    # Chromatin angles
    for j in range(polymer_length-2):
        if j==(region_RR-1) or j==(region_RR-2):
            atomID += 1
            continue
        f.write(str(angleID)+' 1 '+str(atomID)+' '+str(atomID+1)+' '+str(atomID+2)+'\n')
        angleID += 1
        atomID += 1
    atomID += 2
    
    # Filamentous actin angles
    for i in range(number_of_filaments):
        for j in range(filament_length-2):
            f.write(str(angleID)+' 2 '+str(atomID)+' '+str(atomID+1)+' '+str(atomID+2)+'\n')
            angleID += 1
            atomID += 1
        atomID += 2   
    return target_file

os.makedirs('input_files', exist_ok=True)

# List of promoter lengths to make the input files for: 
# Default visitor gene: 3, non-visitor: 1. 
# Usual range = 1, 2, 3
promoter_lengths = [1,2,3]

# Box sizes to consider
# 11, 12 are good box sizes
box_sizes = [10,11,12]

# Is this an actin simulation?
is_actin_simulation = True

# If you want to make input files for other sets of parameters, you can add those here and also make more for loops below
for promoter_length in promoter_lengths:
    for box_size in box_sizes:
        if is_actin_simulation:
            number_of_filaments = 300
        else:
            number_of_filaments = 0
        make_input_file_general(filament_length=3, number_of_filaments=number_of_filaments, box_size=box_size, is_actin_simulation=is_actin_simulation, promoter_length=promoter_length)
        
