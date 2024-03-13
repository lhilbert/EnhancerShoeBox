# EnhancerShoeBox

Molecular Dynamics Simulations of Enhancer-Promoter Interactions in a Model Shoe Box

# Description of contributions

Original code as of December 2023 developed by Dr. Roshan Prizak while working as postdoctoral researcher in the research group of Lennart Hilbert at Karlsruhe Institute of Technology. Further code development and documentation by Dr. Ewa Anna Oprzeska-Zingrebe as postdoctoral researcher in the research group of Lennart Hilbert. Supervision of simulation and code development as well as limited direct contributions from Lennart Hilbert while Professor at Karlsruhe Institute of Technology.

# Shoe-box simulations

Shoe-box simulations are implemented, like the other simulations, using LAMMPS and Python. LAMMPS is periodically called from inside a Python script and these together make up the main simulation run. In between these LAMMPS runs, Python handles "chemical reactions" - creation of bonds, breaking of bonds and change of atom types depending on specific conditions.

## Single simulation

The script `run_single_shoebox.py` contains the base code one can use to run a single simulation repeat of a shoe-box simulation.

An example to run an actin simulation is:

```
python run_single_shoebox.py -b 11 -r 1 -t 10 -o actin/ -m 1 -c Control -p 3 -a 30 -x 80 -z 300
```

An example to run a gene-cluster simulation without actin is:

```
python run_single_shoebox.py -b 11 -r 1 -t 10 -o gene_cluster/ -m 1 -c Control -p 3 -a 30 -x 80 -z 0 -n 0.01
```

The difference between the two simulations is crucially, (a) the output folder name, (b) the value passed with argument `z` (no. of free actin), and (c) the argument "n", representing the probability of nucleation. The output folder name can be anything if your choice, and `z` should be 0 for gene_cluster without actin and >0 for an actin simulation.

Apart from the arguments mentioned next, there are other parameters you can change in the script itself. It's fairly well-documented as of September 2023 but if you have any questions, contact Roshan (roshanprizak@gmail.com).

### Arguments to pass

- b: length of the shoe-box in chromatin monomer units
- r: repeat number
- t: total number of repeats in this batch of parallel runs
- o: output folder (structure) to store the individual runs in
- m: make snapshots of the simulation?
- c: name of the condition - Control, LatB etc.
- p: length of promoter in chromatin monomer units
- a: activation rate
- x: Ser5P threshold for activation
- z: no. of free actin
- n: probability of nucleating new actin filaments

The argument `z` (no. of free actin) should be 0 for a gene-cluster visit simulation without actin, and >0 for an actin simulation.

## Important parameters and other considerations

### Input configuration

The script `make_input_file.py` can generate input configuration files that are consistent with LAMMPS and have the geometry of a shoebox. 

You can use this script to generate input configuration files for a range of parameters. Set `is_actin_simulation` to `True` if generating input files for an actin simulation, and to `False` otherwise.

You can change various aspects of the shoebox simulation in this script - length of each of the fixed polymers, length of the regulatory region, length of the gene etc.

Note the following numbers of different components for various box sizes. 

| Box size | # chromatin monomers | # actin | # Ser5P | # RBP | left anchors | right anchors |
|----------|----------------------|---------|---------|-------|--------------|---------------|
| 9        | 300                  | 240     | 300     | 450   | 1, 100       | 101, 300      |
| 10       | 336                  | 270     | 336     | 500   | 1, 112       | 113, 336      |
| 11       | 370                  | 300     | 370     | 550   | 1, 123       | 124, 370      |
| 12       | 400                  | 300     | 400     | 600   | 1, 133       | 134, 400      |
| 13       | 433                  | 325     | 433     | 650   | 1, 144       | 145, 433      |
| 15       | 500                  | 400     | 500     | 750   | 1, 166       | 167, 500      |
| 18       | 600                  | 500     | 600     | 900   | 1, 200       | 201, 600      |
| 21       | 700                  | 560     | 700     | 1050  | 1, 233       | 234, 700      |
| 24       | 800                  | 650     | 800     | 1200  | 1, 266       | 267, 800      |

### Potentials

One can define and set up the various "soft LJ potentials" in a file called `LJsoft_RSQ.table` using the script `setup_ljsoft_potentials.py`. The main idea behind these soft LJ potentials is to have a softer core for the potentials and forces. When two atoms start to overlap, hardcore repulsion kicks in and pushes them apart. These are set up using the usual Lennard-Jones potentials. However, if the atoms are very close (much closer than hardcore repulsion distances), the forces with the soft LJ potentials don't explode like they would with the usual LJ potentials. This is especially crucial for us because we have "chemical reactions" which deal with changing bonds and atom types, and this can potentially result in large blow-up of forces, making the simulation very unstable.

### Some actin simulations are unstable

With the current setup, actin simulations (when actin can polymerize) are sometimes unstable. Usual errors are atoms escaping the box boundary and bonds becoming very unstable. I haven't figured out the exact source of these errors. At the moment (September 2023), the parallelized code automatically handles these "failed" simulation runs, and restarts them.  

### Other parameters

#### Simulation phases

Each simulation run consists of an initial "Soft" to help the system relax from the initial configuration, followed by a long "Control" phase which makes up the main simulation run. If the simulation is a treament, then the treament occurs at the end of the Control phase, and the next and last phase is the "Treatment" phase.

#### Simulation length

Each simulation run consists of `NRuns + treatment_duration` "Python timesteps". `NRuns` is usually 2000 (and comprises the Soft and Control phases) and `treatment_duration` depends on the treament chosen (and comprises the Treament phase). Each Python timestep consists of `tRun = 200` LAMMPS timesteps, with `dt = 0.005` being the duration of a LAMMPS timestep. 

#### Gene activation

After an initial period (`t_activation_on`), the inactive gene becomes an induced gene. From that point on, at regular intervals (decided by `p_gene_activation`, the rate of activation), we count the number of Ser5P around the promoter of the gene (in a radius of `pol2release_radius`), and if it exceeds `ser5p_to_activate`, change the gene to an active gene.

#### Actin polymerization

After an initial period (`t_polymerization_on`), two actin monomers within a distance of `monomerEffectRadius` can spontaneously nucleate to form an actin filament with a probability of `p_nucleation`. Similarly, actin polymerization can occur at the end of an actin filament (faster on the plus end and slower on the minus end) if a monomer is in the vicinity (within a distance of `filEndEffectRadius`). Also, at a certain rate, depolymerization can occur at the end of an actin filament.

#### RBP -> RNP

After exploring a lot of different frameworks for RBP-RNP ratios and conversions, we finally settled on starting the simulation with only RBP, and slowly ramping up RNP (while decreasing RBP) over the entire simulation run to 90% RNP and 10% RBP as the final value. This is implemented independent of gene activations, and is assumed to represent transcription elsewhere (outside the shoebox) over time.

### Inhibitors

To implement any inhibitors or treatments, in the script `run_single_shoebox.py`, change the variable `treatment_duration` depending on the condition, and at the end of `NRuns` Python timesteps (the Control duration), depending on the treament, change different sets of parameters specifying interactions or other processes.

## Locations of data

### Gene-cluster visit without actin

The data from simulations for the gene-cluster shoebox simulations without actin is contained in the folder `GENE_CLUSTER_SHOEBOX`. Inside the folder are two folders `box10` and `box11` containing many subfolders for various parameters that can give rise to the contact map. I used `box11` to run 100 repeats for each parameter set, and this is our main dataset. The folder `snapshot_videos` contains snapshots and videos with `tImageDump = 33` at 30 fps, making sure the videos are smooth. The folder `contact_maps` contains the contact maps and other plots from the `box11` folder, generated using `contact_maps.ipynb`. A couple of these (parameter distributions for different regions and contact map) are already assembled in `Simulations_contact_map.svg`. 

This folder also contains the previous figure (threshold 40, RNP not ramped over time but only RBP in the system and a few other differences, which we now know are crucial) in `Simulations_contact_map_old.svg`. Use this only for reference, for the schematic or for presentations. Generate other panels of the figure (snapshots + tracks) using `contact_maps.ipynb` and other code.

### Gene-cluster visit with actin

The data from simulations for the actin shoebox simulations is contained in the folder `ACTIN_SHOEBOX`. Inside the folder are many folders of the form `box{x}-{y}`, each containing subfolders for various parameters / conditions. In `box{x}-{y}`, `x` refers to the box length as before, and `y` refers to the strength of the attrative potential between the regulatory element and Pol II Ser5P. These values are as indicated in the script `setup_ljsoft_potentials.py`. The setting 'weak' seems to give the best results. 

This folder also contains the actual code used to generate these folders. 

- Single run: `run_single_ACTIN_SMALL_BOX.py`
- Parallel runs: `run_singleCond_parallel.py`
- Soft potentials: `ljsoft_potential.py`

However, these are not as well commented as their equivalent files as described in this README, and can be replaced by the ones described here.

### Older actin simulations

The folder `ACTIN_SHOEBOX_Aug2022` contains older actin simulations, especially important/useful might be Ser5P concentration scans. 

## Parallel runs

The script `run_parallel_shoebox.sh` contains the base code one can use to run many repeats of `run_single_shoebox.py` in parallel. 

The following code would run 100 repeats of a shoebox simulation in a box of length 11 in the "Control" condition:

```
./run_parallel_shoebox.sh 1 100 11 Control
```

There are many variables one needs to set in the script itself. A few important ones are:

- N: no. of repeats to run in parallel (best if less than no. of CPU cores)
- THRESHOLD: Ser5P threshold for activation
- ACTIVATION: activation rate
- ACTIN: no. of free actin
- PNUC: probability of nucleating new actin filaments

As before, the argument `ACTIN` (no. of free actin) should be 0 for a gene-cluster visit simulation without actin, and >0 for an actin simulation.

### Arguments to pass

In the order to be passed (`./run_parallel_shoebox.sh 1 100 11 Control`), 

- REPEAT_START: start no. of repeat (starting number for output folders run1, run2 etc.)
- REPEAT_END: end no. of repeat
- BOX_LENGTH: length of the shoe-box in chromatin monomer units
- CONDITION: name of the condition - Control, LatB etc.

### Outputs from the simulation

Outputs from a run are stored in corresponding folder such as `run1`, `run2` etc. inside the specified output folder. Some other outputs are stored in single files directly in the specified output folder.

#### Snapshots

Snapshots are stored in the folder `image_files` inside the run's folder. 

#### Synthetic microscopy images

Synthetic microscopy images are stored in the folder `microscopy_files` inside the run's folder.

#### Plots

Some output plots like the gene track (ser5P around promoter and cluster-promoter distance), RNP vs time, Ser5P distribution along the length of the box are stored in the folder `figures` inside the run's folder.

#### Active durations

Active durations for each run are stored in a file `active_duration.txt` in the output folder, with each run getting a line in the file. Each line has three entries of the form `run_number,active_duration,total_length`. The units of `active_duration` and `total_length` are Python timesteps. For Control, `total_length` is 1850 and not 2000 because gene induction and activation start at `t_induction_on = t_activation_on = 150`.

## Other scripts/code

### Making a video from snapshots

The package `ffmpeg` (install with `brew install ffmpeg`) can be used to make a video from the snapshots. 

To make a video from the png files of run11 inside gene_cluster folder with a framerate of 30 fps, the following code would work. Of course, make sure that `tImageDump` in `run_single_shoebox.py` is set to a low enough value (like 33) for a smooth video.

```
ffmpeg -r 30 -f image2 -pattern_type glob -i "gene_cluster/run11/image_files/*?png" -vcodec libx264 -crf 20 -pix_fmt yuv420p gene_cluster/run11/complete.mp4
```

To cut a section out (first two and half minutes) from this video, the following code would work.

```
ffmpeg -i gene_cluster/run11/complete.mp4 -ss -00:00:03 -to 00:02:30 -c:v copy -c:a copy gene_cluster/run11/initial.mp4
```

`ss` is the start time of the section and `to` is the end time of the section. When you want the initial bits, it is better to give a negative value for `ss`.

### Making a gene track figure

While a gene track figure is already made by `run_single_shoebox.py` and saved in the `figures` folder, the scripts `plotTracks.py` and `plotInhibitorTrack.py` (both older versions currently) can be fairly easily adapted to prepare figures for the new data.

### Contact maps and gene design plots

In order to make these plots, first run (may need to adapt the code slightly) to make a summary file, called `summary_contact_grouped.txt`, for the given boxes and conditions. This groups each consecutive sets of 5 repeats each and computes averages for each set. Then, use the Jupyter notebook `contact_maps.ipynb` to make the contact maps, gene design plots and other related figures.

## Installation of LAMMPS + conda environment

### Homebrew

>Ensure you have a M1 version of brew (should be in /opt/homebrew)

If you don't have it, then run the following:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
```

You might also need to add brew's location to $PATH. The instructions should be displayed there.

After you make sure brew is present, if not already done before (independent of conda environment):

```
brew install cmake
brew install libpng
brew install ffmpeg
brew install jpeg
```

### Conda environment

If you don't have conda, download and install the M1 version of miniconda from https://docs.conda.io/projects/miniconda/en/latest/. 

Create a new environment `polymersimulations` (or any other name you like), and install the packages as below:

```
conda create -n polymersimulations 
conda activate polymersimulations
conda install -c conda-forge pandas scikit-learn matplotlib seaborn bokeh ipython voro
conda install -c conda-forge shapely
```

In case the Python script can't see LAMMPS or some other problem arises after finishing the next steps too, an alternative is to use the YML file `polymersimulations.yml` to create a conda environment called `polymersimulations` with versions of Python and other packages as given in the file (a little bit old). 

```
conda env create -f polymersimulations.yml
```

### LAMMPS

The following instructions are based on these links (modified for installation on an Apple Silicon based machine):

1. https://docs.lammps.org/Python_head.html
2. https://docs.lammps.org/Python_install.html#installing-the-lammps-python-module-and-shared-library (install as a shared library inside a virtual environment)

Download LAMMPS source code from GitHub at https://github.com/lammps/lammps/releases, unzip the folder into your wanted location and cd into the LAMMPS folder. 

NOTE: Not all versions of LAMMPS releases might work with M1. The version "Update 2 for Stable release 23 June 2022" (https://github.com/lammps/lammps/releases/tag/stable_23Jun2022_update2) definitely works. If you want, you can try newer versions, and if they don't work, go back to the above older version that works.

Run the following:

```
mkdir build-shared
cd build-shared
cmake -D BUILD_SHARED_LIBS=ON \
    -D LAMMPS_EXCEPTIONS=ON \
    -D PKG_PYTHON=ON \
    -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -D PYTHON_EXECUTABLE=$CONDA_PREFIX/bin/python \
    -D PKG_MOLECULE=ON \
    -D PKG_RIGID=ON \
    -D PKG_GRANULAR=ON \
    -D PKG_ASPHERE=ON \
    -D PKG_FEP=ON \
    -D PKG_EXTRA-PAIR=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-DUMP=ON \
    -D PKG_EXTRA-FIX=ON \
    -D PKG_EXTRA-MOLECULE=ON \
    -D PKG_MC=ON \
    -D PKG_MOLFILE=ON \
    -D PKG_REACTION=ON \
    -D PKG_VORONOI=ON \
    -D PKG_OPENMP=ON \
    -D PKG_OPT=ON \
    -D WITH_PNG=yes \
    -D WITH_FFMPEG=yes \
    ../cmake
```

The following step will take some time to run:

```
make
make install
```

At this point, LAMMPS should be ready to be run from inside Python in the conda environment. Run one of the shoe-box scripts to test if everything went well. If there are some issues, try the following and then run the script again.

```
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

Put following into activate.d/env_vars.sh
```
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
```
Put following into deactivate.d/env_vars.sh
```
DYLD_LIBRARY_PATH=:$DYLD_LIBRARY_PATH:
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH//:$CONDA_PREFIX\/lib:/:}
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH#:}
DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH%:}
```

If it still doesn't work, make sure to create the conda environment from the YML file, and try with the older version of LAMMPS (https://github.com/lammps/lammps/releases/tag/stable_23Jun2022_update2).