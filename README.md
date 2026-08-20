## EnhancerShoeBox

Molecular Dynamics Simulations of Enhancer-Promoter Interactions in a Model Shoe Box

## Description of contributions
Original code as of December 2023 developed by Dr. Roshan Prizak while working as postdoctoral researcher in the research group of Lennart Hilbert at Karlsruhe Institute of Technology. Further code development and documentation by Dr. Ewa Anna Oprzeska-Zingrebe as postdoctoral researcher in the research group of Lennart Hilbert. Supervision of simulation and code development as well as limited direct contributions from Lennart Hilbert while Professor at Karlsruhe Institute of Technology.

## 1. Installation of LAMMPS + conda environment

### Homebrew (Mac OS)

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

## 2. Important parameters and other considerations

### Input configuration

The script `make_input_file.py` can generate input configuration files that are consistent with LAMMPS and have the geometry of a shoebox. 

You can use this script to generate input configuration files for a range of parameters. Set `is_actin_simulation` to `True` if generating input files for an actin simulation, and to `False` otherwise.

You can change various aspects of the shoebox simulation in this script - length of each of the fixed polymers, length of the regulatory region, length of the gene etc. The parameters to change are declared in the last section of the script:

- List of promoter lengths to make the input files for: 
- Default visitor gene: 3, non-visitor: 1. Usual range = 1, 2, 3
```
promoter_lengths = [1,2,3]
```

- Box sizes to consider. 11, 12 are good box sizes
```
box_sizes = [10,11,12]
```

- Is this an actin simulation? Two possibilities: True or False
```
is_actin_simulation = True
```

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

Alternatively, the script `make_input_file_flags.py` allows to declare the desired values of promoter lengths, box sizes and actin presence directly in the command line, without the need to modify the script itself. Use:

```
python3 make_input_file_flags.py -p <list of promoter lengths> -b <list of box sizes> -a <is actin simulation? 1 or 0>
```
For example,

```
python3 make_input_file_flags.py -p 1,2 -b 10,11 -a 0
```
creates input files for the systems defined by promoter lengths 1 and 2, bix sizes 10 and 11 and without actin (-a 0). If no flags are declared, the script generates input files for default configurations of promoter lengths 1,2 and 3, box sizes 10, 11 and 12, and without actin.

The command 
```
python3 make_input_file_flags.py -h
```
or
```
python3 make_input_file_flags.py --help
```
print help information about the use of the script.

The input files generated bo both scripts are stored in the newly created folder /input_files.

### Potentials

One can define and set up the various "soft LJ potentials" in a file called `LJsoft_RSQ.table` using the script `setup_ljsoft_potentials.py`. The main idea behind these soft LJ potentials is to have a softer core for the potentials and forces. When two atoms start to overlap, hardcore repulsion kicks in and pushes them apart. These are set up using the usual Lennard-Jones potentials. However, if the atoms are very close (much closer than hardcore repulsion distances), the forces with the soft LJ potentials don't explode like they would with the usual LJ potentials. This is especially crucial for us because we have "chemical reactions" which deal with changing bonds and atom types, and this can potentially result in large blow-up of forces, making the simulation very unstable.

Parameters that can be customized by the user inside the script:
 - lam=0.2 (scaling factor which relates LJ epsilon with epsilon_0)
 - n=1 (exponential factor)
 - alpha=0.5 (LJ scaling factor)

LJsoft_RSQ.table defines the following parameters for certain types of interactions (eg. chromatin-chromatin, actin-actin, chromatin-regulatory element etc.):

- column 1: consecutive order number for the N number of steps to count up to rmax with increment dr (N, dr and rmax to be declared inside the script)
- column 2: distance r rounded
- column 3: energy of the Lennard-Jones soft potential
- column 4: force of the Lennard-Jones soft potential

### Some actin simulations are unstable

With the current setup, actin simulations (when actin can polymerize) are sometimes unstable. Usual errors are atoms escaping the box boundary and bonds becoming very unstable. I haven't figured out the exact source of these errors. As of September 2023, the parallelized code automatically handles these "failed" simulation runs, and restarts them.  

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
#### RNAs in cells are associated with RNA-binding proteins (RBPs) to form ribonucleoprotein (RNP) complexes.

After exploring a lot of different frameworks for RBP-RNP ratios and conversions, we finally settled on starting the simulation with only RBP, and slowly ramping up RNP (while decreasing RBP) over the entire simulation run to 90% RNP and 10% RBP as the final value. This is implemented independent of gene activations, and is assumed to represent transcription elsewhere (outside the shoebox) over time.

### Inhibitors

To implement any inhibitors or treatments, in the script `run_single_shoebox.py`, change the variable `treatment_duration` depending on the condition, and at the end of `NRuns` Python timesteps (the Control duration), depending on the treament, change different sets of parameters specifying interactions or other processes.


## 3. Running the simulations without actin

The simulations should be run in conda environment.
Activate environment "polymersimulations":

```
conda activate polymersimulations
```

To deactivate the environment, type:

```
conda deactivate
```

### Running a single simulation

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

### Location of data

The generated data is stored in the output folder declared by the -o argument (eg. ~/gene_cluster in the example above). The output folder contains the following subfolders:

 - parallel_counter (valid only for parallel runs)
 - run1 
 
 And the output files (all appended across repeats in this condition folder, written by `run_single_shoebox.py`):
- active_duration.txt --> Active durations for each run. Each line has three entries of the form `run_number,active_duration,total_length`. The units of `active_duration` and `total_length` are Python timesteps. For Control, `total_length` is 1850 and not 2000 because gene induction and activation start at `t_induction_on = t_activation_on = 150`
- Each time step of each repeat is classified into one of four states based on its position relative to the nearest activation event: `induced` (baseline, before any activation-window influence), `approaching` (within a window before an activation event), `active` (the event_log==2 window itself), `receding` (within a window after an activation event). `dist_*.txt` and `ddist_*.txt` below are one flat file per state, pooling values from all time steps (across all repeats in the condition) that fall in that state.
- dist_active.txt / dist_approaching.txt / dist_induced.txt / dist_receding.txt --> one Reg(enhancer)-TSS(gene) distance value per line (in nm; same quantity as `d_reg_gene` in geneTrack.txt), for every time step classified into that state.
- ddist_active.txt / ddist_approaching.txt / ddist_induced.txt / ddist_receding.txt --> same as above, but each value is the *change* in that Reg-TSS distance from one time step to the next (`dist[t+1] - dist[t]`), for every time step (at time `t`) classified into that state.
- gene_stats.txt  --> one line per time step (from `t_induction_on` onward), columns: (1) is gene in contact with cluster, i.e. 1 if the Reg-TSS distance at that time step is below `pol2release_radius`, else 0; (2) the Reg-TSS distance itself, in nm (`d_rg * sig_chromatin`, same quantity as `dist_*.txt` and geneTrack.txt's `d_reg_gene`); (3) Ser5P around the gene TSS (not the promoter); (4) a Ser2P *proxy*, not an actual Ser2P count — 1 if the gene is in the "active" state (`event_log==2`) at that time step, else 0.

 
 The folder "run1" stores the main simulation data, distributed into subfolders and output files. They include:
 - figures
 - image_files --> simulation snapshots; can be combined into video (see: next sections)
 - microscopy_files --> synthetic microscopy images
 - geneTrack.txt --> columns: time, promoter_position_x, promoter_position_y, promoter_position_z, d_reg_promoter, d_reg_gene, Ser5P_around_promoter, Ser5P_around_gene, gene_state (0: inactive, 1: induced, 2: active)
 - ser5pAroundCluster.txt --> for each time step (row), the number of Ser5P (Pol II) atoms found within each of a series of concentric distance shells around the regulatory (enhancer) region (columns, labelled by each shell's outer radius, from a `cluster_r` grid spanning 0.65 to 2.5 in box units). Distance here is atom-to-atom (nearest regulatory-region atom to each Ser5P atom, via `cdist`), *not* distance to the regulatory region's center of mass: for each shell, the script counts all Ser5P atoms within that radius of any regulatory-region atom (a running/cumulative count as the radius grows), then takes the difference between consecutive radii, so each column's value is the number of Ser5P atoms newly captured within that shell.
 
 The subfolder "figures" stores the following files:
- gene_1_ddrg_events.pdf --> histogram of the distribution of the distances between enhancer (Reg) and transcription start site (TSS) for the genes at different states
- gene_ddrg_distr.pdf --> distribution of the Reg-TSS distances
- gene_drg_distr.pdf --> distribution of the Reg-TSS distances
- gene_rg.pdf --> nr of Ser5P around gene TSS vs. Reg-TSS distance. Green dots mark transcriptionally active genes, red - transcriptionally inactive. Line marks the activation threshold.
- gene_rp.pdf --> nr of Ser5P around promoter vs. Reg-Promoter distance
- gene_rp_time.pdf --> nr of Ser5P around promoter and Reg-promoter distance vs. time [min]
- n_rnp.pdf --> number of RNPs vs. time
- ser5p_cluster.pdf --> heatmap version of `ser5pAroundCluster.txt` (see above): rows are time steps, columns are the concentric distance shells around the regulatory (enhancer) region, color is the number of Ser5P atoms newly captured within each shell.
- ser5p_log.pdf --> white: position of superenhancer; grey: position of the gene center of mass; black: activation threshold; green dots: transcription activity. Left Y-axis: time step, X-axis: position
- ser5p.pdf --> white: position of superenhancer; grey: position of the gene center of mass; black: activation threshold; green dots: transcription activity. Left Y-axis: time step, X-axis: position, right Y-axis: local count of Ser5P

`drg`/`ddrg` (used in the `figures/gene_*.pdf` filenames above) and `dist`/`ddist` (used in the condition-level `dist_*.txt`/`ddist_*.txt` flat files) name the same two quantities, just in different contexts: `drg`/`dist` is the raw Reg(enhancer)-TSS(gene) distance in nm at a given time step; `ddrg`/`ddist` is the change in that distance from one time step to the next (`dist[t+1] - dist[t]`). The `drg`/`ddrg` figures are per-repeat histograms built directly in `run_single_shoebox.py`; the `dist`/`ddist` files are the same underlying values pooled across all repeats in a condition and split out per activation-event state (see the `dist_*.txt`/`ddist_*.txt` description above).

## 4. Parallel runs

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
- PROMOTER: promoter length

The argument `ACTIN` (no. of free actin) should be 0 for a gene-cluster visit simulation without actin, and >0 for an actin simulation.

### Arguments to pass

In the order to be passed (`./run_parallel_shoebox.sh 1 100 11 Control`), 

- REPEAT_START: start no. of repeat (starting number for output folders run1, run2 etc.)
- REPEAT_END: end no. of repeat
- BOX_LENGTH: length of the shoe-box in chromatin monomer units
- CONDITION: name of the condition - Control, LatB etc.

### Outputs from the simulation

Outputs from a run are stored in corresponding folder such as `run1`, `run2` etc. inside the specified output folder (`box<size>`, eg. \box11). Some other outputs are stored in single files directly in the specified output folder. The subfolders for each run store data for various parameters that can give rise to the contact map.

### Synthetic microscopy images

Synthetic microscopy images are stored in the folder `microscopy_files` inside the run's folder.

### Plots

Some output plots like the gene track (ser5P around promoter and cluster-promoter distance), RNP vs time, Ser5P distribution along the length of the box are stored in the folder `figures` inside the run's folder.

### Making a gene track figure

While a gene track figure is already made by `run_single_shoebox.py` and saved in the `figures` folder, the scripts `plotTracks_modif.py` (in the repo root; the current version of the older `plotTracks.py`) and `plotInhibitorTrack.py` (only valid for simulations with transcription inhibitors) can be easily adapted to prepare figures for the new data.

Run `plotTracks_modif.py` in the folder where the "geneTrack.txt" file is located, or declare the path to "geneTrack.txt" directly in the script. The generated output files are:

- gene_track.pdf --> two stacked panels sharing a time axis: Reg-Promoter distance (top) and number of Ser5P around the promoter (bottom), both vs. time. A dash-dotted horizontal line marks the respective activation threshold (`dist_thr`/`S5P_thr`) in each panel, and the trace is highlighted in a heavier grey wherever the gene is in the "active" state.
- gene_track_inset.pdf --> a small, axis-less zoom-in on the Ser5P-around-promoter trace, restricted to a narrow time window around the run's activation events; meant to be dropped into a larger figure as an inset. It is produced whenever `plotTracks_modif.py` is run — but its time window (like the vertical lines below) is hardcoded per example run.

The four vertical lines drawn on both `gene_track.pdf` and `gene_track_inset.pdf` (`t1`-`t4`, in orange) are **not** computed automatically from the data — they are timestamps hardcoded in `plotTracks_modif.py` for two specific example runs (`r==3` or `r==4`), chosen by eye to mark events of interest in those particular runs. To reuse the script on a different `geneTrack.txt`, `t1`-`t4` (and the `gene_track_inset.pdf` time window, which is derived from them) need to be edited by hand to match that run's events.

## 5. Contact maps and gene design plots

In order to make these plots, first run dist_genestages_grouped.py (may need to adapt the code slightly) to make a summary file, called `summary_contact_grouped.txt`, for the given boxes and conditions. This groups each consecutive sets of several repeats each and computes averages for each set. Then, use the Jupyter notebook `contact_maps.ipynb` to make the contact maps, gene design plots and other related figures.

If not in Jupyter environment, you can run 'contact_maps.py' Python script to generate contact maps. The default parameter space declared in the script (can be modified) include:

- promoter lengths = 1,2,3
- activation rate = 1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,75,100
- activation thresholds = 70,75,80

The script creates in the current directory the folder 'contact_maps', which contains the following files:

- contact_all.pdf --> contact map for all the genes
- contactRegionFocused.pdf --> contact map divided by regions
- byRegion.pdf
- byRegion_figure.pdf
- byRegion_new.pdf
- heatmap_pr.pdf --> relation between promoter length and activation rate for different gene designs
- heatmap_pr_figure.pdf 
- heatmap_thr.pdf --> relation between activation rate and threshold for different gene designs
- surface.pdf

`byRegion_new.pdf` and `heatmap_pr_figure.pdf` are both produced by `contact_maps_flags.py`/`contact_maps_modif.py` (they aren't a leftover — the code that saves them is live):

- byRegion_new.pdf --> 3 panels sharing an x-axis: promoter-length composition (as fractions), activation rate, and contact % — each broken out by the four transcription-control classes (NV-in, NV-ac, V-in, V-ac) below. The x-axis is that region/class (positions 0-3 for NV-in/NV-ac/V-in/V-ac), though the code that would label the ticks with those names is currently commented out, so the saved PDF shows unlabeled tick positions.
- heatmap_pr_figure.pdf --> the same promoter-length × activation-rate repeat-count data as `heatmap_pr.pdf` (4 panels, one per region), just with the color scale normalized to each panel's max and all axis ticks/labels/spines/colorbar stripped out — a bare version meant for direct use as a manuscript figure panel, not a raw-data dump.

Different classes of transcription control:

NV-in : non-visiting, inactive
NV-ac: non-visiting, active
V-in: visiting, inactive
V-ac: visiting, active

If the file 'summary_contact_grouped.txt' is not located in the current working directory, run 'contact_maps_flags.py' to define path to the input file:

```
python3 contact_maps_flags.py -f /path/to/file/summary_contact_grouped.txt
```

Options '-h' or '--help' displays "help" message.

### Up-to-date pipeline: gene-cluster visit maps (`VisitorGeneMaps/`)

The workflow above is kept for reference, but the current, up-to-date pipeline for generating gene maps lives in the [`VisitorGeneMaps/`](VisitorGeneMaps/) directory, documented in full in [`VisitorGeneMaps/README.md`](VisitorGeneMaps/README.md). It runs in three stages:

**1. Simulation sweep.** [`run_singleCond_parallel.sh`](VisitorGeneMaps/run_singleCond_parallel.sh) drives [`run_single_GENES_STAGES.py`](VisitorGeneMaps/run_single_GENES_STAGES.py) over the full `Promoter (1-3) × Threshold (10-100) × Activation` grid (plus repeats), writing output under `box<BOX_LENGTH>/<CONDITION>_Promoter<P>_Threshold<T>_Act<A>/run<r>/`.

**2. Aggregation into summary tables.** The `dist_genestages_*.py` family in `VisitorGeneMaps/` reads `geneTrack.txt` per repeat and aggregates into flat `summary_contact_*.txt` tables:

| Script | Output | Grouping |
| --- | --- | --- |
| `dist_genestages_all.py` | `summary_contact_all_Thresholds10-100.txt` | one row per repeat |
| `dist_genestages_all_5percentile.py` | `summary_contact_all_Thresholds10-100_5percentile.txt` | one row per repeat, plus 5-percentile distance columns |
| `dist_genestages_grouped.py` | `summary_contact_grouped_Thresholds10-100_test.txt` | 10 repeats averaged per row |
| `dist_genestages_grouped_5percentile_10xaveraging.py` | `summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt` | 10 repeats averaged per row, plus 5-percentile distance columns |
| `dist_genestages_grouped_cpp.cpp` (compiled as `dist_genestages_grouped_cpp`) | `summary_contact_grouped_Thresholds10-100_cpp.txt` | one row per repeat; C++ re-implementation for speed |

These scripts expect a *complete* sweep (every one of the 330 condition folders, each with its full set of repeats) — running them over a partial sweep silently produces an equally partial, misleading summary table rather than an error.

**3. Final maps.** [`shoebox_summary_maps.py`](VisitorGeneMaps/shoebox_summary_maps.py) reads a grouped summary table together with the microscopy file `gene_cluster_visit_OPoutcome_20260719.csv` and produces two figures, both plotted on the Pol II Ser5P (x) / active-fraction Pol II Ser2P (y) coordinate system:

- **Fig. 1 — interpolation maps** (`shoebox_colormaps.svg`/`.png`): 4 panels, each coloring the (Ser5P, Ser2P) plane by the locally expected value (50 nearest-neighbor weighted average) of a parameter or outcome (5-percentile enhancer–promoter distance, promoter length, activation threshold, activation rate). Microscopy data points are overlaid on the first panel with a reported Pearson r and p-value.
- **Fig. 2 — raw scatter maps** (`shoebox_scatter_maps.svg`/`.png`): 5 panels (A–E), one dot per simulated repeat, showing the raw spread behind Fig. 1's smoothed maps.

See `VisitorGeneMaps/README.md` for the full file/folder layout, script details, and known caveats (e.g. output filenames that don't match their generating script's current default).

## 6. Making video from snapshots

Snapshots are stored in the folder `image_files` inside the run's folder. 

The package `ffmpeg` (install with `brew install ffmpeg` for Mac OS) can be used to make a video from the snapshots. 

To make a video from the png files from the `image_files` folder with a framerate of 30 fps, the following code would work. Before, make sure that `tImageDump` in `run_single_shoebox.py` is set to a low enough value (like 33) for a smooth video.

```
ffmpeg -r 30 -f image2 -pattern_type glob -i "~/image_files/*?png" -vcodec libx264 -crf 20 -pix_fmt yuv420p video.mp4
```

To cut a section out (first two and half minutes) from this video, the following code would work.

```
ffmpeg -i video.mp4 -ss -00:00:03 -to 00:02:30 -c:v copy -c:a copy video_initial.mp4
```

`ss` is the start time of the section and `to` is the end time of the section. When you want the initial bits, it is better to give a negative value for `ss`.

To play the video:

```
ffplay output.mp4
```

