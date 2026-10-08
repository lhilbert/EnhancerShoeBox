# Running the parameter sweep on a Windows workstation

This sheet describes how to run the full simulation sweep with `run_sweep.py` on a Windows workstation.
The sweep covers the factorial grid analysed in the manuscript.
This grid consists of 3 promoter lengths, 10 activation rates and 11 activation thresholds, with 100 repeats each.
That makes 330 parameter combinations and 33,000 simulation runs.

The simulations run inside WSL2, the Linux environment built into Windows.
LAMMPS is used through its Python interface, which is not readily available on native Windows.

## What to expect

| | |
|---|---|
| Run time per simulation | about 7.5 min on one CPU core |
| Total compute | about 4,100 core-hours |
| Wall time with 14 parallel runs on an 8-core / 16-thread CPU | about 16–18 days |
| Disk space | about 0.6 MB per run, about 20 GB in total |
| Memory | well below 1 GB per run |

The runner prints an updated estimate of the remaining time every 50 finished runs.

The runner fills the grid evenly.
It first runs repeat 1 of all 330 combinations, then repeat 2 of all combinations, and so on.
A partially finished sweep therefore has the same number of repeats for every combination, give or take one.
It can be analysed at any point.

## One-time setup

### 1. Install WSL2 with Ubuntu

Open PowerShell as administrator and run:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart the computer when asked.
After the restart, open "Ubuntu 24.04" from the Start menu.
On first start, Ubuntu asks for a new user name and password.
These are independent of the Windows login.

### 2. Install system packages

In the Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y git tmux python3-venv
```

### 3. Create the Python environment

The LAMMPS version is pinned to the version used for testing.
The simulation script uses the `PyLammps` interface, which newer LAMMPS versions may no longer contain.

```bash
python3 -m venv ~/esb-venv
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"' >> ~/esb-venv/bin/activate
source ~/esb-venv/bin/activate
pip install lammps==2025.7.22.4.0 mpich numpy scipy pandas matplotlib seaborn bokeh pillow statsmodels scikit-learn
```

The `mpich` package provides the MPI library that the LAMMPS package needs.
The second line makes this library visible whenever the environment is activated.

Check the installation:

```bash
python -c "from lammps import lammps; print('LAMMPS', lammps(cmdargs=['-log','none','-screen','none']).version())"
```

This should print `LAMMPS 20250722`.

### 4. Get the code

Clone the repository into the Ubuntu home folder.
Do not use a folder under `/mnt/c`, because file access to Windows drives is much slower from WSL.

```bash
cd ~
git clone https://github.com/lhilbert/EnhancerShoeBox
cd EnhancerShoeBox
cd VisitorGeneMaps
```

### 5. Test run

Run two simulations of one parameter combination into a separate test folder:

```bash
python run_sweep.py --workers 2 --promoters 3 --activations 50 --thresholds 70 --repeats 1-2 --out box11_test
```

This takes about 8 minutes.
Afterwards, check the column `duration_s` in `sweep_log.csv` for the run time per simulation.
Then delete the test output:

```bash
rm -r box11_test sweep_log.csv
```

## Windows settings for the duration of the sweep

- Set the computer to never sleep: Settings → System → Power → Screen and sleep.
- Pause Windows updates, so that no automatic restart interrupts the sweep: Settings → Windows Update → Pause updates.
- Keep the Ubuntu window open. It can be minimised.

## Starting the sweep

Start a `tmux` session.
A `tmux` session keeps running when its terminal window is disconnected.

```bash
tmux new -s sweep
source ~/esb-venv/bin/activate
cd ~/EnhancerShoeBox/VisitorGeneMaps
python run_sweep.py --workers 14
```

The option `--workers` sets the number of simulations that run in parallel.
On an 8-core CPU with hyperthreading, 14 uses the machine nearly fully and leaves it usable.
Without the option, the runner uses the number of logical cores minus 2.

To leave the session running in the background, press `Ctrl-b` and then `d`.
To return to it later:

```bash
tmux attach -t sweep
```

## Checking progress

Open a second Ubuntu window and run:

```bash
source ~/esb-venv/bin/activate
cd ~/EnhancerShoeBox/VisitorGeneMaps
python run_sweep.py --status
```

The output states how many runs are finished.
It also states the minimum and maximum number of finished repeats per parameter combination.

The file `sweep_log.csv` contains one line per finished or failed run.
Each line records the parameters, the seed, the attempt number, the return code and the run time.

## Interruptions and restarts

The sweep can be stopped and restarted at any time.

- To stop the sweep, press `Ctrl-C` in the `tmux` session. The running simulations are terminated.
- To continue, run the same command again: `python run_sweep.py --workers 14`.

At each start, the runner checks which runs are finished.
A run counts as finished when its `geneTrack.txt` has 2000 lines and its completion marker exists.
Finished runs are skipped.
Unfinished runs are deleted and run again.
This also applies after a crash, a power cut or a Windows restart.
After a restart, open Ubuntu and repeat the commands from "Starting the sweep".

A run that fails three times in a row is reported as `FAILED` and skipped for the current session.
The next start of the runner tries it again.

## Reproducibility

Each run uses a fixed seed derived from its parameters:
seed = promoter × 10⁹ + threshold × 10⁶ + activation × 10³ + repeat.
The seed is written to `sweep_log.csv`.
With the same LAMMPS version, a run with the same seed reproduces `geneTrack.txt` and `ser5pAroundCluster.txt` exactly.

## Output

The results are written to `~/EnhancerShoeBox/VisitorGeneMaps/box11`.
There is one folder per parameter combination, named `Control_Promoter<P>_Threshold<T>_Act<A>`.
Each contains one folder per repeat, `run1` to `run100`.

In Windows Explorer, the output folder is found at:

```
\\wsl$\Ubuntu-24.04\home\<ubuntu user name>\EnhancerShoeBox\VisitorGeneMaps\box11
```

The aggregation script `dist_genestages_grouped_5percentile_10xaveraging.py` needs the files `run<r>/geneTrack.txt` and `dist_active.txt` from each combination.
To copy the output to an external drive, for example drive D:, run in Ubuntu:

```bash
cp -r ~/EnhancerShoeBox/VisitorGeneMaps/box11 /mnt/d/
```

Per-run figures and image dumps are switched off in this sweep.
To create figures for individual runs, run `run_single_GENES_STAGES.py` directly with the option `-g 1`.
