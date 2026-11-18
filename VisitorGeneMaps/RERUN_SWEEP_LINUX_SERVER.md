# Running the parameter sweep on a shared Linux server

This sheet describes how to run the simulation sweep with `run_sweep.py` on the group's shared Linux server over `ssh`.
All commands assume the login `hilbertlab` and the clone in `~/Lennart/EnhancerShoeBox`.

The grid consists of 3 lengths of the promoter-proximal affinity region, 10 activation rates and 11 activation thresholds.
That makes 330 parameter combinations.
With 30 repeats each, the sweep comprises 9,900 simulation runs.
With 100 repeats each, it comprises 33,000 runs.

## What to expect

These numbers were measured on 1 November 2026 with the fixed code (commit `64fbeb8`) and 32 parallel runs.

| | 30 repeats | 100 repeats |
|---|---|---|
| Simulation runs | 9,900 | 33,000 |
| Time per run, 32 in parallel | about 24 min | about 24 min |
| Wall time | about 5 days | about 17 days |
| Disk space | about 6 GB | about 20 GB |
| Memory | about 20 GB in total | same |

Each simulation runs on a single CPU core.
The runner fills the grid evenly.
It first runs repeat 1 of all 330 combinations, then repeat 2 of all combinations, and so on.
A sweep started with `--repeats 1-100` can therefore be aggregated at any time, for example after 30 repeats.

## The shared account

All tmux sessions and processes belong to the account `hilbertlab`, not to the connection.
Every login as `hilbertlab` sees the same sessions, whether from the Mac or from a phone.
Other group members who log in as `hilbertlab` see them too and can attach to them.

## 1. Log in and check the server

```bash
ssh hilbertlab@<server>
nproc; free -g; df -h ~; python3 --version; which git tmux
```

Check the following:

- `nproc` shows 64 logical CPUs (32 physical cores).
- `free -g` shows about 20 GB of free memory.
- `df -h ~` shows enough free disk space, about 6 GB for 30 repeats and 20 GB for 100 repeats.
- `which tmux` prints a path. If it prints nothing, use the alternative in step 7.

## 2. Create the Python environment (once)

```bash
python3 -m venv ~/esb-venv
echo 'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"' >> ~/esb-venv/bin/activate
source ~/esb-venv/bin/activate
pip install lammps==2025.7.22.4.0 mpich numpy scipy pandas matplotlib seaborn bokeh pillow statsmodels scikit-learn
python -c "from lammps import lammps; print('LAMMPS', lammps(cmdargs=['-log','none','-screen','none']).version())"
```

The last command should print `LAMMPS 20250722`.
The LAMMPS version is pinned, because the simulation script uses the `PyLammps` interface, which newer LAMMPS versions may no longer contain.

## 3. Get or update the code

The first time:

```bash
mkdir -p ~/Lennart && cd ~/Lennart
git clone https://github.com/lhilbert/EnhancerShoeBox
```

Every time before a new sweep:

```bash
cd ~/Lennart/EnhancerShoeBox
git pull
git log --oneline -1
```

The last command shows the commit that the sweep will use.
It must match the latest commit on GitHub.

## 4. Use a fresh output folder after every code change

The runner skips every run that it finds finished in the output folder.
Runs from an older code version would therefore be mixed into the new data without warning.
After every change to the simulation code, start with an empty output folder.
Move the old folder aside instead of deleting it:

```bash
cd ~/Lennart/EnhancerShoeBox/VisitorGeneMaps
mv box11 box11_old_<date>
mv sweep_log.csv sweep_log_old_<date>.csv
```

## 5. Test run (about 10 minutes)

```bash
cd ~/Lennart/EnhancerShoeBox/VisitorGeneMaps
source ~/esb-venv/bin/activate
python run_sweep.py --workers 2 --promoters 3 --activations 50 --thresholds 70 --repeats 1-2 --out box11_test
tail -2 sweep_log.csv
rm -r box11_test sweep_log.csv
```

The column `duration_s` in `sweep_log.csv` gives the run time per simulation in seconds.
The column `complete` must be 1.

## 6. Start the sweep

```bash
tmux new -s sweep
cd ~/Lennart/EnhancerShoeBox/VisitorGeneMaps
source ~/esb-venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 python run_sweep.py --workers 32 --repeats 1-100
```

The `export` line keeps every simulation on a single thread.
`nice -n 19` runs the sweep at the lowest priority, so it yields to other users.
All simulations started by the runner inherit this priority.

The runner first prints how many runs are finished and how many are still to do.
After that, it prints a progress line only after every 50 finished runs.
The first progress line therefore appears only after about 40 minutes.

To leave the sweep running, press `Ctrl-b` and then `d`.
After that, the `ssh` connection can be closed.

**Why 32 parallel runs.**
Use one run per physical core (32), not one per hardware thread (64).
Earlier sweeps with 56 parallel runs were about 25 times slower per run than a run alone.
The cause was contention for the shared CPU caches, because every run reads its pair-potential tables at random positions.
The simulation now uses smaller tables (`pair_style table linear 2000` instead of `table lookup 30000`), with equal or better force accuracy.
With these tables and 32 parallel runs, the throughput is about five times higher than before.

## 7. Alternative without tmux

```bash
cd ~/Lennart/EnhancerShoeBox/VisitorGeneMaps
source ~/esb-venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup nice -n 19 python run_sweep.py --workers 32 --repeats 1-100 > sweep.out 2>&1 &
echo $! > sweep.pid
```

The output goes to `sweep.out`.
To stop the sweep cleanly, run `kill -INT $(cat sweep.pid)`.

## 8. Get an overview

To list the tmux sessions and attach to one:

```bash
tmux ls
tmux attach -t sweep
```

Leave again with `Ctrl-b` and then `d`.

To check what is running:

```bash
pgrep -af run_sweep.py
pgrep -fc run_single_GENES_STAGES.py
top -u hilbertlab
```

`pgrep -af run_sweep.py` must show exactly one runner.
Two lines mean that two sweeps are running at the same time.
`pgrep -fc` counts the running simulations and should show 32.
In `top`, column `NI` should show 19 and `%CPU` about 100 per simulation.
Leave `top` with `q`.

To check the progress of the sweep:

```bash
cd ~/Lennart/EnhancerShoeBox/VisitorGeneMaps
source ~/esb-venv/bin/activate
python run_sweep.py --status --repeats 1-100
tail -5 sweep_log.csv
```

The status shows the number of finished runs and the minimum and maximum number of finished repeats per condition.
`sweep_log.csv` gets one line per finished run.
Its last column is the run time in seconds, which should stay around 1,400 with 32 parallel runs.

To find the working folder of a running sweep, if unsure:

```bash
readlink /proc/$(pgrep -f run_sweep.py)/cwd
```

If the sweep was started without `nice`, lower its priority afterwards:

```bash
renice -n 19 -p $(pgrep -u $USER -f "run_s")
```

## 9. Interruptions and restarts

To stop the sweep, attach to its tmux session and press `Ctrl-C`.
Wait until the runner prints "Stopped.".
The running simulations are terminated.

To continue after a stop, a crash or a server reboot, repeat step 6 without changing the code.
Finished runs are skipped.
Unfinished runs are deleted and run again.

## 10. Copy the results to the Mac

Run on the Mac, at any time during or after the sweep:

```bash
rsync -av hilbertlab@<server>:~/Lennart/EnhancerShoeBox/VisitorGeneMaps/box11/ "/Volumes/Extreme Pro/Analysis_EnhancerShoebox/rerun_box11/"
```

`rsync` copies only files that are new or changed.
The same command can therefore be repeated later to fetch additional repeats.

## 11. Aggregate the results

From the folder `VisitorGeneMaps`, on the server or on the Mac:

```bash
python dist_genestages_grouped_5percentile_10xaveraging.py --box-dir <path to box11> --n-runs 60
```

`--n-runs 60` uses the first 60 finished runs of each parameter combination, in run-number order.
A run counts as finished when its completion marker exists and its `geneTrack.txt` has 2000 lines.
Interrupted runs are skipped, and the next finished run of the same combination takes their place.
With 60 runs, each combination contributes 6 groups of 10 runs, which gives 1980 rows in total.
The script prints one line per combination and reports whether all combinations have the same number of runs.
`--max-repeats N` instead uses only runs numbered 1 to N, without replacing interrupted runs.

The maps, the microscopy comparison and the mutual information then follow with

```bash
python shoebox_summary_maps.py --out-dir <folder with the summary table>
```

## Working from a phone

The sweep can be started and monitored from a phone with an SSH app.
Examples are Termius or Blink Shell on iOS, and Termius or JuiceSSH on Android.

- If the server is only reachable from within the KIT network, connect the phone to the KIT VPN first.
- Always start the sweep inside `tmux` (step 6). Mobile connections drop easily, and `tmux` keeps the sweep running when they do.
- After a dropped connection, log in again and run `tmux attach -t sweep`.
- Copy the commands from this sheet instead of typing them.
- Step 10 (copying the results) needs the Mac.
