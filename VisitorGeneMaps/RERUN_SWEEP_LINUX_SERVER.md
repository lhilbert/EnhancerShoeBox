# Running the parameter sweep on a shared Linux server

This sheet describes how to run the simulation sweep with `run_sweep.py` on a shared Linux server over `ssh`.
It is written for a first round of 30 repeats per parameter combination.
The same commands extend the sweep to 100 repeats later.

The grid consists of 3 promoter lengths, 10 activation rates and 11 activation thresholds.
That makes 330 parameter combinations.
With 30 repeats each, the first round comprises 9,900 simulation runs.

## What to expect

| | 30 repeats | 100 repeats |
|---|---|---|
| Simulation runs | 9,900 | 33,000 |
| Compute | about 1,200 core-hours | about 4,100 core-hours |
| Wall time with 56 parallel runs on a 32-core / 64-thread CPU | about 1–1.5 days | about 4–4.5 days |
| Disk space | about 6 GB | about 20 GB |
| Memory | about 15–30 GB in total | same |

Each simulation runs on a single CPU core and takes about 7.5 minutes.
The runner prints an updated estimate of the remaining time every 50 finished runs.
With `nice -n 19`, the sweep yields to other users, so the wall time grows when the server is busy.

The runner fills the grid evenly.
It first runs repeat 1 of all 330 combinations, then repeat 2 of all combinations, and so on.

Replace `<user>` and `<server>` below with the login name and address of the server.

## 1. Log in and check the server

```bash
ssh <user>@<server>
nproc; free -g; df -h ~; python3 --version; which git tmux
```

Check the following:

- `nproc` shows the number of logical CPUs.
- `free -g` shows enough free memory, about 30 GB for 56 parallel runs.
- `df -h ~` shows enough free disk space in the home folder, about 6 GB for 30 repeats and 20 GB for 100 repeats.
- `which tmux` prints a path. If it prints nothing, use the alternative in step 6.

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
If `python3 -m venv` fails, ask the server administrator to install the package `python3-venv`.

## 3. Get the code (once)

```bash
cd ~
git clone https://github.com/lhilbert/EnhancerShoeBox
cd EnhancerShoeBox/VisitorGeneMaps
```

If the home folder is short of space, clone the repository into a data partition with enough free space instead.
Adjust the paths in the following steps accordingly.

## 4. Test run (about 8 minutes)

```bash
python run_sweep.py --workers 2 --promoters 3 --activations 50 --thresholds 70 --repeats 1-2 --out box11_test
cat sweep_log.csv
rm -r box11_test sweep_log.csv
```

The column `duration_s` in `sweep_log.csv` gives the run time per simulation in seconds.

## 5. Start the sweep

```bash
tmux new -s sweep
source ~/esb-venv/bin/activate
cd ~/EnhancerShoeBox/VisitorGeneMaps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 python run_sweep.py --workers 56 --repeats 1-30
```

The `export` line keeps every simulation on a single thread.
`nice -n 19` runs the sweep at the lowest priority.
All simulations started by the runner inherit this priority.
With 56 parallel runs, a few of the 64 threads stay free for other users.

To leave the sweep running, press `Ctrl-b` and then `d`.
After that, the `ssh` connection can be closed.

## 6. Alternative without tmux

```bash
source ~/esb-venv/bin/activate
cd ~/EnhancerShoeBox/VisitorGeneMaps
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
nohup nice -n 19 python run_sweep.py --workers 56 --repeats 1-30 > sweep.out 2>&1 &
echo $! > sweep.pid
```

The output goes to `sweep.out`.
To stop the sweep cleanly, run `kill -INT $(cat sweep.pid)`.

## 7. Check progress

To see the live output:

```bash
tmux attach -t sweep
```

Leave again with `Ctrl-b` and then `d`.

To check progress without attaching:

```bash
source ~/esb-venv/bin/activate
cd ~/EnhancerShoeBox/VisitorGeneMaps
python run_sweep.py --status --repeats 1-30
```

To check priority and threads of the running processes:

```bash
ps -u $USER -o pid,ni,nlwp,pcpu,cmd | grep run_s
```

Column `NI` should show 19.
Column `%CPU` should be at most about 100 per simulation, which corresponds to one core.

If the sweep was started without `nice`, lower its priority afterwards:

```bash
renice -n 19 -p $(pgrep -u $USER -f "run_s")
```

## 8. Interruptions and restarts

To stop the sweep, press `Ctrl-C` in the `tmux` session.
The running simulations are terminated.

To continue after a stop, a crash or a server reboot, repeat step 5.
Finished runs are skipped.
Unfinished runs are deleted and run again.

## 9. Copy the results to the Mac

Run on the Mac, after the sweep has finished:

```bash
rsync -av <user>@<server>:~/EnhancerShoeBox/VisitorGeneMaps/box11/ "/Volumes/Extreme Pro/Analysis_EnhancerShoebox/rerun_box11/"
```

`rsync` copies only files that are new or changed.
The same command can therefore be repeated later to fetch additional repeats.

## 10. Aggregate the results

From the folder `VisitorGeneMaps`, on the server or on the Mac:

```bash
python dist_genestages_grouped_5percentile_10xaveraging.py --box-dir <path to box11> --max-repeats 30
```

With 30 repeats, each parameter combination contributes 3 groups of 10 runs, which gives 990 rows in total.
The script reports whether all combinations have the same number of finished runs.

## 11. Extend to 100 repeats

Start the sweep again with the full repeat range:

```bash
nice -n 19 python run_sweep.py --workers 56 --repeats 1-100
```

Use the same `tmux` and `export` lines as in step 5.
Repeats 1–30 are skipped, because they are already finished.
The remaining 70 repeats take about 3 more days.
Afterwards, aggregate without the option `--max-repeats`.

## Working from a phone

The sweep can be started and monitored from a phone with an SSH app.
Examples are Termius or Blink Shell on iOS, and Termius or JuiceSSH on Android.

- If the server is only reachable from within the KIT network, connect the phone to the KIT VPN first.
- Always start the sweep inside `tmux` (step 5). Mobile connections drop easily, and `tmux` keeps the sweep running when they do.
- After a dropped connection, log in again and run `tmux attach -t sweep`.
- Copy the commands from this sheet instead of typing them.
- Step 9 (copying the results) needs the Mac.
