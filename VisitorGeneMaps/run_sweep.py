"""Run and manage the full parameter sweep of run_single_GENES_STAGES.py.

The sweep covers the factorial grid analysed in the manuscript:
promoter length 1-3, 10 activation rates, 11 activation thresholds,
100 repeats each (33,000 runs). Control thresholds (1, 370) are left out.

Runs are ordered by repeat number across the whole grid (repeat 1 of all
330 conditions, then repeat 2, ...), so the grid fills up evenly.

The script can be started again at any time with the same command.
Finished repeats are skipped. Unfinished repeats (e.g. after a crash,
reboot or Ctrl-C) are deleted and run again.

Each repeat gets a fixed seed derived from its parameters, so every run
can be reproduced individually. Figures and image dumps are switched off.

Usage (from this folder):
    python run_sweep.py --workers 16             # run or resume the sweep
    python run_sweep.py --status                 # show progress only
    python run_sweep.py --workers 16 --promoters 3 --repeats 1-10   # subset

A run is considered finished when parallel_counter/run<r>.txt exists and
run<r>/geneTrack.txt has 2000 lines.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

STOP = threading.Event()   # set on Ctrl-C: no new runs, no retries
RUNNING = set()            # simulation processes currently running
RUNNING_LOCK = threading.Lock()

PROMOTERS = [1, 2, 3]
ACTIVATIONS = [1, 5, 10, 15, 20, 25, 30, 50, 75, 100]
THRESHOLDS = [10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100]
BOX = 11
CONDITION = "Control"
N_STEPS = 2000          # Python steps per trajectory (NRuns in the simulation script)
MAX_ATTEMPTS = 3        # attempts per repeat within one session

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_SCRIPT = "run_single_GENES_STAGES.py"
LOG_FILE = os.path.join(HERE, "sweep_log.csv")


def condition_folder(out_root, p, t, a):
    return os.path.join(out_root, f"{CONDITION}_Promoter{p}_Threshold{t}_Act{a}")


def seed_for(p, t, a, r):
    """Unique, reproducible seed per repeat."""
    return p * 10**9 + t * 10**6 + a * 10**3 + r


def is_finished(folder, r):
    marker = os.path.join(folder, "parallel_counter", f"run{r}.txt")
    track = os.path.join(folder, f"run{r}", "geneTrack.txt")
    if not (os.path.isfile(marker) and os.path.isfile(track)):
        return False
    with open(track) as f:
        return sum(1 for _ in f) == N_STEPS


def clean_unfinished(folder, r):
    shutil.rmtree(os.path.join(folder, f"run{r}"), ignore_errors=True)
    for name in (f"run{r}.txt", f"progress_run{r}.txt"):
        path = os.path.join(folder, "parallel_counter", name)
        if os.path.isfile(path):
            os.remove(path)


def write_log(row):
    new = not os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["finished_at", "promoter", "threshold", "activation", "repeat",
                        "seed", "attempt", "returncode", "complete", "duration_s"])
        w.writerow(row)


def run_one(out_root, p, t, a, r, total):
    folder = condition_folder(out_root, p, t, a)
    seed = seed_for(p, t, a, r)
    # The simulation script expects the output folder with a trailing separator
    out_arg = os.path.relpath(folder, HERE).replace(os.sep, "/") + "/"
    cmd = [sys.executable, SIM_SCRIPT,
           "-b", str(BOX), "-r", str(r), "-t", str(total), "-o", out_arg,
           "-m", "0", "-c", CONDITION, "-p", str(p), "-a", str(a), "-x", str(t),
           "-g", "0", "-q", "1", "-s", str(seed)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if STOP.is_set():
            return None
        clean_unfinished(folder, r)
        os.makedirs(os.path.join(folder, f"run{r}"), exist_ok=True)
        start = time.time()
        with open(os.path.join(folder, f"run{r}", "console.log"), "w") as log:
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT, env=env)
            with RUNNING_LOCK:
                RUNNING.add(proc)
            proc.wait()
            with RUNNING_LOCK:
                RUNNING.discard(proc)
        done = is_finished(folder, r)
        if STOP.is_set() and not done:
            return None   # interrupted by the user: not a failure, redone at the next start
        write_log([datetime.now().isoformat(timespec="seconds"), p, t, a, r, seed,
                   attempt, proc.returncode, int(done), round(time.time() - start)])
        if done:
            return True
    return False


def parse_range(text):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="simulations run in parallel (default: number of cores minus 2)")
    ap.add_argument("--out", default=f"box{BOX}", help="output folder (default: box11)")
    ap.add_argument("--repeats", default="1-100", help="repeat range, e.g. 1-100")
    ap.add_argument("--promoters", default=None, help="subset, e.g. 1,3")
    ap.add_argument("--activations", default=None, help="subset, e.g. 1,50")
    ap.add_argument("--thresholds", default=None, help="subset, e.g. 10,70")
    ap.add_argument("--status", action="store_true", help="only report progress")
    args = ap.parse_args()

    out_root = os.path.join(HERE, args.out)
    repeats = parse_range(args.repeats)
    pick = lambda text, full: [int(x) for x in text.split(",")] if text else full
    promoters = pick(args.promoters, PROMOTERS)
    activations = pick(args.activations, ACTIVATIONS)
    thresholds = pick(args.thresholds, THRESHOLDS)
    total = len(repeats)

    # Repeat number is the outer loop: all conditions get repeat 1, then all get
    # repeat 2, and so on. The grid therefore fills up evenly, and a partial sweep
    # has (almost) the same number of repeats for every condition.
    jobs = [(p, t, a, r) for r in repeats for p in promoters for a in activations for t in thresholds]
    todo = [j for j in jobs if not is_finished(condition_folder(out_root, *j[:3]), j[3])]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} runs finished, {len(todo)} to do.")
    per_cond = {}
    for j in jobs:
        per_cond.setdefault(j[:3], 0)
    for j in set(jobs) - set(todo):
        per_cond[j[:3]] += 1
    print(f"Finished repeats per condition: min {min(per_cond.values())}, "
          f"max {max(per_cond.values())} (over {len(per_cond)} conditions).")
    if args.status or not todo:
        return

    print(f"Running with {args.workers} parallel workers. Press Ctrl-C to stop; "
          f"start the same command again to resume. Log: {LOG_FILE}", flush=True)
    n_ok = n_fail = 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_one, out_root, *j, total): j for j in todo}
    try:
        for fut in as_completed(futures):
            p, t, a, r = futures[fut]
            result = fut.result()
            if result is None:
                continue
            if result:
                n_ok += 1
            else:
                n_fail += 1
                print(f"FAILED after {MAX_ATTEMPTS} attempts: promoter {p}, threshold {t}, "
                      f"activation {a}, repeat {r}", flush=True)
            n_done = n_ok + n_fail
            if n_done % 50 == 0 or n_done == len(todo):
                hours = (time.time() - t0) / 3600
                eta = hours / n_done * (len(todo) - n_done)
                print(f"{datetime.now():%Y-%m-%d %H:%M}  {n_done}/{len(todo)} done, "
                      f"{n_fail} failed, about {eta:.1f} h remaining", flush=True)
    except KeyboardInterrupt:
        print("\nStopping: terminating running simulations ...", flush=True)
        STOP.set()
        with RUNNING_LOCK:
            for proc in RUNNING:
                proc.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        print(f"Stopped. {n_ok} runs completed in this session. "
              f"Unfinished runs are redone at the next start.")
        return
    pool.shutdown(wait=True)
    print(f"Session finished: {n_ok} runs completed, {n_fail} failed. "
          f"Start the same command again to retry failed runs.")


if __name__ == "__main__":
    main()
