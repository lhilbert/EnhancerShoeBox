# new_simulations

## Simulation output directories (e.g. `box11/`)

Directories named `box<BOX_LENGTH>/` (e.g. `box11/`) hold the raw output of the
shoebox simulation sweeps. Each contains one subfolder per parameter
condition, named:

```
<CONDITION>_Promoter<PROMOTER>_Threshold<THRESHOLD>_Act<ACTIVATION>/
```

e.g. `Control_Promoter1_Threshold100_Act50/`. Inside each condition folder:

- `run1/`, `run2/`, … `runN/` — one subfolder per repeat, each containing
  `geneTrack.txt`, `ser5pAroundCluster.txt`, `figures/`, `image_files/`,
  `summaries/`.
- `parallel_counter/` — a `run<N>.txt` marker file per completed repeat, used
  to track progress across parallel jobs.
- `gene_stats.txt`, `dist_*.txt`, `ddist_*.txt`, `active_duration.txt` —
  aggregated stats appended across repeats.

These are produced by:

- **[`run_singleCond_parallel.sh`](run_singleCond_parallel.sh)** — bash driver
  that loops over the full `PROMOTERS × ACTIVATIONS × THRESHOLDS` grid
  (`THRESHOLDS=(1 10 20 30 40 50 60 70 75 80 90 100 370)`) and over repeats,
  builds the `box<N>/<CONDITION>_Promoter..._Threshold..._Act.../run<r>/`
  output path, and launches repeats in parallel (semaphore via
  `run_with_lock`/`open_sem`), skipping any repeat whose
  `figures/ser5p_cluster.pdf` already exists. Its threshold grid is the one
  that includes `1` and `370`, which is why folder names like
  `Control_Promoter1_Threshold1_Act75` and
  `Control_Promoter1_Threshold370_Act10` appear in `box11/`.
- **[`run_single_GENES_STAGES.py`](run_single_GENES_STAGES.py)** — the actual
  simulation, invoked once per repeat by the driver above (`python
  run_single_GENES_STAGES.py -b ... -r ... -t ... -o ... -p ... -a ... -x
  ...`). Writes the per-repeat files (`run<r>/geneTrack.txt`,
  `run<r>/ser5pAroundCluster.txt`, `parallel_counter/run<r>.txt`) and appends
  to the condition-level files (`gene_stats.txt`, `active_duration.txt`,
  `dist_{induced,approaching,active,receding}.txt`,
  `ddist_{induced,approaching,active,receding}.txt`).

The current sweep (330 conditions, seeded, resumable, no per-run figures) is
run with [`run_sweep.py`](run_sweep.py) instead, which calls
`run_single_GENES_STAGES.py` once per repeat with a fixed seed
(promoter·10⁹ + threshold·10⁶ + activation·10³ + repeat). See
[`RERUN_SWEEP_LINUX_SERVER.md`](RERUN_SWEEP_LINUX_SERVER.md) for running it.

A near-identical sibling pair, `run_parallel_shoebox.sh` /
`run_single_shoebox.py`, writes the same file layout but hardcodes a single
fixed condition (`THRESHOLD=80`, `ACTIVATION=30`, `PROMOTER=3`) rather than
looping over a grid; it is used for one-off single-condition runs, not for
generating a full sweep like `box11/`.

## From raw output to summary tables

The per-repeat and per-condition files above are aggregated across all
conditions/repeats into the flat `summary_contact_*.txt` tables (one row per
repeat or per group of repeats, columns `Promoter, Threshold, Activation,
S5PInt, S2PInt, Contact, DistActivation, ...`) by a family of
`dist_genestages_*.py` scripts. Each reads `geneTrack.txt` per repeat under
`box11/<CONDITION>_Promoter<P>_Threshold<T>_Act<A>/run<r>/`, computes mean
Pol II Ser5P, percent-active (Ser2P proxy), percent-in-contact, mean
activation distance, and (for the `_5percentile` variants) the 5th-percentile
enhancer–gene and enhancer–promoter distances, then writes one summary row
per repeat (or, for the averaging variants, one row per block of consecutive
repeats). They all loop over `thresholds = [10, 20, ..., 100]` only (11
values), which is why their output filenames say `Thresholds10-100` — the
`Threshold1` and `Threshold370` control conditions in `box11/` are
deliberately excluded from these summaries.

| Script | Output | Grouping |
| --- | --- | --- |
| [`dist_genestages_all.py`](dist_genestages_all.py) | `summary_contact_all_Thresholds10-100.txt` | one row per repeat |
| [`dist_genestages_all_5percentile.py`](dist_genestages_all_5percentile.py) | `summary_contact_all_Thresholds10-100_5percentile.txt` | one row per repeat, plus 5-percentile distance columns |
| [`dist_genestages_grouped.py`](dist_genestages_grouped.py) | `summary_contact_grouped_Thresholds10-100_test.txt` | 10 repeats averaged per row |
| [`dist_genestages_grouped_5percentile_10xaveraging.py`](dist_genestages_grouped_5percentile_10xaveraging.py) | `summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt` | 10 repeats averaged per row, plus 5-percentile distance columns |
| [`dist_genestages_grouped_cpp.cpp`](dist_genestages_grouped_cpp.cpp) (compiled as `dist_genestages_grouped_cpp`) | `summary_contact_grouped_Thresholds10-100_cpp.txt` | one row per repeat; C++ re-implementation of the grouped logic for speed |

Run any of these from this directory with e.g. `python dist_genestages_all.py`
(needs `seaborn` and `shapely` installed, which are used but not listed in
`pyproject.toml`). The `shoebox_summary_maps.py` figures below then read
whichever `summary_contact_*.txt` table matches the variable set and grouping
they need.

**Current aggregation.** The manuscript uses
`dist_genestages_grouped_5percentile_10xaveraging.py` with these options:

| Option | Meaning |
| --- | --- |
| `--box-dir DIR` | folder with the condition folders (default `box11`) |
| `--n-runs N` | use the first N finished runs per condition; interrupted runs are skipped and replaced by later ones |
| `--max-repeats N` | use only runs numbered 1 to N (no replacement) |
| `--from-step S` | first Python step of the common time window for all readouts (default 150, gene induction) |
| `--legacy` | time windows of the original analysis, for comparison only |
| `--output FILE` | output table |

A run counts as finished when `parallel_counter/run<r>.txt` exists and
`run<r>/geneTrack.txt` has 2000 lines. Each `geneTrack.txt` is read once, and
one progress line per condition is printed. Only complete groups of 10 runs
enter the table. The manuscript table was made with
`--box-dir <sweep folder> --n-runs 60` (330 conditions × 6 groups = 1980 rows).

Note: `summary_contact_grouped.txt` and
`summary_contact_grouped_Thresholds10-100.txt` share `dist_genestages_grouped.py`'s
column layout but not its current output filename
(`..._Thresholds10-100_test.txt`). To regenerate either exactly as currently
named, adjust the output filename in the script to match before running it.

**Prerequisite: a complete `box11/` sweep.** These aggregation scripts loop
over the full `Promoter (1-3) × Threshold (10-100) × Activation` grid and
expect every one of the 330 resulting condition folders under `box11/` to be
present, each with its full set of repeats. Producing that complete sweep
means running [`run_singleCond_parallel.sh`](run_singleCond_parallel.sh)
(which drives [`run_single_GENES_STAGES.py`](run_single_GENES_STAGES.py)) for
every condition in that grid, not just a subset — an aggregation run over a
partial `box11/` silently yields an equally partial, misleading summary
table rather than an error. The `summary_contact_*.txt` files currently
checked into this repository were built from the original complete
330-condition sweep and reproduce the previously published figures. The rerun
sweep with the fixed code (November 2026) and its aggregated table are archived
separately;
`box11/` as currently present on disk is a smaller, partial set of leftover
condition folders and is not sufficient to regenerate those summaries — it is
intentionally left out of the analysis rather than used to overwrite them.

## Summary maps: `shoebox_summary_maps.py`

Run with `python shoebox_summary_maps.py` from this directory, or with
`python shoebox_summary_maps.py --out-dir <folder>` to read the summary table
from and write all outputs to another folder. `--summary` and `--microscopy`
set the two input files individually. The microscopy table
`gene_cluster_visit_OPoutcome_20260719.csv` is taken from this directory if it
is not in the output folder. `--n-perm` sets the number of permutations for the
mutual information (default 1000). The plot range is that of the published
figure (Ser5P 0–20, active fraction 0–0.20) and extends in steps of 5 and 0.05
where the data reach beyond it. The script produces two figures, both plotted
on the same Pol II Ser5P (x) / active-fraction Pol II Ser2P (y) coordinate
system, and one table:

- **Fig. 1 — interpolation maps (`shoebox_colormaps.svg`/`.png`)**: one row
  of 4 panels. Each simulated repeat is a point in (Ser5P, Ser2P) space
  carrying a parameter or outcome value (5-percentile enhancer–promoter
  distance, promoter length, activation threshold, activation rate). A
  smooth background is built by evaluating a weighted average over each
  grid point's 50 nearest simulated neighbors, so the color at any
  (Ser5P, Ser2P) location is the *locally expected* value of that
  variable — i.e. "if a gene sits at this Ser5P/Ser2P state, what
  parameter/outcome does the simulation predict on average?". The
  simulated points' convex hull is outlined in black, and transformed
  microscopy data points are overlaid on the first panel for comparison,
  with a reported Pearson r and p-value (also printed to stdout). The panel
  labels give the mutual information between the joint Ser5P/Ser2P readout
  and each design parameter, as the share of the parameter's entropy that is
  removed by knowing the readouts.

- **Fig. 2 — raw scatter maps (`shoebox_scatter_maps.svg`/`.png`)**: 2 rows,
  5 panels (A–E), one dot per simulated repeat (no averaging/interpolation).
  Top row (A, B): 5-percentile enhancer–promoter and enhancer–gene distance.
  Bottom row (C, D, E): promoter length, activation threshold (both linear
  color scale), and activation rate (log color scale). These show the raw
  spread/density behind Fig. 1's smoothed maps — useful for checking that a
  given region of Ser5P/Ser2P space is actually well-sampled before trusting
  the interpolated value there, and for spotting outliers or sub-populations
  that averaging would hide.

- **Mutual information table (`mutual_information_knn_k3.csv`)**: for each
  design parameter and for Ser5P alone, Ser2P alone and both combined, the
  mutual information I in bits, estimated without binning by the
  nearest-neighbour estimator for a continuous and a discrete variable
  (Kraskov et al. 2004; Ross 2014; k = 3, standardized readouts); the chance
  level I0, the mean over stratified permutations (the tested parameter is
  shuffled among conditions within each combination of the other two
  parameters, all groups of a condition keeping one value); the parameter
  entropy H; the share s = (I − I0)/H; and the one-sided permutation P value.

## One-command rerun analysis: `run_rerun_analysis.py`

`python run_rerun_analysis.py --box-dir <sweep folder> --n-runs 60` checks how
many finished runs each condition has, runs the aggregation with `--n-runs`,
and then `shoebox_summary_maps.py`, writing everything to
`<drive>/rerun_analysis_<N>runs/` unless `--out-dir` is given. Without
`--box-dir`, it searches `--drive` for the folder holding the condition
folders. `--skip-aggregation` redraws the maps from an existing table.
