import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import numpy as np
from matplotlib.colors import ListedColormap, Normalize, LogNorm
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import adjusted_mutual_info_score
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy import stats
from scipy.special import digamma
import argparse
import os
from pathlib import Path
from matplotlib.patches import Rectangle

# --------------------------
# Input and output locations
# --------------------------
# By default the script reads and writes in the current working directory,
# which reproduces the previous behaviour. --out-dir moves both the summary
# table it reads and the figures it writes to another folder, for example the
# analysis folder of a sweep on an external drive. --summary and --microscopy
# override the two input files individually. The microscopy table is looked up
# next to the script when it is not found in the output folder, because it is
# part of the repository and not of the simulation output.

SUMMARY_NAME = "summary_contact_grouped_Thresholds10-100_5percentile_10xaveraged.txt"
MICROSCOPY_NAME = "gene_cluster_visit_OPoutcome_20260719.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", default=None,
                    help="folder for the figures and the default input location (default: current folder)")
parser.add_argument("--summary", default=None,
                    help=f"summary table from the aggregation step (default: <out-dir>/{SUMMARY_NAME})")
parser.add_argument("--microscopy", default=None,
                    help=f"microscopy table (default: {MICROSCOPY_NAME} in <out-dir>, else next to this script)")
parser.add_argument("--n-perm", type=int, default=1000,
                    help="permutations for the mutual-information P values (default: 1000)")
parser.add_argument("--bare", action="store_true",
                    help="leave out panel titles, axis labels and the legend/MI label text, "
                         "so the panels can be placed under the figure's own Inkscape labels. "
                         "The white legend and MI boxes keep their size and position.")
args, _unknown = parser.parse_known_args()

script_dir = Path(__file__).resolve().parent
base_dir = Path(args.out_dir).resolve() if args.out_dir else Path.cwd().resolve()
base_dir.mkdir(parents=True, exist_ok=True)

summary_path = Path(args.summary).resolve() if args.summary else base_dir / SUMMARY_NAME

if args.microscopy:
    microscopy_path = Path(args.microscopy).resolve()
elif (base_dir / MICROSCOPY_NAME).is_file():
    microscopy_path = base_dir / MICROSCOPY_NAME
else:
    microscopy_path = script_dir / MICROSCOPY_NAME

print(f"Summary table:   {summary_path}")
print(f"Microscopy data: {microscopy_path}")
print(f"Figures go to:   {base_dir}")

# --------------------------
# Define Parula colormap manually
# --------------------------
parula_cm_data = np.array([
    [0.2081, 0.1663, 0.5292],
    [0.1976, 0.2675, 0.7060],
    [0.0712, 0.3960, 0.7040],
    [0.0329, 0.5656, 0.6195],
    [0.1809, 0.7490, 0.4920],
    [0.4393, 0.8671, 0.2546],
    [0.7008, 0.9024, 0.1600],
    [0.9469, 0.8871, 0.1462],
    [0.9932, 0.9062, 0.1439],
    [0.9932, 0.9500, 0.2500],
    [0.9994, 0.9994, 0.1620]
])
x_old = np.linspace(0, 1, parula_cm_data.shape[0])
x_new = np.linspace(0, 1, 256)
parula_interp = np.zeros((256, 3))
for i in range(3):
    f = interp1d(x_old, parula_cm_data[:, i], kind="cubic", bounds_error=False, fill_value="extrapolate")
    parula_interp[:, i] = f(x_new)
parula_cmap = ListedColormap(parula_interp)

# Manual colorbar limits for each subplot (edit these values as needed)
manual_cmap_limits = {
    "SE-Promoter 5-perc. dist. [nm]": (200.0, 500.0),
    "SE-P 5% dist [nm]": (200.0, 500.0),
    "Promoter length [monomers]": (1.0, 3.0),
    "Threshold [no. activators]": (0.0, 100.0),
    "Activation [1/min]": (1e-1, 1e1),
}

# --------------------------
# Load the data
# --------------------------

# Load the data from the updated input file
data = pd.read_csv(summary_path)
print(f"Rows in summary table: {len(data)}")
data.columns = ["Promoter", "Threshold", "Activation", "Runs", "S5PInt", "S2PInt", "Contact", "DistActivation", "5percentRG", "5percentRP"]



# Convert columns to numeric type to avoid non-numeric errors
data["Threshold"] = pd.to_numeric(data["Threshold"], errors="coerce")
data["Promoter"] = pd.to_numeric(data["Promoter"], errors="coerce")
# Convert to 1/min: the simulation uses a per-step activation probability of a/1000
# with 0.6 s per Python step, so k_act = (a/1000)/(0.6 s) = 0.1*a 1/min
data["Activation"] = pd.to_numeric(data["Activation"], errors="coerce")*0.1
data["5percentRP"] = pd.to_numeric(data["5percentRP"], errors="coerce")

# Remove rows with a Threshold below a chosen cutoff
threshold_cutoff = 500.0  # Adjust this value as needed
activation_cutoff = 500.0  # Adjust this value as needed
data = data.loc[data["Threshold"] <= threshold_cutoff].copy()
data = data.loc[data["Activation"] <= activation_cutoff].copy()


# Set global styling for fonts
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Arial',
    'font.weight': 'regular',
    'mathtext.default': 'regular',
})

# Create the figure with four evenly spaced subplots
fig, axes = plt.subplots(
    figsize=(11, 2.2),
    nrows=1,
    ncols=4,
    sharex=True,
    sharey=True,
    constrained_layout=False,
)
axes = np.atleast_1d(axes)

# Spread the subplots across the full canvas area
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.14, top=0.92, wspace=0.28, hspace=0.02)


# --- Correction to enable comparability to experimental outcomes ---

S2P_vals = 0.01 * data["S2PInt"].values
S5P_vals = 0.1 * data["S5PInt"].values + 20 * S2P_vals  # <-- Adjusted to include S2P contribution


# --------------------------
# Create interpolated colormap (N-nearest neighbors + weighted averaging)
# --------------------------

# --- Create background grid ---
# Plot range: the range of the original figure (Ser5P 0-20, active fraction 0-0.20),
# extended in steps of 5 and 0.05 where the simulated data reach beyond it
x_plot_min, y_plot_min = 0.0, 0.0
x_plot_max = max(20.0, 5.0 * np.ceil(S5P_vals.max() / 5.0))
y_plot_max = max(0.20, 0.05 * np.ceil(S2P_vals.max() / 0.05))
support_S5P = np.linspace(x_plot_min, x_plot_max, 200)
support_S2P = np.linspace(y_plot_min, y_plot_max, 200)
x_ticks = np.arange(0, x_plot_max + 1e-9, 5.0)
y_ticks = np.round(np.arange(0, y_plot_max + 1e-9, 0.05), 2)

# Normalize grid coordinates for fair distance calculation
norm_S5P = (support_S5P - S5P_vals.mean()) / S5P_vals.std()
norm_S2P = (support_S2P - S2P_vals.mean()) / S2P_vals.std()

grid_S5P, grid_S2P = np.meshgrid(norm_S5P, norm_S2P)

# Nearest neighbors and averaging parameters
N_neighbors=50
averaging_KK=0.25

# Normalize coordinates for fair distance calculation
coords = np.column_stack([
    (S5P_vals - S5P_vals.mean()) / S5P_vals.std(),
    (S2P_vals - S2P_vals.mean()) / S2P_vals.std()
])
 
# Fit neighbor search
nbrs = NearestNeighbors(n_neighbors=N_neighbors).fit(coords)

# Prepare grid values
Z_promoter = np.zeros_like(grid_S5P)
Z_activation = np.zeros_like(grid_S5P)
Z_threshold = np.zeros_like(grid_S5P)
Z_RPdist = np.zeros_like(grid_S5P)

promoter_vals = data["Promoter"].values
activation_vals = data["Activation"].values
threshold_vals = data["Threshold"].values
RPdist_vals = data["5percentRP"].values

for i in range(grid_S5P.shape[0]):
    for j in range(grid_S5P.shape[1]):
        # normalize grid coords
        gx = grid_S5P[i, j]
        gy = grid_S2P[i, j]

        # find nearest neighbors
        distances, indices = nbrs.kneighbors([[gx, gy]], n_neighbors=N_neighbors)

        # weighted average of their values
        grid_dist_vec = distances[0].flatten()
        weights = averaging_KK / (averaging_KK + grid_dist_vec)
        Z_promoter[i, j] = np.sum(weights * promoter_vals[indices[0]]) / np.sum(weights)
        Z_activation[i, j] = np.sum(weights * activation_vals[indices[0]]) / np.sum(weights)
        Z_threshold[i, j] = np.sum(weights * threshold_vals[indices[0]]) / np.sum(weights)
        Z_RPdist[i, j] = np.sum(weights * RPdist_vals[indices[0]]) / np.sum(weights)

# --------------------------
# Mutual information between map location and design parameters
# --------------------------
# Bin-free nearest-neighbour estimator for a continuous readout and a discrete
# parameter (Ross 2014, PLoS ONE 9:e87357). This is the construction used by
# scikit-learn's mutual_info_classif, applied here to the joint two-dimensional
# readout. k sets the neighbourhood size and thereby the resolution; k = 3 is the
# conventional default. The readouts are standardized as for the maps.
# Chance level: stratified permutation test. Within each combination of the other
# two parameters, the values of the tested parameter are shuffled among conditions,
# and all groups of one condition keep the same label.
# Reported share = (MI - mean permuted MI) / H(parameter): the fraction of the
# parameter's uncertainty that is removed by knowing the location in the map.
MI_K = 3
MI_N_PERM = args.n_perm

S5P_norm = (S5P_vals - np.nanmean(S5P_vals)) / np.nanstd(S5P_vals)
S2P_norm = (S2P_vals - np.nanmean(S2P_vals)) / np.nanstd(S2P_vals)
rng = np.random.default_rng(0)
readouts = {
    "S5P": S5P_norm[:, None],
    "S2P": S2P_norm[:, None],
    "S5P+S2P": np.column_stack([S5P_norm, S2P_norm]),
}
# A tiny jitter separates tied values, which the estimator assumes to be absent.
readouts = {name: X + 1e-10 * rng.standard_normal(X.shape) for name, X in readouts.items()}


def mi_knn(X, y, k=MI_K):
    """Mutual information (nats) between continuous X (n x d) and discrete labels y."""
    n = len(y)
    radius = np.empty(n)
    k_all = np.empty(n)
    counts = np.empty(n)
    for label in np.unique(y):
        mask = y == label
        count = mask.sum()
        kk = min(k, count - 1)
        r = NearestNeighbors(n_neighbors=kk).fit(X[mask]).kneighbors()[0][:, -1]
        radius[mask] = np.nextafter(r, 0)
        k_all[mask] = kk
        counts[mask] = count
    m_all = KDTree(X).query_radius(X, radius, count_only=True)
    return (digamma(n) + np.mean(digamma(k_all))
            - np.mean(digamma(counts)) - np.mean(digamma(m_all)))


def entropy(y):
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return -np.sum(p * np.log(p))


design = ["Promoter", "Threshold", "Activation"]
condition_key = data[design].astype(str).agg("_".join, axis=1).values


def stratified_permutations(col, y, n_perm):
    others = [c for c in design if c != col]
    strata = data[others].astype(str).agg("_".join, axis=1).values
    blocks = []
    for s_key in np.unique(strata):
        idx = np.where(strata == s_key)[0]
        members = [idx[condition_key[idx] == c] for c in np.unique(condition_key[idx])]
        blocks.append((members, np.array([y[m[0]] for m in members])))
    for _ in range(n_perm):
        y_perm = np.empty_like(y)
        for members, labels in blocks:
            for m, label in zip(members, rng.permutation(labels)):
                y_perm[m] = label
        yield y_perm


mi_results = {}
mi_rows = []
for target_col in design:
    y = np.unique(data[target_col].values, return_inverse=True)[1]   # one category per sampled value
    perms = list(stratified_permutations(target_col, y, MI_N_PERM))
    H_y = entropy(y)
    mi_results[target_col] = {}
    for name, X in readouts.items():
        mi = mi_knn(X, y)
        null = np.array([mi_knn(X, y_perm) for y_perm in perms])
        share = (mi - null.mean()) / H_y
        p_value = (1 + np.sum(null >= mi)) / (1 + MI_N_PERM)
        mi_results[target_col][name] = {"share": share, "P": p_value}
        mi_rows.append({
            "parameter": target_col, "readout": name, "k": MI_K,
            "MI_bits": mi / np.log(2), "null_MI_bits": null.mean() / np.log(2),
            "H_parameter_bits": H_y / np.log(2), "share_percent": 100 * share,
            "P_stratified": p_value, "n_permutations": MI_N_PERM, "n_points": len(y),
        })
mi_table_path = os.path.join(base_dir, f"mutual_information_knn_k{MI_K}.csv")
pd.DataFrame(mi_rows).to_csv(mi_table_path, index=False, float_format="%.4g")


def format_p(p):
    return f"<{1 / (1 + MI_N_PERM):.0e}" if p <= 1 / (1 + MI_N_PERM) else f"{p:.3g}"


print(f"Mutual information, map location vs design parameter "
      f"(k={MI_K}, {MI_N_PERM} stratified permutations; share of the parameter's entropy):")
for target_col in design:
    r = mi_results[target_col]
    print(f"  {target_col}: " + ", ".join(
        f"{name} {100 * r[name]['share']:.1f}% (P{'' if format_p(r[name]['P']).startswith('<') else '='}"
        f"{format_p(r[name]['P'])})" for name in readouts))
print(f"Table saved as: {mi_table_path}")

# Adjusted MI between the binned readouts and the simulated 5-percentile distance
# (quantile bins: each bin holds about the same number of points)
n_bins = 10


def quantile_codes(values, n_bins):
    edges = np.unique(np.nanquantile(values, np.linspace(0, 1, n_bins + 1))[1:-1])
    return np.digitize(values, edges)


S5P_codes = quantile_codes(S5P_norm, n_bins)
S2P_codes = quantile_codes(S2P_norm, n_bins)
joint_codes = S5P_codes * (S2P_codes.max() + 1) + S2P_codes
sim_target_codes = quantile_codes(data["5percentRP"].values, n_bins)
mi_sim_distance_joint = adjusted_mutual_info_score(joint_codes, sim_target_codes)
mi_sim_distance_s5p = adjusted_mutual_info_score(S5P_codes, sim_target_codes)
mi_sim_distance_s2p = adjusted_mutual_info_score(S2P_codes, sim_target_codes)
print(f"Adjusted MI with the simulated 5%ile distance: S5P={mi_sim_distance_s5p:.4f}, "
      f"S2P={mi_sim_distance_s2p:.4f}, S5P+S2P={mi_sim_distance_joint:.4f}")


# Plot each variable in its own subplot
variable_specs = [
    ("SE-Promoter 5-perc. dist. [nm]", Z_RPdist, "linear"),
    ("Promoter length [monomers]", Z_promoter, "linear"),
    ("Threshold [no. activators]", Z_threshold, "linear"),
    ("Activation [1/min]", Z_activation, "log"),
]

# collect MI annotations during plotting and draw them after the legend
mi_annotations = []  # list of tuples (ax_index, text)

for ax_index, (ax, (title, Z_values, scale)) in enumerate(zip(axes, variable_specs)):
    plot_vals = Z_values[np.isfinite(Z_values)]
    if plot_vals.size == 0:
        continue

    manual_vmin, manual_vmax = manual_cmap_limits.get(title, (None, None))

    if scale == "log":
        positive_vals = plot_vals[plot_vals > 0]
        vmin = max(manual_vmin if manual_vmin is not None else np.nanmin(positive_vals) if positive_vals.size else 1e-6, 1e-6)
        vmax = manual_vmax if manual_vmax is not None else np.nanmax(plot_vals)
        if np.isclose(vmin, vmax):
            vmax = vmin * 10
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = manual_vmin if manual_vmin is not None else np.nanmin(plot_vals)
        vmax = manual_vmax if manual_vmax is not None else np.nanmax(plot_vals)
        if np.isclose(vmin, vmax):
            vmin -= 0.5
            vmax += 0.5
        norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = parula_cmap.reversed() if title == "SE-Promoter 5-perc. dist. [nm]" else parula_cmap

    im = ax.imshow(
        Z_values,
        extent=(support_S5P.min(), support_S5P.max(), support_S2P.min(), support_S2P.max()),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        alpha=1.0,
    )

    points = np.column_stack((S5P_vals, S2P_vals))
    hull = ConvexHull(points)
    hull_points = np.append(hull.vertices, hull.vertices[0]) 
    ax.plot(
        points[hull_points, 0],
        points[hull_points, 1],
        linestyle="-",
        color="black",
        linewidth=0.75,
        alpha=0.6,
        label="Simulation\ncoverage")

    if not args.bare:
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Pol II Ser5P")
        ax.set_ylabel("Active fraction (Pol II Ser2P)")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='y', which='minor', labelleft=False)

    # add mutual information annotation for the parameter-specific subplots
    mi_key = None
    if title == "Promoter length [monomers]":
        mi_key = "Promoter"
    elif title == "Threshold [no. activators]":
        mi_key = "Threshold"
    elif title == "Activation [1/min]":
        mi_key = "Activation"

    if mi_key is not None:
        info = mi_results[mi_key]["S5P+S2P"]
        mi_text = f"MI(S5P,S2P)\n{100 * info['share']:.1f}% of H"
        # store annotation text for later; we'll draw boxes matching subplot 1's legend
        mi_annotations.append((ax_index, mi_text))

    cbar = plt.colorbar(im, ax=ax)
    if title == "Promoter length [monomers]":
        cbar.set_ticks(np.arange(1.0, 3.01, 0.5))
    cbar.set_label("")


# --------------------------
# Comparison to microscopy data
# --------------------------
# --- Reading in additional microscopy data for matching ---
microscopy_data_path = microscopy_path
microscopy_data = pd.read_csv(microscopy_data_path)
microscopy_data.columns = ["stage_gene", "five_percent_distance_nm", "ser5p_level", "ser2p_level"]

# Convert microscopy measurements to numeric values for interpolation-style use
for col in ["five_percent_distance_nm", "ser5p_level", "ser2p_level"]:
    microscopy_data[col] = pd.to_numeric(microscopy_data[col], errors="coerce")

# Keep them as plain arrays, similar to the raw data points used for the interpolation
microscopy_S5P = microscopy_data["ser5p_level"].to_numpy(dtype=float)
microscopy_S2P = microscopy_data["ser2p_level"].to_numpy(dtype=float)
microscopy_5PrctDist = 1000 * microscopy_data["five_percent_distance_nm"].to_numpy(dtype=float)
# Multiply by 1000 to convert from micrometers to nanometers, matching the units of the simulation-derived 5-percentile-distance field.

# Fit an affine transform for the microscopy coordinates so they align with the
# simulation-derived 5-percentile-distance field using a nearest-neighbor loss.
sim_S5P = S5P_vals.astype(float)
sim_S2P = S2P_vals.astype(float)
sim_values = data["5percentRP"].values.astype(float)

valid = np.isfinite(microscopy_S5P) & np.isfinite(microscopy_S2P) & np.isfinite(microscopy_5PrctDist)
mic_x = microscopy_S5P[valid]
mic_y = microscopy_S2P[valid]
mic_values = microscopy_5PrctDist[valid]

# ---
# z-score transformation of microscopy coordinates to match the simulation distribution

trafo_mic_S5P = microscopy_S5P.copy()
trafo_mic_S5P = (trafo_mic_S5P-trafo_mic_S5P.mean())/trafo_mic_S5P.std()
trafo_mic_S5P = trafo_mic_S5P * S5P_vals.std() + S5P_vals.mean()

trafo_mic_S2P = microscopy_S2P.copy()
trafo_mic_S2P = (trafo_mic_S2P-trafo_mic_S2P.mean())/trafo_mic_S2P.std()
trafo_mic_S2P = trafo_mic_S2P * S2P_vals.std() + S2P_vals.mean()

trafo_mic_5PrctDist = microscopy_5PrctDist.copy()
trafo_mic_5PrctDist = (trafo_mic_5PrctDist-trafo_mic_5PrctDist.mean())/trafo_mic_5PrctDist.std()
trafo_mic_5PrctDist = trafo_mic_5PrctDist * sim_values.std() + sim_values.mean()

preds = np.zeros_like(trafo_mic_5PrctDist)
for i, (xt, yt, zt) in enumerate(zip(trafo_mic_S5P, trafo_mic_S2P, trafo_mic_5PrctDist)):
    distances = np.sqrt((sim_S5P - xt) ** 2 + (sim_S2P - yt) ** 2)
    idx = np.argsort(distances)[:N_neighbors]
    nn_distances = distances[idx]
    weights = averaging_KK / (averaging_KK + nn_distances)
    weights = weights / np.sum(weights)
    preds[i] = np.sum(weights * sim_values[idx])


r, p = stats.pearsonr(preds, trafo_mic_5PrctDist)
print(f"Microscopy comparison: r={r:.4f}, P={p:.3e}, n={len(trafo_mic_5PrctDist)} "
      f"(nearest neighbours in unstandardized Ser5P/Ser2P coordinates)")
p_str = f"{p:.0e}"
if "e" in p_str:
    mantissa, exp = p_str.split("e")
    mantissa_f = float(mantissa)
    if mantissa_f.is_integer():
        mantissa_fmt = f"{int(mantissa_f)}"
    else:
        mantissa_fmt = f"{mantissa_f}"
    p_label = rf"$P={mantissa_fmt}\times10^{{{int(exp)}}}$"
else:
    p_label = rf"$P={p:.3g}$"

# Overlay the transformed microscopy coordinates onto the first subplot
first_ax = axes[0]
first_panel_vmin, first_panel_vmax = manual_cmap_limits.get("SE-P 5% dist [nm]")

first_panel_norm = Normalize(vmin=first_panel_vmin, vmax=first_panel_vmax)
first_ax.scatter(
    trafo_mic_S5P,
    trafo_mic_S2P,
    c=trafo_mic_5PrctDist,
    cmap=parula_cmap.reversed(),
    norm=first_panel_norm,
    s=15,
    edgecolors="black",
    linewidths=0.5,
    alpha=0.9,
        label=f"Microscopy\nr={r:.3f}\n{p_label}",
)

leg = first_ax.legend(loc="upper right", frameon=True, fontsize=8)
if args.bare:
    # keep the box and the handles, drop only the lettering
    for _t in leg.get_texts():
        _t.set_color("none")
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('none')
frame.set_alpha(0.9)


# draw the figure to get accurate renderer measurements, then align MI boxes to the legend bbox
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
leg_bbox_disp = leg.get_window_extent(renderer=renderer)
leg_bbox_fig = leg_bbox_disp.transformed(fig.transFigure.inverted())
leg_left, leg_right = leg_bbox_fig.x0, leg_bbox_fig.x1
leg_top, leg_height = leg_bbox_fig.y1, leg_bbox_fig.y1 - leg_bbox_fig.y0

# draw matching white boxes and left-justified text for stored MI annotations
for ax_idx, text in mi_annotations:
    # skip the first subplot since it already contains the boxed legend
    if ax_idx == 0:
        continue
    ax = axes[ax_idx]
    ax_pos = ax.get_position()
    # compute left/right in axis-relative coords so the rectangle lives inside the axis
    left_ax = (leg_left - ax_pos.x0) / ax_pos.width
    width_ax = (leg_right - leg_left) / ax_pos.width
    # clamp to axis bounds
    left_ax = max(0.01, min(left_ax, 0.95))
    width_ax = max(0.1, min(width_ax, 0.94 - left_ax))
    # height relative to axis (convert legend height from figure coords)
    height_ax = leg_height / ax_pos.height
    # place the rectangle near the top of the axis
    rect_y_ax = 1.0 - height_ax - 0.02
    if rect_y_ax < 0.01:
        rect_y_ax = 0.01
    # compute horizontal intersection between legend span and this axis (figure coords)
    pad = 0.005
    rect_left = max(leg_left, ax_pos.x0 + pad)
    rect_right = min(leg_right, ax_pos.x1 - pad)
    # extend left edge a bit for subplots 2-4 so box is slightly wider
    extra_left = 0.1
    if ax_idx != 0:
        rect_left = max(ax_pos.x0 + pad, rect_left - extra_left)

    # fallback width if intersection is too small or non-existent
    min_width = 0.06
    if rect_right - rect_left < min_width:
        rect_width_fig = min_width
        # try to place the box near the right edge of the axis (but inside)
        rect_left = max(ax_pos.x1 - rect_width_fig - pad, ax_pos.x0 + pad)
    else:
        rect_width_fig = rect_right - rect_left

    # choose a rectangle height large enough to cover the text
    rect_height_fig = max(min(leg_height * 0.9, 0.12), 0.07)
    # ensure rectangle height isn't larger than the axis height
    rect_height_fig = min(rect_height_fig, ax_pos.height - 0.01)

    # vertical position: just inside the top of the axis
    rect_y_fig = ax_pos.y1 - rect_height_fig - 0.01
    if rect_y_fig < ax_pos.y0 + 0.005:
        rect_y_fig = ax_pos.y0 + 0.005

    # create rectangle in figure coordinates and clip to the axis patch so it only appears inside the axis
    rect = Rectangle((rect_left, rect_y_fig), rect_width_fig, rect_height_fig,
                     transform=fig.transFigure, facecolor='white', alpha=0.95, edgecolor='none', zorder=4)
    fig.patches.append(rect)
    rect.set_clip_path(ax.patch)

    # place the text inside the rectangle (figure coords) and clip it to the axis as well
    text_x_fig = rect_left + 0.03 * rect_width_fig
    text_y_fig = rect_y_fig + rect_height_fig - 0.12 * rect_height_fig
    t = fig.text(text_x_fig, text_y_fig, text, transform=fig.transFigure, ha='left', va='top', fontsize=8,
                 family='Arial', fontweight='regular', zorder=5,
                 color=("none" if args.bare else "black"))
    t.set_clip_path(ax.patch)


if args.bare:
    # The legend box is laid out from the label text, so its size depends on the
    # available font. Hide it here and keep the figure's existing legend, rather
    # than shifting it against the labels placed on top. The MI boxes above were
    # already measured against it.
    leg.set_visible(False)

# --- saving to file ---
# Save the plot as an SVG file
plt.savefig(os.path.join(base_dir, "shoebox_colormaps.svg"), format="svg")
# Also save a PNG preview for quick viewing
try:
    plt.savefig(os.path.join(base_dir, "shoebox_colormaps_preview.png"), format="png", dpi=150)
except Exception:
    pass
plt.close(fig)


# ============================================================
# SECOND FIGURE: raw-data scatter maps
# Top row    : 5-percentile enhancer-promoter / enhancer-gene distance
# Bottom row : promoter length (linear), threshold (linear),
#              activation rate (log)
# Reproduces the point-cloud scatter style of new_maps_5percentile.py,
# new_maps_FigA.py, new_maps_FigB_log_parula.py and
# new_maps_promoter_parula.py, using the same `data` table, the same
# manual_cmap_limits, and the same parula colormap already defined
# above for consistency with the first figure. The x/y coordinates use
# the same corrected Pol II Ser5P value and active-fraction Pol II
# Ser2P value (S5P_vals / S2P_vals) computed for Fig. 1, so both
# figures share one coordinate system.
# ============================================================

scatter_x = S5P_vals
scatter_y = S2P_vals

# Font spec for Fig. 2 only (kept local so Fig. 1, already drawn and
# saved above, is unaffected).
FIG2_FONT = "Arial"
FIG2_PANEL_LETTER_SIZE = 12
FIG2_LABEL_SIZE = 10
FIG2_TICK_SIZE = 8

fig2 = plt.figure(figsize=(10, 6.6))
gs2 = fig2.add_gridspec(
    nrows=2, ncols=6,
    wspace=1.0, hspace=0.6,
    left=0.06, right=0.96, top=0.94, bottom=0.07,
)

ax_rp = fig2.add_subplot(gs2[0, 0:2])
ax_rg = fig2.add_subplot(gs2[0, 2:4])
ax_promoter = fig2.add_subplot(gs2[1, 0:2])
ax_threshold = fig2.add_subplot(gs2[1, 2:4])
ax_activation = fig2.add_subplot(gs2[1, 4:6])


def _style_scatter_axis(ax):
    # Light gray background improves contrast for bright-yellow points at
    # the top of the parula colormap, which are otherwise hard to see
    # against pure white.
    ax.set_facecolor("0.88")
    ax.set_xlabel("Pol II Ser5P", fontsize=FIG2_LABEL_SIZE, fontfamily=FIG2_FONT, fontweight="regular")
    ax.set_ylabel("Active fraction (Pol II Ser2P)", fontsize=FIG2_LABEL_SIZE, fontfamily=FIG2_FONT, fontweight="regular")
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_min, y_plot_max)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="y", which="minor", labelleft=False)
    ax.tick_params(axis="both", which="major", labelsize=FIG2_TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FIG2_FONT)


# (panel letter, axis, data column, panel title, (vmin, vmax), color scale)
scatter_panels = [
    ("A", ax_rp, "5percentRP", "5-percentile enhancer-promoter distance [nm]",
     manual_cmap_limits["SE-Promoter 5-perc. dist. [nm]"], "linear"),
    ("B", ax_rg, "5percentRG", "5-percentile enhancer-gene distance [nm]",
     manual_cmap_limits["SE-Promoter 5-perc. dist. [nm]"], "linear"),
    ("C", ax_promoter, "Promoter", "Promoter length [monomers]",
     manual_cmap_limits["Promoter length [monomers]"], "linear"),
    ("D", ax_threshold, "Threshold", "Activation threshold [no. activators]",
     manual_cmap_limits["Threshold [no. activators]"], "linear"),
    ("E", ax_activation, "Activation", "Activation rate [1/min]",
     manual_cmap_limits["Activation [1/min]"], "log"),
]

for letter, ax, col, title, (vmin, vmax), scale in scatter_panels:
    c_vals = data[col].values
    norm = LogNorm(vmin=vmin, vmax=vmax) if scale == "log" else Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        scatter_x, scatter_y, c=c_vals,
        cmap=parula_cmap, norm=norm,
        edgecolor="none", s=6,
    )
    _style_scatter_axis(ax)
    ax.set_title(title, fontsize=FIG2_LABEL_SIZE, fontfamily=FIG2_FONT, fontweight="regular")

    # Panel letter, placed just outside the top-left corner of each axis,
    # close enough to its own title that it stays clear of the x-axis
    # label of the panel in the row above.
    ax.text(
        -0.22, 1.12, letter, transform=ax.transAxes,
        fontsize=FIG2_PANEL_LETTER_SIZE, fontfamily=FIG2_FONT, fontweight="bold",
        ha="left", va="bottom",
    )

    cbar = fig2.colorbar(sc, ax=ax)
    if title == "Promoter length [monomers]":
        cbar.set_ticks(np.arange(1.0, 3.01, 0.5))
    elif scale == "log":
        cbar.set_ticks([vmin, 1.0, vmax])
        cbar.set_ticklabels([r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$"])
    cbar.set_label("")
    cbar.ax.tick_params(labelsize=FIG2_TICK_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(FIG2_FONT)

# --- saving to file ---
fig2.savefig(os.path.join(base_dir, "shoebox_scatter_maps.svg"), format="svg")
try:
    fig2.savefig(os.path.join(base_dir, "shoebox_scatter_maps_preview.png"), format="png", dpi=150)
except Exception:
    pass
plt.close(fig2)