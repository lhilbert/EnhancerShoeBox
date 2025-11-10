import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm, ListedColormap
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

# ==============================================================
# Load the data
# ==============================================================
data = pd.read_csv("summary_contact_grouped_Thresholds10-100.txt")
data.columns = ["Order", "Promoter", "Threshold", "Activation",
                "S5PInt", "S2PInt", "Contact", "DistActivation"]

# Convert to numeric (avoid problems with non-numeric rows)
data["Activation"] = pd.to_numeric(data["Activation"], errors="coerce")
data["S5PInt"] = pd.to_numeric(data["S5PInt"], errors="coerce")
data["S2PInt"] = pd.to_numeric(data["S2PInt"], errors="coerce")
data["Promoter"] = pd.to_numeric(data["Promoter"], errors="coerce")

# Drop rows with missing essential values (optional but safe)
data = data.dropna(subset=["Activation", "S5PInt", "S2PInt"])

# ==============================================================
# Compute Z-scores for Ser5P and Ser2P (for distance normalization)
# ==============================================================
S5P_mean, S5P_std = data["S5PInt"].mean(), data["S5PInt"].std(ddof=0)
S2P_mean, S2P_std = data["S2PInt"].mean(), data["S2PInt"].std(ddof=0)

# Guard against zero std
if S5P_std == 0:
    S5P_std = 1.0
if S2P_std == 0:
    S2P_std = 1.0

data["S5PInt_z"] = (data["S5PInt"] - S5P_mean) / S5P_std
data["S2PInt_z"] = (data["S2PInt"] - S2P_mean) / S2P_std

# ==============================================================
# Define Parula colormap manually (interpolated to 256)
# ==============================================================
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
cmap = ListedColormap(parula_interp)

# ==============================================================
# Function: Create background using N-nearest neighbors (Activation values)
# ==============================================================
def create_background_only(df, output_file,
                           N_neighbors=20,
                           averaging_KK=0.1,
                           grid_size=300,
                           vmin=1.0,
                           vmax=100.0):
    """
    Generate a background by distance-weighted averaging of Activation values
    using N-nearest neighbors. Color mapping uses a fixed LogNorm(vmin, vmax)
    so that colors correspond to powers-of-ten consistently (1 -> blue, 10 -> green, 100 -> yellow).
    """
    # Prepare grid in original coordinate system
    x = np.linspace(df["S5PInt"].min(), df["S5PInt"].max(), grid_size)
    y = np.linspace(df["S2PInt"].min(), df["S2PInt"].max(), grid_size)
    X, Y = np.meshgrid(x, y)

    # Arrays of original data
    S5P_vals = df["S5PInt"].values
    S2P_vals = df["S2PInt"].values
    activation_vals = df["Activation"].values

    # Normalize coordinates for distance computation (z-score using sample mean/std)
    coords = np.column_stack([
        (S5P_vals - S5P_vals.mean()) / (S5P_vals.std(ddof=0) if S5P_vals.std(ddof=0) != 0 else 1.0),
        (S2P_vals - S2P_vals.mean()) / (S2P_vals.std(ddof=0) if S2P_vals.std(ddof=0) != 0 else 1.0)
    ])

    # Use a safe neighbor count
    n_neighbors_used = min(int(N_neighbors), max(1, len(coords)))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors_used).fit(coords)

    # Prepare the Z grid
    Z = np.zeros_like(X, dtype=float)

    # Compute weighted averages across the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gx = (X[i, j] - S5P_vals.mean()) / (S5P_vals.std(ddof=0) if S5P_vals.std(ddof=0) != 0 else 1.0)
            gy = (Y[i, j] - S2P_vals.mean()) / (S2P_vals.std(ddof=0) if S2P_vals.std(ddof=0) != 0 else 1.0)

            distances, indices = nbrs.kneighbors([[gx, gy]])
            grid_dist_vec = distances[0] ** 2

            # Avoid division by zero or ill-conditioned situations:
            weights = averaging_KK / (averaging_KK + grid_dist_vec)
            denom = np.sum(weights)
            if denom > 0:
                Z[i, j] = np.sum(weights * activation_vals[indices[0]]) / denom
            else:
                Z[i, j] = np.nan

    # ==============================================================
    # Prepare Z for plotting: clip to [vmin, vmax] 
    # ==============================================================
    # Replace NaNs with vmin so they are shown as the lowest color 
    Z_plot = np.copy(Z)
    Z_plot = np.where(np.isnan(Z_plot), vmin, Z_plot)

    # Clip to the requested display range [vmin, vmax]
    Z_plot = np.clip(Z_plot, vmin, vmax)

    # ==============================================================
    # Plot
    # ==============================================================
    fig, ax = plt.subplots(figsize=(8, 6))

    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(
        Z_plot,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm
    )
    im.set_clim(vmin, vmax)

    # Axis labels
    ax.set_xlabel("PolII Ser5P")
    ax.set_ylabel("PolII Ser2P")

    # Minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='y', which='minor', labelleft=False)

    # Colorbar setup (explicit ticks at 10^0, 10^1, 10^2)
    cbar = plt.colorbar(im, ax=ax)
    ticks = [1, 10, 100]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
    cbar.set_label("Activation rate [1/min]")

    # Save figure
    plt.savefig(output_file, format="svg", bbox_inches="tight")
    plt.close(fig)


# ==============================================================
# Run 
# ==============================================================
create_background_only(
    data,
    output_file="FigB_background_log_parula_zscore_originalscale_neighbors_x10-100.svg",
    N_neighbors=50,
    averaging_KK=0.2,
    grid_size=200,
    vmin=1.0,
    vmax=100.0
)

