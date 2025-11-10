import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull

# ===============================================================
# Define Parula colormap manually (interpolated to 256 levels)
# ===============================================================
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

# ===============================================================
# Load and prepare data
# ===============================================================
data = pd.read_csv("summary_contact_grouped_Thresholds10-100.txt")
data.columns = [
    "Order", "Promoter", "Threshold", "Activation",
    "S5PInt", "S2PInt", "Contact", "DistActivation"
]

# Convert numeric columns
data["Threshold"] = pd.to_numeric(data["Threshold"], errors="coerce")
data["S5PInt"] = pd.to_numeric(data["S5PInt"], errors="coerce")
data["S2PInt"] = pd.to_numeric(data["S2PInt"], errors="coerce")

# Drop NaNs for safety
data = data.dropna(subset=["Threshold", "S5PInt", "S2PInt"])

# ===============================================================
# Compute Z-scores (used only internally for interpolation)
# ===============================================================
mean_S5P = data["S5PInt"].mean()
std_S5P = data["S5PInt"].std()
mean_S2P = data["S2PInt"].mean()
std_S2P = data["S2PInt"].std()

data["S5PInt_z"] = (data["S5PInt"] - mean_S5P) / std_S5P
data["S2PInt_z"] = (data["S2PInt"] - mean_S2P) / std_S2P

# ===============================================================
# Define neighbor-weighted background generation function
# ===============================================================
def create_background_only(df, output_file, N_neighbors=20, averaging_KK=0.1):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Input data arrays
    S5P_vals = df["S5PInt"].values
    S2P_vals = df["S2PInt"].values
    promoter_vals = df["Threshold"].values

    # Compute Z-scores for neighbor search
    coords = np.column_stack([
        (S5P_vals - S5P_vals.mean()) / S5P_vals.std(),
        (S2P_vals - S2P_vals.mean()) / S2P_vals.std()
    ])

    # Fit neighbor search model
    nbrs = NearestNeighbors(n_neighbors=N_neighbors).fit(coords)

    # Create grid in original scale
    x = np.linspace(S5P_vals.min(), S5P_vals.max(), 300)
    y = np.linspace(S2P_vals.min(), S2P_vals.max(), 300)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Compute weighted average over nearest neighbors
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Normalize grid coordinates to Z-score space
            gx = (X[i, j] - S5P_vals.mean()) / S5P_vals.std()
            gy = (Y[i, j] - S2P_vals.mean()) / S2P_vals.std()

            # Find nearest neighbors
            distances, indices = nbrs.kneighbors([[gx, gy]])

            # Distance-weighted average of Threshold values
            grid_dist_vec = distances[0] ** 2
            weights = averaging_KK / (averaging_KK + grid_dist_vec)
            Z[i, j] = np.sum(weights * promoter_vals[indices[0]]) / np.sum(weights)

    # Plot
    im = ax.imshow(
        Z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="auto",
        cmap=parula_cmap,
    )

    ax.set_xlabel("PolII Ser5P")
    ax.set_ylabel("PolII Ser2P")

    # Minor ticks (adjust for raw data scale)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.tick_params(axis="y", which="minor", labelleft=False)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Activation threshold")
    
    # --------------------------
    # Add dashed convex hull outline around actual data points
    # --------------------------
    points = np.column_stack((S5P_vals, S2P_vals))
    hull = ConvexHull(points)
    hull_points = np.append(hull.vertices, hull.vertices[0])  # close the loop
    ax.plot(
        points[hull_points, 0],
        points[hull_points, 1],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Data boundary"
)

# Optional: add legend for clarity
#ax.legend(loc="upper right", frameon=False)

    # Save output
    plt.savefig(output_file, format="svg")
    plt.close(fig)

# ===============================================================
# Run the background generation
# ===============================================================
create_background_only(data, "FigA_background_only_parula_zscore_originalscale_neighbors_x10-100_outline.svg")

