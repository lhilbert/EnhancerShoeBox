# %% [markdown]
# Plot interpolated 5-percentile distance backgrounds using Parula colormap
# (based on N-nearest-neighbor weighted averaging with z-scoring)

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

# --------------------------
# Load input data
# --------------------------
data_file = "summary_contact_all_Thresholds10-100_5percentile.txt"
df = pd.read_csv(data_file)

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

# --------------------------
# Helper function: create background with N-neighbor weighted averaging
# --------------------------
def create_background(df, z_column, cbar_label, output_file,
                      N_neighbors=20, averaging_KK=0.1, grid_size=300):
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Create background grid ---
    x = np.linspace(df["S5PInt"].min(), df["S5PInt"].max(), grid_size)
    y = np.linspace(df["S2PInt"].min(), df["S2PInt"].max(), grid_size)
    X, Y = np.meshgrid(x, y)

    # Original values
    S5P_vals = df["S5PInt"].values
    S2P_vals = df["S2PInt"].values
    Z_vals = df[z_column].values  # color values (5percentRG or 5percentRP)

    # Normalize coordinates for distance calculation (z-scoring)
    coords = np.column_stack([
        (S5P_vals - S5P_vals.mean()) / S5P_vals.std(),
        (S2P_vals - S2P_vals.mean()) / S2P_vals.std()
    ])

    # Fit nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=N_neighbors).fit(coords)

    # Prepare grid result
    Z_grid = np.zeros_like(X)

    # Weighted averaging over nearest neighbors
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gx = (X[i, j] - S5P_vals.mean()) / S5P_vals.std()
            gy = (Y[i, j] - S2P_vals.mean()) / S2P_vals.std()
            distances, indices = nbrs.kneighbors([[gx, gy]])

            # Compute weighted average
            grid_dist_vec = distances[0] ** 2
            weights = averaging_KK / (averaging_KK + grid_dist_vec)
            Z_grid[i, j] = np.sum(weights * Z_vals[indices[0]]) / np.sum(weights)

    # --- Plot background ---
    im = ax.imshow(
        Z_grid,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="auto",
        cmap=parula_cmap
    )

    # --- Axis formatting ---
    ax.set_xlabel("PolII Ser5P")
    ax.set_ylabel("PolII Ser2P")
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="both", which="both", direction="in")

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # Save as SVG
    plt.tight_layout()
    plt.savefig(output_file, format="svg", dpi=300)
    plt.close(fig)
    print(f"Saved: {output_file}")

# --------------------------
# 1. Interpolated enhancer–gene distance background
# --------------------------
create_background(
    df=df,
    z_column="5percentRG",
    cbar_label="5-percentile enhancer-gene distance [nm]",
    output_file="5percentile_Reg-gene_background.svg",
    N_neighbors=30,
    averaging_KK=0.15
)

# --------------------------
# 2. Interpolated enhancer–promoter distance background
# --------------------------
create_background(
    df=df,
    z_column="5percentRP",
    cbar_label="5-percentile enhancer-promoter distance [nm]",
    output_file="5percentile_Reg-promoter_background.svg",
    N_neighbors=30,
    averaging_KK=0.15
)

print("All interpolated background figures created successfully.")

