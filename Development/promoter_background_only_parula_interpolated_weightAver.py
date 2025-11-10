import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors

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
# Load the data
# --------------------------
data = pd.read_csv('summary_contact_grouped_Thresholds10-100.txt')
data["Promoter"] = pd.to_numeric(data["Promoter"], errors="coerce")
data["Activation"] = pd.to_numeric(data["Activation"], errors="coerce")

# --------------------------
# Function to create background only (N-nearest neighbors + weighted averaging)
#Smoothness:
#averaging_KK = 0.1   # moderately smoother
#averaging_KK = 0.2   # noticeably smoother
#averaging_KK = 0.5   # quite soft transitions
#N_neighbors control how many points are averaged
#N_neighbors (e.g., 10–20): local detail is preserved, but can be noisy or patchy.
#N_neighbors (e.g., 50–100): smoother, more continuous background, but fine details are blurred.
# --------------------------
def create_background_only(df, output_file, N_neighbors=20, averaging_KK=0.1):
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Create background grid ---
    #Grid size:
    #300 × 300	High detail
    #200 × 200	40,000	Moderate detail
    #100 × 100	10,000	less detailed
    #75 × 75	5,625	coarse image
    
    x = np.linspace(df["S5PInt"].min(), df["S5PInt"].max(), 300)
    y = np.linspace(df["S2PInt"].min(), df["S2PInt"].max(), 300)
    X, Y = np.meshgrid(x, y)

    S2P_vals = df["S2PInt"].values
    S5P_vals = df["S5PInt"].values
    promoter_vals = df["Promoter"].values

    # Normalize coordinates for fair distance calculation
    coords = np.column_stack([
        (S5P_vals - S5P_vals.mean()) / S5P_vals.std(),
        (S2P_vals - S2P_vals.mean()) / S2P_vals.std()
    ])

    # Fit neighbor search
    nbrs = NearestNeighbors(n_neighbors=N_neighbors).fit(coords)

    # Prepare grid values
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # normalize grid coords
            gx = (X[i, j] - S5P_vals.mean()) / S5P_vals.std()
            gy = (Y[i, j] - S2P_vals.mean()) / S2P_vals.std()

            # find nearest neighbors
            distances, indices = nbrs.kneighbors([[gx, gy]])

            # weighted average of their promoter values
            grid_dist_vec = distances[0] ** 2
            weights = averaging_KK / (averaging_KK + grid_dist_vec)
            Z[i, j] = np.sum(weights * promoter_vals[indices[0]]) / np.sum(weights)
            
    # increase the value of averaging_KK for smoother blur; smaller averaging_KK makes sharper transitions

    # --- Plot background ---
    im = ax.imshow(
        Z,
        extent=(x.min(), x.max(), y.min(), y.max()),
        origin="lower",
        aspect="auto",
        cmap=parula_cmap,
        alpha=1.0
    )

    # --- Axis labels ---
    ax.set_xlabel("PolII Ser5P")
    ax.set_ylabel("PolII Ser2P")

    # Minor ticks
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='y', which='minor', labelleft=False)

    # --- Colorbar ---
    cbar = plt.colorbar(im, ax=ax)
    im.set_clim(1, 3)  # enforce full scale
    cbar.set_ticks([1, 1.5, 2, 2.5, 3])
    cbar.set_label("Promoter length")

    # Save
    plt.savefig(output_file, format="svg")
    plt.close(fig)

# --------------------------
# Run
# --------------------------
create_background_only(data, "promoter_background_only_parula_interpolated_neighbors_x10-100.svg")

