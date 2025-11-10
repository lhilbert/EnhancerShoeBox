import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d

# Define Parula colormap manually
parula_cm_data = np.array([
    #[0.2081, 0.1663, 0.5292],
    #[0.1906, 0.4075, 0.5565],
    #[0.2068, 0.5791, 0.5555],
    #[0.3692, 0.7883, 0.3827],
    #[0.7040, 0.8753, 0.1178],
    #[0.9932, 0.9062, 0.1439]
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
    #f = interp1d(x_old, parula_cm_data[:, i], kind="cubic")
    f = interp1d(x_old, parula_cm_data[:, i], kind="cubic", bounds_error=False, fill_value="extrapolate")
    parula_interp[:, i] = f(x_new)
parula_cmap = ListedColormap(parula_interp)

# --------------------------
# Load the data
# --------------------------
data = pd.read_csv("summary_contact_grouped_Thresholds10-100.txt")
data.columns = [
    "Order", "Promoter", "Threshold", "Activation",
    "S5PInt", "S2PInt", "Contact", "DistActivation"
]

# Convert numeric columns
data["Threshold"] = pd.to_numeric(data["Threshold"], errors="coerce")
data["S5PInt"] = pd.to_numeric(data["S5PInt"], errors="coerce")
data["S2PInt"] = pd.to_numeric(data["S2PInt"], errors="coerce")

# --------------------------
# Choose colormap (prefer parula)
# --------------------------
#cmap = cm.get_cmap("parula") if "parula" in plt.colormaps() else cm.get_cmap("viridis")
cmap = parula_cmap

# --------------------------
# Create grid & background field
# --------------------------
x = np.linspace(data["S5PInt"].min(), data["S5PInt"].max(), 300)
y = np.linspace(data["S2PInt"].min(), data["S2PInt"].max(), 300)
X, Y = np.meshgrid(x, y)

# Interpolate Threshold values onto grid
Z = griddata(
    points=(data["S5PInt"], data["S2PInt"]),
    values=data["Threshold"],
    xi=(X, Y),
    method="nearest",
)

# Blur to soften discrete edges
Z = gaussian_filter(Z, sigma=5)   # increase sigma for smoother background

# --------------------------
# Plot
# --------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# Background shading only (fully opaque)
im = ax.imshow(
    Z,
    extent=(x.min(), x.max(), y.min(), y.max()),
    origin="lower",
    aspect="auto",
    cmap=cmap,
    alpha=1.0,  # fully opaque
)

# Labels
ax.set_xlabel("PolII Ser5P")
ax.set_ylabel("PolII Ser2P")

# Minor ticks
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.tick_params(axis="x", which="minor", labelbottom=False)
ax.tick_params(axis="y", which="minor", labelleft=False)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Activation threshold")

# Save SVG
plt.savefig("FigA_background_only_parula_x10-100.svg", format="svg")
plt.close(fig)

