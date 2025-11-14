# %% [markdown]
# Plot 5-percentile distance figures using custom Parula colormap

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d

# --------------------------
# Load input data
# --------------------------
data_file = "summary_contact_grouped_Thresholds10-100_5percentile_20xaveraged.txt"
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
# Helper function to plot scatter graph
# --------------------------
def plot_scatter(x, y, c, cmap, cbar_label, output_name):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=c, cmap=cmap, edgecolor='none', s=6)
    plt.xlabel("PolII Ser5P")
    plt.ylabel("PolII Ser2P")

    # Set tick intervals
    plt.gca().xaxis.set_major_locator(MultipleLocator(50))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(10))

    # Add colorbar on the right
    cbar = plt.colorbar(sc)
    cbar.set_label(cbar_label)

    # Tight layout and save as SVG
    plt.tight_layout()
    plt.savefig(output_name, format='svg', dpi=300)
    plt.close()

# --------------------------
# 1. 5percentile_Reg-gene.svg
# --------------------------
plot_scatter(
    x=df["S5PInt"],
    y=df["S2PInt"],
    c=df["5percentRG"],
    cmap=parula_cmap,
    cbar_label="5-percentile enhancer-gene distance [nm]",
    output_name="5percentile_Reg-gene_20xaveraging.svg"
)

# --------------------------
# 2. 5percentile_Reg-promoter.svg
# --------------------------
plot_scatter(
    x=df["S5PInt"],
    y=df["S2PInt"],
    c=df["5percentRP"],
    cmap=parula_cmap,
    cbar_label="5-percentile enhancer-promoter distance [nm]",
    output_name="5percentile_Reg-promoter_20xaveraging.svg"
)

print("Figures saved: 5percentile_Reg-gene.svg and 5percentile_Reg-promoter.svg")

