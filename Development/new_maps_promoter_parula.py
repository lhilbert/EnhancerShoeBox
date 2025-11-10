import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator 
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d

# --------------------------
# Define Parula colormap manually
# --------------------------
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
 

# Load the data, specifying that the first row is the header
data = pd.read_csv('summary_contact_all_Thresholds10-100.txt')
#data.columns = ["Order", "Promoter", "Threshold", "Activation", "S5PInt", "S2PInt", "Contact", "DistActivation"]
data.columns = ["Order", "Promoter", "Threshold", "Activation", "Runs", "S5PInt", "S2PInt", "Contact", "DistActivation"]

# Convert 'Promoter', 'Threshold' and 'Activation' columns to numeric type
data["Promoter"] = pd.to_numeric(data["Promoter"], errors="coerce")
data["Activation"] = pd.to_numeric(data["Activation"], errors="coerce")
data["Threshold"] = pd.to_numeric(data["Threshold"], errors="coerce")

# Define function to create a scatter plot and save as SVG
# "s" defines the size of scatter points
def create_scatter_plot(df, title, output_file):
    fig, ax = plt.subplots(figsize=(8, 6))
    #sc = ax.scatter(df["S5PInt"], df["S2PInt"], c=df["Promoter"], cmap="viridis", s=10)
    sc = ax.scatter(df["S5PInt"], df["S2PInt"], c=df["Promoter"], cmap=parula_cmap, s=0.1)

    # Set axis labels
    ax.set_xlabel("PolII Ser5P")
    ax.set_ylabel("PolII Ser2P")
    
    # Remove tick labels from both axes
    #ax.set_xticks([])
    #ax.set_yticks([])
    
    # ----- Add minor ticks -----
    # Every 25 on the X-axis
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    # Every 1 on the Y-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Hide the labels for the minor ticks (keep the tick marks)
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    ax.tick_params(axis='y', which='minor', labelleft=False)


    # Add color bar with the new label for promoter length
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Promoter length")
    cbar.set_ticks([1, 2, 3])  # Show only 1, 2, 3

    
    # Save the plot as an SVG file
    plt.savefig(output_file, format="svg")
    plt.close(fig)  # Close the figure to free memory

# 1. Create a collective plot for all 'Activation' values with the new filename
create_scatter_plot(data, "All Promoter Lengths", "colormaps_promoter_parula_all_x10-100.svg")

# 2. Create individual plots for each unique 'Activation' value
#unique_activations = data["Activation"].unique()
#for activation_value in unique_activations:
    # Filter the data for the current activation value
    #filtered_data = data[data["Activation"] == activation_value]
    #output_filename = f"colormaps_FigA_ActivRate{activation_value}_promoter.svg"
    #create_scatter_plot(filtered_data, f"Activation Rate {activation_value}", output_filename)


