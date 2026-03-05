#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    # Ask for input
    user_input = input("Enter three values, separated by a single space: promoter length, activation threshold, activation rate\n")

    parts = user_input.strip().split()
    if len(parts) != 3:
        print("Invalid number of entered variables")
        sys.exit(1)

    p, x, a = parts

    # Collect all valid p, x, a values from existing folders
    valid_p = set()
    valid_x = set()
    valid_a = set()

    for folder in os.listdir("."):
        if folder.startswith("Control_Promoter") and os.path.isdir(folder):
            # Expected format: Control_Promoter[p]_Threshold[x]_Act[a]
            try:
                # Remove prefix
                rest = folder.replace("Control_Promoter", "")
                parts = rest.split("_")
                p_val = parts[0]             # like "3"
                x_val = parts[1].replace("Threshold", "")  # like "30"
                a_val = parts[2].replace("Act", "")        # like "75"

                valid_p.add(p_val)
                valid_x.add(x_val)
                valid_a.add(a_val)
            except Exception:
                # Skip folders that don’t match the pattern
                continue

    # Validate entered values
    errors = False
    if p not in valid_p:
        print(f"promoter length: {p}. Error: no such value")
        errors = True
    if x not in valid_x:
        print(f"threshold: {x}. Error: no such value")
        errors = True
    if a not in valid_a:
        print(f"activation rate: {a}. Error: no such value")
        errors = True

    if errors:
        sys.exit(1)

    # Build target folder name
    target_folder = f"Control_Promoter{p}_Threshold{x}_Act{a}"

    gene_file = os.path.join(target_folder, "gene_stats.txt")

    if not os.path.exists(gene_file):
        print(f"Error: file gene_stats.txt not found in {target_folder}")
        sys.exit(1)

    # Load data
    try:
        data = np.loadtxt(gene_file, delimiter=",")
    except Exception as e:
        print(f"Error loading gene_stats.txt: {e}")
        sys.exit(1)
        
    # Extract relevant columns
    # col2 = SE-gene distance [nm] (index 1)
    # col4 = transcription state (index 3): 0 or 1
    distances = data[:, 1]
    states = data[:, 3]

    # Separate values
    dist_not_transcribing = distances[states == 0]
    dist_transcribing = distances[states == 1]

    # Output filenames
    out_not = f"SE-G_dist_nottranscribing_p{p}_x{x}_a{a}.svg"
    out_yes = f"SE-G_dist_transcribing_p{p}_x{x}_a{a}.svg"
    out_kde = f"SE-G_dist_kernel_distribution_p{p}_x{x}_a{a}.svg"

    # Plot histogram for NOT transcribing; to make bins thicker use bins=30
    #plt.figure()
    #plt.hist(dist_not_transcribing, bins=60)
    #plt.xlabel("SE–Gene distance [nm]")
    #plt.ylabel("Count")
    #plt.title("Distance distribution – not transcribing")
    #plt.savefig(out_not)
    #plt.close()

    # Plot histogram for transcribing
    #plt.figure()
    #plt.hist(dist_transcribing, bins=30)
    #plt.xlabel("SE–Gene distance [nm]")
    #plt.ylabel("Count")
    #plt.title("Distance distribution – transcribing")
    #plt.savefig(out_yes)
    #plt.close()
    
    
    # === Determine global axis limits ===
    xmin = min(np.min(dist_not_transcribing), np.min(dist_transcribing))
    xmax = max(np.max(dist_not_transcribing), np.max(dist_transcribing))

    # Calculate histogram heights to get a shared Y limit
    counts_not, bin_edges = np.histogram(dist_not_transcribing, bins=60, range=(xmin, xmax))
    counts_yes, _ = np.histogram(dist_transcribing, bins=60, range=(xmin, xmax))

    ymax = max(np.max(counts_not), np.max(counts_yes))

    # === Function to plot one histogram with Gaussian fit ===
    def plot_histogram(data, title, outfile):
        plt.figure()

        # Histogram
        counts, bins, _ = plt.hist(
            data,
            bins=60,
            range=(xmin, xmax),
            alpha=0.7,
            edgecolor='black'
        )

        # Gaussian trend line
        mu, sigma = np.mean(data), np.std(data)
        xvals = np.linspace(xmin, xmax, 400)
        gaussian = norm.pdf(xvals, mu, sigma)

        # scale Gaussian to histogram height
        #gaussian_scaled = gaussian * (ymax / np.max(gaussian))
        gaussian_scaled = gaussian * (np.max(counts) / np.max(gaussian))

        plt.plot(xvals, gaussian_scaled, color="red", linewidth=2)

        # Labels
        plt.xlabel("SE–Gene distance [nm]")
        plt.ylabel("Count")
        plt.title(title)

        # Shared axis limits
        plt.xlim(xmin, xmax)
        plt.ylim(0, ymax * 1.05)

        plt.savefig(outfile)
        plt.close()

    # === Create the two histograms ===
    plot_histogram(
        dist_not_transcribing,
        "Distance distribution – not transcribing",
        out_not
    )

    plot_histogram(
        dist_transcribing,
        "Distance distribution – transcribing",
        out_yes
    )
    
    # ===============================================================
    # Kernel density distribution for both gene states
    # ===============================================================
    from scipy.stats import gaussian_kde

    # Define x-axis range (shared for both curves)
    xmin = min(np.min(dist_not_transcribing), np.min(dist_transcribing))
    xmax = max(np.max(dist_not_transcribing), np.max(dist_transcribing))
    xvals = np.linspace(xmin, xmax, 500)

    # Kernel density estimation
    #kde_not = gaussian_kde(dist_not_transcribing, bw_method=0.2)   # bandwidth factor (~10–20 nm)
    #kde_yes = gaussian_kde(dist_transcribing, bw_method=0.2)
    kde_not = gaussian_kde(dist_not_transcribing, bw_method=0.1)  #sigma 10 nm
    kde_yes = gaussian_kde(dist_transcribing, bw_method=0.1)

    y_not = kde_not(xvals)
    y_yes = kde_yes(xvals)

    # Plot the combined KDE figure
    plt.figure(figsize=(8, 5))

    plt.plot(xvals, y_not, color="blue", linewidth=2, label="Not transcribing")
    plt.plot(xvals, y_yes, color="grey", linewidth=2, label="Transcribing")

    plt.xlabel("SE–Gene distance [nm]")
    plt.ylabel("Kernel density")
    plt.title("Kernel distribution of enhancer–gene distances")

    plt.legend(loc="upper right")
    plt.xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig(out_kde)
    plt.close()



    print("Histograms saved as:")
    print(" ", out_not)
    print(" ", out_yes)
    print(" ", out_kde)


if __name__ == "__main__":
    main()
