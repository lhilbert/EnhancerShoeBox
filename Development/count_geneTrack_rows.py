import os

# Path to the main folder
base_dir = "box11"

# Output file
output_file = "frames_per_run.txt"

with open(output_file, "w") as out:
    for root, dirs, files in os.walk(base_dir):
        # Look for geneTrack.txt
        if "geneTrack.txt" in files:
            # Check if we are in a run folder
            parts = root.split(os.sep)
            if len(parts) >= 3 and parts[-2].startswith("Control_Promoter") and parts[-1].startswith("run"):
                subfolder = parts[-2]
                run_folder = parts[-1]
                gene_file = os.path.join(root, "geneTrack.txt")

                # Count rows in the geneTrack.txt file
                try:
                    with open(gene_file, "r") as f:
                        num_rows = sum(1 for _ in f)
                except Exception as e:
                    num_rows = f"Error: {e}"

                # Write to output file
                out.write(f"{subfolder}: {run_folder}\t{num_rows}\n")

print(f"✅ Done! Results saved in '{output_file}'")

