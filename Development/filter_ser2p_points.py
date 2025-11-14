#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Input and output file names
input_file = "summary_contact_grouped_Thresholds10-100_test.txt"
output_file = "stochastic_ser2p_points_test.txt"

# Read the input file (tab or space delimited)
df = pd.read_csv(input_file, sep=None, engine='python')

# Check if 'S2PInt' column exists
if 'S2PInt' not in df.columns:
    raise ValueError("Input file must contain a 'S2PInt' column.")

# Define threshold
threshold = 0.54054
eps = 1e-3  # tolerance for floating-point comparison

# Compute masks
is_zero = np.isclose(df['S2PInt'], 0, atol=eps)
is_multiple = np.isclose((df['S2PInt'] / threshold) % 1, 0, atol=eps)

# Count values
n_discrete = int((is_zero | is_multiple).sum())
n_random = int(len(df) - n_discrete)

# Filter rows that are NOT multiples of threshold and not zero
filtered_df = df[~(is_zero | is_multiple)]

# Save output
filtered_df.to_csv(output_file, sep='\t', index=False)

# Print summary
print(f"{n_discrete} discrete Ser2P values, {n_random} randomly scattered Ser2P points")


#For summary_contact_grouped_Thresholds10-100.txt, the output reads: "4809 discrete Ser2P values, 1791 randomly scattered Ser2P points"
#For summary_contact_grouped_Thresholds10-100_int.txt, the output reads: "4809 discrete Ser2P values, 1791 randomly scattered Ser2P points"
#For summary_contact_all_Thresholds10-100.txt, the output reads: "31119 discrete Ser2P values, 1881 randomly scattered Ser2P points"
#For summary_contact_grouped_Thresholds10-100_cpp.txt, the output reads: "4743 discrete Ser2P values, 1857 randomly scattered Ser2P points"

