import numpy as np
import pandas as pd
import os
import argparse
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, Delaunay
from collections import defaultdict

# ==============================
# ARGUMENT PARSING
# ==============================

parser = argparse.ArgumentParser(description="Surface analysis of condensates")
parser.add_argument("-d", "--dir", required=True,
                    help="Run directory (e.g. run44)")
args = parser.parse_args()

run_dir = args.dir

# Extract run number from folder name
run_number = ''.join(filter(str.isdigit, run_dir))

# ==============================
# FILE PATHS
# ==============================

ser5p_file = os.path.join(run_dir, "ser5p_position.txt")

phase_files = {
    "active": os.path.join(run_dir, f"dist_active_run{run_number}.txt"),
    "induced": os.path.join(run_dir, f"dist_induced_run{run_number}.txt"),
    "approaching": os.path.join(run_dir, f"dist_approaching_run{run_number}.txt"),
    "receding": os.path.join(run_dir, f"dist_receding_run{run_number}.txt")
}

output_main = os.path.join(run_dir, f"surface_metrics_run{run_number}.txt")
output_by_state = os.path.join(run_dir, f"surface_by_state_run{run_number}.txt")

# surface_metrics.txt saves per timestep: timestep, gene phase, cluster_size, A_convex, A_alpha, A_sasa, Rg, roughness (A-area)
# output_by_state.txt averages per active, induced, approaching, receding 

# ==============================
# PHYSICAL PARAMETERS
# ==============================

sig_ser5p = 18.0  # nm (diameter)
particle_radius = sig_ser5p / 2

# DBSCAN parameters
eps = 1.2 * sig_ser5p  # 1.3 times PolII diameter; other good eps to try: 1.0 and 1.5
min_samples = 5

# Alpha-shape parameters
alpha = 1.5 * sig_ser5p  # larger alpha --> smoother surface; smaller alpha --> more detailed surface

# SASA probe radius (like solvent probe)
probe_radius = 5.0  # nm

# ==============================
# FUNCTIONS
# ==============================

def load_phase_data(phase_files):
    phase_map = {}
    for phase, file in phase_files.items():
        if not os.path.exists(file):
            continue
        data = np.loadtxt(file)
        for row in data:
            timestep = int(row[0])
            phase_map[timestep] = phase
    return phase_map


def alpha_shape_3D(points, alpha):
    if len(points) < 4:
        return np.array([])

    tetra = Delaunay(points)
    tetrapos = points[tetra.simplices]

    normsq = np.sum(tetrapos**2, axis=2)[:, :, None]
    ones = np.ones((tetrapos.shape[0], tetrapos.shape[1], 1))

    a = np.linalg.det(np.concatenate((tetrapos, ones), axis=2))
    Dx = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [1, 2]], ones), axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 2]], ones), axis=2))
    Dz = np.linalg.det(np.concatenate((normsq, tetrapos[:, :, [0, 1]], ones), axis=2))
    c = np.linalg.det(np.concatenate((normsq, tetrapos), axis=2))

    r = np.sqrt(Dx**2 + Dy**2 + Dz**2 - 4 * a * c) / (2 * np.abs(a))

    tetras = tetra.simplices[r.flatten() < alpha]

    tri_combinations = np.array([(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
    triangles = tetras[:, tri_combinations].reshape(-1,3)
    triangles = np.sort(triangles, axis=1)

    tri_dict = defaultdict(int)
    for tri in triangles:
        tri_dict[tuple(tri)] += 1

    triangles = np.array([tri for tri in tri_dict if tri_dict[tri] == 1])

    return triangles


def triangle_areas(points, triangles):
    if len(triangles) == 0:
        return 0.0
    p0 = points[triangles[:,0]]
    p1 = points[triangles[:,1]]
    p2 = points[triangles[:,2]]

    cross = np.cross(p1 - p0, p2 - p0)
    return np.sum(0.5 * np.linalg.norm(cross, axis=1))


def radius_of_gyration(points):
    center = np.mean(points, axis=0)
    return np.sqrt(np.mean(np.sum((points - center)**2, axis=1)))


def sasa_approx(points, radius, probe):
    N = len(points)
    exposed = 0
    cutoff = 2 * (radius + probe)

    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.sum(dists < cutoff) - 1

        if neighbors < 10:    # SASA threshold, can be tuned further
            exposed += 1

    return exposed * 4 * np.pi * (radius + probe)**2


def estimate_surface_tension(A, V):
    """
    Laplace-like proxy:
    gamma ~ A / V^(2/3)
    (dimensionless, comparative only)
    """
    if V <= 0:
        return 0
    return A / (V ** (2/3))


# ==============================
# LOAD DATA
# ==============================

print(f"Processing directory: {run_dir}")

df = pd.read_csv(ser5p_file, header=0,
                 names=["timestep", "particle_id", "x", "y", "z"])
                 
# Enforce numeric types
df = df.astype({
    "timestep": int,
    "particle_id": int,
    "x": float,
    "y": float,
    "z": float
})

phase_map = load_phase_data(phase_files)

timesteps = sorted(df["timestep"].unique())

results = []

# ==============================
# MAIN LOOP
# ==============================

for ts in timesteps:

    frame = df[df["timestep"] == ts]
    points = frame[["x", "y", "z"]].values

    if len(points) < 10:
        continue

    # ---- DBSCAN clustering----
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    unique_labels = set(labels) - {-1}
    if len(unique_labels) == 0:
        continue
    
    # ---- Select largest cluster ----
    largest_cluster = max(unique_labels,
                          key=lambda l: np.sum(labels == l))

    cluster_points = points[labels == largest_cluster]

    if len(cluster_points) < 4:
        continue

    # ---- Convex hull ----
    hull = ConvexHull(cluster_points)
    A_convex = hull.area
    V = hull.volume

    # ---- Alpha shape ----
    triangles = alpha_shape_3D(cluster_points, alpha)
    A_alpha = triangle_areas(cluster_points, triangles)

    # ---- Rg Radius of gyration----
    Rg = radius_of_gyration(cluster_points)

    # ---- Roughness ----
    roughness = A_alpha / A_convex if A_convex > 0 else 0

    # ---- SASA like----
    A_sasa = sasa_approx(cluster_points, particle_radius, probe_radius)

    # ---- Surface tension proxy ----
    gamma = estimate_surface_tension(A_alpha, V)

    # ---- Phase ----
    phase = phase_map.get(ts, "unknown")

    results.append([
        ts, phase, len(cluster_points),
        A_convex, A_alpha, A_sasa,
        V, Rg, roughness, gamma
    ])

# ==============================
# SAVE OUTPUT
# ==============================

header = ("timestep phase cluster_size "
          "A_convex A_alpha A_sasa "
          "volume Rg roughness gamma")

np.savetxt(output_main, results, fmt="%s", header=header)

print(f"Saved: {output_main}")

# ---- Aggregate by phase ----

df_out = pd.DataFrame(results, columns=[
    "timestep","phase","cluster_size",
    "A_convex","A_alpha","A_sasa",
    "volume","Rg","roughness","gamma"
])

df_state = df_out.groupby("phase").mean(numeric_only=True)

np.savetxt(output_by_state,
           df_state.values,
           fmt="%.6f",
           header=" ".join(df_state.columns))

print(f"Saved: {output_by_state}")

print("Done.")
