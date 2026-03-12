#This script uses 3D Gaussian density field and isosurface extraction to produce a smooth droplet-like surface that tightly follows the particle cloud
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib.animation import FFMpegWriter

# Load data
data = np.loadtxt("ser5p_position.txt", delimiter=",", skiprows=1)

timesteps = data[:,0]
x = data[:,2]
y = data[:,3]
z = data[:,4]

# Fixing the range of X, Y, Z axes
# Global coordinate limits for stable movie
global_xmin, global_xmax = x.min(), x.max()
global_ymin, global_ymax = y.min(), y.max()
global_zmin, global_zmax = z.min(), z.max()

frame = np.max(timesteps)

mask = timesteps == frame

points = np.column_stack((x[mask], y[mask], z[mask]))

print("Particles in frame:", len(points))

# ---- Create density grid ----

grid_size = 50

xmin,xmax = points[:,0].min(), points[:,0].max()
ymin,ymax = points[:,1].min(), points[:,1].max()
zmin,zmax = points[:,2].min(), points[:,2].max()

H, edges = np.histogramdd(
    points,
    bins=grid_size,
    range=[[xmin,xmax],[ymin,ymax],[zmin,zmax]]
)

# Smooth density
density = gaussian_filter(H, sigma=2)

# ---- Extract smooth surface ----

verts, faces, normals, values = marching_cubes(density, level=np.max(density)*0.2)

# Rescale vertices to real coordinates
scale = np.array([
    (xmax-xmin)/grid_size,
    (ymax-ymin)/grid_size,
    (zmax-zmin)/grid_size
])

verts = verts*scale + np.array([xmin,ymin,zmin])

# ---- Plot ----

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')

# particles
ax.scatter(points[:,0], points[:,1], points[:,2], s=20)

# smooth shell
ax.plot_trisurf(
    verts[:,0],
    verts[:,1],
    faces,
    verts[:,2],
    alpha=0.1,
    color="red"
)

#Parameter controlling isosurface opacity:
#alpha = 1.0   → completely opaque
#alpha = 0.5   → semi-transparent
#alpha = 0.1   → very transparent
#alpha = 0.0   → invisible

ax.set_title(f"Smooth condensate surface (step {int(frame)})")

plt.savefig(f"smooth_condensate_{int(frame)}.jpg", dpi=300)
plt.show()

# ------------------------------------------------------------
# GENERATE MOVIE OF CONDENSATE EVOLUTION OVER TIME
# ------------------------------------------------------------

from matplotlib.animation import FFMpegWriter

# Load timesteps where gene is active
#try:
    #active_data = np.loadtxt("dist_active.txt")
    #active_frames = set(active_data[:,0].astype(int))
#except:
    #active_frames = set()

# ------------------------------------------------------------
# Load timesteps where gene is active (dist_active_run*.txt)
# ------------------------------------------------------------

active_frames = set()
active_file = None

for f in os.listdir():
    if f.startswith("dist_active_run") and f.endswith(".txt"):
        active_file = f
        break

if active_file is None:
    print("Warning: dist_active_run*.txt not found. All frames will be red.")
else:
    active_data = np.loadtxt(active_file)

    # handle both 1-column and multi-column files
    if active_data.ndim == 1:
        active_frames = set(active_data.astype(int))
    else:
        active_frames = set(active_data[:,0].astype(int))

    print("Using activation file:", active_file)
    print("Number of active frames:", len(active_frames))    
    
     

unique_steps = np.unique(timesteps)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

writer = FFMpegWriter(fps=15)

with writer.saving(fig, "condensate_evolution.mp4", dpi=150):

    for frame_i in unique_steps:

        print(f"Generating 3D movie of cluster evolution ... Time step: {int(frame_i)}")

        ax.cla()

        mask = timesteps == frame_i
        points = np.column_stack((x[mask], y[mask], z[mask]))

        if len(points) < 5:
            continue

        xmin,xmax = points[:,0].min(), points[:,0].max()
        ymin,ymax = points[:,1].min(), points[:,1].max()
        zmin,zmax = points[:,2].min(), points[:,2].max()

        H,_ = np.histogramdd(
            points,
            bins=grid_size,
            range=[[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        )

        density = gaussian_filter(H, sigma=2)

        try:
            verts, faces, normals, values = marching_cubes(
                density,
                level=np.max(density)*0.2
            )
        except:
            continue

        scale = np.array([
            (xmax-xmin)/grid_size,
            (ymax-ymin)/grid_size,
            (zmax-zmin)/grid_size
        ])

        verts = verts*scale + np.array([xmin,ymin,zmin])

        ax.set_xlim(global_xmin, global_xmax)
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_zlim(global_zmin, global_zmax)

        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1,1,1])

        ax.scatter(points[:,0], points[:,1], points[:,2], s=15)

        if int(frame_i) in active_frames:
            shell_color = "gray"
        else:
            shell_color = "red"

        ax.plot_trisurf(
            verts[:,0],
            verts[:,1],
            faces,
            verts[:,2],
            alpha=0.1,
            color=shell_color,
            linewidth=0
        )

        ax.set_title(f"Step {int(frame_i)}")

        writer.grab_frame()

plt.close(fig)

print("Movie saved as condensate_evolution.mp4")
exit()
