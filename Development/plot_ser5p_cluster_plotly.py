#This script uses 3D Gaussian density field and isosurface extraction to produce a smooth droplet-like surface that tightly follows the particle cloud
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

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
# ---- reduce mesh complexity (10× smaller file) ----
#faces = faces[::5]

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
# INTERACTIVE 3D CONDENSATE EVOLUTION (SMOOTH SURFACE)
# ------------------------------------------------------------

import plotly.graph_objects as go

active_file = None

try:
    # Find the file dist_active_run*.txt
    for f in os.listdir():
        if f.startswith("dist_active_run") and f.endswith(".txt"):
            active_file = f
            break

    if active_file is None:
        raise FileNotFoundError("Error: dist_active_run*.txt. No such file")

    active_data = np.loadtxt(active_file)

    # Works for both 1-column and multi-column files
    if active_data.ndim == 1:
        active_frames = set(active_data.astype(int))
    else:
        active_frames = set(active_data[:,0].astype(int))

except Exception as e:
    print(e)
    active_frames = set()
    
print("Activation file:", active_file)
print("Number of active frames:", len(active_frames))

#selecting only every 10th frame for visualization
unique_steps = np.unique(timesteps)[::10]

frames_plotly = []

for frame_i in unique_steps:

    print(f"Preparing interactive frame {int(frame_i)}")

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
    # ---- reduce mesh complexity only for movie ----
    faces_movie = faces[::3]

    # ---- Particle cloud ----
    scatter = go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode="markers",
        marker=dict(
            size=3,
            color="blue",
            opacity=0.6
        )
    )

    # ---- Shell color ----
    if int(frame_i) in active_frames:
        shell_color = "gray"
    else:
        shell_color = "red"

    # ---- Smooth condensate shell ----
    mesh = go.Mesh3d(
        x=verts[:,0],
        y=verts[:,1],
        z=verts[:,2],
        i=faces_movie[:,0],
        j=faces_movie[:,1],
        k=faces_movie[:,2],
        opacity=0.35,
        color=shell_color,
        flatshading=False,
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            roughness=0.4,
            specular=0.1
        )
    )

    frame = go.Frame(
        data=[scatter, mesh],
        name=str(int(frame_i))
    )

    frames_plotly.append(frame)

fig = go.Figure(
    data=frames_plotly[0].data,
    frames=frames_plotly
)

fig.update_layout(
    scene=dict(
        xaxis=dict(range=[global_xmin, global_xmax], autorange=False),    #autorange=false prevents the axes range to adapt to the data range --> stable visualization box
        yaxis=dict(range=[global_ymin, global_ymax], autorange=False),
        zaxis=dict(range=[global_zmin, global_zmax], autorange=False),
        aspectmode="cube"
    ),
    updatemenus=[{
        "type":"buttons",
        "buttons":[
            dict(
                label="Play",
                method="animate",
                args=[None,{
                    "frame":{"duration":80,"redraw":False},
                    "fromcurrent":True,
                    "transition":{"duration":0}
                }]
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None],{
                    "frame":{"duration":0},
                    "mode":"immediate"
                }]
            )
        ]
    }],
    sliders=[{
        "steps":[
            {"args":[[f.name],{"frame":{"duration":0},"mode":"immediate"}],
             "label":f.name,
             "method":"animate"}
            for f in frames_plotly
    ]
}]
    
    
)

fig.write_html("condensate_evolution_interactive.html")

print("Interactive visualization saved as condensate_evolution_interactive.html")
