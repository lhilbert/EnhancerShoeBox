import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Load cluster particle positions
data = np.loadtxt("ser5p_position.txt", delimiter=",", skiprows=1)

timesteps = data[:,0]
x = data[:,2]
y = data[:,3]
z = data[:,4]

# Choose timestep to visualize
# frame = 100 allows to select the time step for visualization (here eg. timestep = 100)
frame = np.max(timesteps)   # The last frame at the end of the simulation

mask = timesteps == frame

points = np.column_stack((x[mask], y[mask], z[mask]))

print("Particles in frame:", len(points))
#print("Particles in frame:", np.sum(mask))

# Create 3D figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

# Plot Ser5P particles
#ax.scatter(x[mask], y[mask], z[mask], s=20)
ax.scatter(points[:,0], points[:,1], points[:,2], s=20)

# ---- Compute convex hull (condensate surface) ----
if len(points) > 4:

    hull = ConvexHull(points)

    for simplex in hull.simplices:
        triangle = points[simplex]

        ax.plot_trisurf(
            triangle[:,0],
            triangle[:,1],
            triangle[:,2],
            color="red",
            alpha=0.15,
            linewidth=0
        )

#Labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_title(f"Ser5P condensate at step {int(frame)}")

# Save image
plt.savefig(f"cluster_shell_step_{int(frame)}.jpg", dpi=300)

# Show interactive plot
plt.show()

