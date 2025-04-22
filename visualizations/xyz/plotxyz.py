import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the files
before = pd.read_csv(r'project\visualizations\positions_before.txt')
after = pd.read_csv(r'project\visualizations\positions_after.txt')

# Ensure both files have same number of points
assert before.shape == after.shape, "Files must have the same number of rows."

# Extract coordinates
x0, y0, z0 = before['x'], before['y'], before['z']
x1, y1, z1 = after['x'], after['y'], after['z']

# Compute deltas
dx = x1 - x0
dy = y1 - y0
dz = z1 - z0
distance = np.sqrt(dx**2 + dy**2 + dz**2)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot before and after points
ax.scatter(x0, y0, z0, c='blue', label='Before')
ax.scatter(x1, y1, z1, c='red', label='After')

# Draw arrows from before to after
for i in range(len(before)):
    ax.quiver(x0[i], y0[i], z0[i], dx[i], dy[i], dz[i], color='gray', arrow_length_ratio=0.1)

# Axis labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Movement of Points (Before → After)')
ax.legend()

plt.tight_layout()
plt.show()
