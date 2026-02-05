import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from pathlib import Path

rng = np.random.default_rng(2025)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# Create custom colormap from white to theme color
theme_color = COLORS[0]  # "#1B435E"
# Convert hex to RGB (normalized 0-1)
theme_rgb = tuple(int(theme_color[i:i+2], 16)/255.0 for i in (1, 3, 5))
white_rgb = (1.0, 1.0, 1.0)
# Create colormap from white (low density) to theme color (high density)
custom_cmap = LinearSegmentedColormap.from_list('white_to_theme', [white_rgb, theme_rgb], N=256)

# =========================
# Figure 3: Many-point strategies (4 panels in 2x2 grid)
# =========================

# Generate a "many points" dataset (correlated cloud with a hint of curvature)
n = 1500
x = rng.normal(0, 1.0, n)
y = 0.9 * x + rng.normal(0, 0.55, n) + 0.25 * np.sin(2.2 * x)

# Scale to 0..10 so all panels share the same limits
x = (x - x.min()) / (x.max() - x.min()) * 10
y = (y - y.min()) / (y.max() - y.min()) * 10

# Set limits with small margin to ensure all points are visible
x_lim = (-0.5, 10.5)
y_lim = (-0.5, 10.5)

fig = plt.figure(figsize=(10, 10), dpi=220)

# 2x2 grid layout
w = 0.38  # panel width
h = 0.38  # panel height

# Calculate positions for 2x2 grid
# Top row: panels 1 and 2
# Bottom row: panels 3 and 4
top_y = 0.58  # Increased from 0.52 to add more space
bottom_y = 0.08
left_x = 0.08
right_x = 0.55

axes = []

# Panel 1 (top-left): original scatter
ax1 = fig.add_axes([left_x, top_y, w, h])
axes.append(ax1)

# Panel 2 (top-right): transparency scatter
ax2 = fig.add_axes([right_x, top_y, w, h])
axes.append(ax2)

# Panel 3 (bottom-left): 2D histogram
ax3_main = fig.add_axes([left_x, bottom_y, w, h])
axes.append(ax3_main)

# Panel 4 (bottom-right): hexbin
ax4_main = fig.add_axes([right_x, bottom_y, w, h])
axes.append(ax4_main)

def clean_axes(ax):
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    # Show x and y axes with ticks
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

# Panel 1: original scatter
ax = axes[0]
ax.scatter(x, y, s=20, color=COLORS[0], edgecolors='none')
ax.set_title("Original data, 1500 points", fontsize=9)
clean_axes(ax)

# Panel 2: transparency scatter
ax = axes[1]
ax.scatter(x, y, s=25, alpha=0.12, color=COLORS[0], edgecolors='none')
ax.set_title("Plot with transparency", fontsize=9)
clean_axes(ax)

# Panel 3: 2D histogram
ax = axes[2]
ax.set_facecolor("white")
H, xedges, yedges, im = ax.hist2d(x, y, bins=30, cmap=custom_cmap, vmin=0)
im.set_clim(vmin=0)
ax.set_title("2D histogram", fontsize=9)
clean_axes(ax)

# Panel 4: hexbin
ax = axes[3]
ax.set_facecolor("white")
hb = ax.hexbin(x, y, gridsize=28, cmap=custom_cmap, mincnt=0)
ax.set_title("Hexbin", fontsize=9)
clean_axes(ax)

# Save the figure
output_path = Path("../images/figure_3.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220)
print(f"Figure 3 saved to {output_path.absolute()}")

plt.close(fig)

