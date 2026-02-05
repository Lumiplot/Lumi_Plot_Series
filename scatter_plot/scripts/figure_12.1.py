import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

rng = np.random.default_rng(12345)

# Color palette
COLORS = ["#1B435E", "#FF7D2D", "#FAC846", "#A0C382", "#5F9B8C"]  # Dark Blue, Crusta, Bright Sun, Olivine, Patina

# ----------------------------
# 1) EV Battery Performance Dataset
# X: Charge rate (C rate, standardized)
# Y: Capacity fade after 100 cycles (standardized)
# Groups: Chemistry A, B, C
# Story: Chemistries separate cleanly. Within each chemistry, higher charge rate 
#        relates differently to degradation. Marginals show one chemistry lives 
#        in a safer operating range.
# ----------------------------
def make_group(name, n, mean_x, mean_y, slope, x_sd=0.55, noise=0.45):
    x = rng.normal(mean_x, x_sd, n)
    y = mean_y + slope*(x - mean_x) + rng.normal(0, noise, n)
    return pd.DataFrame({"chemistry": name, "charge_rate_z": x, "capacity_fade_z": y})

# Chemistry A: Lower charge rates, moderate degradation (safer operating range)
# Chemistry B: Mid-range charge rates, higher degradation
# Chemistry C: Higher charge rates, variable degradation
df = pd.concat([
    make_group("Chemistry A", 170, mean_x=-1.05, mean_y=-1.00, slope=0.18, x_sd=0.55, noise=0.40),
    make_group("Chemistry B",  90, mean_x=-0.55, mean_y= 0.85, slope=0.40, x_sd=0.50, noise=0.38),
    make_group("Chemistry C", 130, mean_x= 1.05, mean_y= 0.55, slope=0.75, x_sd=0.55, noise=0.42),
], ignore_index=True)

# Mild outliers to make it feel real
out = pd.DataFrame({
    "chemistry": ["Chemistry A", "Chemistry B", "Chemistry C", "Chemistry C"],
    "charge_rate_z": [-2.3, -1.2, 2.2, 1.9],
    "capacity_fade_z": [-2.1, 2.4, 2.6, 2.1],
})
df = pd.concat([df, out], ignore_index=True)

df["charge_rate_z"] = df["charge_rate_z"].clip(-3, 3)
df["capacity_fade_z"] = df["capacity_fade_z"].clip(-3, 3)

# ----------------------------
# Helpers: 1D KDE via histogram smoothing
# ----------------------------
def smooth_hist_density(values, grid, bandwidth=0.18):
    bins = np.linspace(grid.min(), grid.max(), 70)
    hist, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    dx = centers[1] - centers[0]
    kx = np.arange(-int(4*bandwidth/dx), int(4*bandwidth/dx)+1) * dx
    kernel = np.exp(-0.5*(kx/bandwidth)**2)
    kernel /= kernel.sum() * dx

    sm = np.convolve(hist, kernel, mode="same")
    return np.interp(grid, centers, sm, left=0, right=0)

def add_marginals(ax_top, ax_right, xvals, yvals, codes, x_grid, y_grid, color_fn):
    for code in np.unique(codes):
        d = smooth_hist_density(xvals[codes == code], x_grid, bandwidth=0.18)
        ax_top.fill_between(x_grid, 0, d, alpha=0.22, color=color_fn(code))
        ax_top.plot(x_grid, d, linewidth=1.2, alpha=0.9, color=color_fn(code))
    ax_top.set_xlim(-3, 3)
    ax_top.set_xticks([]); ax_top.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax_top.spines[sp].set_visible(False)

    for code in np.unique(codes):
        d = smooth_hist_density(yvals[codes == code], y_grid, bandwidth=0.18)
        ax_right.fill_betweenx(y_grid, 0, d, alpha=0.22, color=color_fn(code))
        ax_right.plot(d, y_grid, linewidth=1.2, alpha=0.9, color=color_fn(code))
    ax_right.set_ylim(-3, 3)
    ax_right.set_xticks([]); ax_right.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax_right.spines[sp].set_visible(False)

# ----------------------------
# Create single panel figure with margin KDE
# ----------------------------
chemistry_order = ["Chemistry A", "Chemistry B", "Chemistry C"]
cat = pd.Categorical(df["chemistry"], categories=chemistry_order, ordered=True)
codes = cat.codes

# Use custom color palette
color = lambda code: COLORS[code % len(COLORS)]

x_grid = np.linspace(-3, 3, 400)
y_grid = np.linspace(-3, 3, 400)

fig = plt.figure(figsize=(5.2, 5.2), dpi=220)
gs = GridSpec(2, 2, figure=fig, height_ratios=[0.24, 1.0], width_ratios=[1.0, 0.26], hspace=0.05, wspace=0.05)

ax_top = fig.add_subplot(gs[0, 0])
ax_main = fig.add_subplot(gs[1, 0])
ax_right = fig.add_subplot(gs[1, 1])
ax_corner = fig.add_subplot(gs[0, 1])
ax_corner.axis("off")

handles = [Line2D([0], [0], marker="o", linestyle="", markersize=6,
                  markerfacecolor=color(i), markeredgecolor="none", alpha=0.9)
           for i in range(len(chemistry_order))]

# Scatter plot with explicit colors
scatter_colors = [color(c) for c in codes]
ax_main.scatter(df["charge_rate_z"], df["capacity_fade_z"], s=18, alpha=0.60, c=scatter_colors, edgecolors="none")
ax_main.set_xlim(-3, 3); ax_main.set_ylim(-3, 3)
ax_main.set_xlabel("Charge rate (standardized)")
ax_main.set_ylabel("Capacity fade after 100 cycles (standardized)")
ax_main.grid(True, linestyle=":", linewidth=0.7, alpha=0.45)
ax_main.legend(handles, chemistry_order, frameon=True, fontsize=8, loc="upper left")

# Add margin KDE plots
add_marginals(ax_top, ax_right,
              df["charge_rate_z"].to_numpy(), df["capacity_fade_z"].to_numpy(), codes,
              x_grid, y_grid, color)

fig.suptitle("EV Battery Performance: Charge rate vs Capacity fade across chemistries",
             fontsize=12, y=0.98)

# Save the figure
output_path = Path("../images/figure_12.1.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, facecolor="white", dpi=220)
print(f"Figure 12.1 saved to {output_path.absolute()}")

plt.close(fig)

