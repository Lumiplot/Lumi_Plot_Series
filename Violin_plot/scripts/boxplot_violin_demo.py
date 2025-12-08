"""
Demonstration: Identical Boxplots, Different Violin Plots

This script creates four groups with very different distributions:
- Group A: roughly normal (single peak in the middle)
- Group B: very bimodal (two peaks, little mass near the median)
- Group C: heavily right-skewed
- Group D: single, fat central hump (very concentrated near the median)

Each group is warped using a monotone piecewise-linear map so that all have
the same min, Q1, median, Q3, max = 0, 25, 50, 75, 100.

Result: Boxplots look identical (they only use the five-number summary),
but violin plots reveal the very different underlying distributions.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

n = 400  # points per group

# 1. Base shapes in "Z-space"
Z_A = np.random.normal(0, 1.0, n)  # unimodal
Z_B = np.concatenate([
    np.random.normal(-2, 0.35, n//2),
    np.random.normal(2, 0.35, n//2)
])  # bimodal
Z_C = np.random.exponential(1.0, n)  # skewed
Z_D = np.random.beta(8, 8, n)  # Beta(8,8): big peak in the middle
Z_D = Z_D * 4 - 2  # put it on a similar scale as others


def warp_to_common_box(z, targets=(0, 25, 50, 75, 100)):
    """
    Monotone piecewise-linear transform so that:
    min -> targets[0], Q1 -> targets[1], median -> targets[2],
    Q3 -> targets[3], max -> targets[4].
    
    Parameters
    ----------
    z : array-like
        Input values to transform
    targets : tuple of float
        Target values for (min, Q1, median, Q3, max)
    
    Returns
    -------
    y : ndarray
        Transformed values with the specified quantiles
    """
    q_probs = np.array([0, 0.25, 0.5, 0.75, 1.0])
    z_q = np.quantile(z, q_probs)
    y_t = np.array(targets, dtype=float)
    y = np.empty_like(z, dtype=float)
    
    # Apply segment-wise linear mapping based on which quantile interval z falls into
    for i in range(4):
        z0, z1 = z_q[i], z_q[i+1]
        y0, y1 = y_t[i], y_t[i+1]
        # Handle boundaries: include endpoints for first and last segments
        if i == 0:
            mask = (z >= z0) & (z <= z1)
        elif i == 3:
            mask = (z > z0) & (z <= z1)
        else:
            mask = (z > z0) & (z <= z1)
        
        # Linear interpolation on that segment
        y[mask] = y0 + (z[mask] - z0) * (y1 - y0) / (z1 - z0 + 1e-9)
    
    return y


# Warp each group to have the same five-number summary
targets = (0, 25, 50, 75, 100)
Y_A = warp_to_common_box(Z_A, targets)
Y_B = warp_to_common_box(Z_B, targets)
Y_C = warp_to_common_box(Z_C, targets)
Y_D = warp_to_common_box(Z_D, targets)  # same min/Q1/median/Q3/max as A/B/C

# Put into one tidy data frame
df = pd.DataFrame({
    "group": (["A"] * n) + (["B"] * n) + (["C"] * n) + (["D"] * n),
    "value": np.concatenate([Y_A, Y_B, Y_C, Y_D])
})

# Quick check: five-number summaries per group
print("Five-number summaries per group:")
print(df.groupby("group")["value"].describe()[["min", "25%", "50%", "75%", "max"]])
print("\nNote: All groups should have nearly identical summaries!")

# Create visualizations
plt.figure(figsize=(18, 5))

# Define orange color
orange_color = '#ff7f0e'

# Set consistent y-axis limits for all plots
y_min = df['value'].min()
y_max = df['value'].max()
y_padding = (y_max - y_min) * 0.25  # Add 25% padding on both sides
y_lim = (y_min - y_padding, y_max + y_padding)

# Panel 1: Raw Data Points
plt.subplot(1, 3, 1)
sns.stripplot(data=df, x="group", y="value", color=orange_color, alpha=0.3, size=2)
plt.ylim(y_lim)
plt.title("Raw Data Points\n(Actual distribution)", fontsize=12, fontweight='bold')
plt.ylabel("Value")
plt.grid(axis='y', alpha=0.3)

# Panel 2: Boxplots (with reduced width and opacity)
plt.subplot(1, 3, 2)
box_plot = sns.boxplot(data=df, x="group", y="value", color=orange_color, width=0.4)
plt.ylim(y_lim)
# Set opacity for boxplot fill
for patch in box_plot.artists:
    patch.set_facecolor(orange_color)
    patch.set_alpha(1.0)
    patch.set_edgecolor('black')  # Keep edges visible
plt.title("Boxplots\n(Identical five-number summaries)", fontsize=12, fontweight='bold')
plt.ylabel("Value")
plt.grid(axis='y', alpha=0.3)

# Panel 3: Violin Plots (with opacity)
plt.subplot(1, 3, 3)
violin_plot = sns.violinplot(data=df, x="group", y="value", inner="quartile", color=orange_color)
plt.ylim(y_lim)
# Set opacity for violin plot elements
for patch in violin_plot.collections:
    patch.set_alpha(1.0)
plt.title("Violin Plots\n(Reveal different distributions)", fontsize=12, fontweight='bold')
plt.ylabel("Value")
plt.grid(axis='y', alpha=0.3)

plt.subplots_adjust(wspace=0.5)
plt.savefig("../images/boxplot_violin_comparison.png", dpi=150, bbox_inches='tight')
print("\nPlot saved as '../images/boxplot_violin_comparison.png'")
plt.show()

# Optional: Save the dataset
df.to_csv("../data/demo_dataset.csv", index=False)
print("Dataset saved as '../data/demo_dataset.csv'")

