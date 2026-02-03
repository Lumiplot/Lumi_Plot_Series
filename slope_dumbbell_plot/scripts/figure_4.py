import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

# Set font to Inter for all plots
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# -------------------------
# Theme Colors
# -------------------------
COLOR_2024 = "#292F36"  # Dark Gray / Charcoal
COLOR_2025 = "#A41F13"  # Reddish-Brown / Burnt Orange
COLOR_TARGET = "#8F7A6E"  # Medium Brown / Taupe
COLOR_LINE = "#E0DBD8"  # Light Gray / Beige - for connecting lines
COLOR_BG = "#FAF5F1"  # Off-White / Cream - for backgrounds

# -------------------------
# Load data from CSV
# -------------------------
# Data is in the parent directory (slope_dumbbell folder)
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "survey_data.csv")
df = pd.read_csv(data_path)

# Rename columns to match script expectations
df = df.rename(columns={
    "Category": "attribute",
    "2024 Q4": "competitor",
    "2025 Q4": "our_company"
})

# Remove any rows with missing data
df = df.dropna()

# -------------------------
# Figure 4a: Diverging bar chart showing differences
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Calculate differences and sort by difference
d_diverging = df.copy()
d_diverging["difference"] = d_diverging["our_company"] - d_diverging["competitor"]
d_sorted = d_diverging.sort_values("difference").reset_index(drop=True)

y = list(range(len(d_sorted)))
ax.set_yticks(y)
ax.set_yticklabels(d_sorted["attribute"])
ax.invert_yaxis()

# Create diverging bars - all start from zero
# Positive differences (improvements) go right, negative (declines) go left
for i, row in d_sorted.iterrows():
    diff = row["difference"]
    if diff >= 0:
        # Positive: brown color, extend to the right from zero
        ax.barh(i, diff, height=0.6, left=0, color=COLOR_TARGET, alpha=0.8)
        # Add value label
        ax.text(diff / 2, i, f"+{int(diff)}", ha="center", va="center", 
                fontsize=9, weight="bold", color="white")
    else:
        # Negative: red color, extend to the left from zero (negative width extends left)
        ax.barh(i, diff, height=0.6, left=0, color=COLOR_2025, alpha=0.8)
        # Add value label
        ax.text(diff / 2, i, f"{int(diff)}", ha="center", va="center", 
                fontsize=9, weight="bold", color="white")

# Add vertical line at zero
ax.axvline(0, color=COLOR_2024, linewidth=1, linestyle='-', alpha=0.5)

# Set x limits with minimum of -15
max_diff = max(abs(d_sorted["difference"]))
ax.set_xlim(-15, max(max_diff + 2, 15))
ax.set_xlabel("Change from 2024 Q4 to 2025 Q4 (%)")
ax.set_title("Employee satisfaction survey - Change by category")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_4a.png")
plt.savefig(output_path, dpi=200)
plt.close()

# -------------------------
# Figure 4b: Dumbbell version of diverging bars chart
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Calculate differences and sort by difference (same as figure 4a)
d_dumbbell = df.copy()
d_dumbbell["difference"] = d_dumbbell["our_company"] - d_dumbbell["competitor"]
d_sorted_dumbbell = d_dumbbell.sort_values("difference").reset_index(drop=True)

y = list(range(len(d_sorted_dumbbell)))
# Remove y-axis ticks and labels - we'll place labels next to data points instead
ax.set_yticks([])
ax.invert_yaxis()

# Add subtle vertical reference lines at major tick marks
for x_tick in [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25]:
    ax.axvline(x_tick, color=COLOR_LINE, linewidth=0.5, alpha=0.2, linestyle='--', zorder=0)

# Create dumbbell plot - line from baseline (0) to change value, no dots on baseline
for i, row in d_sorted_dumbbell.iterrows():
    diff = row["difference"]
    
    # Connect baseline (0) to change value
    ax.plot([0, diff], [i, i], linewidth=2.5, color=COLOR_LINE, alpha=0.6, zorder=1)
    
    # Change point - color based on positive/negative (NO dot at zero)
    if diff >= 0:
        # Positive change: brown color
        ax.scatter([diff], [i], s=150, facecolors=COLOR_TARGET, edgecolors=COLOR_TARGET, linewidths=1, zorder=4)
        # Add value label on the dot (centered)
        ax.text(diff, i, f"+{int(diff)}", ha="center", va="center", fontsize=7, 
                weight="bold", color="white", zorder=5)
        # Add category label next to the data point
        ax.text(diff + 1.5, i, row["attribute"], ha="left", va="center", fontsize=9, 
                color=COLOR_2024, zorder=5)
    else:
        # Negative change: red color
        ax.scatter([diff], [i], s=150, facecolors=COLOR_2025, edgecolors=COLOR_2025, linewidths=1, zorder=4)
        # Add value label on the dot (centered)
        ax.text(diff, i, f"{int(diff)}", ha="center", va="center", fontsize=7, 
                weight="bold", color="white", zorder=5)
        # Add category label next to the data point
        ax.text(diff - 1.5, i, row["attribute"], ha="right", va="center", fontsize=9, 
                color=COLOR_2024, zorder=5)

# Add vertical line at zero
ax.axvline(0, color=COLOR_2024, linewidth=1.5, linestyle='-', alpha=0.5, zorder=2)

# Set x limits to -25 to 25
ax.set_xlim(-25, 25)
ax.set_xlabel("Change from 2024 Q4 to 2025 Q4 (%)")
ax.set_title("Employee satisfaction survey - Change by category")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

# Legend (no baseline dot since we removed it)
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_TARGET, 
               markeredgecolor=COLOR_TARGET, markersize=10, markeredgewidth=1, label='Positive change'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_2025, 
               markeredgecolor=COLOR_2025, markersize=10, markeredgewidth=1, label='Negative change')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_4b.png")
plt.savefig(output_path, dpi=200)
plt.close()

print("Saved: figure_4a.png, figure_4b.png")
