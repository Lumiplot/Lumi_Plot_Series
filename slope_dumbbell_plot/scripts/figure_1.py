import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Set font to Inter for all plots
# Suppress font warnings if Inter is not installed
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']

# -------------------------
# Theme Colors
# -------------------------
# New theme color palette
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

TARGET = 60  # percent (adjust as needed)

# Sort data by 2025 Q4 values (descending) - highest values at top
d = df.copy().sort_values("our_company", ascending=False).reset_index(drop=True)

# -------------------------
# 1) Dumbbell with circles
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# y positions: top attribute at top
y = list(range(len(d)))
ax.set_yticks(y)
ax.set_yticklabels(d["attribute"])
ax.invert_yaxis()

# Connecting lines with difference labels
for i, row in d.iterrows():
    ax.plot([row["competitor"], row["our_company"]], [i, i], linewidth=2, color=COLOR_LINE, alpha=0.6)
    # Calculate difference and show on line
    diff = row["our_company"] - row["competitor"]
    mid_x = (row["competitor"] + row["our_company"]) / 2
    diff_sign = "+" if diff >= 0 else ""
    ax.text(mid_x, i, f"{diff_sign}{int(diff)}", ha="center", va="center", fontsize=7, 
            color=COLOR_TARGET, weight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_BG, edgecolor="none", alpha=0.8))

# Circles (2024 Q4 = light fill, 2025 Q4 = filled with theme color) - reduced size
ax.scatter(d["competitor"], y, s=150, facecolors=COLOR_BG, edgecolors=COLOR_2024, linewidths=1, zorder=3)
ax.scatter(d["our_company"], y, s=150, facecolors=COLOR_2025, edgecolors=COLOR_2025, zorder=4)

# Target vertical line
ax.axvline(TARGET, linewidth=2, linestyle="--", alpha=0.8, color=COLOR_TARGET, zorder=2)
# Target label - same style as bars target zone
ax.text(TARGET + 1, -0.7, f"TARGET {TARGET}%", ha="left", va="center", fontsize=9, color=COLOR_TARGET)

ax.set_xlim(40, 75)
ax.set_ylim(-1.2, len(d) - 0.5)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Proper legend with visual markers
from matplotlib.patches import Circle
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BG, 
               markeredgecolor=COLOR_2024, markersize=10, markeredgewidth=1, label='2024 Q4'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_2025, 
               markeredgecolor=COLOR_2025, markersize=10, markeredgewidth=1, label='2025 Q4')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_1a.png")
plt.savefig(output_path, dpi=200)
plt.close()


# -------------------------
# 2) Slopegraph (Competitor -> Our Company)
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Add small offsets to stagger overlapping points
def add_jitter(values, jitter_amount=0.8):
    """Add small vertical offsets to overlapping values"""
    jittered = values.copy()
    value_counts = {}
    for idx, val in enumerate(values):
        if val in value_counts:
            value_counts[val].append(idx)
        else:
            value_counts[val] = [idx]
    
    for val, indices in value_counts.items():
        if len(indices) > 1:
            # Stagger overlapping points
            for i, idx in enumerate(indices):
                offset = (i - (len(indices) - 1) / 2) * jitter_amount
                jittered[idx] = val + offset
    return jittered

# Apply jitter to both sides
competitor_jittered = add_jitter(d["competitor"].values)
our_company_jittered = add_jitter(d["our_company"].values)

x0, x1 = 0, 1
for i, row in d.iterrows():
    comp_val = competitor_jittered[i]
    our_val = our_company_jittered[i]
    ax.plot([x0, x1], [comp_val, our_val], linewidth=2, color=COLOR_LINE, alpha=0.6, zorder=1)
    ax.scatter([x0], [comp_val], s=30, color=COLOR_2024, zorder=3)
    ax.scatter([x1], [our_val], s=30, color=COLOR_2025, zorder=3)

# Right side labels (attribute names only)
for i, row in d.iterrows():
    our_val = our_company_jittered[i]
    ax.text(x1 + 0.03, our_val, row["attribute"],
            ha="left", va="center", fontsize=9, color=COLOR_2025)

# Target reference (optional, matches the example concept)
ax.axhline(TARGET, linewidth=1, linestyle="--", alpha=0.5, color=COLOR_TARGET)
ax.text(x0 - 0.03, TARGET, f"TARGET", ha="right", va="bottom", fontsize=7, color=COLOR_TARGET)

ax.set_xlim(-0.2, 1.2)
ax.set_xticks([x0, x1])
ax.set_xticklabels(["2024 Q4", "2025 Q4"])
ax.set_ylim(40, 75)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_1b.png")
plt.savefig(output_path, dpi=200)
plt.close()


# -------------------------
# 3) Horizontal bar comparison + shaded target zone
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

y = list(range(len(d)))
ax.set_yticks(y)
ax.set_yticklabels(d["attribute"])
ax.invert_yaxis()

# Shaded target zone (>= target)
ax.axvspan(TARGET, 100, alpha=0.15, color=COLOR_TARGET)
ax.text(TARGET + 1, len(d) - 0.3, f"TARGET {TARGET}%", fontsize=9, color=COLOR_TARGET)

# Offset bars so they don't overlap
y_offset_2024 = [yi - 0.15 for yi in y]
y_offset_2025 = [yi + 0.15 for yi in y]

# Competitor bars (outlined with black)
ax.barh(y_offset_2024, d["competitor"], height=0.3, facecolor="none", edgecolor=COLOR_2024, linewidth=0.8, label="2024 Q4")

# Our company bars (filled with red, outlined with same red)
ax.barh(y_offset_2025, d["our_company"], height=0.3, facecolor=COLOR_2025, edgecolor=COLOR_2025, linewidth=0.8, label="2025 Q4")

ax.set_xlim(0, 100)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_1c.png")
plt.savefig(output_path, dpi=200)
plt.close()


# -------------------------
# 4) Diverging bar chart showing differences
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Calculate differences
d["difference"] = d["our_company"] - d["competitor"]

# Sort by difference for better visualization
d_sorted = d.sort_values("difference")

y = list(range(len(d_sorted)))
ax.set_yticks(y)
ax.set_yticklabels(d_sorted["attribute"])
ax.invert_yaxis()

# Create diverging bars - all start from zero
# Positive differences (improvements) go right, negative (declines) go left
for i, row in d_sorted.iterrows():
    diff = row["difference"]
    if diff >= 0:
        # Positive: teal color, extend to the right from zero
        ax.barh(i, diff, height=0.6, left=0, color=COLOR_TARGET, alpha=0.8)
        # Add value label
        ax.text(diff / 2, i, f"+{int(diff)}", ha="center", va="center", 
                fontsize=9, weight="bold", color="white")
    else:
        # Negative: orange color, extend to the left from zero (negative width extends left)
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
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_1d.png")
plt.savefig(output_path, dpi=200)
plt.close()

print("Saved: figure_1a.png, figure_1b.png, figure_1c.png, figure_1d.png")
