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

TARGET = 60  # percent

# -------------------------
# 1) BAD: Unranked (alphabetical), no difference numbers, small dots, black and white
# Figure 2a
# -------------------------
d_bad = df.copy().sort_values("attribute", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.6, 5.6))

y = list(range(len(d_bad)))
ax.set_yticks(y)
ax.set_yticklabels(d_bad["attribute"])
ax.invert_yaxis()

# Connecting lines - simple, no difference labels
for i, row in d_bad.iterrows():
    ax.plot([row["competitor"], row["our_company"]], [i, i], 
            linewidth=1, color='black', alpha=0.4)

# Small dots - black and white (swapped)
ax.scatter(d_bad["competitor"], y, s=30, color='white', edgecolors='black', 
           linewidths=1, marker='o', zorder=3)
ax.scatter(d_bad["our_company"], y, s=30, color='black', marker='o', zorder=3)

ax.set_xlim(40, 75)
ax.set_ylim(-0.5, len(d_bad) - 0.5)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

# Simple legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=6, markeredgewidth=1, label='2024 Q4'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markeredgecolor='black', markersize=6, markeredgewidth=1, label='2025 Q4')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_2a.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 2) BETTER: Ordered by 2025 data (descending)
# Figure 2b
# -------------------------
d_better = df.copy().sort_values("our_company", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.6, 5.6))

y = list(range(len(d_better)))
ax.set_yticks(y)
ax.set_yticklabels(d_better["attribute"])
ax.invert_yaxis()

# Connecting lines - no difference labels yet
for i, row in d_better.iterrows():
    ax.plot([row["competitor"], row["our_company"]], [i, i], 
            linewidth=2, color=COLOR_LINE, alpha=0.6)

# Small dots - still black and white (swapped)
ax.scatter(d_better["competitor"], y, s=30, color='white', edgecolors='black', 
           linewidths=1, marker='o', zorder=3)
ax.scatter(d_better["our_company"], y, s=30, color='black', marker='o', zorder=3)

ax.set_xlim(40, 75)
ax.set_ylim(-0.5, len(d_better) - 0.5)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=6, markeredgewidth=1, label='2024 Q4'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
               markeredgecolor='black', markersize=6, markeredgewidth=1, label='2025 Q4')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_2b.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 3) EVEN BETTER: Shows number difference, large circles with different colors
# Figure 2c
# -------------------------
d_even_better = df.copy().sort_values("our_company", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.6, 5.6))

y = list(range(len(d_even_better)))
ax.set_yticks(y)
ax.set_yticklabels(d_even_better["attribute"])
ax.invert_yaxis()

# Connecting lines with difference labels
for i, row in d_even_better.iterrows():
    ax.plot([row["competitor"], row["our_company"]], [i, i], 
            linewidth=2, color=COLOR_LINE, alpha=0.6)
    # Calculate difference and show on line
    diff = row["our_company"] - row["competitor"]
    mid_x = (row["competitor"] + row["our_company"]) / 2
    diff_sign = "+" if diff >= 0 else ""
    ax.text(mid_x, i, f"{diff_sign}{int(diff)}", ha="center", va="center", 
            fontsize=7, color=COLOR_TARGET, weight="bold", 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_BG, 
                     edgecolor="none", alpha=0.8))

# Large circles with different colors
ax.scatter(d_even_better["competitor"], y, s=150, facecolors=COLOR_BG, 
           edgecolors=COLOR_2024, linewidths=1, zorder=3)
ax.scatter(d_even_better["our_company"], y, s=150, facecolors=COLOR_2025, 
           edgecolors=COLOR_2025, zorder=4)

ax.set_xlim(40, 75)
ax.set_ylim(-0.5, len(d_even_better) - 0.5)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BG, 
               markeredgecolor=COLOR_2024, markersize=10, markeredgewidth=1, label='2024 Q4'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_2025, 
               markeredgecolor=COLOR_2025, markersize=10, markeredgewidth=1, label='2025 Q4')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_2c.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

# -------------------------
# 4) GREAT: Shows annotation of the reference line (target line)
# Figure 2d
# -------------------------
d_great = df.copy().sort_values("our_company", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.6, 5.6))

y = list(range(len(d_great)))
ax.set_yticks(y)
ax.set_yticklabels(d_great["attribute"])
ax.invert_yaxis()

# Connecting lines with difference labels
for i, row in d_great.iterrows():
    ax.plot([row["competitor"], row["our_company"]], [i, i], 
            linewidth=2, color=COLOR_LINE, alpha=0.6)
    # Calculate difference and show on line
    diff = row["our_company"] - row["competitor"]
    mid_x = (row["competitor"] + row["our_company"]) / 2
    diff_sign = "+" if diff >= 0 else ""
    ax.text(mid_x, i, f"{diff_sign}{int(diff)}", ha="center", va="center", 
            fontsize=7, color=COLOR_TARGET, weight="bold", 
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_BG, 
                     edgecolor="none", alpha=0.8))

# Large circles with different colors
ax.scatter(d_great["competitor"], y, s=150, facecolors=COLOR_BG, 
           edgecolors=COLOR_2024, linewidths=1, zorder=3)
ax.scatter(d_great["our_company"], y, s=150, facecolors=COLOR_2025, 
           edgecolors=COLOR_2025, zorder=4)

# Target vertical line with annotation
ax.axvline(TARGET, linewidth=2, linestyle="--", alpha=0.8, color=COLOR_TARGET, zorder=2)
# Target label annotation
ax.text(TARGET + 1, -0.7, f"TARGET {TARGET}%", ha="left", va="center", 
        fontsize=9, color=COLOR_TARGET, weight="bold")

ax.set_xlim(40, 75)
ax.set_ylim(-1.2, len(d_great) - 0.5)
ax.set_xlabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_BG, 
               markeredgecolor=COLOR_2024, markersize=10, markeredgewidth=1, label='2024 Q4'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_2025, 
               markeredgecolor=COLOR_2025, markersize=10, markeredgewidth=1, label='2025 Q4')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_2d.png")
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Saved: {output_path}")

print("\nAll dumbbell progression plots generated successfully!")
print("  - figure_2a.png: Unranked, no differences, small dots, B&W")
print("  - figure_2b.png: Ordered by 2025, small dots, B&W")
print("  - figure_2c.png: Ordered, shows differences, large colored circles")
print("  - figure_2d.png: All features + target line annotation")
