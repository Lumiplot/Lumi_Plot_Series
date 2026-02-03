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

# Sort data alphabetically by attribute name
d = df.copy().sort_values("attribute", ascending=True).reset_index(drop=True)

# -------------------------
# Vertical bar comparison (no target reference)
# -------------------------
fig, ax = plt.subplots(figsize=(5.6, 5.6))

x = list(range(len(d)))
ax.set_xticks(x)
ax.set_xticklabels(d["attribute"], rotation=45, ha="right")

# Offset bars so they don't overlap
x_offset_2024 = [xi - 0.15 for xi in x]
x_offset_2025 = [xi + 0.15 for xi in x]

# Competitor bars (outlined with black)
ax.bar(x_offset_2024, d["competitor"], width=0.3, facecolor="none", edgecolor=COLOR_2024, linewidth=0.8, label="2024 Q4")

# Our company bars (filled with red, outlined with same red)
ax.bar(x_offset_2025, d["our_company"], width=0.3, facecolor=COLOR_2025, edgecolor=COLOR_2025, linewidth=0.8, label="2025 Q4")

ax.set_ylim(0, 80)
ax.set_ylabel("% of employees agree")
ax.set_title("Employee satisfaction survey results")
ax.legend(loc="upper right", frameon=False)

# Clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "images", "figure_0.png")
plt.savefig(output_path, dpi=200)
plt.close()

print("Saved: figure_0.png")
